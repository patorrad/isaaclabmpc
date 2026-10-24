"""UCT search for viewpoints and pushes that reveal a hidden target (after mppi_playground/mcts.py).

State: the TCP (tool tip) position with the tool pointing straight down, plus the
known objects' boxes as the path has pushed them. Actions:
  move       : move at `speed` for `action_time` along one of `n_dirs` unit directions
  push macro : push object k from one of its 4 faces by `push_dist` (if `push_macros`):
               rise to a safe height above all objects, move over the contact point beside
               that face, descend to push height, sweep through. A push needs this exact
               sequence, which random single moves rarely find; as one action it is depth 1.
  view macro : go to one of the `n_views` best viewpoints (if `n_views` > 0): candidate
               viewpoints on a `view_grid` grid at height `view_z` are ranked once, at the root,
               by the belief mass they reveal; travelling there is one action.

Tool model (tool pointing down, tip at the TCP):
  rod      : segment tip → tip + coupling_bottom, radius rod_radius (the part that pushes)
  coupling : segment tip + coupling_bottom → tip + coupling_top, radius coupling_radius
  sensor   : sensor_height above the tip, looking down with a square fov (disk of the
             inscribed half-angle is used, since the sensor's yaw is free)

Push model (quasi-static, like the playground's pushable walls): when the tip, below an
object's top, moves into its footprint from the side, the object translates with the
tip's horizontal motion. A push that would drive the object out of the tote or into
another object jams the tool (collision). Pressing down onto an object, the coupling
touching an object, or the tool touching a wall / the floor / leaving the workspace is
a collision (terminal).

Reward (accumulated along the path; a node's value is its expected return-to-go):
  + w_info * probability mass of floor cells newly visible after the action (cells
    already claimed earlier on the same path count once)
  - w_push per action that pushes, - w_dturn * (direction change / pi)
  - w_obstacle * (1 - d / margin)^2 when the tip is within `margin` of a wall below the rim
  - w_len per metre travelled (macros travel further than single moves)
  - w_collision on collision (terminal); a path that has claimed all but `done_eps` of the
    mass is complete (terminal, no penalty)
  non-terminal leaf: w_heur * sum_c p_c exp(-|tip - c|^2 / 2 sigma^2) over unclaimed cells
Random rollouts use move actions only; push macros are evaluated by the tree itself.

numpy only.
"""

import math
import random
import time

import numpy as np

from .belief import Box, FloorBelief, rects_overlap


def directions(n_dirs: int) -> np.ndarray:
    """6 (axes), 14 (axes + corners) or 26 (axes + corners + edges) unit directions."""
    axes = [v for i in range(3) for v in (np.eye(3)[i], -np.eye(3)[i])]
    corners = [np.array([x, y, z]) / math.sqrt(3) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    edges = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for a in (-1, 1):
            for b in (-1, 1):
                v = np.zeros(3); v[i], v[j] = a, b
                edges.append(v / math.sqrt(2))
    sets = {6: axes, 14: axes + corners, 26: axes + corners + edges}
    if n_dirs not in sets:
        raise ValueError(f"n_dirs must be 6, 14 or 26, got {n_dirs}")
    return np.array(sets[n_dirs])


def segments_hit_boxes(p0, p1, centers, halves, yaws, pad=0.0):
    """Slab test for K segments against B upright boxes → (K, B) bool."""
    c, s = np.cos(yaws), np.sin(yaws)                                    # (B,)
    a = p0[:, None, :] - centers[None]                                   # (K, B, 3)
    d = (p1 - p0)[:, None, :].repeat(len(centers), axis=1)
    # rotate world → box frame about z
    ax = c * a[..., 0] + s * a[..., 1]; ay = -s * a[..., 0] + c * a[..., 1]
    dx = c * d[..., 0] + s * d[..., 1]; dy = -s * d[..., 0] + c * d[..., 1]
    A = np.stack([ax, ay, a[..., 2]], -1); D = np.stack([dx, dy, d[..., 2]], -1)
    h = (halves + pad)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-h - A) / D, (h - A) / D
    par = D == 0
    inside = np.abs(A) <= h
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    te, tx = tmin.max(-1), tmax.min(-1)
    return (te <= tx) & (tx >= 0.0) & (te <= 1.0)


def _box_arrays(boxes):
    if not boxes:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return (np.array([b.center for b in boxes], float), np.array([b.half for b in boxes], float),
            np.array([b.yaw for b in boxes], float))


def _contains_xy(xy, centers, halves, yaws, pad):
    """Is point xy inside each box's padded footprint? → (B,) bool."""
    c, s = np.cos(yaws), np.sin(yaws)
    d = xy[None] - centers[:, :2]
    lx = np.abs(c * d[:, 0] + s * d[:, 1]); ly = np.abs(-s * d[:, 0] + c * d[:, 1])
    return (lx <= halves[:, 0] + pad) & (ly <= halves[:, 1] + pad)


def search(tcp, belief: FloorBelief, objects: list, walls: list, interior, p: dict, seed: int = 0) -> dict:
    """Plan from `tcp` (3,). objects / walls: lists of Box. interior: ((x0, x1), (y0, y1)) tote floor."""
    t_start = time.perf_counter()
    rng = random.Random(seed)
    tcp = np.asarray(tcp, float)
    dirs = directions(int(p["n_dirs"]))
    n_moves = len(dirs)
    n_sub = max(1, int(p.get("n_sub", 3)))
    step = p["speed"] * p["action_time"] / n_sub
    lo, hi = np.asarray(p["workspace_lo"], float), np.asarray(p["workspace_hi"], float)
    # A start above the ceiling (e.g. the robot's home pose) is allowed for this search, so the
    # tree can plan its way down instead of finding nothing.
    hi = hi.copy()
    hi[2] = max(hi[2], float(np.asarray(tcp, float)[2]) + 1e-6)
    floor_z = belief.floor_z
    rod_r, cpl_r = p["rod_radius"], p["coupling_radius"]
    cpl_lo, cpl_hi, sens_h = p["coupling_bottom"], p["coupling_top"], p["sensor_height"]
    tan_half = math.tan(p["fov"] / 2)
    w_info, w_push, w_col = p["w_info"], p.get("w_push", 0.0), p["w_collision"]
    w_obs, margin, w_dturn = p.get("w_obstacle", 0.0), max(p.get("margin", 0.03), 1e-6), p.get("w_dturn", 0.0)
    w_heur, sigma = p.get("w_heur", 0.1), p.get("heur_sigma", 0.1)
    rim_z = max((w.top for w in walls), default=floor_z)
    outer = p.get("outer_margin", 0.25)
    (ix0, ix1), (iy0, iy1) = interior

    wc, wh, wy = _box_arrays(walls)
    oc0, oh, oy = _box_arrays(objects)

    # Candidate cells: those the target may still be in.
    flat_p = belief.p.ravel()
    cand = np.flatnonzero(flat_p > 0)
    cand_xy = belief.cells.reshape(-1, 2)[cand]
    cand_p = flat_p[cand]
    cand_floor = np.column_stack([cand_xy, np.full(len(cand), floor_z)])

    # Per-box scalars for the pure-Python contact checks (a handful of boxes; numpy call overhead
    # would dominate). The tote interior is axis-aligned, so wall checks are bounds checks.
    o_cos, o_sin = np.cos(oy).tolist(), np.sin(oy).tolist()
    o_hx, o_hy, o_hz = oh[:, 0].tolist(), oh[:, 1].tolist(), oh[:, 2].tolist()
    n_obj = len(oc0)
    n_push = 4 * n_obj if p.get("push_macros", True) else 0
    push_dist, w_len = p.get("push_dist", 0.06), p.get("w_len", 0.0)
    gap = p.get("push_gap", 0.01)                        # clearance from the face before the sweep
    done_eps = p.get("done_eps", 0.02)                   # unclaimed mass below this: path complete (terminal)
    faces = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))   # outward normals in the box frame

    def in_footprint(x, y, oc, k, pad):
        dx, dy = x - oc[k][0], y - oc[k][1]
        return (abs(o_cos[k] * dx + o_sin[k] * dy) <= o_hx[k] + pad
                and abs(-o_sin[k] * dx + o_cos[k] * dy) <= o_hy[k] + pad)

    def inside_interior(x, y, pad):
        return ix0 + pad <= x <= ix1 - pad and iy0 + pad <= y <= iy1 - pad

    def collides(tip, oc):
        """Tool vs workspace, floor, tote walls (below the rim), and coupling vs objects."""
        x, y, z = tip
        if not lo[2] <= z <= hi[2]:
            return True
        # Below the rim the tool must stay over the tote; with the whole tool above the rim it may
        # also be outside it (by `outer_margin`), e.g. where the robot starts.
        m = outer if z >= rim_z + 0.01 else 0.0
        if not (lo[0] - m <= x <= hi[0] + m and lo[1] - m <= y <= hi[1] + m):
            return True
        if z < floor_z + rod_r:
            return True
        if z < rim_z and not inside_interior(x, y, rod_r):                 # rod below the rim
            return True
        if z + cpl_lo < rim_z and not inside_interior(x, y, cpl_r):        # coupling below the rim
            return True
        for k in range(n_obj):                                             # coupling onto an object
            if z + cpl_lo < oc[k][2] + o_hz[k] and in_footprint(x, y, oc, k, cpl_r):
                return True
        return False

    def footprint_inside_tote(oc, k):
        c, s_ = o_cos[k], o_sin[k]
        ex = abs(c) * o_hx[k] + abs(s_) * o_hy[k]          # half extents of the rotated footprint's AABB
        ey = abs(s_) * o_hx[k] + abs(c) * o_hy[k]
        return ix0 <= oc[k][0] - ex and oc[k][0] + ex <= ix1 and iy0 <= oc[k][1] - ey and oc[k][1] + ey <= iy1

    def push(tip, new_tip, oc):
        """Returns (oc', pushed, jammed)."""
        if not n_obj:
            return oc, False, False
        hit = [k for k in range(n_obj)
               if new_tip[2] < oc[k][2] + o_hz[k] and in_footprint(new_tip[0], new_tip[1], oc, k, rod_r)]
        if not hit:
            return oc, False, False
        dx, dy = new_tip[0] - tip[0], new_tip[1] - tip[1]
        if abs(dx) + abs(dy) < 1e-9 and new_tip[2] > tip[2]:
            return oc, False, False         # lifting straight up out of contact is always free
        for k in hit:
            if tip[2] >= oc[k][2] + o_hz[k] and in_footprint(tip[0], tip[1], oc, k, rod_r):
                return oc, False, True      # pressing down onto an object
        if abs(dx) + abs(dy) < 1e-9:
            return oc, False, True
        oc = oc.copy()
        for k in hit:
            oc[k, 0] += dx
            oc[k, 1] += dy
        for k in hit:                        # pushed box must stay in the tote and clear of the others
            if not footprint_inside_tote(oc, k):
                return oc, True, True
            bk = Box(oc[k], oh[k], oy[k])
            for m in range(n_obj):
                if m != k and rects_overlap(bk, Box(oc[m], oh[m], oy[m])):
                    return oc, True, True
        return oc, True, False

    def visible_gain(tip, oc, claimed):
        """Probability mass of unclaimed candidate cells the sensor would see from here."""
        s = tip + np.array([0.0, 0.0, sens_h])
        h = s[2] - floor_z
        if h <= 0 or not len(cand):
            return 0.0, claimed
        r = h * tan_half
        idx = np.flatnonzero(~claimed & (np.sum((cand_xy - s[:2]) ** 2, axis=1) <= r * r))
        if not len(idx):
            return 0.0, claimed
        ends = cand_floor[idx]
        starts = np.repeat(s[None], len(idx), axis=0)
        # Stop 2 mm above the floor so a cell isn't occluded by the box it sits next to at floor level.
        ends = ends + np.array([0.0, 0.0, 0.002])
        occ = np.zeros(len(idx), bool)
        if len(wc):
            occ |= segments_hit_boxes(starts, ends, wc, wh, wy).any(1)
        if len(oc):
            occ |= segments_hit_boxes(starts, ends, oc, oh, oy).any(1)
        vis = idx[~occ]
        if not len(vis):
            return 0.0, claimed
        claimed = claimed.copy()
        claimed[vis] = True
        return float(cand_p[vis].sum()), claimed

    def wall_prox(tip):
        if tip[2] >= rim_z:
            return 0.0
        d = min(tip[0] - ix0, ix1 - tip[0], tip[1] - iy0, iy1 - tip[1])
        return w_obs * (1 - d / margin) ** 2 if 0 <= d < margin else 0.0

    def move_to(tip, target, oc):
        """Straight line in substeps of at most `step`; push model + collision on each.
        → (tip, oc, pushed, collided)."""
        dist = float(np.linalg.norm(target - tip))
        n = max(1, int(math.ceil(dist / step - 1e-9)))
        pushed_any = False
        for i in range(1, n + 1):
            new_tip = tip + (target - tip) / (n - i + 1)
            oc, pushed, jammed = push(tip, new_tip, oc)
            pushed_any |= pushed
            if jammed or collides(new_tip, oc):
                return new_tip, oc, pushed_any, True
            tip = new_tip
        return tip, oc, pushed_any, False

    def macro_waypoints(tip, oc, k, face):
        """Rise → over the contact point → down to push height → sweep `push_dist` into face `face` of box k."""
        nx_l, ny_l = faces[face]
        c, s_ = o_cos[k], o_sin[k]
        nx, ny = c * nx_l - s_ * ny_l, s_ * nx_l + c * ny_l           # face normal in world
        reach = (o_hx[k] if nx_l else o_hy[k]) + rod_r + gap
        sx, sy = oc[k][0] + nx * reach, oc[k][1] + ny * reach
        bottom = oc[k][2] - o_hz[k]
        z_push = max(floor_z + rod_r + 0.003, min(bottom + 1.2 * o_hz[k], bottom + 2 * o_hz[k] - 0.01))
        z_safe = max(tip[2], max(oc[m][2] + o_hz[m] for m in range(n_obj)) + 0.02)
        return [np.array([tip[0], tip[1], z_safe]), np.array([sx, sy, z_safe]),
                np.array([sx, sy, z_push]),
                np.array([sx - nx * (push_dist + gap), sy - ny * (push_dist + gap), z_push])]

    # View macros: rank candidate viewpoints by the mass they reveal from the root state.
    # Viewpoints sit a fixed clearance above the tallest known object (not at the current TCP height:
    # high viewpoints near the ceiling are slow for the arm to reach and move between).
    top_obj = max((oc0[m][2] + o_hz[m] for m in range(n_obj)), default=floor_z)
    view_z = p.get("view_z") or min(max(top_obj + p.get("view_clearance", 0.12), floor_z + 0.2),
                                    float(p["workspace_hi"][2]) - 0.02)
    views = []
    if int(p.get("n_views", 0)) > 0:
        g = p.get("view_grid", 0.05)
        cands = [np.array([x, y, view_z])
                 for x in np.arange(ix0 + cpl_r, ix1 - cpl_r + 1e-9, g)
                 for y in np.arange(iy0 + cpl_r, iy1 - cpl_r + 1e-9, g)]
        scored = [(visible_gain(v, oc0, np.zeros(len(cand), bool))[0], i) for i, v in enumerate(cands)]
        scored = [t for t in sorted(scored, reverse=True) if t[0] > 0][: int(p["n_views"])]
        views = [cands[i] for _, i in scored]
    n_act = n_moves + n_push + len(views)

    def simulate(tip, oc, claimed, a, prev):
        """One action → (tip, oc, claimed, term, reward, pushed, waypoints); term 1 = collision."""
        if a >= n_moves + n_push:                       # view macro
            v = views[a - n_moves - n_push]
            z_safe = max(tip[2], v[2])
            pts = [np.array([tip[0], tip[1], z_safe]), np.array([v[0], v[1], z_safe]), v]
            turn = 0.0
        elif a < n_moves:
            pts = [tip + step * n_sub * dirs[a]]
            turn = 0.0 if not (0 <= prev < n_moves) else \
                math.acos(max(-1.0, min(1.0, float(dirs[prev] @ dirs[a])))) / math.pi
        else:
            k, face = divmod(a - n_moves, 4)
            pts = macro_waypoints(tip, oc, k, face)
            turn = 0.0
        start, pushed_any = tip, False
        travelled = 0.0
        for target in pts:
            travelled += float(np.linalg.norm(target - tip))
            tip, oc, pushed, collided = move_to(tip, target, oc)
            pushed_any |= pushed
            if collided:
                return tip, oc, claimed, 1, -w_col, pushed_any, pts
        gain, claimed = visible_gain(tip, oc, claimed)
        done = float(cand_p[~claimed].sum()) < done_eps        # nothing left to find on this path
        r = (w_info * gain - (w_push if pushed_any else 0.0) - w_dturn * turn - w_len * travelled
             - wall_prox(tip))
        return tip, oc, claimed, (2 if done else 0), r, pushed_any, pts

    def heuristic(tip, claimed):
        free = ~claimed
        if not free.any():
            return 0.0
        d2 = np.sum((cand_xy[free] - tip[:2]) ** 2, axis=1)
        return w_heur * float(np.sum(cand_p[free] * np.exp(-d2 / (2 * sigma ** 2))))

    def rollout(tip, oc, claimed, depth, prev):
        total = 0.0
        for _ in range(depth):
            a = rng.randrange(n_moves)      # moves only: random macros mostly collide and drown the signal
            tip, oc, claimed, term, r, _, _ = simulate(tip, oc, claimed, a, prev)
            prev = a
            total += r
            if term:
                return total
        return total + heuristic(tip, claimed)

    # Tree as parallel lists; node 0 is the root. REWARD[n] = reward of the action into n.
    claimed0 = np.zeros(len(cand), bool)
    TIP, OC, CL, PTS = [tcp], [oc0], [claimed0], [[tcp]]
    PARENT, DEPTH, ACT, PUSH = [-1], [0], [-1], [False]
    N, Wsum, TERM, REWARD = [0], [0.0], [0], [0.0]
    children, untried = [[]], [list(range(n_act))]
    c_uct, max_depth, roll_depth = p["c"], int(p["depth"]), int(p["rollout_depth"])

    for _ in range(int(p["iterations"])):
        node, path = 0, [0]
        while not TERM[node] and DEPTH[node] < max_depth:
            if untried[node]:
                a = untried[node].pop(rng.randrange(len(untried[node])))
                tip, oc, cl, term, r, pushed, pts = simulate(TIP[node], OC[node], CL[node], a, ACT[node])
                child = len(TIP)
                TIP.append(tip); OC.append(oc); CL.append(cl); PTS.append(pts)
                PARENT.append(node); DEPTH.append(DEPTH[node] + 1); ACT.append(a); PUSH.append(pushed)
                N.append(0); Wsum.append(0.0); TERM.append(term); REWARD.append(r)
                children.append([]); untried.append(list(range(n_act)))
                children[node].append(child)
                node = child
                path.append(node)
                break
            log_n = math.log(N[node])
            node = max(children[node], key=lambda ch: REWARD[ch] + Wsum[ch] / N[ch] + c_uct * math.sqrt(log_n / N[ch]))
            path.append(node)

        if TERM[node]:
            g = 0.0
        elif DEPTH[node] >= max_depth:
            g = heuristic(TIP[node], CL[node])
        else:
            g = rollout(TIP[node], OC[node], CL[node], roll_depth, ACT[node])
        for n in reversed(path):
            N[n] += 1
            Wsum[n] += g
            g += REWARD[n]

    # Best path: most visited child (then value), stopping before a collision.
    best, pushes, node = [tcp.copy()], [], 0
    while children[node]:
        nxt = max(children[node], key=lambda ch: (N[ch], REWARD[ch] + Wsum[ch] / N[ch]))
        if TERM[nxt] == 1:
            break
        node = nxt
        best.extend(PTS[node])      # a macro contributes its corner points
        pushes.append(PUSH[node])

    return {
        "best_path": np.array(best),
        "best_actions": [("move" if ACT[n] < n_moves else
                          f"push obj {(ACT[n] - n_moves) // 4} face {(ACT[n] - n_moves) % 4}" if ACT[n] < n_moves + n_push
                          else f"view {ACT[n] - n_moves - n_push}")
                         for n in _path_nodes(children, N, REWARD, Wsum, TERM)],
        "views": np.array(views).reshape(-1, 3),
        "best_pushes": pushes,
        "push_planned": any(pushes),
        "best_gain": float(cand_p[CL[node]].sum()) if len(cand) else 0.0,   # mass the best path reveals
        "expected_return": Wsum[0] / max(N[0], 1),
        "num_nodes": len(TIP),
        "iterations": int(p["iterations"]),
        "ms": (time.perf_counter() - t_start) * 1000,
        "tree": _tree_segments(TIP, PTS, PARENT, N, TERM),
    }


def _path_nodes(children, N, REWARD, Wsum, TERM):
    """Nodes on the most-visited path (same rule as best_path)."""
    out, node = [], 0
    while children[node]:
        nxt = max(children[node], key=lambda ch: (N[ch], REWARD[ch] + Wsum[ch] / N[ch]))
        if TERM[nxt] == 1:
            break
        out.append(nxt)
        node = nxt
    return out


def _tree_segments(TIP, PTS, PARENT, N, TERM):
    """Every expanded edge of the tree as line segments, for debugging/visualisation.

    Returns {"seg": (S, 6) [x0, y0, z0, x1, y1, z1], "visits": (S,), "term": (S,)}; a macro action
    contributes one segment per leg (its real path), all carrying the child's visits and terminal flag.
    """
    seg, visits, term = [], [], []
    for n in range(1, len(TIP)):
        a = TIP[PARENT[n]]
        for b in PTS[n]:
            seg.append(np.concatenate([a, b]))
            visits.append(N[n])
            term.append(TERM[n])
            a = b
    return {"seg": np.array(seg).reshape(-1, 6), "visits": np.array(visits), "term": np.array(term)}


def pick_waypoint(path: np.ndarray, lookahead: float):
    """Point on `path` about `lookahead` metres of arc from its start (its end if shorter); None if no path."""
    if path is None or len(path) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    i = int(np.searchsorted(arc, lookahead))
    return path[min(i, len(path) - 1)]
