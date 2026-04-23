# isaaclabmpc

MPPI (Model Predictive Path Integral) controller using **Isaac Lab** as the
parallel physics back-end.  Ported from
[genesismpc](../genesismpc) which used Genesis physics.

## Architecture

```
isaaclabmpc/
├── isaaclab_mpc/
│   ├── planner/
│   │   ├── isaaclab_wrapper.py   # Wraps SimulationContext + InteractiveScene
│   │   └── mppi_isaaclab.py      # MPPIIsaacLabPlanner (zerorpc server)
│   └── utils/
│       └── transport.py          # Tensor ↔ bytes serialisation
├── robots/
│   └── ur16e.py                  # UR16E ArticulationCfg (UrdfFileCfg)
└── examples/
    └── ur16e_reach/
        ├── planner.py            # zerorpc MPPI server
        ├── runner.py             # Standalone control loop (no RPC)
        └── config.yaml           # MPPI + physics parameters
```

**Key design choices** — same as genesismpc:

| Concept | Implementation |
|---------|---------------|
| `num_envs == num_samples` | Each Isaac Lab env IS one MPPI trajectory |
| `dynamics()` | `set_joint_velocity_target` + `sim.step()` |
| State reset | `write_joint_state_to_sim` + `scene.reset()` each planning cycle |
| RPC transport | zerorpc + `torch_to_bytes` / `bytes_to_torch` |

## Prerequisites

```bash
conda activate env_isaaclab

# Install mppi_torch (editable, from genesismpc companion repo)
pip install -e /home/paolo/Documents/mppi_torch

# Install this package
pip install -e /home/paolo/Documents/isaaclabmpc

# Install zerorpc
pip install zerorpc
```

## Running the ur16e_reach example

### Option A — standalone runner (single process, no viewer by default)

```bash
cd /home/paolo/Documents/isaaclabmpc
python examples/ur16e_reach/runner.py
```

Add `--no-headless` (or remove the default `headless=False`) to open the
Isaac Lab viewer and watch the robot reach towards the goal.

### Option B — zerorpc server (same pattern as genesismpc)

**Terminal 1** — start the MPPI planner server:
```bash
cd /home/paolo/Documents/isaaclabmpc
python examples/ur16e_reach/planner.py
```

**Terminal 2** — connect any zerorpc client (e.g. a ROS bridge or a custom
world viewer) to `tcp://127.0.0.1:4242` and call:

```python
import zerorpc, torch, io

def t2b(t):
    b = io.BytesIO(); torch.save(t, b); b.seek(0); return b.read()
def b2t(b):
    return torch.load(io.BytesIO(b), weights_only=False)

c = zerorpc.Client()
c.connect("tcp://127.0.0.1:4242")

# Send current joint state and get back optimal action
DOF = 6
q  = torch.zeros(DOF)
dq = torch.zeros(DOF)
dof_state = t2b(torch.cat([q, dq]))
root_state = t2b(torch.zeros(13))          # unused for fixed-base

action_bytes = c.compute_action_tensor(dof_state, root_state)
action = b2t(action_bytes)                 # shape (6,) joint velocities
```

## Configuration

Edit `examples/ur16e_reach/config.yaml`:

| Key | Description |
|-----|-------------|
| `goal` | Target EE position `[x, y, z]` in metres |
| `ee_link_name` | Body name for end-effector cost |
| `mppi.num_samples` | MPPI parallel trajectories (= num_envs) |
| `mppi.horizon` | Roll-out length in physics steps |
| `mppi.u_min/u_max` | Joint velocity limits [rad/s] |
| `isaaclab.dt` | Physics timestep [s] |

## UR16e robot

The robot is loaded from the URDF at:
```
../genesismpc/assets/ur_description/urdf/ur16e.urdf
```
Isaac Lab converts it to USD automatically on first run (cached in
`/tmp/isaaclab/`).  The URDF path is resolved relative to `robots/ur16e.py`
and expects the `genesismpc` repo to be at `../genesismpc` relative to this
repo.  Change `UR16E_URDF_PATH` in [robots/ur16e.py](robots/ur16e.py) if
your layout differs.
