
Proximity-based u_max clamping: reduce action magnitude when TCP is within e.g. 5cm of any object/table.

Larger dead-band on object→goal cost (already commented out in some objectives — re-enable).

Increase smoothing in the bridge as goal_distance shrinks (variable smoothing_alpha).

Async MPPI on its own thread so planning dt actually matches execution dt — this is the real fix for the 18x overshoot and we've discussed it before.