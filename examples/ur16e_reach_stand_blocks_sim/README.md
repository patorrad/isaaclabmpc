Once you have a checkpoint, the full workflow is:

Terminal 1 (openpi env):


cd ~/Documents/openpi
python scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_ur16e_push \
    --policy.dir checkpoints/pi05_ur16e_push/run_01/5000
Terminal 2 (isaaclab env):


cd /home/paolo/Documents/isaaclabmpc
python examples/ur16e_reach_stand_blocks_sim/world_pi05.py
The --action_scale flag lets you tune how aggressively the delta actions are applied, and --action_execute controls how many steps of each chunk run before re-querying the policy.


############################################

Terminal 1 — MPPI planner (env_isaaclab)


cd /home/paolo/Documents/isaaclabmpc
conda activate env_isaaclab
python examples/ur16e_reach_stand_blocks_sim/planner.py
Terminal 2 — Demo collector (env_isaaclab)


conda activate env_isaaclab
python examples/ur16e_reach_stand_blocks_sim/collect_demos.py \
    --n_episodes 100 --out_dir /tmp/ur16e_reach_blocks_demos
Convert to LeRobot format (openpi env)


conda activate openpi
cd /home/paolo/Documents/isaaclabmpc
python examples/ur16e_reach_stand_blocks_sim/convert_to_lerobot.py \
    --in_dir /tmp/ur16e_reach_blocks_demos
Train (openpi env)


cd /home/paolo/Documents/openpi
conda activate openpi
python scripts/train.py --config pi05_ur16e_push --exp_name run_02
# checkpoints → checkpoints/pi05_ur16e_push/run_02/
Serve policy (openpi env)


python scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_ur16e_push \
    --policy.dir checkpoints/pi05_ur16e_push/run_02/5000
Test (env_isaaclab)


conda activate env_isaaclab
cd /home/paolo/Documents/isaaclabmpc
python examples/ur16e_reach_stand_blocks_sim/world_pi05.py