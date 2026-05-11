"""Convert collected .npz episodes to a LeRobot dataset.

Run this in the openpi conda environment (which has lerobot installed):

    conda activate openpi
    python examples/ur16e_reach_stand_blocks_sim/convert_to_lerobot.py \\
        --in_dir /tmp/ur16e_reach_blocks_demos \\
        --repo_id local/ur16e_reach_blocks

The resulting dataset is saved to:
    ~/.cache/huggingface/lerobot/local/ur16e_reach_blocks/
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "base_rgb": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_rgb": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "joints": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joints"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  type=str, required=True,
                        help="Directory containing episode_XXXX.npz files")
    parser.add_argument("--repo_id", type=str, default="local/ur16e_reach_blocks")
    args = parser.parse_args()

    episodes = sorted(glob.glob(os.path.join(args.in_dir, "episode_*.npz")))
    if not episodes:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.in_dir}")
    print(f"Found {len(episodes)} episodes in {args.in_dir}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="ur16e",
        fps=60,
        features=FEATURES,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    for ep_path in episodes:
        data = np.load(ep_path, allow_pickle=True)
        base_rgb  = data["base_rgb"]   # (T,224,224,3)
        wrist_rgb = data["wrist_rgb"]
        joints    = data["joints"]     # (T,6)
        actions   = data["actions"]    # (T,7)
        task      = str(data["task"])

        T = len(joints)
        for t in range(T):
            dataset.add_frame({
                "base_rgb":  base_rgb[t],
                "wrist_rgb": wrist_rgb[t],
                "joints":    joints[t].astype(np.float32),
                "actions":   actions[t].astype(np.float32),
                "task":      task,
            })
        dataset.save_episode()
        print(f"  {os.path.basename(ep_path)} — {T} frames")

    print(f"\nDone. Dataset saved to ~/.cache/huggingface/lerobot/{args.repo_id}/")


if __name__ == "__main__":
    main()
