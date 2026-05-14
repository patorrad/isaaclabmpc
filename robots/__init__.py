"""Portable path export for the stand URDF asset.

Uses os.path so this resolves correctly regardless of working directory or
install location (mirrors the pattern in assets/robots/ur16e.py).
"""
import os

STAND_URDF_PATH: str = os.path.join(os.path.dirname(__file__), "stand", "stand.urdf")
