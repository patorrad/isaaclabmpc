import os as _os

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
STAND_URDF_PATH = _os.path.normpath(_os.path.join(_THIS_DIR, "stand", "stand.urdf"))

from robots.ur16e import UR16E_CFG  # noqa: F401
