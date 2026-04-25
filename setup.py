from setuptools import setup, find_packages

setup(
    name="isaaclab_mpc",
    version="0.1.0",
    packages=find_packages(include=["isaaclab_mpc", "isaaclab_mpc.*", "robots"]),
    python_requires=">=3.11",
    install_requires=[
        # mppi_torch must be installed separately (editable install from
        # /home/paolo/Documents/mppi_torch):
        #   pip install -e /home/paolo/Documents/mppi_torch
        "zerorpc",
        "pyyaml",
        "torch",
    ],
    description="MPPI controller with Isaac Lab as the dynamics model",
)
