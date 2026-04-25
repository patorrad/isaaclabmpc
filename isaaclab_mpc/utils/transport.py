"""Torch tensor serialization for zerorpc transport.

Same interface as genesismpc/utils/transport.py — tensors are packed into
bytes using torch.save so that dtype, shape and device info are preserved.
"""

import io
import torch


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff, weights_only=False)
