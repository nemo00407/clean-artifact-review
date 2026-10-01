"""Handle-to-kernel adaptation (clean-state selection, deletion-free input).

Loader functions accept only gateway handle types - never paths or raw
bytes - so the trainer cannot be pointed at undeclared state.
"""
import numpy as np

from .gateway import BaseModelHandle, CheckpointHandle, DatasetHandle


def base_params(handle):
    if not isinstance(handle, BaseModelHandle):
        raise TypeError("expected BaseModelHandle")
    return dict(handle.params)


def checkpoint_lora(handle):
    """Mutable float32 copy of the clean checkpoint adapter for training."""
    if not isinstance(handle, CheckpointHandle):
        raise TypeError("expected CheckpointHandle")
    return {k: np.array(v, dtype=np.float32) for k, v in handle.lora.items()}


def training_tokens(handle):
    if not isinstance(handle, DatasetHandle):
        raise TypeError("expected DatasetHandle")
    return handle.tokens
