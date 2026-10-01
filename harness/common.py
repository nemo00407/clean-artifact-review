"""Shared harness configuration and synthetic data generation (non-TCB).

The pretraining corpus is deletion-independent public text; user records
only ever enter shard datasets, never the base model. This mirrors the
trust story: W_base is a deployment-time constant, per-shard LoRA adapters
carry all user influence.
"""
import hashlib
import os
import pathlib

import numpy as np

# The deployment under test. Overridable because the enclave sees the host's
# storage at a mount of its own, and because that storage is untrusted in either
# location: nothing about the guarantee depends on where these bytes sit.
DATA_DIR = pathlib.Path(os.environ.get("CLEAN_DATA_DIR")
                        or pathlib.Path(__file__).resolve().parents[1] / "data")

SEALED_MOUNT = pathlib.Path("/sealed")

# Set by a harness that runs a scenario against a private copy of a deployment,
# so that what the scenario persists does not reach the real sealed state.
SEALED_OVERRIDE = None


def sealed_dir():
    """Where the platform keeps what it seals for the enclave: the signing
    seed, the per-shard record heads, the version counters.

    Under Gramine this is an encrypted mount whose key the hardware derives
    from MRENCLAVE, so an image with a different measurement cannot read it.
    Without one it is an ordinary file beside the data, which is what the
    prototype used before it had hardware to seal with. Resolved on each call
    because a sweep may point `DATA_DIR` at a different deployment.
    """
    if SEALED_OVERRIDE is not None:
        return SEALED_OVERRIDE
    return SEALED_MOUNT if SEALED_MOUNT.is_dir() else DATA_DIR

MODEL_CONFIG = {
    "n_layers": 2, "d_model": 128, "n_heads": 4, "d_ff": 256,
    "vocab": 256, "seq_len": 64, "lora_rank": 4, "lora_alpha": 8,
}
PRETRAIN = {"lr": 0.02, "momentum": 0.9, "epochs": 15, "batch_size": 32}
FINETUNE = {"lr": 0.05, "momentum": 0.9, "epochs": 4, "batch_size": 16}

K_SHARDS = 4
N_SLICES = 4
RECORDS_PER_SLICE = 32

_ADJ = ["quick", "calm", "bright", "small", "heavy", "quiet", "sharp", "plain"]
_NOUN = ["fox", "river", "stone", "cloud", "field", "lamp", "train", "door"]
_VERB = ["crosses", "follows", "watches", "carries", "passes", "circles"]
_PLACE = ["market", "harbor", "garden", "bridge", "valley", "station"]


def unlearn_config():
    """The public configuration c bound into the ARC."""
    return dict(MODEL_CONFIG, **FINETUNE)


def pad_to_seq(text, seq_len=64):
    raw = text.encode("ascii")[:seq_len]
    return raw + b" " * (seq_len - len(raw))


def gen_pretrain_corpus(n, seed=1234):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        s = (f"the {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} "
             f"the {rng.choice(_NOUN)} near the {rng.choice(_PLACE)} . ")
        rows.append(np.frombuffer(pad_to_seq(s), dtype=np.uint8))
    return np.stack(rows)


def gen_user_record(uid, rng):
    s = (f"user {uid:04d} likes item {rng.integers(100):02d} and item "
         f"{rng.integers(100):02d} at the {rng.choice(_PLACE)} . ")
    return np.frombuffer(pad_to_seq(s), dtype=np.uint8)


def shard_of(uid, k=K_SHARDS):
    digest = hashlib.sha256(str(uid).encode()).digest()
    return int.from_bytes(digest[:8], "big") % k


def harness_r(label: str) -> bytes:
    """Fixed, labeled randomness for reproducible harness runs."""
    return hashlib.sha256(b"clean-harness:" + label.encode()).digest()
