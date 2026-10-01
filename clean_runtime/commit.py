"""Canonical serialization and cryptographic commitments (spec section 4).

All hashes are SHA-256. Signed messages use domain-separation tags and
length-prefixed field encoding to rule out concatenation ambiguity.
"""
import hashlib
import json
import struct

import numpy as np


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def tensor_bytes(arr) -> bytes:
    """Canonical bytes of a tensor: uint32 LE ndim + shape header, then
    little-endian float32 payload in C order."""
    a = np.ascontiguousarray(arr, dtype="<f4")
    header = struct.pack("<I", a.ndim) + b"".join(
        struct.pack("<I", d) for d in a.shape
    )
    return header + a.tobytes()


def params_hash(params: dict) -> bytes:
    """Commitment over a named parameter set, keys in sorted order."""
    h = hashlib.sha256()
    for name in sorted(params):
        encoded = name.encode()
        h.update(struct.pack("<I", len(encoded)) + encoded)
        h.update(tensor_bytes(params[name]))
    return h.digest()


def canonical_config(config: dict) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def config_hash(config: dict) -> bytes:
    return sha256(canonical_config(config))


def record_hash(salt: bytes, payload: bytes) -> bytes:
    """Per-record commitment; the salt keeps record hashes dictionary-attack
    resistant so manifests can be shared with auditors."""
    return sha256(bytes(salt) + bytes(payload))


def merkle_root(leaves) -> bytes:
    """Binary SHA-256 Merkle tree; an odd node is paired with itself."""
    if len(leaves) == 0:
        return sha256(b"")
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [sha256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def dataset_root(indices, salts, payloads) -> bytes:
    """Merkle root over record hashes ordered by ascending record index."""
    order = np.argsort(np.asarray(indices), kind="stable")
    leaves = [record_hash(salts[i], payloads[i]) for i in order]
    return merkle_root(leaves)


def hashes_root(indices, record_hashes) -> bytes:
    """Merkle root over pre-computed record hashes ordered by index (used
    for H(Delta), where the runtime never sees the deleted payloads)."""
    order = np.argsort(np.asarray(indices), kind="stable")
    leaves = [bytes(record_hashes[i]) for i in order]
    return merkle_root(leaves)


def encode_signed_message(tag: bytes, fields) -> bytes:
    """Domain-separated, length-prefixed message encoding for signatures."""
    out = bytearray(tag)
    for field in fields:
        f = bytes(field)
        out += struct.pack(">I", len(f)) + f
    return bytes(out)
