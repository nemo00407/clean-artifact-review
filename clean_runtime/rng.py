"""TEE RNG emulation (spec section 4.6).

Training randomness is derived inside the trust boundary and is
deletion-independent by construction: the host has no interface to supply a
seed. The generator is counter-based (Philox), so execution is deterministic
for a fixed attested value r.
"""
import hashlib

import numpy as np


def derive_r(sk_seed: bytes, nonce: bytes, invocation_counter: int) -> bytes:
    """r = SHA256(sk_seed || "rng" || nonce || counter). sk_seed never leaves
    the (emulated) enclave; see attest module."""
    h = hashlib.sha256()
    h.update(sk_seed)
    h.update(b"rng")
    h.update(nonce)
    h.update(invocation_counter.to_bytes(8, "big"))
    return h.digest()


class TeeRng:
    """Deterministic PRNG handle exposed to the trainer.

    Exposes only sampling primitives - no seed setter exists, so host-chosen
    randomness cannot enter the input domain (attack vector 5)."""

    def __init__(self, r: bytes):
        if not isinstance(r, (bytes, bytearray)) or len(r) != 32:
            raise ValueError("r must be 32 bytes of attested randomness")
        self._r = bytes(r)
        key = int.from_bytes(self._r[:16], "big")
        self._gen = np.random.Generator(np.random.Philox(key=key))

    @property
    def r(self) -> bytes:
        return self._r

    def normal(self, shape, scale=1.0):
        return self._gen.normal(0.0, scale, size=shape).astype(np.float32)

    def permutation(self, n):
        return self._gen.permutation(n)
