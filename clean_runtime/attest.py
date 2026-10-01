"""Attestation emulation: enclave identity, key sealing, certificate issuance.

Emulates the hardware attestation primitives of a CPU/GPU TEE (spec section
4.5). MRENCLAVE is the SHA-256 measurement of the runtime source tree
(computed by the gateway, the I/O mediation point) and stands in for the
hardware platform measurement; the Ed25519 signing key emulates the
hardware-sealed enclave key sk_TEE. Certificates are issued strictly by
this module: the ARC only after the runtime's state transition returns,
the CP_Cert only at checkpoint creation time.
"""
import json
import struct

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from . import commit, dlog
from .rng import derive_r

CP_TAG = b"CLEAN-CP-v1"
ARC_TAG = b"CLEAN-ARC-v1"


def _int(v: int) -> bytes:
    return struct.pack(">q", int(v))


def setup(entropy: bytes):
    """Emulated hardware Setup(1^lambda): derive the sealed enclave secret
    and publish the verification key. Returns (sealed_state, pk_bytes)."""
    sk_seed = commit.sha256(bytes(entropy) + b"clean-sk-seed")
    sealed = json.dumps({"sk_seed": sk_seed.hex()}).encode()
    key = Ed25519PrivateKey.from_private_bytes(sk_seed)
    pk = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return sealed, pk


def cp_cert_message(h_cp, h_d_prefix, h_record, shard_id, slice_idx, version,
                    mrenclave):
    return commit.encode_signed_message(
        CP_TAG, [h_cp, h_d_prefix, h_record, _int(shard_id), _int(slice_idx),
                 _int(version), mrenclave])


def arc_message(fields, mrenclave):
    """Canonical ARC payload in the field order of spec section 4.4."""
    return commit.encode_signed_message(
        ARC_TAG, [fields["c_base"], fields["h_l_ckpt"], fields["h_l_new"],
                  fields["h_dclean"], fields["h_delta"], fields["h_record"],
                  fields["h_r"], fields["h_config"],
                  _int(fields["shard_id"]), _int(fields["cp_version"]),
                  mrenclave, fields["nonce"]])


class Enclave:
    """In-enclave signing context reconstructed from sealed state.

    Also holds the head of each shard's deletion record. Only the head is
    enclave-resident; the body lives in host storage and is checked against
    this value before it is believed. The platform persists the head exactly
    as it persists the sealed key, and this module performs no file access.
    """

    def __init__(self, sealed_state: bytes, mrenclave: bytes,
                 record_heads=None, max_versions=None):
        state = json.loads(bytes(sealed_state))
        self._sk_seed = bytes.fromhex(state["sk_seed"])
        self._key = Ed25519PrivateKey.from_private_bytes(self._sk_seed)
        self.mrenclave = bytes(mrenclave)
        self.pk = self._key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw)
        self._record_head = {int(k): (bytes.fromhex(v)
                                      if isinstance(v, str) else bytes(v))
                             for k, v in dict(record_heads or {}).items()}
        self._max_version = {int(k): int(v)
                             for k, v in dict(max_versions or {}).items()}

    def next_version(self, shard_id) -> int:
        """A version number no certificate of this shard has carried before.

        Unlearning certifies the states it produces, and those coexist with
        the ladder it resumed from, so the number has to come from a counter
        the enclave advances and never rewinds.
        """
        v = self._max_version.get(int(shard_id), -1) + 1
        self._max_version[int(shard_id)] = v
        return v

    def note_version(self, shard_id, version) -> None:
        """Record a version issued outside next_version (initial training)."""
        k = int(shard_id)
        self._max_version[k] = max(self._max_version.get(k, -1), int(version))

    def max_versions(self) -> dict:
        return dict(self._max_version)

    def record_head(self, shard_id) -> bytes:
        """Head of the shard's deletion record; genesis if it has none."""
        return self._record_head.get(int(shard_id), dlog.genesis(shard_id))

    def record_heads(self) -> dict:
        """Export the heads for the platform to persist."""
        return {k: v.hex() for k, v in self._record_head.items()}

    def append_deletion(self, shard_id, record_hashes) -> bytes:
        """Advance the head over one invocation's deletion set."""
        head = dlog.extend(self.record_head(shard_id), record_hashes)
        self._record_head[int(shard_id)] = head
        return head

    def derive_r(self, nonce: bytes, invocation_counter: int = 0) -> bytes:
        """Attested training randomness; the host has no seed interface."""
        return derive_r(self._sk_seed, nonce, invocation_counter)

    def sign_cp_cert(self, h_cp, h_d_prefix, shard_id, slice_idx,
                     version, h_record=None) -> bytes:
        if h_record is None:
            h_record = self.record_head(shard_id)
        self.note_version(shard_id, version)
        return self._key.sign(cp_cert_message(
            h_cp, h_d_prefix, h_record, shard_id, slice_idx, version,
            self.mrenclave))

    def sign_arc(self, fields) -> bytes:
        return self._key.sign(arc_message(fields, self.mrenclave))
