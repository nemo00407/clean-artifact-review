"""Verified read path for invocation inputs that leave enclave memory.

An input admitted at the gateway is split into fixed-size blocks. The blocks
themselves are handed to a backend the host controls; only a constant-size
commitment per block is retained inside the enclave. Every later fetch
re-derives the commitment of the returned bytes and compares it against the
one recorded at admission, so an invocation either observes exactly the
values admitted at T3 or raises `BlockIntegrityError` and aborts before any
certificate is issued.

This is what makes the temporal half of input closure a mechanism rather
than an assumption. The host may alter, replay, or roll back any block it
stores; each such alteration is detected on the next fetch. What remains
assumed is only that the recorded commitments -- 32 bytes per block -- are
integrity-protected while resident, which is the baseline TEE guarantee
already relied on in the trust anchors.

While a value is resident in enclave memory it is protected by the same
guarantee and needs no re-derivation; verification is required exactly at
the boundary crossing, which is why it lives on the fetch path -- after the
transfer completes, on the device side, so that the transport itself needs
no trust (see `device`).
"""
import struct

import numpy as np

from . import commit

BLOCK_ROWS = 32


class BlockIntegrityError(Exception):
    """A block returned by the backend does not match the commitment recorded
    at admission. The invocation aborts without issuing a certificate."""


def encode_block(arr) -> bytes:
    """Canonical opaque encoding of one block; the backend sees only these."""
    a = np.ascontiguousarray(arr)
    dtype = a.dtype.str.encode()
    header = struct.pack("<II", len(dtype), a.ndim) + dtype
    header += b"".join(struct.pack("<I", d) for d in a.shape)
    return header + a.tobytes()


def decode_block(blob: bytes) -> np.ndarray:
    n_dtype, ndim = struct.unpack("<II", blob[:8])
    off = 8
    dtype = blob[off:off + n_dtype].decode()
    off += n_dtype
    shape = struct.unpack("<%dI" % ndim, blob[off:off + 4 * ndim])
    off += 4 * ndim
    return np.frombuffer(blob[off:], dtype=dtype).reshape(shape)


def block_commitment(blob: bytes) -> bytes:
    return commit.sha256(blob)


class EnclaveResidentBackend:
    """Backend for inputs that stay inside enclave memory for the whole
    invocation. Present so the read path is identical in both deployments:
    verification still runs, and is simply never expected to fail."""

    def __init__(self):
        self._blobs = {}

    def put(self, key, blob):
        self._blobs[key] = blob

    def get(self, key):
        return self._blobs[key]


class VerifiedRowView:
    """Row-addressable view over blocked rows held by an untrusted backend.

    Indexing with an array of row ids fetches the covering blocks, verifies
    each against its recorded commitment, and gathers the requested rows.
    `verify=False` exists solely to build the -NoReverify ablation of the
    evaluation; deployments must never construct it.
    """

    def __init__(self, backend, key_prefix, commitments, n_rows,
                 block_rows=BLOCK_ROWS, verify=True, path=None):
        self._backend = backend
        self._path = path
        self._prefix = key_prefix
        self._commitments = tuple(commitments)
        self._n = int(n_rows)
        self._block_rows = int(block_rows)
        self._verify = bool(verify)

    def __len__(self):
        return self._n

    @property
    def n_blocks(self):
        return len(self._commitments)

    def _fetch(self, b):
        blob = self._backend.get("%s/%d" % (self._prefix, b))
        if self._path is not None:
            blob = self._path.transfer(blob)
        if self._verify and block_commitment(blob) != self._commitments[b]:
            raise BlockIntegrityError(
                "block %d of %s does not match the commitment recorded at "
                "admission; it was altered in storage or in flight"
                % (b, self._prefix))
        return decode_block(blob)

    def __getitem__(self, rows):
        scalar = np.isscalar(rows) or (isinstance(rows, np.generic)
                                       and np.ndim(rows) == 0)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self._n):
            raise IndexError("row out of range")
        blocks = rows // self._block_rows
        out = None
        for b in np.unique(blocks):
            sel = np.nonzero(blocks == b)[0]
            block = self._fetch(int(b))
            gathered = block[rows[sel] - int(b) * self._block_rows]
            if out is None:
                out = np.empty((rows.size,) + gathered.shape[1:],
                               dtype=gathered.dtype)
            out[sel] = gathered
        if out is None:
            return np.empty((0,), dtype=np.int64)
        return out[0] if scalar else out


class RowWindow:
    """A contiguous range of an admitted view, still fetched through it.

    Slice-wise retraining reads one slice at a time. Materializing the slice
    would take its blocks out of the verified path once and for all; this
    forwards every access to the underlying view instead, so the commitment is
    still re-derived on the fetch that serves each batch.
    """

    def __init__(self, view, lo, hi):
        self._view = view
        self._lo, self._hi = int(lo), int(hi)

    def __len__(self):
        return self._hi - self._lo

    def __getitem__(self, rows):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= len(self)):
            raise IndexError("row out of window")
        return self._view[rows + self._lo]


def admit(backend, key_prefix, rows, block_rows=BLOCK_ROWS, verify=True,
          path=None):
    """Split `rows` into blocks, hand them to the backend, and return a view
    that re-verifies every block on fetch. Called by the gateway at admission;
    the commitments returned inside the view are the enclave-resident part."""
    rows = np.ascontiguousarray(rows)
    commitments = []
    for b, start in enumerate(range(0, len(rows), block_rows)):
        blob = encode_block(rows[start:start + block_rows])
        backend.put("%s/%d" % (key_prefix, b), blob)
        commitments.append(block_commitment(blob))
    return VerifiedRowView(backend, key_prefix, commitments, len(rows),
                           block_rows, verify, path)
