"""Host-controlled block storage (NOT part of the TCB).

Stands in for the untrusted storage the shard is swapped out to when it does
not stay resident in enclave memory. The honest backend returns what it was
given; `RollbackBackend` models a malicious host that serves a stale version
of a block after the invocation has already committed to the current one.

This is the adversary's side of the boundary, so it lives in the harness and
is never counted in the trusted computing base -- exactly like the red-team
code in `attacks.py` and the weakened verifier of the ablations.
"""


class HonestBackend:
    def __init__(self):
        self._blobs = {}

    def put(self, key, blob):
        self._blobs[key] = blob

    def get(self, key):
        return self._blobs[key]


class RollbackBackend:
    """Serves the admitted blocks until `trigger_after` fetches have gone by,
    then substitutes a stale blob for `target_key` on every later fetch.

    The stale blob is one the enclave itself produced earlier -- the same
    shape of attack as checkpoint substitution, moved onto the time axis:
    the bytes were legitimate at some point, just not at the point they are
    served. Nothing about them is malformed, so only a commitment recorded
    at admission distinguishes them.
    """

    def __init__(self, target_key, stale_blob, trigger_after=1):
        self._blobs = {}
        self.target_key = target_key
        self.stale_blob = stale_blob
        self.trigger_after = trigger_after
        self.fetches = 0
        self.substitutions = 0
        self.first_substitution_at = None

    def put(self, key, blob):
        self._blobs[key] = blob

    def get(self, key):
        self.fetches += 1
        if key == self.target_key and self.fetches > self.trigger_after:
            self.substitutions += 1
            if self.first_substitution_at is None:
                self.first_substitution_at = self.fetches
            return self.stale_blob
        return self._blobs[key]


class TamperingPath:
    """A PCIe path that alters bytes in flight (NOT part of the TCB).

    Models a compromised driver, a DMA engine writing into the staging
    buffer, or an interconnect the host controls. Storage is untouched: the
    backend still holds exactly what was admitted, and only the copy that
    arrives on the device differs. Nothing on the storage side can detect
    this, which is why the commitment check has to sit after the transfer
    rather than before it.
    """

    def __init__(self, trigger_after=1, flip_at=-1):
        self.transfers = 0
        self.bytes_moved = 0
        self.trigger_after = trigger_after
        self.flip_at = flip_at
        self.tampered = 0
        self.first_tamper_at = None

    def transfer(self, blob: bytes) -> bytes:
        self.transfers += 1
        self.bytes_moved += len(blob)
        if self.transfers > self.trigger_after:
            self.tampered += 1
            if self.first_tamper_at is None:
                self.first_tamper_at = self.transfers
            b = bytearray(blob)
            b[self.flip_at] ^= 0x01
            return bytes(b)
        return blob
