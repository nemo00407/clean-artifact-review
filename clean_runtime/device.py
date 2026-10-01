"""Device boundary: the GPU-TEE side of the runtime.

The trainer payload executes inside a `DeviceContext`, which models the
isolated memory of a confidential GPU. Values reach it only by transfer
across a `PCIePath`, and that path is deliberately *not* trusted for
integrity. Every block is checked against its admission commitment after
arrival, on the device side, so a transfer that delivers bytes other than
those admitted aborts the invocation instead of entering the state
transition. The driver, the DMA engine, and the interconnect therefore have
to be trusted only to deliver something, never to deliver the right thing.

Values already resident in device memory are not re-checked. That is the one
place the hardware anchor of Section 2.2 is load-bearing: enclave memory is
integrity-protected while it is enclave memory, which is why verification
belongs at the boundary crossing and nowhere else.

What is emulated here is the isolation of device memory, exactly as
MRENCLAVE and sealing are emulated elsewhere in this prototype. What is real
is the structure: a single ingress, verification positioned after it, and no
route by which the payload can reach back across the boundary.
"""
from types import MappingProxyType


class PCIePath:
    """The host-controlled transport between enclave and device.

    Untrusted. The honest implementation delivers what it was handed; a
    malicious one may alter, replay, or drop bytes in flight, and the
    commitment check on the far side is what makes that harmless.
    """

    def __init__(self):
        self.transfers = 0
        self.bytes_moved = 0

    def transfer(self, blob: bytes) -> bytes:
        self.transfers += 1
        self.bytes_moved += len(blob)
        return blob


class DeviceContext:
    """Isolated device memory plus the single path that reaches it."""

    def __init__(self, path=None):
        self.path = path if path is not None else PCIePath()
        self._resident = {}

    def place(self, name, value):
        """Move a value across the boundary once, to stay resident for the
        invocation. Used for state the payload reads on every batch and that
        is small enough to keep on the device -- the frozen base model."""
        self._resident[name] = value
        return value

    def resident(self, name):
        return self._resident[name]

    def has(self, name):
        return name in self._resident

    @property
    def contents(self):
        return MappingProxyType(self._resident)
