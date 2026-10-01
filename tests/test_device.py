"""Device boundary (device.py): verification sits after the transfer.

The point of these tests is that the transport is allowed to be hostile.
Storage holds exactly what was admitted; only the copy arriving on the
device differs, so a check performed before the crossing would pass. The
mechanism is sound precisely because the check happens on the far side.
"""
import numpy as np
import pytest

from clean_runtime import device, store


class _FlippingPath:
    """Alters one byte of every block after the first transfer."""

    def __init__(self, trigger_after=0):
        self.transfers = 0
        self.trigger_after = trigger_after

    def transfer(self, blob):
        self.transfers += 1
        if self.transfers > self.trigger_after:
            b = bytearray(blob)
            b[-1] ^= 0x01
            return bytes(b)
        return blob


def _rows(n=70, cols=8):
    return np.arange(n * cols, dtype=np.int64).reshape(n, cols)


def test_honest_path_is_transparent():
    dev = device.DeviceContext()
    rows = _rows()
    view = store.admit(store.EnclaveResidentBackend(), "d", rows,
                       path=dev.path)
    assert np.array_equal(view[np.array([0, 40, 69])],
                          rows[np.array([0, 40, 69])])
    assert dev.path.transfers == 3
    assert dev.path.bytes_moved > 0


def test_tampering_in_flight_is_caught_on_the_device_side():
    backend = store.EnclaveResidentBackend()
    rows = _rows()
    path = _FlippingPath()
    view = store.admit(backend, "d", rows, path=path)
    # storage is untouched: the admitted blob still matches its commitment
    assert store.block_commitment(backend.get("d/0")) == view._commitments[0]
    with pytest.raises(store.BlockIntegrityError):
        view[np.array([0])]


def test_transfer_tampering_survives_the_noreverify_ablation():
    """Without the check the corrupted bytes reach the payload, which is what
    makes the ablation an honest demonstration rather than a formality."""
    rows = _rows()
    view = store.admit(store.EnclaveResidentBackend(), "d", rows,
                       verify=False, path=_FlippingPath())
    assert not np.array_equal(view[np.array([31])], rows[np.array([31])])


def test_resident_values_cross_the_boundary_once():
    dev = device.DeviceContext()
    obj = {"embed": np.zeros((4, 4), dtype=np.float32)}
    dev.place("base", obj)
    assert dev.has("base") and dev.resident("base") is obj
    assert "base" in dev.contents
    with pytest.raises(TypeError):
        dev.contents["base"] = None
