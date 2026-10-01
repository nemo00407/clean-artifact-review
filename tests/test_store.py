"""Verified read path (store.py): the temporal half of input closure.

These tests are the evidence that an input admitted at the gateway cannot be
changed underneath the state transition. Spatial confinement is checked
statically by test_confinement.py; this is the dynamic counterpart.
"""
import numpy as np
import pytest

from clean_runtime import store


def _rows(n=70, cols=8):
    return np.arange(n * cols, dtype=np.int64).reshape(n, cols)


def test_roundtrip_matches_admitted_rows():
    rows = _rows()
    view = store.admit(store.EnclaveResidentBackend(), "d", rows)
    assert len(view) == len(rows)
    assert view.n_blocks == 3  # 70 rows at 32 rows/block
    order = np.array([0, 5, 31, 32, 33, 69, 12])
    assert np.array_equal(view[order], rows[order])
    assert np.array_equal(view[3], rows[3])  # scalar indexing


def test_mutation_after_admission_is_detected():
    rows = _rows()
    backend = store.EnclaveResidentBackend()
    view = store.admit(backend, "d", rows)
    tampered = rows.copy()
    tampered[0, 0] += 1
    backend.put("d/0", store.encode_block(tampered[:store.BLOCK_ROWS]))
    with pytest.raises(store.BlockIntegrityError):
        view[np.array([0])]


def test_stale_block_replay_is_detected():
    """A block that was legitimate earlier is still refused once the
    invocation has committed to a different one -- checkpoint substitution
    moved onto the time axis."""
    first, second = _rows(), _rows() + 1000
    backend = store.EnclaveResidentBackend()
    stale = store.encode_block(first[:store.BLOCK_ROWS])
    view = store.admit(backend, "d", second)
    backend.put("d/0", stale)
    with pytest.raises(store.BlockIntegrityError):
        view[np.array([0, 1])]


def test_untouched_blocks_still_read_after_tampering_elsewhere():
    """Detection is per block, so the check is genuinely on the fetch path
    and not a one-shot validation at admission."""
    rows = _rows()
    backend = store.EnclaveResidentBackend()
    view = store.admit(backend, "d", rows)
    backend.put("d/2", store.encode_block(rows[:6] * 7))
    assert np.array_equal(view[np.array([0, 31])], rows[np.array([0, 31])])
    with pytest.raises(store.BlockIntegrityError):
        view[np.array([64])]


def test_noreverify_ablation_admits_the_substitution():
    """The -NoReverify ablation must actually be weaker; otherwise the
    security matrix would credit a check that does no work."""
    rows = _rows()
    backend = store.EnclaveResidentBackend()
    view = store.admit(backend, "d", rows, verify=False)
    tampered = rows.copy()
    tampered[0, 0] += 1
    backend.put("d/0", store.encode_block(tampered[:store.BLOCK_ROWS]))
    assert view[np.array([0])][0, 0] == rows[0, 0] + 1


def test_view_has_no_write_path():
    view = store.admit(store.EnclaveResidentBackend(), "d", _rows())
    with pytest.raises(TypeError):
        view[0] = 1
    with pytest.raises(IndexError):
        view[np.array([10 ** 6])]
