import json

import numpy as np
import pytest

from clean_runtime import commit, dlog, gateway

T = 8  # token length for fixture records


def _build_fixture(tmp_path, cp_slice_idx=0):
    rng = np.random.default_rng(7)
    base_path = tmp_path / "base.npz"
    np.savez(base_path, w=rng.normal(size=(4, 4)).astype(np.float32))

    n = 12
    index = np.arange(100, 100 + n, dtype=np.uint32)
    salt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    tokens = rng.integers(0, 256, size=(n, T), dtype=np.uint8)
    shard_path = tmp_path / "shard_0.npz"
    np.savez(shard_path, index=index, salt=salt, tokens=tokens)
    meta = {"shard_id": 0, "slice_bounds": [[0, 4], [4, 8], [8, 12]]}
    (tmp_path / "shard_0.npz.meta.json").write_text(json.dumps(meta))

    cp_path = tmp_path / "cp_0_1.npz"
    np.savez(cp_path, a=np.zeros((4, 2), dtype=np.float32))
    cp_meta = {"shard_id": 0, "slice_idx": cp_slice_idx, "version": 1,
               "h_d_prefix": "00", "h_config_train": "00",
               "h_record": dlog.genesis(0).hex()}
    (tmp_path / "cp_0_1.npz.meta.json").write_text(json.dumps(cp_meta))
    return base_path, cp_path, shard_path, index, salt, tokens


def _delta_for(index, salt, tokens, rows):
    return [(int(index[r]), commit.record_hash(salt[r], tokens[r]).hex())
            for r in rows]


def _open(tmp_path, delta, cp_slice_idx=0):
    base_path, cp_path, shard_path, index, salt, tokens = _build_fixture(
        tmp_path, cp_slice_idx)
    return gateway.open_invocation(base_path, cp_path, shard_path, delta,
                                   b"\x01" * 32, {"lr": 0.1}), index, salt, tokens


def test_physical_exclusion(tmp_path):
    base_path, cp_path, shard_path, index, salt, tokens = _build_fixture(tmp_path)
    delta = _delta_for(index, salt, tokens, [5, 9])  # rows in slices 1 and 2
    inv = gateway.open_invocation(base_path, cp_path, shard_path, delta,
                                  b"\x01" * 32, {"lr": 0.1})
    # retrain region is slices 1..2 (rows 4..12) minus the two deleted rows
    assert len(inv.data) == 6
    assert not set(int(i) for i, _ in delta) & set(inv.data.indices.tolist())
    # deleted payload bytes are absent from the trainer-visible buffer
    for r in (5, 9):
        assert not any(np.array_equal(row, tokens[r]) for row in inv.data.tokens)
    # trained prefix rows (slice 0) are not part of the retraining input
    assert set(inv.data.indices.tolist()) == {int(index[r]) for r in range(4, 12)} - {int(index[5]), int(index[9])}


def test_handles_are_frozen(tmp_path):
    inv, *_ = _open(tmp_path, [])
    # the dataset is reached only through the verified read path, which has
    # no write entry point at all -- stronger than a read-only buffer
    with pytest.raises(TypeError):
        inv.data.tokens[0, 0] = 1
    with pytest.raises(ValueError):
        next(iter(inv.base.params.values()))[0, 0] = 1
    with pytest.raises(TypeError):
        inv.config["lr"] = 9


def test_dirty_checkpoint_rejected(tmp_path):
    base_path, cp_path, shard_path, index, salt, tokens = _build_fixture(
        tmp_path, cp_slice_idx=1)
    delta = _delta_for(index, salt, tokens, [5])  # row 5 is inside slice 1 prefix
    with pytest.raises(ValueError, match="not clean"):
        gateway.open_invocation(base_path, cp_path, shard_path, delta,
                                b"\x01" * 32, {})


def test_bad_delta_hash_rejected(tmp_path):
    base_path, cp_path, shard_path, index, salt, tokens = _build_fixture(tmp_path)
    delta = [(int(index[5]), "ab" * 32)]
    with pytest.raises(ValueError, match="hash mismatch"):
        gateway.open_invocation(base_path, cp_path, shard_path, delta,
                                b"\x01" * 32, {})


def test_unknown_delta_index_rejected(tmp_path):
    base_path, cp_path, shard_path, *_ = _build_fixture(tmp_path)
    with pytest.raises(ValueError, match="not present"):
        gateway.open_invocation(base_path, cp_path, shard_path,
                                [(999, "00" * 32)], b"\x01" * 32, {})


def test_dclean_commitment_matches_manifest_minus_delta(tmp_path):
    base_path, cp_path, shard_path, index, salt, tokens = _build_fixture(tmp_path)
    delta = _delta_for(index, salt, tokens, [6])
    inv = gateway.open_invocation(base_path, cp_path, shard_path, delta,
                                  b"\x01" * 32, {})
    keep = [r for r in range(4, 12) if r != 6]
    expected = commit.dataset_root(index[keep], salt[keep], tokens[keep])
    assert inv.data.h_dclean == expected
