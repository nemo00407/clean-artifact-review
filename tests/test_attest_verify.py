"""End-to-end attestation tests on a tiny model: completeness plus one
verifier rejection per binding (unit-level preview of the T5 attack suite)."""
import ast
import json
import pathlib

import numpy as np
import pytest

from clean_runtime import attest, commit, gateway, kernel, runtime
from clean_runtime.rng import TeeRng
from verifier import verify

CFG = {"n_layers": 1, "d_model": 8, "n_heads": 2, "d_ff": 16, "vocab": 17,
       "seq_len": 5, "lora_rank": 2, "lora_alpha": 4, "lr": 0.05,
       "momentum": 0.9, "epochs": 2, "batch_size": 4}
NONCE = b"\x21" * 32


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(3)
    enclave = attest.Enclave(attest.setup(b"test-entropy")[0],
                             gateway.runtime_measurement())

    base = kernel.base_init(CFG, TeeRng(b"\x05" * 32))
    np.savez(tmp_path / "base.npz", **base)

    n = 12
    index = np.arange(200, 200 + n, dtype=np.uint32)
    salt = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    tokens = rng.integers(0, CFG["vocab"], size=(n, CFG["seq_len"]),
                          dtype=np.uint8)
    np.savez(tmp_path / "shard_0.npz", index=index, salt=salt, tokens=tokens)
    bounds = [[0, 4], [4, 8], [8, 12]]
    (tmp_path / "shard_0.npz.meta.json").write_text(
        json.dumps({"shard_id": 0, "slice_bounds": bounds}))
    manifest = [[int(ix), commit.record_hash(s, t).hex()]
                for ix, s, t in zip(index, salt, tokens)]

    lora = kernel.lora_init(CFG, TeeRng(b"\x06" * 32))
    prefix = list(range(4))  # checkpoint has seen slice 0
    h_prefix = commit.dataset_root(index[prefix], salt[prefix], tokens[prefix])
    h_record = enclave.record_head(0)
    cert = enclave.sign_cp_cert(commit.params_hash(lora), h_prefix, 0, 0, 1,
                                h_record)
    meta = {"shard_id": 0, "slice_idx": 0, "version": 1,
            "h_d_prefix": h_prefix.hex(), "h_record": h_record.hex(),
            "cert": cert.hex(),
            "h_config_train": commit.config_hash(CFG).hex()}
    gateway.write_adapter(tmp_path / "cp.npz", lora, meta)

    delta = [(int(index[5]), commit.record_hash(salt[5], tokens[5]).hex())]
    inv = gateway.open_invocation(
        tmp_path / "base.npz", tmp_path / "cp.npz", tmp_path / "shard_0.npz",
        delta, enclave.derive_r(NONCE), CFG, nonce=NONCE, enclave=enclave)
    new_lora, arc, _ = runtime.unlearn(inv, enclave)
    expectations = dict(
        nonce=NONCE, delta=delta, manifest=manifest, slice_bounds=bounds,
        cp_meta=meta, cp_cert=cert, c_base=commit.params_hash(base),
        h_config=commit.config_hash(CFG), mrenclave=enclave.mrenclave)
    return enclave, arc, expectations, manifest


def _verify(enclave, arc, exp):
    return verify.verify_arc(enclave.pk, arc, **exp)


def test_completeness(world):
    enclave, arc, exp, _ = world
    accept, reason = _verify(enclave, arc, exp)
    assert accept == 1, reason


def test_replay_rejected(world):
    enclave, arc, exp, _ = world
    accept, reason = _verify(enclave, arc, dict(exp, nonce=b"\x22" * 32))
    assert accept == 0 and "signature" in reason


def test_skip_attack_rejected(world):
    # host returns the old adapter as the "new" one
    enclave, arc, exp, _ = world
    forged = dict(arc, h_l_new=arc["h_l_ckpt"])
    accept, reason = _verify(enclave, forged, exp)
    assert accept == 0 and "signature" in reason


def test_dataset_tampering_rejected(world):
    enclave, arc, exp, _ = world
    forged = dict(arc, h_dclean=commit.sha256(b"other-data"))
    accept, reason = _verify(enclave, forged, exp)
    assert accept == 0 and "D_clean" in reason


def test_delta_substitution_rejected(world):
    # verifier's deletion request differs from what the ARC was bound to
    enclave, arc, exp, manifest = world
    other = [(manifest[6][0], manifest[6][1])]
    accept, reason = _verify(enclave, arc, dict(exp, delta=other))
    assert accept == 0


def test_checkpoint_substitution_rejected(world):
    enclave, arc, exp, _ = world
    fake_meta = dict(exp["cp_meta"], version=2)
    accept, reason = _verify(enclave, arc, dict(exp, cp_meta=fake_meta))
    assert accept == 0 and "certificate" in reason


def test_unclean_checkpoint_rejected(world):
    # deletion target inside the checkpoint's trained prefix
    enclave, arc, exp, manifest = world
    dirty = [(manifest[2][0], manifest[2][1])]
    accept, reason = _verify(enclave, arc, dict(exp, delta=dirty))
    assert accept == 0 and "not clean" in reason


def test_wrong_mrenclave_rejected(world):
    enclave, arc, exp, _ = world
    accept, reason = _verify(enclave, arc,
                             dict(exp, mrenclave=commit.sha256(b"evil")))
    assert accept == 0


def test_verifier_is_independent():
    """The verifier must not import the runtime it is auditing."""
    vdir = pathlib.Path(__file__).resolve().parent.parent / "verifier"
    for path in vdir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                mod = getattr(node, "module", "") or ""
                assert "clean_runtime" not in mod
                assert all("clean_runtime" not in n for n in names)
