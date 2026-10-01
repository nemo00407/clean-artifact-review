"""SISA initial training at 1B parameters, with the GPU payload in the slot.

The same lifecycle `harness/sisa_train.py` runs for the 0.9M NumPy model:
build the shards, train each one slice by slice, certify the state at every
boundary. Two things differ, and only two.

The payload is `harness/gpu_kernel.py`: LoRA over TinyLlama-1.1B on the GPU.
Everything upstream of it -- the gateway, the verified read path, the
commitments, the certificates -- is the runtime's own code, unchanged.

The base model is not written into `base.npz`. Equation (1) commits to it
once at deployment as C_base and re-hashes only the adapter thereafter, so
what the file carries is that commitment; the payload checks the weights it
loads against it. At 1.1e9 parameters this is the difference between a 4.4 GB
artifact per checkpoint and a 2.3 MB one.
"""
import json
import time

import numpy as np

from clean_runtime import attest, commit, gateway
from clean_runtime.rng import TeeRng

from . import common, gpu_kernel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
K_SHARDS = 2                 # the claim is bitwise equality at scale, not a large N
N_SLICES = 4
RECORDS_PER_SLICE = 8
SEQ_LEN = 256

# The 1B artifacts live beside the 0.9M ones, not on top of them, and the layout
# is smaller. Both go onto `common`, because the helpers that address a record
# by slice -- `run_unlearn.pick_delta` above all -- read the layout from there,
# and a helper that reads the 0.9M layout while addressing a 1B shard indexes
# past the end of the manifest and selects nothing at all.
common.DATA_DIR = common.DATA_DIR.parent / "data-1b"
common.K_SHARDS = K_SHARDS
common.N_SLICES = N_SLICES
common.RECORDS_PER_SLICE = RECORDS_PER_SLICE


def config(c_base_hex):
    return {"model_id": MODEL_ID, "c_base_hex": c_base_hex,
            "lora_rank": 4, "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "batch_size": 2, "epochs": 1, "lr": 1e-4, "seq_len": SEQ_LEN}


def _record_tokens(uid, rng, vocab=32000):
    """One user's record. Token ids rather than text: the property under test
    is bitwise equality of the adapter, which does not turn on what the
    corpus says."""
    return rng.integers(1, vocab, size=SEQ_LEN, dtype=np.int64)


def _write_shard(k, records, out_dir):
    index = np.array([r["index"] for r in records], dtype=np.uint32)
    salt = np.stack([r["salt"] for r in records])
    tokens = np.stack([r["tokens"] for r in records])
    np.savez(out_dir / f"shard_{k}.npz", index=index, salt=salt, tokens=tokens)
    bounds = [[j * RECORDS_PER_SLICE, (j + 1) * RECORDS_PER_SLICE]
              for j in range(N_SLICES)]
    (out_dir / f"shard_{k}.npz.meta.json").write_text(
        json.dumps({"shard_id": k, "slice_bounds": bounds}))
    (out_dir / f"shard_{k}.manifest.json").write_text(json.dumps(
        [[int(ix), commit.record_hash(s, t).hex()]
         for ix, s, t in zip(index, salt, tokens)]))
    return index, salt, tokens


def _save_cp(out, enclave, k, version, slice_idx, lora, prefix_rows,
             index, salt, tokens, cfg):
    h_prefix = commit.dataset_root(index[prefix_rows], salt[prefix_rows],
                                   tokens[prefix_rows])
    cert = enclave.sign_cp_cert(commit.params_hash(lora), h_prefix, k,
                                slice_idx, version, enclave.record_head(k))
    gateway.write_adapter(out / f"cp_{k}_v{version}.npz", lora, {
        "shard_id": k, "slice_idx": slice_idx, "version": version,
        "h_d_prefix": h_prefix.hex(),
        "h_record": enclave.record_head(k).hex(), "cert": cert.hex(),
        "h_config_train": commit.config_hash(cfg).hex()})


def main():
    out = common.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    digest, root = gpu_kernel.base_commitment(MODEL_ID)
    cfg = config(digest.hex())
    # base.npz carries C_base, not the weights: see the module docstring.
    base = {"c_base": np.frombuffer(digest, dtype=np.uint8).copy()}
    np.savez(out / "base.npz", **base)
    (out / "config.json").write_text(json.dumps(cfg, sort_keys=True))
    print(f"[base] {MODEL_ID}\n       C_base {digest.hex()[:32]}\n       {root}")

    mre = gateway.runtime_measurement()
    sealed, pk = attest.setup(b"clean-1b-entropy")
    (common.sealed_dir() / "enclave.sealed").write_bytes(sealed)
    enclave = attest.Enclave(sealed, mre)
    (out / "deployment.json").write_text(json.dumps({
        "pk": pk.hex(), "mrenclave": mre.hex(),
        "c_base": commit.params_hash(base).hex(),
        "h_config": commit.config_hash(cfg).hex()}, sort_keys=True, indent=1))

    per_shard = N_SLICES * RECORDS_PER_SLICE
    shards = {k: [] for k in range(K_SHARDS)}
    rng = np.random.default_rng(99)
    uid = idx = 0
    while any(len(v) < per_shard for v in shards.values()):
        k = common.shard_of(uid, K_SHARDS)
        if len(shards[k]) < per_shard:
            shards[k].append({"index": idx, "uid": uid,
                              "salt": rng.integers(0, 256, 16, dtype=np.uint8),
                              "tokens": _record_tokens(uid, rng)})
            idx += 1
        uid += 1

    for k in range(K_SHARDS):
        index, salt, tokens = _write_shard(k, shards[k], out)
        lora = gpu_kernel.lora_init(cfg, TeeRng(common.harness_r(f"lora-init-{k}")))
        _save_cp(out, enclave, k, 0, -1, lora, [], index, salt, tokens, cfg)
        for j in range(N_SLICES):
            lo, hi = j * RECORDS_PER_SLICE, (j + 1) * RECORDS_PER_SLICE
            lora, losses = gpu_kernel.train(
                base, lora, tokens[lo:hi],
                TeeRng(common.harness_r(f"train-{k}-{j}")), cfg)
            _save_cp(out, enclave, k, j + 1, j, lora, list(range(hi)),
                     index, salt, tokens, cfg)
            print(f"[shard {k}] slice {j}: loss {losses[-1]:.4f}", flush=True)

    # The version counter is enclave state: `next_version` must never hand out a
    # number a certificate has already carried, so a host that could rewind it
    # could get two different states certified under one version. It goes where
    # the sealed key goes, which on a platform that seals is not the data
    # directory.
    (common.sealed_dir() / "enclave.versions.json").write_text(
        json.dumps({str(k): v for k, v in enclave.max_versions().items()}))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
