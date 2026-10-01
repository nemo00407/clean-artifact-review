"""SISA initial-training driver (non-TCB, emulates the trusted lifecycle).

Pretrains the frozen base model on public text, then trains one LoRA
adapter per SISA shard slice-by-slice, materializing a checkpoint after
every slice with its dataset-prefix commitment - the clean rollback points
consumed by the unlearning runtime. In deployment this phase runs inside
the TEE (Trusted Initialization assumption); certificates over these
checkpoints are attached in Stage T3.
"""
import json
import time

import numpy as np

from clean_runtime import attest, commit, gateway, kernel
from clean_runtime.rng import TeeRng

from . import common


def _write_shard(k, records, out_dir):
    index = np.array([r["index"] for r in records], dtype=np.uint32)
    salt = np.stack([r["salt"] for r in records])
    tokens = np.stack([r["tokens"] for r in records])
    path = out_dir / f"shard_{k}.npz"
    np.savez(path, index=index, salt=salt, tokens=tokens)
    n = common.RECORDS_PER_SLICE
    bounds = [[j * n, (j + 1) * n] for j in range(common.N_SLICES)]
    (out_dir / f"shard_{k}.npz.meta.json").write_text(
        json.dumps({"shard_id": k, "slice_bounds": bounds}))
    manifest = [[int(ix), commit.record_hash(s, t).hex()]
                for ix, s, t in zip(index, salt, tokens)]
    (out_dir / f"shard_{k}.manifest.json").write_text(json.dumps(manifest))
    return index, salt, tokens


def main():
    out = common.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Pretrain the base model on deletion-independent public text.
    pre_cfg = dict(common.MODEL_CONFIG, **common.PRETRAIN)
    base = kernel.base_init(pre_cfg, TeeRng(common.harness_r("base-init")))
    corpus = common.gen_pretrain_corpus(256)
    base, pre_losses = kernel.train(base, None, corpus,
                                    TeeRng(common.harness_r("pretrain")),
                                    pre_cfg, train_base=True)
    np.savez(out / "base.npz", **base)
    cfg = common.unlearn_config()
    (out / "config.json").write_text(json.dumps(cfg, sort_keys=True))
    print(f"[base] pretrain loss {pre_losses[0]:.3f} -> {pre_losses[-1]:.3f}")

    # 1b. Enclave setup: sealed key, runtime measurement, public deployment
    # commitments (the verifier's trust anchors).
    mre = gateway.runtime_measurement()
    sealed, pk = attest.setup(b"clean-demo-entropy")
    (common.sealed_dir() / "enclave.sealed").write_bytes(sealed)
    enclave = attest.Enclave(sealed, mre)
    (out / "deployment.json").write_text(json.dumps({
        "pk": pk.hex(), "mrenclave": mre.hex(),
        "c_base": commit.params_hash(base).hex(),
        "h_config": commit.config_hash(cfg).hex()}, sort_keys=True, indent=1))

    # 2. Assign synthetic users to shards until every slice is full.
    per_shard = common.N_SLICES * common.RECORDS_PER_SLICE
    shard_records = {k: [] for k in range(common.K_SHARDS)}
    data_rng = np.random.default_rng(99)
    uid, next_index = 0, 0
    while any(len(v) < per_shard for v in shard_records.values()):
        k = common.shard_of(uid)
        if len(shard_records[k]) < per_shard:
            shard_records[k].append({
                "index": next_index, "uid": uid,
                "salt": data_rng.integers(0, 256, 16, dtype=np.uint8),
                "tokens": common.gen_user_record(uid, data_rng)})
            next_index += 1
        uid += 1

    # 3. Per shard: slice-by-slice LoRA training with checkpointing.
    fin_cfg = common.unlearn_config()
    for k in range(common.K_SHARDS):
        index, salt, tokens = _write_shard(k, shard_records[k], out)
        lora = kernel.lora_init(fin_cfg, TeeRng(common.harness_r(f"lora-init-{k}")))
        _save_cp(out, enclave, k, 0, -1, lora, [], index, salt, tokens, fin_cfg)
        for j in range(common.N_SLICES):
            lo, hi = j * common.RECORDS_PER_SLICE, (j + 1) * common.RECORDS_PER_SLICE
            lora, losses = kernel.train(
                base, lora, tokens[lo:hi],
                TeeRng(common.harness_r(f"train-{k}-{j}")), fin_cfg)
            _save_cp(out, enclave, k, j + 1, j, lora, list(range(hi)),
                     index, salt, tokens, fin_cfg)
            print(f"[shard {k}] slice {j}: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    # The version counter is enclave state: `next_version` must never hand out a
    # number a certificate has already carried, so a host that could rewind it
    # could get two different states certified under one version. It goes where
    # the sealed key goes, which on a platform that seals is not the data
    # directory.
    (common.sealed_dir() / "enclave.versions.json").write_text(
        json.dumps({str(k): v for k, v in enclave.max_versions().items()}))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


def _save_cp(out, enclave, k, version, slice_idx, lora, prefix_rows,
             index, salt, tokens, cfg):
    h_prefix = commit.dataset_root(index[prefix_rows],
                                   salt[prefix_rows], tokens[prefix_rows])
    h_cp = commit.params_hash(lora)
    h_record = enclave.record_head(k)
    cert = enclave.sign_cp_cert(h_cp, h_prefix, k, slice_idx, version,
                                h_record)
    meta = {"shard_id": k, "slice_idx": slice_idx, "version": version,
            "h_d_prefix": h_prefix.hex(), "h_record": h_record.hex(),
            "cert": cert.hex(),
            "h_config_train": commit.config_hash(cfg).hex()}
    gateway.write_adapter(out / f"cp_{k}_v{version}.npz", lora, meta)


if __name__ == "__main__":
    main()
