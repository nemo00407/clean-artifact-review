"""E1 - Non-interference two-world experiment (evaluation.md section 4).

World A: the shard contains Delta; the gateway physically excludes it.
World B: a counterfactual shard in which Delta never existed.
Same clean checkpoint, same attested randomness r, same config.
Claim under test (Theorem 2): H(W_A) == H(W_B), bit for bit.

Negative controls (the equality criterion must have discriminative power):
- LossMask: the ORIGINAL paper's design - the full retrain region including
  Delta is loaded and deleted rows' gradient contributions are masked out
  (grad ⊙ (1 - D_tombstone)). Fails the criterion: batch composition and
  loss normalization depend on Delta's presence, and Delta bytes enter
  runtime memory.
- Tainted: fine-tuning from the final (Delta-trained) checkpoint on the
  retained data, without clean rollback. Fails: residual influence.
"""
import json

import numpy as np

from clean_runtime import commit, gateway, kernel, runtime

from . import common, run_unlearn


def build_counterfactual(shard_id, delta, out_path):
    """Rewrite the shard as if the deleted records never existed."""
    src = common.DATA_DIR / f"shard_{shard_id}.npz"
    meta = json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.npz.meta.json").read_text())
    with np.load(src) as npz:
        index, salt, tokens = (npz["index"].copy(), npz["salt"].copy(),
                               npz["tokens"].copy())
    deleted = {int(ix) for ix, _ in delta}
    keep = np.array([i for i, ix in enumerate(index) if int(ix) not in deleted])
    removed_before = np.cumsum(
        [1 if int(ix) in deleted else 0 for ix in index])

    def shift(row):
        return int(row - (removed_before[row - 1] if row > 0 else 0))

    bounds = [[shift(lo), shift(hi)] for lo, hi in meta["slice_bounds"]]
    np.savez(out_path, index=index[keep], salt=salt[keep], tokens=tokens[keep])
    (common.DATA_DIR / (out_path.name + ".meta.json")).write_text(
        json.dumps({"shard_id": shard_id, "slice_bounds": bounds}))
    return out_path


def run_world_arc(shard_path, delta, cp_version, shard_id=0,
                  nonce_label="two-world"):
    """As run_world, but also returns the certificate the invocation issued,
    so that completeness can be checked on the same runs."""
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(nonce_label)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{shard_id}_v{cp_version}.npz",
        shard_path, delta, enclave.derive_r(nonce), config, nonce=nonce)
    new_lora, arc, _ = runtime.unlearn(inv, enclave)
    return commit.params_hash(new_lora), arc


def run_world(shard_path, delta, cp_version, shard_id=0,
              nonce_label="two-world"):
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(nonce_label)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{shard_id}_v{cp_version}.npz",
        shard_path, delta, enclave.derive_r(nonce), config, nonce=nonce)
    new_lora, _, _ = runtime.unlearn(inv, enclave)
    return commit.params_hash(new_lora)


def loss_mask_train(shard_id, delta, cp_version, nonce_label="two-world"):
    """The original design as an ablation: retrain region loaded WITH the
    deleted rows; their gradient contribution is subtracted per batch
    (algebraically identical to grad ⊙ (1 - D_tombstone) under mean loss)."""
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    enclave = run_unlearn.load_enclave()
    r = enclave.derive_r(common.harness_r(nonce_label))
    with np.load(common.DATA_DIR / f"shard_{shard_id}.npz") as npz:
        index, tokens = npz["index"].copy(), npz["tokens"].copy()
    meta = json.loads((common.DATA_DIR / f"shard_{shard_id}.npz.meta.json"
                       ).read_text())
    cp = gateway.load_checkpoint(common.DATA_DIR / f"cp_{shard_id}_v{cp_version}.npz")
    base = gateway.load_base(common.DATA_DIR / "base.npz").params
    lo = meta["slice_bounds"][cp.meta["slice_idx"] + 1][0]
    region = np.arange(lo, meta["slice_bounds"][-1][1])
    deleted = {int(ix) for ix, _ in delta}
    del_mask = np.array([int(index[r_]) in deleted for r_ in region])
    data = tokens[region]  # NOTE: includes the deleted rows

    params = {k: np.array(v, dtype=np.float32) for k, v in cp.lora.items()}
    from clean_runtime.rng import TeeRng
    rng = TeeRng(r)
    mom = {}
    for _ in range(config["epochs"]):
        order = rng.permutation(len(data))
        for s in range(0, len(data), config["batch_size"]):
            rows = order[s:s + config["batch_size"]]
            batch = data[rows].astype(np.int64)
            _, cache = kernel.forward(base, params, batch, config)
            grads = kernel.backward(base, params, cache, config)
            drows = rows[del_mask[rows]]
            if len(drows):
                _, dcache = kernel.forward(base, params,
                                           data[drows].astype(np.int64), config)
                dgrads = kernel.backward(base, params, dcache, config)
                frac = len(drows) / len(rows)
                grads = {k: grads[k] - frac * dgrads[k] for k in grads}
            kernel.sgd_step(params, grads, mom, config["lr"],
                            config["momentum"])
    return commit.params_hash(params)


def tainted_finetune(shard_id, delta, nonce_label="two-world"):
    """No clean rollback: fine-tune the final (Delta-trained) adapter on the
    retained records only."""
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    enclave = run_unlearn.load_enclave()
    r = enclave.derive_r(common.harness_r(nonce_label))
    with np.load(common.DATA_DIR / f"shard_{shard_id}.npz") as npz:
        index, tokens = npz["index"].copy(), npz["tokens"].copy()
    deleted = {int(ix) for ix, _ in delta}
    keep = np.array([i for i, ix in enumerate(index) if int(ix) not in deleted])
    cp = gateway.load_checkpoint(
        common.DATA_DIR / f"cp_{shard_id}_v{common.N_SLICES}.npz")  # tainted
    base = gateway.load_base(common.DATA_DIR / "base.npz").params
    lora = {k: np.array(v, dtype=np.float32) for k, v in cp.lora.items()}
    from clean_runtime.rng import TeeRng
    new_lora, _ = kernel.train(base, lora, tokens[keep], TeeRng(r), config)
    return commit.params_hash(new_lora)


SWEEP_SHARDS = [0, 1, 2, 3]
SWEEP_SLICES = [1, 2, 3]
SWEEP_COUNTS = [1, 2, 4, 8]
DETERMINISM_REPEATS = 20


def _pick(shard_id, slice_idx, count):
    if slice_idx is not None:
        return run_unlearn.pick_delta(shard_id, slice_idx, count)
    a = run_unlearn.pick_delta(shard_id, 2, (count + 1) // 2)
    b = run_unlearn.pick_delta(shard_id, 3, count // 2)
    return a + b


def _configs():
    """(label, shard, cp_version, slice_idx, count) over the grid, plus the
    spanning-slice cases the single-slice grid cannot express."""
    for shard in SWEEP_SHARDS:
        for sl in SWEEP_SLICES:
            for n in SWEEP_COUNTS:
                yield (f"s{shard} slice{sl} n{n}", shard, sl, sl, n)
    for shard in SWEEP_SHARDS:
        for n in (2, 6):
            yield (f"s{shard} slices2-3 n{n}", shard, 2, None, n)


def sweep():
    """Two-world equality and completeness over the whole grid."""
    rows = []
    for label, shard, cp_v, sl, n in _configs():
        delta = _pick(shard, sl, n)
        cf = build_counterfactual(
            shard, delta, common.DATA_DIR / f"shard_cf_{shard}.npz")
        h_a, arc = run_world_arc(
            common.DATA_DIR / f"shard_{shard}.npz", delta, cp_v,
            shard_id=shard)
        h_b = run_world(cf, [], cp_v, shard_id=shard)
        accept, reason, _ = run_unlearn.verify_run(
            arc, shard_id=shard, nonce_label="two-world", delta=delta,
            clean_version=cp_v)
        rows.append((label, shard, sl, n, h_a == h_b, accept == 1))
        print(f"[{label}] equal={h_a == h_b} verifies={accept == 1}"
              + ("" if accept == 1 else f" ({reason})"))
    return rows


def determinism(repeats=DETERMINISM_REPEATS):
    """Repeat one invocation unchanged; adapters and ARC bindings must be
    byte-identical every time (Assumption 3 as realized by the prototype)."""
    delta = _pick(0, 2, 1)
    seen = set()
    arc_seen = set()
    for _ in range(repeats):
        h, arc = run_world_arc(common.DATA_DIR / "shard_0.npz", delta, 2)
        seen.add(h)
        arc_seen.add(bytes(arc["signature"]))
    return repeats, len(seen), len(arc_seen)


def controls():
    """Validity ablations: the equality test must be able to fail. These are
    CLEAN with one mechanism disabled, not competing designs."""
    delta = _pick(0, 2, 1)
    cf = build_counterfactual(0, delta, common.DATA_DIR / "shard_cf_nc.npz")
    h_b = run_world(cf, [], 2)
    return [
        ("CLEAN-NoExclusion", loss_mask_train(0, delta, 2) != h_b),
        ("CLEAN-NoCleanCP", tainted_finetune(0, delta) != h_b),
    ]


def main():
    rows = sweep()
    reps, distinct, arc_distinct = determinism()
    ctrl = controls()

    equal = sum(1 for r in rows if r[4])
    verified = sum(1 for r in rows if r[5])
    lines = ["# E1 - Non-interference (two-world equality), swept", "",
             f"Each configuration runs the invocation twice: once on the real "
             f"shard with the gateway excluding Delta, once on a "
             f"counterfactual shard in which those records were never "
             f"written. Same certified checkpoint, same attested randomness, "
             f"same configuration. Equality is bitwise over the canonical "
             f"adapter serialization.", "",
             f"**{equal} of {len(rows)}** configurations equal; "
             f"**{verified} of {len(rows)}** issued a certificate the "
             f"verifier accepts (Theorem 1).", "",
             "| configuration | shard | slice | records | equal | verifies |",
             "|---|---:|---|---:|---|---|"]
    for label, shard, sl, n, eq, ok in rows:
        lines.append(f"| {label} | {shard} | {sl if sl is not None else '2-3'}"
                     f" | {n} | {'yes' if eq else 'NO'} | "
                     f"{'yes' if ok else 'NO'} |")
    lines += ["",
              f"Determinism: {reps} repeats of one invocation produced "
              f"{distinct} distinct adapter hash(es) and {arc_distinct} "
              f"distinct signature(s).", "",
              "Validity ablations (the criterion must be able to fail):", ""]
    for name, differs in ctrl:
        lines.append(f"- `{name}`: differs from the counterfactual world: "
                     f"**{differs}**")
    lines.append("")
    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "noninterference.md").write_text("\n".join(lines))

    dat = ["shard slice records equal verifies"]
    for label, shard, sl, n, eq, ok in rows:
        dat.append(f"{shard} {sl if sl is not None else 23} {n} "
                   f"{1 if eq else 0} {1 if ok else 0}")
    (out / "noninterference.dat").write_text("\n".join(dat) + "\n")
    (out / "noninterference.meta.json").write_text(json.dumps(
        {"configs": len(rows), "equal": equal, "verified": verified,
         "determinism_repeats": reps, "distinct_adapters": distinct,
         "distinct_signatures": arc_distinct,
         "controls": {n: bool(d) for n, d in ctrl}}, indent=1))
    print(f"-> {out / 'noninterference.md'} ({equal}/{len(rows)} equal, "
          f"{verified}/{len(rows)} verify, determinism {distinct} distinct)")


if __name__ == "__main__":
    main()
