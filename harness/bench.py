"""Commitment cost and SISA rollback measurements.

E1  Commitment / attestation cost: composite C(W) vs full-model hash, with
    extrapolation to 7B/70B; CP_Cert / ARC signing; gateway exclusion cost.
E2  SISA rollback benefit: retrain cost vs deletion slice position.

All figures are real prototype measurements. Emitted to results/.
"""
import json
import time

import numpy as np

from clean_runtime import attest, commit, gateway, kernel, runtime
from clean_runtime.rng import TeeRng

from . import common, run_unlearn

RESULTS = None


def _load_base():
    with np.load(common.DATA_DIR / "base.npz") as npz:
        return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------- E7

def e7_commitment(config, base):
    # composite C(W) = H(C_base || H(L)) vs full-model hash H(W_base || L)
    lora = kernel.lora_init(config, TeeRng(b"\x01" * 32))
    lora_bytes = sum(commit.tensor_bytes(v).__sizeof__() for v in lora.values())

    def time_it(fn, reps=50):
        fn()
        t0 = time.time()
        for _ in range(reps):
            fn()
        return (time.time() - t0) / reps

    c_base = commit.params_hash(base)  # precomputed once at deployment

    def composite():
        # C_base is a deployment-time constant; only H(L) is recomputed
        return commit.sha256(c_base + commit.params_hash(lora))

    def full_model_hash():
        return commit.params_hash({**base, **lora})

    t_comp = time_it(composite)
    t_full = time_it(full_model_hash)

    base_bytes = sum(v.size * 4 for v in base.values())
    adapter_bytes = sum(v.size * 4 for v in lora.values())

    # hashing throughput (bytes/s) from the full-model measurement
    thrpt = base_bytes / t_full if t_full > 0 else 0

    # extrapolate to 7B / 70B (LoRA rank-16 adapter ~ 0.2% of params)
    ext = []
    for name, params in [("7B", 7e9), ("70B", 70e9)]:
        full_s = params * 4 / thrpt
        adapter_frac = 0.002
        comp_s = params * adapter_frac * 4 / thrpt
        ext.append((name, full_s, comp_s, full_s / comp_s))

    # attestation primitives
    enclave = run_unlearn.load_enclave()
    fields = runtime.arc_fields(
        _fake_inv(config, enclave), c_base, commit.params_hash(lora),
        commit.params_hash(lora))
    t_sign = time_it(lambda: enclave.sign_arc(fields), reps=200)
    return dict(t_comp=t_comp, t_full=t_full, base_bytes=base_bytes,
                adapter_bytes=adapter_bytes, thrpt=thrpt, ext=ext,
                t_sign=t_sign)


class _FakeInv:
    pass


def _fake_inv(config, enclave):
    inv = _FakeInv()
    inv.data = _FakeInv()
    inv.data.h_dclean = b"\x00" * 32
    inv.h_delta = b"\x00" * 32
    inv.h_record = b"\x00" * 32
    inv.rng = TeeRng(enclave.derive_r(b"\x00" * 32))
    inv.h_config = commit.config_hash(config)
    inv.cp = _FakeInv()
    inv.cp.meta = {"shard_id": 0, "version": 2}
    inv.nonce = b"\x00" * 32
    return inv


# ---------------------------------------------------------------- E8

def e8_rollback(config, base):
    """Retrain cost as a function of the deletion slice: deleting from a
    later slice rolls back to a later checkpoint and retrains fewer slices."""
    per_shard = {}
    for shard in range(common.K_SHARDS):
        for target_slice in range(common.N_SLICES):
            cp_version = target_slice  # roll back to CP before this slice
            delta = run_unlearn.pick_delta(shard, target_slice, 1)
            t0 = time.time()
            run_unlearn.run(shard_id=shard, target_slice=target_slice,
                            clean_version=cp_version, delta=delta,
                            out_name=None)
            per_shard.setdefault(target_slice, []).append(time.time() - t0)
    rows = []
    for target_slice in sorted(per_shard):
        xs = per_shard[target_slice]
        rows.append((target_slice, common.N_SLICES - target_slice,
                     float(np.median(xs)), float(min(xs)), float(max(xs)),
                     len(xs)))
    # from-scratch baseline: retrain all slices from CP_0, over every shard
    scratches = []
    for shard in range(common.K_SHARDS):
        delta = run_unlearn.pick_delta(shard, 0, 1)
        t0 = time.time()
        run_unlearn.run(shard_id=shard, target_slice=0, clean_version=0,
                        delta=delta, out_name=None)
        scratches.append(time.time() - t0)
    return rows, float(np.median(scratches))


# ---------------------------------------------------------------- main

def main():
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    base = _load_base()
    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)

    e7 = e7_commitment(config, base)
    lines = ["# E7 - Commitment and Attestation Cost", "",
             f"Prototype adapter {e7['adapter_bytes']/1024:.1f} KiB, base "
             f"{e7['base_bytes']/1024:.1f} KiB. Hashing throughput "
             f"{e7['thrpt']/1e6:.0f} MB/s.", "",
             "| operation | time |", "|---|---:|",
             f"| composite C(W) = H(C_base || H(L)) | {e7['t_comp']*1e6:.1f} us |",
             f"| full-model hash H(W_base || L) | {e7['t_full']*1e6:.1f} us |",
             f"| ARC signing (Ed25519) | {e7['t_sign']*1e6:.1f} us |", "",
             "Composite commitment only re-hashes the adapter each unlearning "
             "(C_base is a deployment-time constant). Extrapolated to LLM "
             "scale (rank-16 LoRA ~ 0.2% of parameters):", "",
             "| model | full-model hash | adapter-only C(W) | speedup |",
             "|---|---:|---:|---:|"]
    for name, full_s, comp_s, sp in e7["ext"]:
        lines.append(f"| {name} | {full_s*1e3:.0f} ms | {comp_s*1e3:.1f} ms | "
                     f"{sp:.0f}x |")
    lines.append("")
    (out / "commitment_cost.md").write_text("\n".join(lines))
    print(f"[E7] composite {e7['t_comp']*1e6:.1f}us vs full "
          f"{e7['t_full']*1e6:.1f}us; 70B speedup "
          f"{e7['ext'][1][3]:.0f}x")

    e8rows, scratch = e8_rollback(config, base)
    lines = ["# E8 - SISA Clean-Rollback Benefit", "",
             f"Deleting from a later slice rolls back to a later clean "
             f"checkpoint and retrains fewer slices. Each row is the median "
             f"over {common.K_SHARDS} shards.", "",
             "| deletion slice | slices retrained | median (s) | min | max |",
             "|---:|---:|---:|---:|---:|"]
    for sl, n, med, lo, hi, _k in e8rows:
        lines.append(f"| {sl} | {n} | {med:.2f} | {lo:.2f} | {hi:.2f} |")
    lines += ["",
              f"From-scratch (roll back to CP_0, retrain all "
              f"{common.N_SLICES} slices): {scratch:.2f}s median. SISA "
              f"localizes the retraining region to the slices after the "
              f"deletion point.", ""]
    (out / "rollback.dat").write_text(
        "slice slices_retrained median lo hi\n"
        + "\n".join(f"{sl} {n} {med:.4f} {lo:.4f} {hi:.4f}"
                    for sl, n, med, lo, hi, _k in e8rows)
        + f"\n# from_scratch {scratch:.4f}\n")
    (out / "rollback.md").write_text("\n".join(lines))
    print(f"[E8] slice0->{e8rows[0][2]:.2f}s slice3->{e8rows[-1][2]:.2f}s "
          f"scratch {scratch:.2f}s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
