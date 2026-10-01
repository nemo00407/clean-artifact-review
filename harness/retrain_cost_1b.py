"""E18 - Retraining cost against deletion position, at 1B parameters.

E8 measured this on the 0.9M NumPy model: retraining falls by a constant amount
per slice the deletion spares, because a deletion later in the shard resumes
from a later certified checkpoint. The shape is SISA's and the payload is not
supposed to matter, so the question is whether the relationship survives a real
GPU trainer, where a fixed per-invocation cost -- kernel selection, allocator
warm-up, the base model's residency -- sits underneath the part that scales.

That fixed cost is also why the first invocation is measured and thrown away.
CUDA selects and caches kernels on first use, so an unwarmed invocation reports
the cache miss rather than the retraining, and reports it only for whichever
configuration happened to run first.
"""
import json
import time

import numpy as np

from clean_runtime import runtime

from . import common, gpu_kernel, run_unlearn, sisa_train_1b   # noqa: F401

runtime.kernel = gpu_kernel          # importing sisa_train_1b sets the layout

REPS = 3


def _invocation(shard, target_slice):
    """One World-A invocation; returns the seconds spent in the state
    transition, which is what `run_unlearn.run` times."""
    delta = run_unlearn.pick_delta(shard, target_slice, 1)
    assert len(delta) == 1, f"no record selected in slice {target_slice}"
    _lora, _arc, _losses, elapsed = run_unlearn.run(
        shard_id=shard, target_slice=target_slice, clean_version=target_slice,
        delta=delta, out_name=None, nonce_label=f"e18-s{shard}-sl{target_slice}")
    return elapsed


def sweep():
    print("warm-up invocation (discarded)", flush=True)
    _invocation(0, common.N_SLICES - 1)

    per_slice = {}
    for shard in range(common.K_SHARDS):
        for target_slice in range(common.N_SLICES):
            for _ in range(REPS):
                el = _invocation(shard, target_slice)
                per_slice.setdefault(target_slice, []).append(el)
            print(f"  shard {shard} slice {target_slice}: "
                  f"{per_slice[target_slice][-REPS:]}", flush=True)
    rows = []
    for sl in sorted(per_slice):
        xs = per_slice[sl]
        rows.append({"slice": sl, "retrained": common.N_SLICES - sl,
                     "median": float(np.median(xs)), "min": float(min(xs)),
                     "max": float(max(xs)), "n": len(xs)})
    return rows


def main():
    rows = sweep()
    cfg = json.loads((common.DATA_DIR / "config.json").read_text())
    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)

    # The saving per slice spared, as the slope of a straight line through the
    # medians. Reported with the residual so that "constant" is a measurement
    # and not a description of the shape.
    xs = np.array([r["retrained"] for r in rows], dtype=float)
    ys = np.array([r["median"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = float(np.max(np.abs(ys - (slope * xs + intercept))))

    lines = ["# E18 - Retraining cost against deletion position at 1B", "",
             f"`{cfg['model_id']}`, LoRA rank {cfg['lora_rank']} on "
             f"`{', '.join(cfg.get('target_modules', ['q_proj', 'v_proj']))}`, "
             f"{common.K_SHARDS} shards $\\times$ {common.N_SLICES} slices "
             f"$\\times$ {common.RECORDS_PER_SLICE} records. Each row is "
             f"{rows[0]['n']} invocations per slice position, over all shards. "
             f"A deletion in slice 0 resumes from the base model, so that row "
             f"is the cost of rebuilding the shard outright.", "",
             "| deletion slice | slices retrained | median (s) | min | max |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['slice']} | {r['retrained']} | {r['median']:.2f} | "
                     f"{r['min']:.2f} | {r['max']:.2f} |")
    fixed = (f"a fixed component of {intercept:.2f}\\,s that the invocation "
             f"pays whatever it retrains" if abs(intercept) >= 0.05
             else "no fixed component this measurement can separate from zero")
    lines += ["",
              f"Retraining falls by {slope:.2f}\\,s per slice the deletion "
              f"spares, within {resid:.2f}\\,s of a straight line across all "
              f"{len(rows)} positions, and the line has {fixed} "
              f"(intercept {intercept:.2f}\\,s). The relationship is SISA's "
              f"and it is the payload-independent one the 0.9\\,M model also "
              f"shows; the two sets of absolute times are not comparable, "
              f"because a slice here holds "
              f"{common.RECORDS_PER_SLICE} records against 32 there.", ""]
    (out / "retrain_1b.md").write_text("\n".join(lines))
    (out / "retrain_1b.dat").write_text(
        "slice retrained median min max\n" + "\n".join(
            f"{r['slice']} {r['retrained']} {r['median']} {r['min']} {r['max']}"
            for r in rows) + "\n")
    (out / "retrain_1b.json").write_text(json.dumps(
        {"config": cfg, "rows": rows, "slope_s_per_slice": float(slope),
         "intercept_s": float(intercept), "max_residual_s": resid}, indent=1))
    print(f"\n{slope:.2f} s per slice spared (residual {resid:.2f} s) "
          f"-> {out / 'retrain_1b.md'}")


if __name__ == "__main__":
    main()
