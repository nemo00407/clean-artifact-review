"""RQ1 at 1B parameters: is the adapter still bitwise identical?

The 0.9M sweep answers this on a two-layer NumPy model, which is smaller than
the benchmarks of the cryptographic work the paper criticises, making the
number easy to dismiss. This runs the same experiment against TinyLlama-1.1B
on the GPU.

Only two things are swapped: the data directory, and the payload the runtime
calls. `runtime.kernel` is a module-level binding, so pointing it at
`harness/gpu_kernel.py` is exactly the substitution a deployment performs at
build time -- and, as Track 1 measured, one that changes MRENCLAVE. The
gateway, the verified read path, the commitments, the certificates and the
verifier are the runtime's own code and are not touched.

World A holds the records and the gateway withholds them; World B is a shard
rebuilt as though they had never been written, with the slice boundaries
shifted so that each slice keeps the same records. Same checkpoint, same
attested randomness, same configuration. The criterion is SHA-256 over the
canonical adapter serialization.
"""
import json
import time

import numpy as np

from clean_runtime import commit, gateway, runtime

from . import common, gpu_kernel, run_unlearn, sisa_train_1b
from .two_world import build_counterfactual

# Importing sisa_train_1b points `common` at the 1B deployment and its layout.

runtime.kernel = gpu_kernel          # the payload substitution under test

SWEEP_SHARDS = [0, 1]
SWEEP_SLICES = [1, 2, 3]
SWEEP_COUNTS = [1, 2]


def _configs():
    for shard in SWEEP_SHARDS:
        for sl in SWEEP_SLICES:
            for n in SWEEP_COUNTS:
                yield (f"s{shard} slice{sl} n{n}", shard, sl, sl, n)


def run_world(shard_path, delta, cp_version, shard_id, nonce_label):
    cfg = json.loads((common.DATA_DIR / "config.json").read_text())
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(nonce_label)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{shard_id}_v{cp_version}.npz",
        shard_path, delta, enclave.derive_r(nonce), cfg, nonce=nonce)
    new_lora, arc, _ = runtime.unlearn(inv, enclave)
    return commit.params_hash(new_lora), arc


def sweep():
    rows = []
    for label, shard, cp_v, sl, n in _configs():
        t0 = time.time()
        delta = run_unlearn.pick_delta(shard, sl, n)
        # Two identical worlds would agree for a reason that has nothing to do
        # with non-interference, so the request has to be real before the
        # comparison means anything.
        assert len(delta) == n, (
            f"{label}: asked to delete {n} records, selected {len(delta)}")
        cf = build_counterfactual(shard, delta,
                                  common.DATA_DIR / f"shard_cf_{shard}.npz")
        # Both worlds are given the same nonce, so r is the same in both:
        # Definition 2 holds r fixed and varies only the deleted content.
        h_a, arc = run_world(common.DATA_DIR / f"shard_{shard}.npz", delta,
                             cp_v, shard, f"tw1b-{label}")
        h_b, _ = run_world(cf, [], cp_v, shard, f"tw1b-{label}")
        accept, reason, _ = run_unlearn.verify_run(
            arc, shard_id=shard, nonce_label=f"tw1b-{label}", delta=delta,
            clean_version=cp_v)
        rows.append({"label": label, "shard": shard, "slice": sl, "n": n,
                     # Recorded so the artifact itself shows the request was
                     # not empty: an empty one makes the two worlds the same
                     # invocation and the comparison vacuous.
                     "deleted": [int(ix) for ix, _ in delta],
                     "equal": h_a == h_b, "accept": int(accept),
                     "reason": reason, "h_a": h_a.hex(), "h_b": h_b.hex(),
                     "seconds": time.time() - t0})
        print(f"{label:18s} equal={h_a == h_b}  verifies={accept}  "
              f"{time.time() - t0:5.1f}s", flush=True)
    return rows


def main():
    rows = sweep()
    eq = sum(r["equal"] for r in rows)
    ok = sum(r["accept"] for r in rows)
    cfg = json.loads((common.DATA_DIR / "config.json").read_text())

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    lines = ["# E16 - Two-world equality at 1B parameters", "",
             f"Payload: `harness/gpu_kernel.py`, LoRA rank "
             f"{cfg['lora_rank']} over `{cfg['model_id']}` on the GPU. The "
             f"gateway, read path, commitments, certificates and verifier are "
             f"the runtime's own code, unchanged.", "",
             "| configuration | records deleted | $L^A = L^B$ | verifies "
             "| seconds |",
             "|---|---|---|---|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {', '.join(map(str, r['deleted']))} | "
                     f"{'yes' if r['equal'] else '**NO**'} | "
                     f"{'yes' if r['accept'] else '**NO**'} | {r['seconds']:.1f} |")
    lines += ["", f"{eq}/{len(rows)} configurations produced byte-identical "
                  f"adapters across the two worlds; {ok}/{len(rows)} issued a "
                  f"certificate the independent verifier accepted.", ""]
    (out / "two_world_1b.md").write_text("\n".join(lines))
    (out / "two_world_1b.json").write_text(json.dumps(
        {"config": cfg, "rows": rows, "equal": eq, "accepted": ok}, indent=1))
    print(f"\n{eq}/{len(rows)} equal, {ok}/{len(rows)} verified "
          f"-> {out / 'two_world_1b.md'}")
    assert eq == len(rows) and ok == len(rows), "1B two-world sweep failed"


if __name__ == "__main__":
    main()
