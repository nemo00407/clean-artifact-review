"""E11/E12 - What the boundary costs (non-TCB harness).

Two sweeps that the earlier point measurements could not express.

E11, verification cost. `Verify` recomputes a Merkle root over the published
manifest minus the deletion request and checks two signatures. It never
reads the model, so its cost is a function of the dataset it audits and not
of the parameters it certifies. We sweep the manifest length and the size of
the deletion request and measure the real verifier end to end, over
certificates the enclave actually signed.

E12, temporal-closure overhead. Closure in time is bought by splitting the
admitted dataset into blocks, retaining one commitment per block inside the
enclave, and re-deriving it on every fetch. The block size sets the trade:
smaller blocks mean more commitments resident and more re-derivations,
larger blocks mean fewer of both but more bytes hashed per fetch. We sweep it
and measure the retraining time with the check on and off.
"""
import json
import time

import numpy as np

from clean_runtime import commit, dlog, gateway, runtime, store
from verifier import verify

from . import common, run_unlearn


def _time(fn, reps):
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return float(np.median(xs)), float(np.percentile(xs, 99))


# ----------------------------------------------------- E11 verification cost

def _synthetic_case(enclave, dep, n_records, n_deleted, n_slices=4):
    """A well-formed certificate over a manifest of `n_records` rows.

    The enclave signs it, so the verifier runs its whole path -- deletion
    binding, checkpoint certificate, clean-prefix test, the H(D_clean)
    recomputation and both signature checks -- exactly as on a real run.
    """
    idx = list(range(n_records))
    hashes = [commit.sha256(b"rec%d" % i) for i in idx]
    manifest = [[i, h.hex()] for i, h in zip(idx, hashes)]
    per = n_records // n_slices
    bounds = [[j * per, (j + 1) * per] for j in range(n_slices)]
    bounds[-1][1] = n_records

    slice_idx = 1                      # trained prefix = slices 0..1
    prefix_end = bounds[slice_idx][1]
    deleted = idx[prefix_end:prefix_end + n_deleted]
    delta = [(i, hashes[i].hex()) for i in deleted]

    h_prefix = commit.hashes_root(np.array(idx[:prefix_end]),
                                  hashes[:prefix_end])
    h_cp = commit.sha256(b"ckpt")
    head = dlog.genesis(0)                 # a shard with no deletions yet
    cert = enclave.sign_cp_cert(h_cp, h_prefix, 0, slice_idx, 2, head)
    cp_meta = {"shard_id": 0, "slice_idx": slice_idx, "version": 2,
               "h_d_prefix": h_prefix.hex(), "h_record": head.hex(),
               "cert": cert.hex()}

    dset = set(deleted)
    retained = [(i, hashes[i]) for i in idx[prefix_end:] if i not in dset]
    h_dclean = verify._root_of(retained)
    h_delta = verify._root_of([(i, hashes[i]) for i in deleted])

    arc = {"c_base": bytes.fromhex(dep["c_base"]), "h_l_ckpt": h_cp,
           "h_l_new": commit.sha256(b"new"), "h_dclean": h_dclean,
           "h_delta": h_delta,
           "h_record": dlog.extend(head, [hashes[i] for i in deleted]),
           "h_r": commit.sha256(b"r"),
           "h_config": bytes.fromhex(dep["h_config"]),
           "shard_id": 0, "cp_version": 2,
           "nonce": common.harness_r("cost-sweep"),
           "mrenclave": enclave.mrenclave}
    arc["signature"] = enclave.sign_arc(arc)

    kw = dict(nonce=arc["nonce"], delta=delta, manifest=manifest,
              slice_bounds=bounds, cp_meta=cp_meta, cp_cert=cert,
              c_base=arc["c_base"], h_config=arc["h_config"],
              mrenclave=enclave.mrenclave)
    return arc, kw


def verification_cost(sizes=(128, 512, 2048, 8192, 32768),
                      deletions=(1, 8, 64, 512), reps=200):
    enclave = run_unlearn.load_enclave()
    dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
    pk = bytes.fromhex(dep["pk"])

    by_manifest = []
    for n in sizes:
        arc, kw = _synthetic_case(enclave, dep, n, 1)
        accept, reason = verify.verify_arc(pk, arc, **kw)
        assert accept == 1, f"synthetic case n={n} did not verify: {reason}"
        med, p99 = _time(lambda: verify.verify_arc(pk, arc, **kw), reps)
        by_manifest.append((n, med, p99))
        print(f"manifest {n:6d} records -> {med:.3f} ms (p99 {p99:.3f})")

    by_delta = []
    n = max(sizes)
    for d in deletions:
        arc, kw = _synthetic_case(enclave, dep, n, d)
        accept, reason = verify.verify_arc(pk, arc, **kw)
        assert accept == 1, f"synthetic case d={d} did not verify: {reason}"
        med, p99 = _time(lambda: verify.verify_arc(pk, arc, **kw), reps)
        by_delta.append((d, med, p99))
        print(f"deletion set {d:5d} records -> {med:.3f} ms (p99 {p99:.3f})")
    return by_manifest, by_delta


# ------------------------------------------------ E12 temporal-closure cost

def _retrain_reference(delta, reps=3):
    """One real invocation: how long it takes and how many fetches it makes."""
    from . import fetch_probe
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r("blocksize")
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    best, n_rows, fetches = None, None, None
    for _ in range(reps):
        backend = fetch_probe._CountingBackend()
        t0 = time.perf_counter()
        inv = gateway.open_invocation(
            common.DATA_DIR / "base.npz",
            common.DATA_DIR / "cp_0_v2.npz",
            common.DATA_DIR / "shard_0.npz",
            delta, enclave.derive_r(nonce), config, nonce=nonce,
            backend=backend)
        n_rows = len(inv.data.tokens)
        runtime.unlearn(inv, enclave)
        el = (time.perf_counter() - t0) * 1000
        best = el if best is None else min(best, el)
        fetches = backend.fetches
    return best, n_rows, fetches


def temporal_overhead(block_sizes=(4, 8, 16, 32), reps=2000,
                      wide=(4, 16, 64, 256, 1024, 4096)):
    """The cost of closure in time, decomposed.

    Trying to read this off end-to-end wall clock does not work: the check
    is microseconds against seconds of retraining, so the difference sits
    inside run-to-run variance. We therefore measure the re-derivation
    directly and compose it with the fetch count of a real invocation.
    """
    delta = run_unlearn.pick_delta(0, 2, 1)
    retrain_ms, n_rows, fetches = _retrain_reference(delta)
    print(f"reference invocation: {n_rows} rows, {fetches} fetches, "
          f"{retrain_ms:.1f} ms")

    rows = []
    for bs in block_sizes:
        blob = store.encode_block(np.zeros((bs, 64), dtype=np.uint8))
        med, _ = _time(lambda: store.block_commitment(blob), reps)
        per_fetch_us = med * 1000.0
        resident = -(-n_rows // bs)            # commitments held in enclave
        # A run reads every block once per batch; scale the observed fetch
        # count by how the block count changes with block size.
        base_blocks = -(-n_rows // store.BLOCK_ROWS)
        run_fetches = fetches * resident / base_blocks
        total_ms = run_fetches * med
        assert resident >= 2, (
            f"block size {bs} collapses the {n_rows}-row dataset to one "
            f"block; the composed per-run column would be meaningless")
        rows.append((bs, len(blob), resident, per_fetch_us, run_fetches,
                     total_ms, 100.0 * total_ms / retrain_ms))
        print(f"block {bs:4d} rows ({len(blob):6d} B, {resident:3d} resident): "
              f"{per_fetch_us:7.2f} us/fetch, {total_ms:7.3f} ms/run, "
              f"{100.0 * total_ms / retrain_ms:6.4f}% of retraining")

    # Per-fetch cost alone, over a range the prototype dataset cannot reach.
    # This is pure hashing and does not depend on the dataset at all.
    curve = []
    for bs in wide:
        blob = store.encode_block(np.zeros((bs, 64), dtype=np.uint8))
        med, _ = _time(lambda: store.block_commitment(blob), reps)
        curve.append((bs, len(blob), med * 1000.0))
        print(f"  per-fetch at {bs:5d} rows ({len(blob):7d} B): "
              f"{med * 1000.0:8.2f} us")
    return retrain_ms, n_rows, fetches, rows, curve


def main():
    by_manifest, by_delta = verification_cost()
    blocks_result = temporal_overhead()

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)

    lines = ["# E11 - Verification cost", "",
             "Certificates signed by the enclave over synthetic manifests. "
             "The verifier runs its whole path in every measurement. It never "
             "reads the model, so no series here varies with parameter "
             "count.", "",
             "| manifest (records) | median (ms) | p99 (ms) |", "|---:|---:|---:|"]
    for n, med, p99 in by_manifest:
        lines.append(f"| {n} | {med:.3f} | {p99:.3f} |")
    lines += ["", "| deletion set (records) | median (ms) | p99 (ms) |",
              "|---:|---:|---:|"]
    for d, med, p99 in by_delta:
        lines.append(f"| {d} | {med:.3f} | {p99:.3f} |")
    lines += ["",
              f"Manifest length spans {by_manifest[0][0]} to "
              f"{by_manifest[-1][0]} records, a factor of "
              f"{by_manifest[-1][0] // by_manifest[0][0]}, for a cost ratio of "
              f"{by_manifest[-1][1] / by_manifest[0][1]:.1f}.", ""]
    (out / "verification_cost.md").write_text("\n".join(lines))
    (out / "verification_cost.dat").write_text(
        "n median p99\n"
        + "\n".join(f"{n} {m:.4f} {p:.4f}" for n, m, p in by_manifest) + "\n")
    (out / "verification_cost_delta.dat").write_text(
        "d median p99\n"
        + "\n".join(f"{d} {m:.4f} {p:.4f}" for d, m, p in by_delta) + "\n")

    retrain_ms, n_rows, fetches, blocks, curve = blocks_result
    lines = ["# E12 - Temporal-closure overhead", "",
             f"The admitted dataset is split into blocks; one commitment per "
             f"block stays in the enclave and is re-derived on every fetch. "
             f"The reference invocation admits **{n_rows}** rows and performs "
             f"**{fetches}** fetches in **{retrain_ms:.0f} ms** of "
             f"retraining. Per-fetch cost is measured directly, because the "
             f"check is microseconds against seconds and does not surface "
             f"above run-to-run variance end to end.", "",
             "| block (rows) | bytes | commitments resident | per fetch (us) "
             "| per run (ms) | share of retraining |",
             "|---:|---:|---:|---:|---:|---:|"]
    for bs, nb, res, us, rf, tot, pct in blocks:
        lines.append(f"| {bs} | {nb} | {res} | {us:.2f} | {tot:.3f} | "
                     f"{pct:.4f}% |")
    lines += ["",
              "Smaller blocks keep more commitments resident and re-derive "
              "more often; larger blocks hash more bytes per fetch. The "
              "guarantee is identical at every point on the curve, so the "
              "block size is a memory/throughput choice and not a security "
              "one.", ""]
    (out / "temporal_overhead.md").write_text("\n".join(lines))
    (out / "temporal_overhead.dat").write_text(
        "block bytes resident per_fetch_us per_run_ms pct\n"
        + "\n".join(f"{b} {nb} {r} {us:.4f} {t:.5f} {p:.6f}"
                    for b, nb, r, us, _rf, t, p in blocks) + "\n")
    (out / "block_commitment_cost.dat").write_text(
        "block bytes us\n"
        + "\n".join(f"{b} {nb} {us:.4f}" for b, nb, us in curve) + "\n")
    print(f"-> {out / 'verification_cost.md'}, {out / 'temporal_overhead.md'}")


if __name__ == "__main__":
    main()
