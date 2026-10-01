"""E13 - Compound erasure across invocations (non-TCB harness).

Theorem 2 speaks about one invocation. The property regulation asks for is
stronger: a record deleted today must still be absent tomorrow. It is not
automatic, because the checkpoints a shard accumulated before a deletion still
carry the deleted record, and a later deletion that resumes from one of them
puts it back.

This module exercises the mechanism that closes the gap. A completed deletion
is appended to the shard's record, the enclave retains the record's head, and
every invocation re-runs the clean-prefix test against the union of its own
request and everything the record already holds. Three things are measured:
that sequential deletions still verify, that the ladder the unlearning issues
keeps the second deletion cheap, and that the two ways of restoring a deleted
record are refused.

Every run here writes state -- the record, the heads, new checkpoints -- so it
works on a private copy of the dataset and leaves the shared one alone.
"""
import contextlib
import json
import pathlib
import shutil
import tempfile
import uuid

import numpy as np

from clean_runtime import commit, dlog, gateway, runtime, store
from clean_runtime.gateway import DatasetHandle, Invocation
from clean_runtime.rng import TeeRng

from . import common, run_unlearn

SHARD = 0


@contextlib.contextmanager
def isolated_data():
    """Run against a copy of a deployment, so persisted deletions stay local.

    Two things have to be copied, not one. The data is host storage and a
    temporary directory is the right place for it. The sealed state is not: it
    is the signing seed and the record heads, and on a platform that seals them
    the copy has to stay inside the sealing boundary, so it goes in a
    subdirectory of wherever the sealed state already lives.
    """
    original, original_sealed = common.DATA_DIR, common.sealed_dir()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="clean-compound-"))
    sealed = original_sealed / f"isolated-{uuid.uuid4().hex[:8]}"
    try:
        # A flat, contents-only copy. `shutil.copytree` copies metadata the
        # platform need not implement -- Gramine returns ENOSYS for it -- and it
        # would also recurse into the temporary directory that lives under the
        # deployment. A deployment directory is a flat set of files, so copying
        # exactly those is both sufficient and portable.
        data = tmp / "data"
        data.mkdir(parents=True)
        for f in sorted(original.iterdir()):
            if f.is_file():
                shutil.copyfile(f, data / f.name)
        sealed.mkdir(parents=True, exist_ok=True)
        for name in ("enclave.sealed", run_unlearn.HEADS, run_unlearn.VERSIONS):
            src = original_sealed / name
            if src.exists():
                shutil.copyfile(src, sealed / name)
        common.DATA_DIR = data
        common.SEALED_OVERRIDE = sealed
        yield common.DATA_DIR
    finally:
        common.DATA_DIR, common.SEALED_OVERRIDE = original, None
        shutil.rmtree(tmp, ignore_errors=True)
        # Best effort: on an encrypted mount, removal does not behave the way it
        # does on a filesystem the process owns -- Gramine leaves the host file
        # in place and the directory entry can go before the directory does. The
        # host clears what is left between runs, which is the only place that can
        # be done anyway.
        shutil.rmtree(sealed, ignore_errors=True)


def _metas(shard_id):
    out = []
    for p in sorted(common.DATA_DIR.glob(f"cp_{shard_id}_v*.npz.meta.json")):
        out.append(json.loads(p.read_text()))
    return out


def _slice_bounds(shard_id):
    return json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.npz.meta.json").read_text()
    )["slice_bounds"]


def _manifest(shard_id):
    return json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.manifest.json").read_text())


def select_checkpoint(shard_id, delta, record):
    """The latest certified checkpoint an honest provider may resume from.

    Its trained prefix must be free of the request in hand and of every
    deletion already in the record, which is exactly the test the gateway
    will redo.
    """
    manifest = _manifest(shard_id)
    bounds = _slice_bounds(shard_id)
    deleted_now = {int(ix) for ix, _ in delta}
    best = None
    for meta in _metas(shard_id):
        later = dlog.entries_after(shard_id, record,
                                   bytes.fromhex(meta["h_record"]))
        if later is None:
            continue
        forgotten = dlog.deleted_hashes(later)
        prefix_end = 0 if meta["slice_idx"] < 0 \
            else bounds[meta["slice_idx"]][1]
        clean = True
        for row, (ix, h) in enumerate(manifest):
            if row >= prefix_end:
                break
            if int(ix) in deleted_now or bytes.fromhex(h) in forgotten:
                clean = False
                break
        if clean and (best is None or meta["version"] > best["version"]):
            best = meta
    return best


def _delete(shard_id, delta, label, slices_expected=None):
    """One honest deletion, persisted, then verified."""
    record = run_unlearn.load_record(shard_id)
    cp = select_checkpoint(shard_id, delta, record)
    assert cp is not None, "no certified checkpoint is clean for this request"
    _, arc, _, elapsed = run_unlearn.run(
        shard_id=shard_id, clean_version=cp["version"], delta=delta,
        nonce_label=label, out_name=None, persist=True)
    accept, reason, _ = run_unlearn.verify_run(
        arc, shard_id=shard_id, nonce_label=label, delta=delta,
        clean_version=cp["version"], record=record)
    retrained = len(_slice_bounds(shard_id)) - 1 - cp["slice_idx"]
    if slices_expected is not None:
        assert retrained == slices_expected, (
            f"{label}: resumed from slice {cp['slice_idx']}, retraining "
            f"{retrained} slices, expected {slices_expected}")
    return {"version": cp["version"], "slice_idx": cp["slice_idx"],
            "slices_retrained": retrained, "accept": accept,
            "reason": reason, "seconds": elapsed,
            "h_record": arc["h_record"].hex()}


def resurrection_arc(shard_id, delta, poisoned_version, record):
    """Bypass the gateway and resume from a checkpoint the record forbids.

    The gateway refuses this, so there is no certificate to present. To ask
    what the auditor alone could do, the red team assembles the invocation by
    hand, exactly as it does for the other post-admission attacks.
    """
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r("resurrection")
    with np.load(common.DATA_DIR / f"shard_{shard_id}.npz") as npz:
        index, salt, tokens = npz["index"], npz["salt"], npz["tokens"]
    meta = json.loads(
        (common.DATA_DIR
         / f"cp_{shard_id}_v{poisoned_version}.npz.meta.json").read_text())
    bounds = _slice_bounds(shard_id)
    lo = bounds[meta["slice_idx"] + 1][0]
    deleted = {int(ix) for ix, _ in delta}
    rows = np.array([r for r in range(lo, bounds[-1][1])
                     if int(index[r]) not in deleted])
    data = DatasetHandle(index[rows], salt[rows], tokens[rows],
                         commit.dataset_root(index[rows], salt[rows],
                                             tokens[rows]))
    dh = {int(i): bytes.fromhex(h) for i, h in delta}
    delta_idx = np.array(sorted(dh), dtype=np.uint32)
    h_delta = commit.hashes_root(delta_idx, [dh[int(i)] for i in delta_idx])
    base = gateway.load_base(common.DATA_DIR / "base.npz")
    cp = gateway.load_checkpoint(
        common.DATA_DIR / f"cp_{shard_id}_v{poisoned_version}.npz")
    inv = Invocation(base, cp, data, TeeRng(enclave.derive_r(nonce)),
                     json.loads((common.DATA_DIR / "config.json").read_text()),
                     h_delta, delta_idx, nonce)
    inv.delta_hashes = tuple(dh[int(i)] for i in delta_idx)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc, meta


def sealed_snapshot():
    """What the host can copy out and put back: the sealed files, and the
    record body it was already holding."""
    sealed = common.sealed_dir()
    files = {}
    for name in (run_unlearn.HEADS, run_unlearn.VERSIONS):
        p = sealed / name
        files[name] = p.read_bytes() if p.exists() else None
    return files, run_unlearn.load_record(SHARD)


def sealed_restore(snapshot):
    """Put the enclave back in a world it has already left.

    Sealing binds the state to the enclave identity and not to the moment it
    was written, so the restored files decrypt under the same measurement and
    the enclave resumes from them without noticing.
    """
    files, record = snapshot
    sealed = common.sealed_dir()
    for name, blob in files.items():
        p = sealed / name
        if blob is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(blob)
    run_unlearn.save_record(SHARD, record)


def rollback_case(snapshot, known_head, erased):
    """Restore the pre-deletion sealed state and certify in the old world.

    Returns what three auditors make of the resulting certificate: one
    auditing this shard for the first time, one carrying the head it saw
    when the restored deletion was certified, and the second with the
    freshness check ablated away. `erased` is what the shard had been
    asked to forget between the snapshot and now, which is what the restored
    world puts back.
    """
    sealed_restore(snapshot)
    delta = run_unlearn.pick_delta(SHARD, 3, 1)
    record = run_unlearn.load_record(SHARD)
    cp = select_checkpoint(SHARD, delta, record)
    assert cp is not None, "the restored world offers no checkpoint at all"
    _, arc, _, _ = run_unlearn.run(
        shard_id=SHARD, clean_version=cp["version"], delta=delta,
        nonce_label="sealed-rollback", out_name=None, persist=False)

    # A certificate is worth reporting only if it does restore something:
    # the state it resumes from has to be carrying one of the records the
    # shard had already been asked to forget.
    manifest = _manifest(SHARD)
    bounds = _slice_bounds(SHARD)
    meta = json.loads((common.DATA_DIR
                       / f"cp_{SHARD}_v{cp['version']}.npz.meta.json"
                       ).read_text())
    prefix_end = bounds[meta["slice_idx"]][1]
    restored = [int(ix) for row, (ix, h) in enumerate(manifest)
                if row < prefix_end and bytes.fromhex(h) in erased]
    assert restored, "the restored checkpoint carries no deleted record"

    kw = dict(shard_id=SHARD, nonce_label="sealed-rollback", delta=delta,
              clean_version=cp["version"], record=record)
    first_time = run_unlearn.verify_run(arc, **kw)[0]
    stateful = run_unlearn.verify_run(arc, known_head=known_head, **kw)[0]
    ablated = run_unlearn.verify_run(
        arc, known_head=known_head,
        unsafe_skip=frozenset({"freshness"}), **kw)[0]
    return {"version": cp["version"], "slice_idx": meta["slice_idx"],
            "restored": restored,
            "first_time_accepts": first_time == 1,
            "stateful_rejects": stateful == 0,
            "ablation_accepts": ablated == 1}


def run():
    results = {}
    with isolated_data():
        # What a host can copy out before it lets the first deletion run.
        snapshot = sealed_snapshot()

        d1 = run_unlearn.pick_delta(SHARD, 2, 1)
        results["first"] = _delete(SHARD, d1, "compound-1", slices_expected=2)

        record_after_1 = run_unlearn.load_record(SHARD)
        results["record_entries"] = len(record_after_1)

        # The ladder the first deletion issued lets the second resume one
        # slice back instead of falling all the way to a pre-deletion state.
        d2 = run_unlearn.pick_delta(SHARD, 3, 1)
        results["second"] = _delete(SHARD, d2, "compound-2",
                                    slices_expected=1)

        # Attack 1: resume from a checkpoint whose prefix contains the record
        # deleted first. The gateway refuses it.
        poisoned = max(m["version"] for m in _metas(SHARD)
                       if m["slice_idx"] == 2 and m["version"] <= 4)
        record = run_unlearn.load_record(SHARD)
        try:
            run_unlearn.run(shard_id=SHARD, clean_version=poisoned, delta=d2,
                            nonce_label="resurrect", out_name=None)
            results["resurrection_refused"] = False
        except ValueError as e:
            results["resurrection_refused"] = "earlier invocation" in str(e)

        # Attack 2: serve a record with the first deletion removed, so the
        # poisoned checkpoint looks clean. The retained head refuses it.
        try:
            # the record as it stood before any deletion
            run_unlearn.save_record(SHARD, [])
            run_unlearn.run(shard_id=SHARD, clean_version=poisoned, delta=d2,
                            nonce_label="rollback", out_name=None)
            results["record_rollback_refused"] = False
        except ValueError as e:
            results["record_rollback_refused"] = (
                "head retained by the enclave" in str(e))
        finally:
            run_unlearn.save_record(SHARD, record)

        # What the auditor alone can do: an ARC from a run that bypassed the
        # gateway is rejected by the full verifier and accepted once the
        # compound-erasure check is ablated away.
        arc, meta = resurrection_arc(SHARD, d2, poisoned, record)
        dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
        from verifier import verify
        kw = dict(nonce=common.harness_r("resurrection"), delta=d2,
                  manifest=_manifest(SHARD), slice_bounds=_slice_bounds(SHARD),
                  cp_meta=meta, cp_cert=bytes.fromhex(meta["cert"]),
                  c_base=bytes.fromhex(dep["c_base"]),
                  h_config=bytes.fromhex(dep["h_config"]),
                  mrenclave=bytes.fromhex(dep["mrenclave"]), record=record)
        pk = bytes.fromhex(dep["pk"])
        results["verifier_rejects"] = verify.verify_arc(pk, arc, **kw)[0] == 0
        results["ablation_accepts"] = verify.verify_arc(
            pk, arc, **dict(kw, unsafe_skip=frozenset({"record"})))[0] == 1

        # Attack 3: restore the sealed state from before the first deletion
        # and certify in the world that comes back. It leaves the copy in a
        # state of the host's choosing, so it runs last.
        results["rollback"] = rollback_case(
            snapshot, bytes.fromhex(results["first"]["h_record"]),
            dlog.deleted_hashes(run_unlearn.load_record(SHARD)))
    return results


def main():
    r = run()
    for k, v in r.items():
        print(f"  {k}: {v}")
    ok = (r["first"]["accept"] == 1 and r["second"]["accept"] == 1
          and r["resurrection_refused"] and r["record_rollback_refused"]
          and r["verifier_rejects"] and r["ablation_accepts"]
          and r["rollback"]["stateful_rejects"]
          and r["rollback"]["ablation_accepts"])

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    lines = ["# E13 - Compound erasure across invocations", "",
             "Two deletions on the same shard, in sequence, each persisted to "
             "the shard's deletion record.", "",
             "| step | resumed from | slices retrained | verifies |",
             "|---|---|---:|---|"]
    for key, name in (("first", "delete from slice 2"),
                      ("second", "delete from slice 3")):
        s = r[key]
        lines.append(f"| {name} | v{s['version']} (slice {s['slice_idx']}) | "
                     f"{s['slices_retrained']} | "
                     f"{'yes' if s['accept'] == 1 else 'NO'} |")
    lines += ["",
              f"The second deletion resumes from a checkpoint the first one "
              f"issued, so it retrains {r['second']['slices_retrained']} "
              f"slice rather than falling back to a pre-deletion state.", "",
              "Restoring the first deletion, four ways:", "",
              f"- resuming from a checkpoint whose prefix holds it: refused by "
              f"the gateway: **{r['resurrection_refused']}**",
              f"- serving a record with that entry removed: refused against "
              f"the retained head: **{r['record_rollback_refused']}**",
              f"- an ARC from a run that bypassed the gateway: rejected by the "
              f"full verifier: **{r['verifier_rejects']}**; accepted with the "
              f"compound-erasure check ablated: **{r['ablation_accepts']}**",
              f"- restoring the sealed state from before the first deletion "
              f"and certifying in the world that comes back: the run resumes "
              f"from v{r['rollback']['version']} (slice "
              f"{r['rollback']['slice_idx']}), whose trained prefix carries "
              f"record {r['rollback']['restored'][0]}. An auditor holding no "
              f"earlier head accepts it: "
              f"**{r['rollback']['first_time_accepts']}**. An auditor "
              f"carrying the head it saw when that deletion was certified "
              f"rejects it: **{r['rollback']['stateful_rejects']}**; with "
              f"the freshness check ablated it accepts again: "
              f"**{r['rollback']['ablation_accepts']}**",
              ""]
    (out / "compound.md").write_text("\n".join(lines))
    (out / "compound.meta.json").write_text(json.dumps(r, indent=1))
    print(f"-> {out / 'compound.md'} (all expectations met: {ok})")
    assert ok, "compound erasure did not behave as specified"


if __name__ == "__main__":
    main()


def resurrection_case():
    """Build the resurrection attack and everything an auditor needs to judge
    it, on a shard that has already had one deletion.

    Returned as (arc, verify-overrides) so the attack matrix can evaluate it
    against every configuration exactly like the others.
    """
    with isolated_data():
        d1 = run_unlearn.pick_delta(SHARD, 2, 1)
        record0 = run_unlearn.load_record(SHARD)
        cp1 = select_checkpoint(SHARD, d1, record0)
        run_unlearn.run(shard_id=SHARD, clean_version=cp1["version"],
                        delta=d1, nonce_label="compound-1", out_name=None,
                        persist=True)
        record = run_unlearn.load_record(SHARD)
        d2 = run_unlearn.pick_delta(SHARD, 3, 1)
        poisoned = max(m["version"] for m in _metas(SHARD)
                       if m["slice_idx"] == 2 and m["version"] <= 4)
        arc, meta = resurrection_arc(SHARD, d2, poisoned, record)
        over = {"delta": d2, "cp_meta": meta,
                "cp_cert": bytes.fromhex(meta["cert"]), "record": record,
                "nonce": common.harness_r("resurrection")}
    return arc, over
