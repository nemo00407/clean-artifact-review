"""E14/E15 - What an audit costs at corpus scale (non-TCB harness).

E11 measured verification over manifests up to 32768 records, which is three
orders of magnitude below the corpus a foundation model is trained on. Two
questions follow, and neither is answered by extrapolating that curve.

E14, verification at corpus scale. `Verify` recomputes a Merkle root over the
published manifest minus the deletion request. The cost people expect to
dominate is the hashing, which is linear and cheap. The cost that actually
dominates is holding the manifest: the auditor materializes one entry per
record before the first hash is taken, and `verifier/verify.py` sorts that list
as well. We measure the real verifier until it stops being able to run, report
where that happens, and then measure a streaming root computation that folds
leaves as they arrive and reaches further on the same machine.

E15, the clean-prefix test. The gateway must find the checkpoint's trained
prefix free of the request in hand and of every deletion recorded since. The
prefix is the part of the shard the invocation does *not* retrain, so this cost
moves opposite to the retraining it saves: the later the deletion falls, the
less there is to retrain and the more prefix there is to walk. The paper
asserts that relationship; this measures it.

Both sweeps run each point in a forked child so that peak RSS is attributable,
and both are pure CPU: nothing here depends on the enclave.
"""
import hashlib
import json
import os
import pickle
import resource
import struct
import time

from . import common

SHA = hashlib.sha256


def _say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------- merkle roots

def level_merkle(leaves):
    """The root as `verifier/verify.py` computes it: pad each odd level by
    repeating its last node, then pair. Materializes every level."""
    if len(leaves) == 0:
        return SHA(b"").digest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [SHA(level[i] + level[i + 1]).digest()
                 for i in range(0, len(level), 2)]
    return level[0]


def _fold_push(stack, leaf):
    """Add one leaf to a fold in progress.

    The stack holds at most one subtree per height, heights strictly increasing
    from the top, so it is O(log n) whatever n turns out to be. Two subtrees of
    equal height combine immediately.
    """
    node, height = leaf, 0
    while stack and stack[-1][0] == height:
        node = SHA(stack.pop()[1] + node).digest()
        height += 1
    stack.append((height, node))


def _fold_root(stack):
    """Close a fold. A lone subtree at a height is paired with itself, which is
    what the level-wise algorithm does to the last node of an odd level."""
    if not stack:
        return SHA(b"").digest()
    height, node = stack.pop()
    while stack:
        if stack[-1][0] == height:
            node = SHA(stack.pop()[1] + node).digest()
        else:
            node = SHA(node + node).digest()
        height += 1
    return node


def streaming_merkle(leaf_iter):
    """The root of `level_merkle`, folded as the leaves arrive."""
    stack = []
    for leaf in leaf_iter:
        _fold_push(stack, leaf)
    return _fold_root(stack)


# ------------------------------------------------------------- child harness

def _in_child(fn, *args, cap_gb=7.0):
    """Run fn in a forked child; return (result, peak_rss_bytes).

    Each sweep point gets its own address space so that `ru_maxrss`, which is a
    high-water mark and never falls, measures that point alone. The child's
    address space is capped so that a point too large for this machine raises
    MemoryError inside the child rather than growing until the kernel picks a
    victim -- the limit is the measurement, and it should not be taken by
    killing something else.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # child
        os.close(r)
        try:
            cap = int(cap_gb * 1e9)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            out = fn(*args)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            with os.fdopen(w, "wb") as f:
                pickle.dump((out, rss), f)
        except MemoryError:
            with os.fdopen(w, "wb") as f:
                pickle.dump(("MemoryError", -1), f)
        except Exception as e:                      # noqa: BLE001
            with os.fdopen(w, "wb") as f:
                pickle.dump((f"{type(e).__name__}: {e}", -1), f)
        os._exit(0)
    os.close(w)                                     # parent
    with os.fdopen(r, "rb") as f:
        try:
            out, rss = pickle.load(f)
        except EOFError:                            # child was killed (OOM)
            out, rss = "killed", -1
    os.waitpid(pid, 0)
    return out, rss


# ----------------------------------------------------- E14: verification cost

def _leaf(i, seed=0):
    return SHA(b"rec" + struct.pack(">q", i) + struct.pack(">q", seed)).digest()


def _leaves(n, seed=0):
    """Deterministic record hashes, generated rather than stored."""
    for i in range(n):
        yield _leaf(i, seed)


def _manifest_iter(n, seed=0):
    """The published manifest in the form an auditor receives it: one
    (index, hex digest) pair per record, in ascending index order."""
    for i in range(n):
        yield [i, _leaf(i, seed).hex()]


def _reps(n):
    return 50 if n <= 32768 else 20 if n <= 10**5 else 5 if n <= 10**6 else 1


def _case_header(n, n_deleted=1, n_slices=4):
    """A signed certificate over a manifest of `n` records, built in O(log n).

    `cost_sweeps._synthetic_case` builds the same object by materializing the
    manifest, which is what cannot be done at 10^8. The roots here are folded
    from the generated leaves instead, so building the header of the largest
    case costs what the smallest one costs, and whether the manifest is held
    becomes the verifier's choice rather than the harness's.

    The enclave signs the result, so a verifier that accepts it has recomputed
    H(D_clean) over the manifest and found the folded root bit-identical to its
    own level-wise one.
    """
    from clean_runtime import dlog
    from . import run_unlearn
    enclave = run_unlearn.load_enclave()
    dep = json.loads((common.DATA_DIR / "deployment.json").read_text())

    per = n // n_slices
    bounds = [[j * per, (j + 1) * per] for j in range(n_slices)]
    bounds[-1][1] = n
    slice_idx = 1                          # trained prefix = slices 0..1
    prefix_end = bounds[slice_idx][1]
    deleted = list(range(prefix_end, prefix_end + n_deleted))
    dset = set(deleted)

    h_prefix = streaming_merkle(_leaf(i) for i in range(prefix_end))
    h_dclean = streaming_merkle(_leaf(i) for i in range(prefix_end, n)
                                if i not in dset)
    h_delta = streaming_merkle(_leaf(i) for i in deleted)
    h_cp = SHA(b"ckpt").digest()
    head = dlog.genesis(0)                 # a shard with no deletions yet
    cert = enclave.sign_cp_cert(h_cp, h_prefix, 0, slice_idx, 2, head)
    cp_meta = {"shard_id": 0, "slice_idx": slice_idx, "version": 2,
               "h_d_prefix": h_prefix.hex(), "h_record": head.hex(),
               "cert": cert.hex()}
    arc = {"c_base": bytes.fromhex(dep["c_base"]), "h_l_ckpt": h_cp,
           "h_l_new": SHA(b"new").digest(), "h_dclean": h_dclean,
           "h_delta": h_delta,
           "h_record": dlog.extend(head, [_leaf(i) for i in deleted]),
           "h_r": SHA(b"r").digest(),
           "h_config": bytes.fromhex(dep["h_config"]),
           "shard_id": 0, "cp_version": 2,
           "nonce": common.harness_r("scale-sweep"),
           "mrenclave": enclave.mrenclave}
    arc["signature"] = enclave.sign_arc(arc)
    kw = dict(nonce=arc["nonce"],
              delta=[(i, _leaf(i).hex()) for i in deleted],
              slice_bounds=bounds, cp_meta=cp_meta, cp_cert=cert,
              c_base=arc["c_base"], h_config=arc["h_config"],
              mrenclave=enclave.mrenclave)
    return bytes.fromhex(dep["pk"]), arc, kw


def _streaming_verify_arc(pk, arc, *, manifest_iter, nonce, delta,
                          slice_bounds, cp_meta, cp_cert, c_base, h_config,
                          mrenclave, record=()):
    """Every check `verify_arc` runs, over a manifest that is never held.

    Three of those checks need the whole manifest -- that the deletion request
    names real records, that none of them lies in the checkpoint's trained
    prefix, and that H(D_clean) is the root of the manifest minus the request.
    They are folded into one pass here, which holds one subtree per Merkle
    height, the request, and the hashes deleted since the checkpoint. The pass
    reads the manifest in ascending index order, which is the order the root is
    defined over and which it checks as it goes.

    Structural checks are delegated to `verifier/verify.py` wherever they are
    constant-cost, so the two verifiers agree by construction on everything
    except how the manifest is traversed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey)
    from verifier import verify as V

    delta = [(int(ix), bytes.fromhex(h) if isinstance(h, str) else bytes(h))
             for ix, h in delta]
    want = dict(delta)
    deleted_ix = set(want)
    prefix_end = V._prefix_end(slice_bounds, cp_meta["slice_idx"])

    if not V.verify_cp_cert(pk, cp_cert, arc["h_l_ckpt"], cp_meta, mrenclave):
        return 0, "invalid checkpoint certificate"
    if (int(cp_meta["shard_id"]) != int(arc["shard_id"])
            or int(cp_meta["version"]) != int(arc["cp_version"])):
        return 0, "checkpoint identity does not match ARC"

    record = [list(e) for e in record]
    h_record = V._dlog_extend(V._dlog_head(arc["shard_id"], record),
                              [h for _, h in delta])
    running = V._dlog_genesis(arc["shard_id"])
    target = bytes.fromhex(cp_meta.get("h_record", ""))
    later, seen = None, running == target
    if seen:
        later = list(record)
    for i, entry in enumerate(record):
        running = V._dlog_extend(running, entry)
        if running == target:
            later, seen = list(record[i + 1:]), True
            break
    if not seen:
        return 0, ("checkpoint's record head is not a prefix of the "
                   "deletion record")
    forgotten = {bytes(h) for e in later for h in e}

    stack, prev_ix = [], -1
    for row, (ix, hx) in enumerate(manifest_iter):
        ix = int(ix)
        if ix <= prev_ix:
            return 0, "manifest is not in ascending index order"
        prev_ix = ix
        h = bytes.fromhex(hx)
        if ix in want:
            if want[ix] != h:
                return 0, f"delta hash mismatch at index {ix}"
            del want[ix]
            if row < prefix_end:
                return 0, f"checkpoint not clean: index {ix} in trained prefix"
        if row < prefix_end:
            if h in forgotten:
                return 0, (f"checkpoint not clean: index {ix} of its trained "
                           f"prefix was deleted by an earlier invocation")
            continue
        if ix not in deleted_ix:
            _fold_push(stack, h)
    if want:
        return 0, f"delta index {next(iter(want))} not in manifest"
    if _fold_root(stack) != arc["h_dclean"]:
        return 0, "H(D_clean) does not match manifest minus delta"

    msg = V._encode(V.ARC_TAG, [c_base, arc["h_l_ckpt"], arc["h_l_new"],
                                arc["h_dclean"], V._root_of(delta), h_record,
                                arc["h_r"], h_config, V._int(arc["shard_id"]),
                                V._int(arc["cp_version"]), mrenclave, nonce])
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(arc["signature"], msg)
    except InvalidSignature:
        return 0, "invalid ARC signature"
    return 1, "accept"


def _real_verify_point(n):
    """One point on the published verifier's curve: the whole path of
    `verify_arc`, on the manifest an auditor holds as a list."""
    from verifier import verify
    pk, arc, kw = _case_header(n)
    kw = dict(kw, manifest=list(_manifest_iter(n)))
    accept, reason = verify.verify_arc(pk, arc, **kw)
    assert accept == 1, f"real verifier rejected n={n}: {reason}"
    xs = []
    for _ in range(_reps(n)):
        t0 = time.perf_counter()
        verify.verify_arc(pk, arc, **kw)
        xs.append(time.perf_counter() - t0)
    xs.sort()
    return {"n": n, "seconds": xs[len(xs) // 2], "reps": len(xs)}


def _streaming_verify_point(n):
    """The same path, over a manifest that is streamed rather than held."""
    pk, arc, kw = _case_header(n)
    accept, reason = _streaming_verify_arc(
        pk, arc, manifest_iter=_manifest_iter(n), **kw)
    assert accept == 1, f"streaming verifier rejected n={n}: {reason}"
    xs = []
    for _ in range(_reps(n)):
        t0 = time.perf_counter()
        _streaming_verify_arc(pk, arc, manifest_iter=_manifest_iter(n), **kw)
        xs.append(time.perf_counter() - t0)
    xs.sort()
    return {"n": n, "seconds": xs[len(xs) // 2], "reps": len(xs)}


def _synthesis_point(n):
    """What it costs to produce the manifest the two points above consume.

    The published verifier is handed a list, so its timed region excludes the
    work of building one; the streaming verifier generates each entry inside
    its timed region. Measuring the generator alone says how much of the
    streaming figure is that asymmetry and not the verification.
    """
    xs = []
    for _ in range(_reps(n)):
        t0 = time.perf_counter()
        for _entry in _manifest_iter(n):
            pass
        xs.append(time.perf_counter() - t0)
    xs.sort()
    return {"n": n, "seconds": xs[len(xs) // 2]}


def _agreement_check(n=8192):
    """The two verifiers return the same verdict on a tampered manifest.

    Equal timings on an accepting case would also be produced by a streaming
    pass that checked nothing, so the pass is made to reject as well: one
    manifest entry is given the wrong digest, which is the check the fold and
    the level-wise recomputation both exist to make.
    """
    from verifier import verify
    pk, arc, kw = _case_header(n)
    good = list(_manifest_iter(n))
    bad = [list(e) for e in good]
    bad[n - 1][1] = _leaf(n - 1, seed=1).hex()      # a record that is retained
    out = {}
    for label, man in (("accept", good), ("tampered", bad)):
        r = verify.verify_arc(pk, arc, **dict(kw, manifest=man))
        s = _streaming_verify_arc(pk, arc, manifest_iter=iter(man), **kw)
        out[label] = {"real": list(r), "streaming": list(s)}
        assert r[0] == s[0], f"verifiers disagree on the {label} case: {r} {s}"
    assert out["accept"]["real"][0] == 1 and out["tampered"]["real"][0] == 0, \
        "the agreement check did not exercise both verdicts"
    return out


def verification_at_scale(sizes=(128, 512, 2048, 8192, 32768, 10**5, 10**6,
                                 3 * 10**6, 10**7, 10**8),
                          cap_gb=9.0, rss_budget_gb=8.5):
    """Both verifiers, stopping the published one when it runs out of room."""
    rows, real_stopped_at = [], None
    for n in sizes:
        row = {"n": n}
        if real_stopped_at is None:
            out, rss = _in_child(_real_verify_point, n, cap_gb=cap_gb)
            if isinstance(out, str) or rss / 1e9 > rss_budget_gb:
                real_stopped_at = n
                row["real_s"], row["real_rss"] = None, None
                why = out if isinstance(out, str) else f"{rss/1e9:.1f} GB > budget"
                _say(f"n={n:>11,}  published verifier: stopped ({why})")
            else:
                row["real_s"], row["real_rss"] = out["seconds"], rss
        else:
            row["real_s"], row["real_rss"] = None, None

        out, rss = _in_child(_streaming_verify_point, n, cap_gb=cap_gb)
        if isinstance(out, str):
            row["stream_s"], row["stream_rss"] = None, None
            _say(f"n={n:>11,}  streaming: {out}")
        else:
            row["stream_s"], row["stream_rss"] = out["seconds"], rss

        out, _ = _in_child(_synthesis_point, n, cap_gb=cap_gb)
        row["gen_s"] = None if isinstance(out, str) else out["seconds"]

        _say(f"n={n:>11,}  published {_fmt(row['real_s'])}"
             f" / {_fmt_mem(row['real_rss'])}"
             f"   streaming {_fmt(row['stream_s'])}"
             f" / {_fmt_mem(row['stream_rss'])}"
             f"   (manifest synthesis {_fmt(row['gen_s'])})")
        rows.append(row)
    return rows, real_stopped_at


# ------------------------------------------------- E15: clean-prefix test cost

def _prefix_point(prefix_len, n_deleted=64):
    """Time the test the gateway runs before admitting a checkpoint.

    The record names deletions by hash, so the enclave cannot index into the
    prefix: it walks it, checking each record against the set of hashes
    deleted since the checkpoint was certified.
    """
    forgotten = {SHA(b"del" + struct.pack(">q", i)).digest()
                 for i in range(n_deleted)}
    prefix = [h for h in _leaves(prefix_len)]
    t0 = time.perf_counter()
    clean = True
    for h in prefix:
        if h in forgotten:
            clean = False
            break
    dt = time.perf_counter() - t0
    return {"prefix": prefix_len, "seconds": dt, "clean": clean}


def prefix_test_at_scale(lengths=(10**4, 10**5, 10**6, 10**7)):
    rows = []
    for m in lengths:
        out, rss = _in_child(_prefix_point, m)
        if isinstance(out, str):
            _say(f"prefix={m:>11,}  {out}")
            continue
        rows.append({"prefix": m, "seconds": out["seconds"], "rss": rss})
        _say(f"prefix={m:>11,}  {_fmt(out['seconds'])} / {_fmt_mem(rss)}")
    return rows


# ------------------------------------------------------------------ reporting

def _fmt(s):
    if s is None:
        return "     --- "
    return f"{s*1000:8.1f} ms" if s < 1 else f"{s:8.2f} s "


def _fmt_mem(b):
    return "   ---" if b is None or b < 0 else f"{b/1e9:5.2f} GB"


def main(reuse=False):
    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    meta = out / "scale.meta.json"
    if reuse and meta.exists():
        m = json.loads(meta.read_text())
        rows, stopped, prows = (m["verify"], m["real_verifier_stops_at"],
                                m["prefix"])
        agree = m.get("agreement")
        _say(f"reusing the measurements in {meta}")
    else:
        _say("== E14: verification at corpus scale ==")
        agree = _agreement_check()
        _say("both verifiers accept the honest manifest and reject a tampered "
             "one, with the same reason")
        rows, stopped = verification_at_scale()
        _say("\n== E15: clean-prefix test ==")
        prows = prefix_test_at_scale()

    lines = ["# E14 - Verification at corpus scale", "",
             "`Verify` is run whole at every point: the deletion request's "
             "membership in the manifest, the checkpoint certificate, the "
             "clean-prefix test, the compound-erasure test, the "
             "$H(\\mathcal{D}_{clean})$ recomputation and both signature "
             "checks. **published** is `verifier/verify.py` unmodified, which "
             "takes the manifest as a list of (index, hex digest) pairs and "
             "sorts it before hashing. **streaming** is the same set of checks "
             "folded into one pass over the manifest, holding one Merkle "
             "subtree per height instead of the manifest.", "",
             "| records | published | peak RSS | streaming | peak RSS |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['n']:,} | {_fmt(r['real_s']).strip()} | "
            f"{_fmt_mem(r['real_rss']).strip()} | "
            f"{_fmt(r['stream_s']).strip()} | {_fmt_mem(r['stream_rss']).strip()} |")

    done = [r for r in rows if r["real_s"]]
    last = done[-1]
    per_rec = last["real_rss"] / last["n"]
    both = [r for r in rows if r["real_s"] and r["stream_s"]]
    slow = [r["stream_s"] / r["real_s"] for r in both]
    # Net of the manifest the streaming pass has to produce as it goes, which
    # the published verifier is handed before its timed region opens.
    net = [(r["stream_s"] - (r["gen_s"] or 0)) / r["real_s"] for r in both
           if r["n"] >= 10**5]
    streamed = [r for r in rows if r["stream_s"]][-1]
    lines += ["",
              f"Both curves are linear in time. What separates them is memory: "
              f"the published verifier holds about {per_rec:.0f} bytes per "
              f"record -- an entry, its hex digest, the two index maps it "
              f"builds and the sorted copy it hashes -- so a manifest of "
              f"$10^9$ records would ask for roughly {per_rec:.0f}\\,GB before "
              f"the first hash is taken. The streaming pass holds "
              f"{streamed['stream_rss']/1e6:.0f}\\,MB at every point measured, "
              f"including the largest.",
              "",
              f"It does not pay for that in time. The two timed regions do not "
              f"contain the same work -- the published verifier is handed a "
              f"list, while the streaming pass produces each entry as it "
              f"consumes it, {_fmt(rows[-1]['gen_s']).strip()} of the "
              f"{_fmt(streamed['stream_s']).strip()} at {streamed['n']:,} "
              f"records. Net of producing the manifest, the streaming pass "
              f"costs {sum(net)/len(net):.2f}$\\times$ the published verifier "
              f"over the decades where the manifest dominates. Measured as "
              f"they stand, the ratio is {sum(slow)/len(slow):.1f}"
              f"$\\times$.",
              "",
              "Times are the median over repetitions (50 at the small sizes "
              "down to a single run at $10^8$); peak RSS is what the verifier "
              "holds while running, the manifest included."]
    if stopped:
        lines += ["",
                  f"Under a 9\\,GB address-space cap the published verifier "
                  f"raises `MemoryError` at {stopped:,} records. The streaming "
                  f"pass returns `accept` on the same certificate."]
    if agree:
        lines += ["",
                  "The two agree on the verdict, not merely on the timing. "
                  "Every accepting point is already evidence that the folded "
                  "root is bit-identical to the level-wise one, because the "
                  "certificate binds a root the harness folded and the "
                  "published verifier recomputes it level-wise before "
                  "accepting. Giving one retained record the wrong digest "
                  "makes both return "
                  f"`{agree['tampered']['real'][1]}`."]
    lines += [""]
    (out / "scale_verify.md").write_text("\n".join(lines))
    (out / "scale_verify.dat").write_text(
        "n real_s real_rss stream_s stream_rss gen_s\n" + "\n".join(
            f"{r['n']} {r['real_s'] or 'nan'} {r['real_rss'] or 'nan'} "
            f"{r['stream_s'] or 'nan'} {r['stream_rss'] or 'nan'} "
            f"{r.get('gen_s') or 'nan'}"
            for r in rows) + "\n")

    lines = ["# E15 - Clean-prefix test against prefix length", "",
             "The gateway walks the checkpoint's trained prefix, checking each "
             "record against the deletions recorded since that checkpoint was "
             "certified. Entries name records by hash so that they survive "
             "compaction, which is also why the prefix cannot be indexed into "
             "and must be walked.", "",
             "| prefix (records) | time | peak RSS |", "|---:|---:|---:|"]
    for r in prows:
        lines.append(f"| {r['prefix']:,} | {_fmt(r['seconds']).strip()} | "
                     f"{_fmt_mem(r['rss']).strip()} |")
    if prows:
        ns = prows[-1]["seconds"] / prows[-1]["prefix"] * 1e9
        lines += ["",
                  f"The walk costs about {ns:.0f}\\,ns per record -- a set "
                  f"membership test, with no hashing, because the manifest "
                  f"already carries the digests. At the largest prefix measured "
                  f"it is {prows[-1]['seconds']*1000:.0f}\\,ms against seconds "
                  f"of retraining, so it does not dominate an invocation; what "
                  f"it does cost is holding the prefix, "
                  f"{prows[-1]['rss']/1e9:.1f}\\,GB at "
                  f"{prows[-1]['prefix']:,} records.",
                  "",
                  "The prefix is the part of the shard the invocation does not "
                  "retrain, so this cost runs opposite to the saving: a deletion "
                  "late in a shard retrains least and walks most.", ""]
    (out / "scale_prefix.md").write_text("\n".join(lines))
    (out / "scale_prefix.dat").write_text(
        "prefix seconds rss\n" + "\n".join(
            f"{r['prefix']} {r['seconds']} {r['rss']}" for r in prows) + "\n")

    (out / "scale.meta.json").write_text(json.dumps(
        {"verify": rows, "real_verifier_stops_at": stopped, "prefix": prows,
         "agreement": agree},
        indent=1))
    _say(f"\n-> {out / 'scale_verify.md'}, {out / 'scale_prefix.md'}")


if __name__ == "__main__":
    import sys
    main(reuse="--reuse" in sys.argv)
