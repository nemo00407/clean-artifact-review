"""E3 - Compliance-cheating attack suite (evaluation.md section 4).

Red-team attacks produce genuinely enclave-signed ARCs over tampered inputs
(a malicious host controls what the runtime is fed, not the signing key).
Each structural attack is run against (a) the full CLEAN verifier, which
must REJECT, and (b) the verifier ablation with the relevant binding
removed, which then ACCEPTS - proving the binding is load-bearing. Attacks
defeated purely by signature unforgeability have no corresponding ablation.
An honest run is the completeness control; randomness manipulation is
blocked at the API layer.
"""
import json

import numpy as np

from clean_runtime import commit, device, gateway, rng, runtime, store
from clean_runtime.gateway import DatasetHandle, Invocation
from clean_runtime.rng import TeeRng
from verifier import verify

from . import common, run_unlearn, untrusted_store

SHARD, SLICE, CPV = 0, 2, 2
NONCE_LABEL = "attack-nonce"


def _config():
    return json.loads((common.DATA_DIR / "config.json").read_text())


def _context():
    dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
    manifest = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.manifest.json").read_text())
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    cp_meta = json.loads(
        (common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz.meta.json").read_text())
    delta = run_unlearn.pick_delta(SHARD, SLICE)
    exp = dict(nonce=common.harness_r(NONCE_LABEL), delta=delta,
               manifest=manifest, slice_bounds=bounds, cp_meta=cp_meta,
               cp_cert=bytes.fromhex(cp_meta["cert"]),
               c_base=bytes.fromhex(dep["c_base"]),
               h_config=bytes.fromhex(dep["h_config"]),
               mrenclave=bytes.fromhex(dep["mrenclave"]))
    return dep, exp, delta, manifest


def _honest_arc(delta):
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz",
        common.DATA_DIR / f"shard_{SHARD}.npz",
        delta, enclave.derive_r(nonce), _config(), nonce=nonce)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _shard():
    with np.load(common.DATA_DIR / f"shard_{SHARD}.npz") as npz:
        return npz["index"].copy(), npz["salt"].copy(), npz["tokens"].copy()


def _red_team_dataset_tampering(delta):
    """Host runs the enclave over the retrain region WITHOUT excluding Delta
    (an un-tombstoned shard), yielding a validly signed ARC bound to the
    tampered H(D_clean)."""
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    index, salt, tokens = _shard()
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    lo = bounds[CPV][0]  # retrain region start for the clean checkpoint
    rows = np.arange(lo, bounds[-1][1])  # NOTE: Delta rows NOT removed
    data = DatasetHandle(index[rows], salt[rows], tokens[rows],
                         commit.dataset_root(index[rows], salt[rows],
                                             tokens[rows]))
    dh = {int(i): bytes.fromhex(h) for i, h in delta}
    delta_idx = np.array(sorted(dh), dtype=np.uint32)
    h_delta = commit.hashes_root(delta_idx, [dh[int(i)] for i in delta_idx])
    base = gateway.load_base(common.DATA_DIR / "base.npz")
    cp = gateway.load_checkpoint(common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz")
    inv = Invocation(base, cp, data, TeeRng(enclave.derive_r(nonce)),
                     _config(), h_delta, delta_idx, nonce)
    # A real enclave derives the record head from the request it was handed;
    # these invocations bypass the gateway, so the red team supplies it.
    inv.delta_hashes = tuple(dh[int(i)] for i in delta_idx)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _red_team_checkpoint_substitution(delta):
    """Host runs the enclave from a checkpoint whose contents match no
    TEE-issued certificate (adapter bytes tampered), yielding a validly
    signed ARC bound to an uncertified H(L_ckpt)."""
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    cp = gateway.load_checkpoint(common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz")
    from clean_runtime.gateway import CheckpointHandle
    tampered = {k: np.array(v, dtype=np.float32) + 0.01
                for k, v in cp.lora.items()}
    cp_t = CheckpointHandle(tampered, dict(cp.meta))
    index, salt, tokens = _shard()
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    lo = bounds[CPV][0]
    deleted = {int(i) for i, _ in delta}
    rows = np.array([r for r in range(lo, bounds[-1][1])
                     if int(index[r]) not in deleted])
    data = DatasetHandle(index[rows], salt[rows], tokens[rows],
                         commit.dataset_root(index[rows], salt[rows],
                                             tokens[rows]))
    dh = {int(i): bytes.fromhex(h) for i, h in delta}
    delta_idx = np.array(sorted(deleted), dtype=np.uint32)
    h_delta = commit.hashes_root(delta_idx, [dh[int(i)] for i in delta_idx])
    base = gateway.load_base(common.DATA_DIR / "base.npz")
    inv = Invocation(base, cp_t, data, TeeRng(enclave.derive_r(nonce)),
                     _config(), h_delta, delta_idx, nonce)
    # A real enclave derives the record head from the request it was handed;
    # these invocations bypass the gateway, so the red team supplies it.
    inv.delta_hashes = tuple(dh[int(i)] for i in delta_idx)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _red_team_unclean_checkpoint(dirty):
    """Host bypasses the gateway's clean guard: deletes a record that lies
    in the checkpoint's trained prefix, so retraining from that checkpoint
    cannot remove its influence, yet the enclave still signs the ARC."""
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    index, salt, tokens = _shard()
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    lo = bounds[CPV][0]
    deleted = {int(i) for i, _ in dirty}
    rows = np.array([r for r in range(lo, bounds[-1][1])
                     if int(index[r]) not in deleted])
    data = DatasetHandle(index[rows], salt[rows], tokens[rows],
                         commit.dataset_root(index[rows], salt[rows],
                                             tokens[rows]))
    dh = {int(i): bytes.fromhex(h) for i, h in dirty}
    delta_idx = np.array(sorted(dh), dtype=np.uint32)
    h_delta = commit.hashes_root(delta_idx, [dh[int(i)] for i in delta_idx])
    base = gateway.load_base(common.DATA_DIR / "base.npz")
    cp = gateway.load_checkpoint(common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz")
    inv = Invocation(base, cp, data, TeeRng(enclave.derive_r(nonce)),
                     _config(), h_delta, delta_idx, nonce)
    # A real enclave derives the record head from the request it was handed;
    # these invocations bypass the gateway, so the red team supplies it.
    inv.delta_hashes = tuple(dh[int(i)] for i in delta_idx)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _red_team_mid_invocation_rollback(delta, verify_reads=True):
    """Host lets the invocation admit a correctly excluded D_clean, then
    serves a stale copy of one dataset block once training has started.

    The stale bytes are the pre-deletion contents of the same block, so the
    deleted records re-enter the state transition *after* H(D_clean) was
    committed. Nothing about the block is malformed and H(D_clean) still
    matches the manifest, so no verifier binding can distinguish the run:
    only a commitment recorded at admission and checked on the fetch path
    can. Returns (arc, backend); with `verify_reads` the fetch raises and no
    certificate is issued at all.
    """
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    index, _, tokens = _shard()
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    cp_meta = json.loads(
        (common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz.meta.json").read_text())

    region = np.arange(bounds[cp_meta["slice_idx"] + 1][0], bounds[-1][1])
    stale = store.encode_block(tokens[region][:store.BLOCK_ROWS])
    backend = untrusted_store.RollbackBackend(
        "dclean/tokens/0", stale, trigger_after=1)

    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz",
        common.DATA_DIR / f"shard_{SHARD}.npz",
        delta, enclave.derive_r(nonce), _config(), nonce=nonce,
        backend=backend, verify=verify_reads)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc, backend


def _red_team_transfer_tampering(delta, verify_reads=True):
    """Host leaves storage intact but corrupts blocks in flight across the
    PCIe path, modelling a compromised driver or DMA engine.

    Nothing on the storage side is wrong: the backend still holds exactly
    what the gateway admitted, and a check performed before the transfer
    would pass. Only the copy that arrives on the device differs. This is
    what the commitment check buys by sitting after arrival---the transport
    has to be trusted to deliver something, never to deliver the right
    thing. Returns (arc, path).
    """
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    path = untrusted_store.TamperingPath(trigger_after=1)
    dev = device.DeviceContext(path=path)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz",
        common.DATA_DIR / f"shard_{SHARD}.npz",
        delta, enclave.derive_r(nonce), _config(), nonce=nonce,
        verify=verify_reads, device=dev)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc, path


def _red_team_config_steering(delta):
    """Host runs the enclave over correctly excluded data from a certified
    clean checkpoint, but selects the training configuration as a function
    of the deletion request.

    Every data binding verifies and no deleted record is read, yet the
    output depends on Delta through c, exactly as it would through a
    host-chosen seed. Only H(c) in the certificate, checked against the
    configuration published at deployment, separates this run from an
    honest one."""
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(NONCE_LABEL)
    steered = dict(_config())
    steered["epochs"] = int(steered["epochs"]) + len(delta)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{SHARD}_v{CPV}.npz",
        common.DATA_DIR / f"shard_{SHARD}.npz",
        delta, enclave.derive_r(nonce), steered, nonce=nonce)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _v(pk, arc, exp, **over):
    e = dict(exp)
    e.update(over)
    return verify.verify_arc(pk, arc, **e)


def run():
    dep, exp, delta, manifest = _context()
    pk = bytes.fromhex(dep["pk"])
    rows = []

    # completeness control
    assert _v(pk, _honest_arc(delta), exp)[0] == 1, "honest run must verify"

    # 1. Retraining skipping via external forgery (no enclave invocation):
    #    fabricate an ARC with an invalid signature. Caught by unforgeability;
    #    no verifier ablation recovers it.
    honest = _honest_arc(delta)
    forged = dict(honest, signature=b"\x00" * 64)
    rows.append(("Retraining Skipping", _v(pk, forged, exp)[0], None,
                 "external forgery; caught by ARC unforgeability"))

    # 2. Certificate replay: genuine old ARC replayed under a fresh session.
    full = _v(pk, honest, exp, nonce=common.harness_r("fresh-session"))[0]
    weak = _v(pk, honest, exp, nonce=common.harness_r("fresh-session"),
              unsafe_skip=frozenset({"nonce"}))[0]
    rows.append(("Certificate Replay", full, weak,
                 "freshness: nonce binding; NoNonce ablation accepts"))

    # 3. Dataset tampering: enclave-signed ARC over an un-tombstoned shard.
    arc3 = _red_team_dataset_tampering(delta)
    rows.append(("Dataset Tampering", _v(pk, arc3, exp)[0],
                 _v(pk, arc3, exp, unsafe_skip=frozenset({"dclean"}))[0],
                 "H(D_clean) recomputation; NoDataBind ablation accepts"))

    # 4. Checkpoint substitution: enclave-signed ARC from an uncertified CP.
    arc4 = _red_team_checkpoint_substitution(delta)
    rows.append(("Checkpoint Substitution", _v(pk, arc4, exp)[0],
                 _v(pk, arc4, exp, unsafe_skip=frozenset({"cp_cert"}))[0],
                 "CP_Cert binding; NoCPCert ablation accepts"))

    # 4b. Unclean checkpoint: deletion target lies in the trained prefix of
    #     the presented checkpoint (host bypassed the gateway clean guard).
    dirty = [(manifest[1][0], manifest[1][1])]
    arc5 = _red_team_unclean_checkpoint(dirty)
    rows.append(("Unclean Checkpoint",
                 _v(pk, arc5, exp, delta=dirty)[0],
                 _v(pk, arc5, exp, delta=dirty,
                    unsafe_skip=frozenset({"clean"}))[0],
                 "clean-rollback judgment; NoClean ablation accepts"))

    # 6. Mid-invocation shard rollback: a stale block is served after the
    #    invocation committed to H(D_clean). Refused on the fetch path, so no
    #    ARC exists to verify. With re-verification removed the run completes
    #    and the resulting ARC passes the FULL verifier -- the one attack no
    #    verifier binding can recover, which is why the check must sit in the
    #    runtime rather than in the auditor.
    try:
        _red_team_mid_invocation_rollback(delta)
        full6 = 1  # certificate issued despite the substitution
    except store.BlockIntegrityError:
        full6 = 0
    arc6, backend6 = _red_team_mid_invocation_rollback(delta,
                                                       verify_reads=False)
    assert backend6.substitutions > 0, "rollback backend never fired"
    rows.append(("Mid-Invocation Shard Rollback", full6,
                 _v(pk, arc6, exp)[0],
                 "block re-verification on fetch; NoReverify ablation "
                 "yields an ARC the full verifier accepts"))

    # 7. PCIe transfer tampering: storage is untouched; the bytes are altered
    #    in flight. Caught for the same reason and by the same check, which
    #    is the point: the check sits on the device side of the boundary, so
    #    the transport needs no trust. Without it the run completes and the
    #    ARC again passes the full verifier.
    try:
        _red_team_transfer_tampering(delta)
        full7 = 1
    except store.BlockIntegrityError:
        full7 = 0
    arc7, path7 = _red_team_transfer_tampering(delta, verify_reads=False)
    assert path7.tampered > 0, "tampering path never fired"
    rows.append(("PCIe Transfer Tampering", full7, _v(pk, arc7, exp)[0],
                 "commitment checked after arrival on the device; "
                 "NoReverify ablation yields an ARC the full verifier accepts"))

    # 8. Configuration steering: the fourth input of R, chosen as a function
    #    of Delta. No data binding is violated; only H(c) catches it.
    arc8 = _red_team_config_steering(delta)
    rows.append(("Configuration Steering", _v(pk, arc8, exp)[0],
                 _v(pk, arc8, exp, unsafe_skip=frozenset({"config"}))[0],
                 "H(c) against the deployed configuration; "
                 "NoConfigBind ablation accepts"))
    return honest, rows


def randomness_manipulation():
    try:
        rng.TeeRng(b"host-chosen-short-seed")
        return False, "TeeRng accepted host seed (unexpected)"
    except ValueError:
        return True, ("no seed interface; r is enclave-derived and bound "
                      "via H(r) in the ARC")


def main():
    honest, rows = run()
    ok_r, note_r = randomness_manipulation()

    dep, exp, _, _ = _context()
    pk = bytes.fromhex(dep["pk"])
    import time
    samples = []
    for _ in range(2000):
        t0 = time.perf_counter()
        verify.verify_arc(pk, honest, **exp)
        samples.append((time.perf_counter() - t0) * 1000)
    vmed, vp99 = float(np.median(samples)), float(np.percentile(samples, 99))

    verdict = {0: "REJECT", 1: "ACCEPT", None: "-"}
    lines = ["# E3 - Attack Interception Matrix", "",
             "Red-team attacks present genuinely enclave-signed ARCs over "
             "tampered inputs. The full verifier rejects every attack; the "
             "verifier ablation with the named binding removed then accepts, "
             "showing each defense is load-bearing.", "",
             "| attack vector | CLEAN (full) | verifier ablation | defense |",
             "|---|---|---|---|"]
    for name, full, weak, note in rows:
        wcol = f"{verdict[weak]} (vuln.)" if weak == 1 else verdict[weak]
        lines.append(f"| {name} | {verdict[full]} | {wcol} | {note} |")
    lines.append(f"| Randomness Manipulation | REJECT (API) | - | {note_r} |")
    lines += ["",
              f"Honest run verifies (completeness). Verify() over 2000 calls: "
              f"median {vmed:.3f} ms, p99 {vp99:.3f} ms.", ""]

    all_rejected = all(f == 0 for _, f, _, _ in rows) and ok_r
    ablations_expose = all(w == 1 for _, _, w, _ in rows if w is not None)
    lines.append(f"All attacks rejected by full verifier: {all_rejected}. "
                 f"Every structural ablation re-exposes its attack: "
                 f"{ablations_expose}.")

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "attacks.md").write_text("\n".join(lines))
    for name, full, weak, _ in rows:
        print(f"[{name}] full={verdict[full]} ablation={verdict[weak]}")
    print(f"[Randomness Manipulation] api_reject={ok_r}")
    print(f"all_rejected={all_rejected} ablations_expose={ablations_expose} "
          f"verify median {vmed:.3f}ms")


if __name__ == "__main__":
    main()
