"""E3b - Attack x ablation cross product (non-TCB harness).

Every attack of the threat model is run against the full system and against
each configuration with exactly one defense removed. A cell records whether
the adversary obtains a certificate the auditor accepts.

The diagonal shows each defense is necessary; the off-diagonal shows the
defenses are orthogonal, i.e. no binding silently covers another's attack.
Two attacks are refused before a certificate exists: the post-admission
substitutions, which the runtime aborts on the read path, and randomness
manipulation, which has no interface to attempt. For those, the column that
removes the responsible defense is the only one in which a certificate
appears at all -- and it is then accepted by every verifier configuration,
which is why that defense cannot live in the auditor.
"""
import json

from clean_runtime import gateway, rng, runtime, store
from verifier import verify

from . import attacks, common, compound, run_unlearn

# A configuration is the full system, or the full system with one defense
# removed. Verifier ablations name a binding in verify_arc's unsafe_skip;
# the runtime ablation disables the commitment check on the read path.
RUNTIME_ABLATION = "-NoReverify"
# (name, short label for the figure, bindings removed)
COLUMNS = [
    ("full", "full", frozenset()),
    ("-NoNonce", "Nonce", frozenset({"nonce"})),
    ("-NoDataBind", "Data", frozenset({"dclean"})),
    ("-NoDeltaBind", "Delta", frozenset({"delta_binding"})),
    # The deletion request is bound three ways: through H(D_clean), which the
    # auditor recomputes from the manifest minus its own request, through
    # H(Delta) in the signed payload, and through the record head the payload
    # also carries. No one of them alone admits a substituted deletion set, so
    # the conjunction is the informative ablation.
    ("-NoDataBind -NoDeltaBind -NoRecord", "Request",
     frozenset({"dclean", "delta_binding", "record"})),
    ("-NoCPCert", "CPCert", frozenset({"cp_cert"})),
    ("-NoClean", "Clean", frozenset({"clean"})),
    ("-NoConfigBind", "Config", frozenset({"config"})),
    ("-NoRecord", "Record", frozenset({"record"})),
    (RUNTIME_ABLATION, "Reverify", None),
]

ACCEPT, REJECT, NO_CERT = "accept", "reject", "none"


def _red_team_deletion_substitution(requested, actual):
    """Host runs the enclave honestly, but on a deletion set of its own.

    Every input is well formed and the run is genuinely attested; the
    certificate simply binds H(Delta') for the set the host chose rather
    than the one the data owner asked for. This is the separating case
    between Definition 1 and Definition 2: soundness holds -- the
    certificate does not lie about what ran -- while binding fails.
    """
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(attacks.NONCE_LABEL)
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{attacks.SHARD}_v{attacks.CPV}.npz",
        common.DATA_DIR / f"shard_{attacks.SHARD}.npz",
        actual, enclave.derive_r(nonce), attacks._config(), nonce=nonce)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def _producers(delta, manifest):
    """name -> callable(runtime_ablated) -> (arc, verify_overrides) or None.

    Returning None means the attack never yields a certificate in that
    configuration, because it was refused before one could be issued.
    """
    dirty = [(manifest[1][0], manifest[1][1])]
    other = run_unlearn.pick_delta(attacks.SHARD, attacks.SLICE + 1, 1)

    def skipping(_ablated):
        honest = attacks._honest_arc(delta)
        return dict(honest, signature=b"\x00" * 64), {}

    def replay(_ablated):
        return attacks._honest_arc(delta), {
            "nonce": common.harness_r("fresh-session")}

    def deletion_substitution(_ablated):
        return _red_team_deletion_substitution(delta, other), {}

    def dataset_tampering(_ablated):
        return attacks._red_team_dataset_tampering(delta), {}

    def checkpoint_substitution(_ablated):
        return attacks._red_team_checkpoint_substitution(delta), {}

    def unclean_checkpoint(_ablated):
        return attacks._red_team_unclean_checkpoint(dirty), {"delta": dirty}

    def rollback(ablated):
        if not ablated:
            return None  # refused on the read path; no certificate exists
        arc, backend = attacks._red_team_mid_invocation_rollback(
            delta, verify_reads=False)
        assert backend.substitutions > 0, "rollback backend never fired"
        return arc, {}

    def transfer(ablated):
        if not ablated:
            return None
        arc, path = attacks._red_team_transfer_tampering(
            delta, verify_reads=False)
        assert path.tampered > 0, "tampering path never fired"
        return arc, {}

    def config_steering(_ablated):
        return attacks._red_team_config_steering(delta), {}

    def randomness(_ablated):
        return None  # no seed interface exists to attempt it

    resurrection = {}

    def deletion_resurrection(_ablated):
        # Needs a shard with a deletion already behind it, so it is built on
        # a private copy and carries its own expectations.
        if not resurrection:
            arc, over = compound.resurrection_case()
            resurrection["v"] = (arc, over)
        return resurrection["v"]

    # Ordered as the threat model enumerates them: evidence without an
    # execution, then an execution on host-chosen inputs (one entry per
    # bound field), then an input changed after it was admitted.
    return [
        ("Retraining Skipping", skipping),
        ("Certificate Replay", replay),
        ("Deletion-Set Substitution", deletion_substitution),
        ("Checkpoint Substitution", checkpoint_substitution),
        ("Unclean Checkpoint", unclean_checkpoint),
        ("Deletion Resurrection", deletion_resurrection),
        ("Dataset Tampering", dataset_tampering),
        ("Randomness Manipulation", randomness),
        ("Configuration Steering", config_steering),
        ("Mid-Invocation Rollback", rollback),
        ("Transfer Tampering", transfer),
    ]


def build():
    """Returns (attack_names, column_names, cells) with cells[i][j] in
    {accept, reject, none}."""
    dep, exp, delta, manifest = attacks._context()
    pk = bytes.fromhex(dep["pk"])
    producers = _producers(delta, manifest)

    names = [n for n, _ in producers]
    cols = [c for c, _, _ in COLUMNS]
    short = [t for _, t, _ in COLUMNS]
    cells = []
    for name, produce in producers:
        # one honest-runtime artifact, one runtime-ablated artifact
        normal = produce(False)
        ablated = produce(True)
        row = []
        for col, _short, skip in COLUMNS:
            if col == RUNTIME_ABLATION:
                art, use_skip = ablated, frozenset()
            else:
                art, use_skip = normal, skip
            if art is None:
                row.append(NO_CERT)
                continue
            arc, over = art
            kw = dict(exp)
            kw.update(over)
            kw["unsafe_skip"] = use_skip
            row.append(ACCEPT if verify.verify_arc(pk, arc, **kw)[0] == 1
                       else REJECT)
        cells.append(row)
        print(f"[{name}] " + " ".join(f"{c}={v}" for c, v in zip(short, row)))
    return names, cols, short, cells


def main():
    names, cols, short, cells = build()
    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)

    sym = {ACCEPT: "**accept**", REJECT: "reject", NO_CERT: "--"}
    lines = ["# E3b - Attack x ablation cross product", "",
             "Each column is the system with one defense removed. A cell says "
             "whether the adversary obtains a certificate the auditor "
             "accepts. `--` means the attack was refused before any "
             "certificate existed.", "",
             "| attack | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, row in zip(names, cells):
        lines.append(f"| {name} | " + " | ".join(sym[v] for v in row) + " |")

    accepts = [(names[i], cols[j])
               for i, row in enumerate(cells)
               for j, v in enumerate(row) if v == ACCEPT]
    full_col = cols.index("full")
    no_accept_full = all(row[full_col] != ACCEPT for row in cells)
    lines += ["",
              f"Attacks: {len(names)}. Configurations: {len(cols)}. "
              f"Cells accepting: {len(accepts)} of {len(names) * len(cols)}.",
              f"No attack succeeds against the full system: {no_accept_full}.",
              ""]
    (out / "attack_matrix.md").write_text("\n".join(lines))

    # pgfplots matrix input: x=column index, y=row index, v in {0,1,2}
    code = {REJECT: 0, ACCEPT: 1, NO_CERT: 2}
    dat = ["x y v"]
    for i, row in enumerate(cells):
        for j, v in enumerate(row):
            dat.append(f"{j} {len(names) - 1 - i} {code[v]}")
    (out / "attack_matrix.dat").write_text("\n".join(dat) + "\n")
    (out / "attack_matrix.labels.json").write_text(
        json.dumps({"attacks": names, "columns": cols, "short": short},
                   indent=1))

    print(f"-> {out / 'attack_matrix.md'} ({len(accepts)} accepting cells, "
          f"full-system clean={no_accept_full})")


if __name__ == "__main__":
    main()
