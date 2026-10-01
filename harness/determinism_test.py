"""E2 - Deterministic execution contract check (spec section 5).

Re-runs the identical unlearning invocation N times through fresh gateway
invocations and requires byte-identical adapters and ARC binding fields.
Empirical evidence for the fixed-input determinism premise of the Deletion
Non-Interference theorem.
"""
from clean_runtime import commit

from . import common, run_unlearn


def main(n=5, write=True):
    outputs = []
    for _ in range(n):
        new_lora, arc, _, _ = run_unlearn.run(out_name=None)
        outputs.append((commit.params_hash(new_lora),
                        {k: v for k, v in arc.items() if k != "signature"}))
    h0, arc0 = outputs[0]
    ok = True
    for i, (h, arc) in enumerate(outputs[1:], start=2):
        if h != h0 or arc != arc0:
            ok = False
    print(f"determinism {'OK' if ok else 'FAILED'}: {n} runs, "
          f"adapter hash {h0.hex()[:16]}...")
    if write:
        out = common.DATA_DIR.parent / "results"
        out.mkdir(exist_ok=True)
        lines = ["# E2 - Deterministic Execution Contract", "",
                 f"Identical invocation re-run **{n}** times through fresh "
                 "gateway invocations.", "",
                 f"- adapter hash (all runs): `{h0.hex()}`",
                 f"- byte-identical adapters and ARC fields: **{ok}**", "",
                 "Ed25519 is deterministic, so the ARC is identical bit for "
                 "bit as well. Single-host result; cross-hardware determinism "
                 "is the deployment obligation of the execution contract "
                 "(Assumption 3), not proven here.", ""]
        (out / "determinism.md").write_text("\n".join(lines))
    return ok


if __name__ == "__main__":
    main()
