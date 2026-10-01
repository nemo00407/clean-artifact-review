"""E4 - TCB LOC breakdown (evaluation.md section 4).

Counts non-blank, non-comment source lines of each runtime (TCB) module and
tags what each is trusted for, then contrasts with the non-TCB components.
Emitted as results/tcb_loc.md for the paper's TCB audit table.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "clean_runtime"

TRUSTED_FOR = {
    "gateway.py": "input confinement (sole mediation point; physical exclusion of Delta)",
    "loader.py": "clean-state selection / deletion-free input parsing",
    "rng.py": "deletion-independent attested randomness",
    "kernel.py": "non-semantic (numerical behavior carries no security obligation)",
    "commit.py": "canonical serialization and commitments",
    "attest.py": "MRENCLAVE, CP_Cert / ARC issuance timing",
    "dlog.py": "compound erasure (append-only deletion record; the enclave "
               "retains only its head)",
    "store.py": "temporal input closure (block commitments; re-verification "
                "on every fetch from untrusted storage)",
    "device.py": "device boundary (single ingress; verification positioned "
                 "after the transfer, so the transport needs no trust)",
    "runtime.py": "closed-input state transition; ARC issued only after success",
    "__init__.py": "package doc / import policy",
}


def sloc(path):
    """Non-blank, non-comment physical lines, excluding pure docstring lines."""
    src = path.read_text()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_lines.add(ln)
    count = 0
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in doc_lines:
            continue
        count += 1
    return count


def main():
    rows, total = [], 0
    for path in sorted(RUNTIME.glob("*.py")):
        n = sloc(path)
        total += n
        rows.append((path.name, n, TRUSTED_FOR.get(path.name, "")))

    non_tcb = {}
    for sub in ("verifier", "harness"):
        d = ROOT / sub
        non_tcb[sub] = sum(sloc(p) for p in d.glob("*.py"))

    lines = ["# E4 - Trusted Computing Base (LOC breakdown)", "",
             "Effective source lines (non-blank, non-comment, docstrings "
             "excluded) of the CLEAN runtime. The training kernel is inside "
             "the TCB for confinement but is non-semantic: its numerical "
             "behavior carries no security obligation.", "",
             "| module | SLOC | trusted for |", "|---|---:|---|"]
    for name, n, tag in rows:
        lines.append(f"| `{name}` | {n} | {tag} |")
    lines.append(f"| **total TCB** | **{total}** | |")
    lines += ["",
              "Non-TCB components (not trusted for the erasure guarantee):", "",
              "| component | SLOC | role |", "|---|---:|---|",
              f"| `verifier/` | {non_tcb['verifier']} | client-side checking, "
              "imports nothing from the runtime |",
              f"| `harness/` | {non_tcb['harness']} | data generation, "
              "attacks, benchmarks |", "",
              f"The {total}-line TCB is small enough to audit directly, in "
              "contrast to the millions of lines in a PyTorch/CUDA training "
              "stack or the circuit + trusted-setup TCB of SNARK-based "
              "verifiable unlearning. Input confinement is additionally "
              "checked mechanically by an AST audit (tests/test_confinement.py).",
              ""]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "tcb_loc.md").write_text("\n".join(lines))
    print(f"TCB total {total} SLOC "
          f"(kernel non-semantic {dict(( (n,s) for n,s,_ in rows)).get('kernel.py')})")
    for name, n, _ in rows:
        print(f"  {n:4d}  {name}")
    print(f"non-TCB: verifier {non_tcb['verifier']}, "
          f"harness {non_tcb['harness']} -> {out / 'tcb_loc.md'}")


if __name__ == "__main__":
    main()
