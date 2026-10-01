"""Regenerate the full evaluation result set (Stage T5).

Runs the SISA initialization then every experiment, writing results/*.md.
Intended entry point for reproducing the numbers cited in the paper (P6).
"""
import subprocess
import sys

STEPS = [
    ("SISA init", "harness.sisa_train"),
    ("E1 non-interference sweep (+ determinism, completeness)",
     "harness.two_world"),
    ("E3 attack interception", "harness.attacks"),
    ("E3b attack x ablation cross product", "harness.attack_matrix"),
    ("E9 refusal point of post-admission substitution",
     "harness.fetch_probe"),
    ("E10 mutation testing of the confinement audit", "harness.mutation"),
    ("E13 compound erasure across invocations", "harness.compound"),
    ("E11/E12 verification and temporal-closure cost",
     "harness.cost_sweeps"),
    ("E7/E8 commitment and rollback benchmarks", "harness.bench"),
    ("E17 certificate production, in the enclave and outside",
     "harness.cert_cost"),
    # E14/E15 sweep to 10^8 records and can take ten minutes; they are pure
    # CPU and independent of the rest, so they run last.
    ("E14/E15 audit cost at corpus scale", "harness.scale_sweeps"),
]


def main():
    import os
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    for label, mod in STEPS:
        print(f"\n=== {label} ({mod}) ===", flush=True)
        subprocess.run([sys.executable, "-m", mod], check=True, env=env)
    # E4 (TCB) and E5 (comparison) are source-derived / literature tables
    subprocess.run([sys.executable, "tools/loc_report.py"], check=True, env=env)
    print("\nresults/ regenerated (E5 comparison.md is literature-sourced, "
          "edited by hand).")


if __name__ == "__main__":
    main()
