# Result files

Every number the paper reports is produced by a module under `harness/` and
written here. `results/` holds the host runs; `results-sgx/` holds the runs
performed inside the SGX enclave, kept separate because a deployment is bound
to the image that produced it (the certificates name a measurement, and the
enclave's measurement is not the host's).

These files are a run of the code as released here, on the platform
`NOTES-hardware.md` records. Verdicts are reproducible: every equality, every
accept and every reject comes out the same on a re-run. Timings are medians over
repetitions and move by a few percent between runs, and further under load, so
the figures quoted in the paper and the figures here agree to within that
variation rather than exactly.

Regenerate everything with `python -m harness.run_all` from the repository
root. Individual experiments are listed below; the top-level `README.md`
explains the environment and gives the per-experiment commands.

| file | paper label | what it records |
|---|---|---|
| `noninterference.md` | E1 | two-world bitwise equality, swept over shard, slice and deletion size, plus the two controls with one mechanism disabled |
| `attacks.md` | E3 | the individual attacks and the ablation that re-exposes each |
| `attack_matrix.md` | E3b | the attack × ablation cross product behind the evaluation's matrix figure |
| `tcb_loc.md` | E4 | trusted-computing-base line counts per runtime module |
| `comparison.md` | E5 | comparison against SNARK, zkPoT, PoUL and MIA auditing; the figures for the other systems come from their own papers, recorded in `NOTES-baselines.md` |
| `commitment_cost.md` | E7 | composite commitment against hashing the whole model |
| `rollback.md` | E8 | retraining cost against the position of the deletion |
| `abort_point.md` | E9 | where a post-admission substitution is refused on the read path |
| `mutation.md` | E10 | mutation testing of the confinement audit: which planted escapes it catches |
| `verification_cost.md` | E11 | verification cost against manifest length |
| `temporal_overhead.md` | E12 | cost of re-deriving a block commitment on every fetch |
| `compound.md` | E13 | two deletions in sequence on one shard, and four attempts to restore the first |
| `scale_verify.md` | E14 | audit cost at corpus scale, to 10^8 manifest records |
| `scale_prefix.md` | E15 | clean-prefix test against prefix length |
| `two_world_1b.md` | E16 | two-world bitwise equality at 1.1B parameters |
| `cert_cost.md` | E17 | certificate production, inside the enclave and outside |
| `retrain_1b.md` | E18 | retraining cost against deletion position at 1.1B parameters |

`NOTES-hardware.md` records the platform every measurement was taken on.

The `.dat` files are the same data in the column form the paper's plots read.
The `.json` and `.meta.json` files carry the raw per-run values behind each
table, including the ones the prose quotes but the tables do not list.

`comparison.md` is the one table that is not measured here: it restates what
the compared systems report about themselves, and `NOTES-baselines.md` says
which claim came from where.
