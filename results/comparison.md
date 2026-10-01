# E5 - Verification-Mechanism Comparison

This is the one table in `results/` the harness does not measure. The CLEAN row
is measured here; every other row restates what the cited work reports about
itself, at the scale that work reports. `NOTES-baselines.md` records which
figure came from which paper.

**Certifies** names the evidence object. **Trusted for correctness** names what
must be right for that evidence to mean erasure.

| mechanism | certifies | trusted for correctness | prove | verify |
|---|---|---|---|---|
| MIA auditing | statistical forgetting | the audit itself | shadow models | minutes |
| PoUL (Weng et al., TIFS 2024) | prescribed procedure ran | the enclaved procedure | --- | constant, unmeasured |
| SNARK (Eisenhofer et al., SaTML 2025) | prescribed update ran | the prescribed algorithm | 5 min | 108 s |
| zkPoT (Kaizen, CCS 2024) | gradient descent ran | the prescribed algorithm | 882 s, 466 GB | 130 ms |
| **CLEAN (this work)** | **deleted records were not an input** | **the mediation layer (521 lines)** | **85 us** | **0.34 ms** |

Every row assumes a malicious provider except MIA auditing, whose verdicts are
stated against an honest-but-curious one.

The rows are not measured on a common workload, so the scales matter. The CLEAN
figures are measurements at a 128-record manifest (`cert_cost.md` for the prove
column, `verification_cost.md` for the verify column). SNARK is on 100 data
points, and a proof-system variant of it trades verification down to under a
second for 34 min of proving plus a one-time 8 min of client preprocessing.
zkPoT is per gradient-descent iteration on a 10.1M-parameter model. PoUL's
constant verification cost is a design property for which no latency is
reported.

## What the comparison is about

The third column is the one that carries the argument. The cryptographic route
proves that a prescribed computation ran, so the prescribed algorithm has to be
the right one. The enclave route as PoUL uses it proves the same thing inside a
TEE, so the enclaved procedure has to implement unlearning correctly. CLEAN
certifies an information-flow property of the execution instead, which holds
whatever the payload computes, and leaves the payload's correctness outside the
security argument. What stays trusted is the mediation layer, and `tcb_loc.md`
measures it.

The cost columns follow from that. Evidence generation is a hash and a signature
over commitments, independent of parameter count (`commitment_cost.md`,
`cert_cost.md`). The audit reads no model parameter, so its cost follows the
length of the dataset manifest: `verification_cost.md` at prototype scale,
`scale_verify.md` out to 10^8 records.

## Sources

- Weng et al., Proof of Unlearning: Definitions and Instantiation, IEEE TIFS 2024.
- Eisenhofer et al., Verifiable and Provably Secure Machine Unlearning, IEEE SaTML 2025.
- Abbaszadeh et al., Zero-Knowledge Proofs of Training for Deep Neural Networks (Kaizen), ACM CCS 2024.
- Hayes et al., Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy, IEEE SaTML 2025.
- Zhang et al., Verification of Machine Unlearning is Fragile, ICML 2024.
