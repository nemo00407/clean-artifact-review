# E1 - Non-interference (two-world equality), swept

Each configuration runs the invocation twice: once on the real shard with the gateway excluding Delta, once on a counterfactual shard in which those records were never written. Same certified checkpoint, same attested randomness, same configuration. Equality is bitwise over the canonical adapter serialization.

**56 of 56** configurations equal; **56 of 56** issued a certificate the verifier accepts (Theorem 1).

| configuration | shard | slice | records | equal | verifies |
|---|---:|---|---:|---|---|
| s0 slice1 n1 | 0 | 1 | 1 | yes | yes |
| s0 slice1 n2 | 0 | 1 | 2 | yes | yes |
| s0 slice1 n4 | 0 | 1 | 4 | yes | yes |
| s0 slice1 n8 | 0 | 1 | 8 | yes | yes |
| s0 slice2 n1 | 0 | 2 | 1 | yes | yes |
| s0 slice2 n2 | 0 | 2 | 2 | yes | yes |
| s0 slice2 n4 | 0 | 2 | 4 | yes | yes |
| s0 slice2 n8 | 0 | 2 | 8 | yes | yes |
| s0 slice3 n1 | 0 | 3 | 1 | yes | yes |
| s0 slice3 n2 | 0 | 3 | 2 | yes | yes |
| s0 slice3 n4 | 0 | 3 | 4 | yes | yes |
| s0 slice3 n8 | 0 | 3 | 8 | yes | yes |
| s1 slice1 n1 | 1 | 1 | 1 | yes | yes |
| s1 slice1 n2 | 1 | 1 | 2 | yes | yes |
| s1 slice1 n4 | 1 | 1 | 4 | yes | yes |
| s1 slice1 n8 | 1 | 1 | 8 | yes | yes |
| s1 slice2 n1 | 1 | 2 | 1 | yes | yes |
| s1 slice2 n2 | 1 | 2 | 2 | yes | yes |
| s1 slice2 n4 | 1 | 2 | 4 | yes | yes |
| s1 slice2 n8 | 1 | 2 | 8 | yes | yes |
| s1 slice3 n1 | 1 | 3 | 1 | yes | yes |
| s1 slice3 n2 | 1 | 3 | 2 | yes | yes |
| s1 slice3 n4 | 1 | 3 | 4 | yes | yes |
| s1 slice3 n8 | 1 | 3 | 8 | yes | yes |
| s2 slice1 n1 | 2 | 1 | 1 | yes | yes |
| s2 slice1 n2 | 2 | 1 | 2 | yes | yes |
| s2 slice1 n4 | 2 | 1 | 4 | yes | yes |
| s2 slice1 n8 | 2 | 1 | 8 | yes | yes |
| s2 slice2 n1 | 2 | 2 | 1 | yes | yes |
| s2 slice2 n2 | 2 | 2 | 2 | yes | yes |
| s2 slice2 n4 | 2 | 2 | 4 | yes | yes |
| s2 slice2 n8 | 2 | 2 | 8 | yes | yes |
| s2 slice3 n1 | 2 | 3 | 1 | yes | yes |
| s2 slice3 n2 | 2 | 3 | 2 | yes | yes |
| s2 slice3 n4 | 2 | 3 | 4 | yes | yes |
| s2 slice3 n8 | 2 | 3 | 8 | yes | yes |
| s3 slice1 n1 | 3 | 1 | 1 | yes | yes |
| s3 slice1 n2 | 3 | 1 | 2 | yes | yes |
| s3 slice1 n4 | 3 | 1 | 4 | yes | yes |
| s3 slice1 n8 | 3 | 1 | 8 | yes | yes |
| s3 slice2 n1 | 3 | 2 | 1 | yes | yes |
| s3 slice2 n2 | 3 | 2 | 2 | yes | yes |
| s3 slice2 n4 | 3 | 2 | 4 | yes | yes |
| s3 slice2 n8 | 3 | 2 | 8 | yes | yes |
| s3 slice3 n1 | 3 | 3 | 1 | yes | yes |
| s3 slice3 n2 | 3 | 3 | 2 | yes | yes |
| s3 slice3 n4 | 3 | 3 | 4 | yes | yes |
| s3 slice3 n8 | 3 | 3 | 8 | yes | yes |
| s0 slices2-3 n2 | 0 | 2-3 | 2 | yes | yes |
| s0 slices2-3 n6 | 0 | 2-3 | 6 | yes | yes |
| s1 slices2-3 n2 | 1 | 2-3 | 2 | yes | yes |
| s1 slices2-3 n6 | 1 | 2-3 | 6 | yes | yes |
| s2 slices2-3 n2 | 2 | 2-3 | 2 | yes | yes |
| s2 slices2-3 n6 | 2 | 2-3 | 6 | yes | yes |
| s3 slices2-3 n2 | 3 | 2-3 | 2 | yes | yes |
| s3 slices2-3 n6 | 3 | 2-3 | 6 | yes | yes |

Determinism: 20 repeats of one invocation produced 1 distinct adapter hash(es) and 1 distinct signature(s).

Validity ablations (the criterion must be able to fail):

- `CLEAN-NoExclusion`: differs from the counterfactual world: **True**
- `CLEAN-NoCleanCP`: differs from the counterfactual world: **True**
