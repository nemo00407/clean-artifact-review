# E12 - Temporal-closure overhead

The admitted dataset is split into blocks; one commitment per block stays in the enclave and is re-derived on every fetch. The reference invocation admits **63** rows and performs **20** fetches in **2106 ms** of retraining. Per-fetch cost is measured directly, because the check is microseconds against seconds and does not surface above run-to-run variance end to end.

| block (rows) | bytes | commitments resident | per fetch (us) | per run (ms) | share of retraining |
|---:|---:|---:|---:|---:|---:|
| 4 | 275 | 16 | 1.04 | 0.167 | 0.0079% |
| 8 | 531 | 8 | 1.46 | 0.117 | 0.0056% |
| 16 | 1043 | 4 | 2.31 | 0.093 | 0.0044% |
| 32 | 2067 | 2 | 4.05 | 0.081 | 0.0038% |

Smaller blocks keep more commitments resident and re-derive more often; larger blocks hash more bytes per fetch. The guarantee is identical at every point on the curve, so the block size is a memory/throughput choice and not a security one.
