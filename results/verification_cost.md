# E11 - Verification cost

Certificates signed by the enclave over synthetic manifests. The verifier runs its whole path in every measurement. It never reads the model, so no series here varies with parameter count.

| manifest (records) | median (ms) | p99 (ms) |
|---:|---:|---:|
| 128 | 0.329 | 0.386 |
| 512 | 0.715 | 0.757 |
| 2048 | 2.266 | 2.416 |
| 8192 | 8.849 | 9.395 |
| 32768 | 36.791 | 39.962 |

| deletion set (records) | median (ms) | p99 (ms) |
|---:|---:|---:|
| 1 | 38.586 | 40.019 |
| 8 | 36.540 | 40.129 |
| 64 | 36.636 | 37.573 |
| 512 | 37.137 | 40.603 |

Manifest length spans 128 to 32768 records, a factor of 256, for a cost ratio of 111.9.
