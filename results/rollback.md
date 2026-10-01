# E8 - SISA Clean-Rollback Benefit

Deleting from a later slice rolls back to a later clean checkpoint and retrains fewer slices. Each row is the median over 4 shards.

| deletion slice | slices retrained | median (s) | min | max |
|---:|---:|---:|---:|---:|
| 0 | 4 | 4.27 | 4.25 | 4.36 |
| 1 | 3 | 3.19 | 3.19 | 3.20 |
| 2 | 2 | 2.12 | 2.12 | 2.12 |
| 3 | 1 | 1.05 | 1.04 | 1.09 |

From-scratch (roll back to CP_0, retrain all 4 slices): 4.26s median. SISA localizes the retraining region to the slices after the deletion point.
