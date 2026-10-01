# E9 - Refusal point of post-admission substitution

An unattacked invocation performs **20** block fetches. The host substitutes from fetch *k*+1 onward; the table gives the fetch on which the invocation aborted. No certificate was issued in any row.

| k | storage: substituted at | aborted at | transfer: substituted at | aborted at |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 2 | 2 |
| 2 | 3 | 3 | 3 | 3 |
| 3 | 4 | 4 | 4 | 4 |
| 4 | 5 | 5 | 5 | 5 |
| 5 | 6 | 6 | 6 | 6 |
| 6 | 7 | 7 | 7 | 7 |
| 7 | 8 | 8 | 8 | 8 |
| 8 | 10 | 10 | 9 | 9 |
| 9 | 10 | 10 | 10 | 10 |
| 10 | 13 | 13 | 11 | 11 |
| 11 | 13 | 13 | 12 | 12 |
| 12 | 13 | 13 | 13 | 13 |

Aborted on the fetch carrying the substitution in every case: True. Substituted blocks consumed by the trainer: 0.

The commitment is re-derived per fetch, so refusal lands on the carrying fetch regardless of how far into the run the substitution happens. The storage adversary targets one block key, so its substitution first lands on the next fetch of that block rather than on fetch k+1; the transfer adversary corrupts whatever crosses next.
