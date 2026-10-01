# E15 - Clean-prefix test against prefix length

The gateway walks the checkpoint's trained prefix, checking each record against the deletions recorded since that checkpoint was certified. Entries name records by hash so that they survive compaction, which is also why the prefix cannot be indexed into and must be walked.

| prefix (records) | time | peak RSS |
|---:|---:|---:|
| 10,000 | 0.4 ms | 0.03 GB |
| 100,000 | 4.0 ms | 0.03 GB |
| 1,000,000 | 39.1 ms | 0.11 GB |
| 10,000,000 | 390.3 ms | 0.91 GB |

The walk costs about 39\,ns per record -- a set membership test, with no hashing, because the manifest already carries the digests. At the largest prefix measured it is 390\,ms against seconds of retraining, so it does not dominate an invocation; what it does cost is holding the prefix, 0.9\,GB at 10,000,000 records.

The prefix is the part of the shard the invocation does not retrain, so this cost runs opposite to the saving: a deletion late in a shard retrains least and walks most.
