# E3b - Attack x ablation cross product

Each column is the system with one defense removed. A cell says whether the adversary obtains a certificate the auditor accepts. `--` means the attack was refused before any certificate existed.

| attack | full | -NoNonce | -NoDataBind | -NoDeltaBind | -NoDataBind -NoDeltaBind -NoRecord | -NoCPCert | -NoClean | -NoConfigBind | -NoRecord | -NoReverify |
|---|---|---|---|---|---|---|---|---|---|---|
| Retraining Skipping | reject | reject | reject | reject | reject | reject | reject | reject | reject | reject |
| Certificate Replay | reject | **accept** | reject | reject | reject | reject | reject | reject | reject | reject |
| Deletion-Set Substitution | reject | reject | reject | reject | **accept** | reject | reject | reject | reject | reject |
| Checkpoint Substitution | reject | reject | reject | reject | reject | **accept** | reject | reject | reject | reject |
| Unclean Checkpoint | reject | reject | reject | reject | reject | reject | **accept** | reject | reject | reject |
| Deletion Resurrection | reject | reject | reject | reject | **accept** | reject | reject | reject | **accept** | reject |
| Dataset Tampering | reject | reject | **accept** | reject | **accept** | reject | reject | reject | reject | reject |
| Randomness Manipulation | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Configuration Steering | reject | reject | reject | reject | reject | reject | reject | **accept** | reject | reject |
| Mid-Invocation Rollback | -- | -- | -- | -- | -- | -- | -- | -- | -- | **accept** |
| Transfer Tampering | -- | -- | -- | -- | -- | -- | -- | -- | -- | **accept** |

Attacks: 11. Configurations: 10. Cells accepting: 11 of 110.
No attack succeeds against the full system: True.
