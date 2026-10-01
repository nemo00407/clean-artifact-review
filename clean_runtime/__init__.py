"""CLEAN input-closed unlearning runtime (TCB).

Modules in this package form the trusted computing base of the CLEAN
prototype. The runtime is trusted only for *input confinement*: every piece
of information reaching the trainer enters through the Secure Input Gateway
(`gateway.py`), and the deleted set never enters the input domain. Numerical
behavior of the training kernel carries no security obligation
(non-semantic TCB).

Import policy (enforced by tests/test_confinement.py):
  - allowed everywhere: numpy, hashlib, hmac, struct, json, intra-package
  - gateway.py additionally: file I/O (it is the sole mediation point)
  - attest.py additionally: cryptography (Ed25519 signing)
"""
