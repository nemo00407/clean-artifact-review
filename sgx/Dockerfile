# CLEAN's control plane under Gramine-SGX.
#
# The base image carries Gramine 1.9 and the SGX PSW client libraries. We add
# only what clean_runtime imports: numpy for the array handles, cryptography
# for Ed25519. Nothing here is part of the TCB measurement argument -- the
# enclave measurement covers the files the manifest marks as trusted, and the
# image is just how they get onto the machine.
FROM gramineproject/gramine:stable-jammy

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-numpy python3-cryptography \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
