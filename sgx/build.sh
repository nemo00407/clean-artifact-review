#!/usr/bin/env bash
# Build and run the CLEAN enclave probe. Everything happens in the container;
# the host contributes only /dev/sgx_enclave and the directories mounted below.
set -euo pipefail
cd "$(dirname "$0")"
APP="${APP:-/app/probe.py}"
APPARGS="${APPARGS:-}"
SIZE="${1:-1G}"
RA="${2:-none}"
mkdir -p build sealed
docker run --rm --device /dev/sgx_enclave -e APP="$APP" -e APPARGS="$APPARGS" \
  -v "$PWD/app":/app:ro -v "$PWD/build":/build -v "$PWD/sealed":/sealed \
  -v "$PWD/probe.manifest.template":/probe.manifest.template:ro \
  -v "$PWD/enclave-key.pem":/key.pem:ro \
  clean-sgx:1 "cd /build && gramine-manifest \
      -Dlog_level=error -Dentrypoint=/usr/bin/python3.10 \
      -Darch_libdir=/lib/x86_64-linux-gnu \
      -Denclave_size=$SIZE -Dra_type=$RA \
      /probe.manifest.template probe.manifest \
    && gramine-sgx-sign --key /key.pem --manifest probe.manifest \
      --output probe.manifest.sgx >/dev/null \
    && gramine-sgx probe ${APP:-/app/probe.py} ${APPARGS:-}"
