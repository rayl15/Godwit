#!/usr/bin/env bash
# Kernel comparison that survives a fanless machine.
#
# Each configuration runs in its own process with a cooldown, and the order is
# rotated every round. Running them together made position in the run worth 2x
# — larger than any real difference between the kernels.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.."

BIN=.build/release/godwit
ROUNDS=${ROUNDS:-4}
COOLDOWN=${COOLDOWN:-8}
VARIANTS=(naive persistent staged)

for shape in "5760 2880" "2880 2880"; do
  set -- $shape
  echo "=== ${1}x${2} ==="
  for round in $(seq 1 "$ROUNDS"); do
    # Rotate the order so no variant keeps a favourable slot.
    offset=$(( (round - 1) % ${#VARIANTS[@]} ))
    for i in $(seq 0 $(( ${#VARIANTS[@]} - 1 )) ); do
      v=${VARIANTS[$(( (i + offset) % ${#VARIANTS[@]} ))]}
      sleep "$COOLDOWN"
      printf "r%d " "$round"
      $BIN bench dequant --only "$v" --rows "$1" --cols "$2"
    done
  done
  echo
done
