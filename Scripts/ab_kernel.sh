#!/usr/bin/env bash
# A/B two kernels on a thermally unstable machine.
#
# Absolute throughput here drifts by up to 2x depending on how hot the GPU is,
# which is larger than any difference we are trying to detect. So this measures
# a RATIO instead: each round runs both variants back to back, and thermal state
# — whatever it is at that moment — divides out.
#
# Two further controls:
#   - the order flips every round, so neither variant keeps the cooler slot;
#   - each measurement is its own process, since sharing one heats the later
#     configuration and that alone was worth 2x.
#
# Reports the per-round ratio, the median, and a sign test. With 12 rounds,
# 12/12 in one direction is p ≈ 0.0005; 8/12 is not evidence of anything.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.."

A=${A:-naive}
B=${B:-persistent}
ROWS=${ROWS:-5760}
COLS=${COLS:-2880}
ROUNDS=${ROUNDS:-12}
COOLDOWN=${COOLDOWN:-6}
BIN=.build/release/godwit

run() { $BIN bench dequant --only "$1" --rows "$ROWS" --cols "$COLS" \
        | awk '{for(i=1;i<=NF;i++) if($i=="G") print $(i-1)}'; }

echo "A=$A  B=$B  shape ${ROWS}x${COLS}  $ROUNDS rounds"
wins=0
ratios=()
for r in $(seq 1 "$ROUNDS"); do
  sleep "$COOLDOWN"
  if (( r % 2 == 1 )); then a=$(run "$A"); sleep "$COOLDOWN"; b=$(run "$B")
  else b=$(run "$B"); sleep "$COOLDOWN"; a=$(run "$A"); fi
  ratio=$(echo "scale=4; $b/$a" | bc)
  ratios+=("$ratio")
  (( $(echo "$ratio > 1" | bc) )) && wins=$((wins+1))
  printf "  r%-2d  %s %6s   %s %6s   ratio %s\n" "$r" "$A" "$a" "$B" "$b" "$ratio"
done

printf '%s\n' "${ratios[@]}" | sort -n | awk -v n="$ROUNDS" -v w="$wins" '
  {v[NR]=$1}
  END{ m=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
       printf "\nmedian ratio B/A: %.3f   B wins %d/%d\n", m, w, n }'
