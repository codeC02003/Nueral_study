#!/bin/bash
# Frozen evaluation of A2. Identical to what A1 received - no parameter differs.
cd "$(dirname "$0")"
for L in 0 1 2 3; do
  python3 validate_layer.py --model a2 --layer $L --sae-steps 40000 --seeds 3
done
