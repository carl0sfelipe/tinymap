#!/usr/bin/env bash
# One generate worker per GPU. Usage: scripts/gpus.sh <region.yaml> <lora.safetensors> [extra tinymap args]
# Workers share the run folder through lock files; Ctrl-C stops all, re-run to resume.
set -euo pipefail
REGION=$1; LORA=$2; shift 2
N=$(nvidia-smi -L | wc -l)
mkdir -p logs
trap 'kill 0' INT TERM
for ((i=0; i<N; i++)); do
  CUDA_VISIBLE_DEVICES=$i tinymap generate "$REGION" --lora "$LORA" --gpu 0 "$@" > "logs/gpu$i.log" 2>&1 &
  echo "GPU $i -> logs/gpu$i.log"
  sleep 20   # stagger model loading so disk and RAM are not hit all at once
done
wait
