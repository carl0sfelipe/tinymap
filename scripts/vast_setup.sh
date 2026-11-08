#!/usr/bin/env bash
# Prepara uma máquina nova da vast.ai (imagem pytorch/pytorch com CUDA) para o tinymap.
# Uso, dentro da máquina:
#   git clone -b estudio https://github.com/carl0sfelipe/tinymap && cd tinymap && bash scripts/vast_setup.sh
# Depois: preencher .env e rodar `tinymap doctor regions/belo-horizonte.yaml`.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== sistema"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libvips-dev git curl ca-certificates xvfb >/dev/null
if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi

echo "== python"
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[lab,bridge]"

echo "== render (Node + Chromium headless)"
(cd render && npm install --no-audit --no-fund && npx playwright install --with-deps chromium)

echo "== ai-toolkit (treino da LoRA)"
if [ ! -d ../ai-toolkit ]; then
  git clone -q https://github.com/ostris/ai-toolkit ../ai-toolkit
  (cd ../ai-toolkit && git submodule update --init --recursive -q && python -m pip install -q -r requirements.txt)
fi

echo "== caches no disco grande (modelos ~60 GB)"
mkdir -p /workspace/hf
grep -q '^HF_HOME=' .env 2>/dev/null || echo "HF_HOME=/workspace/hf" >> .env
[ -f .env ] || cp .env.example .env

echo
echo "Pronto. Preencha o .env (GOOGLE_TILES_API_KEY ou CESIUM_ION_TOKEN, HF_TOKEN) e rode:"
echo "  tinymap doctor regions/belo-horizonte.yaml"
