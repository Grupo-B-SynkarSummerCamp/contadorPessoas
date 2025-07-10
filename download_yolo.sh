#!/usr/bin/env bash

set -euo pipefail

# URL oficial do peso YOLOv8n (pré-treinado)
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

# Diretório de destino: ./models (relativo ao local onde o script é executado)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
OUTPUT_PATH="${MODELS_DIR}/yolov8n.pt"

# Cria a pasta models se não existir
mkdir -p "$MODELS_DIR"

echo "Baixando yolov8n.pt para $OUTPUT_PATH..."
# Faz o download seguindo redirecionamentos
curl -L "$MODEL_URL" -o "$OUTPUT_PATH"

echo "Download concluído!"
ls -lh "$OUTPUT_PATH"
