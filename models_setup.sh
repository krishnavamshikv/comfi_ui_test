#!/bin/bash
###############################################################################
# setup_models.sh
#
# Downloads every model needed for the ComfyUI "AI influencer" Phase 1
# pipeline (InstantID + IPAdapter FaceID + ControlNet OpenPose/Depth +
# ReActor face swap) into the correct ComfyUI model folders.
#
# USAGE:
#   bash setup_models.sh
#
# Safe to re-run: already-downloaded files with the correct size are
# skipped. Every run writes a timestamped log to the logs/ folder next
# to this script, plus keeps updating logs/latest.log.
###############################################################################

set -uo pipefail  # NOTE: no -e on purpose — one failed download must not kill the rest

# ---------------------------------------------------------------------------
# CONFIG — edit this if your ComfyUI path ever changes
# ---------------------------------------------------------------------------
COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/setup_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/latest.log"

# Send everything to both the terminal and the log file
exec > >(tee -a "$LOG_FILE" "$LATEST_LOG") 2>&1

FAILED_ITEMS=()
SKIPPED_ITEMS=()
OK_ITEMS=()

echo "==============================================================="
echo " ComfyUI Model Setup — started at $(date)"
echo " ComfyUI dir: $COMFYUI_DIR"
echo " Log file:    $LOG_FILE"
echo "==============================================================="

if [ ! -d "$COMFYUI_DIR" ]; then
  echo "[FATAL] ComfyUI directory not found at $COMFYUI_DIR"
  echo "        Edit the COMFYUI_DIR variable at the top of this script and re-run."
  exit 1
fi

# ---------------------------------------------------------------------------
# Helper function: download_model <url> <dest_path> <min_size_bytes> <label>
# ---------------------------------------------------------------------------
download_model () {
  local url="$1"
  local dest="$2"
  local min_size="$3"
  local label="$4"

  mkdir -p "$(dirname "$dest")"

  # Skip if the file already exists and looks complete (size check)
  if [ -f "$dest" ]; then
    local existing_size
    existing_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$existing_size" -ge "$min_size" ]; then
      echo "[SKIP] $label already present and looks complete ($(numfmt --to=iec "$existing_size" 2>/dev/null || echo "$existing_size bytes")) -> $dest"
      SKIPPED_ITEMS+=("$label")
      return 0
    else
      echo "[INFO] $label exists but looks incomplete (${existing_size} bytes) — re-downloading"
      rm -f "$dest"
    fi
  fi

  echo "[GET]  $label"
  echo "       URL:  $url"
  echo "       Dest: $dest"

  # -c resumes partial downloads, retries handle flaky connections,
  # --content-disposition follows HF's real filename if it differs
  wget -c --tries=5 --timeout=60 --content-disposition \
       -O "$dest" "$url"

  if [ $? -ne 0 ]; then
    echo "[FAIL] wget reported an error for $label"
    FAILED_ITEMS+=("$label")
    return 1
  fi

  local final_size
  final_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)

  if [ "$final_size" -lt "$min_size" ]; then
    echo "[FAIL] $label downloaded but is too small (${final_size} bytes, expected >= ${min_size}). Likely a broken/redirected download."
    FAILED_ITEMS+=("$label")
    return 1
  fi

  echo "[OK]   $label — $(numfmt --to=iec "$final_size" 2>/dev/null || echo "$final_size bytes")"
  OK_ITEMS+=("$label")
  return 0
}

# ---------------------------------------------------------------------------
# 1. Base checkpoint — RealVisXL V5.0 (SDXL photorealistic)
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors?download=true" \
  "${MODELS_DIR}/checkpoints/RealVisXL_V5.0_fp16.safetensors" \
  6000000000 \
  "RealVisXL V5.0 checkpoint"

# ---------------------------------------------------------------------------
# 2. InstantID
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true" \
  "${MODELS_DIR}/instantid/ip-adapter.bin" \
  1500000000 \
  "InstantID ip-adapter model"

download_model \
  "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/InstantID_ControlNet.safetensors" \
  2000000000 \
  "InstantID ControlNet model"

# antelopev2 face analysis (zip -> extracted with python, no unzip needed)
ANTELOPE_DIR="${MODELS_DIR}/insightface/models/antelopev2"
ANTELOPE_ZIP="${MODELS_DIR}/insightface/models/antelopev2.zip"
if [ -f "${ANTELOPE_DIR}/glintr100.onnx" ]; then
  echo "[SKIP] antelopev2 face models already present -> $ANTELOPE_DIR"
  SKIPPED_ITEMS+=("antelopev2 face models")
else
  mkdir -p "$ANTELOPE_DIR"
  echo "[GET]  antelopev2 face analysis models"
  wget -c --tries=5 --timeout=60 -O "$ANTELOPE_ZIP" \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip?download=true"
  if [ $? -eq 0 ] && [ -f "$ANTELOPE_ZIP" ]; then
    python3 -c "import zipfile; zipfile.ZipFile('${ANTELOPE_ZIP}').extractall('${MODELS_DIR}/insightface/models')"
    # handle possible nested folder from the zip
    if [ -d "${ANTELOPE_DIR}/antelopev2" ]; then
      mv "${ANTELOPE_DIR}/antelopev2"/* "${ANTELOPE_DIR}/"
      rmdir "${ANTELOPE_DIR}/antelopev2"
    fi
    rm -f "$ANTELOPE_ZIP"
    if [ -f "${ANTELOPE_DIR}/glintr100.onnx" ]; then
      echo "[OK]   antelopev2 face models extracted"
      OK_ITEMS+=("antelopev2 face models")
    else
      echo "[FAIL] antelopev2 extraction didn't produce expected files — check $ANTELOPE_DIR manually"
      FAILED_ITEMS+=("antelopev2 face models")
    fi
  else
    echo "[FAIL] antelopev2 download failed"
    FAILED_ITEMS+=("antelopev2 face models")
  fi
fi

# ---------------------------------------------------------------------------
# 3. IPAdapter FaceID Plus v2 (SDXL) + matching LoRA + CLIP vision encoder
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin?download=true" \
  "${MODELS_DIR}/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin" \
  800000000 \
  "IPAdapter FaceID Plus v2 (SDXL)"

download_model \
  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors?download=true" \
  "${MODELS_DIR}/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors" \
  50000000 \
  "IPAdapter FaceID Plus v2 LoRA (SDXL)"

download_model \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true" \
  "${MODELS_DIR}/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" \
  2000000000 \
  "CLIP Vision encoder (for IPAdapter)"

# ---------------------------------------------------------------------------
# 4. ControlNet — OpenPose (SDXL) + Depth (SDXL)
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/OpenPoseXL2.safetensors" \
  4000000000 \
  "ControlNet OpenPose (SDXL)"

download_model \
  "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/controlnet-depth-sdxl-1.0_fp16.safetensors" \
  2000000000 \
  "ControlNet Depth (SDXL, fp16)"

# ---------------------------------------------------------------------------
# 5. ReActor — face swap model + face restoration model
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx?download=true" \
  "${MODELS_DIR}/insightface/inswapper_128.onnx" \
  500000000 \
  "ReActor inswapper_128 face-swap model"

download_model \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
  "${MODELS_DIR}/facerestore_models/GFPGANv1.4.pth" \
  300000000 \
  "GFPGAN v1.4 face restoration model"

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo ""
echo "==============================================================="
echo " SUMMARY — finished at $(date)"
echo "==============================================================="
echo "OK (${#OK_ITEMS[@]}):"
for i in "${OK_ITEMS[@]:-}"; do [ -n "$i" ] && echo "   - $i"; done
echo ""
echo "SKIPPED / already present (${#SKIPPED_ITEMS[@]}):"
for i in "${SKIPPED_ITEMS[@]:-}"; do [ -n "$i" ] && echo "   - $i"; done
echo ""
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "FAILED (${#FAILED_ITEMS[@]}):"
  for i in "${FAILED_ITEMS[@]}"; do echo "   - $i"; done
  echo ""
  echo "Re-run this script to retry only the failed/incomplete files."
  echo "Full log saved at: $LOG_FILE"
  exit 1
else
  echo "All models downloaded/verified successfully."
  echo "Full log saved at: $LOG_FILE"
  exit 0
fi