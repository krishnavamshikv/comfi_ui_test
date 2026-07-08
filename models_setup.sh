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
# skipped. Every run writes a clean, readable log to logs/latest.log
# (and a timestamped copy) — no raw wget progress spam.
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

# Everything printed with `log` goes to screen AND to both log files.
# (We do NOT pipe wget's own raw output through this — that's the part
# that looked ugly — we generate our own clean lines instead.)
log () {
  echo -e "$1" | tee -a "$LOG_FILE" "$LATEST_LOG"
}

FAILED_ITEMS=()
SKIPPED_ITEMS=()
OK_ITEMS=()

TOTAL_ITEMS=11
CURRENT_ITEM=0

# ---------------------------------------------------------------------------
# Speed check: wget uses a single connection, which Hugging Face throttles
# per-stream (~10-15MB/s) regardless of your pod's real bandwidth.
# aria2c splits each download into parallel connections and is MUCH faster
# on high-bandwidth boxes like RunPod. We auto-install it if missing.
# ---------------------------------------------------------------------------
USE_ARIA2=0
if command -v aria2c >/dev/null 2>&1; then
  USE_ARIA2=1
else
  log "[SETUP] aria2c not found — installing it for much faster parallel downloads..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y >/dev/null 2>&1 && apt-get install -y aria2 >/dev/null 2>&1
  fi
  if command -v aria2c >/dev/null 2>&1; then
    USE_ARIA2=1
    log "[SETUP] aria2c installed successfully — using parallel downloads."
  else
    log "[SETUP] Could not install aria2c (no apt access) — falling back to single-connection wget (slower)."
  fi
fi

human_size () {
  # Pretty-print bytes as e.g. 6.9G / 350M
  numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1} bytes"
}

log "==============================================================="
log " ComfyUI Model Setup — started at $(date)"
log " ComfyUI dir: $COMFYUI_DIR"
log " Log file:    $LOG_FILE"
log "==============================================================="

if [ ! -d "$COMFYUI_DIR" ]; then
  log "[FATAL] ComfyUI directory not found at $COMFYUI_DIR"
  log "        Edit the COMFYUI_DIR variable at the top of this script and re-run."
  exit 1
fi

# ---------------------------------------------------------------------------
# download_model <url> <dest_path> <expected_size_bytes> <label>
#
# Runs wget quietly in the background, then polls the partial file's size
# every 5 seconds and prints ONE clean status line per check — instead of
# wget's dot-matrix spam. Also prints an overall [n/11] counter.
# ---------------------------------------------------------------------------
download_model () {
  local url="$1"
  local dest="$2"
  local expected_size="$3"
  local label="$4"

  CURRENT_ITEM=$((CURRENT_ITEM + 1))
  local tag="[${CURRENT_ITEM}/${TOTAL_ITEMS}]"

  mkdir -p "$(dirname "$dest")"

  # Skip if already downloaded and looks complete
  if [ -f "$dest" ]; then
    local existing_size
    existing_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$existing_size" -ge "$expected_size" ]; then
      log "$tag SKIP  ${label} — already downloaded ($(human_size "$existing_size"))"
      SKIPPED_ITEMS+=("$label")
      return 0
    else
      log "$tag INFO  ${label} — found incomplete file, re-downloading"
      rm -f "$dest"
    fi
  fi

  log "$tag START ${label}"
  log "       -> $dest"

  local dl_pid
  if [ "$USE_ARIA2" -eq 1 ]; then
    # 16 parallel connections, split into 16 pieces, auto-resume on retry
    aria2c -q -x16 -s16 -k1M --continue=true --max-tries=5 --retry-wait=5 \
      -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url" &
    dl_pid=$!
  else
    wget -q -c --tries=5 --timeout=60 -O "$dest" "$url" &
    dl_pid=$!
  fi

  local last_size=0
  local last_time=$SECONDS

  while kill -0 "$dl_pid" 2>/dev/null; do
    sleep 5
    local cur_size
    cur_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    local now=$SECONDS
    local elapsed=$(( now - last_time ))
    [ "$elapsed" -lt 1 ] && elapsed=1
    local speed_bps=$(( (cur_size - last_size) / elapsed ))
    local pct=0
    if [ "$expected_size" -gt 0 ]; then
      pct=$(( cur_size * 100 / expected_size ))
      [ "$pct" -gt 100 ] && pct=100
    fi
    log "$tag ...   ${pct}% — $(human_size "$cur_size") / ~$(human_size "$expected_size")  ($(human_size "$speed_bps")/s)"
    last_size=$cur_size
    last_time=$now
  done

  wait "$dl_pid"
  local dl_exit=$?

  if [ "$dl_exit" -ne 0 ]; then
    log "$tag FAIL  ${label} — download exited with error code $dl_exit"
    FAILED_ITEMS+=("$label")
    return 1
  fi

  local final_size
  final_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)

  if [ "$final_size" -lt "$expected_size" ]; then
    log "$tag FAIL  ${label} — downloaded but too small ($(human_size "$final_size"), expected ~$(human_size "$expected_size")). Likely a broken/redirected download."
    FAILED_ITEMS+=("$label")
    return 1
  fi

  log "$tag DONE  ${label} — $(human_size "$final_size")"
  OK_ITEMS+=("$label")
  return 0
}

# ---------------------------------------------------------------------------
# 1. Base checkpoint — RealVisXL V5.0 (SDXL photorealistic)
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors?download=true" \
  "${MODELS_DIR}/checkpoints/RealVisXL_V5.0_fp16.safetensors" \
  6900000000 \
  "RealVisXL V5.0 checkpoint"

# ---------------------------------------------------------------------------
# 2. InstantID
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true" \
  "${MODELS_DIR}/instantid/ip-adapter.bin" \
  1690000000 \
  "InstantID ip-adapter model"

download_model \
  "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/InstantID_ControlNet.safetensors" \
  2500000000 \
  "InstantID ControlNet model"

# antelopev2 face analysis (zip -> extracted with python, no unzip needed)
ANTELOPE_DIR="${MODELS_DIR}/insightface/models/antelopev2"
ANTELOPE_ZIP="${MODELS_DIR}/insightface/models/antelopev2.zip"
CURRENT_ITEM=$((CURRENT_ITEM + 1))
TAG="[${CURRENT_ITEM}/${TOTAL_ITEMS}]"
if [ -f "${ANTELOPE_DIR}/glintr100.onnx" ]; then
  log "$TAG SKIP  antelopev2 face models — already present"
  SKIPPED_ITEMS+=("antelopev2 face models")
else
  mkdir -p "$ANTELOPE_DIR"
  log "$TAG START antelopev2 face analysis models"
  wget -q -c --tries=5 --timeout=60 -O "$ANTELOPE_ZIP" \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip?download=true"
  if [ $? -eq 0 ] && [ -f "$ANTELOPE_ZIP" ]; then
    python3 -c "import zipfile; zipfile.ZipFile('${ANTELOPE_ZIP}').extractall('${MODELS_DIR}/insightface/models')"
    if [ -d "${ANTELOPE_DIR}/antelopev2" ]; then
      mv "${ANTELOPE_DIR}/antelopev2"/* "${ANTELOPE_DIR}/"
      rmdir "${ANTELOPE_DIR}/antelopev2"
    fi
    rm -f "$ANTELOPE_ZIP"
    if [ -f "${ANTELOPE_DIR}/glintr100.onnx" ]; then
      log "$TAG DONE  antelopev2 face models extracted"
      OK_ITEMS+=("antelopev2 face models")
    else
      log "$TAG FAIL  antelopev2 extraction didn't produce expected files — check $ANTELOPE_DIR manually"
      FAILED_ITEMS+=("antelopev2 face models")
    fi
  else
    log "$TAG FAIL  antelopev2 download failed"
    FAILED_ITEMS+=("antelopev2 face models")
  fi
fi

# ---------------------------------------------------------------------------
# 3. IPAdapter FaceID Plus v2 (SDXL) + matching LoRA + CLIP vision encoder
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin?download=true" \
  "${MODELS_DIR}/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin" \
  830000000 \
  "IPAdapter FaceID Plus v2 (SDXL)"

download_model \
  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors?download=true" \
  "${MODELS_DIR}/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors" \
  370000000 \
  "IPAdapter FaceID Plus v2 LoRA (SDXL)"

download_model \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true" \
  "${MODELS_DIR}/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" \
  2500000000 \
  "CLIP Vision encoder (for IPAdapter)"

# ---------------------------------------------------------------------------
# 4. ControlNet — OpenPose (SDXL) + Depth (SDXL)
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/OpenPoseXL2.safetensors" \
  5000000000 \
  "ControlNet OpenPose (SDXL)"

download_model \
  "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true" \
  "${MODELS_DIR}/controlnet/controlnet-depth-sdxl-1.0_fp16.safetensors" \
  2500000000 \
  "ControlNet Depth (SDXL, fp16)"

# ---------------------------------------------------------------------------
# 5. ReActor — face swap model + face restoration model
# ---------------------------------------------------------------------------
download_model \
  "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx?download=true" \
  "${MODELS_DIR}/insightface/inswapper_128.onnx" \
  554000000 \
  "ReActor inswapper_128 face-swap model"

download_model \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
  "${MODELS_DIR}/facerestore_models/GFPGANv1.4.pth" \
  349000000 \
  "GFPGAN v1.4 face restoration model"

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
log ""
log "==============================================================="
log " SUMMARY — finished at $(date)"
log "==============================================================="
log "OK (${#OK_ITEMS[@]}):"
for i in "${OK_ITEMS[@]:-}"; do [ -n "$i" ] && log "   - $i"; done
log ""
log "SKIPPED / already present (${#SKIPPED_ITEMS[@]}):"
for i in "${SKIPPED_ITEMS[@]:-}"; do [ -n "$i" ] && log "   - $i"; done
log ""
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  log "FAILED (${#FAILED_ITEMS[@]}):"
  for i in "${FAILED_ITEMS[@]}"; do log "   - $i"; done
  log ""
  log "Re-run this script to retry only the failed/incomplete files."
  log "Full log saved at: $LOG_FILE"
  exit 1
else
  log "All models downloaded/verified successfully."
  log "Full log saved at: $LOG_FILE"
  exit 0
fi