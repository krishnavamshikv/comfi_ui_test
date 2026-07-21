#!/usr/bin/env bash
# ==========================================================================
# AI Influencer MVP — model + custom node bootstrap for ComfyUI on RunPod
# Re-run this on every fresh pod. Idempotent (skips files that already exist).
#
# Usage:
#   export HF_TOKEN=hf_xxx        # needed for the gated FLUX Kontext file
#   cd /workspace/runpod-slim/ComfyUI         # <-- set this to your actual ComfyUI root
#   bash setup_models.sh
# ==========================================================================
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-$(pwd)}"
echo "Installing into: $COMFYUI_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. The FLUX Kontext dev download is a gated"
  echo "repo and will fail without it. Get a token at https://huggingface.co/settings/tokens"
  echo "and accept the license at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev"
fi

mkdir -p "$COMFYUI_DIR"/models/{diffusion_models,text_encoders,vae,controlnet}
mkdir -p "$COMFYUI_DIR"/custom_nodes

dl () {
  # dl <url> <output_path> [use_auth]
  local url="$1" out="$2" auth="${3:-no}"
  if [ -f "$out" ]; then
    echo "SKIP (exists): $out"
    return
  fi
  echo "Downloading -> $out"
  if [ "$auth" = "auth" ]; then
    wget -q --show-progress --header="Authorization: Bearer ${HF_TOKEN:-}" -O "$out" "$url"
  else
    wget -q --show-progress -O "$out" "$url"
  fi
}

# --------------------------------------------------------------------------
# PHASE 1 — FLUX.1 Kontext Dev (character pose + outfit transfer, image)
# --------------------------------------------------------------------------
dl "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" \
   "$COMFYUI_DIR/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" auth

dl "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
   "$COMFYUI_DIR/models/text_encoders/clip_l.safetensors"

dl "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors" \
   "$COMFYUI_DIR/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"

# Flux VAE — pulled from the non-gated schnell repo, identical VAE weights
dl "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" \
   "$COMFYUI_DIR/models/vae/ae.safetensors"

# --------------------------------------------------------------------------
# PHASE 2 — Wan2.1 VACE (image + motion-control video -> video)
# 1.3B = MVP / fast iteration. Swap for the 14B VACE file later for quality.
# --------------------------------------------------------------------------
dl "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" \
   "$COMFYUI_DIR/models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors"

dl "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
   "$COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

dl "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
   "$COMFYUI_DIR/models/vae/wan_2.1_vae.safetensors"

# --------------------------------------------------------------------------
# Custom nodes (needed for video I/O + pose extraction in Phase 2)
# --------------------------------------------------------------------------
cd "$COMFYUI_DIR/custom_nodes"

if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
fi

if [ ! -d "comfyui_controlnet_aux" ]; then
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
fi

if [ ! -d "ComfyUI-Manager" ]; then
  git clone https://github.com/Comfy-Org/ComfyUI-Manager.git
fi

echo ""
echo "Installing custom node python deps..."
for d in ComfyUI-VideoHelperSuite comfyui_controlnet_aux ComfyUI-Manager; do
  if [ -f "$d/requirements.txt" ]; then
    pip install -q -r "$d/requirements.txt" --break-system-packages || true
  fi
done

echo ""
echo "=========================================================="
echo "Done. Restart ComfyUI so it picks up the new nodes/models."
echo "=========================================================="