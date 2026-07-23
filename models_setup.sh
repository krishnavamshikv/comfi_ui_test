#!/usr/bin/env bash
# =============================================================================
#  AI Influencer - ComfyUI model + custom-node bootstrap for RunPod
#  Phase 1: Flux.1-dev + PuLID (face) + ControlNet-Union DWPose (pose) + IP-Adapter (outfit)
#  Phase 2: Wan2.1 VACE 14B (image + pose video -> dancing video, native ComfyUI nodes)
#  Target GPU: 48 GB (A6000 / L40S).  Output: 720 x 1280.  All open-source.
#
#  Run ONCE per RunPod session (nothing is meant to persist):
#     bash download_models.sh
# =============================================================================
set -euo pipefail

# ---- config -----------------------------------------------------------------
COMFYUI="${COMFYUI:-/workspace/ComfyUI}"          # override: COMFYUI=/path bash download_models.sh
JOBS=16                                            # aria2c connections per file

echo ">> ComfyUI dir: $COMFYUI"
[ -d "$COMFYUI" ] || { echo "!! $COMFYUI not found. Set COMFYUI=/your/path"; exit 1; }

# ---- ensure aria2c ----------------------------------------------------------
if ! command -v aria2c >/dev/null 2>&1; then
  echo ">> installing aria2c"
  apt-get update -y && apt-get install -y aria2 curl || (pip install --quiet aria2p; true)
fi

# ---- HuggingFace auth --------------------------------------------------------
# IMPORTANT: do NOT feed an "Authorization" header to aria2c. HF's Xet backend
# 302-redirects to a *pre-signed, byte-range-locked* CDN URL; forwarding an auth
# header onto that signed URL makes the CDN return 403. aria2c downloads the
# default (public) files correctly on its own. The token is only needed for
# *gated* repos, which are pulled with `hf download` (Xet-aware) via hfdl() below.
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
  echo ">> HF_TOKEN detected (used only for gated repos via hfdl)"
  export HF_TOKEN                      # picked up automatically by the hf CLI
  pip install --quiet -U "huggingface_hub[cli,hf_transfer]" >/dev/null 2>&1 || true
  export HF_HUB_ENABLE_HF_TRANSFER=1   # fast parallel chunked download
else
  echo ">> no HF_TOKEN set — public mirrors below need none"
fi

# ---- helper: dl <dest_dir> <out_name> <url>  (PUBLIC files, aria2c) ----------
dl () {
  local dir="$1" out="$2" url="$3"
  mkdir -p "$dir"
  if [ -f "$dir/$out" ]; then echo "   exists: $out"; return 0; fi
  echo ">> $out"
  aria2c -c -x"$JOBS" -s"$JOBS" -k1M --console-log-level=warn --summary-interval=0 \
         -d "$dir" -o "$out" "$url"
}

# ---- helper: hfdl <dest_dir> <out_name> <repo_id> <path_in_repo> -------------
# For GATED / Xet repos (needs HF_TOKEN + you must have accepted the repo terms).
hfdl () {
  local dir="$1" out="$2" repo="$3" path="$4"
  mkdir -p "$dir"
  if [ -f "$dir/$out" ]; then echo "   exists: $out"; return 0; fi
  echo ">> (hf) $out  <-  $repo/$path"
  hf download "$repo" "$path" --local-dir "$dir" >/dev/null
  # flatten if the file landed in a subpath, and normalize the name
  local got; got=$(find "$dir" -type f -name "$(basename "$path")" | head -1)
  [ -n "$got" ] && [ "$got" != "$dir/$out" ] && mv -f "$got" "$dir/$out"
}

# =============================================================================
#  1. CUSTOM NODES
# =============================================================================
CN="$COMFYUI/custom_nodes"
mkdir -p "$CN"
clone () { local url="$1" name; name=$(basename "$url" .git)
  if [ -d "$CN/$name" ]; then echo "   node exists: $name"; else
    echo ">> clone $name"; git clone --depth=1 "$url" "$CN/$name"; fi
  [ -f "$CN/$name/requirements.txt" ] && pip install --quiet -r "$CN/$name/requirements.txt" || true
}
clone https://github.com/ltdrdata/ComfyUI-Manager.git
clone https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git   # PuLID for Flux
clone https://github.com/Fannovel16/comfyui_controlnet_aux.git      # DWPose preprocessor
clone https://github.com/XLabs-AI/x-flux-comfyui.git                # Flux IP-Adapter
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git   # video load/combine
# insightface + onnxruntime for PuLID (in case node requirements missed them)
pip install --quiet insightface onnxruntime-gpu facexlib timm || true

# =============================================================================
#  2. PHASE 1 MODELS  (Flux image pipeline)
# =============================================================================
M="$COMFYUI/models"

# --- Flux base (fp8 UNet, fits 48GB with headroom for PuLID+CN+IPA) ---------
dl "$M/diffusion_models" flux1-dev-fp8.safetensors \
   https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors
# Prefer the OFFICIAL full-precision Flux instead? It's gated — accept terms at
# https://huggingface.co/black-forest-labs/FLUX.1-dev then use (needs HF_TOKEN):
#   hfdl "$M/diffusion_models" flux1-dev.safetensors black-forest-labs/FLUX.1-dev flux1-dev.safetensors
#   hfdl "$M/vae"             ae.safetensors        black-forest-labs/FLUX.1-dev ae.safetensors

# --- Flux text encoders ------------------------------------------------------
dl "$M/text_encoders" t5xxl_fp16.safetensors \
   https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
dl "$M/text_encoders" clip_l.safetensors \
   https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors

# --- Flux VAE (non-gated mirror) --------------------------------------------
dl "$M/vae" ae.safetensors \
   https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors

# --- PuLID-Flux (face identity of the AI character) -------------------------
dl "$M/pulid" pulid_flux_v0.9.1.safetensors \
   https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors
# EVA-CLIP used by PuLID
dl "$M/clip" EVA02_CLIP_L_336_psz14_s6B.pt \
   https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt
# InsightFace antelopev2 (5 onnx) for face detection/embedding
AF="$M/insightface/models/antelopev2"
for f in 1k3d68 2d106det genderage glintr100 scrfd_10g_bnkps; do
  dl "$AF" "$f.onnx" "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/$f.onnx"
done

# --- Flux ControlNet Union Pro (used in OpenPose mode for the reference pose)-
dl "$M/controlnet" FLUX.1-dev-ControlNet-Union-Pro.safetensors \
   https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors

# --- DWPose ONNX for the controlnet_aux preprocessor ------------------------
DW="$CN/comfyui_controlnet_aux/ckpts/yzd-v/DWPose"
dl "$DW" yolox_l.onnx          https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
dl "$DW" dw-ll_ucoco_384.onnx  https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx

# --- XLabs Flux IP-Adapter (transfers the outfit/style from reference) ------
dl "$M/xlabs/ipadapters" flux-ip-adapter-v2.safetensors \
   https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors
# CLIP-Vision for the IP-Adapter
dl "$M/clip_vision" clip-vit-large-patch14.safetensors \
   https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors

# =============================================================================
#  3. PHASE 2 MODELS  (Wan2.1 VACE 14B video pipeline - native ComfyUI nodes)
# =============================================================================
WAN=https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files

# 14B VACE diffusion model. fp16 (~32GB) fits 48GB. For more headroom swap the
# line below for the fp8 build (comment/uncomment as you like).
dl "$M/diffusion_models" wan2.1_vace_14B_fp16.safetensors \
   "$WAN/diffusion_models/wan2.1_vace_14B_fp16.safetensors"
# dl "$M/diffusion_models" wan2.1_vace_14B_fp8_scaled.safetensors \
#    "$WAN/diffusion_models/wan2.1_vace_14B_fp8_scaled.safetensors"

# Wan text encoder (umt5) + VAE
dl "$M/text_encoders" umt5_xxl_fp8_e4m3fn_scaled.safetensors \
   "$WAN/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
dl "$M/vae" wan_2.1_vae.safetensors \
   "$WAN/vae/wan_2.1_vae.safetensors"

echo ""
echo "=============================================================="
echo " DONE. Restart ComfyUI so the new custom nodes are registered:"
echo "   cd $COMFYUI && python main.py --listen 0.0.0.0 --port 8188"
echo " Then load workflow_phase1_api.json and workflow_phase2_api.json"
echo "=============================================================="
