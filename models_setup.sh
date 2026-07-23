#!/usr/bin/env bash
set -e
export GIT_TERMINAL_PROMPT=0

echo "=== 1. Installing System Dependencies ==="
apt-get update && apt-get install -y aria2 git wget

# Define ComfyUI root (standard RunPod PyTorch/ComfyUI template path)
COMFY_DIR="/workspace/runpod-slim/ComfyUI"
if [ ! -d "$COMFY_DIR" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

cd "$COMFY_DIR/custom_nodes"

echo "=== 2. Cloning Essential Custom Nodes ==="
# ComfyUI Manager (for dependency resolution)
[ ! -d "ComfyUI-Manager" ] && git clone https://github.com/ltdrdata/ComfyUI-Manager.git
# IP-Adapter Plus (for Phase 1 face & identity preservation)
[ ! -d "ComfyUI_IPAdapter_plus" ] && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
# Advanced ControlNet & DWPose Aux (for pose extraction from image & video)
[ ! -d "ComfyUI-Advanced-ControlNet" ] && git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
[ ! -d "comfyui_controlnet_aux" ] && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
# VideoHelperSuite (for video loading, frame extraction, and MP4 export)
[ ! -d "ComfyUI-VideoHelperSuite" ] && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
# MimicMotion / AnimateDiff (for Phase 2 dance & expression video generation)
# Corrected MimicMotion Wrapper repository
[ ! -d "ComfyUI-MimicMotionWrapper" ] && git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git
[ ! -d "ComfyUI-AnimateDiff-Evolved" ] && git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

echo "=== 3. Installing Node Python Requirements ==="
pip install -r "$COMFY_DIR/custom_nodes/comfyui_controlnet_aux/requirements.txt"
pip install -r "$COMFY_DIR/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
pip install -r "$COMFY_DIR/custom_nodes/ComfyUI-MimicMotionWrapper/requirements.txt" || true

echo "=== 4. Downloading Models via aria2c (16x Concurrent Connections) ==="
# Function for fast downloading
download_model() {
    local url=$1
    local dir=$2
    local out=$3
    mkdir -p "$dir"
    echo "Downloading $out to $dir..."
    aria2c -x 16 -s 16 -k 1M -d "$dir" -o "$out" "$url"
}

# 1. Base SDXL Model (RealVisXL V4.0 - optimized for photorealistic people/influencers)
download_model \
  "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors" \
  "$COMFY_DIR/models/checkpoints" \
  "RealVisXL_V4.0.safetensors"

# 2. CLIP Vision Models (Required for IP-Adapter)
download_model \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
  "$COMFY_DIR/models/clip_vision" \
  "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# 3. IP-Adapter SDXL Weights (FaceID & General Plus for identity & style)
download_model \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
  "$COMFY_DIR/models/ipadapter" \
  "ip-adapter-plus-face_sdxl_vit-h.safetensors"
download_model \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
  "$COMFY_DIR/models/ipadapter" \
  "ip-adapter-plus_sdxl_vit-h.safetensors"

# 4. ControlNet OpenPose SDXL (For Phase 1 pose matching)
download_model \
  "https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" \
  "$COMFY_DIR/models/controlnet" \
  "controlnet-openpose-sdxl-1.0.safetensors"

# 5. MimicMotion / AnimateDiff Motion Weights (For Phase 2 dance video generation)
download_model \
  "https://huggingface.co/TencentARC/MimicMotion/resolve/main/MimicMotion_1-1.pth" \
  "$COMFY_DIR/models/mimicmotion" \
  "MimicMotion_1-1.pth"
download_model \
  "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt" \
  "$COMFY_DIR/models/animatediff_models" \
  "mm_sdxl_v10_beta.ckpt"

echo "=== Setup Complete! Starting ComfyUI ==="
cd "$COMFY_DIR"
python main.py --listen 0.0.0.0 --port 8188