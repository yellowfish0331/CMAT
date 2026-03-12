#!/bin/bash
# OmniGibson 优化版 headless 运行脚本

set -e

echo "🚀 启动OmniGibson headless渲染（优化版）..."

# 确保在正确的conda环境
if [[ "$CONDA_DEFAULT_ENV" != "omnigibson" ]]; then
    echo "❌ 请先激活omnigibson环境: conda activate omnigibson"
    exit 1
fi

# 检查虚拟显示是否运行
if ! pgrep -f "Xvfb :99" > /dev/null; then
    echo "❌ 虚拟显示未运行，请先执行: ~/setup_headless_display.sh"
    exit 1
fi

# 设置基础环境变量
export DISPLAY=:99
export PYTHONPATH=$PYTHONPATH:/home/pengzelin/yuhuang/HACL/unsup-affordance/behavior1k_omnigibson_render

# 设置OmniGibson headless渲染参数
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_NO_DISPLAY=1

# 强制使用OpenGL而不是Vulkan（解决驱动不兼容问题）
export OMNI_USE_OPENGL=1
export CARB_GRAPHICS_USE_OPENGL=1
export OMNI_BACKEND_RENDERER="OpenGL"

# GPU和渲染相关设置
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export __GL_SYNC_TO_VBLANK=0
export __GL_SYNC_DISPLAY_DEVICE=":99"

# 禁用GUI相关功能
export OMNI_KIT_ALLOW_ROOT=1
export OMNI_KIT_HIDE_UI=1
export OMNI_KIT_DISABLE_INPUT=1

# Isaac Sim特定设置
export CARB_SETTINGS_OVERRIDES='{"renderer": {"backend": "OpenGL"}, "/app/window/hideUi": true, "/app/headless": true}'

echo "✅ 环境设置完成（优化版）"
echo "📺 DISPLAY = $DISPLAY"
echo "🎮 GPU设备 = $NVIDIA_VISIBLE_DEVICES"
echo "🖥️ 渲染后端 = OpenGL"
echo "🐍 Python环境 = $CONDA_DEFAULT_ENV"

# 运行OmniGibson（使用原始参数但添加更完整的headless设置）
echo "🎬 开始渲染..."
python render.py \
    --orientation_root qa_merged/ \
    --og_dataset_root /home/pengzelin/.conda/envs/omnigibson/lib/python3.10/site-packages/omnigibson/data/og_dataset/objects \
    --category_model_list selected_object_models.json \
    --save_path behavior1k
