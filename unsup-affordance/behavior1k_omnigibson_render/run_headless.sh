#!/bin/bash
# OmniGibson headless运行脚本

set -e

echo "🚀 启动OmniGibson headless渲染..."

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

# 设置环境变量
export DISPLAY=:99
export PYTHONPATH=$PYTHONPATH:/home/pengzelin/yuhuang/HACL/unsup-affordance/behavior1k_omnigibson_render

# 设置OmniGibson headless渲染参数
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_NO_DISPLAY=1

echo "✅ 环境设置完成"
echo "📺 DISPLAY = $DISPLAY"
echo "🐍 Python环境 = $CONDA_DEFAULT_ENV"

# 运行OmniGibson（使用原始参数但添加headless相关设置）
echo "🎬 开始渲染..."
python render.py \
    --orientation_root qa_merged/ \
    --og_dataset_root /home/pengzelin/.conda/envs/omnigibson/lib/python3.10/site-packages/omnigibson/data/og_dataset/objects \
    --category_model_list selected_object_models.json \
    --save_path behavior1k
