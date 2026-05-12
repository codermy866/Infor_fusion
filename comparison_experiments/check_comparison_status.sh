#!/bin/bash
# 检查对比实验运行状态

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

echo "=========================================="
echo "📊 对比实验运行状态检查"
echo "=========================================="
echo ""

echo "🔍 正在运行的进程:"
ps aux | grep -E "train_medclip|train_convirt|train_mmformer|train_swin|run_selected_baselines" | grep -v grep | head -15
echo ""

echo "💾 GPU显存使用情况:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "GPU %s: %sMB / %sMB 使用, 利用率: %s%%\n", $1, $2, $3, $4}'
echo ""

echo "📁 实验结果检查:"
for method in MedCLIP ConVIRT mmFormer Swin-T_Fusion; do
    result_dir="$RESULTS_DIR/$method"
    if [ -d "$result_dir" ]; then
        # 检查是否有checkpoint或结果文件
        checkpoints=$(find "$result_dir" -name "*.pth" 2>/dev/null | wc -l)
        jsons=$(find "$result_dir" -name "*.json" 2>/dev/null | wc -l)
        echo "  $method: $checkpoints checkpoints, $jsons json files"
    else
        echo "  $method: 目录不存在"
    fi
done
echo ""

echo "📝 最新日志文件:"
ls -lht "$LOG_DIR"/*_sequential_*.log 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 ")"}'
echo ""

echo "=========================================="
