#!/bin/bash
# 快速检查对比实验运行状态

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

echo "=========================================="
echo "📊 对比实验运行状态"
echo "=========================================="
echo ""

echo "🔍 正在运行的进程:"
ps aux | grep -E "train_bio_cot|train_medclip|train_convirt|train_mmformer|train_swin|run_all_comparisons" | grep -v grep | head -10
echo ""

echo "💾 GPU显存使用情况:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "GPU %s: %sMB / %sMB 使用, 利用率: %s%%\n", $1, $2, $3, $4}'
echo ""

echo "📁 最新日志文件:"
ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -5
echo ""

echo "✅ 实验结果目录:"
ls -d "$RESULTS_DIR"/*/ 2>/dev/null | head -5
echo ""

echo "=========================================="
