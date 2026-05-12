#!/bin/bash
# 实时监控对比实验进度

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

echo "=========================================="
echo "对比实验进度监控"
echo "=========================================="
echo ""

# 检查进程状态
echo "📊 运行中的进程："
ps aux | grep -E "train_convirt|train_mmformer|train_swin|train_medclip" | grep -v grep | awk '{printf "  PID %s: %s\n", $2, $11}'
echo ""

# GPU 使用情况
echo "🎮 GPU 使用情况："
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s: %s/%s MB (%.1f%%), Util: %s%%\n", $1, $2, $3, ($2/$3)*100, $4}'
echo ""

# 最新日志（每个方法）
echo "📝 最新日志状态："
for method in ConVIRT mmFormer "Swin-T_Fusion" MedCLIP; do
    log_file=$(find "$LOG_DIR" -name "${method}_sequential_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        last_line=$(tail -1 "$log_file" 2>/dev/null)
        echo "  $method:"
        echo "    文件: $(basename $log_file)"
        echo "    最后一行: ${last_line:0:80}..."
    else
        echo "  $method: 暂无日志"
    fi
done
echo ""

# 结果目录检查
echo "📁 结果目录："
for dir in MedCLIP ConVIRT mmFormer Swin-T_Fusion; do
    result_dir="$RESULTS_DIR/$dir"
    if [ -d "$result_dir" ]; then
        run_count=$(find "$result_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "  $dir: $run_count 个运行目录"
    else
        echo "  $dir: 目录不存在"
    fi
done
