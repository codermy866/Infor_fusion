#!/bin/bash
# 启动实时监控（后台运行，输出到文件）

cd "$(dirname "$0")"

# 创建监控日志目录
mkdir -p monitor_logs

# 启动实时监控（每30秒刷新一次）
nohup python -u realtime_monitor.py --interval 30 > monitor_logs/realtime_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "✅ 实时监控已启动（后台运行）"
echo "   PID: $!"
echo "   日志: monitor_logs/realtime_monitor_*.log"
echo ""
echo "📊 查看实时监控："
echo "   tail -f monitor_logs/realtime_monitor_*.log"
echo ""
echo "📊 交互式监控（前台运行）："
echo "   python realtime_monitor.py"
echo ""
echo "📊 单次查看："
echo "   python realtime_monitor.py --once"
