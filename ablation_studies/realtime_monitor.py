#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控消融实验进度
显示当前执行阶段、进度、关键指标等
"""
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]  # experiments/exp_bio3.2

EXPERIMENTS = [
    "baseline",
    "w/o_visual_notes",
    "w/o_adaptive_gating",
    "w/o_alignment_loss",
    "w/o_ot_loss",
    "w/o_dual_head",
    "w/o_cross_attn",
    "w/o_hierarchical",
    "w/o_noise_aware",
    "w/o_clinical_evolver",
    "w/o_text_adapter",
]

def get_experiment_status(exp_name):
    """获取实验状态"""
    exp_dir = ROOT / "ablation_studies" / exp_name
    history_files = list(exp_dir.glob("logs/training_history_*.json"))
    
    status = {
        "name": exp_name,
        "status": "未开始",
        "epochs": 0,
        "total_epochs": 20,
        "best_auc": 0.0,
        "current_loss": 0.0,
        "latest_file": None,
        "is_running": False,
        "pid": None,
        "start_time": None,
        "elapsed_time": None,
    }
    
    if not history_files:
        return status
    
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    status["latest_file"] = latest.name
    
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epochs = len(data.get('train_loss', []))
        best_auc = data.get('best_auc', 0.0)
        current_loss = data.get('train_loss', [0.0])[-1] if data.get('train_loss') else 0.0
        
        status["epochs"] = epochs
        status["best_auc"] = best_auc
        status["current_loss"] = current_loss
        
        if epochs >= 20 and best_auc > 0.0:
            status["status"] = "✅ 已完成"
        elif epochs > 0:
            status["status"] = f"🔄 进行中"
        else:
            status["status"] = "⚠️ 未完成"
    except Exception as e:
        status["status"] = f"❌ 错误: {str(e)[:30]}"
    
    return status

def get_running_processes():
    """获取正在运行的实验进程"""
    try:
        result = subprocess.run(
            ["bash", "-lc", "ps aux | grep 'train_bio_cot_v3.2.py' | grep 'ablation_studies' | grep -v grep"],
            capture_output=True,
            text=True
        )
        processes = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    start_time = ' '.join(parts[8:10]) if len(parts) > 9 else 'N/A'
                    
                    # 提取实验名称
                    exp_name = None
                    for exp in EXPERIMENTS:
                        if exp in line:
                            exp_name = exp
                            break
                    
                    if exp_name:
                        processes[exp_name] = {
                            "pid": pid,
                            "cpu": cpu,
                            "mem": mem,
                            "start_time": start_time
                        }
        return processes
    except Exception:
        return {}

def get_current_epoch_from_log(exp_name):
    """从日志文件中提取当前epoch"""
    exp_dir = ROOT / "ablation_studies" / exp_name / "logs"
    log_files = list(exp_dir.glob("nohup_*.log"))
    
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 查找最新的epoch
            import re
            epoch_matches = re.findall(r'Epoch (\d+)/\d+', content)
            if epoch_matches:
                return int(epoch_matches[-1])
            
            # 查找训练阶段
            if 'Epoch' in content:
                lines = content.split('\n')
                for line in reversed(lines):
                    if 'Epoch' in line and '/' in line:
                        match = re.search(r'Epoch (\d+)/\d+', line)
                        if match:
                            return int(match.group(1))
    except Exception:
        pass
    
    return None

def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds/60)}分{int(seconds%60)}秒"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}小时{minutes}分钟"

def display_status(refresh_interval=30):
    """显示实验状态（实时刷新）"""
    import os
    
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 100)
        print("📊 Bio-COT 3.2 消融实验实时监控（整合5.0优势）")
        print("=" * 100)
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"刷新间隔: {refresh_interval}秒（按 Ctrl+C 退出）")
        print("=" * 100)
        print()
        
        # 获取运行中的进程
        running_processes = get_running_processes()
        
        # 获取所有实验状态
        all_status = []
        for exp in EXPERIMENTS:
            status = get_experiment_status(exp)
            
            # 检查是否正在运行
            if exp in running_processes:
                status["is_running"] = True
                status["pid"] = running_processes[exp]["pid"]
                status["cpu"] = running_processes[exp]["cpu"]
                status["mem"] = running_processes[exp]["mem"]
                
                # 从日志获取当前epoch
                current_epoch = get_current_epoch_from_log(exp)
                if current_epoch:
                    status["epochs"] = current_epoch
            
            all_status.append(status)
        
        # 统计信息
        completed = sum(1 for s in all_status if s["status"] == "✅ 已完成")
        running = sum(1 for s in all_status if s["is_running"])
        pending = len(EXPERIMENTS) - completed - running
        
        print(f"📈 总体进度: {completed}/{len(EXPERIMENTS)} 已完成 | {running} 运行中 | {pending} 待执行")
        print()
        
        # 当前正在运行的实验
        running_exps = [s for s in all_status if s["is_running"]]
        if running_exps:
            print("🔄 当前正在运行的实验:")
            print("-" * 100)
            for status in running_exps:
                progress_bar = "█" * int(status["epochs"] / status["total_epochs"] * 20) + "░" * (20 - int(status["epochs"] / status["total_epochs"] * 20))
                print(f"  {status['name']:30s} | {status['status']:15s} | Epoch: {status['epochs']:2d}/{status['total_epochs']:2d} | [{progress_bar}]")
                print(f"    └─ PID: {status.get('pid', 'N/A'):8s} | CPU: {status.get('cpu', '0'):5s}% | Mem: {status.get('mem', '0'):5s}% | Best AUC: {status['best_auc']:.4f} | Loss: {status['current_loss']:.4f}")
            print()
        
        # 所有实验状态
        print("📋 所有实验状态:")
        print("-" * 100)
        print(f"{'实验名称':<30s} | {'状态':<15s} | {'进度':<25s} | {'Best AUC':<10s} | {'当前Loss':<10s}")
        print("-" * 100)
        
        for status in all_status:
            progress = f"{status['epochs']}/{status['total_epochs']}"
            progress_bar = "█" * int(status["epochs"] / status["total_epochs"] * 20) + "░" * (20 - int(status["epochs"] / status["total_epochs"] * 20))
            progress_str = f"[{progress_bar}] {progress}"
            
            print(f"{status['name']:<30s} | {status['status']:<15s} | {progress_str:<25s} | {status['best_auc']:>9.4f} | {status['current_loss']:>9.4f}")
        
        print()
        print("=" * 100)
        print("💡 提示: 按 Ctrl+C 退出监控")
        print("=" * 100)
        
        # 等待刷新
        try:
            time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\n👋 监控已退出")
            break

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实时监控消融实验进度')
    parser.add_argument('--interval', type=int, default=30, help='刷新间隔（秒），默认30秒')
    parser.add_argument('--once', action='store_true', help='只显示一次，不循环刷新')
    
    args = parser.parse_args()
    
    if args.once:
        # 只显示一次
        import os
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 100)
        print("📊 Bio-COT 3.2 消融实验状态（单次查看）")
        print("=" * 100)
        print(f"查看时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        print()
        
        running_processes = get_running_processes()
        all_status = []
        for exp in EXPERIMENTS:
            status = get_experiment_status(exp)
            if exp in running_processes:
                status["is_running"] = True
                status["pid"] = running_processes[exp]["pid"]
                current_epoch = get_current_epoch_from_log(exp)
                if current_epoch:
                    status["epochs"] = current_epoch
            all_status.append(status)
        
        completed = sum(1 for s in all_status if s["status"] == "✅ 已完成")
        running = sum(1 for s in all_status if s["is_running"])
        
        print(f"📈 总体进度: {completed}/{len(EXPERIMENTS)} 已完成 | {running} 运行中 | {len(EXPERIMENTS)-completed-running} 待执行")
        print()
        
        running_exps = [s for s in all_status if s["is_running"]]
        if running_exps:
            print("🔄 当前正在运行的实验:")
            print("-" * 100)
            for status in running_exps:
                progress_bar = "█" * int(status["epochs"] / status["total_epochs"] * 20) + "░" * (20 - int(status["epochs"] / status["total_epochs"] * 20))
                print(f"  {status['name']:30s} | Epoch: {status['epochs']:2d}/{status['total_epochs']:2d} | [{progress_bar}] | Best AUC: {status['best_auc']:.4f}")
            print()
        
        print("📋 所有实验状态:")
        print("-" * 100)
        for status in all_status:
            progress_bar = "█" * int(status["epochs"] / status["total_epochs"] * 20) + "░" * (20 - int(status["epochs"] / status["total_epochs"] * 20))
            print(f"  {status['name']:30s} | {status['status']:15s} | [{progress_bar}] {status['epochs']:2d}/20 | AUC: {status['best_auc']:.4f}")
    else:
        # 循环刷新
        display_status(args.interval)

if __name__ == "__main__":
    main()

