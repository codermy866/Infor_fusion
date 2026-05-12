#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控消融实验进度
"""
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]  # experiments/exp_bio3.2

EXPERIMENTS = [
    "baseline",
    # 3.1特性的消融实验
    "w/o_visual_notes",
    "w/o_adaptive_gating",
    "w/o_alignment_loss",
    "w/o_ot_loss",
    "w/o_dual_head",
    "w/o_cross_attn",
    # 🔥 5.0特性的消融实验（新增）
    "w/o_hierarchical",
    "w/o_noise_aware",
    "w/o_clinical_evolver",
    "w/o_text_adapter",
]

def get_experiment_status(exp_name):
    """获取实验状态"""
    exp_dir = ROOT / "ablation_studies" / exp_name
    history_files = list(exp_dir.glob("logs/training_history_*.json"))
    
    if not history_files:
        return {
            "status": "未开始",
            "epochs": 0,
            "best_auc": 0.0,
            "latest_file": None
        }
    
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epochs = len(data.get('train_loss', []))
        best_auc = data.get('best_auc', 0.0)
        
        if epochs >= 20 and best_auc > 0.0:
            status = "✅ 已完成"
        elif epochs > 0:
            status = f"🔄 进行中 ({epochs}/20)"
        else:
            status = "⚠️ 未完成"
        
        return {
            "status": status,
            "epochs": epochs,
            "best_auc": best_auc,
            "latest_file": latest.name
        }
    except Exception as e:
        return {
            "status": f"❌ 错误: {e}",
            "epochs": 0,
            "best_auc": 0.0,
            "latest_file": None
        }

def get_running_processes():
    """获取正在运行的实验进程"""
    import subprocess
    try:
        result = subprocess.run(
            ["bash", "-lc", "ps aux | grep 'train_bio_cot_v3.2.py' | grep 'ablation_studies' | grep -v grep"],
            capture_output=True,
            text=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    # 提取实验名称
                    exp_name = None
                    for exp in EXPERIMENTS:
                        if exp in line:
                            exp_name = exp
                            break
                    processes.append({
                        "pid": pid,
                        "cpu": cpu,
                        "mem": mem,
                        "exp": exp_name or "unknown"
                    })
        return processes
    except Exception:
        return []

def main():
    print("=" * 80)
    print("📊 消融实验实时监控（整合5.0优势的3.2版本）")
    print("=" * 80)
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 显示实验状态
    print("实验状态:")
    print("-" * 80)
    for exp in EXPERIMENTS:
        status = get_experiment_status(exp)
        print(f"{exp:25s} | {status['status']:20s} | Epochs: {status['epochs']:2d}/20 | Best AUC: {status['best_auc']:.4f}")
        if status['latest_file']:
            print(f"  └─ 最新文件: {status['latest_file']}")
    
    print("\n" + "-" * 80)
    
    # 显示正在运行的进程
    processes = get_running_processes()
    if processes:
        print(f"\n🔄 正在运行的实验 ({len(processes)}个):")
        for p in processes:
            print(f"  PID: {p['pid']:8s} | CPU: {p['cpu']:5s}% | Mem: {p['mem']:5s}% | 实验: {p['exp']}")
    else:
        print("\n⚪ 当前没有正在运行的实验")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

