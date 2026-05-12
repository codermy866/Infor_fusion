#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控对比实验的运行状态
"""

import json
from pathlib import Path
from datetime import datetime
import time

EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / 'comparison_experiments' / 'results'
LOG_DIR = EXP_ROOT / 'comparison_experiments' / 'logs'
CONFIG_FILE = RESULTS_DIR / 'experiment_config.json'
SUMMARY_FILE = RESULTS_DIR / 'results_summary.json'

def get_experiment_status():
    """获取实验运行状态"""
    
    if not CONFIG_FILE.exists():
        return None, "实验尚未开始"
    
    # 读取配置
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    experiments = config.get('experiments', [])
    
    # 读取结果汇总（如果存在）
    summary = None
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    
    status = []
    
    for exp in experiments:
        exp_name = exp['name']
        num_runs = exp.get('num_runs', 5)
        seeds = exp.get('seeds', [])
        
        # 检查每个运行的状态
        runs_status = []
        completed = 0
        
        for run_idx in range(1, num_runs + 1):
            seed = seeds[run_idx - 1] if run_idx <= len(seeds) else None
            run_dir = RESULTS_DIR / exp_name / f'run_{run_idx}_seed_{seed}'
            
            # 检查是否有结果文件
            has_result = False
            if run_dir.exists():
                # 检查是否有checkpoint或结果文件
                checkpoint_files = list(run_dir.rglob('*.pth'))
                result_files = list(run_dir.rglob('*.json'))
                has_result = len(checkpoint_files) > 0 or len(result_files) > 0
            
            # 检查日志文件
            log_files = list(LOG_DIR.glob(f'{exp_name}_run{run_idx}_seed{seed}_*.log'))
            has_log = len(log_files) > 0
            
            if has_result:
                completed += 1
                status_str = "✅ 完成"
            elif has_log:
                # 检查日志最后修改时间
                if log_files:
                    last_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    mtime = datetime.fromtimestamp(last_log.stat().st_mtime)
                    age = datetime.now() - mtime
                    if age.total_seconds() < 300:  # 5分钟内更新过
                        status_str = "🔄 运行中"
                    else:
                        status_str = "⏸️ 可能已停止"
                else:
                    status_str = "❓ 未知"
            else:
                status_str = "⏳ 未开始"
            
            runs_status.append({
                'run': run_idx,
                'seed': seed,
                'status': status_str,
                'has_result': has_result,
                'has_log': has_log
            })
        
        status.append({
            'name': exp_name,
            'description': exp.get('description', ''),
            'completed': completed,
            'total': num_runs,
            'progress': f"{completed}/{num_runs}",
            'runs': runs_status
        })
    
    return status, config

def print_status():
    """打印实验状态"""
    status, config = get_experiment_status()
    
    if status is None:
        print("⚠️ 实验尚未开始")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 对比实验运行状态")
    print(f"{'='*80}\n")
    
    if config:
        start_time = config.get('start_time', '')
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed = datetime.now() - start_dt
                print(f"开始时间: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"已运行时间: {elapsed}")
                print()
            except:
                pass
    
    for exp_status in status:
        exp_name = exp_status['name']
        completed = exp_status['completed']
        total = exp_status['total']
        progress = exp_status['progress']
        
        print(f"🔬 {exp_name}")
        print(f"   描述: {exp_status['description']}")
        print(f"   进度: {progress} ({completed}/{total})")
        
        # 显示每个运行的详细状态
        for run in exp_status['runs']:
            print(f"      Run {run['run']} (Seed {run['seed']}): {run['status']}")
        
        print()
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='监控对比实验状态')
    parser.add_argument('--watch', action='store_true', help='持续监控（每60秒刷新）')
    args = parser.parse_args()
    
    if args.watch:
        print("🔄 持续监控模式（按Ctrl+C退出）")
        try:
            while True:
                print_status()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n✅ 监控已停止")
    else:
        print_status()

