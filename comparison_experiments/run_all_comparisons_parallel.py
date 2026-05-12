#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动执行所有对比实验（支持并行运行）
根据GPU显存情况智能调度，支持多线程并行运行
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
import json
import signal
import threading
from queue import Queue
import psutil

# 添加项目路径
EXP_ROOT = Path(__file__).resolve().parents[1]  # 当前实验目录
EXPERIMENTS_ROOT = EXP_ROOT.parent
REPO_ROOT = EXPERIMENTS_ROOT.parent
EXP_3_0_ROOT = EXPERIMENTS_ROOT / 'exp_bio3.0_improved'

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(EXP_ROOT))

# 对比实验配置（按优先级排序）
COMPARISON_EXPERIMENTS = [
    {
        'name': 'Bio-COT_3.2_Full',
        'script': 'training/train_bio_cot_v3.2.py',
        'config': None,
        'description': 'Bio-COT 3.2 (Full) - Our proposed method',
        'category': 'ours',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024],
        'estimated_memory_mb': 4000  # 预估显存需求（MB）
    },
    {
        'name': 'MedCLIP',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'medclip' / 'train_medclip.py'),
        'config': None,
        'description': 'MedCLIP - Medical domain-specific CLIP model',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024],
        'estimated_memory_mb': 3000
    },
    {
        'name': 'ConVIRT',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'convirt' / 'train_convirt.py'),
        'config': None,
        'description': 'ConVIRT - Contrastive Vision-Representation Transformer',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024],
        'estimated_memory_mb': 3500
    },
    {
        'name': 'mmFormer',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'mmformer' / 'train_mmformer.py'),
        'config': None,
        'description': 'mmFormer - Multi-modal Medical Transformer',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024],
        'estimated_memory_mb': 4000
    },
    {
        'name': 'Swin-T_Fusion',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'swin_t_baseline' / 'train_swin.py'),
        'config': None,
        'description': 'Swin-T + Fusion - Swin-T encoder with simple fusion',
        'category': 'baseline',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024],
        'estimated_memory_mb': 2500
    },
]

# 创建输出目录
OUTPUT_DIR = EXP_ROOT / 'comparison_experiments' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = EXP_ROOT / 'comparison_experiments' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量
running_processes = {}  # {task_id: process}
task_queue = Queue()
results_lock = threading.Lock()
results_summary = []

def get_gpu_memory_info():
    """获取GPU显存信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_idx = int(parts[0])
                    total = int(parts[1])
                    used = int(parts[2])
                    free = int(parts[3])
                    gpus.append({
                        'index': gpu_idx,
                        'total_mb': total,
                        'used_mb': used,
                        'free_mb': free,
                        'utilization': (used / total * 100) if total > 0 else 0
                    })
        return gpus
    except Exception as e:
        print(f"⚠️ 无法获取GPU信息: {e}")
        return []

def find_available_gpu(required_memory_mb, exclude_gpus=None):
    """查找可用的GPU"""
    gpus = get_gpu_memory_info()
    exclude_gpus = exclude_gpus or []
    
    for gpu in gpus:
        if gpu['index'] in exclude_gpus:
            continue
        if gpu['free_mb'] >= required_memory_mb:
            return gpu['index']
    
    return None

def get_running_gpu_processes():
    """获取当前正在运行的GPU进程"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        
        running_pids = set()
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 1:
                    try:
                        pid = int(parts[0])
                        running_pids.add(pid)
                    except:
                        pass
        
        return running_pids
    except:
        return set()

def run_single_experiment(exp_config, run_idx, seed, gpu_id=None):
    """
    运行单个实验（一次运行）
    
    Args:
        exp_config: 实验配置字典
        run_idx: 运行索引（1-5）
        seed: 随机种子
        gpu_id: 指定的GPU ID（None表示自动选择）
    
    Returns:
        (success, log_path, result_path, process)
    """
    exp_name = exp_config['name']
    script_path = exp_config['script']
    description = exp_config['description']
    estimated_memory = exp_config.get('estimated_memory_mb', 3000)
    
    # 创建实验输出目录
    exp_output_dir = OUTPUT_DIR / exp_name / f'run_{run_idx}_seed_{seed}'
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'{exp_name}_run{run_idx}_seed{seed}_{timestamp}.log'
    
    print(f"\n{'='*80}")
    print(f"🚀 开始运行: {exp_name} (Run {run_idx}/5, Seed: {seed})")
    print(f"   描述: {description}")
    print(f"   脚本: {script_path}")
    print(f"   输出: {exp_output_dir}")
    print(f"   日志: {log_file}")
    if gpu_id is not None:
        print(f"   GPU: {gpu_id}")
    print(f"{'='*80}\n")
    
    # 选择GPU
    if gpu_id is None:
        gpu_id = find_available_gpu(estimated_memory)
        if gpu_id is None:
            print(f"⚠️ 没有足够的GPU显存（需要{estimated_memory}MB），等待中...")
            # 等待一段时间后重试
            time.sleep(60)
            gpu_id = find_available_gpu(estimated_memory)
            if gpu_id is None:
                print(f"❌ 仍然没有足够的GPU显存，跳过此任务")
                return False, log_file, exp_output_dir, None
    
    # 构建命令
    if exp_name == 'Bio-COT_3.2_Full':
        # 创建临时配置文件
        temp_config_file = exp_output_dir / 'config_temp.py'
        temp_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        default_config_path = EXP_ROOT / 'config.py'
        if default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            config_content += f"""

# ============================================================
# 自动生成的临时配置（Run {run_idx}, Seed {seed})
# ============================================================
import os
import random
import numpy as np
import torch

# 设置随机种子
random.seed({seed})
np.random.seed({seed})
torch.manual_seed({seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({seed})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 创建配置实例后修改输出目录
config = BioCOT_v3_2_Config()
config.output_dir = '{exp_output_dir / "results"}'
config.checkpoint_dir = '{exp_output_dir / "checkpoints"}'
config.log_dir = '{exp_output_dir / "logs"}'
"""
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
        
        cmd = [
            'python', str(EXP_ROOT / script_path),
            '--config', str(temp_config_file) if temp_config_file.exists() else str(default_config_path),
            '--gpu', str(gpu_id)
        ]
        work_dir = EXP_ROOT
    else:
        # 其他方法
        script_abs = Path(script_path)
        if not script_abs.exists():
            print(f"❌ 错误: 脚本不存在: {script_path}")
            return False, None, None, None
        
        cmd = [
            'python', str(script_abs),
            '--seed', str(seed),
            '--output_dir', str(exp_output_dir),
            '--gpu', str(gpu_id) if '--gpu' in subprocess.run([str(script_abs), '--help'], capture_output=True, text=True).stdout else ''
        ]
        # 如果没有--gpu参数，移除它
        if '--gpu' not in subprocess.run([str(script_abs), '--help'], capture_output=True, text=True).stdout:
            cmd = [c for c in cmd if c != '--gpu' and c != str(gpu_id)]
        
        work_dir = script_abs.parent
    
    # 执行命令
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            log_f.write(f"Experiment: {exp_name}\n")
            log_f.write(f"Run: {run_idx}/5\n")
            log_f.write(f"Seed: {seed}\n")
            log_f.write(f"GPU: {gpu_id}\n")
            log_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"{'='*80}\n\n")
            log_f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(work_dir),
                env=os.environ.copy()
            )
            
            return True, log_file, exp_output_dir, process
                
    except Exception as e:
        print(f"❌ 运行 {exp_name} (Run {run_idx}, Seed {seed}) 时出错: {e}")
        return False, log_file, exp_output_dir, None

def worker_thread():
    """工作线程：从队列中取出任务并执行"""
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        task_id, exp_config, run_idx, seed = task
        
        try:
            success, log_path, result_path, process = run_single_experiment(
                exp_config, run_idx, seed
            )
            
            if process:
                with results_lock:
                    running_processes[task_id] = {
                        'process': process,
                        'exp_name': exp_config['name'],
                        'run_idx': run_idx,
                        'seed': seed,
                        'log_path': log_path,
                        'result_path': result_path,
                        'start_time': datetime.now()
                    }
                
                # 等待进程完成
                return_code = process.wait()
                
                with results_lock:
                    if task_id in running_processes:
                        del running_processes[task_id]
                    
                    results_summary.append({
                        'task_id': task_id,
                        'exp_name': exp_config['name'],
                        'run_idx': run_idx,
                        'seed': seed,
                        'success': return_code == 0,
                        'log_path': str(log_path) if log_path else None,
                        'result_path': str(result_path) if result_path else None
                    })
                
                if return_code == 0:
                    print(f"✅ {exp_config['name']} (Run {run_idx}, Seed {seed}) 完成！")
                else:
                    print(f"❌ {exp_config['name']} (Run {run_idx}, Seed {seed}) 失败！返回码: {return_code}")
        except Exception as e:
            print(f"❌ 任务 {task_id} 执行出错: {e}")
        
        task_queue.task_done()

def run_all_comparisons_parallel(max_parallel=2):
    """并行运行所有对比实验"""
    start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"🎯 开始执行所有对比实验（并行模式）")
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   实验数量: {len(COMPARISON_EXPERIMENTS)}")
    print(f"   每个实验运行次数: 5次")
    print(f"   总运行次数: {len(COMPARISON_EXPERIMENTS) * 5}")
    print(f"   最大并行数: {max_parallel}")
    print(f"{'='*80}\n")
    
    # 检查GPU显存
    gpus = get_gpu_memory_info()
    if gpus:
        print("📊 GPU显存状态:")
        for gpu in gpus:
            print(f"   GPU {gpu['index']}: {gpu['free_mb']}MB / {gpu['total_mb']}MB 可用 "
                  f"(使用率: {gpu['utilization']:.1f}%)")
        print()
    
    # 保存实验配置
    config_file = OUTPUT_DIR / 'experiment_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiments': COMPARISON_EXPERIMENTS,
            'start_time': start_time.isoformat(),
            'output_dir': str(OUTPUT_DIR),
            'log_dir': str(LOG_DIR),
            'max_parallel': max_parallel
        }, f, indent=2, ensure_ascii=False)
    
    # 创建所有任务
    task_id = 0
    for exp_config in COMPARISON_EXPERIMENTS:
        exp_name = exp_config['name']
        num_runs = exp_config['num_runs']
        seeds = exp_config['seeds']
        
        for run_idx in range(1, num_runs + 1):
            seed = seeds[run_idx - 1]
            task_queue.put((task_id, exp_config, run_idx, seed))
            task_id += 1
    
    print(f"📋 已创建 {task_id} 个任务\n")
    
    # 启动工作线程
    threads = []
    for i in range(max_parallel):
        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        threads.append(t)
        print(f"✅ 启动工作线程 {i+1}/{max_parallel}")
    
    print(f"\n🔄 等待所有任务完成...\n")
    
    # 等待所有任务完成
    task_queue.join()
    
    # 停止工作线程
    for _ in range(max_parallel):
        task_queue.put(None)
    
    for t in threads:
        t.join()
    
    # 保存结果汇总
    summary_file = OUTPUT_DIR / 'results_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'experiments': results_summary
        }, f, indent=2, ensure_ascii=False)
    
    # 打印最终汇总
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 所有对比实验执行完成！")
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   总耗时: {duration}")
    print(f"   结果汇总: {summary_file}")
    print(f"{'='*80}\n")
    
    # 统计成功率
    success_count = sum(1 for r in results_summary if r['success'])
    total_count = len(results_summary)
    print(f"📊 总体成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='并行运行所有对比实验')
    parser.add_argument('--max_parallel', type=int, default=2, 
                       help='最大并行数（默认2）')
    args = parser.parse_args()
    
    try:
        run_all_comparisons_parallel(max_parallel=args.max_parallel)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
