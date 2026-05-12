#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动执行所有对比实验（Comparison Experiments）
用于MICCAI论文的Table 1
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
import json
import signal

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
        'config': None,  # 使用默认config
        'description': 'Bio-COT 3.2 (Full) - Our proposed method',
        'category': 'ours',
        'num_runs': 5,  # 运行5次（不同随机种子）
        'seeds': [42, 123, 456, 789, 2024]
    },
    {
        'name': 'MedCLIP',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'medclip' / 'train_medclip.py'),
        'config': None,
        'description': 'MedCLIP - Medical domain-specific CLIP model',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024]
    },
    {
        'name': 'ConVIRT',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'convirt' / 'train_convirt.py'),
        'config': None,
        'description': 'ConVIRT - Contrastive Vision-Representation Transformer',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024]
    },
    {
        'name': 'mmFormer',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'mmformer' / 'train_mmformer.py'),
        'config': None,
        'description': 'mmFormer - Multi-modal Medical Transformer',
        'category': 'sota',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024]
    },
    {
        'name': 'Swin-T_Fusion',
        'script': str(EXP_3_0_ROOT / 'comparison_experiments' / 'baselines' / 'swin_t_baseline' / 'train_swin.py'),
        'config': None,
        'description': 'Swin-T + Fusion - Swin-T encoder with simple fusion',
        'category': 'baseline',
        'num_runs': 5,
        'seeds': [42, 123, 456, 789, 2024]
    },
]

# 创建输出目录
OUTPUT_DIR = EXP_ROOT / 'comparison_experiments' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = EXP_ROOT / 'comparison_experiments' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量：用于信号处理
current_process = None
running_experiments = []

def signal_handler(sig, frame):
    """处理中断信号"""
    global current_process
    print("\n\n⚠️ 收到中断信号，正在安全退出...")
    if current_process:
        print(f"   正在终止当前实验进程...")
        current_process.terminate()
        try:
            current_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            current_process.kill()
    print("✅ 已安全退出")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_single_experiment(exp_config, run_idx, seed):
    """
    运行单个实验（一次运行）
    
    Args:
        exp_config: 实验配置字典
        run_idx: 运行索引（1-5）
        seed: 随机种子
    
    Returns:
        (success, log_path, result_path)
    """
    global current_process
    
    exp_name = exp_config['name']
    script_path = exp_config['script']
    description = exp_config['description']
    
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
    print(f"{'='*80}\n")
    
    # 构建命令
    if exp_name == 'Bio-COT_3.2_Full':
        # Bio-COT 3.2使用训练脚本
        # 注意：train_bio_cot_v3.2.py 使用 --config 参数，需要创建临时配置文件
        import importlib.util
        
        # 创建临时配置文件（基于默认config，但修改输出目录）
        temp_config_file = exp_output_dir / 'config_temp.py'
        temp_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取默认config并修改
        default_config_path = EXP_ROOT / 'config.py'
        if default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 在文件末尾添加配置修改代码
            config_content += f"""

# ============================================================
# 自动生成的临时配置（Run {run_idx}, Seed {seed}）
# ============================================================
import os
import random
import numpy as np
import torch

# 设置随机种子（在创建config实例之前）
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
        ]
        work_dir = EXP_ROOT
    else:
        # 其他方法使用exp_bio3.0_improved中的脚本
        script_abs = Path(script_path)
        if not script_abs.exists():
            print(f"❌ 错误: 脚本不存在: {script_path}")
            return False, None, None
        
        cmd = [
            'python', str(script_abs),
            '--seed', str(seed),
            '--output_dir', str(exp_output_dir),
        ]
        work_dir = script_abs.parent
    
    # 执行命令
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            log_f.write(f"Experiment: {exp_name}\n")
            log_f.write(f"Run: {run_idx}/5\n")
            log_f.write(f"Seed: {seed}\n")
            log_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"{'='*80}\n\n")
            log_f.flush()
            
            current_process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(work_dir),
                env=os.environ.copy()
            )
            
            # 等待完成
            return_code = current_process.wait()
            current_process = None
            
            log_f.write(f"\n{'='*80}\n")
            log_f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Return Code: {return_code}\n")
            
            if return_code == 0:
                print(f"✅ {exp_name} (Run {run_idx}, Seed {seed}) 完成！")
                return True, log_file, exp_output_dir
            else:
                print(f"❌ {exp_name} (Run {run_idx}, Seed {seed}) 失败！返回码: {return_code}")
                return False, log_file, exp_output_dir
                
    except Exception as e:
        print(f"❌ 运行 {exp_name} (Run {run_idx}, Seed {seed}) 时出错: {e}")
        if current_process:
            current_process.terminate()
            current_process = None
        return False, log_file, exp_output_dir

def run_all_comparisons():
    """运行所有对比实验"""
    global running_experiments
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"🎯 开始执行所有对比实验")
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   实验数量: {len(COMPARISON_EXPERIMENTS)}")
    print(f"   每个实验运行次数: 5次")
    print(f"   总运行次数: {len(COMPARISON_EXPERIMENTS) * 5}")
    print(f"{'='*80}\n")
    
    # 保存实验配置
    config_file = OUTPUT_DIR / 'experiment_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiments': COMPARISON_EXPERIMENTS,
            'start_time': start_time.isoformat(),
            'output_dir': str(OUTPUT_DIR),
            'log_dir': str(LOG_DIR)
        }, f, indent=2, ensure_ascii=False)
    
    # 运行每个实验
    results_summary = []
    
    for exp_idx, exp_config in enumerate(COMPARISON_EXPERIMENTS, 1):
        exp_name = exp_config['name']
        num_runs = exp_config['num_runs']
        seeds = exp_config['seeds']
        
        print(f"\n{'#'*80}")
        print(f"# 实验 {exp_idx}/{len(COMPARISON_EXPERIMENTS)}: {exp_name}")
        print(f"{'#'*80}\n")
        
        exp_results = {
            'name': exp_name,
            'description': exp_config['description'],
            'category': exp_config['category'],
            'runs': []
        }
        
        # 运行5次
        for run_idx in range(1, num_runs + 1):
            seed = seeds[run_idx - 1]
            success, log_path, result_path = run_single_experiment(exp_config, run_idx, seed)
            
            exp_results['runs'].append({
                'run_idx': run_idx,
                'seed': seed,
                'success': success,
                'log_path': str(log_path) if log_path else None,
                'result_path': str(result_path) if result_path else None
            })
            
            # 如果失败，等待一下再继续
            if not success:
                print(f"⚠️ 等待5秒后继续下一个运行...")
                time.sleep(5)
        
        # 统计成功率
        success_count = sum(1 for r in exp_results['runs'] if r['success'])
        exp_results['success_rate'] = f"{success_count}/{num_runs}"
        
        results_summary.append(exp_results)
        
        print(f"\n📊 {exp_name} 完成情况: {success_count}/{num_runs} 成功")
    
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
    
    # 打印每个实验的成功率
    print("📊 实验结果汇总:")
    for exp_result in results_summary:
        print(f"   {exp_result['name']}: {exp_result['success_rate']} 成功")
    
    print(f"\n✅ 所有日志保存在: {LOG_DIR}")
    print(f"✅ 所有结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        run_all_comparisons()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        signal_handler(None, None)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
