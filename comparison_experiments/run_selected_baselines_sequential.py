#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按顺序依次运行 MedCLIP / ConVIRT / mmFormer / Swin-T Baseline

- 使用 exp_bio3.0_improved 中已经验证过的训练脚本
- 每个方法调用一次，对应脚本内部会运行 `num_runs` 次（默认=5）
- 输出目录定向到当前实验目录 comparison_experiments/results 下对应文件夹
- 日志输出到当前实验目录 comparison_experiments/logs
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime

# 当前文件路径: .../experiments/exp_infofusion_2026/comparison_experiments/run_selected_baselines_sequential.py
EXP_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = EXP_ROOT.parent
EXP3_ROOT = EXPERIMENTS_ROOT / 'exp_bio3.0_improved'
THIS_DIR = Path(__file__).resolve().parent

LOG_DIR = THIS_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        'name': 'MedCLIP',
        'script': EXP3_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'medclip' / 'train_medclip.py',
        'experiment_name': 'baseline_medclip_v3_2',
        'output_dir': EXP_ROOT / 'comparison_experiments' / 'results' / 'MedCLIP',
        'num_runs': 5,
    },
    {
        'name': 'ConVIRT',
        'script': EXP3_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'convirt' / 'train_convirt.py',
        'experiment_name': 'baseline_convirt_v3_2',
        'output_dir': EXP_ROOT / 'comparison_experiments' / 'results' / 'ConVIRT',
        'num_runs': 5,
    },
    {
        'name': 'mmFormer',
        'script': EXP3_ROOT / 'comparison_experiments' / 'baselines' / 'sota_baselines' / 'mmformer' / 'train_mmformer.py',
        'experiment_name': 'baseline_mmformer_v3_2',
        'output_dir': EXP_ROOT / 'comparison_experiments' / 'results' / 'mmFormer',
        'num_runs': 5,
    },
    {
        'name': 'Swin-T_Fusion',
        'script': EXP3_ROOT / 'comparison_experiments' / 'baselines' / 'swin_t_baseline' / 'train_swin.py',
        'experiment_name': 'baseline_swin_t_v3_2',
        'output_dir': EXP_ROOT / 'comparison_experiments' / 'results' / 'Swin-T_Fusion',
        'num_runs': 5,
    },
]


def run_single(exp_cfg):
    name = exp_cfg['name']
    script = exp_cfg['script']
    exp_name = exp_cfg['experiment_name']
    output_dir = Path(exp_cfg['output_dir'])
    num_runs = exp_cfg['num_runs']

    if not script.exists():
        print(f"❌ 跳过 {name}: 脚本不存在: {script}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'{name}_sequential_{timestamp}.log'

    cmd = [
        sys.executable,
        str(script),
        '--experiment_name', exp_name,
        '--num_runs', str(num_runs),
        '--output_dir', str(output_dir),
        # data_root 使用脚本默认值（/data2/hmy/5Center_datas/...），如需LCO数据可在此处覆盖
    ]

    print("=" * 80)
    print(f"🚀 开始运行对比实验: {name}")
    print(f"   脚本: {script}")
    print(f"   输出目录: {output_dir}")
    print(f"   日志: {log_path}")
    print("=" * 80)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment: {name}\n")
        f.write(f"Script: {script}\n")
        f.write(f"Output dir: {output_dir}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(script.parent),
            env=os.environ.copy(),
        )
        ret = proc.wait()

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Return code: {ret}\n")

    if ret == 0:
        print(f"✅ {name} 完成 (日志: {log_path})")
        return True
    else:
        print(f"❌ {name} 失败 (返回码: {ret})，请查看日志: {log_path}")
        return False


def main():
    print("\n===== 按顺序依次运行 ConVIRT / mmFormer / Swin-T 对比实验 =====\n")
    print("ℹ️  跳过 MedCLIP（已在运行中）\n")
    
    # 跳过 MedCLIP，只运行其他三个
    experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] != 'MedCLIP']
    
    for exp in experiments_to_run:
        ok = run_single(exp)
        if not ok:
            # 出现错误时不中断后续实验，只是提示
            print(f"⚠️ 警告: {exp['name']} 运行失败，将继续尝试后续实验。")

    print("\n🎉 所有指定的对比实验已按顺序尝试执行完毕 (请检查各自日志)。\n")


if __name__ == '__main__':
    main()
