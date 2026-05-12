#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新运行未完成的消融实验
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]  # experiments/exp_bio3.2

CONFIG_PATHS = {
    "baseline": "ablation_studies/baseline/config.py",
    "w/o_visual_notes": "ablation_studies/w/o_visual_notes/config.py",
    "w/o_adaptive_gating": "ablation_studies/w/o_adaptive_gating/config.py",
    "w/o_alignment_loss": "ablation_studies/w/o_alignment_loss/config.py",
    "w/o_ot_loss": "ablation_studies/w/o_ot_loss/config.py",
    "w/o_dual_head": "ablation_studies/w/o_dual_head/config.py",
    "w/o_cross_attn": "ablation_studies/w/o_cross_attn/config.py",
}

def check_experiment_completed(exp_name):
    """检查实验是否完成"""
    exp_dir = ROOT / "ablation_studies" / exp_name
    history_files = list(exp_dir.glob("logs/training_history_*.json"))
    
    if not history_files:
        return False, "无训练历史文件"
    
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epochs = len(data.get('train_loss', []))
        best_auc = data.get('best_auc', 0.0)
        
        # 检查是否完成20个epochs且AUC > 0
        if epochs >= 20 and best_auc > 0.0:
            return True, f"已完成 ({epochs} epochs, AUC={best_auc:.4f})"
        else:
            return False, f"未完成 ({epochs} epochs, AUC={best_auc:.4f})"
    except Exception as e:
        return False, f"读取失败: {e}"

def run_experiment(exp_name, gpu=0):
    """运行单个实验"""
    config_path = ROOT / CONFIG_PATHS[exp_name]
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None, None
    
    log_dir = ROOT / "ablation_studies" / exp_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"nohup_{exp_name.replace('/', '_')}_{ts}.log"
    
    # 使用虚拟环境Python
    venv_python = ROOT.parents[1] / "my_retfound" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)
    
    train_script = ROOT / "training" / "train_bio_cot_v3.2.py"
    
    cmd = [
        str(venv_python),
        "-u",
        str(train_script),
        "--config",
        str(config_path),
        "--gpu",
        str(gpu),
    ]
    
    print(f"🚀 启动实验: {exp_name}")
    print(f"   命令: {' '.join(cmd)}")
    print(f"   日志: {log_path}")
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env
        )
    
    print(f"   PID: {process.pid}")
    return process.pid, log_path

def wait_for_completion(exp_name, pid, log_path, timeout_hours=24):
    """等待实验完成"""
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600
    
    print(f"等待实验完成: {exp_name} (PID: {pid})")
    
    while True:
        # 检查超时
        if time.time() - start_time > timeout_seconds:
            print(f"⚠️ 超时 ({timeout_hours}小时)，停止等待")
            return False
        
        # 检查进程是否还在运行
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"✅ 进程已结束: {exp_name}")
                break
        except Exception:
            pass
        
        # 检查日志中是否有完成标记
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "✅ 训练完成" in content or "训练完成" in content:
                        print(f"✅ 实验完成: {exp_name}")
                        return True
                    # 检查是否有错误
                    if "ValueError" in content or "Exception" in content:
                        last_lines = content.split('\n')[-20:]
                        error_lines = [l for l in last_lines if 'Error' in l or 'Exception' in l]
                        if error_lines:
                            print(f"⚠️ 检测到错误: {exp_name}")
                            print(f"   错误信息: {error_lines[-1]}")
            except Exception as e:
                pass
        
        time.sleep(60)  # 每分钟检查一次
    
    # 最终检查实验是否真的完成
    is_completed, status = check_experiment_completed(exp_name)
    if is_completed:
        print(f"✅ 实验成功完成: {exp_name} - {status}")
        return True
    else:
        print(f"⚠️ 实验可能未完成: {exp_name} - {status}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🔄 重新运行未完成的消融实验")
    print("=" * 80)
    
    # 检查所有实验
    incomplete_experiments = []
    for exp_name in CONFIG_PATHS.keys():
        is_completed, status = check_experiment_completed(exp_name)
        print(f"{exp_name}: {status}")
        if not is_completed:
            incomplete_experiments.append(exp_name)
    
    if not incomplete_experiments:
        print("\n🎉 所有实验都已完成！")
        sys.exit(0)
    
    print(f"\n需要重新运行的实验 ({len(incomplete_experiments)}个):")
    for exp in incomplete_experiments:
        print(f"  - {exp}")
    
    # 按顺序运行
    for i, exp in enumerate(incomplete_experiments, 1):
        print(f"\n{'='*80}")
        print(f"处理实验 {i}/{len(incomplete_experiments)}: {exp}")
        print(f"{'='*80}")
        
        pid, log_path = run_experiment(exp, gpu=0)
        if pid is None:
            print(f"❌ 无法启动实验: {exp}")
            continue
        
        # 等待完成
        success = wait_for_completion(exp, pid, log_path)
        
        if not success:
            print(f"⚠️ 实验可能未成功完成: {exp}")
        
        # 等待一段时间再启动下一个实验
        if i < len(incomplete_experiments):
            print(f"\n等待30秒后启动下一个实验...")
            time.sleep(30)
    
    print("\n" + "=" * 80)
    print("🎉 所有实验处理完成")
    print("=" * 80)

