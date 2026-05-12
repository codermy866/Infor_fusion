#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新运行失败的消融实验
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # experiments/exp_bio3.2

# 需要重新运行的实验
FAILED_EXPERIMENTS = [
    "w/o_visual_notes",  # 训练失败（image_names长度不匹配，AUC=0.0000）
]

CONFIG_PATHS = {
    "w/o_visual_notes": "ablation_studies/w/o_visual_notes/config.py",
}

def run_experiment(exp_name, gpu=0):
    """运行单个实验"""
    config_path = ROOT / CONFIG_PATHS[exp_name]
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    log_dir = ROOT / "ablation_studies" / exp_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
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

if __name__ == "__main__":
    import os
    import time
    
    print("=" * 80)
    print("🔄 重新运行失败的消融实验")
    print("=" * 80)
    
    for exp in FAILED_EXPERIMENTS:
        print(f"\n处理实验: {exp}")
        pid, log_path = run_experiment(exp, gpu=0)
        
        # 等待完成
        print(f"等待实验完成...")
        while True:
            try:
                # 检查进程是否还在运行
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"✅ 实验完成: {exp}")
                    break
                
                # 检查日志中是否有完成标记
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "✅ 训练完成" in content or "训练完成" in content:
                            print(f"✅ 实验完成: {exp}")
                            break
                
                time.sleep(60)  # 每分钟检查一次
            except KeyboardInterrupt:
                print(f"\n⚠️ 用户中断，停止等待")
                break
            except Exception as e:
                print(f"⚠️ 检查时出错: {e}")
                time.sleep(60)
    
    print("\n🎉 所有实验处理完成")
