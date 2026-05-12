#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto Sequential Ablation Runner

需求：
- Baseline 跑完后，自动按顺序启动所有 w/o_* 消融实验
- 严格“一个跑完再跑下一个”
- 每个实验单独输出 nohup 日志
"""

import argparse
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util
import random


ROOT = Path(__file__).resolve().parents[1]  # experiments/exp_bio3.2

EXPERIMENT_ORDER = [
    # ========== 单模块消融实验（移除单个模块）==========
    "baseline",
    "w/o_vlm_retriever",  # 🔥 新增：禁用VLM Retriever
    "w/o_visual_notes",
    "w/o_adaptive_gating",
    "w/o_alignment_loss",
    "w/o_ot_loss",
    "w/o_dual_head",
    "w/o_cross_attn",
]

CONFIG_PATHS = {
    # 单模块消融实验
    "baseline": "ablation_studies/baseline/config.py",
    "w/o_vlm_retriever": "ablation_studies/w/o_vlm_retriever/config.py",  # 🔥 新增
    "w/o_visual_notes": "ablation_studies/w/o_visual_notes/config.py",
    "w/o_adaptive_gating": "ablation_studies/w/o_adaptive_gating/config.py",
    "w/o_alignment_loss": "ablation_studies/w/o_alignment_loss/config.py",
    "w/o_ot_loss": "ablation_studies/w/o_ot_loss/config.py",
    "w/o_dual_head": "ablation_studies/w/o_dual_head/config.py",
    "w/o_cross_attn": "ablation_studies/w/o_cross_attn/config.py",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_config_num_epochs(config_file: Path) -> int:
    """
    从配置文件加载 num_epochs。
    注意：配置文件里可能 import 了 BioCOT_v3_Config，必须只选择“该文件内定义的 Config 类”。
    """
    # 为了避免配置文件 import 路径问题（例如 `from config import BioCOT_v3_Config` 报错），
    # 这里不再执行 import，而是直接从文本解析 num_epochs。
    # 兼容两种写法：
    #   1) num_epochs: int = 20
    #   2) self.num_epochs = 20 （__post_init__ 内）
    try:
        txt = config_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 20

    # 优先解析“类字段定义”
    m = re.search(r"\bnum_epochs\s*:\s*int\s*=\s*(\d+)\b", txt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # 其次解析 __post_init__ 的强制赋值
    m = re.search(r"\bself\.num_epochs\s*=\s*(\d+)\b", txt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    return 20


def _latest_log_for_exp(exp: str) -> Path | None:
    log_dir = ROOT / "ablation_studies" / exp / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("nohup_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _is_exp_running(exp: str) -> bool:
    cfg = CONFIG_PATHS.get(exp)
    if not cfg:
        return False
    # 通过ps匹配config路径（兼容相对/绝对两种传参方式）
    pattern_abs = str((ROOT / cfg).as_posix())
    pattern_rel = cfg
    try:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                (
                    "ps aux | grep \"train_bio_cot_v3.2.py\" | grep -v grep | "
                    f"(grep -F \"{pattern_abs}\" || true) ; "
                    "ps aux | grep \"train_bio_cot_v3.2.py\" | grep -v grep | "
                    f"(grep -F \"{pattern_rel}\" || true)"
                ),
            ],
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def _is_exp_completed(exp: str) -> bool:
    cfg_path = ROOT / CONFIG_PATHS[exp]
    num_epochs = _load_config_num_epochs(cfg_path)
    log_path = _latest_log_for_exp(exp)
    if not log_path or not log_path.exists():
        return False

    # 判定策略：出现 “Epoch {num_epochs}/{num_epochs} - 验证阶段” 或 “训练完成”
    # 兼容中文日志
    done_markers = [
        "✅ 训练完成",
        "训练完成",
    ]
    epoch_val_re = re.compile(rf"Epoch\s+{num_epochs}/{num_epochs}\s+-\s+验证阶段")

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if any(m in content for m in done_markers):
            # 需要同时满足：日志标记完成 + 该实验进程已结束（避免误判导致并行/重复启动）
            return not _is_exp_running(exp)
        if epoch_val_re.search(content):
            return not _is_exp_running(exp)
    except Exception:
        return False

    return False


def _any_other_ablation_training_running(current_exp: str) -> bool:
    """
    严格串行控制：如果当前要启动的实验之外，还有任何 ablation_studies 的 train_bio_cot_v3.2.py 在跑，则返回True。
    这样不会打扰正在训练的进程，只是"等待合适的卡/合适的时机再启动下一步"。
    
    ⚠️ 重要：只检查 ablation_studies 目录下的训练，不会干扰正常训练（不在 ablation_studies 目录下的训练）。
    """
    current_cfg = CONFIG_PATHS.get(current_exp)
    if not current_cfg:
        return False

    current_abs = str((ROOT / current_cfg).as_posix())
    current_rel = current_cfg

    try:
        out = subprocess.check_output(
            ["bash", "-lc", "ps aux | grep \"train_bio_cot_v3.2.py\" | grep -v grep || true"],
            text=True,
        )
    except Exception:
        return False

    for line in out.splitlines():
        # 🔥 关键：只检查 ablation_studies 目录下的训练，忽略正常训练
        if "ablation_studies" not in line:
            continue  # 跳过非消融实验的训练进程
        # 不是当前实验的训练进程，则认为"有其它消融实验在跑"
        if (current_abs not in line) and (current_rel not in line):
            return True
    return False


def _start_exp(exp: str, physical_gpu: int) -> tuple[int, Path]:
    cfg_path = ROOT / CONFIG_PATHS[exp]
    log_dir = ROOT / "ablation_studies" / exp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"nohup_{exp.replace('/', '_')}_{ts}.log"

    # 🔥 使用指定的虚拟环境 Python（确保使用正确的依赖）
    venv_python = ROOT.parents[1] / "my_retfound" / "bin" / "python"
    if not venv_python.exists():
        # 如果指定路径不存在，回退到 sys.executable
        venv_python = Path(sys.executable)
    py = venv_python
    train_script = (ROOT / "training" / "train_bio_cot_v3.2.py").resolve()

    cmd = [
        str(py),
        "-u",
        str(train_script),
        "--config",
        str(cfg_path),
        "--gpu",
        "0",
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        # 关键：使用 CUDA_VISIBLE_DEVICES 将“物理GPU号”映射为 torch 的 cuda:0
        # 这样即使集群/环境设置了 CUDA_VISIBLE_DEVICES，也不会出现 invalid device ordinal
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)

    return p.pid, log_path


def _query_gpus() -> list[dict]:
    """
    返回GPU状态列表：
    [
      { 'index': 0, 'util': 12, 'mem_used': 6806, 'mem_total': 49140, 'mem_free': 42334 },
      ...
    ]
    """
    # 这里用 CUDA_VISIBLE_DEVICES 过滤后的“可见GPU”来选卡，避免选到不可见的卡
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_set = None
    if visible:
        try:
            visible_set = {int(x.strip()) for x in visible.split(",") if x.strip() != ""}
        except Exception:
            visible_set = None

    try:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    gpus: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, used, total = parts
        try:
            idx_i = int(idx)
            util_i = int(util)
            used_i = int(used)
            total_i = int(total)
        except ValueError:
            continue
        if visible_set is not None and idx_i not in visible_set:
            continue
        gpus.append(
            {
                "index": idx_i,
                "util": util_i,
                "mem_used": used_i,
                "mem_total": total_i,
                "mem_free": max(total_i - used_i, 0),
            }
        )
    return gpus


def _pick_gpu(
    allowed_gpus: list[int] | None,
    min_free_mem_mb: int,
    max_util: int,
    strategy: str = "random",
) -> dict | None:
    """
    从可用GPU中选择一个。
    - allowed_gpus: 允许使用的GPU列表；None表示全部
    - min_free_mem_mb: 最小空闲显存阈值（MB）
    - max_util: 最大利用率阈值（%）
    - strategy: random | most_free | least_util
    """
    gpus = _query_gpus()
    if not gpus:
        return None

    if allowed_gpus is not None:
        gpus = [g for g in gpus if g["index"] in allowed_gpus]

    eligible = [g for g in gpus if g["mem_free"] >= min_free_mem_mb and g["util"] <= max_util]
    if not eligible:
        return None

    if strategy == "most_free":
        eligible.sort(key=lambda x: (x["mem_free"], -x["util"]), reverse=True)
        return eligible[0]
    if strategy == "least_util":
        eligible.sort(key=lambda x: (x["util"], -x["mem_free"]))
        return eligible[0]

    # default: random（在eligible中随机，做到“灵活变通/随机调整”）
    return random.choice(eligible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="固定GPU（当不开启auto_gpu时使用）")
    parser.add_argument(
        "--auto_gpu",
        action="store_true",
        help="自动选择空闲GPU（推荐：满足显存/利用率阈值后随机挑选）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="允许使用的GPU列表，例如 '0,1'；留空表示全部",
    )
    parser.add_argument(
        "--min_free_mem_mb",
        type=int,
        default=12000,
        help="自动选卡时的最小空闲显存阈值(MB)，默认12000",
    )
    parser.add_argument(
        "--max_util",
        type=int,
        default=30,
        help="自动选卡时的最大利用率阈值(%)，默认30",
    )
    parser.add_argument(
        "--gpu_strategy",
        type=str,
        default="random",
        choices=["random", "most_free", "least_util"],
        help="选卡策略：random/most_free/least_util",
    )
    parser.add_argument("--interval", type=int, default=120, help="检查间隔(秒)")
    parser.add_argument("--start_from", type=str, default="baseline", help="从哪个实验开始（默认baseline）")
    args = parser.parse_args()

    if args.start_from not in EXPERIMENT_ORDER:
        raise ValueError(f"start_from 必须是 {EXPERIMENT_ORDER} 之一")

    start_idx = EXPERIMENT_ORDER.index(args.start_from)
    plan = EXPERIMENT_ORDER[start_idx:]

    print("=" * 80)
    print("🤖 Auto Sequential Ablation Runner")
    print(f"Root: {ROOT}")
    if args.auto_gpu:
        print("GPU: AUTO")
        print(f"Allowed GPUs: {args.gpus if args.gpus else 'ALL'}")
        print(f"GPU thresholds: free>={args.min_free_mem_mb}MB, util<={args.max_util}%")
        print(f"GPU strategy: {args.gpu_strategy}")
    else:
        print(f"GPU: {args.gpu}")
    print(f"Interval: {args.interval}s")
    print("Plan:")
    for i, exp in enumerate(plan, 1):
        print(f"  [{i}/{len(plan)}] {exp} -> {CONFIG_PATHS[exp]}")
    print("=" * 80, flush=True)

    try:
        for exp in plan:
            cfg_path = ROOT / CONFIG_PATHS[exp]
            if not cfg_path.exists():
                print(f"[{_now()}] ❌ 配置文件不存在: {cfg_path}", flush=True)
                break

            # 若已完成，直接跳过
            if _is_exp_completed(exp):
                print(f"[{_now()}] ✅ 已完成，跳过: {exp}", flush=True)
                continue

            # 若正在跑，等待它完成
            if _is_exp_running(exp):
                log_path = _latest_log_for_exp(exp)
                print(f"[{_now()}] ⏳ 检测到正在运行: {exp} | log={log_path}", flush=True)
            else:
                # 严格串行：若还有其它消融训练在跑，则等待（不打扰正在训练的进程）
                while _any_other_ablation_training_running(exp):
                    print(
                        f"[{_now()}] 🕒 检测到其它消融训练仍在运行，等待 {args.interval}s 后再尝试启动: {exp}",
                        flush=True,
                    )
                    time.sleep(args.interval)

                # 自动挑选GPU（不打扰当前训练；只在“启动下一个实验”时选卡）
                chosen_gpu = args.gpu
                if args.auto_gpu:
                    allowed = None
                    if args.gpus.strip():
                        allowed = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]

                    while True:
                        g = _pick_gpu(
                            allowed_gpus=allowed,
                            min_free_mem_mb=args.min_free_mem_mb,
                            max_util=args.max_util,
                            strategy=args.gpu_strategy,
                        )
                        if g is not None:
                            chosen_gpu = g["index"]
                            print(
                                f"[{_now()}] 🟢 选择GPU={chosen_gpu} "
                                f"(free={g['mem_free']}MB, util={g['util']}%)",
                                flush=True,
                            )
                            break
                        print(
                            f"[{_now()}] 🕒 暂无满足条件的GPU，等待 {args.interval}s 后重试 "
                            f"(free>={args.min_free_mem_mb}MB, util<={args.max_util}%)",
                            flush=True,
                        )
                        time.sleep(args.interval)

                pid, log_path = _start_exp(exp, chosen_gpu)
                print(
                    f"[{_now()}] 🚀 启动实验: {exp} | physical_gpu={chosen_gpu} (CUDA_VISIBLE_DEVICES={chosen_gpu} -> cuda:0) "
                    f"| PID={pid} | log={log_path}",
                    flush=True,
                )

            # 等待完成（带心跳，避免“无声挂掉”）
            last_heartbeat = time.time()
            while True:
                if _is_exp_completed(exp):
                    print(f"[{_now()}] ✅ 实验完成: {exp}", flush=True)
                    break
                now = time.time()
                if now - last_heartbeat >= max(args.interval, 60):
                    print(f"[{_now()}] 💓 仍在等待实验完成: {exp}", flush=True)
                    last_heartbeat = now
                time.sleep(args.interval)

        print(f"[{_now()}] 🎉 计划执行结束", flush=True)
    except Exception:
        import traceback
        print(f"[{_now()}] ❌ 调度器异常退出：", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()


