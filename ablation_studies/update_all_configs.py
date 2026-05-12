#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量更新所有消融实验配置，添加5.0优势"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
configs = {
    "baseline": "baseline/config.py",
    "w/o_visual_notes": "w/o_visual_notes/config.py",
    "w/o_adaptive_gating": "w/o_adaptive_gating/config.py",
    "w/o_alignment_loss": "w/o_alignment_loss/config.py",
    "w/o_ot_loss": "w/o_ot_loss/config.py",
    "w/o_dual_head": "w/o_dual_head/config.py",
    "w/o_cross_attn": "w/o_cross_attn/config.py",
}

# 5.0优势配置代码
v5_features = """
    # 🔥 5.0优势：默认启用所有5.0特性
    use_hierarchical: bool = True  # 分层多尺度特征提取
    use_noise_aware: bool = True  # 噪声感知融合
    use_clinical_evolver: bool = True  # 动态临床查询演化
    use_text_adapter: bool = True  # Text Adapter
    dropout_rate: float = 0.4  # 激进正则化
    drop_path_rate: float = 0.2  # DropPath
    lambda_ortho: float = 0.5  # 正交损失权重
    lambda_noise: float = 0.1  # 噪声正则化损失权重
"""

for exp_name, config_path in configs.items():
    config_file = ROOT / config_path
    if not config_file.exists():
        print(f"⚠️ 配置文件不存在: {config_file}")
        continue
    
    content = config_file.read_text(encoding='utf-8')
    
    # 检查是否已包含5.0配置
    if 'use_hierarchical' in content:
        print(f"✅ {exp_name}: 已包含5.0配置")
        continue
    
    # 在__post_init__之前插入5.0配置
    if 'def __post_init__(self):' in content:
        content = content.replace(
            'def __post_init__(self):',
            v5_features + '\n    def __post_init__(self):'
        )
        config_file.write_text(content, encoding='utf-8')
        print(f"✅ {exp_name}: 已添加5.0配置")
    else:
        print(f"⚠️ {exp_name}: 无法找到插入位置")

print("\n🎉 所有配置更新完成！")
