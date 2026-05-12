#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计分析工具：用于实验统计检验和结果汇总
- McNemar检验：比较两个方法的分类结果
- Wilcoxon符号秩检验：比较AUC分布
- 置信区间计算：95% CI for AUC
- 结果汇总表格生成：自动生成CSV表格
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from scipy.stats import bootstrap
import argparse

def load_results_from_json(json_path: Path) -> Dict:
    """从JSON文件加载实验结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间
    
    Args:
        data: 数据数组
        confidence: 置信水平（默认0.95）
    
    Returns:
        (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    n = len(data)
    
    if n < 2:
        return (data[0], data[0])
    
    # 使用t分布（小样本）或正态分布（大样本）
    if n < 30:
        # t分布
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        margin = t_critical * std_err
    else:
        # 正态分布
        z_critical = stats.norm.ppf(1 - alpha/2)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        margin = z_critical * std_err
    
    mean = np.mean(data)
    return (mean - margin, mean + margin)

def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict:
    """
    McNemar检验：比较两个方法的分类结果
    
    Args:
        y_true: 真实标签
        y_pred1: 方法1的预测
        y_pred2: 方法2的预测
    
    Returns:
        包含p值、统计量等的字典
    """
    # 构建2x2列联表
    # a: 两个方法都正确
    # b: 方法1正确，方法2错误
    # c: 方法1错误，方法2正确
    # d: 两个方法都错误
    
    a = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    d = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    # McNemar统计量（使用连续性校正）
    if b + c > 0:
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        p_value = 1.0
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'contingency_table': {
            'both_correct': int(a),
            'method1_correct_method2_wrong': int(b),
            'method1_wrong_method2_correct': int(c),
            'both_wrong': int(d)
        }
    }

def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
    """
    Wilcoxon符号秩检验：比较两个方法的AUC分布
    
    Args:
        data1: 方法1的AUC值数组
        data2: 方法2的AUC值数组
    
    Returns:
        包含p值、统计量等的字典
    """
    if len(data1) != len(data2):
        raise ValueError("两个数组长度必须相同")
    
    # 计算差值
    diff = data1 - data2
    
    # 执行Wilcoxon检验
    statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff)
    }

def aggregate_results(results_dir: Path, experiment_name: str, num_runs: int = 5) -> Dict:
    """
    聚合多次运行的结果
    
    Args:
        results_dir: 结果目录
        experiment_name: 实验名称
        num_runs: 运行次数
    
    Returns:
        聚合后的结果字典
    """
    aucs = []
    accs = []
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for seed in [42, 123, 456, 789, 2024][:num_runs]:
        # 尝试不同的路径模式
        possible_paths = [
            results_dir / f"seed_{seed}" / "training_history.json",
            results_dir / f"run_{seed}" / "training_history.json",
            results_dir / f"training_history_seed_{seed}.json",
            results_dir / "logs" / f"training_history_*_seed_{seed}.json",
        ]
        
        # 也尝试直接查找所有training_history文件
        history_files = list(results_dir.rglob("training_history*.json"))
        
        if not history_files:
            print(f"⚠️ 未找到 {experiment_name} seed {seed} 的结果文件")
            continue
        
        # 使用第一个找到的文件
        history_file = history_files[0]
        
        try:
            history = load_results_from_json(history_file)
            
            # 提取最佳验证结果
            if 'val_auc' in history:
                best_idx = np.argmax(history['val_auc'])
                aucs.append(history['val_auc'][best_idx])
                accs.append(history['val_acc'][best_idx] if 'val_acc' in history else 0.0)
                sensitivities.append(history['val_sensitivity'][best_idx] if 'val_sensitivity' in history else 0.0)
                specificities.append(history['val_specificity'][best_idx] if 'val_specificity' in history else 0.0)
                f1_scores.append(history['val_f1'][best_idx] if 'val_f1' in history else 0.0)
        except Exception as e:
            print(f"⚠️ 读取 {history_file} 失败: {e}")
            continue
    
    if not aucs:
        return None
    
    # 计算统计量
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    auc_ci = calculate_confidence_interval(np.array(aucs))
    
    return {
        'experiment_name': experiment_name,
        'num_runs': len(aucs),
        'auc': {
            'mean': auc_mean,
            'std': auc_std,
            'ci_lower': auc_ci[0],
            'ci_upper': auc_ci[1],
            'values': aucs
        },
        'accuracy': {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'values': accs
        },
        'sensitivity': {
            'mean': np.mean(sensitivities),
            'std': np.std(sensitivities),
            'values': sensitivities
        },
        'specificity': {
            'mean': np.mean(specificities),
            'std': np.std(specificities),
            'values': specificities
        },
        'f1_score': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores),
            'values': f1_scores
        }
    }

def generate_comparison_table(comparison_results: List[Dict], output_path: Path):
    """生成对比实验CSV汇总表。"""
    rows = []
    for result in comparison_results:
        name = result['experiment_name'].replace('_', ' ').title()
        auc_val = result['auc']['mean']
        auc_std = result['auc']['std']
        sens = result['sensitivity']['mean']
        spec = result['specificity']['mean']
        f1 = result['f1_score']['mean']
        rows.append({
            "method": name,
            "internal_val_auc_mean": auc_val,
            "internal_val_auc_std": auc_std,
            "sensitivity": sens,
            "specificity": spec,
            "f1_score": f1,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"CSV汇总表已保存到: {output_path}")

def generate_ablation_table(ablation_results: List[Dict], full_model_result: Dict, output_path: Path):
    """生成消融实验CSV汇总表。"""
    full_auc = full_model_result['auc']['mean']
    rows = [{
        "method": "Bio-COT 3.2 (Full)",
        "internal_val_auc_mean": full_auc,
        "internal_val_auc_std": full_model_result['auc']['std'],
        "delta_auc_vs_full": 0.0,
    }]

    for result in ablation_results:
        name = result['experiment_name'].replace('w/o_', '').replace('_', ' ').title()
        auc_val = result['auc']['mean']
        auc_std = result['auc']['std']
        delta = auc_val - full_auc
        rows.append({
            "method": f"w/o {name}",
            "internal_val_auc_mean": auc_val,
            "internal_val_auc_std": auc_std,
            "delta_auc_vs_full": delta,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"CSV汇总表已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='统计分析工具')
    parser.add_argument('--comparison_results', type=str, help='对比实验结果目录')
    parser.add_argument('--ablation_results', type=str, help='消融实验结果目录')
    parser.add_argument('--output_dir', type=str, default='analysis/statistical_results', help='输出目录')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 聚合对比实验结果
    comparison_results = []
    if args.comparison_results:
        comparison_dir = Path(args.comparison_results)
        for method_dir in comparison_dir.iterdir():
            if method_dir.is_dir():
                result = aggregate_results(method_dir, method_dir.name)
                if result:
                    comparison_results.append(result)
        
        if comparison_results:
            generate_comparison_table(comparison_results, output_dir / 'comparison_table.csv')
    
    # 聚合消融实验结果
    ablation_results = []
    full_model_result = None
    
    if args.ablation_results:
        ablation_dir = Path(args.ablation_results)
        
        # 查找完整模型结果
        full_model_paths = [
            ablation_dir / 'baseline',
            ablation_dir / 'full_model',
        ]
        
        for path in full_model_paths:
            if path.exists():
                result = aggregate_results(path, 'full_model')
                if result:
                    full_model_result = result
                    break
        
        # 查找消融实验
        for exp_dir in ablation_dir.rglob('w/o_*'):
            if exp_dir.is_dir():
                result = aggregate_results(exp_dir, exp_dir.name)
                if result:
                    ablation_results.append(result)
        
        if ablation_results and full_model_result:
            generate_ablation_table(ablation_results, full_model_result, output_dir / 'ablation_table.csv')
    
    print("\n✅ 统计分析完成！")
    print(f"   结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
