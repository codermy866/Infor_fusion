#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5中心实验专用可视化工具模块
所有可视化函数都在此文件中，确保exp_5centers文件夹完全独立
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("⚠️ seaborn未安装，部分图表将使用matplotlib绘制")

from sklearn.metrics import roc_curve, auc
from scipy import stats
from pathlib import Path


def plot_training_curves(history, output_dir, timestamp):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 训练和验证损失
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练和验证准确率
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[0, 2].plot(epochs, history['val_auc'], label='Val AUC', color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('Validation AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 损失组件
    axes[1, 0].plot(history['train_ot_loss'], label='OT Loss')
    axes[1, 0].plot(history['train_consist_loss'], label='Consist Loss')
    axes[1, 0].plot(history['train_adv_loss'], label='Adv Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(history['val_f1'], label='Val F1', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 清空最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存")


def plot_confusion_matrix(cm, output_dir, timestamp):
    """绘制混淆矩阵（带热图）"""
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    if sns is not None:
        # 使用seaborn绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'})
        # 添加百分比标注
        for i in range(2):
            for j in range(2):
                if cm[i, j] > 0:
                    plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)', 
                            ha='center', va='center', fontsize=10, color='red', weight='bold')
    else:
        # 使用matplotlib绘制混淆矩阵
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, label='Count')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)', 
                        ha='center', va='center', color='black', fontsize=14, weight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix (with Percentages)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存")


def plot_loss_heatmap(history, output_dir, timestamp):
    """绘制损失组件热图（按epoch）"""
    # 准备数据
    epochs = range(1, len(history['train_loss']) + 1)
    loss_data = np.array([
        history['train_cls_loss'],
        history['train_ot_loss'],
        history['train_consist_loss'],
        history['train_adv_loss']
    ])
    
    # 归一化（按行归一化，便于比较）
    loss_data_norm = (loss_data - loss_data.min(axis=1, keepdims=True)) / (loss_data.max(axis=1, keepdims=True) - loss_data.min(axis=1, keepdims=True) + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始损失热图
    im1 = axes[0].imshow(loss_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Component')
    axes[0].set_title('Loss Components Heatmap (Raw)', fontsize=12, weight='bold')
    plt.colorbar(im1, ax=axes[0], label='Loss Value')
    
    # 归一化损失热图
    im2 = axes[1].imshow(loss_data_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('Loss Components Heatmap (Normalized)', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=axes[1], label='Normalized Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_heatmap_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失热图已保存")


def plot_roc_curve(y_true, y_probs, output_dir, timestamp):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curve_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存")


def plot_prediction_distribution(y_true, y_probs, output_dir, timestamp):
    """绘制预测概率分布（直方图、箱线图、小提琴图、热图等专业可视化）"""
    # 确保输入是numpy数组
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_probs, np.ndarray):
        y_probs = np.array(y_probs)
    
    # 确保是一维数组
    y_true = y_true.flatten()
    y_probs = y_probs.flatten()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 按真实标签分组
    neg_probs = y_probs[y_true == 0]
    pos_probs = y_probs[y_true == 1]
    
    # 确保数据不为空且是数组
    if len(neg_probs) == 0:
        neg_probs = np.array([0.0])
    else:
        neg_probs = np.array(neg_probs).flatten()
    
    if len(pos_probs) == 0:
        pos_probs = np.array([0.0])
    else:
        pos_probs = np.array(pos_probs).flatten()
    
    # 1. 直方图（重叠）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(neg_probs, bins=20, alpha=0.7, label='Negative', color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.hist(pos_probs, bins=20, alpha=0.7, label='Positive', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('(A) Histogram Overlay', fontsize=12, weight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    
    # 2. 箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = []
    labels = []
    if len(neg_probs) > 0:
        box_data.append(neg_probs)
        labels.append('Negative')
    if len(pos_probs) > 0:
        box_data.append(pos_probs)
        labels.append('Positive')
    
    if len(box_data) > 0:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, 
                        widths=0.6, showmeans=True, meanline=True)
        colors = ['#3498db', '#e74c3c']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
    ax2.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
    ax2.set_title('(B) Box Plot', fontsize=12, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0, 1])
    
    # 3. 小提琴图
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        import seaborn as sns
        import pandas as pd
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax3, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax3.set_xlabel('True Class', fontsize=11, weight='bold')
        ax3.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax3.set_title('(C) Violin Plot', fontsize=12, weight='bold', pad=10)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_ylim([0, 1])
    except Exception as e:
        # 如果seaborn不可用，使用KDE密度图
        try:
            if len(neg_probs) > 1:
                kde_neg = stats.gaussian_kde(neg_probs)
                x_neg = np.linspace(0, 1, 100)
                ax3.plot(x_neg, kde_neg(x_neg), label='Negative', color='#3498db', linewidth=2)
            if len(pos_probs) > 1:
                kde_pos = stats.gaussian_kde(pos_probs)
                x_pos = np.linspace(0, 1, 100)
                ax3.plot(x_pos, kde_pos(x_pos), label='Positive', color='#e74c3c', linewidth=2)
            ax3.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
            ax3.set_ylabel('Density', fontsize=11, weight='bold')
            ax3.set_title('(C) Kernel Density Estimation', fontsize=12, weight='bold', pad=10)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
        except:
            ax3.hist(neg_probs, bins=20, alpha=0.5, label='Negative', color='#3498db', density=True)
            ax3.hist(pos_probs, bins=20, alpha=0.5, label='Positive', color='#e74c3c', density=True)
            ax3.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
            ax3.set_ylabel('Density', fontsize=11, weight='bold')
            ax3.set_title('(C) Density Histogram', fontsize=12, weight='bold', pad=10)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. 2D热图
    ax4 = fig.add_subplot(gs[1, 0])
    prob_bins = np.linspace(0, 1, 21)
    hist_2d, xedges, yedges = np.histogram2d(y_probs, y_true, bins=[prob_bins, [0, 0.5, 1]])
    im = ax4.imshow(hist_2d.T, cmap='YlOrRd', aspect='auto', origin='lower', 
                    extent=[0, 1, 0, 1], interpolation='bilinear')
    ax4.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax4.set_ylabel('True Label', fontsize=11, weight='bold')
    ax4.set_yticks([0.25, 0.75])
    ax4.set_yticklabels(['Negative (0)', 'Positive (1)'])
    ax4.set_title('(D) Probability vs Label Heatmap', fontsize=12, weight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax4, label='Count')
    cbar.ax.tick_params(labelsize=9)
    
    # 5. 累积分布函数（CDF）
    ax5 = fig.add_subplot(gs[1, 1])
    if len(neg_probs) > 0:
        sorted_neg = np.sort(neg_probs)
        ax5.plot(sorted_neg, np.arange(len(sorted_neg)) / max(len(sorted_neg), 1), 
                 label='Negative', color='#3498db', linewidth=2.5, linestyle='-', marker='o', markersize=3, alpha=0.8)
    if len(pos_probs) > 0:
        sorted_pos = np.sort(pos_probs)
        ax5.plot(sorted_pos, np.arange(len(sorted_pos)) / max(len(sorted_pos), 1), 
                 label='Positive', color='#e74c3c', linewidth=2.5, linestyle='-', marker='s', markersize=3, alpha=0.8)
    ax5.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=11, weight='bold')
    ax5.set_title('(E) Cumulative Distribution Function', fontsize=12, weight='bold', pad=10)
    ax5.legend(frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # 6. Q-Q图
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        if len(neg_probs) > 1 and len(pos_probs) > 1:
            stats.probplot(neg_probs, dist="norm", plot=ax6)
            stats.probplot(pos_probs, dist="norm", plot=ax6)
            ax6.get_lines()[0].set_color('#3498db')
            ax6.get_lines()[0].set_label('Negative')
            ax6.get_lines()[1].set_color('#e74c3c')
            ax6.get_lines()[1].set_label('Positive')
            ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
            ax6.legend(frameon=True, fancybox=True, shadow=True)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor Q-Q plot', 
                    ha='center', va='center', fontsize=12, transform=ax6.transAxes)
            ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
    except:
        ax6.text(0.5, 0.5, 'Q-Q plot\nnot available', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # 7. 统计信息表格
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    try:
        neg_mean, neg_std = np.mean(neg_probs), np.std(neg_probs)
        pos_mean, pos_std = np.mean(pos_probs), np.std(pos_probs)
        neg_median = np.median(neg_probs)
        pos_median = np.median(pos_probs)
        neg_q25, neg_q75 = np.percentile(neg_probs, [25, 75])
        pos_q25, pos_q75 = np.percentile(pos_probs, [25, 75])
        
        stats_text = f"""
        Statistical Summary:
        
        Negative Class (n={len(neg_probs)}):
          Mean ± SD: {neg_mean:.4f} ± {neg_std:.4f}
          Median [IQR]: {neg_median:.4f} [{neg_q25:.4f}, {neg_q75:.4f}]
          Range: [{np.min(neg_probs):.4f}, {np.max(neg_probs):.4f}]
        
        Positive Class (n={len(pos_probs)}):
          Mean ± SD: {pos_mean:.4f} ± {pos_std:.4f}
          Median [IQR]: {pos_median:.4f} [{pos_q25:.4f}, {pos_q75:.4f}]
          Range: [{np.min(pos_probs):.4f}, {np.max(pos_probs):.4f}]
        """
    except:
        stats_text = "Statistics calculation error"
    
    ax7.text(0.1, 0.5, stats_text, fontsize=9.5, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
             alpha=0.9, edgecolor='#dee2e6', linewidth=1.5))
    ax7.set_title('(G) Statistical Summary', fontsize=12, weight='bold', pad=10)
    
    # 8. 密度对比图
    ax8 = fig.add_subplot(gs[2, 1])
    try:
        import seaborn as sns
        import pandas as pd
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.kdeplot(data=df, x='Probability', hue='Class', ax=ax8, 
                   palette=['#3498db', '#e74c3c'], fill=True, alpha=0.6, linewidth=2)
        ax8.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax8.set_ylabel('Density', fontsize=11, weight='bold')
        ax8.set_title('(H) Density Comparison', fontsize=12, weight='bold', pad=10)
        ax8.legend(frameon=True, fancybox=True, shadow=True)
        ax8.set_xlim([0, 1])
    except:
        ax8.hist(neg_probs, bins=20, alpha=0.5, label='Negative', color='#3498db', density=True)
        ax8.hist(pos_probs, bins=20, alpha=0.5, label='Positive', color='#e74c3c', density=True)
        ax8.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax8.set_ylabel('Density', fontsize=11, weight='bold')
        ax8.set_title('(H) Density Histogram', fontsize=12, weight='bold', pad=10)
        ax8.legend(frameon=True, fancybox=True, shadow=True)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # 9. 直方图+KDE
    ax9 = fig.add_subplot(gs[2, 2])
    try:
        import seaborn as sns
        import pandas as pd
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.histplot(data=df, x='Probability', hue='Class', bins=20, 
                    ax=ax9, palette=['#3498db', '#e74c3c'], alpha=0.7, 
                    stat='density', kde=True, line_kws={'linewidth': 2})
        ax9.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax9.set_ylabel('Density', fontsize=11, weight='bold')
        ax9.set_title('(I) Histogram with KDE', fontsize=12, weight='bold', pad=10)
        ax9.legend(frameon=True, fancybox=True, shadow=True)
        ax9.set_xlim([0, 1])
    except:
        ax9.hist([neg_probs, pos_probs], bins=20, alpha=0.7, 
                label=['Negative', 'Positive'], color=['#3498db', '#e74c3c'], 
                edgecolor='black', linewidth=0.5)
        ax9.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax9.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax9.set_title('(I) Side-by-side Histogram', fontsize=12, weight='bold', pad=10)
        ax9.legend(frameon=True, fancybox=True, shadow=True)
    ax9.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Comprehensive Prediction Distribution Analysis', fontsize=16, weight='bold', y=0.995)
    plt.savefig(output_dir / f'prediction_distribution_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 预测分布分析图已保存: prediction_distribution_bio_cot_multimodal_balanced_{timestamp}.png")


def plot_advanced_violin_analysis(y_true, y_probs, output_dir, timestamp):
    """绘制高级小提琴图分析（多子图展示）"""
    # 确保输入是numpy数组
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_probs, np.ndarray):
        y_probs = np.array(y_probs)
    
    # 确保是一维数组
    y_true = y_true.flatten()
    y_probs = y_probs.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Violin Plot Analysis', fontsize=16, weight='bold', y=0.995)
    
    neg_probs = y_probs[y_true == 0]
    pos_probs = y_probs[y_true == 1]
    
    # 确保数据不为空且是数组
    if len(neg_probs) == 0:
        neg_probs = np.array([0.0])
    else:
        neg_probs = np.array(neg_probs).flatten()
    
    if len(pos_probs) == 0:
        pos_probs = np.array([0.0])
    else:
        pos_probs = np.array(pos_probs).flatten()
    
    try:
        import seaborn as sns
        import pandas as pd
        
        # 准备数据
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        # 1. 标准小提琴图
        ax1 = axes[0, 0]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax1, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax1.set_title('(A) Standard Violin Plot', fontsize=12, weight='bold', pad=10)
        ax1.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim([0, 1])
        
        # 2. 小提琴图 + 散点图
        ax2 = axes[0, 1]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax2, 
                      palette=['#3498db', '#e74c3c'], inner=None, width=0.8)
        sns.stripplot(data=df, x='Class', y='Probability', ax=ax2, 
                     color='black', size=3, alpha=0.3, jitter=True)
        ax2.set_title('(B) Violin + Strip Plot', fontsize=12, weight='bold', pad=10)
        ax2.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim([0, 1])
        
        # 3. 小提琴图 + 箱线图
        ax3 = axes[1, 0]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax3, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax3.set_title('(C) Violin + Box Plot', fontsize=12, weight='bold', pad=10)
        ax3.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_ylim([0, 1])
        
        # 4. 小提琴图 + 均值点
        ax4 = axes[1, 1]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax4, 
                      palette=['#3498db', '#e74c3c'], inner='quart', width=0.8)
        # 添加均值点
        neg_mean = np.mean(neg_probs)
        pos_mean = np.mean(pos_probs)
        ax4.scatter([0], [neg_mean], color='yellow', s=100, marker='D', 
                   edgecolors='black', linewidths=1.5, zorder=10, label='Mean')
        ax4.scatter([1], [pos_mean], color='yellow', s=100, marker='D', 
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax4.set_title('(D) Violin + Quartiles + Mean', fontsize=12, weight='bold', pad=10)
        ax4.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax4.set_ylim([0, 1])
        
    except ImportError:
        # 如果没有seaborn，使用matplotlib绘制简化版本
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Seaborn required\nfor violin plots', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'violin_analysis_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 小提琴图分析已保存: violin_analysis_bio_cot_multimodal_balanced_{timestamp}.png")


def plot_loss_boxplot(history, output_dir, timestamp):
    """绘制损失组件的箱线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    loss_data = {
        'CLS Loss': history['train_cls_loss'],
        'OT Loss': history['train_ot_loss'],
        'Consist Loss': history['train_consist_loss'],
        'Adv Loss': history['train_adv_loss']
    }
    
    # 箱线图
    ax1 = axes[0]
    box_data = [loss_data[key] for key in loss_data.keys()]
    bp = ax1.boxplot(box_data, labels=list(loss_data.keys()), patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('Loss Value', fontsize=11)
    ax1.set_title('Loss Components Distribution (Boxplot)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 小提琴图（如果seaborn可用）
    ax2 = axes[1]
    try:
        import seaborn as sns
        data_list = []
        labels_list = []
        for key, values in loss_data.items():
            data_list.extend(values)
            labels_list.extend([key] * len(values))
        df = pd.DataFrame({'Loss': data_list, 'Component': labels_list})
        sns.violinplot(data=df, x='Component', y='Loss', ax=ax2, palette=colors)
        ax2.set_title('Loss Components Distribution (Violin Plot)', fontsize=12, weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except:
        # 如果seaborn不可用，使用分组箱线图
        positions = [1, 2, 3, 4]
        bp2 = ax2.boxplot(box_data, positions=positions, labels=list(loss_data.keys()), patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_ylabel('Loss Value', fontsize=11)
        ax2.set_title('Loss Components Distribution (Grouped Boxplot)', fontsize=12, weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_boxplot_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失箱线图已保存")


def plot_metrics_comparison(history, output_dir, timestamp):
    """绘制指标对比图（箱线图、热图）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 训练指标箱线图
    ax1 = axes[0, 0]
    train_metrics = {
        'Train Loss': history['train_loss'],
        'Train Acc': history['train_acc']
    }
    box_data = [train_metrics[key] for key in train_metrics.keys()]
    bp1 = ax1.boxplot(box_data, labels=list(train_metrics.keys()), patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Training Metrics Distribution', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 验证指标箱线图
    ax2 = axes[0, 1]
    val_metrics = {
        'Val Loss': history['val_loss'],
        'Val Acc': history['val_acc'],
        'Val AUC': history['val_auc'],
        'Val F1': history['val_f1']
    }
    box_data2 = [val_metrics[key] for key in val_metrics.keys()]
    bp2 = ax2.boxplot(box_data2, labels=list(val_metrics.keys()), patch_artist=True)
    colors2 = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Validation Metrics Distribution', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. 指标相关性热图
    ax3 = axes[1, 0]
    metrics_matrix = np.array([
        history['train_loss'],
        history['train_acc'],
        history['val_loss'],
        history['val_acc'],
        history['val_auc'],
        history['val_f1']
    ])
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(metrics_matrix)
    metric_names = ['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Val AUC', 'Val F1']
    im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_yticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, rotation=45, ha='right')
    ax3.set_yticklabels(metric_names)
    ax3.set_title('Metrics Correlation Heatmap', fontsize=12, weight='bold')
    # 添加数值标注
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax3, label='Correlation')
    
    # 4. 指标变化趋势热图（按epoch）
    ax4 = axes[1, 1]
    metrics_trend = np.array([
        history['train_loss'],
        history['train_acc'],
        history['val_loss'],
        history['val_acc'],
        history['val_auc'],
        history['val_f1']
    ])
    # 归一化到[0, 1]
    metrics_trend_norm = (metrics_trend - metrics_trend.min(axis=1, keepdims=True)) / \
                         (metrics_trend.max(axis=1, keepdims=True) - metrics_trend.min(axis=1, keepdims=True) + 1e-10)
    im2 = ax4.imshow(metrics_trend_norm, cmap='viridis', aspect='auto', interpolation='nearest')
    ax4.set_yticks(range(len(metric_names)))
    ax4.set_yticklabels(metric_names)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_title('Normalized Metrics Trend Heatmap', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=ax4, label='Normalized Value')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'metrics_comparison_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 指标对比图已保存（包含箱线图、相关性热图、趋势热图）")


def plot_loss_component_analysis(history, output_dir, timestamp):
    """绘制损失组件的详细分析（多维度可视化）"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # 1. 损失组件趋势（对数尺度）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs, history['train_cls_loss'], label='CLS Loss', color=colors[0], linewidth=2)
    ax1.semilogy(epochs, history['train_ot_loss'], label='OT Loss', color=colors[1], linewidth=2)
    ax1.semilogy(epochs, history['train_consist_loss'], label='Consist Loss', color=colors[2], linewidth=2)
    ax1.semilogy(epochs, history['train_adv_loss'], label='Adv Loss', color=colors[3], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=11, weight='bold')
    ax1.set_title('(A) Loss Components (Log Scale)', fontsize=12, weight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 损失组件箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    loss_data = [
        history['train_cls_loss'],
        history['train_ot_loss'],
        history['train_consist_loss'],
        history['train_adv_loss']
    ]
    bp = ax2.boxplot(loss_data, labels=['CLS', 'OT', 'Consist', 'Adv'], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Loss Value', fontsize=11, weight='bold')
    ax2.set_title('(B) Loss Components Distribution', fontsize=12, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 3. 损失组件小提琴图
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        import seaborn as sns
        import pandas as pd
        loss_df = pd.DataFrame({
            'CLS': history['train_cls_loss'],
            'OT': history['train_ot_loss'],
            'Consist': history['train_consist_loss'],
            'Adv': history['train_adv_loss']
        })
        loss_df_melted = loss_df.melt(var_name='Loss Type', value_name='Loss Value')
        sns.violinplot(data=loss_df_melted, x='Loss Type', y='Loss Value', ax=ax3, 
                      palette=colors, inner='box')
        ax3.set_title('(C) Loss Components Violin Plot', fontsize=12, weight='bold', pad=10)
        ax3.set_ylabel('Loss Value', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    except:
        ax3.text(0.5, 0.5, 'Seaborn required\nfor violin plot', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('(C) Loss Components Violin Plot', fontsize=12, weight='bold', pad=10)
    
    # 4. 损失组件相关性热图
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        import seaborn as sns
        import pandas as pd
        loss_df = pd.DataFrame({
            'CLS': history['train_cls_loss'],
            'OT': history['train_ot_loss'],
            'Consist': history['train_consist_loss'],
            'Adv': history['train_adv_loss']
        })
        corr = loss_df.corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                   ax=ax4, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
        ax4.set_title('(D) Loss Components Correlation', fontsize=12, weight='bold', pad=10)
    except:
        ax4.text(0.5, 0.5, 'Seaborn required\nfor heatmap', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('(D) Loss Components Correlation', fontsize=12, weight='bold', pad=10)
    
    # 5. 损失组件占比（饼图）
    ax5 = fig.add_subplot(gs[1, 1])
    total_loss = (np.mean(history['train_cls_loss']) + 
                  np.mean(history['train_ot_loss']) + 
                  np.mean(history['train_consist_loss']) + 
                  np.mean(history['train_adv_loss']))
    if total_loss > 0:
        sizes = [
            np.mean(history['train_cls_loss']) / total_loss,
            np.mean(history['train_ot_loss']) / total_loss,
            np.mean(history['train_consist_loss']) / total_loss,
            np.mean(history['train_adv_loss']) / total_loss
        ]
        labels = ['CLS', 'OT', 'Consist', 'Adv']
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 11, 'weight': 'bold'})
    ax5.set_title('(E) Average Loss Components Ratio', fontsize=12, weight='bold', pad=10)
    
    # 6. 损失组件累积贡献
    ax6 = fig.add_subplot(gs[1, 2])
    cumulative_cls = np.cumsum(history['train_cls_loss'])
    cumulative_ot = np.cumsum(history['train_ot_loss'])
    cumulative_consist = np.cumsum(history['train_consist_loss'])
    cumulative_adv = np.cumsum(history['train_adv_loss'])
    ax6.fill_between(epochs, 0, cumulative_cls, alpha=0.6, color=colors[0], label='CLS')
    ax6.fill_between(epochs, cumulative_cls, cumulative_cls + cumulative_ot, alpha=0.6, color=colors[1], label='OT')
    ax6.fill_between(epochs, cumulative_cls + cumulative_ot, 
                     cumulative_cls + cumulative_ot + cumulative_consist, 
                     alpha=0.6, color=colors[2], label='Consist')
    ax6.fill_between(epochs, cumulative_cls + cumulative_ot + cumulative_consist,
                     cumulative_cls + cumulative_ot + cumulative_consist + cumulative_adv,
                     alpha=0.6, color=colors[3], label='Adv')
    ax6.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax6.set_ylabel('Cumulative Loss', fontsize=11, weight='bold')
    ax6.set_title('(F) Cumulative Loss Components', fontsize=12, weight='bold', pad=10)
    ax6.legend(frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # 7. 损失组件与性能指标的关系
    ax7 = fig.add_subplot(gs[2, 0])
    ax7_twin = ax7.twinx()
    line1 = ax7.plot(epochs, history['train_cls_loss'], 'o-', color=colors[0], label='CLS Loss', linewidth=2, markersize=4)
    line2 = ax7_twin.plot(epochs, history['val_acc'], 's-', color='#9b59b6', label='Val Acc', linewidth=2, markersize=4)
    ax7.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax7.set_ylabel('CLS Loss', fontsize=11, weight='bold', color=colors[0])
    ax7_twin.set_ylabel('Validation Accuracy', fontsize=11, weight='bold', color='#9b59b6')
    ax7.tick_params(axis='y', labelcolor=colors[0])
    ax7_twin.tick_params(axis='y', labelcolor='#9b59b6')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax7.set_title('(G) CLS Loss vs Validation Accuracy', fontsize=12, weight='bold', pad=10)
    ax7.grid(True, alpha=0.3, linestyle='--')
    
    # 8. 损失组件散点矩阵
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(history['train_cls_loss'], history['train_ot_loss'], 
               alpha=0.6, s=50, c=epochs, cmap='viridis', edgecolors='black', linewidths=0.5)
    ax8.set_xlabel('CLS Loss', fontsize=11, weight='bold')
    ax8.set_ylabel('OT Loss', fontsize=11, weight='bold')
    ax8.set_title('(H) CLS vs OT Loss Scatter', fontsize=12, weight='bold', pad=10)
    cbar = plt.colorbar(scatter, ax=ax8)
    cbar.set_label('Epoch', fontsize=9)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # 9. 损失组件统计摘要
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats_text = f"""
    Loss Components Statistics:
    
    CLS Loss:
      Mean: {np.mean(history['train_cls_loss']):.6f}
      Std: {np.std(history['train_cls_loss']):.6f}
      Min: {np.min(history['train_cls_loss']):.6f}
      Max: {np.max(history['train_cls_loss']):.6f}
    
    OT Loss:
      Mean: {np.mean(history['train_ot_loss']):.6f}
      Std: {np.std(history['train_ot_loss']):.6f}
      Min: {np.min(history['train_ot_loss']):.6f}
      Max: {np.max(history['train_ot_loss']):.6f}
    
    Consist Loss:
      Mean: {np.mean(history['train_consist_loss']):.6f}
      Std: {np.std(history['train_consist_loss']):.6f}
      Min: {np.min(history['train_consist_loss']):.6f}
      Max: {np.max(history['train_consist_loss']):.6f}
    
    Adv Loss:
      Mean: {np.mean(history['train_adv_loss']):.6f}
      Std: {np.std(history['train_adv_loss']):.6f}
      Min: {np.min(history['train_adv_loss']):.6f}
      Max: {np.max(history['train_adv_loss']):.6f}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
             alpha=0.9, edgecolor='#dee2e6', linewidth=1.5))
    ax9.set_title('(I) Statistical Summary', fontsize=12, weight='bold', pad=10)
    
    plt.suptitle('Comprehensive Loss Component Analysis', fontsize=16, weight='bold', y=0.995)
    plt.savefig(output_dir / f'loss_component_analysis_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 损失组件分析图已保存: loss_component_analysis_bio_cot_multimodal_balanced_{timestamp}.png")

