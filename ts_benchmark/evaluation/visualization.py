# -*- coding: utf-8 -*-
"""
可视化模块 - 用于绘制预测结果对比图
"""

import os
from typing import List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import regression_metrics

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def inverse_transform_data(
    data: np.ndarray,
    scaler=None,
    hist_data: np.ndarray = None
) -> np.ndarray:
    """
    反归一化数据

    :param data: 归一化后的数据
    :param scaler: StandardScaler 对象
    :param hist_data: 历史数据（用于获取正确的形状）
    :return: 反归一化后的数据
    """
    if scaler is None:
        return data

    original_shape = data.shape
    if len(original_shape) == 3:
        n_samples, n_timesteps, n_features = original_shape
        data_2d = data.reshape(-1, n_features)
        inverted = scaler.inverse_transform(data_2d)
        return inverted.reshape(n_samples, n_timesteps, n_features)
    else:
        return scaler.inverse_transform(data)


def plot_single_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    time_index: Optional[pd.Index] = None,
    horizon: int = 96,
    figsize: Tuple[int, int] = (16, 8),
    title: str = None,
    save_path: str = None,
    n_channels_to_plot: int = 4,
    alpha: float = 0.8,
    scaler=None,
    channel_names: List[str] = None
) -> plt.Figure:
    """
    绘制单个滚动窗口的预测对比图

    :param actual: 真实值，shape: (horizon, n_channels) 或 (horizon,)
    :param predicted: 预测值，shape: (horizon, n_channels) 或 (horizon,)
    :param time_index: 时间索引
    :param horizon: 预测长度
    :param figsize: 图形大小
    :param title: 图表标题
    :param save_path: 保存路径
    :param n_channels_to_plot: 最多显示的通道数
    :param alpha: 透明度
    :param scaler: StandardScaler对象，用于计算归一化MSE
    :param channel_names: 通道名称列表
    :return: matplotlib Figure 对象
    """
    # 处理一维情况
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)

    n_timesteps, n_channels = actual.shape
    n_channels_to_plot = min(n_channels_to_plot, n_channels)

    # 与 benchmark 中 mse_norm / mae_norm 一致：在 scaler 变换空间计算误差
    mse_norms = None
    mae_norms = None
    if scaler is not None:
        err = scaler.transform(actual) - scaler.transform(predicted)
        mse_norms = np.mean(err ** 2, axis=0)
        mae_norms = np.mean(np.abs(err), axis=0)

    n_rows = n_channels_to_plot
    fig, axes = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows / 2), squeeze=False)

    x = np.arange(n_timesteps)
    if time_index is not None:
        x_labels = time_index
    else:
        x_labels = x

    for i in range(n_rows):
        ax = axes[i, 0]
        ax.plot(x_labels, actual[:, i], label='Actual', color='#2E86AB', linewidth=1.5, alpha=alpha)
        ax.plot(x_labels, predicted[:, i], label='Predicted', color='#E94F37', linewidth=1.5, alpha=alpha, linestyle='--')

        # 获取通道名称
        ch_name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"

        # 计算该通道的误差指标
        mse = np.mean((actual[:, i] - predicted[:, i]) ** 2)
        mae = np.mean(np.abs(actual[:, i] - predicted[:, i]))

        # 如果有归一化MSE，显示mse_norm
        if mse_norms is not None:
            mse_norm = mse_norms[i]
            mae_norm = mae_norms[i]
            ax.set_title(f'{ch_name}: Actual vs Predicted (MSE={mse:.4f}, MSE_norm={mse_norm:.4f}, MAE_norm={mae_norm:.4f})', fontsize=11)
        else:
            ax.set_title(f'{ch_name}: Actual vs Predicted (MSE={mse:.4f}, MAE={mae:.4f})', fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Forecast Comparison (Horizon={horizon})', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_multi_rolling_forecast(
    actual_list: List[np.ndarray],
    predicted_list: List[np.ndarray],
    stride: int = 1,
    horizon: int = 96,
    figsize: Tuple[int, int] = (20, 12),
    title: str = None,
    save_path: str = None,
    max_rollings_to_show: int = 3
) -> plt.Figure:
    """
    绘制多个滚动窗口的预测对比图（展示预测趋势）

    :param actual_list: 真实值列表
    :param predicted_list: 预测值列表
    :param stride: 滚动步长
    :param horizon: 预测长度
    :param figsize: 图形大小
    :param title: 图表标题
    :param save_path: 保存路径
    :param max_rollings_to_show: 最多显示的滚动窗口数
    :return: matplotlib Figure 对象
    """
    n_rollings = min(len(actual_list), max_rollings_to_show)

    # 选择要显示的滚动窗口
    indices = np.linspace(0, len(actual_list) - 1, n_rollings, dtype=int)

    fig, axes = plt.subplots(n_rollings, 2, figsize=(figsize[0], figsize[1] * n_rollings / 3))

    if n_rollings == 1:
        axes = axes.reshape(-1, 2)

    for idx, rolling_idx in enumerate(indices):
        actual = actual_list[rolling_idx]
        predicted = predicted_list[rolling_idx]

        # 处理一维情况
        if actual.ndim == 1:
            actual = actual.reshape(-1, 1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)

        n_timesteps, n_channels = actual.shape
        x = np.arange(n_timesteps)

        # 左图：第一个通道的预测对比
        ax_left = axes[idx, 0]
        ax_left.plot(x, actual[:, 0], label='Actual', color='#2E86AB', linewidth=1.5)
        ax_left.plot(x, predicted[:, 0], label='Predicted', color='#E94F37', linewidth=1.5, linestyle='--')
        ax_left.set_title(f'Rolling #{rolling_idx} - Channel 0', fontsize=11)
        ax_left.set_xlabel('Time Step')
        ax_left.set_ylabel('Value')
        ax_left.legend(loc='upper right')
        ax_left.grid(True, alpha=0.3)

        # 右图：所有通道的预测误差
        ax_right = axes[idx, 1]
        errors = np.abs(actual - predicted)
        for ch in range(min(3, n_channels)):
            ax_right.plot(x, errors[:, ch], label=f'Channel {ch}', alpha=0.7, linewidth=1.2)
        ax_right.set_title(f'Rolling #{rolling_idx} - Absolute Error', fontsize=11)
        ax_right.set_xlabel('Time Step')
        ax_right.set_ylabel('Absolute Error')
        ax_right.legend(loc='upper right')
        ax_right.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Multi-Rolling Forecast Comparison (Stride={stride}, Horizon={horizon})', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_comprehensive_forecast(
    series_name: str,
    actual_list: List[pd.DataFrame],
    predicted_list: List[pd.DataFrame],
    train_data: Optional[pd.DataFrame] = None,
    stride: int = 1,
    horizon: int = 96,
    save_path: str = None,
    figsize: Tuple[int, int] = (20, 16)
) -> plt.Figure:
    """
    绘制综合预测结果图（包含历史数据和预测结果）

    :param series_name: 数据集名称
    :param actual_list: 真实值 DataFrame 列表
    :param predicted_list: 预测值 DataFrame 列表
    :param train_data: 训练数据（可选，用于显示历史上下文）
    :param stride: 滚动步长
    :param horizon: 预测长度
    :param save_path: 保存路径
    :param figsize: 图形大小
    :return: matplotlib Figure 对象
    """
    if len(actual_list) == 0:
        return None

    # 合并所有数据用于绘图
    all_actual = pd.concat(actual_list, axis=0) if isinstance(actual_list[0], pd.DataFrame) else None
    all_predicted = pd.concat(predicted_list, axis=0) if isinstance(predicted_list[0], pd.DataFrame) else None

    if all_actual is None:
        return None

    n_channels = all_actual.shape[1] if all_actual.ndim > 1 else 1
    n_rows = min(4, n_channels)  # 最多显示4个子图

    fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0], figsize[1] * n_rows / 2))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors_actual = plt.cm.Blues(np.linspace(0.4, 0.8, 3))
    colors_pred = plt.cm.Oranges(np.linspace(0.4, 0.8, 3))

    # 左列：预测对比图
    for ch in range(n_rows):
        ax = axes[ch, 0]
        x = np.arange(len(all_actual))

        ax.plot(x, all_actual.values[:, ch], label='Actual', color=colors_actual[2], linewidth=1.5)
        ax.plot(x, all_predicted.values[:, ch], label='Predicted', color=colors_pred[2], linewidth=1.5, linestyle='--', alpha=0.8)

        # 添加滚动窗口分隔线
        for i in range(1, len(actual_list)):
            split_point = i * horizon
            if split_point < len(x):
                ax.axvline(x=split_point, color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'{series_name} - Channel {ch}: Actual vs Predicted', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 右列：误差分析图
    for ch in range(n_rows):
        ax = axes[ch, 1]
        errors = np.abs(all_actual.values[:, ch] - all_predicted.values[:, ch])
        x = np.arange(len(errors))

        ax.fill_between(x, 0, errors, alpha=0.4, color='#E94F37')
        ax.plot(x, errors, color='#E94F37', linewidth=1)

        # 添加滚动窗口分隔线
        for i in range(1, len(actual_list)):
            split_point = i * horizon
            if split_point < len(x):
                ax.axvline(x=split_point, color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'Channel {ch}: Absolute Error', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)

    # 计算总体指标
    overall_mse = np.mean((all_actual.values - all_predicted.values) ** 2)
    overall_mae = np.mean(np.abs(all_actual.values - all_predicted.values))
    overall_rmse = np.sqrt(overall_mse)

    fig.suptitle(
        f'{series_name} - Comprehensive Forecast Analysis\n'
        f'Overall MSE: {overall_mse:.4f} | MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | '
        f'Rollings: {len(actual_list)} | Stride: {stride} | Horizon: {horizon}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_error_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
    figsize: Tuple[int, int] = (14, 6),
    title: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    绘制预测误差分布图

    :param actual: 真实值
    :param predicted: 预测值
    :param figsize: 图形大小
    :param title: 图表标题
    :param save_path: 保存路径
    :return: matplotlib Figure 对象
    """
    errors = (actual - predicted).flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：误差直方图
    ax_left = axes[0]
    ax_left.hist(errors, bins=50, color='#E94F37', alpha=0.7, edgecolor='white')
    ax_left.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    ax_left.axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=1.5, label=f'Mean: {np.mean(errors):.4f}')
    ax_left.set_title('Error Distribution (Histogram)', fontsize=12)
    ax_left.set_xlabel('Prediction Error')
    ax_left.set_ylabel('Frequency')
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    # 右图：Q-Q图（检验误差正态性）
    ax_right = axes[1]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax_right)
    ax_right.set_title('Q-Q Plot (Normality Check)', fontsize=12)
    ax_right.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Error Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def _short_dataset_name(series_name: str) -> str:
    """ETTh2.csv -> ETTh2"""
    base = os.path.basename(series_name)
    return base.replace(".csv", "").replace(".CSV", "")


def _compute_corr(actual: np.ndarray) -> np.ndarray:
    """Compute |Pearson correlation| matrix."""
    n_t, n_c = actual.shape
    if n_c > 1 and n_t > 2:
        corr = np.corrcoef(actual.T)
        return np.abs(np.nan_to_num(corr, nan=0.0))
    return np.array([[1.0]])


def plot_time_domain_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    series_name: str,
    horizon: int,
    model_name: str,
    channel_names: Optional[List[str]],
    scaler,
    save_path: str,
    dpi: int = 150,
) -> None:
    """时域多通道 Actual vs Predicted — 论文 Fig.7 左上风格（热力图单独成图）。"""
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    n_t, n_c = actual.shape
    if channel_names is None or len(channel_names) != n_c:
        channel_names = [f"C{i + 1}" for i in range(n_c)]

    ds = _short_dataset_name(series_name)
    mse_n = (
        regression_metrics.mse_norm(actual, predicted, scaler)
        if scaler is not None
        else float(np.mean((actual - predicted) ** 2))
    )

    try:
        cmap_lines = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap_lines = plt.cm.get_cmap("tab10")

    n_rows = min(n_c, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows), squeeze=False)
    t_idx = np.arange(n_t)
    for i in range(n_rows):
        ax = axes[i, 0]
        color = cmap_lines(i % 10)
        ax.plot(t_idx, actual[:, i], color=color, linewidth=1.2, label="Actual")
        ax.plot(t_idx, predicted[:, i], color=color, linewidth=1.0, linestyle="--", alpha=0.85, label="Predicted")
        mse_i = np.mean((actual[:, i] - predicted[:, i]) ** 2)
        ax.set_title(f"{channel_names[i]}", fontsize=11)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{ds}-H={horizon}-{model_name}  |  mse_norm={mse_n:.4f}  (Time Domain)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_channel_fft(
    actual: np.ndarray,
    predicted: np.ndarray,
    series_name: str,
    horizon: int,
    model_name: str,
    channel_names: Optional[List[str]],
    scaler,
    save_path: str,
    dpi: int = 150,
) -> None:
    """各通道 rFFT 幅值谱 — 论文 Fig.7 下方风格，每通道独立子图，不拥挤。"""
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    n_t, n_c = actual.shape
    if channel_names is None or len(channel_names) != n_c:
        channel_names = [f"C{i + 1}" for i in range(n_c)]

    ds = _short_dataset_name(series_name)

    try:
        cmap_lines = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap_lines = plt.cm.get_cmap("tab10")

    n_rows = n_c
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.2 * n_rows), squeeze=False)
    for i in range(n_c):
        ax = axes[i, 0]
        color = cmap_lines(i % 10)

        # Actual spectrum (solid)
        x_act = actual[:, i] - np.mean(actual[:, i])
        spec_act = np.abs(np.fft.rfft(x_act))
        freq = np.fft.rfftfreq(n_t)

        # Predicted spectrum (dashed)
        x_pred = predicted[:, i] - np.mean(predicted[:, i])
        spec_pred = np.abs(np.fft.rfft(x_pred))

        ax.plot(freq, spec_act, color=color, linewidth=1.2, label="Actual")
        ax.plot(freq, spec_pred, color=color, linewidth=1.0, linestyle="--", alpha=0.85, label="Predicted")

        # Mark dominant freq
        if len(spec_act) > 1:
            k = 1 + int(np.argmax(spec_act[1:]))
            ax.scatter(freq[k], spec_act[k], color="red", s=18, zorder=5, label=f"Peak@freq={freq[k]:.3f}")

        ax.set_title(f"{channel_names[i]} — rFFT", fontsize=10)
        ax.set_xlabel("Frequency", fontsize=8)
        ax.set_ylabel("|rFFT|", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{ds}-H={horizon}-{model_name}  |  rFFT Spectrum per Channel",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_channel_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    series_name: str,
    horizon: int,
    model_name: str,
    channel_names: Optional[List[str]],
    scaler,
    save_path: str,
    dpi: int = 150,
) -> None:
    """通道 |Pearson 相关| 热力图 — 论文 Fig.7 右上风格（独立大图）。"""
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    n_t, n_c = actual.shape
    if channel_names is None or len(channel_names) != n_c:
        channel_names = [f"C{i + 1}" for i in range(n_c)]

    ds = _short_dataset_name(series_name)
    mse_n = (
        regression_metrics.mse_norm(actual, predicted, scaler)
        if scaler is not None
        else float(np.mean((actual - predicted) ** 2))
    )

    abs_corr = _compute_corr(actual)

    fig, ax = plt.subplots(figsize=(max(6, n_c * 1.2), max(5, n_c * 1.0)))
    im = ax.imshow(abs_corr, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_c))
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(channel_names, fontsize=10)
    for i in range(n_c):
        for j in range(n_c):
            text_color = "white" if abs_corr[i, j] > 0.5 else "black"
            ax.text(j, i, f"{abs_corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color=text_color)
    ax.set_title(
        f"{ds}-H={horizon}-{model_name}  |  |Channel Correlation|  (mse_norm={mse_n:.4f})\n"
        f"Proxy for Mopformer channel attention (arXiv:2412.10859 Fig.7)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Pearson r|")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_forecast_visualization(
    series_name: str,
    actual_data: np.ndarray,
    predicted_data: np.ndarray,
    save_dir: str,
    train_data: np.ndarray = None,
    stride: int = 1,
    horizon: int = 96,
    model_name: str = "Model",
    scaler=None,
    channel_names: List[str] = None
) -> dict:
    """
    创建完整的预测可视化并保存

    :param series_name: 数据集名称（如 ETTh1）
    :param actual_data: 真实值
    :param predicted_data: 预测值
    :param save_dir: 保存目录
    :param train_data: 训练数据（可选）
    :param stride: 滚动步长
    :param horizon: 预测长度
    :param model_name: 模型名称（如 Mopformer）
    :param scaler: StandardScaler对象，用于计算归一化MSE
    :param channel_names: 通道名称列表（如 HUFL, HULL, MUFL 等）
    :return: 包含保存路径的字典
    """
    os.makedirs(save_dir, exist_ok=True)

    saved_files = {}
    safe_series_name = series_name.replace('/', '_').replace('\\', '_')
    # 生成标题：ETTh1-H=336-Mopformer
    title_prefix = f"{series_name}-H={horizon}-{model_name}"

    # 1. 保存综合预测图（如果提供的是 DataFrame 列表）
    if isinstance(actual_data, list) and len(actual_data) > 0:
        comprehensive_path = os.path.join(save_dir, f'{safe_series_name}_comprehensive.png')
        try:
            plot_comprehensive_forecast(
                series_name=series_name,
                actual_list=actual_data,
                predicted_list=predicted_data,
                train_data=train_data,
                stride=stride,
                horizon=horizon,
                save_path=comprehensive_path
            )
            saved_files['comprehensive'] = comprehensive_path
        except Exception as e:
            print(f"Warning: Failed to create comprehensive plot for {series_name}: {e}")

    # 2. 保存多滚动窗口对比图
    if isinstance(actual_data, list) and len(actual_data) > 1:
        multi_rolling_path = os.path.join(save_dir, f'{safe_series_name}_multi_rolling.png')
        try:
            plot_multi_rolling_forecast(
                actual_list=actual_data,
                predicted_list=predicted_data,
                stride=stride,
                horizon=horizon,
                save_path=multi_rolling_path
            )
            saved_files['multi_rolling'] = multi_rolling_path
        except Exception as e:
            print(f"Warning: Failed to create multi-rolling plot for {series_name}: {e}")

    # 3. 保存第一个滚动窗口的详细对比图（主要图表）
    if isinstance(actual_data, list):
        first_actual = actual_data[0]
        first_predicted = predicted_data[0]
    else:
        first_actual = actual_data
        first_predicted = predicted_data

    single_forecast_path = os.path.join(save_dir, f'{safe_series_name}_forecast.png')
    try:
        if isinstance(first_actual, pd.DataFrame):
            actual_vals = first_actual.values
            pred_vals = first_predicted.values
        else:
            actual_vals = first_actual
            pred_vals = first_predicted

        plot_single_forecast(
            actual=actual_vals,
            predicted=pred_vals,
            horizon=horizon,
            title=title_prefix,
            save_path=single_forecast_path,
            scaler=scaler,
            channel_names=channel_names
        )
        saved_files['single_forecast'] = single_forecast_path
    except Exception as e:
        print(f"Warning: Failed to create single forecast plot for {series_name}: {e}")

    # 5. 论文 Fig.7 风格三张独立图：时域、rFFT、通道相关热力图
    try:
        if isinstance(actual_data, list):
            av = actual_data[0].values if isinstance(actual_data[0], pd.DataFrame) else actual_data[0]
            pv = predicted_data[0].values if isinstance(predicted_data[0], pd.DataFrame) else predicted_data[0]
        elif isinstance(actual_data, pd.DataFrame):
            av, pv = actual_data.values, predicted_data.values
        else:
            av, pv = actual_data, predicted_data

        # 5a. 时域多通道
        time_path = os.path.join(save_dir, f'{safe_series_name}_time_domain.png')
        plot_time_domain_forecast(
            actual=av, predicted=pv, series_name=series_name,
            horizon=horizon, model_name=model_name, channel_names=channel_names,
            scaler=scaler, save_path=time_path,
        )
        saved_files['time_domain'] = time_path

        # 5b. 各通道 rFFT
        fft_path = os.path.join(save_dir, f'{safe_series_name}_rfft.png')
        plot_channel_fft(
            actual=av, predicted=pv, series_name=series_name,
            horizon=horizon, model_name=model_name, channel_names=channel_names,
            scaler=scaler, save_path=fft_path,
        )
        saved_files['rfft'] = fft_path

        # 5c. 通道相关热力图
        corr_path = os.path.join(save_dir, f'{safe_series_name}_channel_corr.png')
        plot_channel_correlation(
            actual=av, predicted=pv, series_name=series_name,
            horizon=horizon, model_name=model_name, channel_names=channel_names,
            scaler=scaler, save_path=corr_path,
        )
        saved_files['channel_corr'] = corr_path
    except Exception as e:
        print(f"Warning: Failed to create time_domain / rfft / channel_corr plots for {series_name}: {e}")

    # 4. 保存误差分布图
    if isinstance(actual_data, list):
        all_actual = np.concatenate([a.values if isinstance(a, pd.DataFrame) else a for a in actual_data], axis=0)
        all_predicted = np.concatenate([p.values if isinstance(p, pd.DataFrame) else p for p in predicted_data], axis=0)
    else:
        all_actual = actual_data
        all_predicted = predicted_data

    error_dist_path = os.path.join(save_dir, f'{safe_series_name}_error_dist.png')
    try:
        plot_error_distribution(
            actual=all_actual,
            predicted=all_predicted,
            title=title_prefix,
            save_path=error_dist_path
        )
        saved_files['error_distribution'] = error_dist_path
    except Exception as e:
        print(f"Warning: Failed to create error distribution plot for {series_name}: {e}")

    return saved_files
