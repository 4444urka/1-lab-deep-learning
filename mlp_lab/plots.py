"""Графики: EDA, кривые обучения, сетка экспериментов, градиенты, активации.

Все функции принимают данные явно и возвращают ``matplotlib.figure.Figure``,
чтобы их можно было сохранить или доработать снаружи.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from mlp_lab.activations import get_activation
from mlp_lab.training import History

# Фиксированный порядок категориальных цветов: цвет закреплён за сущностью, а не за рангом.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "vlag"
INK = "#52514e"


def set_style() -> None:
    """Единый спокойный стиль: тонкая сетка, без верхней/правой рамки."""
    sns.set_theme(context="notebook", style="whitegrid", palette=CATEGORICAL)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "axes.titleweight": "bold",
            "text.parse_math": False,  # «$11-$25k» — текст, а не формула
            "figure.dpi": 100,
        }
    )


def _palette(labels: Sequence) -> dict:
    return {label: CATEGORICAL[i % len(CATEGORICAL)] for i, label in enumerate(labels)}


# --------------------------------------------------------------------------- EDA


def plot_class_distribution(y: pd.Series, class_names: Sequence[str] | None = None, title: str = "Распределение классов"):
    """Горизонтальные столбцы: число объектов и доля каждого класса."""
    counts = y.value_counts().sort_index()
    labels = [class_names[i] for i in counts.index] if class_names is not None else counts.index.astype(str)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(counts) + 1))
    ax.barh(labels, counts.values, color=[_palette(labels)[label] for label in labels], height=0.6)
    for i, value in enumerate(counts.values):
        ax.text(value, i, f"  {value}  ({value / counts.sum():.1%})", va="center", color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_title(title)
    ax.set_xlabel("объектов")
    return fig


def plot_missing(df: pd.DataFrame, title: str = "Пропуски по колонкам"):
    """Доля пропусков по колонкам (только колонки с пропусками) и матрица пропусков."""
    share = df.isna().mean().sort_values(ascending=False)
    share = share[share > 0]
    fig, (ax_bar, ax_mat) = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * len(share) + 1)), width_ratios=[1, 1.4])
    ax_bar.barh(share.index, share.values, color=CATEGORICAL[0], height=0.6)
    ax_bar.invert_yaxis()
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bar.set_title(title)
    sns.heatmap(df[share.index].isna().T, cbar=False, cmap=["#f4f3ef", CATEGORICAL[0]], ax=ax_mat, xticklabels=False)
    ax_mat.set_title("Матрица пропусков (синий — пропуск)")
    ax_mat.set_xlabel("объекты")
    fig.tight_layout()
    return fig


def plot_numeric_by_class(
    df: pd.DataFrame, columns: Sequence[str], target: str, class_names: Sequence[str] | None = None, ncols: int = 4, log_cols: Sequence[str] = ()
):
    """Плотности (KDE) числовых признаков по классам; ``log_cols`` рисуются в шкале log1p."""
    labels = class_names or sorted(df[target].unique())
    data = df.assign(**{target: df[target].map(dict(enumerate(labels))) if class_names else df[target]})
    data = data.assign(**{col: np.log1p(data[col]) for col in log_cols})
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    for ax, col in zip(axes.flat, columns, strict=False):
        sns.kdeplot(
            data=data, x=col, hue=target, hue_order=labels, palette=_palette(labels),
            common_norm=False, ax=ax, linewidth=1.5, warn_singular=False,
            legend=ax is axes.flat[0],
        )
        ax.set_ylabel("")
        ax.set_title(f"log1p({col})" if col in log_cols else col, fontsize=10)
        ax.set_xlabel("")
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Распределения числовых признаков по классам", y=1.0)
    fig.tight_layout()
    return fig


def plot_box_by_class(
    df: pd.DataFrame, columns: Sequence[str], target: str, class_names: Sequence[str] | None = None, ncols: int = 3
):
    """Boxplot признаков по классам — медианы, разброс, выбросы."""
    labels = class_names or sorted(df[target].unique())
    data = df.assign(**{target: df[target].map(dict(enumerate(labels))) if class_names else df[target]})
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, columns, strict=False):
        sns.boxplot(
            data=data, x=target, y=col, hue=target, order=labels, hue_order=labels, palette=_palette(labels),
            ax=ax, linewidth=1, fliersize=2, width=0.6, legend=False,
        )
        ax.set_xlabel("")
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_categorical_vs_target(
    df: pd.DataFrame, columns: Sequence[str], target: str, class_names: Sequence[str] | None = None, ncols: int = 2
):
    """Нормированные stacked bar: состав классов внутри каждой категории признака."""
    labels = class_names or sorted(df[target].unique())
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.4 * nrows), squeeze=False)
    colors = _palette(labels)
    for ax, col in zip(axes.flat, columns, strict=False):
        table = pd.crosstab(df[col].fillna("missing"), df[target], normalize="index")
        table.columns = [labels[c] if class_names else c for c in table.columns]
        counts = df[col].fillna("missing").value_counts()
        table = table.loc[counts.index]
        left = np.zeros(len(table))
        for label in labels:
            ax.barh(table.index.astype(str), table[label], left=left, color=colors[label], label=label, height=0.7, edgecolor="white", linewidth=1)
            left += table[label].to_numpy()
        ax.set_yticks(range(len(table)), [f"{idx} (n={counts[idx]})" for idx in table.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(col, fontsize=10)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, columns: Sequence[str], method: str = "spearman", title: str | None = None):
    """Матрица корреляций (нижний треугольник), дивергентная шкала с нулём в центре."""
    corr = df[list(columns)].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    size = 0.45 * len(columns) + 2
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr, mask=mask, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0, square=True, ax=ax,
        annot=len(columns) <= 25, fmt=".1f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.6}, linewidths=0.5,
    )
    ax.set_title(title or f"Корреляции ({method})")
    return fig


# ------------------------------------------------------------------- обучение


def plot_history(history: History, metrics: Sequence[str] = ("loss", "macro_f1"), title: str = ""):
    """Кривые train/valid по эпохам для каждой метрики; вертикаль — лучшая эпоха."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 3.8), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=True):
        epochs = np.arange(1, history.n_epochs + 1)
        ax.plot(epochs, history.train[metric], label="train", color=CATEGORICAL[0])
        ax.plot(epochs, history.valid[metric], label="valid", color=CATEGORICAL[1])
        ax.axvline(history.best_epoch + 1, color=INK, linestyle="--", linewidth=1, label=f"лучшая эпоха ({history.best_epoch + 1})")
        ax.set_xlabel("эпоха")
        ax.set_title(metric)
        ax.legend(frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_histories(histories: Mapping[str, History], metric: str = "loss", split: str = "valid", title: str | None = None):
    """Сравнение нескольких моделей по одной метрике на одной выборке."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, history) in enumerate(histories.items()):
        values = getattr(history, split)[metric]
        ax.plot(np.arange(1, len(values) + 1), values, label=name, color=CATEGORICAL[i % len(CATEGORICAL)])
    ax.set_xlabel("эпоха")
    ax.set_title(title or f"{split} {metric}")
    ax.legend(frameon=False)
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], normalize: str | None = "true", title: str = ""):
    """Матрица ошибок; по умолчанию нормирована по строкам (доля от истинного класса)."""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d", cmap=SEQUENTIAL_CMAP, ax=ax,
        xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1 if normalize else None, cbar=False,
    )
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истина")
    ax.set_title(title or "Матрица ошибок")
    ax.tick_params(axis="x", rotation=20)
    return fig


# ------------------------------------------------------------------- сетка


def plot_grid_heatmap(summary: pd.DataFrame, value: str, index: str = "depth", columns: str = "width", title: str | None = None, fmt: str = ".3f"):
    """Heatmap метрики по сетке гиперпараметров (последовательная шкала)."""
    table = summary.pivot(index=index, columns=columns, values=value)
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 2, 0.7 * table.shape[0] + 1.5))
    sns.heatmap(table, annot=True, fmt=fmt, cmap=SEQUENTIAL_CMAP, ax=ax, cbar_kws={"shrink": 0.7}, linewidths=0.5)
    ax.set_title(title or value)
    return fig


def plot_metric_vs_params(results: pd.DataFrame, metric: str, hue: str = "depth", title: str | None = None):
    """Метрика от числа параметров (лог-шкала); линии соединяют одну глубину."""
    summary = results.groupby([hue, "width"]).agg(n_params=("n_params", "first"), value=(metric, "mean"), std=(metric, "std")).reset_index()
    levels = sorted(summary[hue].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, level in enumerate(levels):
        part = summary[summary[hue] == level].sort_values("n_params")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.errorbar(part["n_params"], part["value"], yerr=part["std"], marker="o", markersize=5, capsize=2, color=color, label=f"{hue}={level}")
    ax.set_xscale("log")
    ax.set_xlabel("число параметров (log)")
    ax.set_title(title or metric)
    ax.legend(frameon=False, ncol=2)
    return fig


# --------------------------------------------------------------- градиенты


def plot_gradient_norms(norms: Mapping[str, pd.DataFrame], column: str = "weight_grad_norm", title: str | None = None):
    """Нормы градиентов по слоям (от входа к выходу) для нескольких случаев."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, frame) in enumerate(norms.items()):
        ax.plot(np.arange(1, len(frame) + 1), frame[column].to_numpy(), marker="o", markersize=6, label=name, color=CATEGORICAL[i % len(CATEGORICAL)])
    ax.set_yscale("log")
    ax.set_xlabel("номер Linear-слоя (1 — вход → скрытый 1, последний — скрытый → выход)")
    ax.set_ylabel(column + " (log)")
    ax.set_title(title or column)
    ax.legend(frameon=False)
    return fig


def plot_input_gradients(grads: Mapping[str, np.ndarray], feature_names: Sequence[str], top_k: int = 15, title: str = "Градиент loss по входным признакам"):
    """Признаки с наибольшим |dL/dx| (saliency) для одного или нескольких объектов."""
    frame = pd.DataFrame({name: np.ravel(g) for name, g in grads.items()}, index=feature_names)
    top = frame.abs().max(axis=1).sort_values(ascending=False).index[:top_k]
    fig, axes = plt.subplots(1, len(grads), figsize=(5.5 * len(grads), 0.32 * top_k + 1.5), sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, frame.columns, strict=True):
        values = frame.loc[top, name]
        ax.barh(top, values, color=np.where(values > 0, CATEGORICAL[1], CATEGORICAL[0]), height=0.6)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(name, fontsize=10)
    axes.flat[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_weight_gradient_heatmaps(report, max_layers: int | None = None, title: str = "dL/dW по слоям"):
    """Heatmap матриц dL/dW каждого Linear-слоя (общая дивергентная шкала на слой)."""
    layers = [layer for layer in report.layers.values() if layer.weight_grad is not None][:max_layers]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.2), squeeze=False)
    for ax, layer in zip(axes.flat, layers, strict=True):
        grad = layer.weight_grad
        limit = np.abs(grad).max() or 1.0
        ax.imshow(grad, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
        ax.set_title(f"{layer.name}  {grad.shape[0]}×{grad.shape[1]}", fontsize=9)
        ax.set_xlabel("вход")
        ax.set_ylabel("выход")
        ax.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- активации


def plot_activations(names: Sequence[str], x_range: tuple[float, float] = (-5, 5), n_points: int = 1001, ncols: int = 4):
    """Сетка графиков: функция активации и её производная."""
    x = np.linspace(*x_range, n_points)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        act = get_activation(name)
        ax.plot(x, act(x), color=CATEGORICAL[0], label="f(x)")
        ax.plot(x, act.derivative(x), color=CATEGORICAL[1], linestyle="--", linewidth=1.5, label="f'(x)")
        ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        ax.axvline(0, color="#c3c2b7", linewidth=0.8)
        ax.set_title(name, fontsize=11)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig


def plot_softmax(logits: Sequence[float], temperatures: Sequence[float] = (0.5, 1.0, 2.0, 5.0)):
    """Softmax одного вектора логитов при разных температурах: x / T."""
    act = get_activation("softmax")
    logits = np.asarray(logits, dtype=np.float64)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(3.4 * len(temperatures), 3), sharey=True)
    for ax, temperature in zip(np.atleast_1d(axes), temperatures, strict=True):
        probs = act(logits / temperature)
        ax.bar(range(len(logits)), probs, color=CATEGORICAL[0], width=0.6)
        for i, p in enumerate(probs):
            ax.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(logits)), [f"{v:g}" for v in logits])
        ax.set_title(f"T = {temperature:g}", fontsize=10)
        ax.set_xlabel("логит")
    fig.suptitle("softmax(x / T)")
    fig.tight_layout()
    return fig
