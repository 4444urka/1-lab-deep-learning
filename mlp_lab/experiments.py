"""Сетка экспериментов «глубина × ширина» для MLP."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mlp_lab.config import ModelConfig, TrainConfig, set_seed
from mlp_lab.models import MLP, count_parameters
from mlp_lab.training import Trainer

HiddenDimsFn = Callable[[int, int], list[int]]


def constant_width(depth: int, width: int) -> list[int]:
    """Все скрытые слои одной ширины: ``[width] * depth``."""
    return [width] * depth


def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    **trainer_kwargs,
) -> Trainer:
    """Собирает MLP, обучает на ``data["train"]`` с валидацией на ``data["valid"]``.

    Сид ``train_config.seed`` фиксируется до создания модели, поэтому он задаёт
    и начальные веса, и порядок батчей — запуск полностью воспроизводим (на CPU).

    Возвращает ``Trainer``; история обучения доступна в ``trainer.history``.
    """
    set_seed(train_config.seed)
    trainer = Trainer(MLP(model_config), train_config, **trainer_kwargs)
    trainer.history = trainer.fit(*data["train"], *data["valid"])
    return trainer


def run_depth_width_grid(
    base_model: ModelConfig,
    base_train: TrainConfig,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    depths: Iterable[int],
    widths: Iterable[int],
    seeds: Iterable[int],
    hidden_dims_fn: HiddenDimsFn = constant_width,
    cache_path: str | Path | None = None,
    rerun: bool = False,
    **trainer_kwargs,
) -> pd.DataFrame:
    """Обучает модель для каждой комбинации (depth, width, seed).

    Остальные параметры берутся из ``base_model``/``base_train``. Метрики
    фиксируются на лучшей по valid loss эпохе. Если ``cache_path`` задан и файл
    существует (и ``rerun=False``), результаты читаются из него.

    Returns:
        DataFrame: по строке на запуск — depth, width, seed, n_params, best_epoch,
        n_epochs, fit_time_sec и ``train_*``/``valid_*`` для каждой метрики.
    """
    if cache_path is not None and Path(cache_path).exists() and not rerun:
        return pd.read_csv(cache_path)

    rows = []
    for depth, width, seed in itertools.product(depths, widths, seeds):
        model_cfg = replace(base_model, hidden_dims=hidden_dims_fn(depth, width))
        trainer = train_model(model_cfg, replace(base_train, seed=seed), data, **trainer_kwargs)
        history = trainer.history
        row = {
            "depth": depth,
            "width": width,
            "seed": seed,
            "n_params": count_parameters(trainer.model),
            "best_epoch": history.best_epoch,
            "n_epochs": history.n_epochs,
            "fit_time_sec": history.fit_time_sec,
        }
        for split in ("train", "valid"):
            for key in getattr(history, split):
                row[f"{split}_{key}"] = history.at_best(split, key)
        rows.append(row)

    results = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(cache_path, index=False)
    return results


def summarize_grid(results: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Среднее и std метрик по сидам для каждой пары (depth, width)."""
    metrics = list(metrics)
    summary = results.groupby(["depth", "width"])[metrics + ["n_params"]].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.drop(columns="n_params_std").rename(columns={"n_params_mean": "n_params"}).reset_index()
