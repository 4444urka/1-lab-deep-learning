"""Конфигурации модели и обучения, выбор устройства и фиксация сидов."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ModelConfig:
    """Описание архитектуры MLP.

    Attributes:
        input_dim: размерность входного вектора признаков.
        output_dim: размерность выхода (число классов или целевых переменных).
        hidden_dims: ширины скрытых слоёв; длина списка — глубина сети.
        activation: имя функции активации (см. ``mlp_lab.models.TORCH_ACTIVATIONS``).
        dropout: вероятность dropout после каждого скрытого слоя (0 — выключен).
        batch_norm: добавлять ли BatchNorm1d после каждого скрытого Linear.
    """

    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False


@dataclass
class TrainConfig:
    """Гиперпараметры обучения.

    Attributes:
        lr: шаг оптимизатора.
        weight_decay: L2-регуляризация (decoupled, AdamW).
        batch_size: размер мини-батча для обучения.
        eval_batch_size: размер батча при оценке (без градиентов можно больше).
        max_epochs: максимальное число эпох.
        patience: число эпох без улучшения valid loss до остановки;
            ``None`` — early stopping выключен, сохраняются веса последней эпохи.
        min_delta: минимальное улучшение valid loss, которое считается улучшением.
        use_class_weights: взвешивать ли loss обратно частоте классов.
        device: ``"cpu"`` | ``"mps"`` | ``"cuda"`` | ``"auto"``. По умолчанию CPU: для небольших
            табличных данных он быстрее MPS и даёт воспроизводимый результат.
        seed: сид для воспроизводимости.
        verbose: печатать прогресс каждые ``verbose`` эпох (0 — молча).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    eval_batch_size: int = 4096
    max_epochs: int = 300
    patience: int | None = 25
    min_delta: float = 0.0
    use_class_weights: bool = True
    device: str = "cpu"
    seed: int = 42
    verbose: int = 0


def resolve_device(name: str = "mps") -> torch.device:
    """Возвращает доступное устройство; при недоступности запрошенного — CPU.

    ``"auto"`` выбирает первое доступное из cuda → mps → cpu.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Фиксирует сиды Python, NumPy и PyTorch (включая MPS/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
