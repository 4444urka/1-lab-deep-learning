"""MLP, собираемый из ``ModelConfig``."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from mlp_lab.config import ModelConfig

TORCH_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


def build_activation(name: str) -> nn.Module:
    """Создаёт модуль активации по имени."""
    try:
        return TORCH_ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"Неизвестная активация {name!r}; доступны: {sorted(TORCH_ACTIVATIONS)}") from exc


class MLP(nn.Module):
    """Многослойный перцептрон произвольной глубины и ширины.

    Каждый скрытый блок: ``Linear → [BatchNorm1d] → activation → [Dropout]``.
    Выходной слой — ``Linear`` без активации (логиты), поэтому модель подходит
    и для классификации (с ``CrossEntropyLoss``), и для регрессии (с ``MSELoss``).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for width in config.hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(build_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Число параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
