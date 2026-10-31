"""Извлечение градиентов по слоям при инференсе одного объекта (или батча)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class LayerGradients:
    """Градиенты одного слоя.

    Attributes:
        name: имя модуля внутри модели (``named_modules``).
        weight_grad: dL/dW (или ``None``, если у слоя нет весов).
        bias_grad: dL/db (или ``None``).
        output: выход слоя на прямом проходе.
        output_grad: dL/d(выход слоя) — градиент, пришедший в слой сверху.
    """

    name: str
    weight_grad: np.ndarray | None
    bias_grad: np.ndarray | None
    output: np.ndarray
    output_grad: np.ndarray | None


@dataclass
class GradientReport:
    """Результат ``GradientExtractor.extract``."""

    loss: float
    outputs: np.ndarray
    input_grad: np.ndarray
    layers: dict[str, LayerGradients] = field(default_factory=dict)

    def norms(self) -> pd.DataFrame:
        """L2-нормы градиентов и активаций по слоям (в порядке прямого прохода)."""

        def _norm(array: np.ndarray | None) -> float:
            return float(np.linalg.norm(array)) if array is not None else np.nan

        return pd.DataFrame(
            [
                {
                    "layer": name,
                    "weight_grad_norm": _norm(layer.weight_grad),
                    "bias_grad_norm": _norm(layer.bias_grad),
                    "output_grad_norm": _norm(layer.output_grad),
                    "output_norm": _norm(layer.output),
                }
                for name, layer in self.layers.items()
            ]
        ).set_index("layer")


class GradientExtractor:
    """Собирает градиенты по всем слоям заданных типов для любой ``nn.Module``.

    Используется как context manager: hooks регистрируются при входе и
    гарантированно снимаются при выходе.

    Example:
        >>> with GradientExtractor(model) as extractor:
        ...     report = extractor.extract(x, target=torch.tensor([2]))
        >>> report.norms()

    Args:
        model: модель.
        layer_types: типы модулей, по которым собираются градиенты.
        loss_fn: функция потерь ``(outputs, target) -> scalar``; по умолчанию CrossEntropy.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_types: tuple[type[nn.Module], ...] = (nn.Linear,),
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model
        self.layer_types = layer_types
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._outputs: dict[str, torch.Tensor] = {}
        self._output_grads: dict[str, torch.Tensor] = {}

    def __enter__(self) -> GradientExtractor:
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types):
                self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_forward_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._outputs[name] = output.detach()
            if output.requires_grad:
                output.register_hook(lambda grad: self._output_grads.__setitem__(name, grad.detach()))

        return hook

    def extract(self, x: torch.Tensor, target: torch.Tensor | None = None) -> GradientReport:
        """Прямой и обратный проход для ``x``.

        Модель переводится в режим eval (dropout выключен, BatchNorm использует
        накопленную статистику). Если ``target`` не задан, градиент считается от
        логита предсказанного класса — «почему модель выбрала этот класс».
        """
        if not self._handles:
            raise RuntimeError("GradientExtractor нужно использовать внутри блока with")
        self._outputs.clear()
        self._output_grads.clear()
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = x.detach().to(device).clone().requires_grad_(True)
        outputs = self.model(x)
        if target is None:
            loss = outputs.gather(1, outputs.argmax(dim=1, keepdim=True)).sum()
        else:
            loss = self.loss_fn(outputs, target.to(device))
        loss.backward()

        modules = dict(self.model.named_modules())
        layers = {}
        for name, output in self._outputs.items():
            module = modules[name]
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            layers[name] = LayerGradients(
                name=name,
                weight_grad=_to_numpy(weight.grad) if weight is not None and weight.grad is not None else None,
                bias_grad=_to_numpy(bias.grad) if bias is not None and bias.grad is not None else None,
                output=_to_numpy(output),
                output_grad=_to_numpy(self._output_grads[name]) if name in self._output_grads else None,
            )

        report = GradientReport(
            loss=float(loss.item()),
            outputs=_to_numpy(outputs.detach()),
            input_grad=_to_numpy(x.grad),
            layers=layers,
        )
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)
        return report


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
