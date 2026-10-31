"""Функции активации и их производные на NumPy.

Все функции зарегистрированы в реестре и доступны по имени::

    act = get_activation("leaky_relu", alpha=0.1)
    y, dy = act(x), act.derivative(x)

Новая активация добавляется декоратором ``@register("name", derivative=...)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import erf

ArrayFn = Callable[..., np.ndarray]

SELU_ALPHA = 1.6732632423543772
SELU_SCALE = 1.0507009873554805


@dataclass(frozen=True)
class Activation:
    """Функция активации с производной и параметрами по умолчанию.

    Attributes:
        name: имя в реестре.
        fn: ``f(x, **params)``.
        derivative_fn: ``f'(x, **params)``; для неэлементных функций (softmax) — якобиан.
        params: параметры (например, ``alpha`` у LeakyReLU).
        elementwise: применяется ли функция к каждому элементу независимо.
    """

    name: str
    fn: ArrayFn
    derivative_fn: ArrayFn
    params: dict[str, Any] = field(default_factory=dict)
    elementwise: bool = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(np.asarray(x, dtype=np.float64), **self.params)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.derivative_fn(np.asarray(x, dtype=np.float64), **self.params)


_REGISTRY: dict[str, Activation] = {}


def register(name: str, derivative: ArrayFn, elementwise: bool = True, **default_params) -> Callable[[ArrayFn], ArrayFn]:
    """Декоратор: регистрирует функцию активации вместе с производной."""

    def decorator(fn: ArrayFn) -> ArrayFn:
        _REGISTRY[name] = Activation(name, fn, derivative, dict(default_params), elementwise)
        return fn

    return decorator


def get_activation(name: str, **params) -> Activation:
    """Возвращает активацию по имени; ``params`` переопределяют значения по умолчанию."""
    try:
        base = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Неизвестная активация {name!r}; доступны: {list_activations()}") from exc
    unknown = set(params) - set(base.params)
    if unknown:
        raise ValueError(f"У {name!r} нет параметров {sorted(unknown)}; допустимые: {sorted(base.params)}")
    return Activation(base.name, base.fn, base.derivative_fn, {**base.params, **params}, base.elementwise)


def list_activations(elementwise: bool | None = None) -> list[str]:
    """Имена зарегистрированных активаций (опционально — только (не)элементные)."""
    return [name for name, act in _REGISTRY.items() if elementwise is None or act.elementwise == elementwise]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Численно устойчивая форма: не переполняется exp при больших |x|.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _std_normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


@register("linear", derivative=lambda x: np.ones_like(x))
def linear(x: np.ndarray) -> np.ndarray:
    """Тождественная функция: f(x) = x."""
    return x.copy()


@register("step", derivative=lambda x: np.zeros_like(x))
def step(x: np.ndarray) -> np.ndarray:
    """Ступенчатая функция Хевисайда: 1 при x > 0, иначе 0. Производная 0 (в нуле не определена)."""
    return (x > 0).astype(np.float64)


@register("sigmoid", derivative=lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)))
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Логистическая сигмоида: 1 / (1 + e^-x), диапазон (0, 1)."""
    return _sigmoid(x)


@register("tanh", derivative=lambda x: 1.0 - np.tanh(x) ** 2)
def tanh(x: np.ndarray) -> np.ndarray:
    """Гиперболический тангенс, диапазон (-1, 1), центрирован в нуле."""
    return np.tanh(x)


@register("relu", derivative=lambda x: (x > 0).astype(np.float64))
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x)."""
    return np.maximum(0.0, x)


@register("leaky_relu", derivative=lambda x, alpha: np.where(x > 0, 1.0, alpha), alpha=0.01)
def leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    """Leaky ReLU: x при x > 0, иначе alpha·x."""
    return np.where(x > 0, x, alpha * x)


@register("prelu", derivative=lambda x, alpha: np.where(x > 0, 1.0, alpha), alpha=0.25)
def prelu(x: np.ndarray, alpha: float) -> np.ndarray:
    """PReLU: как Leaky ReLU, но alpha — обучаемый параметр (здесь задаётся явно)."""
    return np.where(x > 0, x, alpha * x)


@register("elu", derivative=lambda x, alpha: np.where(x > 0, 1.0, alpha * np.exp(np.minimum(x, 0.0))), alpha=1.0)
def elu(x: np.ndarray, alpha: float) -> np.ndarray:
    """ELU: x при x > 0, иначе alpha·(e^x − 1)."""
    return np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0.0)))


@register(
    "selu",
    derivative=lambda x: SELU_SCALE * np.where(x > 0, 1.0, SELU_ALPHA * np.exp(np.minimum(x, 0.0))),
)
def selu(x: np.ndarray) -> np.ndarray:
    """SELU: scale·ELU(x, alpha) с константами, дающими самонормализацию."""
    return SELU_SCALE * np.where(x > 0, x, SELU_ALPHA * np.expm1(np.minimum(x, 0.0)))


@register("gelu", derivative=lambda x: _std_normal_cdf(x) + x * _std_normal_pdf(x))
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU: x·Φ(x), где Φ — функция распределения N(0, 1)."""
    return x * _std_normal_cdf(x)


@register("silu", derivative=lambda x: _sigmoid(x) * (1.0 + x * (1.0 - _sigmoid(x))))
def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x·sigmoid(x)."""
    return x * _sigmoid(x)


def _mish_derivative(x: np.ndarray) -> np.ndarray:
    tanh_sp = np.tanh(_softplus(x))
    return tanh_sp + x * _sigmoid(x) * (1.0 - tanh_sp**2)


@register("mish", derivative=_mish_derivative)
def mish(x: np.ndarray) -> np.ndarray:
    """Mish: x·tanh(softplus(x))."""
    return x * np.tanh(_softplus(x))


@register("softplus", derivative=lambda x: _sigmoid(x))
def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus: ln(1 + e^x) — гладкая версия ReLU."""
    return _softplus(x)


@register("softsign", derivative=lambda x: 1.0 / (1.0 + np.abs(x)) ** 2)
def softsign(x: np.ndarray) -> np.ndarray:
    """Softsign: x / (1 + |x|), диапазон (-1, 1), насыщается медленнее tanh."""
    return x / (1.0 + np.abs(x))


@register("hard_sigmoid", derivative=lambda x: np.where((x > -3.0) & (x < 3.0), 1.0 / 6.0, 0.0))
def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """Hard sigmoid: кусочно-линейная аппроксимация сигмоиды, clip(x/6 + 1/2, 0, 1)."""
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


@register(
    "hard_tanh",
    derivative=lambda x, min_val, max_val: np.where((x > min_val) & (x < max_val), 1.0, 0.0),
    min_val=-1.0,
    max_val=1.0,
)
def hard_tanh(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Hard tanh: clip(x, min_val, max_val)."""
    return np.clip(x, min_val, max_val)


@register("relu6", derivative=lambda x: np.where((x > 0.0) & (x < 6.0), 1.0, 0.0))
def relu6(x: np.ndarray) -> np.ndarray:
    """ReLU6: min(max(0, x), 6)."""
    return np.clip(x, 0.0, 6.0)


def _softmax_jacobian(x: np.ndarray, axis: int) -> np.ndarray:
    s = np.moveaxis(softmax(x, axis), axis, -1)
    return s[..., :, None] * (np.eye(s.shape[-1]) - s[..., None, :])


@register("softmax", derivative=_softmax_jacobian, elementwise=False, axis=-1)
def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Softmax: e^{x_i} / Σ_j e^{x_j} по оси ``axis``. Производная — якобиан diag(s) − s·sᵀ."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def torch_reference(name: str, **params) -> Callable:
    """Эталонная реализация из PyTorch (для сверки). Импорт torch ленивый."""
    import torch
    import torch.nn.functional as F

    act = get_activation(name, **params)
    p = act.params
    references = {
        "linear": lambda t: t,
        # Сравнение не дифференцируемо: autograd не строит граф, градиент считаем нулевым.
        "step": lambda t: (t > 0).to(t.dtype),
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "leaky_relu": lambda t: F.leaky_relu(t, negative_slope=p.get("alpha", 0.0)),
        "prelu": lambda t: F.prelu(t, torch.tensor([p.get("alpha", 0.0)], dtype=t.dtype)),
        "elu": lambda t: F.elu(t, alpha=p.get("alpha", 1.0)),
        "selu": F.selu,
        "gelu": F.gelu,
        "silu": F.silu,
        "mish": F.mish,
        "softplus": F.softplus,
        "softsign": F.softsign,
        "hard_sigmoid": F.hardsigmoid,
        "hard_tanh": lambda t: F.hardtanh(t, p.get("min_val", -1.0), p.get("max_val", 1.0)),
        "relu6": F.relu6,
        "softmax": lambda t: F.softmax(t, dim=p.get("axis", -1)),
    }
    return references[name]


def verify_against_torch(names: list[str] | None = None, n_points: int = 10_000, seed: int = 0, atol: float = 1e-6):
    """Сверяет значения и производные с PyTorch (autograd) на случайных точках.

    Точки берутся из U(-6, 6): вероятность попасть точно в точку излома
    (где производная определяется по соглашению) равна нулю.

    Returns:
        DataFrame: по строке на активацию — максимальные абсолютные ошибки и флаги совпадения.
    """
    import pandas as pd
    import torch

    rng = np.random.default_rng(seed)
    rows = []
    for name in names or list_activations():
        act = get_activation(name)
        x = rng.uniform(-6, 6, size=(n_points // 10, 10) if not act.elementwise else n_points)
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y_t = torch_reference(name)(x_t)
        value_err = float(np.abs(act(x) - y_t.detach().numpy()).max())

        if act.elementwise:
            grad_t = torch.autograd.grad(y_t.sum(), x_t)[0].numpy() if y_t.requires_grad else np.zeros_like(x)
            grad_err = float(np.abs(act.derivative(x) - grad_t).max())
        else:
            jac_t = torch.stack([torch.autograd.functional.jacobian(torch_reference(name), row) for row in x_t.detach()])
            grad_err = float(np.abs(act.derivative(x) - jac_t.numpy()).max())

        rows.append(
            {
                "activation": name,
                "value_max_abs_err": value_err,
                "grad_max_abs_err": grad_err,
                "value_ok": value_err < atol,
                "grad_ok": grad_err < atol,
            }
        )
    return pd.DataFrame(rows).set_index("activation")
