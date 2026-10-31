"""Цикл обучения с early stopping и записью истории."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from mlp_lab.config import TrainConfig, resolve_device, set_seed

MetricFn = Callable[[np.ndarray, np.ndarray], float]
"""Метрика: ``(y_true, outputs) -> float``, где ``outputs`` — сырые выходы модели."""


def classification_metrics() -> dict[str, MetricFn]:
    """Набор метрик для классификации; предсказание — argmax по логитам."""

    def _wrap(fn: Callable[[np.ndarray, np.ndarray], float]) -> MetricFn:
        return lambda y_true, outputs: float(fn(y_true, outputs.argmax(axis=1)))

    return {
        "macro_f1": _wrap(lambda y, p: f1_score(y, p, average="macro")),
        "accuracy": _wrap(accuracy_score),
        "balanced_accuracy": _wrap(balanced_accuracy_score),
    }


def compute_class_weights(y: np.ndarray, n_classes: int | None = None) -> torch.Tensor:
    """Веса классов, обратные частоте: ``n_samples / (n_classes * count_c)``."""
    n_classes = n_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = len(y) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class History:
    """История обучения по эпохам.

    ``train``/``valid`` — словари ``{имя_метрики: [значение по эпохам]}``,
    ключ ``"loss"`` присутствует всегда.
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    valid: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = -1
    stopped_epoch: int = -1
    fit_time_sec: float = 0.0

    def log(self, split: str, values: Mapping[str, float]) -> None:
        store = getattr(self, split)
        for key, value in values.items():
            store.setdefault(key, []).append(float(value))

    def at_best(self, split: str, key: str) -> float:
        """Значение метрики на лучшей (по valid loss) эпохе."""
        return getattr(self, split)[key][self.best_epoch]

    @property
    def n_epochs(self) -> int:
        return len(self.train.get("loss", []))


class Trainer:
    """Обучает модель мини-батчами, оценивает на train/valid после каждой эпохи.

    Данные целиком переносятся на устройство один раз — для табличных данных
    это быстрее, чем ``DataLoader``. Лучшая эпоха выбирается по valid loss;
    при включённом early stopping (``patience``) в конце восстанавливаются её веса.

    Args:
        model: любой ``nn.Module``.
        config: гиперпараметры обучения.
        loss_fn: функция потерь; по умолчанию ``CrossEntropyLoss``
            (с весами классов, если ``config.use_class_weights``).
        metrics: метрики ``(y_true, outputs) -> float``; по умолчанию
            ``classification_metrics()``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        loss_fn: nn.Module | None = None,
        metrics: Mapping[str, MetricFn] | None = None,
    ):
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.metrics = dict(metrics) if metrics is not None else classification_metrics()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(np.array(array), dtype=dtype, device=self.device)

    def _target_dtype(self) -> torch.dtype:
        return torch.long if isinstance(self.loss_fn, nn.CrossEntropyLoss) else torch.float32

    def _resolve_loss(self, y_train: np.ndarray) -> None:
        if self.loss_fn is None:
            weight = compute_class_weights(y_train) if self.config.use_class_weights else None
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.loss_fn = self.loss_fn.to(self.device)

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Сырые выходы модели (логиты) в режиме eval."""
        self.model.eval()
        X_t = X if isinstance(X, torch.Tensor) else self._to_tensor(X, torch.float32)
        outputs = [
            self.model(X_t[i : i + self.config.eval_batch_size])
            for i in range(0, len(X_t), self.config.eval_batch_size)
        ]
        return torch.cat(outputs).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> dict[str, float]:
        """Loss и метрики на выборке."""
        X_t = X if isinstance(X, torch.Tensor) else self._to_tensor(X, torch.float32)
        y_t = y if isinstance(y, torch.Tensor) else self._to_tensor(y, self._target_dtype())
        outputs = self.predict(X_t)
        loss = self.loss_fn(torch.as_tensor(outputs, device=self.device), y_t).item()
        y_np = y_t.cpu().numpy()
        return {"loss": loss, **{name: fn(y_np, outputs) for name, fn in self.metrics.items()}}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> History:
        """Обучает модель и возвращает историю."""
        cfg = self.config
        set_seed(cfg.seed)
        self._resolve_loss(np.asarray(y_train))
        target_dtype = self._target_dtype()
        X_tr, y_tr = self._to_tensor(X_train, torch.float32), self._to_tensor(y_train, target_dtype)
        X_va, y_va = self._to_tensor(X_valid, torch.float32), self._to_tensor(y_valid, target_dtype)
        generator = torch.Generator().manual_seed(cfg.seed)

        history = History()
        best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
        start = time.perf_counter()
        for epoch in range(cfg.max_epochs):
            self.model.train()
            for idx in torch.randperm(len(X_tr), generator=generator).split(cfg.batch_size):
                idx = idx.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model(X_tr[idx]), y_tr[idx])
                loss.backward()
                self.optimizer.step()

            history.log("train", self.evaluate(X_tr, y_tr))
            valid_scores = self.evaluate(X_va, y_va)
            history.log("valid", valid_scores)

            if valid_scores["loss"] < best_loss - cfg.min_delta:
                best_loss, history.best_epoch, epochs_without_improvement = valid_scores["loss"], epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1

            if cfg.verbose and (epoch + 1) % cfg.verbose == 0:
                print(f"epoch {epoch + 1:4d}  train_loss={history.train['loss'][-1]:.4f}  valid_loss={valid_scores['loss']:.4f}")
            if cfg.patience is not None and epochs_without_improvement >= cfg.patience:
                break

        history.stopped_epoch = epoch
        history.fit_time_sec = time.perf_counter() - start
        if cfg.patience is not None and best_state is not None:
            self.model.load_state_dict(best_state)
        return history
