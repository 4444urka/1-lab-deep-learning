"""mlp_lab — переиспользуемые компоненты для обучения и исследования MLP.

Модули:
    config       — dataclass-конфиги модели и обучения, выбор устройства, сиды;
    models       — MLP, собираемый из конфига;
    training     — Trainer с early stopping и историей обучения;
    experiments  — сетка экспериментов «глубина × ширина»;
    gradients    — извлечение градиентов по слоям через hooks;
    activations  — функции активации и их производные на numpy;
    plots        — графики (EDA, обучение, сетка, градиенты, активации);
    data         — загрузка и предобработка датасета SUPPORT2 (специфично для задачи).
"""

from mlp_lab.config import ModelConfig, TrainConfig, resolve_device, set_seed
from mlp_lab.models import MLP, count_parameters
from mlp_lab.training import History, Trainer, classification_metrics, compute_class_weights

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "resolve_device",
    "set_seed",
    "MLP",
    "count_parameters",
    "History",
    "Trainer",
    "classification_metrics",
    "compute_class_weights",
]
