"""Загрузка и предобработка датасета SUPPORT2 (UCI id 880).

Модуль специфичен для задачи: здесь сосредоточены все решения о колонках,
таргете и заполнении пропусков. Общая часть — ``train_valid_test_split``.

Таргет ``sfdm2`` — функциональное состояние пациента через 2 месяца после
включения в исследование (порядковая шкала). Класс ``Coma or Intub`` (41 объект)
объединён с соседним ``SIP>=30`` в класс ``severe``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/880/data.csv"
NA_VALUES = ["NaN", ""]

TARGET = "sfdm2"
TARGET_MAP = {
    "no(M2 and SIP pres)": "no_disability",
    "adl>=4 (>=5 if sur)": "adl_ge4",
    "SIP>=30": "severe",
    "Coma or Intub": "severe",
    "<2 mo. follow-up": "died_lt2mo",
}
CLASS_NAMES = ["no_disability", "adl_ge4", "severe", "died_lt2mo"]

# Известны только после 3-го дня (исходы, длительность, стоимость, поздние решения).
LEAKAGE_COLS = ["death", "hospdead", "d.time", "slos", "charges", "totcst", "totmcst", "avtisst", "dnr", "dnrday"]
# Готовые прогнозы другой модели и врача — не учим сеть копировать чужие предсказания.
PREDICTION_COLS = ["surv2m", "surv6m", "prg2m", "prg6m"]
# Идентификатор и дубли: dzclass — огрубление dzgroup, adlp/adls сведены в adlsc.
REDUNDANT_COLS = ["id", "dzclass", "adlp", "adls"]

CATEGORICAL_COLS = ["sex", "race", "dzgroup", "income", "ca"]
BINARY_COLS = ["diabetes", "dementia"]
NUMERIC_COLS = [
    "age", "num.co", "edu", "scoma", "sps", "aps", "hday", "adlsc",
    "meanbp", "hrt", "resp", "temp", "sod", "pafi", "alb", "ph",
    "wblc", "bili", "crea", "bun", "urine", "glucose",
]
# Сильно скошенные вправо — перед стандартизацией применяем log1p.
LOG_COLS = ["wblc", "bili", "crea", "bun", "urine", "glucose"]
# «Нормальные» значения, рекомендованные авторами SUPPORT для пропусков в анализах:
# если анализ не делали, врач, скорее всего, считал показатель нормальным.
LAB_NORMAL_VALUES = {"alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01, "bun": 6.51, "wblc": 9.0, "urine": 2502.0}
# Для этих анализов рекомендаций нет — заполняем медианой, но флаг пропуска тоже добавляем.
LAB_MEDIAN_COLS = ["ph", "glucose"]


def download_support2(path: str | Path, url: str = DATA_URL) -> Path:
    """Скачивает CSV, если файла ещё нет. Возвращает путь к файлу."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_raw(path: str | Path) -> pd.DataFrame:
    """Читает исходный CSV; строка ``"NaN"`` трактуется как пропуск."""
    return pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)


def missing_indicator_cols() -> list[str]:
    """Имена бинарных флагов пропуска, которые добавляет ``prepare_frame``."""
    return [f"{col}_missing" for col in [*LAB_NORMAL_VALUES, *LAB_MEDIAN_COLS]]


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Очищает таблицу: выкидывает строки без таргета и лишние колонки,
    объединяет классы, добавляет флаги пропусков и заполняет анализы нормами.

    Заполнение константами из литературы не использует статистик данных, поэтому
    его можно делать до разбиения — утечки из valid/test в train нет. Медианы
    и масштабирование считаются позже, в ``build_preprocessor``, только по train.

    Returns:
        X: признаки (до кодирования), y: метки классов 0..3 в порядке ``CLASS_NAMES``.
    """
    df = raw.dropna(subset=[TARGET]).reset_index(drop=True)
    y = df[TARGET].map(TARGET_MAP).map({name: i for i, name in enumerate(CLASS_NAMES)}).astype(int)
    X = df.drop(columns=[TARGET, *LEAKAGE_COLS, *PREDICTION_COLS, *REDUNDANT_COLS])

    for col in [*LAB_NORMAL_VALUES, *LAB_MEDIAN_COLS]:
        X[f"{col}_missing"] = X[col].isna().astype(int)
    X = X.fillna(LAB_NORMAL_VALUES)
    return X[[*NUMERIC_COLS, *BINARY_COLS, *missing_indicator_cols(), *CATEGORICAL_COLS]], y.rename("target")


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: медианы, log1p, стандартизация, one-hot. Обучается только на train."""
    linear_cols = [col for col in NUMERIC_COLS if col not in LOG_COLS]
    return ColumnTransformer(
        [
            (
                "log_num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                LOG_COLS,
            ),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), linear_cols),
            ("bin", "passthrough", [*BINARY_COLS, *missing_indicator_cols()]),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )


def train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Делит данные на train/valid/test (по умолчанию 70/15/15) со стратификацией по ``y``."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=seed
    )
    valid_share = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rest, y_rest, test_size=valid_share, stratify=y_rest if stratify else None, random_state=seed
    )
    return {"train": (X_train, y_train), "valid": (X_valid, y_valid), "test": (X_test, y_test)}


def transform_splits(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]], preprocessor: ColumnTransformer
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Обучает препроцессор на train и применяет ко всем выборкам.

    Returns:
        словарь ``{split: (X float32, y int64)}`` и имена признаков после кодирования.
    """
    preprocessor.fit(splits["train"][0])
    arrays = {
        name: (preprocessor.transform(X).astype(np.float32), y.to_numpy(dtype=np.int64))
        for name, (X, y) in splits.items()
    }
    return arrays, list(preprocessor.get_feature_names_out())
