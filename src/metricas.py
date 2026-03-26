"""Métricas e utilitários de avaliação."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _contagens_binarias(
    y_pred: NDArray, y_true: NDArray, limiar: float = 0.5
) -> tuple[int, int, int, int]:
    """Retorna (TP, TN, FP, FN) para uso interno."""
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred e y_true devem ter o mesmo shape.")
    y_hat = (y_pred >= limiar).astype(np.int64).reshape(-1)
    y = y_true.astype(np.int64).reshape(-1)
    tp = int(np.sum((y_hat == 1) & (y == 1)))
    tn = int(np.sum((y_hat == 0) & (y == 0)))
    fp = int(np.sum((y_hat == 1) & (y == 0)))
    fn = int(np.sum((y_hat == 0) & (y == 1)))
    return tp, tn, fp, fn


def acuracia_binaria(y_pred: NDArray, y_true: NDArray, limiar: float = 0.5) -> float:
    """Acurácia para classificação binária com saída probabilística."""
    tp, tn, fp, fn = _contagens_binarias(y_pred, y_true, limiar)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precisao_binaria(y_pred: NDArray, y_true: NDArray, limiar: float = 0.5) -> float:
    """Precisão: TP / (TP + FP). Quão confiável é quando prediz positivo."""
    tp, _, fp, _ = _contagens_binarias(y_pred, y_true, limiar)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_binario(y_pred: NDArray, y_true: NDArray, limiar: float = 0.5) -> float:
    """Recall (sensibilidade): TP / (TP + FN). Quão bem encontra os positivos."""
    tp, _, _, fn = _contagens_binarias(y_pred, y_true, limiar)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_binario(y_pred: NDArray, y_true: NDArray, limiar: float = 0.5) -> float:
    """F1-Score: média harmônica entre precisão e recall."""
    p = precisao_binaria(y_pred, y_true, limiar)
    r = recall_binario(y_pred, y_true, limiar)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def matriz_confusao_binaria(
    y_pred: NDArray, y_true: NDArray, limiar: float = 0.5
) -> NDArray:
    """Retorna matriz 2x2: [[TN, FP], [FN, TP]]."""
    tp, tn, fp, fn = _contagens_binarias(y_pred, y_true, limiar)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

