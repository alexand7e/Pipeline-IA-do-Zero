"""Funções de perda (loss) usadas no treinamento."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bce_binaria(y_pred: NDArray, y_true: NDArray, eps: float = 1e-12) -> float:
    """Binary cross-entropy média.

    Espera y_pred e y_true com shape (n, 1). y_pred deve estar em (0, 1).
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred e y_true devem ter o mesmo shape.")
    y_pred_clip = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred_clip) + (1.0 - y_true) * np.log(1.0 - y_pred_clip))
    return float(np.mean(loss))


def grad_bce_com_sigmoid(y_pred: NDArray, y_true: NDArray) -> NDArray:
    """Gradiente simplificado para saída Sigmoid com BCE.

    Quando a última ativação é sigmoid e a perda é BCE, o gradiente em relação
    ao pré-ativação z da última camada simplifica para: (y_pred - y_true) / n.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred e y_true devem ter o mesmo shape.")
    n = y_true.shape[0]
    return (y_pred - y_true) / max(1, n)

