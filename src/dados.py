"""Geração de datasets sintéticos para demos de ML."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gerar_dados_duas_luas(
    n_amostras: int = 1000,
    ruido: float = 0.12,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Gera um dataset 2D do tipo 'duas luas' (classificação binária).

    Retorna:
        X: shape (n_amostras, 2)
        y: shape (n_amostras, 1) com valores 0.0 ou 1.0
    """
    if n_amostras < 2:
        raise ValueError("n_amostras deve ser >= 2.")
    if ruido < 0:
        raise ValueError("ruido deve ser >= 0.")

    rng = np.random.default_rng(seed)
    n0 = n_amostras // 2
    n1 = n_amostras - n0

    t0 = rng.uniform(0.0, np.pi, size=n0)
    x0 = np.cos(t0)
    y0 = np.sin(t0)

    t1 = rng.uniform(0.0, np.pi, size=n1)
    x1 = 1.0 - np.cos(t1)
    y1 = -np.sin(t1) - 0.5

    X0 = np.stack([x0, y0], axis=1)
    X1 = np.stack([x1, y1], axis=1)
    X = np.vstack([X0, X1]).astype(np.float64)

    y = np.vstack([np.zeros((n0, 1)), np.ones((n1, 1))]).astype(np.float64)

    if ruido > 0:
        X += rng.normal(loc=0.0, scale=ruido, size=X.shape)

    idx = rng.permutation(n_amostras)
    return X[idx], y[idx]

