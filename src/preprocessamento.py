"""Funções de split e transformação (fit/transform) para o pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


def dividir_treino_validacao_teste(
    X: NDArray,
    y: NDArray,
    frac_treino: float = 0.7,
    frac_validacao: float = 0.15,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Divide (X, y) em treino/validação/teste com embaralhamento."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo número de amostras.")
    if not (0 < frac_treino < 1):
        raise ValueError("frac_treino deve estar em (0, 1).")
    if not (0 <= frac_validacao < 1):
        raise ValueError("frac_validacao deve estar em [0, 1).")
    if frac_treino + frac_validacao >= 1:
        raise ValueError("frac_treino + frac_validacao deve ser < 1.")

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    Xs, ys = X[idx], y[idx]

    n_treino = int(round(n * frac_treino))
    n_valid = int(round(n * frac_validacao))

    X_treino = Xs[:n_treino]
    y_treino = ys[:n_treino]

    X_valid = Xs[n_treino : n_treino + n_valid]
    y_valid = ys[n_treino : n_treino + n_valid]

    X_teste = Xs[n_treino + n_valid :]
    y_teste = ys[n_treino + n_valid :]

    return X_treino, y_treino, X_valid, y_valid, X_teste, y_teste


@dataclass(frozen=True)
class Padronizador:
    """Padronização Z-score: (X - média) / desvio."""

    media: NDArray
    desvio: NDArray

    @classmethod
    def ajustar(cls, X_treino: NDArray, eps: float = 1e-8) -> "Padronizador":
        if X_treino.ndim != 2:
            raise ValueError("X_treino deve ser uma matriz 2D.")
        media = X_treino.mean(axis=0, keepdims=True)
        desvio = X_treino.std(axis=0, keepdims=True)
        desvio = np.maximum(desvio, eps)
        return cls(media=media, desvio=desvio)

    def transformar(self, X: NDArray) -> NDArray:
        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D.")
        return (X - self.media) / self.desvio

