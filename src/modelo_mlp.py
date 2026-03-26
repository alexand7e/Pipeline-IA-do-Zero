"""Modelo MLP do zero: camadas densas, ativações e backprop."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray) -> NDArray:
    return np.maximum(0.0, z)


def grad_relu(z: NDArray) -> NDArray:
    return (z > 0.0).astype(np.float64)


def sigmoid(z: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-z))


def grad_sigmoid(z: NDArray) -> NDArray:
    s = sigmoid(z)
    return s * (1.0 - s)


def _inicializar_pesos(
    rng: np.random.Generator, n_entrada: int, n_saida: int, ativacao: str
) -> NDArray:
    if ativacao.lower() == "relu":
        escala = np.sqrt(2.0 / n_entrada)
    else:
        escala = np.sqrt(1.0 / n_entrada)
    return rng.normal(loc=0.0, scale=escala, size=(n_entrada, n_saida)).astype(np.float64)


@dataclass
class CamadaDensa:
    """Camada totalmente conectada: z = XW + b; a = ativacao(z)."""

    W: NDArray
    b: NDArray
    ativacao: str

    _X: NDArray | None = None
    _z: NDArray | None = None

    dW: NDArray | None = None
    db: NDArray | None = None

    def forward(self, X: NDArray) -> NDArray:
        self._X = X
        self._z = X @ self.W + self.b
        if self.ativacao == "relu":
            return relu(self._z)
        if self.ativacao == "sigmoid":
            return sigmoid(self._z)
        raise ValueError(f"Ativação desconhecida: {self.ativacao}")

    def backward(self, grad_saida: NDArray, *, grad_ja_eh_pre_ativacao: bool = False) -> NDArray:
        if self._X is None or self._z is None:
            raise RuntimeError("forward deve ser chamado antes de backward.")

        if grad_ja_eh_pre_ativacao:
            grad_z = grad_saida
        else:
            if self.ativacao == "relu":
                grad_z = grad_saida * grad_relu(self._z)
            elif self.ativacao == "sigmoid":
                grad_z = grad_saida * grad_sigmoid(self._z)
            else:
                raise ValueError(f"Ativação desconhecida: {self.ativacao}")

        self.dW = self._X.T @ grad_z
        self.db = np.sum(grad_z, axis=0, keepdims=True)
        return grad_z @ self.W.T

    def passo_sgd(self, taxa_aprendizado: float) -> None:
        if self.dW is None or self.db is None:
            raise RuntimeError("backward deve ser chamado antes de atualizar os pesos.")
        self.W -= taxa_aprendizado * self.dW
        self.b -= taxa_aprendizado * self.db


class MLP:
    """MLP mínima para classificação binária (saída Sigmoid)."""

    def __init__(self, camadas: list[CamadaDensa]) -> None:
        self.camadas = camadas

    @classmethod
    def criar(
        cls,
        n_entrada: int,
        tamanhos_ocultos: list[int],
        seed: int = 42,
    ) -> "MLP":
        rng = np.random.default_rng(seed)
        tamanhos = [n_entrada, *tamanhos_ocultos, 1]
        camadas: list[CamadaDensa] = []
        for i in range(len(tamanhos) - 1):
            n_in, n_out = tamanhos[i], tamanhos[i + 1]
            ativacao = "relu" if i < len(tamanhos) - 2 else "sigmoid"
            W = _inicializar_pesos(rng, n_in, n_out, ativacao)
            b = np.zeros((1, n_out), dtype=np.float64)
            camadas.append(CamadaDensa(W=W, b=b, ativacao=ativacao))
        return cls(camadas=camadas)

    def forward(self, X: NDArray) -> NDArray:
        saida = X
        for camada in self.camadas:
            saida = camada.forward(saida)
        return saida

    def backward(self, grad_ultima: NDArray, *, grad_ultima_eh_pre_ativacao: bool) -> None:
        grad = grad_ultima
        for i in range(len(self.camadas) - 1, -1, -1):
            camada = self.camadas[i]
            grad = camada.backward(
                grad,
                grad_ja_eh_pre_ativacao=(grad_ultima_eh_pre_ativacao and i == len(self.camadas) - 1),
            )

    def passo_sgd(self, taxa_aprendizado: float) -> None:
        for camada in self.camadas:
            camada.passo_sgd(taxa_aprendizado)

