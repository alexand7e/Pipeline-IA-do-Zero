"""Orquestra os 7 passos básicos do pipeline em uma demo reproduzível."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from src.dados import gerar_dados_duas_luas
from src.metricas import acuracia_binaria, matriz_confusao_binaria
from src.modelo_mlp import MLP
from src.perdas import bce_binaria, grad_bce_com_sigmoid
from src.preprocessamento import Padronizador, dividir_treino_validacao_teste


@dataclass(frozen=True)
class ArtefatosPipeline:
    X_treino: NDArray
    y_treino: NDArray
    X_valid: NDArray
    y_valid: NDArray
    X_teste: NDArray
    y_teste: NDArray
    padronizador: Padronizador
    modelo: MLP
    historico: dict[str, list[float]]


def treinar_mlp(
    modelo: MLP,
    X_treino: NDArray,
    y_treino: NDArray,
    X_valid: NDArray,
    y_valid: NDArray,
    epochs: int = 200,
    taxa_aprendizado: float = 0.05,
    tamanho_lote: int = 64,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Treina uma MLP com BCE (binária) e SGD por mini-batch."""
    rng = np.random.default_rng(seed)
    n = X_treino.shape[0]
    historico: dict[str, list[float]] = {"loss_treino": [], "loss_valid": [], "acc_valid": []}

    for _epoca in range(epochs):
        idx = rng.permutation(n)
        Xs, ys = X_treino[idx], y_treino[idx]

        for ini in range(0, n, tamanho_lote):
            Xb = Xs[ini : ini + tamanho_lote]
            yb = ys[ini : ini + tamanho_lote]

            y_pred = modelo.forward(Xb)
            grad_z = grad_bce_com_sigmoid(y_pred, yb)
            modelo.backward(grad_z, grad_ultima_eh_pre_ativacao=True)
            modelo.passo_sgd(taxa_aprendizado)

        y_pred_treino = modelo.forward(X_treino)
        y_pred_valid = modelo.forward(X_valid)

        loss_treino = bce_binaria(y_pred_treino, y_treino)
        loss_valid = bce_binaria(y_pred_valid, y_valid)
        acc_valid = acuracia_binaria(y_pred_valid, y_valid)

        historico["loss_treino"].append(loss_treino)
        historico["loss_valid"].append(loss_valid)
        historico["acc_valid"].append(acc_valid)

    return historico


def executar_pipeline(
    n_amostras: int = 1200,
    ruido: float = 0.12,
    seed: int = 42,
    tamanhos_ocultos: list[int] | None = None,
    epochs: int = 200,
    taxa_aprendizado: float = 0.05,
    tamanho_lote: int = 64,
) -> tuple[ArtefatosPipeline, dict[str, object]]:
    """Executa a demo completa: dados -> preparo -> treino -> avaliação."""
    if tamanhos_ocultos is None:
        tamanhos_ocultos = [16, 16]

    X, y = gerar_dados_duas_luas(n_amostras=n_amostras, ruido=ruido, seed=seed)

    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = dividir_treino_validacao_teste(
        X, y, frac_treino=0.7, frac_validacao=0.15, seed=seed
    )

    pad = Padronizador.ajustar(X_treino)
    X_treino_p = pad.transformar(X_treino)
    X_valid_p = pad.transformar(X_valid)
    X_teste_p = pad.transformar(X_teste)

    modelo = MLP.criar(n_entrada=X_treino_p.shape[1], tamanhos_ocultos=tamanhos_ocultos, seed=seed)
    historico = treinar_mlp(
        modelo,
        X_treino_p,
        y_treino,
        X_valid_p,
        y_valid,
        epochs=epochs,
        taxa_aprendizado=taxa_aprendizado,
        tamanho_lote=tamanho_lote,
        seed=seed,
    )

    y_pred_teste = modelo.forward(X_teste_p)
    loss_teste = bce_binaria(y_pred_teste, y_teste)
    acc_teste = acuracia_binaria(y_pred_teste, y_teste)
    cm = matriz_confusao_binaria(y_pred_teste, y_teste)

    artefatos = ArtefatosPipeline(
        X_treino=X_treino_p,
        y_treino=y_treino,
        X_valid=X_valid_p,
        y_valid=y_valid,
        X_teste=X_teste_p,
        y_teste=y_teste,
        padronizador=pad,
        modelo=modelo,
        historico=historico,
    )

    relatorio: dict[str, object] = {
        "loss_teste": float(loss_teste),
        "acc_teste": float(acc_teste),
        "matriz_confusao": cm,
        "ultimas": {
            "loss_treino": historico["loss_treino"][-1],
            "loss_valid": historico["loss_valid"][-1],
            "acc_valid": historico["acc_valid"][-1],
        },
    }

    return artefatos, relatorio

