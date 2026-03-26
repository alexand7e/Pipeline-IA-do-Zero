"""Orquestra os 7 passos básicos do pipeline em uma demo reproduzível."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
from numpy.typing import NDArray

from src.dados import gerar_dados_duas_luas
from src.metricas import (
    acuracia_binaria,
    f1_binario,
    matriz_confusao_binaria,
    precisao_binaria,
    recall_binario,
)
from src.modelo_mlp import MLP
from src.perdas import bce_binaria, grad_bce_com_sigmoid
from src.preprocessamento import Padronizador, dividir_treino_validacao_teste

logger = logging.getLogger(__name__)


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
    paciencia: int = 20,
) -> dict[str, list[float]]:
    """Treina uma MLP com BCE (binária) e SGD por mini-batch.

    Args:
        paciencia: número de épocas sem melhora na loss de validação
                   antes de interromper o treino (early stopping).
                   Use 0 para desativar.
    """
    rng = np.random.default_rng(seed)
    n = X_treino.shape[0]
    historico: dict[str, list[float]] = {"loss_treino": [], "loss_valid": [], "acc_valid": []}

    melhor_loss_valid = float("inf")
    epocas_sem_melhora = 0

    for epoca in range(1, epochs + 1):
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

        logger.debug("época=%d  loss_treino=%.4f  loss_valid=%.4f  acc_valid=%.4f",
                     epoca, loss_treino, loss_valid, acc_valid)

        if paciencia > 0:
            if loss_valid < melhor_loss_valid:
                melhor_loss_valid = loss_valid
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= paciencia:
                    logger.info("Early stopping na época %d (paciência=%d)", epoca, paciencia)
                    break

    return historico


def executar_pipeline(
    n_amostras: int = 1200,
    ruido: float = 0.12,
    seed: int = 42,
    tamanhos_ocultos: list[int] | None = None,
    epochs: int = 200,
    taxa_aprendizado: float = 0.05,
    tamanho_lote: int = 64,
    paciencia: int = 20,
) -> tuple[ArtefatosPipeline, dict[str, object]]:
    """Executa a demo completa: dados -> preparo -> treino -> avaliação."""
    if tamanhos_ocultos is None:
        tamanhos_ocultos = [16, 16]

    logger.info("Passo 1 — gerando %d amostras (seed=%d, ruído=%.2f)", n_amostras, seed, ruido)
    X, y = gerar_dados_duas_luas(n_amostras=n_amostras, ruido=ruido, seed=seed)

    logger.info("Passo 2/3 — split e padronização")
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = dividir_treino_validacao_teste(
        X, y, frac_treino=0.7, frac_validacao=0.15, seed=seed
    )

    pad = Padronizador.ajustar(X_treino)
    X_treino_p = pad.transformar(X_treino)
    X_valid_p = pad.transformar(X_valid)
    X_teste_p = pad.transformar(X_teste)

    logger.info("Passo 4/5 — treino MLP %s (lr=%.3f, lote=%d, paciência=%d)",
                tamanhos_ocultos, taxa_aprendizado, tamanho_lote, paciencia)
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
        paciencia=paciencia,
    )

    logger.info("Passo 6 — avaliação no conjunto de teste (%d amostras)", X_teste_p.shape[0])
    y_pred_teste = modelo.forward(X_teste_p)
    loss_teste = bce_binaria(y_pred_teste, y_teste)
    acc_teste = acuracia_binaria(y_pred_teste, y_teste)
    prec_teste = precisao_binaria(y_pred_teste, y_teste)
    rec_teste = recall_binario(y_pred_teste, y_teste)
    f1_teste = f1_binario(y_pred_teste, y_teste)
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
        "precisao_teste": float(prec_teste),
        "recall_teste": float(rec_teste),
        "f1_teste": float(f1_teste),
        "matriz_confusao": cm,
        "epocas_treinadas": len(historico["loss_treino"]),
        "ultimas": {
            "loss_treino": historico["loss_treino"][-1],
            "loss_valid": historico["loss_valid"][-1],
            "acc_valid": historico["acc_valid"][-1],
        },
    }

    logger.info(
        "Resultado — loss=%.4f  acc=%.4f  precisão=%.4f  recall=%.4f  F1=%.4f  (épocas=%d)",
        loss_teste, acc_teste, prec_teste, rec_teste, f1_teste,
        relatorio["epocas_treinadas"],
    )

    return artefatos, relatorio

