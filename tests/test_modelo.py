import numpy as np

from src.dados import gerar_dados_duas_luas
from src.modelo_mlp import MLP
from src.perdas import bce_binaria
from src.pipeline import treinar_mlp
from src.preprocessamento import Padronizador, dividir_treino_validacao_teste


def test_mlp_forward_retorna_probabilidades() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2)).astype(np.float64)
    modelo = MLP.criar(n_entrada=2, tamanhos_ocultos=[8, 8], seed=0)
    y_pred = modelo.forward(X)
    assert y_pred.shape == (10, 1)
    assert np.all((y_pred > 0.0) & (y_pred < 1.0))


def test_treino_reduz_loss_validacao_em_media() -> None:
    X, y = gerar_dados_duas_luas(n_amostras=400, ruido=0.10, seed=3)
    X_tr, y_tr, X_va, y_va, _, _ = dividir_treino_validacao_teste(
        X, y, frac_treino=0.7, frac_validacao=0.2, seed=3
    )
    pad = Padronizador.ajustar(X_tr)
    X_tr = pad.transformar(X_tr)
    X_va = pad.transformar(X_va)

    modelo = MLP.criar(n_entrada=2, tamanhos_ocultos=[16, 16], seed=3)
    y_pred_ini = modelo.forward(X_va)
    loss_ini = bce_binaria(y_pred_ini, y_va)

    hist = treinar_mlp(
        modelo,
        X_tr,
        y_tr,
        X_va,
        y_va,
        epochs=60,
        taxa_aprendizado=0.05,
        tamanho_lote=64,
        seed=3,
    )
    assert hist["loss_valid"][0] > hist["loss_valid"][-1]

    y_pred_fim = modelo.forward(X_va)
    loss_fim = bce_binaria(y_pred_fim, y_va)
    assert loss_fim < loss_ini

