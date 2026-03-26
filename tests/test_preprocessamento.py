import numpy as np

from src.preprocessamento import Padronizador, dividir_treino_validacao_teste


def test_dividir_treino_validacao_teste_tamanhos() -> None:
    X = np.zeros((100, 2), dtype=np.float64)
    y = np.zeros((100, 1), dtype=np.float64)
    X_tr, y_tr, X_va, y_va, X_te, y_te = dividir_treino_validacao_teste(
        X, y, frac_treino=0.7, frac_validacao=0.15, seed=1
    )
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_va.shape[0] == y_va.shape[0]
    assert X_te.shape[0] == y_te.shape[0]
    assert X_tr.shape[0] + X_va.shape[0] + X_te.shape[0] == 100


def test_padronizador_zscore_media_aproxima_zero_no_treino() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.normal(loc=10.0, scale=3.0, size=(200, 2)).astype(np.float64)
    pad = Padronizador.ajustar(X_tr)
    Xp = pad.transformar(X_tr)
    assert np.allclose(Xp.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Xp.std(axis=0), 1.0, atol=1e-10)

