import numpy as np

from src.dados import gerar_dados_duas_luas


def test_gerar_dados_duas_luas_shapes_e_labels() -> None:
    X, y = gerar_dados_duas_luas(n_amostras=101, ruido=0.0, seed=123)
    assert X.shape == (101, 2)
    assert y.shape == (101, 1)
    assert set(np.unique(y.reshape(-1)).tolist()) <= {0.0, 1.0}


def test_gerar_dados_duas_luas_reprodutivel_com_seed() -> None:
    X1, y1 = gerar_dados_duas_luas(n_amostras=200, ruido=0.12, seed=7)
    X2, y2 = gerar_dados_duas_luas(n_amostras=200, ruido=0.12, seed=7)
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)

