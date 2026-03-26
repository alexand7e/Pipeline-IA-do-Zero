import numpy as np

from src.metricas import (
    acuracia_binaria,
    f1_binario,
    matriz_confusao_binaria,
    precisao_binaria,
    recall_binario,
)


def _make(y_pred_list, y_true_list):
    y_pred = np.array(y_pred_list, dtype=np.float64).reshape(-1, 1)
    y_true = np.array(y_true_list, dtype=np.float64).reshape(-1, 1)
    return y_pred, y_true


def test_acuracia_perfeita() -> None:
    y_pred, y_true = _make([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert acuracia_binaria(y_pred, y_true) == 1.0


def test_acuracia_pior_caso() -> None:
    y_pred, y_true = _make([0.9, 0.1, 0.8, 0.2], [0, 1, 0, 1])
    assert acuracia_binaria(y_pred, y_true) == 0.0


def test_precisao_sem_falso_positivo() -> None:
    # TP=2, FP=0 → precisão=1.0
    y_pred, y_true = _make([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
    assert precisao_binaria(y_pred, y_true) == 1.0


def test_precisao_com_falso_positivo() -> None:
    # pred=[1,1,1,0], true=[1,0,0,0] → TP=1, FP=2 → 1/3
    y_pred, y_true = _make([0.9, 0.8, 0.7, 0.1], [1, 0, 0, 0])
    assert abs(precisao_binaria(y_pred, y_true) - 1 / 3) < 1e-9


def test_recall_sem_falso_negativo() -> None:
    # TP=2, FN=0 → recall=1.0
    y_pred, y_true = _make([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
    assert recall_binario(y_pred, y_true) == 1.0


def test_recall_com_falso_negativo() -> None:
    # pred=[1,0,0,0], true=[1,1,1,0] → TP=1, FN=2 → 1/3
    y_pred, y_true = _make([0.9, 0.1, 0.2, 0.1], [1, 1, 1, 0])
    assert abs(recall_binario(y_pred, y_true) - 1 / 3) < 1e-9


def test_f1_perfeito() -> None:
    y_pred, y_true = _make([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
    assert f1_binario(y_pred, y_true) == 1.0


def test_f1_zero_quando_sem_predicoes_positivas() -> None:
    # nenhuma predição positiva → precisão=0 → F1=0
    y_pred, y_true = _make([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0])
    assert f1_binario(y_pred, y_true) == 0.0


def test_matriz_confusao_shape_e_valores() -> None:
    # pred=[1,0,1,0], true=[1,0,0,1] → TN=1, FP=1, FN=1, TP=1
    y_pred, y_true = _make([0.9, 0.1, 0.8, 0.2], [1, 0, 0, 1])
    cm = matriz_confusao_binaria(y_pred, y_true)
    assert cm.shape == (2, 2)
    assert cm[1, 1] == 1  # TP
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN
