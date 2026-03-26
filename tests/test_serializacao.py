import numpy as np

from src.pipeline import executar_pipeline
from src.serializacao import carregar_pacote, salvar_pacote


def test_salvar_e_carregar_pacote_preserva_predicoes(tmp_path) -> None:
    artefatos, _ = executar_pipeline(
        n_amostras=300,
        ruido=0.12,
        seed=5,
        tamanhos_ocultos=[8, 8],
        epochs=40,
        taxa_aprendizado=0.05,
        tamanho_lote=64,
    )

    caminho = tmp_path / "pacote.json"
    salvar_pacote(artefatos.modelo, artefatos.padronizador, caminho)

    modelo2, pad2 = carregar_pacote(caminho)
    X = artefatos.X_teste[:25]
    y1 = artefatos.modelo.forward(X)
    y2 = modelo2.forward(X)

    assert np.allclose(pad2.media, artefatos.padronizador.media)
    assert np.allclose(pad2.desvio, artefatos.padronizador.desvio)
    assert np.allclose(y1, y2)

