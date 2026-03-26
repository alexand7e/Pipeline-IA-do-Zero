from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import executar_pipeline
from src.serializacao import salvar_pacote


def _tentar_plotar_dados(X: np.ndarray, y: np.ndarray, caminho: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    y_flat = y.reshape(-1)
    ax.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], s=12, alpha=0.8, label="Classe 0")
    ax.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], s=12, alpha=0.8, label="Classe 1")
    ax.set_title("Dataset sintético: duas luas (padronizado)")
    ax.legend()
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def main() -> None:
    artefatos, relatorio = executar_pipeline(
        n_amostras=1200,
        ruido=0.12,
        seed=42,
        tamanhos_ocultos=[16, 16],
        epochs=250,
        taxa_aprendizado=0.05,
        tamanho_lote=64,
    )

    print("Passo 6 — Avaliação (teste)")
    print(f"- loss_teste: {relatorio['loss_teste']:.6f}")
    print(f"- acc_teste:  {relatorio['acc_teste']:.4f}")
    print("- matriz_confusao (TN FP / FN TP):")
    print(relatorio["matriz_confusao"])

    out_dir = Path("artefatos")
    salvar_pacote(artefatos.modelo, artefatos.padronizador, out_dir / "pacote_pipeline.json")
    _tentar_plotar_dados(artefatos.X_treino, artefatos.y_treino, out_dir / "dados_treino.png")

    print("Passo 7 — Empacotamento")
    print(f"- salvo em: {(out_dir / 'pacote_pipeline.json').as_posix()}")


if __name__ == "__main__":
    main()
