from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import executar_pipeline
from src.serializacao import salvar_pacote


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo completa do pipeline de IA.")
    p.add_argument("--n-amostras", type=int, default=1200, metavar="N",
                   help="Número de amostras do dataset (padrão: 1200)")
    p.add_argument("--epochs", type=int, default=250, metavar="E",
                   help="Épocas máximas de treino (padrão: 250)")
    p.add_argument("--lr", type=float, default=0.05, metavar="LR",
                   help="Taxa de aprendizado (padrão: 0.05)")
    p.add_argument("--lote", type=int, default=64, metavar="B",
                   help="Tamanho do mini-batch (padrão: 64)")
    p.add_argument("--paciencia", type=int, default=20, metavar="P",
                   help="Early stopping: épocas sem melhora (0=desativado, padrão: 20)")
    p.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    p.add_argument("--saida", type=Path, default=Path("artefatos"),
                   help="Diretório de saída para artefatos")
    p.add_argument("--verbose", action="store_true",
                   help="Exibe log por época (DEBUG)")
    return p.parse_args()


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
    args = _parse_args()

    nivel_log = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=nivel_log, format="%(levelname)s  %(name)s — %(message)s")

    artefatos, relatorio = executar_pipeline(
        n_amostras=args.n_amostras,
        ruido=0.12,
        seed=args.seed,
        tamanhos_ocultos=[16, 16],
        epochs=args.epochs,
        taxa_aprendizado=args.lr,
        tamanho_lote=args.lote,
        paciencia=args.paciencia,
    )

    print("\nPasso 6 — Avaliação (teste)")
    print(f"  épocas treinadas : {relatorio['epocas_treinadas']}")
    print(f"  loss             : {relatorio['loss_teste']:.6f}")
    print(f"  acurácia         : {relatorio['acc_teste']:.4f}")
    print(f"  precisão         : {relatorio['precisao_teste']:.4f}")
    print(f"  recall           : {relatorio['recall_teste']:.4f}")
    print(f"  F1-Score         : {relatorio['f1_teste']:.4f}")
    print("  matriz de confusão (TN FP / FN TP):")
    print(relatorio["matriz_confusao"])

    out_dir = args.saida
    salvar_pacote(artefatos.modelo, artefatos.padronizador, out_dir / "pacote_pipeline.json")
    _tentar_plotar_dados(artefatos.X_treino, artefatos.y_treino, out_dir / "dados_treino.png")

    print(f"\nPasso 7 — Empacotamento")
    print(f"  salvo em: {(out_dir / 'pacote_pipeline.json').as_posix()}")


if __name__ == "__main__":
    main()
