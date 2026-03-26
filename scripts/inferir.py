from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serializacao import carregar_pacote


def main() -> None:
    pacote = Path("artefatos") / "pacote_pipeline.json"
    modelo, pad = carregar_pacote(pacote)

    X_novo = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [-0.8, 0.9],
        ],
        dtype=np.float64,
    )

    X_novo_p = pad.transformar(X_novo)
    prob = modelo.forward(X_novo_p)
    pred = (prob >= 0.5).astype(int)

    print(f"Pacote carregado: {pacote.as_posix()}")
    for i in range(X_novo.shape[0]):
        print(f"- X={X_novo[i].tolist()} -> prob={float(prob[i, 0]):.4f} -> classe={int(pred[i, 0])}")


if __name__ == "__main__":
    main()
