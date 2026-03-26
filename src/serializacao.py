"""Salvar e carregar o modelo treinado em um formato simples (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from src.modelo_mlp import CamadaDensa, MLP
from src.preprocessamento import Padronizador


def _ndarray_para_lista(x: NDArray) -> list[list[float]]:
    return x.astype(float).tolist()


def salvar_modelo(modelo: MLP, caminho: str | Path) -> None:
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tipo": "MLP",
        "camadas": [
            {
                "ativacao": camada.ativacao,
                "W": _ndarray_para_lista(camada.W),
                "b": _ndarray_para_lista(camada.b),
            }
            for camada in modelo.camadas
        ],
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_modelo(caminho: str | Path) -> MLP:
    payload = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if payload.get("tipo") != "MLP":
        raise ValueError("Arquivo não parece conter um modelo MLP válido.")

    camadas: list[CamadaDensa] = []
    for item in payload["camadas"]:
        W = np.array(item["W"], dtype=np.float64)
        b = np.array(item["b"], dtype=np.float64)
        ativacao = str(item["ativacao"])
        camadas.append(CamadaDensa(W=W, b=b, ativacao=ativacao))
    return MLP(camadas=camadas)


def salvar_pacote(modelo: MLP, padronizador: Padronizador, caminho: str | Path) -> None:
    """Salva modelo + preprocessamento para inferência consistente."""
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tipo": "PipelineMLP",
        "preprocessamento": {
            "media": _ndarray_para_lista(padronizador.media),
            "desvio": _ndarray_para_lista(padronizador.desvio),
        },
        "modelo": {
            "tipo": "MLP",
            "camadas": [
                {
                    "ativacao": camada.ativacao,
                    "W": _ndarray_para_lista(camada.W),
                    "b": _ndarray_para_lista(camada.b),
                }
                for camada in modelo.camadas
            ],
        },
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_pacote(caminho: str | Path) -> tuple[MLP, Padronizador]:
    payload = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if payload.get("tipo") != "PipelineMLP":
        raise ValueError("Arquivo não parece conter um pacote de pipeline válido.")

    pp = payload["preprocessamento"]
    media = np.array(pp["media"], dtype=np.float64)
    desvio = np.array(pp["desvio"], dtype=np.float64)
    pad = Padronizador(media=media, desvio=desvio)

    modelo_payload = payload["modelo"]
    if modelo_payload.get("tipo") != "MLP":
        raise ValueError("Pacote não contém um modelo MLP válido.")

    camadas: list[CamadaDensa] = []
    for item in modelo_payload["camadas"]:
        W = np.array(item["W"], dtype=np.float64)
        b = np.array(item["b"], dtype=np.float64)
        ativacao = str(item["ativacao"])
        camadas.append(CamadaDensa(W=W, b=b, ativacao=ativacao))
    return MLP(camadas=camadas), pad
