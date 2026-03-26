from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from src.pipeline import executar_pipeline
from src.serializacao import carregar_pacote, salvar_pacote


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _carregar_ou_treinar_pacote(path: Path) -> tuple[object, object]:
    if path.exists():
        return carregar_pacote(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    artefatos, _ = executar_pipeline(
        n_amostras=1200,
        ruido=0.12,
        seed=42,
        tamanhos_ocultos=[16, 16],
        epochs=250,
        taxa_aprendizado=0.05,
        tamanho_lote=64,
    )
    salvar_pacote(artefatos.modelo, artefatos.padronizador, path)
    return carregar_pacote(path)


class ServicoInferencia:
    def __init__(self, caminho_pacote: Path) -> None:
        self.caminho_pacote = caminho_pacote
        self.modelo, self.pad = _carregar_ou_treinar_pacote(caminho_pacote)

    def prever(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xp = self.pad.transformar(X)
        prob = self.modelo.forward(Xp)
        classe = (prob >= 0.5).astype(int)
        return prob, classe


def criar_handler(servico: ServicoInferencia):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/health"):
                _json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "modelo": "MLP binária",
                        "caminho_pacote": str(servico.caminho_pacote.as_posix()),
                    },
                )
                return
            _json_response(self, 404, {"erro": "rota não encontrada"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                _json_response(self, 404, {"erro": "rota não encontrada"})
                return

            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                payload = json.loads(raw.decode("utf-8"))
                X = np.array(payload["X"], dtype=np.float64)
                if X.ndim != 2 or X.shape[1] != 2:
                    raise ValueError("X deve ter shape (n, 2).")
                prob, classe = servico.prever(X)
                _json_response(
                    self,
                    200,
                    {
                        "prob": prob.reshape(-1).astype(float).tolist(),
                        "classe": classe.reshape(-1).astype(int).tolist(),
                    },
                )
            except Exception as e:
                _json_response(self, 400, {"erro": str(e)})

        def log_message(self, _format: str, *_args) -> None:
            return

    return Handler


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    caminho_pacote = Path(os.environ.get("MODEL_PATH", "artefatos/pacote_pipeline.json"))

    servico = ServicoInferencia(caminho_pacote=caminho_pacote)
    httpd = ThreadingHTTPServer((host, port), criar_handler(servico))
    httpd.serve_forever()


if __name__ == "__main__":
    main()

