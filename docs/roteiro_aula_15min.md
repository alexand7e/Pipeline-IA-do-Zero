# Roteiro de aula (15 min) — Pipeline básico de IA

## 0) Setup (30s)

- Mostrar o repositório e a ideia: “pipeline completo, do zero, em PT-BR”.
- Rodar (ou mostrar o output já rodado): `python scripts/rodar_pipeline.py`.

## 1) Problema e métrica (1 min)

- “O que estamos tentando prever?” → classe 0 vs classe 1.
- “Como sabemos se está bom?” → acurácia e matriz de confusão.
- Ponto didático: métrica vem antes do modelo.

## 2) Dados (2 min)

- Dataset sintético “duas luas”:
  - é não-linear → justifica rede neural
  - é 2D → fácil de visualizar
- Mostrar o arquivo: [dados.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/dados.py)

Frase-chave:
- “Sem dado, não existe modelo. Sem dado representativo, o modelo não generaliza.”

## 3) Preparação (2 min)

- Padronização Z-score:
  - ajusta apenas no treino
  - aplica em validação/teste
- Mostrar o arquivo: [preprocessamento.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

Gancho:
- “O mesmo pré-processamento precisa acontecer em produção.”

## 4) Split (1 min)

- Treino/validação/teste:
  - treino aprende
  - validação guia decisões
  - teste é a checagem final
- “Se eu mexo no teste o tempo todo, eu treino no teste sem perceber.”

## 5) Modelo e treino (4 min)

- MLP mínima:
  - ReLU nas camadas ocultas
  - Sigmoid na saída para probabilidade
- Treinamento:
  - loss: BCE
  - otimização: SGD em mini-batches
- Mostrar o arquivo: [modelo_mlp.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/modelo_mlp.py)
- Mostrar o orquestrador: [pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)

Frases-chave:
- “Forward calcula saída; backward calcula gradiente; SGD atualiza pesos.”
- “Rede neural é cálculo + otimização: nada mágico.”

## 6) Avaliação e análise de erro (2 min)

- Mostrar:
  - loss e acc no teste
  - matriz de confusão
- Discutir rapidamente FP/FN:
  - onde o modelo erra pode importar mais que a acurácia

## 7) Empacotamento e próximos passos (2 min)

- O que vai para produção:
  - pesos do modelo
  - parâmetros do pré-processamento
- Mostrar salvamento/carregamento:
  - [serializacao.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/serializacao.py)
  - `python scripts/inferir.py`

Próximos passos (citar, sem implementar):
- busca de hiperparâmetros
- calibração de probabilidade
- monitoramento de drift de dados
- re-treino e versionamento de modelos

