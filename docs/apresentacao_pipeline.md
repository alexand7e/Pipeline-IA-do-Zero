# Pipeline básico de um projeto de IA (do zero) — guia didático

Este documento explica, de forma direta, **tudo o que foi implementado** no projeto **Pipeline-IA-do-Zero**, organizado na sequência lógica de um **pipeline real** de Machine Learning.

O projeto é intencionalmente “do zero” (NumPy) para deixar as ideias claras: **dados → preparo → treino → avaliação → pacote de inferência → serviço**.

Referências do projeto:

- Mapa dos 7 passos: [pipeline\_7\_passos.md](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docs/pipeline_7_passos.md)
- Roteiro curto (15 min): [roteiro\_aula\_15min.md](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docs/roteiro_aula_15min.md)

***

## O “problema real” (e a versão didática usada aqui)

Em projetos reais, você quase sempre quer prever **uma decisão binária**:

- fraude vs não fraude
- churn vs não churn
- aprovado vs reprovado

No mundo real, as features podem ser dezenas/centenas. Aqui, para ficar visual e didático, usamos **apenas 2 features** e um dataset sintético **não-linear** (“duas luas”).

Pense assim:

- Feature 1 e Feature 2 representam dois sinais reais (ex.: valor e horário; renda e idade; cliques e tempo de sessão)
- O objetivo é prever `classe ∈ {0, 1}`

***

## Visão geral do pipeline (7 passos)

Fluxo (simplificado):

```text
        ┌───────────────┐
        │ 1) Problema   │  define: alvo + métricas (acc, F1, recall…)
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 2) Dados       │  gera/obtém X, y  (seed → reprodutível)
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 3) Preparo     │  padronização Z-score (fit só no treino)
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 4) Split       │  70% treino / 15% val / 15% teste
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 5) Treino      │  MLP + BCE + SGD + early stopping
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 6) Avaliação   │  acc · precisão · recall · F1 · cm
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 7) Empacotar   │  modelo + preprocessamento → JSON
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ Produção       │  script / HTTP + Docker / logging
        └───────────────┘
```

Onde acontece no código:

- Orquestração do pipeline: [pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- Demo completa (treina/avalia/salva pacote): [rodar\_pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/scripts/rodar_pipeline.py)

***

## Convenções usadas no projeto (inputs/outputs)

Representação de dados:

- `X`: matriz de features com shape `(n, 2)`
- `y`: rótulo binário com shape `(n, 1)` e valores `0.0` ou `1.0`
- `prob`: saída do modelo com shape `(n, 1)` e valores em `(0, 1)`
- `classe`: `prob >= 0.5`

Exemplo (mini-batch):

```text
Xb: (64, 2)  ->  modelo.forward  ->  y_pred: (64, 1)
yb: (64, 1)  ->  loss (BCE)      ->  gradiente para backward
```

***

## Passo 1) Definir problema e métrica

Objetivo:

- Decidir **o que prever** (target) e **como avaliar** (métrica)

No projeto:

- Problema: **classificação binária**
- Métrica principal: **acurácia**
- Métricas complementares: **precisão, recall e F1-Score**
- Análise de erro: **matriz de confusão** (para entender FP/FN)

Representação simplificada:

```text
prob >= 0.5  → classe 1
prob <  0.5  → classe 0

matriz 2x2: [[TN, FP],
            [FN, TP]]

precisão = TP / (TP + FP)   → confiança quando prediz positivo
recall   = TP / (TP + FN)   → cobertura dos positivos reais
F1       = 2 * P * R / (P + R)
```

Ponto didático: acurácia sozinha engana em datasets desbalanceados.
Se a classe 1 ocorre 5% do tempo, um modelo que diz "0 sempre" tem 95% de acurácia — e zero de recall.

Código:

- [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

***

## Passo 2) Obter dados

Objetivo:

- Construir/obter um dataset com `X` e `y` de forma reproduzível

No projeto:

- Dataset sintético “duas luas”, com controle de:
  - `n_amostras`
  - `ruido`
  - `seed`

Representação simplificada do gerador:

```text
gerar_dados_duas_luas(n, ruido, seed)
  cria meia-lua classe 0
  cria meia-lua classe 1 (deslocada)
  adiciona ruído gaussiano
  embaralha
  retorna X (n,2), y (n,1)
```

Código:

- [gerar\_dados\_duas\_luas](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/dados.py)

***

## Passo 3) Preparar os dados (pré-processamento)

Objetivo:

- Aplicar transformações **consistentes** entre treino e produção

No projeto:

- Padronização Z-score:
  - `fit` no treino (calcula média e desvio)
  - `transform` em treino/val/teste com os mesmos parâmetros

Representação simplificada:

```text
ajustar(X_treino):
  media = mean(X_treino)
  desvio = std(X_treino)

transformar(X):
  (X - media) / desvio
```

Ponto didático (evitar vazamento):

- Se você ajusta a transformação usando validação/teste, você “olha” o futuro sem perceber.

Código:

- [Padronizador](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

***

## Passo 4) Dividir em treino/validação/teste

Objetivo:

- Medir generalização de forma honesta

No projeto:

- Embaralhamento com `seed` e split:
  - treino: 70%
  - validação: 15%
  - teste: 15%

Representação simplificada:

```text
idx = permutacao(seed)
(X, y) = (X[idx], y[idx])

n_treino = round(n * frac_treino)
n_valid  = round(n * frac_validacao)

treino: [0 : n_treino)
valid : [n_treino : n_treino+n_valid)
teste : [n_treino+n_valid : n)
```

Código:

- [dividir\_treino\_validacao\_teste](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

***

## Passo 5) Treinar o modelo

Objetivo:

- Aprender uma função `X → prob(classe=1)` a partir de exemplos

No projeto:

- Modelo: **MLP** mínima (camadas densas)
  - ReLU nas camadas ocultas
  - Sigmoid na saída
- Loss: **BCE** (binary cross-entropy)
- Otimização: **SGD** com mini-batches

Representação simplificada do modelo:

```text
X (n,2)
  → Dense(2 → 16) + ReLU
  → Dense(16 → 16) + ReLU
  → Dense(16 → 1)  + Sigmoid
  = prob (n,1)
```

Representação simplificada do loop de treino:

```text
melhor_loss_valid = ∞
epocas_sem_melhora = 0

para cada época (até epochs ou early stopping):
  embaralhar treino
  para cada mini-batch:
    y_pred = forward(Xb)
    grad_z_ultimo = (y_pred - yb) / n
    backward(grad_z_ultimo)
    sgd_update(taxa_aprendizado)

  medir loss_treino, loss_valid, acc_valid
  registrar no histórico (logging)

  se loss_valid melhorou:
    melhor_loss_valid = loss_valid
    epocas_sem_melhora = 0
  senão:
    epocas_sem_melhora += 1
    se epocas_sem_melhora >= paciencia:
      parar (early stopping)
```

Early stopping (`paciencia`):

- Evita overfitting sem precisar acertar `epochs` na mão
- Configurável: `--paciencia 0` desativa
- Produz `epocas_treinadas` no relatório — você vê quando o modelo convergiu

Onde está cada peça:

- Forward/backward e SGD: [modelo\_mlp.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/modelo_mlp.py)
- BCE e gradiente simplificado (BCE + Sigmoid): [perdas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/perdas.py)
- Função de treino do pipeline: [treinar\_mlp](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)

***

## Passo 6) Avaliar e analisar erros

Objetivo:

- Medir performance no **teste**
- Entender *como* o modelo erra (não só “quanto”)

No projeto:

- Relatório completo de saída:
  - `loss_teste` — BCE no conjunto de teste
  - `acc_teste` — acurácia
  - `precisao_teste` — confiança nas predições positivas
  - `recall_teste` — cobertura dos positivos reais
  - `f1_teste` — balanço precisão/recall
  - `matriz_confusao` — mapa de erros 2×2
  - `epocas_treinadas` — indica quando o early stopping atuou

Representação simplificada:

```text
prob_teste = modelo.forward(X_teste_padronizado)

acc      = (TP + TN) / total
precisão = TP / (TP + FP)
recall   = TP / (TP + FN)
F1       = 2 * precisão * recall / (precisão + recall)

cm = [[TN, FP],
      [FN, TP]]
```

Leitura rápida da matriz de confusão:

- **FP** (alarme falso): modelo disse 1, era 0 — custoso em spam/fraude
- **FN** (silêncio falso): modelo disse 0, era 1 — custoso em diagnóstico/segurança

Código:

- Avaliação na execução do pipeline: [executar\_pipeline](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- Métricas: [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

***

## Passo 7) Empacotar para inferência (produção)

Objetivo:

- Garantir que o modelo possa prever fora do notebook/script
- Levar para produção **modelo + pré-processamento** (não apenas pesos)

No projeto:

- Salvamos um JSON (“pacote de pipeline”) com:
  - média e desvio do padronizador
  - pesos e vieses de cada camada

Representação simplificada do pacote:

```json
{
  "tipo": "PipelineMLP",
  "preprocessamento": {
    "media": [[...]],
    "desvio": [[...]]
  },
  "modelo": {
    "tipo": "MLP",
    "camadas": [
      { "ativacao": "relu",    "W": [[...]], "b": [[...]] },
      { "ativacao": "relu",    "W": [[...]], "b": [[...]] },
      { "ativacao": "sigmoid", "W": [[...]], "b": [[...]] }
    ]
  }
}
```

Código:

- Salvar/carregar pacote: [serializacao.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/serializacao.py)
- Exemplo de inferência lendo pacote: [inferir.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/scripts/inferir.py)

***

## “Produção” no projeto: inferência via HTTP (e Docker)

Objetivo:

- Simular como esse pipeline vira um serviço de inferência

No projeto:

- Serviço HTTP minimalista:
  - `GET /health` — healthcheck (para orquestradores como K8s/Compose)
  - `POST /predict` com JSON `{ “X”: [[x1, x2], ...] }`
- Se não existir pacote salvo, treina e salva automaticamente; caso exista, apenas carrega.
- Logging ativo: cada requisição e evento importante é registrado no stdout do container.

Representação simplificada do `/predict`:

```text
recebe X (n,2)
Xp = pad.transformar(X)
prob = modelo.forward(Xp)
classe = prob >= 0.5
retorna { “prob”: [...], “classe”: [...] }
```

Logs que o servidor emite:

```text
INFO  __main__ — Pacote encontrado em artefatos/... — carregando.
INFO  __main__ — Servidor iniciado em http://0.0.0.0:8000
INFO  __main__ — HTTP GET /health
INFO  __main__ — HTTP POST /predict
```

Variáveis de ambiente configuráveis (sem rebuildar a imagem):

| Variável     | Padrão                           | O que controla                   |
| ------------ | -------------------------------- | -------------------------------- |
| `PORT`       | `8000`                           | Porta interna do servidor        |
| `MODEL_PATH` | `artefatos/pacote_pipeline.json` | Caminho do pacote de inferência  |
| `PACIENCIA`  | `20`                             | Early stopping no treino inicial |

Código:

- Serviço HTTP: [serve.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/serve.py)

Infra:

- Dockerfile: [Dockerfile](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/Dockerfile)
- Compose: [docker-compose.yml](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docker-compose.yml)

Exemplo de request:

```json
{
  “X”: [
    [0.0, 0.0],
    [1.0, 0.2],
    [-0.8, 0.9]
  ]
}
```

***

## Qualidade e confiabilidade: testes automatizados

Objetivo:

- Garantir que peças do pipeline funcionem e não quebrem com mudanças

No projeto, há testes para:

- geração de dados
- pré-processamento (split e padronização)
- modelo (forward/backward, treino reduz loss)
- métricas (acurácia, precisão, recall, F1 — casos-limite incluídos)
- serialização (salvar/carregar preserva predições)

Testes de métricas — por que são importantes:

- Testam F1 = 0 quando o modelo não prediz nenhum positivo
- Testam precisão = 1 quando não há falso positivo
- Garantem que a matriz de confusão bate com os contadores manuais

Exemplos importantes:

- “salvar e carregar pacote preserva predições”: [test\_serializacao.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/tests/test_serializacao.py)
- “métricas de classificação (9 casos)”: [test\_metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/tests/test_metricas.py)

***

## Como rodar (atalhos)

Demo completa (treina, avalia, salva pacote em `artefatos/`):

```bash
python scripts/rodar_pipeline.py
```

Com hiperparâmetros diferentes (sem editar código):

```bash
# 300 épocas, lr menor, early stopping mais paciente, log detalhado
python scripts/rodar_pipeline.py --epochs 300 --lr 0.01 --paciencia 30 --verbose

# Desativar early stopping
python scripts/rodar_pipeline.py --paciencia 0

# Ver todas as opções
python scripts/rodar_pipeline.py --help
```

Inferência local usando o pacote salvo:

```bash
python scripts/inferir.py
```

Serviço HTTP local com Docker Compose (porta 8080 → 8000 interno):

```bash
docker compose up --build
```

Testar o serviço:

```bash
# Healthcheck
curl http://localhost:8080/health

# Predição
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[0.0, 0.0], [1.0, 0.2], [-0.8, 0.9]]}'
```

Configurar o serviço via variável de ambiente (sem rebuildar):

```bash
docker compose run -e PACIENCIA=50 pipeline-ia
```

Testes:

```bash
pytest -q
```

