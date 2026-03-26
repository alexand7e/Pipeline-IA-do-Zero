# Pipeline básico de um projeto de IA (do zero) — guia didático

Este documento explica, de forma direta, **tudo o que foi implementado** no projeto **Pipeline-IA-do-Zero**, organizado na sequência lógica de um **pipeline real** de Machine Learning.

O projeto é intencionalmente “do zero” (NumPy) para deixar as ideias claras: **dados → preparo → treino → avaliação → pacote de inferência → serviço**.

Referências do projeto:
- Mapa dos 7 passos: [pipeline_7_passos.md](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docs/pipeline_7_passos.md)
- Roteiro curto (15 min): [roteiro_aula_15min.md](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docs/roteiro_aula_15min.md)

---

## O “problema real” (e a versão didática usada aqui)

Em projetos reais, você quase sempre quer prever **uma decisão binária**:
- fraude vs não fraude
- churn vs não churn
- aprovado vs reprovado

No mundo real, as features podem ser dezenas/centenas. Aqui, para ficar visual e didático, usamos **apenas 2 features** e um dataset sintético **não-linear** (“duas luas”).

Pense assim:
- Feature 1 e Feature 2 representam dois sinais reais (ex.: valor e horário; renda e idade; cliques e tempo de sessão)
- O objetivo é prever `classe ∈ {0, 1}`

---

## Visão geral do pipeline (7 passos)

Fluxo (simplificado):

```text
        ┌───────────────┐
        │ 1) Problema   │  define: alvo + métrica + como avaliar
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 2) Dados       │  gera/obtém X, y
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 3) Preparo     │  padronização (fit no treino)
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 4) Split       │  treino/val/teste
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 5) Treino      │  MLP + BCE + SGD
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 6) Avaliação   │  acc + matriz de confusão
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ 7) Empacotar   │  salvar modelo + preprocessamento
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ Produção       │  inferência via script/HTTP/Docker
        └───────────────┘
```

Onde acontece no código:
- Orquestração do pipeline: [pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- Demo completa (treina/avalia/salva pacote): [rodar_pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/scripts/rodar_pipeline.py)

---

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

---

## Passo 1) Definir problema e métrica

Objetivo:
- Decidir **o que prever** (target) e **como avaliar** (métrica)

No projeto:
- Problema: **classificação binária**
- Métrica principal: **acurácia**
- Métrica de apoio: **matriz de confusão** (para entender FP/FN)

Representação simplificada:

```text
prob >= 0.5  → classe 1
prob <  0.5  → classe 0

matriz 2x2: [[TN, FP],
            [FN, TP]]
```

Código:
- [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

---

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
- [gerar_dados_duas_luas](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/dados.py)

---

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

---

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
- [dividir_treino_validacao_teste](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

---

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
para cada época:
  embaralhar treino
  para cada mini-batch:
    y_pred = forward(Xb)
    grad_z_ultimo = (y_pred - yb) / n
    backward(grad_z_ultimo)
    sgd_update(taxa_aprendizado)

  medir loss_treino, loss_valid, acc_valid
```

Onde está cada peça:
- Forward/backward e SGD: [modelo_mlp.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/modelo_mlp.py)
- BCE e gradiente simplificado (BCE + Sigmoid): [perdas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/perdas.py)
- Função de treino do pipeline: [treinar_mlp](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)

---

## Passo 6) Avaliar e analisar erros

Objetivo:
- Medir performance no **teste**
- Entender *como* o modelo erra (não só “quanto”)

No projeto:
- Calcula:
  - `loss_teste`
  - `acc_teste`
  - `matriz_confusao`

Representação simplificada:

```text
prob_teste = modelo.forward(X_teste_padronizado)
acc_teste  = mean((prob_teste >= 0.5) == y_teste)
cm         = [[TN, FP],
              [FN, TP]]
```

Código:
- Avaliação na execução do pipeline: [executar_pipeline](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- Métricas: [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

---

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

---

## “Produção” no projeto: inferência via HTTP (e Docker)

Objetivo:
- Simular como esse pipeline vira um serviço de inferência

No projeto:
- Serviço HTTP minimalista:
  - `GET /health`
  - `POST /predict` com JSON `{ "X": [[x1, x2], ...] }`
- Se não existir pacote salvo, treina e salva automaticamente; caso exista, apenas carrega.

Representação simplificada do `/predict`:

```text
recebe X (n,2)
Xp = pad.transformar(X)
prob = modelo.forward(Xp)
classe = prob >= 0.5
retorna prob e classe
```

Código:
- Serviço HTTP: [serve.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/serve.py)

Infra:
- Dockerfile: [Dockerfile](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/Dockerfile)
- Compose: [docker-compose.yml](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/docker-compose.yml)

Exemplo de request:

```json
{
  "X": [
    [0.0, 0.0],
    [1.0, 0.2],
    [-0.8, 0.9]
  ]
}
```

---

## Qualidade e confiabilidade: testes automatizados

Objetivo:
- Garantir que peças do pipeline funcionem e não quebrem com mudanças

No projeto, há testes para:
- geração de dados
- pré-processamento (split e padronização)
- modelo (forward/backward)
- serialização (salvar/carregar preserva predições)

Exemplo importante:
- “salvar e carregar pacote preserva predições”: [test_serializacao.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/tests/test_serializacao.py)

---

## Como rodar (atalhos)

Demo completa (treina, avalia, salva pacote em `artefatos/`):

```bash
python scripts/rodar_pipeline.py
```

Inferência local usando o pacote salvo:

```bash
python scripts/inferir.py
```

Serviço HTTP local com Docker:

```bash
docker build -t pipeline-ia-do-zero .
docker run --rm -p 8000:8000 pipeline-ia-do-zero
```

Ou com Compose:

```bash
docker compose up --build
```

Testes:

```bash
pytest -q
```

