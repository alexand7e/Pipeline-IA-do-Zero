# Pipeline Básico de IA (do zero)

Projeto de portfólio em **PT-BR** que implementa, com **NumPy**, um pipeline completo e enxuto de Machine Learning para um modelo simples de rede neural (MLP) em **classificação binária**.

A proposta é servir como **case didático** para uma aula de 15 minutos sobre “pipeline básico de um modelo de IA”, cobrindo os **7 passos essenciais** com código legível, testes e um script de demonstração.

---

## O que este projeto demonstra

- Implementação do pipeline com foco em clareza (sem frameworks de ML)
- Rede neural MLP mínima (forward, backward e SGD)
- Reprodutibilidade (seeds, split determinístico)
- Avaliação + análise de erro (acurácia e matriz de confusão)
- Salvamento/carregamento do modelo para inferência

---

## Os 7 passos do pipeline (mapa do projeto)

1. **Definir o problema e a métrica**  
   Objetivo: classificar pontos 2D em duas classes. Métrica: acurácia (e matriz de confusão).
2. **Obter dados**  
   Geração de um dataset sintético “duas luas” (evita dependências e facilita visualização).
3. **Preparar os dados**  
   Padronização (Z-score) aprendida no treino e aplicada em validação/teste.
4. **Dividir em treino/validação/teste**  
   Split com seed e embaralhamento para avaliação honesta.
5. **Treinar o modelo**  
   MLP (ReLU + Sigmoid) treinada com mini-batches e gradiente descendente estocástico.
6. **Avaliar e analisar erros**  
   Métricas, matriz de confusão e checagens rápidas de overfitting via validação.
7. **Empacotar para inferência e próximos passos**  
   Serialização em JSON e script de inferência; seção final sugere monitoramento e iteração.

---

## Estrutura do projeto

```
Pipeline-IA-do-Zero/
├── src/
│   ├── dados.py              # geração do dataset sintético
│   ├── preprocessamento.py   # split + padronização (fit/transform)
│   ├── modelo_mlp.py         # MLP do zero (camadas densas)
│   ├── perdas.py             # BCE (binária) e gradientes
│   ├── metricas.py           # acurácia + matriz de confusão
│   ├── serializacao.py       # salvar/carregar modelo em JSON
│   └── pipeline.py           # orquestra os 7 passos
├── scripts/
│   ├── rodar_pipeline.py     # demo completa (treino + avaliação + export)
│   └── inferir.py            # exemplo de inferência com modelo salvo
├── docs/
│   ├── pipeline_7_passos.md  # explicação objetiva do pipeline
│   └── roteiro_aula_15min.md # roteiro de apresentação (15 min)
└── tests/
    ├── test_dados.py
    ├── test_preprocessamento.py
    ├── test_modelo.py
    └── test_serializacao.py
```

---

## Dependências

- Python 3.11+
- NumPy
- Matplotlib (opcional, apenas para visualizações no script)
- pytest (testes)

Instalação:

```bash
pip install numpy matplotlib pytest
```

---

## Como rodar

### Demo: pipeline completo

```bash
python scripts/rodar_pipeline.py
```

### Inferência com modelo salvo

```bash
python scripts/inferir.py
```

### Deploy local com Docker (emulando produção)

Sobe um serviço HTTP com:
- `GET /health`
- `POST /predict`

Build e run:

```bash
docker build -t pipeline-ia-do-zero .
docker run --rm -p 8000:8000 pipeline-ia-do-zero
```

Teste rápido:

```bash
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary @scripts/exemplo_requisicao.json
```

Alternativa com Compose:

```bash
docker compose up --build
```

### Testes

```bash
pytest -q
```
