# Pipeline básico de um modelo de IA (7 passos)

Este documento é o “mapa mental” do projeto: um pipeline completo, do zero, para classificação binária com uma MLP.

## 1) Definir problema e métrica

- Problema: classificar pontos 2D em duas classes.
- Saída do modelo: probabilidade (0–1).
- Métrica principal: acurácia no teste.
- Métrica de apoio: matriz de confusão (para entender FP/FN).

Onde está no código:
- [pipeline.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

## 2) Obter dados

Aqui usamos um dataset sintético (“duas luas”) para:
- evitar dependências externas e discussão de licenças
- facilitar a visualização do limite de decisão
- ter um dataset não-linear (onde uma MLP faz sentido)

Onde está no código:
- [dados.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/dados.py)

## 3) Preparar os dados (pré-processamento)

Passo essencial: padronizar os atributos (Z-score) usando **apenas o treino**:
- ajusta média e desvio no treino
- aplica a transformação em validação e teste

Onde está no código:
- [preprocessamento.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

## 4) Dividir em treino/validação/teste

Objetivo: medir generalização.
- treino: aprende pesos
- validação: ajuda a enxergar overfitting durante o desenvolvimento
- teste: só para o “resultado final”

Onde está no código:
- [dividir_treino_validacao_teste](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/preprocessamento.py)

## 5) Treinar o modelo

Modelo: MLP com camadas densas
- ativações: ReLU nas camadas ocultas e Sigmoid na saída
- otimização: SGD com mini-batches
- loss: BCE binária

Onde está no código:
- [modelo_mlp.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/modelo_mlp.py)
- [perdas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/perdas.py)
- [treinar_mlp](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)

## 6) Avaliar e analisar erros

No teste, reportamos:
- loss_teste (BCE)
- acc_teste
- matriz de confusão

Onde está no código:
- [executar_pipeline](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/pipeline.py)
- [metricas.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/metricas.py)

## 7) Empacotar para inferência (e próximos passos)

Um modelo “de verdade” precisa rodar em inferência com o mesmo pré-processamento do treino.

Este projeto salva um pacote JSON com:
- pesos e vieses do modelo
- média e desvio do padronizador

Onde está no código:
- [serializacao.py](file:///c:/Users/alexa/OneDrive/Documentos/06-personal-projects/aula-ai-mentor/Pipeline-IA-do-Zero/src/serializacao.py)

Como ver funcionando:

```bash
python scripts/rodar_pipeline.py
python scripts/inferir.py
```

