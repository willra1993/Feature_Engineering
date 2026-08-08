# Feature Engineering para risco de crédito

[![CI](https://github.com/willra1993/Feature_Engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/willra1993/Feature_Engineering/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Projeto de preparação de dados para um problema de classificação de risco de crédito. O objetivo é transformar uma base cadastral bruta em atributos consistentes e reproduzíveis, prontos para alimentar modelos de machine learning sem introduzir vazamento entre treino e teste.

## Contexto

A base contém 50.000 solicitações de cartão, 54 colunas e a variável-alvo `MAU_PAGADOR`. Na amostra disponível, 13.041 registros (26,08%) pertencem à classe positiva. O notebook documenta a exploração original; o pipeline em Python oferece uma execução limpa e testável da preparação e de um modelo de referência.

Este projeto trata o modelo apenas como baseline técnico. Uma decisão real de crédito exige validação temporal, análise de viés, explicabilidade, monitoramento e revisão das exigências regulatórias aplicáveis.

## Decisões técnicas

- valores sentinela como `NULL`, campos em branco e `#DIV/0!` são tratados como ausentes;
- identificadores, atributos constantes e campos de cardinalidade muito alta são removidos do baseline;
- variáveis numéricas recebem imputação pela mediana e normalização Min-Max;
- variáveis categóricas recebem imputação pela moda e codificação one-hot tolerante a categorias inéditas;
- todas as transformações aprendidas são ajustadas somente nos dados de treino por meio de um `Pipeline` do scikit-learn;
- a divisão é estratificada e determinística;
- o baseline usa regressão logística com pesos balanceados para reduzir o impacto do desbalanceamento da classe-alvo.

## Resultado do baseline

Com a divisão padrão (`test_size=0.20`, `random_state=42`), o pipeline produz o seguinte ponto de referência:

| Métrica | Resultado |
| --- | ---: |
| Acurácia | 0,5748 |
| Acurácia balanceada | 0,5803 |
| Precisão | 0,3262 |
| Recall | 0,5916 |
| F1-score | 0,4206 |
| ROC AUC | 0,6127 |

Esses valores servem para verificar a execução e comparar experimentos futuros. Eles não representam uma estimativa de desempenho em produção.

## Como executar

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/willra1993/Feature_Engineering.git
cd Feature_Engineering
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Linux ou macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale o projeto e execute o baseline:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
feature-engineering --data dataset.txt --report artifacts/metrics.json
```

O comando imprime as métricas no terminal e, quando `--report` é informado, salva o mesmo resultado em JSON. Para abrir a análise exploratória:

```bash
python -m pip install -e ".[notebook]"
jupyter lab Feature_Engineering.ipynb
```

## Qualidade e testes

As verificações locais reproduzem o que é executado automaticamente no GitHub Actions:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Os testes cobrem o contrato de preparação dos dados e o comportamento do pipeline diante de valores ausentes e categorias não observadas no treino.

## Limitações

- o dataset não contém documentação de origem, período de referência ou licença de redistribuição;
- o experimento usa uma divisão aleatória, adequada para demonstração, mas insuficiente para estimar desempenho futuro em produção;
- a remoção de campos de alta cardinalidade é uma escolha conservadora de baseline, não uma conclusão definitiva sobre seu valor preditivo;
- métricas agregadas não substituem análise de custo de erro, calibração, estabilidade e equidade entre grupos.
