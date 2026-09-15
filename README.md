# Previsão de Adesão a Empréstimo Pessoal (Bank Personal Loan)

Script em Python para análise exploratória e previsão da **adesão de clientes de um banco a um empréstimo pessoal**, comparando os desempenhos de `RandomForestClassifier` e `LogisticRegression`, com ajuste de limiar de decisão (threshold) para priorizar recall, validação cruzada estratificada e visualizações completas (correlações, matrizes de confusão e distribuição de probabilidades).

## 📋 Sumário

- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Estrutura do dataset](#-estrutura-do-dataset)
- [Como usar](#-como-usar)
- [Exemplos de uso](#-exemplos-de-uso)
- [Saídas geradas](#-saídas-geradas)
- [Observações](#-observações)
- [Licença](#-licença)

## ✅ Requisitos

- Python 3.9+
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## 🚀 Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/previsao-emprestimo-pessoal.git
   cd previsao-emprestimo-pessoal
   ```

2. (Opcional, mas recomendado) Crie um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

   Ou crie um arquivo `requirements.txt` com o conteúdo abaixo e instale com `pip install -r requirements.txt`:

   ```
   numpy>=1.23.0
   pandas>=1.5.0
   matplotlib>=3.6.0
   seaborn>=0.12.0
   scikit-learn>=1.2.0
   ```

## 📊 Estrutura do dataset

O script espera um arquivo `Bank_Personal_Loan_Modelling.csv` no mesmo diretório, com (entre outras) as seguintes colunas:

| Coluna                | Tipo             | Uso no script                                                        |
|-----------------------|------------------|------------------------------------------------------------------------|
| `ID`                  | -                | Descartada                                                              |
| `ZIP Code`            | -                | Descartada                                                              |
| `Age`                 | Numérica         | Padronizada (`StandardScaler`)                                          |
| `Experience`          | Numérica         | Usada apenas na análise de correlação; **removida** antes do treino devido à forte correlação com `Age` (também usada para descartar registros com valores negativos) |
| `Income`              | Numérica         | Padronizada (`StandardScaler`)                                          |
| `Family`              | Numérica         | Padronizada (`StandardScaler`)                                          |
| `CCAvg`               | Numérica         | Convertida para `float` (tratamento de valores com `/` no lugar de `.`) e padronizada |
| `Education`           | Categórica       | One-Hot Encoding                                                        |
| `Mortgage`            | Numérica         | Padronizada (`StandardScaler`)                                          |
| `Securities Account`  | Binária (0/1)    | Passthrough (mantida como está)                                          |
| `CD Account`          | Binária (0/1)    | Passthrough (mantida como está)                                          |
| `Online`              | Binária (0/1)    | Passthrough (mantida como está)                                          |
| `CreditCard`          | Binária (0/1)    | Passthrough (mantida como está)                                          |
| `Personal Loan`       | Binária (0/1)    | **Target** (indica se o cliente aceitou o empréstimo pessoal)          |

> 💡 O dataset "Bank Personal Loan Modelling" é amplamente utilizado em tutoriais de ML e pode ser encontrado publicamente, por exemplo, no [Kaggle](https://www.kaggle.com/datasets/teertha/personal-loan-modeling).

Coloque o arquivo `Bank_Personal_Loan_Modelling.csv` na raiz do projeto, no mesmo diretório do script, antes de executá-lo.

## 🖥️ Como usar

Execute o script diretamente, sem parâmetros:

```bash
python loan_predictor.py
```

O script executa, em sequência:
1. Carregamento do `Bank_Personal_Loan_Modelling.csv` e remoção das colunas `ID` e `ZIP Code`.
2. Remoção de registros com valores negativos na coluna `Experience` e correção de formatação da coluna `CCAvg` (substituição de `/` por `.` quando o dado estiver em formato de texto).
3. Cálculo e exibição (em heatmaps) das correlações de **Pearson** e **Spearman** entre os atributos e a variável alvo `Personal Loan`.
4. Remoção da coluna `Experience` do dataset de treino, devido à sua forte correlação positiva com `Age`.
5. Separação em features (`df_x`) e target (`df_y`).
6. Divisão treino/teste (80/20, com `stratify` no target).
7. Pré-processamento via `ColumnTransformer`: `StandardScaler` para colunas numéricas, `OneHotEncoder(drop='first')` para a coluna categórica `Education` e `passthrough` para as colunas binárias.
8. Treinamento de dois modelos: `RandomForestClassifier` (com `class_weight='balanced'`) e `LogisticRegression` (com `class_weight='balanced'`).
9. Ajuste do **limiar de decisão (threshold)** para `0.25`, priorizando recall na identificação de clientes propensos a aceitar o empréstimo.
10. Impressão de acurácia e relatório de classificação para os dois modelos.
11. Validação cruzada estratificada (`StratifiedKFold`, 5 folds) comparando a consistência dos dois modelos.
12. Exibição de um painel com matrizes de confusão e histogramas de distribuição das probabilidades previstas por classe.

## 💡 Exemplos de uso

### Executando o script como está

```bash
python loan_predictor.py
```

O script exibirá, na sequência, dois painéis gráficos (`plt.show()`):
1. Matrizes de correlação de Pearson e Spearman.
2. Matrizes de confusão e histogramas de separação de probabilidades para os dois modelos.

E imprimirá no console, entre outras informações:

```
=================================================================
 CORRELAÇÃO COM A VARIÁVEL ALVO (PERSONAL LOAN) 
=================================================================

--- PEARSON (Linear) ---
Personal Loan    1.000000
Income           0.502...
...

--- DESEMPENHO: RANDOM FOREST ---
Acuracia Geral: 0.97xx
              precision    recall  f1-score   support
           0       ...
           1       ...
```

### Ajustando o limiar de decisão (threshold)

Para priorizar precisão em vez de recall (ou vice-versa), altere a variável `novo_threshold` no script:

```python
novo_threshold = 0.40  # valor original: 0.25
```

E execute novamente:

```bash
python loan_predictor.py
```

### Reutilizando os modelos treinados em uma sessão interativa

Como o script não encapsula a lógica em funções, a forma mais simples de reaproveitar os modelos (`rf_model`, `lr_model`) e o pré-processador (`preprocessor`) após a execução é rodá-lo com `python -i`, mantendo as variáveis disponíveis no console:

```bash
python -i loan_predictor.py
```

```python
>>> nova_amostra_processada = preprocessor.transform(x_test.head(1))
>>> rf_model.predict_proba(nova_amostra_processada)
```

## 📊 Saídas geradas

- **Console**: correlações de Pearson e Spearman com a variável alvo `Personal Loan`, acurácia e relatório de classificação (`RandomForest` e `LogisticRegression`) com o threshold ajustado, e resultados da validação cruzada (média e desvio padrão da acurácia em 5 folds).
- **Gráficos (via `plt.show()`)**:
  - Painel 1: heatmaps de correlação de Pearson e Spearman entre os atributos.
  - Painel 2: matrizes de confusão dos dois modelos (rótulos "Recusou"/"Aceitou") e histogramas comparando a distribuição das probabilidades previstas para as classes "Não Contratou" e "Contratou", com o limiar de decisão destacado.

## ⚠️ Observações

- O script está estruturado de forma **sequencial/procedural** (sem funções ou classes) e deve ser executado de uma só vez, do início ao fim.
- O caminho do dataset está fixado na variável `DATASET_PATH = 'Bank_Personal_Loan_Modelling.csv'`; para usar outro arquivo, edite essa variável diretamente no script.
- A coluna `Experience` é utilizada na análise de correlação e na limpeza de dados (remoção de valores negativos), mas é **removida do conjunto de treino** por apresentar forte correlação positiva com `Age`, evitando multicolinearidade.
- A coluna `CCAvg` recebe um tratamento condicional (`if df['CCAvg'].dtype == 'object'`) para converter valores no formato `"1/5"` (com barra) para `1.5` (com ponto), antes de ser usada como numérica.
- O limiar de decisão (`novo_threshold = 0.25`) foi definido para priorizar recall (reduzir falsos negativos, ou seja, clientes propensos ao empréstimo que seriam erroneamente classificados como não propensos); ajuste esse valor conforme o objetivo do negócio (maior precisão vs. maior recall).
- Os hiperparâmetros dos modelos (`n_estimators=100`, `max_depth=20` para o Random Forest; `max_iter=1000` para a Regressão Logística) estão fixos no código; ajuste-os conforme necessário para o seu dataset.
- A validação cruzada (`cross_val_score`) é executada sobre os dados **não escalonados/originais** (`df_x`, `df_y`), pois o próprio pipeline (`pip_rf`/`pip_lr`) já inclui o pré-processamento em cada fold.

## 📄 Licença

Este projeto está licenciado sob os termos da licença MIT. Sinta-se livre para usar, modificar e distribuir.
