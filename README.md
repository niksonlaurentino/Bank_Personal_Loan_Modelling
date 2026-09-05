# Bank_Personal_Loan_Modelling
Modelo de Machine Learning para previsão de adesão a empréstimos pessoais bancários.

Este repositório contém uma solução completa de Aprendizado de Máquina para prever a conversão de clientes bancários em campanhas de empréstimo pessoal. O projeto abrange desde a limpeza e análise exploratória de dados até a comparação entre algoritmos e otimização do limiar de decisão (threshold).

---

## 📌 Visão Geral do Problema

O objetivo do banco é identificar clientes com alta probabilidade de aceitar uma oferta de empréstimo pessoal. Como a maioria dos clientes recusa a oferta (classe desbalanceada), o foco principal da modelagem é maximizar o Recall da classe positiva, reduzindo os Falsos Negativos (clientes potenciais que não seriam abordados).

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* Linguagem: Python 3.x
* Manipulação e Análise de Dados: pandas, numpy
* Visualização de Dados: matplotlib, seaborn
* Machine Learning & Pré-processamento: scikit-learn
  - ColumnTransformer, StandardScaler, OneHotEncoder
  - RandomForestClassifier, LogisticRegression
  - train_test_split, StratifiedKFold, cross_val_score
  - classification_report, confusion_matrix, ConfusionMatrixDisplay

---

## 🚀 Estrutura do Pipeline de Dados

O script executa as seguintes etapas estruturadas:

1. Limpeza e Tratamento de Dados:
   * Remoção de identificadores irrelevantes (ID, ZIP Code).
   * Eliminação de inconsistências (registros com Experience < 0).
   * Saneamento de formato e tipagem da coluna CCAvg.

2. Análise de Correlação e Multicolinearidade:
   * Cálculo das matrizes de correlação de Pearson (linear) e Spearman (monotônica).
   * Remoção da variável Experience devido à altíssima colinearidade com a variável Age.

3. Engenharia de Recursos e Pré-processamento:
   * Uso de ColumnTransformer para evitar vazamento de dados (data leakage).
   * Variáveis Numéricas: Padronização Z-Score (StandardScaler).
   * Variáveis Categóricas: Codificação OneHotEncoder(drop='first').
   * Variáveis Binárias: Mantidas originais.

4. Modelagem e Balanceamento:
   * Divisão estratificada dos dados (80% treino / 20% teste).
   * Aplicação do argumento class_weight='balanced' nos algoritmos Random Forest e Regressão Logística.

5. Otimização do Limiar de Decisão (Decision Thresholding):
   * Ajuste do ponto de corte de classificação de 0.50 para 0.25, priorizando a captura de potenciais clientes (Recall).

6. Validação Cruzada e Diagnóstico:
   * Avaliação de consistência através de StratifiedKFold com 5 divisões (folds).
   * Geradores de Matriz de Confusão e gráficos de densidade de probabilidade (KDE).

---

## 📊 Estrutura de Pré-processamento das Colunas

| Categoria de Atributo | Colunas Incluídas | Tratamento Aplicado |
| :--- | :--- | :--- |
| Numéricos | Age, Income, Family, CCAvg, Mortgage | StandardScaler() |
| Categóricos Multi-classe | Education | OneHotEncoder(drop='first') |
| Binários | Securities Account, CD Account, Online, CreditCard | Passthrough (Mantidos originais) |

---

## ⚙️ Como Executar o Projeto

1. Clone o repositório em sua máquina local:
   git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git

2. Certifique-se de ter as bibliotecas necessárias instaladas:
   pip install numpy pandas matplotlib seaborn scikit-learn

3. Mantenha o arquivo de dados Bank_Personal_Loan_Modelling.csv no mesmo diretório do script Python.

4. Execute o arquivo do script:
   python nome_do_seu_script.py

---

## 📈 Resultados e Diagnósticos Esperados

Ao rodar o script, são gerados:
* Matrizes de Correlação Comparativas: Pearson vs. Spearman em painel lado a lado.
* Métricas de Classificação: Acurácia geral, Precision, Recall e F1-Score para ambos os modelos com corte em 0.25.
* Métricas de Validação Cruzada: Acurácia média e desvio padrão ao longo dos 5 folds.
