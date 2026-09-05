# IMPORTACAO DE BIBLIOTECAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#ESTILIZACAO DE GRAFICOS
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

# CARREGAMENTO DE DATASET
DATASET_PATH = 'Bank_Personal_Loan_Modelling.csv'
df = pd.read_csv(DATASET_PATH)

# DROPS DE DADOS 
df = df.drop(columns=['ID', 'ZIP Code'], errors='ignore') #DROP DE COLUNAS ID, ZIP CODE
df = df.drop(df[df['Experience'] < 0].index) #DROP DE VALORES NEGATIVOS DA COLUNA EXPERIENCE
if df['CCAvg'].dtype == 'object': # TIPIFICACAO E REPLACE DE DADOS DA COLUNA CCAVG
    df['CCAvg'] = df['CCAvg'].astype(str).str.replace('/', '.').astype(float)

# ANALISE DE CORRELACAO DE PEARSON E SPEARMAN
cols_analise = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Personal Loan'] 
corr_pearson = df[cols_analise].corr(method='pearson')
corr_spearman = df[cols_analise].corr(method='spearman')
# GRAFICOS DAS CORRELACOES
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(
    corr_pearson, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    ax=axes[0], 
    cbar=False
)
axes[0].set_title("Correlação de Pearson (Relação Linear)", fontsize=12, fontweight='bold')
sns.heatmap(
    corr_spearman, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    ax=axes[1], 
    cbar=True
)
axes[1].set_title("Correlação de Spearman (Relação Monotônica)", fontsize=12, fontweight='bold')
plt.suptitle("Análise Exploratória: Matrizes de Correlação dos Atributos", fontsize=14, fontweight='bold')
plt.show()

print("=" * 65)
print(" CORRELAÇÃO COM A VARIÁVEL ALVO (PERSONAL LOAN) ")
print("=" * 65)
print("\n--- PEARSON (Linear) ---")
print(corr_pearson['Personal Loan'].sort_values(ascending=False))

print("\n--- SPEARMAN (Postos/Ranks) ---")
print(corr_spearman['Personal Loan'].sort_values(ascending=False))

df = df.drop(columns=['Experience'], errors='ignore') #DEVIDO CORRELACAO FORTEMENTE POSITIVA AGE E EXPERIENCE, DROPADO A COLUNA EXPERIENCE

#SEPARACAO DE FEATURES E TARGET
df_x = df.drop(columns=['Personal Loan'])
df_y = df['Personal Loan']

#SEPARACAO EM TREINO E TESTE
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=42, stratify=df_y
)

#LISTAS DE COLUNAS PARA TRATAMENTO DE ESCALONAMENTO
numeric_cols = ['Age', 'Income', 'Family', 'CCAvg', 'Mortgage']
cat_cols = ['Education'] 
bin_cols = ['Securities Account', 'CD Account', 'Online', 'CreditCard']

#PREPROCESSAMENTO DE ESCALONAMENTO
preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ]
)
#ESCALONAMENTO DE DADOS
x_train_scaled = preprocessor.fit_transform(x_train)
x_test_scaled = preprocessor.transform(x_test)

#APLICACAO DOS DOIS ALGORITMOS - RF E LR
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced'
)

lr_model = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000
)

# TREINAMENTO (AJUSTE) DOS MODELOS COM OS DADOS JÁ ESCALONADOS
rf_model.fit(x_train_scaled, y_train)
lr_model.fit(x_train_scaled, y_train)

#NOVO LIMITE DE THRESHOLD DE 25%
novo_threshold = 0.25

#APLICACAO DE NOVO THRESHOLD - PRIORIZANDO O RECALL
probs_rf = rf_model.predict_proba(x_test_scaled)[:, 1]
pred_rf_custom = (probs_rf >= novo_threshold).astype(int)

probs_lr = lr_model.predict_proba(x_test_scaled)[:, 1]
pred_lr_custom = (probs_lr >= novo_threshold).astype(int)

#AVALIACAO

print("=" * 65)
print(f" AVALIAÇÃO DE DESEMPENHO (LIMIAR DE CORTE AJUSTADO PARA {novo_threshold}) ")
print("=" * 65)

print("\n--- DESEMPENHO: RANDOM FOREST ---")
print(f'Acuracia Geral: {accuracy_score(y_test, pred_rf_custom):.4f}')
print(classification_report(y_test, pred_rf_custom))

print("\n--- DESEMPENHO: REGRESSÃO LOGÍSTICA ---")
print(f'Acuracia Geral: {accuracy_score(y_test, pred_lr_custom):.4f}')
print(classification_report(y_test, pred_lr_custom))


# VALIDACAO CRUZADA ESTRATIFICADA (CROSS-VALIDATION)
pip_rf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf_model)])
pip_lr = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lr_model)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_rf = cross_val_score(pip_rf, df_x, df_y, cv=skf, n_jobs=-1)
scores_lr = cross_val_score(pip_lr, df_x, df_y, cv=skf, n_jobs=-1)

print("\n" + "=" * 65)
print(" VALIDAÇÃO CRUZADA COMPARATIVA (CONSISTÊNCIA EM 5 FOLDS) ")
print("=" * 65)
print(f'Random Forest   - Acuracia Media: {scores_rf.mean():.4f} (Desvio Padrao: {scores_rf.std():.4f})')
print(f'Reg. Logistica  - Acuracia Media: {scores_lr.mean():.4f} (Desvio Padrao: {scores_lr.std():.4f})')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cm_rf = confusion_matrix(y_test, pred_rf_custom)
cm_lr = confusion_matrix(y_test, pred_lr_custom)

#MATRIZ DE CONFUSAO
ConfusionMatrixDisplay(cm_rf, display_labels=['Recusou', 'Aceitou']).plot(
    ax=axes[0, 0], cmap='Blues', colorbar=False
)
axes[0, 0].set_title(f'Random Forest (Matriz de Confusão - Corte {novo_threshold})')
axes[0, 0].grid(False)

ConfusionMatrixDisplay(cm_lr, display_labels=['Recusou', 'Aceitou']).plot(
    ax=axes[0, 1], cmap='Greens', colorbar=False
)
axes[0, 1].set_title(f'Regressão Logística (Matriz de Confusão - Corte {novo_threshold})')
axes[0, 1].grid(False)

#GRAFICO DA MATRIZ
sns.histplot(probs_rf[y_test == 0], color='red', label='Classe 0 (Não Contratou)', ax=axes[1, 0], kde=True, stat="density", alpha=0.4)
sns.histplot(probs_rf[y_test == 1], color='green', label='Classe 1 (Contratou)', ax=axes[1, 0], kde=True, stat="density", alpha=0.4)
axes[1, 0].axvline(novo_threshold, color='blue', linestyle='--', label=f'Limiar ({novo_threshold})')
axes[1, 0].set_title('Random Forest: Separação de Probabilidades')
axes[1, 0].set_xlabel('Probabilidade Calculada de Aceitar o Empréstimo')
axes[1, 0].legend()

sns.histplot(probs_lr[y_test == 0], color='red', label='Classe 0 (Não Contratou)', ax=axes[1, 1], kde=True, stat="density", alpha=0.4)
sns.histplot(probs_lr[y_test == 1], color='green', label='Classe 1 (Contratou)', ax=axes[1, 1], kde=True, stat="density", alpha=0.4)
axes[1, 1].axvline(novo_threshold, color='blue', linestyle='--', label=f'Limiar ({novo_threshold})')
axes[1, 1].set_title('Regressão Logística: Separação de Probabilidades')
axes[1, 1].set_xlabel('Probabilidade Calculada de Aceitar o Empréstimo')
axes[1, 1].legend()

plt.suptitle("Painel Comparativo de Desempenho e Diagnóstico dos Modelos", fontsize=14, fontweight='bold')
plt.show()