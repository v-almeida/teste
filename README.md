import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from google.colab import files
import io

# Carregar o arquivo
uploaded = files.upload()
df = pd.read_excel(io.BytesIO(uploaded['Lista NPS Positivo_V4 (11).xlsx']))

# Definindo as colunas X e y
X = df['nome']  # Coluna com texto
y = df['reacao']  # Coluna alvo

# Verificar valores nulos
print("Valores nulos em X:", X.isnull().sum())
print("Valores nulos em y:", y.isnull().sum())

# Remover linhas com valores nulos em y
df = df.dropna(subset=['nome', 'reacao'])

# Redefinindo X e y após remoção de NaNs
X = df['nome']
y = df['reacao']

# Verificar o balanceamento percentual da base de dados
print("Distribuição percentual das classes:")
print(y.value_counts(normalize=True))

# Verificar o número de classes em y
unique_classes = y.unique()
print(f"Número de classes únicas em y: {len(unique_classes)}")

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Transformar o texto em TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treinando e otimizando o modelo RandomForest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
model_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
model_rf.fit(X_train_tfidf, y_train)
y_pred_rf = model_rf.predict(X_test_tfidf)
y_pred_proba_rf = model_rf.predict_proba(X_test_tfidf)

# Treinando o modelo LogisticRegression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
y_pred_proba_lr = model_lr.predict_proba(X_test_tfidf)

# Função para calcular métricas
def calcular_metricas_multiclass(y_test, y_pred, modelo_nome):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia - {modelo_nome}: {accuracy:.2f}")
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precisão - {modelo_nome}: {precision:.2f}")
    print(f"Recall - {modelo_nome}: {recall:.2f}")
    print(f"F1 Score - {modelo_nome}: {f1:.2f}")

# Calcular métricas
calcular_metricas_multiclass(y_test, y_pred_rf, "Random Forest")
calcular_metricas_multiclass(y_test, y_pred_lr, "Logistic Regression")

# Função para plotar a curva ROC para multiclass
def plotar_curva_roc(y_test, y_pred_proba, modelo_nome):
    y_test_bin = label_binarize(y_test, classes=unique_classes)
    n_classes = y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Classe {unique_classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {modelo_nome}')
    plt.legend(loc="lower right")
    plt.show()

# Plotar a curva ROC para Random Forest e Logistic Regression
plotar_curva_roc(y_test, y_pred_proba_rf, "Random Forest")
plotar_curva_roc(y_test, y_pred_proba_lr, "Logistic Regression")

# Função para plotar gráficos de pontos de corte
def plotar_graficos_pontos_de_corte(y_test, y_pred_proba):
    thresholds = np.arange(0, 1.01, 0.01)
    precisions = []
    recalls = []
    f1_scores = []
    
    # Calcular precisão e recall para diferentes thresholds
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_threshold, average='weighted')
        recall = recall_score(y_test, y_pred_threshold, average='weighted')
        f1 = f1_score(y_test, y_pred_threshold, average='weighted')
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Plotar os gráficos
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precisions, label='Precisão', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precisão e Recall vs Threshold')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, precisions, label='Precisão', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='orange')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.axvline(x=0.7, color='purple', linestyle='--', label='Threshold 0.7')
    plt.axvline(x=0.3, color='brown', linestyle='--', label='Threshold 0.3')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Análise de Thresholds')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Executar para Random Forest
plotar_graficos_pontos_de_corte(y_test, y_pred_proba_rf)
