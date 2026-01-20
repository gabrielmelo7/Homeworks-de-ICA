import numpy as np
import pandas as pd
from utils.qda_class import qda
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


def plot_decision_boundaries(qda_obj):
    # Pegar as 2 melhores features
    top_2 = qda_obj.columns[:2] 
    X = qda_obj.train_x[top_2].values
    y = qda_obj.train_y.values
    
    # Treinar uma QDA com essas 2 features
    qda_viz = QuadraticDiscriminantAnalysis()
    qda_viz.fit(X, y)
    
    # Malha de pontos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Prever para cada ponto da malha
    Z = qda_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=40)

    handles, _ = scatter.legend_elements()
    class_names = ["Hazardous", "Poor", "Moderate", "Good"]
    
    ax.legend(handles, class_names, title="Classes")
    
    ax.set_xlabel(top_2[0])
    ax.set_ylabel(top_2[1])
    ax.set_title("Fronteiras de Decisão QDA")
    
    return fig

def analyze_variance_structure(qda_obj):
    df = pd.concat([qda_obj.train_x[qda_obj.columns], qda_obj.train_y], axis=1)
    
    print("\nAnálise de Dispersão por Classe (Justificativa QDA)")
    for label in [1, 2, 3, 4]:
        subset = df[df.iloc[:, -1] == label].iloc[:, :-1]
        # Determinante da matriz de covariância (volume da nuvem de dados)
        # Usamos log-determinante para estabilidade numérica
        cov_matrix = np.cov(subset, rowvar=False)
        log_det = np.linalg.slogdet(cov_matrix)[1]
        
        print(f"Classe {label}: Log-Determinante da Covariância = {log_det:.2f}")


for i in range(3, 10):
    qda_obj = qda("HW3/data/data_yeojohnson_zscore.csv", i)
    
    # Matriz de Confusão
    fig = qda_obj.confusion_matrix_plot()
    fig.savefig(f"HW3/images/qda_confusion_{i}.png")
    plt.close()

    # Estatísticas
    print(qda_obj.principal_stats())

# Ilustração das fronteiras de decisão com 2 features
fig = plot_decision_boundaries(qda_obj)
fig.savefig(f"HW3/images/decision_boundaries_qda.png")
plt.close()

# Análise da covariância das classes
analyze_variance_structure(qda_obj)