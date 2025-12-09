import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_pls(x, y, n_components):
    # x: matriz (N, D) padronizada
    # y: vetor (N, 1) padronizado
    # n_components: inteiro (número de componentes)

    N, D = x.shape

    W = np.zeros((D, n_components)) # Pesos
    P = np.zeros((D, n_components)) # Loadings de X
    Q = np.zeros((n_components, 1)) # Loadings de Y 

    x_res = x.copy()
    y_res = y.copy()

    for i in range(n_components):

        # w reflete a covariância entre X e Y
        w = np.dot(x_res.T, y_res)
        w = w / np.linalg.norm(w)

        # t é a projeção dos dados X na direção w
        t = np.dot(x_res, w)

        # loading p (Regressão de X em t)
        # p = (X_res.T * t) / (t.T * t) -> Como t é vetor, t.T * t é um escalar
        denom = np.dot(t.T, t)
        p = np.dot(x_res.T, t) / denom

        # loading q (Regressão de Y em t)
        # q = (Y_res.T * t) / (t.T * t)
        q = np.dot(y_res.T, t) / denom

        # Removemos a parte explicada pelo componente atual
        # X_novo = X_velho - (t * p.T)
        # Y_novo = Y_velho - (t * q) 
        x_res = x_res - np.dot(t, p.T)
        y_res = y_res - (t * q)  
        
        # Armazenar nos vetores principais (preenchendo as colunas i)
        W[:, i] = w.ravel() # ravel garante que seja um array 1D
        P[:, i] = p.ravel()
        Q[i] = q

    # Calcular o vetor de coeficientes Beta
    # Beta = W * (P.T * W)^-1 * Q
    pt_w = np.dot(P.T, W)
    pt_w_inv = np.linalg.pinv(pt_w)
    term1 = np.dot(W, pt_w_inv)
    
    beta = np.dot(term1, Q)
    
    return beta

def predict_pls(x, beta):
    # x: matriz (N, D) padronizada
    # beta: vetor (D, 1)
    
    # Predição: Y = X * Beta
    y_pred = np.dot(x, beta)
    
    return y_pred

def select_optimal_k(cv_results_df, threshold=0.01):
    """
    Escolhe o K baseado no princípio da parcimônia.
    """
    # Ordenar por componente (Crescente)
    df = cv_results_df.copy()
    df = df.sort_values(by='n_components', ascending=True).reset_index(drop=True)
    
    # Calcular a diferença percentual
    # Linha Atual (K=2) - Linha Anterior (K=1)
    # Como RMSE cai, diff é negativo. Invertemos para ficar positivo.
    df['improvement'] = -df['Mean_RMSE'].diff()
    
    # Calcular % de melhora em relação ao passo anterior
    df['pct_improvement'] = df['improvement'] / df['Mean_RMSE'].shift(1)
    
    # Análise para decisão
    print("\n--- Análise de Ganho ---")
    print(df[['n_components', 'Mean_RMSE', 'pct_improvement']])

    # Assumimos K=1 como base
    best_k = int(df.iloc[0]['n_components'])
    
    # Começamos a verificar do segundo (K=2) em diante
    for i in range(1, len(df)):
        improvement = df.iloc[i]['pct_improvement']
        current_k = int(df.iloc[i]['n_components'])
        
        # Só aceitamos o novo K se ele melhorar o erro mais que o threshold
        if improvement >= threshold:
            best_k = current_k
        else:
            print(f"\nPonto De Corte: Adicionar o componente {current_k} melhorou apenas {improvement:.2%}.")
            print(f"Decisão: Parar e manter {best_k} componentes.")
            break
            
    return best_k

def plot_pls_selection(cv_results, best_k):
    """
    Plota o RMSE e R2 em eixos duplos para visualizar a escolha do K.
    
    Args:
        cv_results: DataFrame retornado pela validação cruzada.
        best_k: O inteiro K escolhido pela sua função select_optimal_k.
    """

    # Garantir ordenação correta com base no número de componentes
    df = cv_results.sort_values(by='n_components', ascending=True)
    
    # Estilo
    sns.set_style("white") 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # RMSE (Lado Esquerdo - Vermelho)
    color_rmse = '#E63946' # Vermelho elegante
    ax1.set_xlabel('Número de Componentes (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (Erro)', color=color_rmse, fontsize=12, fontweight='bold')
    ax1.plot(df['n_components'], df['Mean_RMSE'], color=color_rmse, marker='o', linewidth=2, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, axis='x', linestyle=':', alpha=0.6)

    # R2 (Lado Direito - Azul)
    ax2 = ax1.twinx()  
    color_r2 = '#1D3557' # Azul marinho elegante
    ax2.set_ylabel('R² (Variância Explicada)', color=color_r2, fontsize=12, fontweight='bold')
    ax2.plot(df['n_components'], df['Mean_R2'], color=color_r2, marker='s', linestyle='--', linewidth=2, label='R²')
    ax2.tick_params(axis='y', labelcolor=color_r2)
    ax2.grid(False)

    # Uma linha vertical onde foi feita a escolha
    plt.axvline(x=best_k, color='gray', linestyle='-.', alpha=0.8, linewidth=1.5)

    rmse_at_k = df[df['n_components']==best_k]['Mean_RMSE'].values[0]
    
    plt.text(best_k + 0.1, rmse_at_k, f' Escolha Parcimoniosa\n (K={best_k})', 
             color='gray', fontsize=10, verticalalignment='bottom')

    plt.title('Seleção de Componentes PLS', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_test, y_pred, categories=None, title_suffix=""):
    """
    Gera gráficos de regressão em layout vertical (ideal para colunas de artigos).
    Cores opacas e linha de tendência destacada.
    """

    # Garantir arrays 1D
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    residuals = y_test - y_pred

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    sns.set_style("whitegrid")

    # Gráfico Predito vs. Real
    # Linha de Tendência
    sns.regplot(x=y_test, y=y_pred, scatter=False, ax=ax1, 
                color='black', line_kws={'linewidth': 2.5, 'linestyle': '-'}, label='Tendência PLS')

    sns.scatterplot(x=y_test, y=y_pred, ax=ax1, 
                    hue=categories, 
                    alpha=0.9,  
                    edgecolor='k', s=60) 
    
    # Linha Ideal
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color='#444444', linestyle='--', lw=2, label='Ideal')

    ax1.set_ylabel('Valor Predito (Modelo)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Valor Real (Observado)', fontsize=12, fontweight='bold') 
    ax1.set_title(f'Acurácia do Modelo {title_suffix}', fontsize=13)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Gráfico dos resíduos
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, 
                    hue=categories, 
                    alpha=0.9, edgecolor='k', s=60, legend=False) 
    
    # Linha Zero
    ax2.axhline(0, color='black', linestyle='-', lw=1.5)
    
    ax2.set_xlabel('Valor Predito', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resíduos (Real - Predito)', fontsize=12, fontweight='bold')
    ax2.set_title('Análise de Resíduos', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    # plt.savefig('fig2_vertical_results.png', dpi=300, bbox_inches='tight')
    plt.show()