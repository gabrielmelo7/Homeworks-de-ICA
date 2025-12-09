import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils.pls_regression_module import train_pls, predict_pls, select_optimal_k, plot_pls_selection, plot_regression_results
from utils.crossvalidation import cross_validation_pls 
from utils.train_test_split import train_test_split


df = pd.read_csv('HW2/data/data_yeojohnson.csv')

map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }

df['Air Quality'] = df['Air Quality'].map(map_values)

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(df, target_variable='SO2')

colunas_X = df.drop('SO2', axis=1).columns
colunas_Y = ['SO2']

X_train = pd.DataFrame(X_train, columns=colunas_X)
X_test  = pd.DataFrame(X_test,  columns=colunas_X)
y_train = pd.DataFrame(y_train, columns=colunas_Y)
y_test  = pd.DataFrame(y_test,  columns=colunas_Y)

# print(f"{X_train}")
# print(f"{X_test}")
# print(f"{y_train}")
# print(f"{y_test}")

print("\n--- Validação Cruzada ---")

cv_results = cross_validation_pls(X_train, y_train, max_components=9, folds=10)

print(cv_results)

best_k = select_optimal_k(cv_results, 0.005)
print(f"\nMelhor número de componentes encontrado: {best_k}")

print(f"\n--- Comparação no Test Set (K={best_k}) ---")

mean_X = X_train.mean(); std_X = X_train.std()
mean_y = y_train.mean(); std_y = y_train.std()

# Padronizar Treino
X_train_z = (X_train - mean_X) / std_X
y_train_z = (y_train - mean_y) / std_y

# Treinar o modelo
beta_custom = train_pls(X_train_z.values, y_train_z.values, n_components=best_k)

# Testar o modelo (Padronizando teste com médias do treino)
X_test_z = (X_test - mean_X) / std_X
y_pred_z = predict_pls(X_test_z.values, beta_custom)

# Despadronizar a saída para escala real do data yeojohnson
y_pred_custom = (y_pred_z * std_y.values.item()) + mean_y.values.item()

# O sklearn faz a padronização internamente
pls_sklearn = PLSRegression(n_components=best_k, scale=True)
pls_sklearn.fit(X_train, y_train)
y_pred_sklearn = pls_sklearn.predict(X_test)

# Comparando os resultados
rmse_custom = np.sqrt(mean_squared_error(y_test, y_pred_custom))
r2_custom = r2_score(y_test, y_pred_custom)

rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print(f"{'Métrica':<15} | {'Seu Modelo':<15} | {'Scikit-Learn':<15}")
print("-" * 50)
print(f"{'RMSE':<15}   | {rmse_custom:.5f}          | {rmse_sklearn:.5f}")
print(f"{'R2':<15}   | {r2_custom:.5f}          | {r2_sklearn:.5f}")

plot_pls_selection(cv_results, best_k)

categorias_numericas = X_test['Air Quality']

map_reverso = {
    1: 'Good',
    2: 'Moderate',
    3: 'Poor',
    4: 'Hazardous'
}
categorias_nomes = categorias_numericas.map(map_reverso)

plot_regression_results(y_test, y_pred_custom, categories=categorias_nomes, title_suffix="(PLS K=3)")