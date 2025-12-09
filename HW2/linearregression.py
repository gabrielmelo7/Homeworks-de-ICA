import numpy as np
from utils.olsregression import ols_regression
from utils.train_test_split import train_test_split 
from utils.rmse import root_mean_square_error
from utils.plotcompare import plotcompare
from utils.r_squared import r2 
from utils.crossvalOLS import cross_validation
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

data_df = pd.read_csv("data/data_yeojohnson.csv") 

map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }

inverse_map = {v: k for k, v in map_values.items()}

data_df['Air Quality'].replace(map_values, inplace=True)

x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(data_df, "SO2")

ols_model = ols_regression()
ols_model.fit(x_train_set, y_train_set)
model = LinearRegression()
model.fit(x_train_set, y_train_set)

ols_predictions = ols_model.predict(x_test_set) 
model_predictions = model.predict(x_test_set)

ols_rmse = root_mean_square_error(ols_predictions, y_test_set)
ols_r2 = r2(ols_predictions, y_test_set)
model_rmse = root_mean_square_error(model_predictions, y_test_set)
model_r2 = r2(model_predictions, y_test_set)

print(f"Ols rmse= {ols_rmse}, Scikit-learn rmse= {model_rmse}\n")
print(f"Ols r squared= {ols_r2}, Scikit-learn r squared= {model_r2}\n")
print(f"Ols weights = {ols_model}\n Scikit-learn weights= {np.hstack([model.intercept_, model.coef_])}\n")

# K-fold cross validation of the OLS model
rmse, rmse_std, r2, r2_std = cross_validation(data_df, "SO2", ols_model, 5)

print(f"Cross val rmse: {rmse}\n")
print(f"Cross val rmse_std: {rmse_std}\n")
print(f"Cross val r2: {r2}\n")
print(f"Cross val r2_std: {r2_std}\n")

# K-fold cross validation of the scikit model
rmse, rmse_std, r2, r2_std = cross_validation(data_df, "SO2", model, 5)

print(f"Scikit-learn rmse: {rmse}\n")
print(f"Scikit-learn rmse_std: {rmse_std}\n")
print(f"Scikit-learn r2: {r2}\n")
print(f"Scikit-learn r2_std: {r2_std}\n")

class_values = x_test_set[:, 8]
class_values = np.array([inverse_map[val] for val in class_values])

class_values

fig = plotcompare(y_test_set, ols_predictions, class_values)
fig.savefig(os.path.join("./results", r"plot_comparison.png"))


