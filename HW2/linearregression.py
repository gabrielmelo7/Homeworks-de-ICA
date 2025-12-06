import numpy as np
from utils.olsregression import ols_regression
from utils.train_test_split import train_test_split 
from utils.rmse import root_mean_square_error
from utils.r_squared import r2 
from sklearn.linear_model import LinearRegression
import pandas as pd

data_df = pd.read_csv("data/data_yeojohnson_zscore.csv") 

map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }

data_df['Air Quality'].replace(map_values, inplace=True)

x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(data_df, "SO2")

ols_model = ols_regression(x_train_set, y_train_set)
model = LinearRegression()
model.fit(x_train_set, y_train_set)

ones = np.ones((x_test_set.shape[0], 1))
x_test_aug = np.hstack([ones, x_test_set])

ols_predictions = x_test_aug @ ols_model 
model_predictions = model.predict(x_test_set)

ols_rmse = root_mean_square_error(ols_predictions, y_test_set)
ols_r2 = r2(ols_predictions, y_test_set)
model_rmse = root_mean_square_error(model_predictions, y_test_set)
model_r2 = r2(model_predictions, y_test_set)

print(f"Ols rmse= {ols_rmse}, Scikit-learn rmse= {model_rmse}\n")
print(f"Ols r squared= {ols_r2}, Scikit-learn r squared= {model_r2}\n")
print(f"Ols weights = {ols_model}\n Scikit-learn weights= {np.hstack([model.intercept_, model.coef_])}\n")
