import pandas as pd
import numpy as np
from utils.ridge_regression_module import Ridge_regression
from utils.crossvalidation import cross_validation_ridge as cvr
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from utils.r_squared import r2
from utils.rmse import root_mean_square_error
from utils.Comparison_plot import comparison_predicted_real,skewness_res_plot,correlation_plot
import matplotlib.pyplot as plt



transformed_data = pd.read_csv("./data/data_yeojohnson.csv")

print(transformed_data.SO2.max(), transformed_data.SO2.min())
#splitting the dataset:
train, test = np.split(transformed_data.sample(frac=1, random_state=42), [int(.8*len(transformed_data))])

# Splitting into training, validation and test:
#adopting 80% - training
# 20% - test

x_train = train.drop(columns=["SO2"]); x_test = test.drop(columns=["SO2"])

y_train = train.SO2; y_test = test.SO2

#CROSS-VALIDATION SCRATCH:
CROSS_VALIDATION_SCRATCH = cvr(x_train,y_train,np.arange(0,80,5))
#CROSS-VALIDATION BUILT-IN_RMSE;
alphas = np.arange(0,80,5)
ridge_cv_RMSE = RidgeCV(alphas,cv=10,scoring='neg_root_mean_squared_error')
map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }
x_train_ridge = x_train.copy(); x_train_ridge["Air Quality"] = x_train_ridge["Air Quality"].map(map_values)
x_test_ridge = x_test.copy();x_test_ridge["Air Quality"] = x_test_ridge["Air Quality"].map(map_values)
CROSS_VALIDATION_BUILT_IN_LAMBDA = ridge_cv_RMSE.fit(x_train_ridge,y_train).alpha_
CROSS_VALIDATION_BUILT_IN_RMSE = ridge_cv_RMSE.fit(x_train_ridge,y_train).best_score_
print(f"Comparação entre Scratch X Built-in - Lambdas - CROSS VALIDATION:\n\nSCRATCH: {CROSS_VALIDATION_SCRATCH.lamb.iloc[0]} \n\nBUILT-IN: {CROSS_VALIDATION_BUILT_IN_LAMBDA}")
print(f"Comparação entre Scratch X Built-in - RMSE - CROSS VALIDATION:\n\nSCRATCH: {CROSS_VALIDATION_SCRATCH.Mean_RMSE.iloc[0]} \n\nBUILT-IN: {-CROSS_VALIDATION_BUILT_IN_RMSE}")
#CROSS-VALIDATION BUILT-IN_R2;
ridge_cv = RidgeCV(alphas,cv=10,scoring='r2')
map_values = {
        'Hazardous': 4,
        'Poor': 3,
        'Moderate': 2,
        'Good': 1
        }
CROSS_VALIDATION_BUILT_IN_R2 = ridge_cv.fit(x_train_ridge,y_train).best_score_

print(f"Comparação entre Scratch X Built-in - R2 - CROSS VALIDATION:\n\nSCRATCH: {CROSS_VALIDATION_SCRATCH.Mean_r2.iloc[0]} \n\nBUILT-IN: {CROSS_VALIDATION_BUILT_IN_R2}")

def rmse_r2_model_sklean(ridge,y_test,x_test):
    y_pred = ridge.predict(x_test)
    RMSE = root_mean_square_error(y_pred,y_test)
    R2=r2(y_pred,y_test)
    return RMSE, R2

ridge = Ridge_regression(x_train,y_train,x_test,y_test)
RMSE_BUILTIN, R2_BUILTIN = rmse_r2_model_sklean(ridge_cv,y_test,x_test_ridge)

print(f"Comparação entre Scratch X Built-in no conjunto de testes:\n\nScratch:\n\nRMSE:{ridge.RSME_scratch}\t\tR2:{ridge.r2_scratch}\n\nBuilt-in:\n\nRMSE:{RMSE_BUILTIN}\t\tR2:{R2_BUILTIN}")


print(f'\n\n\nbethas for best lambda: {ridge.ridge_bethas}\n')
print(f'20 primeiras predições: {ridge.predicted_ridge(x_test_ridge)[0:20]}\n')

plt.axes(comparison_predicted_real(ridge.ridge_bethas,y_test,x_test))
plt.savefig("results/Comparison_ridge_real_pred.png",dpi=300)

skewness_res_plot(transformed_data).savefig("results/skewness-resolved.png",dpi=150)

correlation_plot(transformed_data).savefig("results/correlation_plot.png",dpi=150)





