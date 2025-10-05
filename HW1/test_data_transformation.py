import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.data_transformations import (
    standard_zscore,
    boxcox_transform, 
    yeojohnson_transform, 
    spatial_sign_transform
);

caminho = 'C:/Users/filip/OneDrive/Documentos/Eng-Comp/2025.2/Inteligência Computacional Aplicada/Homeworks-de-ICA/HW1/data/updated_pollution_dataset.csv'
df = pd.read_csv(caminho)

df_copy = df.iloc[:,:-1]

scaled_data = standard_zscore(df_copy)
yeojohnson_data = yeojohnson_transform(df_copy)
euclidean_data = spatial_sign_transform(df_copy)

print(scaled_data)
