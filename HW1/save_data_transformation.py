import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# Importing developed functions for qq_plots and data normalization
from utils.qq_plots import create_qq_plots
from utils.data_transformations import (
    standard_zscore,
    boxcox_transform, 
    yeojohnson_transform, 
    spatial_sign_transform
);

path = pathlib.Path('HW1/data/updated_pollution_dataset.csv')
df = pd.read_csv(path)

numeric_cols = df.select_dtypes(include=[np.number])

# Applying zscore
zscore_data = standard_zscore(numeric_cols)

# Applying boxcox
boxcox_data, lambda_values, shifts = boxcox_transform(numeric_cols)

# Applying yeojohnson
yeojohnson_data = yeojohnson_transform(numeric_cols)

# Applying spatial_sign
spatial_sign_data = spatial_sign_transform(numeric_cols)

# Combining yeojohnson with zscore
# yeojohnson -> make distributions normal
# zscore -> requires a normal distribution for better effectiveness
yeojohnson_zscore_data = standard_zscore(yeojohnson_data)

# Adding the last non-numeric column
zscore_data['Air Quality'] = df['Air Quality']
boxcox_data['Air Quality'] = df['Air Quality']
yeojohnson_data['Air Quality'] = df['Air Quality']
spatial_sign_data['Air Quality'] = df['Air Quality']
yeojohnson_zscore_data['Air Quality'] = df['Air Quality']

# Saving transformed data
zscore_data.to_csv('HW1/data_transformations/data_zscore.csv', index=False)
boxcox_data.to_csv('HW1/data_transformations/data_boxcox.csv', index=False)
yeojohnson_data.to_csv('HW1/data_transformations/data_yeojohnson.csv', index=False)
spatial_sign_data.to_csv('HW1/data_transformations/data_spatial_sign.csv', index=False)
yeojohnson_zscore_data.to_csv('HW1/data_transformations/data_yeojohnson_zscore.csv', index=False)

create_qq_plots(zscore_data)
plt.savefig('HW1/results/zscore_qq.png')

create_qq_plots(boxcox_data)
plt.savefig('HW1/results/boxcox_qq.png')

create_qq_plots(yeojohnson_data)
plt.savefig('HW1/results/yeojohnson_qq.png')

create_qq_plots(spatial_sign_data)
plt.savefig('HW1/results/spatial_sign_qq.png')

create_qq_plots(yeojohnson_zscore_data)
plt.savefig('HW1/results/yeojohnson_zscore_qq.png')
