# Importing relevant libraries and functions
from utils.correlation_matrix_plot import correlation_matrix_plot 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# We begin by importing the data with pandas
air_data = pd.read_csv("./data/updated_pollution_dataset.csv")
# Now we apply the function 
corr_matrix, scatter_matrix = correlation_matrix_plot(air_data, "Air Quality")

corr_matrix.figure.savefig(os.path.join("./results", r"air_quality_correlation_matrix.png"))
scatter_matrix.savefig(os.path.join("./results", r"AirQualityPairPlot.png"))



