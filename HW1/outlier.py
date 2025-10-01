import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2

# matplotlib.use("module://matplotlib-backend-kitty")

from matplotlib import patches
from matplotlib import pyplot as plt

PATH = "./data/SensorLog.csv"


def create_qq_plots(data):
    """
    Loads a CSV file, creates a Q-Q plot for each numeric column,
    and saves the resulting grid of plots to an image file.
    """

    df = data

    if "Timestamp" in df.columns:
        df_numeric = df.drop(columns=["Timestamp"])
    else:
        df_numeric = df

    features = df_numeric.columns
    print(f"Found features: {features.tolist()}")

    num_features = len(features)
    n_cols = 3  # You can change this to have more or fewer plots per row
    n_rows = (num_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        stats.probplot(df_numeric[feature].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot for {feature}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def find_and_plot_outliers(df, col1, col2, confidence_level=0.95):
    """
    Creates and displays a 2D confidence ellipse for two specified columns
    of a DataFrame.

    Args:
        df (pd.DataFrame): The input data.
        col1 (str): The name of the column for the x-axis.
        col2 (str): The name of the column for the y-axis.
        confidence_level (float): The confidence level for the ellipse boundary.
    """
    try:
        data_2d = df[[col1, col2]].to_numpy()
    except KeyError:
        print(
            f"Error: One or both columns ('{col1}', '{col2}') not found in the DataFrame."
        )
        return

    centerpoint = np.mean(data_2d, axis=0)
    covariance = np.cov(data_2d, rowvar=False)
    lambda_, v = np.linalg.eig(covariance)
    order = lambda_.argsort()[::-1]
    lambda_ = lambda_[order]
    v = v[:, order]
    lambda_ = np.sqrt(lambda_)

    cutoff_2d = chi2.ppf(confidence_level, df=2)

    ellipse = patches.Ellipse(
        xy=(centerpoint[0], centerpoint[1]),
        width=lambda_[0] * cutoff_2d * 2,
        height=lambda_[1] * cutoff_2d * 2,
        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
        edgecolor="#d63031",
        facecolor="#0984e3",
        alpha=0.4,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data_2d[:, 0], data_2d[:, 1], zorder=1)
    ax.add_artist(ellipse)

    plt.title(f"{confidence_level * 100}% Confidence Ellipse for {col1} vs. {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(PATH, sep=";").iloc[:, 1:]
    df.head()
    create_qq_plots(df)
    find_and_plot_outliers(df, "TempIn_1 (°C)", "HumIn_1 (%)")
