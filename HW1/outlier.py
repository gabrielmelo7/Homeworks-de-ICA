import numpy as np
import pandas as pd
from scipy.stats import chi2
import os

# matplotlib.use("module://matplotlib-backend-kitty")
from matplotlib import patches
from matplotlib import pyplot as plt
from utils.mahalanobis import mahalanobis_distances
from utils.qq_plots import create_qq_plots

PATH = "./data_transformations/data_yeojohnson.csv"


def find_and_plot_outliers(
    df, col1, col2, confidence_level=0.95, label_col="Air Quality"
):
    """
    Calculates Mahalanobis distance for the entire dataset to find outliers,
    then plots a 2D confidence ellipse for two specified columns.

    Args:
        df (pd.DataFrame): The input data. Should be numeric only.
        col1 (str): The name of the column for the plot's x-axis.
        col2 (str): The name of the column for the plot's y-axis.
        confidence_level (float): The confidence level for the outlier cutoff.
    """

    # == Outlier Detection ==
    features_df = df.drop(columns=[label_col])
    data_array = features_df.to_numpy()
    degrees_of_freedom = features_df.shape[1]
    cutoff = chi2.ppf(confidence_level, df=degrees_of_freedom)

    distances, cov = mahalanobis_distances(data_array)

    is_outlier = distances > cutoff
    outlier_indexes = np.where(is_outlier)[0]

    print(f"Found {len(outlier_indexes)} outliers in {len(df)} samples.")

    # == Exporting the outliers to a csv file ==
    outliers = df[is_outlier]
    print(outliers)
    outliers.to_csv(os.path.join("results", r"outliers.csv"))

    # == 2D Confidence Ellipse Plot (for the chosen features) ==
    try:
        data_2d = df[[col1, col2]].to_numpy()
    except KeyError:
        print(
            f"\nError: One or both columns ('{col1}', '{col2}') not found for plotting."
        )
        return

    # Calculate properties for the 2D data
    centerpoint_2d = np.mean(data_2d, axis=0)
    cov_2d = np.cov(data_2d, rowvar=False)

    # Get and sort eigenvalues/vectors for correct orientation of the ellipse
    lambda_, v = np.linalg.eig(cov_2d)
    order = lambda_.argsort()[::-1]
    lambda_ = lambda_[order]
    v = v[:, order]
    lambda_ = np.sqrt(lambda_)
    angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))

    # Calculate the cutoff for the 2D plot
    cutoff_2d = np.sqrt(chi2.ppf(confidence_level, df=2))

    # Ellipse patch
    ellipse = patches.Ellipse(
        xy=(centerpoint_2d[0], centerpoint_2d[1]),
        width=lambda_[0] * cutoff_2d * 2,
        height=lambda_[1] * cutoff_2d * 2,
        angle=angle,
        edgecolor="red",
        facecolor="cyan",
        alpha=0.4,
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.add_artist(ellipse)
    ax.scatter(data_2d[:, 0], data_2d[:, 1])

    plt.title(f"{confidence_level * 100}% Confidence Ellipse for {col1} vs. {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    df = pd.read_csv(PATH, sep=",")

    qq_plot_figure = create_qq_plots(df)
    plt.show()
    find_and_plot_outliers(df, "PM2.5", "NO2", confidence_level=0.99)


if __name__ == "__main__":
    main()
