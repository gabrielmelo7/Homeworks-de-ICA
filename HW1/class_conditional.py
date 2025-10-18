import pandas as pd
import math
from scipy.stats import skew
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

PATH = "./data/updated_pollution_dataset.csv"


def compute_class_conditional_statistics(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    results = {}
    by_class = df.groupby("Air Quality")
    aux_df = by_class.get_group(cls)

    for column in aux_df.columns:
        if column == "Air Quality":
            continue
        results[column] = {
            "mean": np.mean(aux_df[column]),
            "std": np.std(aux_df[column]),
            "skewness": skew(aux_df[column]),
        }

    return pd.DataFrame(results).T


def plot_class_conditional(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_features = len(numeric_cols)

    n_cols_hist = 3
    n_rows_hist = math.ceil(n_features / n_cols_hist)

    fig_hist, axes_hist = plt.subplots(
        n_rows_hist, n_cols_hist, figsize=(15, 4 * n_rows_hist)
    )
    axes_hist = axes_hist.flatten()

    for i, column in enumerate(numeric_cols):
        ax = axes_hist[i]

        sns.histplot(
            data=df,
            x=column,
            hue="Air Quality",
            kde=True,
            ax=ax,
            element="step",
            stat="density",
            common_norm=False,
        )

    for j in range(i + 1, len(axes_hist)):
        axes_hist[j].set_visible(False)

    plt.suptitle("Class-Conditional Histograms", fontsize=16, y=1.03)
    plt.savefig(
        os.path.join("./results/class_conditional/", "histograms_class_conditional.png")
    )
    plt.tight_layout()
    plt.show()

    # Box Plots
    i = 0
    plt.figure(figsize=(15, 20))
    for column in numeric_cols:
        plt.subplot(3, 3, i + 1)
        sns.boxplot(data=df, x="Air Quality", y=column)
        plt.title(f"{column} by Class")
        i += 1

    plt.savefig(os.path.join("./results/class_conditional/", "boxplots.png"))
    plt.tight_layout()
    plt.show()


def main():
    try:
        df = pd.read_csv(PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {PATH}")

    har_df = compute_class_conditional_statistics(df, cls="Hazardous")
    poor_df = compute_class_conditional_statistics(df, cls="Poor")
    moderate_df = compute_class_conditional_statistics(df, cls="Moderate")
    good_df = compute_class_conditional_statistics(df, cls="Good")

    har_df.to_csv(os.path.join("./results", r"har_stats.csv"))
    poor_df.to_csv(os.path.join("./results", r"poor_stats.csv"))
    moderate_df.to_csv(os.path.join("./results", r"moderate_stats.csv"))
    good_df.to_csv(os.path.join("./results", r"good_stats.csv"))

    plot_class_conditional(df)


if __name__ == "__main__":
    main()
