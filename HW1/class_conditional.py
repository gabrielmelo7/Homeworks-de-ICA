import pandas as pd
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
    numeric_cols = df.select_dtypes(include=[np.number])

    # Histograms
    for column in numeric_cols:
        sns.displot(
            data=df,
            x=column,
            col="Air Quality",
            col_wrap=2,
            kde=True,
            kind="hist",
        )
        plt.suptitle(f"Histograms of {column} by Class", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join("./results", f"{column}_histograms.png"))
        plt.show()

    # Box Plots
    i = 0
    plt.figure(figsize=(10, 6))
    for column in numeric_cols:
        plt.subplot(3, 3, i + 1)
        sns.boxplot(data=df, x="Air Quality", y=column)
        plt.title(f"{column} by Class")
        plt.savefig(os.path.join("./results", "boxplots.png"))
        i += 1

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
