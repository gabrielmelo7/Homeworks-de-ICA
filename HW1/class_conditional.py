import pandas as pd
from scipy.stats import skew
import numpy as np
import os

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

    return pd.DataFrame(results)


def plot_class_conditional(df):
    # TODO
    pass


if __name__ == "__main__":
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
