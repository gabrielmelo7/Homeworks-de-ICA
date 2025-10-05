import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def create_qq_plots(data : pd.DataFrame) -> plt.Figure:
    """
    Create a Q-Q plot for each numerical column of a DataFrame and return the Matplotlib figure object.

    Args:
        data (pd.DataFrame): O DataFrame de entrada.

    Returns:
        plt.Figure: O objeto da figura contendo a grade de Q-Q plots.
    """

    numeric_df = data.select_dtypes(include='number')
    features = numeric_df.columns

    print(f"Found features: {features.tolist()}")

    num_features = len(features)
    n_cols = 3
    n_rows = (num_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        stats.probplot(numeric_df[feature].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot for {feature}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    
    return fig