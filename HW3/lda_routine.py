from utils.lda_class import lda
import matplotlib.pyplot as plt
import time

"""
Loops for descending order in correlation with Air Quality labels.

Executes a routine to calculate and visualize performance across different 
clustering granularities (from 3 to 9 classes):

* **Correlation Matrix:** Saves heatmap of feature correlations.
* **Variance Explained:** Visualizes the informative power of Linear Discriminants (LDs).
* **Confusion Matrix:** Saves the prediction heatmap.
* **Separation Plot:** Visualizes the class separation in the LDA projected space.

Console Output (principal_stats):
    Prints a detailed evaluation report containing:
    - Global Metrics: Overall Accuracy
    - Per-Class Metrics: Precision, Recall (Sensitivity), and F1-Score for each of the i classes.
    - Aggregated Metrics: F1-Score Macro Average (balanced) and F1-Score Weighted Average.

Files are saved automatically to the 'images/' directory.
"""

for i in range(3,10):
    
    lda_obj = lda("data/data_yeojohnson_zscore.csv",i)

    #plotting correlation matrix:
    fig = lda_obj.corr_matrix_plot();fig.tight_layout();fig.savefig("images/Correlation_matrix.png");plt.close()
    #Variance explanation by LDA:
    fig = lda_obj.lda_variance_plot();fig.tight_layout();fig.savefig("images/Variance_lda_plot.png");plt.close()
    #Confusion matrix
    fig = lda_obj.confusion_matrix_plot(); fig.tight_layout();fig.savefig("images/Confusion Matrix.png");plt.close()
    #Separation between labels explained in LDA domain:
    fig = lda_obj.lda_separation_plot(); fig.tight_layout();fig.savefig(f"images/separation/Separation btwn labels - {i}.png");plt.close()
    #Principal stats:
    print(lda_obj.principal_stats())

lda_obj = lda("data/data_yeojohnson_zscore.csv")
fig = lda_obj.lda_plot();fig.tight_layout(); fig.savefig(f"images/lda_plot.png",dpi=100);plt.close()
fig = lda_obj.lda_3d_plot(); fig.savefig(f"images/lda_3d_plot.png",bbox_inches='tight');plt.close()

    #time.sleep(0.5)
