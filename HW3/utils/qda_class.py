import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from utils.train_test_split import splitter

class qda:
    def __init__(self, dataset, selection=9):
        dataset = pd.read_csv("HW3/data/data_yeojohnson_zscore.csv")

        self.train_y, self.train_x, self.test_y, self.test_x = splitter(dataset)
        
        df_corr = pd.concat((self.train_x, self.train_y), axis=1)
        self.corr = df_corr.corr()
        
        self.qda = QuadraticDiscriminantAnalysis()

        self.columns = self.rank_correlation(selection)
        self.qda_training()

    def rank_correlation(self, selected):
        target_name = self.train_y.name
        corrs = np.abs(self.corr[target_name])
        d_sorted_pos = np.argsort(corrs)[::-1]
        d_sorted_labels = self.corr.columns[d_sorted_pos]

        return d_sorted_labels[1:selected+1]
    
    def qda_training(self):
        print(f"Training QDA com {len(self.columns)} features")
        self.qda.fit(self.train_x[self.columns], self.train_y)

    def qda_prediction(self):
        print("Predicting")
        return self.qda.predict(self.test_x[self.columns])

    def corr_matrix_plot(self):
        mask = np.triu(np.ones_like(self.corr, dtype=bool))
        fig, axs = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        sns.heatmap(self.corr, mask=mask, cmap='rocket', annot=True, fmt='.2f')
        return fig
    
    def confusion_matrix_plot(self):
        predicted = self.qda_prediction()
        cm = confusion_matrix(self.test_y, predicted)
        self.cof_matrix = cm # Salva para usar nas métricas

        class_names = ["Hazardous", "Poor", "Moderate", "Good"]
        
        fig, axs = plt.subplots(1, 1, figsize=(8,6), dpi=150)
        sns.heatmap(cm, cmap="Blues", annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        return fig

    def principal_stats(self):
        
        if not hasattr(self, 'cof_matrix'):
            self.confusion_matrix_plot()
            plt.close()
            
        cm = self.cof_matrix
        total = np.sum(cm)
        accuracy = np.trace(cm) / total
        
        # Cálculos por classe
        precision = np.diag(cm) / np.sum(cm, axis=0) 
        recall = np.diag(cm) / np.sum(cm, axis=1)  
        f1 = 2 * (precision * recall) / (precision + recall)
        
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1 = np.nan_to_num(f1)
        
        macro_f1 = np.mean(f1)
        weighted_f1 = np.sum(f1 * np.sum(cm, axis=1)) / total

        # Formatação do Output
        B, C, G, Y, R, M = "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m", "\033[35m"
        
        details = "\n".join([
            f" {M}Class {i+1}:{R}  "
            f"{C}Rec:{R} {G}{recall[i]:.4f}{R}  "
            f"{C}Prec:{R} {G}{precision[i]:.4f}{R}  "
            f"{C}F1:{R} {G}{f1[i]:.4f}{R}"
            for i in range(len(cm))
        ])

        return (
            f"\n{B}{Y}╔{'═'*53}╗{R}\n"
            f"{B}{Y}║             QDA - EVALUATION - {len(self.columns)} FEATURES            ║{R}\n"
            f"{B}{Y}╚{'═'*53}╝{R}\n"
            f"{B}[GLOBAL]{R} Acc: {G}{accuracy:.4f}{R} | Macro F1: {G}{macro_f1:.4f}{R}\n"
            f"{B}[DETAILS]{R}\n{details}\n"
        )
