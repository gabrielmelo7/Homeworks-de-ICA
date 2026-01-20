import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from utils.train_test_split import splitter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

"""
Primary steps, loading dataset

Input:
dataset <- path for dataset to be used. Prefence for yeojohnson_zscores.
"""

class lda:
    def __init__(self,dataset,selection=9):
        dataset = pd.read_csv(dataset)
        self.train_y,self.train_x, self.test_y, self.test_x = splitter(dataset)
        df = pd.concat((self.train_x,self.train_y),axis=1)
        #df.rename(columns={'Proximity_to_Industrial_Areas':'Prox_Industrial','Population_Density':'Pop_Density'},inplace=True)
        self.corr = df.corr()
        self.lda = LinearDiscriminantAnalysis(n_components=3) # 4 classes
        self.columns = self.rank_correlation(selection)
        self.lda_training()

    def corr_matrix_plot(self):
        mask = np.triu(np.ones_like(self.corr,dtype=bool))
        fig, axs = plt.subplots(1,1,figsize=(12,8),dpi=150)
        sns.heatmap(self.corr,mask=mask,cmap='rocket',annot=True,fmt='.2f')
        return fig
    
    def rank_correlation(self,selected):
        d_sorted_pos = np.argsort(np.abs(np.array(self.corr["Air Quality"])))[::-1]
        d_sorted = np.array(self.corr["Air Quality"])[d_sorted_pos]
        d_sorted_labels = np.array(self.corr.columns)[d_sorted_pos]
        final_rank = d_sorted_labels[1:selected+1]
        return final_rank

    
    def lda_training(self):
        print("Training 💪")
        #Built-in Function:
        self.lda.fit(self.train_x[self.columns], self.train_y)
        variance = self.lda.explained_variance_ratio_
        print(f"Variancia explicada:\n{variance}")

    def lda_variance_plot(self):
        variance = self.lda.explained_variance_ratio_
        df = pd.DataFrame(variance.reshape(1,3),columns=["LD1","LD2","LD3"])
        fig,axs = plt.subplots(1,1,figsize=(8,6),dpi=150)
        sns.barplot(df)
        return fig
        

    def lda_prediction(self):
        print("Predicting 🎯")
        return self.lda.predict(self.test_x[self.columns])
    
    def lda_separation_plot(self):
        X_r = self.lda.transform(self.test_x[self.columns])
        df = pd.DataFrame({"LD1":X_r[:,0],"labels":self.test_y})
        fig,axs = plt.subplots(1,1,figsize=(8,6),dpi=150)
        sns.kdeplot(df, x="LD1", hue="labels",fill=True, palette='viridis')
        return fig

    def confusion_matrix_plot(self):
        predicted = self.lda_prediction()
        fig,axs = plt.subplots(1,1,figsize=(8,6),dpi=150)
        self.cof_matrix= confusion_matrix(self.test_y,predicted)
        sns.heatmap(self.cof_matrix, cmap="Blues",annot=True, fmt='d')
        return fig
    
    def recall(self):
        recall_list = []
        n_l, n_c = self.cof_matrix.shape
        for i in range(n_l):
            sum_t = np.sum(self.cof_matrix[:,i])
            recall = self.cof_matrix[i,i]/sum_t
            recall_list.append(recall)
        return np.array(recall_list,dtype=np.float64)
    
    def precision(self):
        precision_list = []
        n_l, n_c = self.cof_matrix.shape
        for i in range(n_c):
            sum_t = np.sum(self.cof_matrix[i,:])
            precision = self.cof_matrix[i,i]/sum_t
            precision_list.append(precision)
        #print(precision_list)
        return np.array(precision_list,dtype=np.float64)
    
    def accuracy(self):
        sum_t = 0
        n_l,n_c = self.cof_matrix.shape
        for i in range(n_c):
            sum_t += self.cof_matrix[i,i]

        accuracy = sum_t/np.sum(self.cof_matrix)
        #print(accuracy)
        return accuracy
    
    def f1_score(self):
        f1_score_list = []
        n_l, n_c = self.cof_matrix.shape
        for i in range(n_c):
            sum_t = np.sum(self.cof_matrix[i,:])
            sum_t += np.sum(self.cof_matrix[:,i])
            f1_score = 2*self.cof_matrix[i,i]/sum_t
            f1_score_list.append(f1_score)
        #print("F1_score:",f1_score_list)
        return np.array(f1_score_list)
    
    def f1_average_weighted(self):
        f1_list = self.f1_score()
        sum_t = np.sum(self.cof_matrix,axis=1)
        weighted_sum = f1_list*sum_t
        average_w = np.sum(weighted_sum)/np.sum(sum_t)
        #print(average_w)
        return average_w
    
    def f1_average_simple(self):
        f1_list = self.f1_score()
        avg = np.sum(f1_list)/f1_list.shape[0]
        #print(avg)
        return avg




    def principal_stats(self):
        accuracy = self.accuracy()
        f1_simple = self.f1_average_simple()
        f1_weighted = self.f1_average_weighted()
        recall_vals = self.recall()
        precision_vals = self.precision()
        f1_vals = self.f1_score()

        # --- 2. O Código de Impressão (Cores ANSI) ---
        # B=Bold, C=Cyan, G=Green, Y=Yellow, R=Reset, M=Magenta
        B, C, G, Y, R, M = "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m", "\033[35m"

        return(
            f"\n{B}{Y}╔{'═'*53}╗{R}\n"
            f"{B}{Y}║             LDA - AVALIATION METRICS - {len(self.columns)}            ║{R}\n"
            f"{B}{Y}╚{'═'*53}╝{R}\n\n"
            f"{B}[GLOBAL STATS]{R}\n"
            f" {C}Accuracy:{R} {G}{accuracy:.4f}{R}  |  "
            f"{C}F1 (Macro):{R} {G}{f1_simple:.4f}{R}  |  "
            f"{C}F1 (Weighted):{R} {G}{f1_weighted:.4f}{R}\n\n"
            f"{B}[BY CLASS DETAILS]{R}\n" + 
            "\n".join([
                f" {M}Class {i}:{R}  "
                f"{C}Recall:{R} {G}{recall_vals[i]:.4f}{R}  "
                f"{C}Precision:{R} {G}{precision_vals[i]:.4f}{R}  "
                f"{C}F1:{R} {G}{f1_vals[i]:.4f}{R}" 
                for i in range(4)
            ]) + "\n"
        )
    

    def lda_plot(self):
        test_y = self.test_y.copy()
        label_remapping = {'1': "Hazardous", '2': "Poor",'3':"Moderate",'4':"Good"}
        fig, axs = plt.subplots(1,1,figsize=(12,8))
        X_r = self.lda.transform(self.test_x[self.columns])
        X = X_r[:,0]
        Y = X_r[:,1]
        sns.scatterplot(x=X,y=Y,hue=test_y,s=70,palette=sns.color_palette('viridis',4))
        axs.set_xlabel("LDA_1")
        axs.set_ylabel("LDA_2")
        handles, labels = axs.get_legend_handles_labels()
        print(labels)

        label_mapped = [label_remapping[item] for item in labels]

        plt.legend(handles,label_mapped)
        
        return fig
    
    def lda_3d_plot(self):
        test_y = self.test_y.copy()
        label_remapping = {1: "Hazardous", 2: "Poor",3:"Moderate",4:"Good"}
        X_r = self.lda.transform(self.test_x[self.columns])
        X = X_r[:,0]
        Y = X_r[:,1]
        Z = X_r[:,2]
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        cmap = ListedColormap(sns.color_palette('viridis',256).as_hex())
        sc = ax.scatter(X,Y,Z, s=40, c=test_y, marker='o', cmap=cmap)
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_zlabel("LD3")
        handles, _ = sc.legend_elements()
        plt.legend(handles,['Hazardous',"Poor","Moderate","Good"])

        
        return fig
        
        


