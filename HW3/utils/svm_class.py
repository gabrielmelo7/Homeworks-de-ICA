from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import pyplot as plt
from sklearn import metrics


class SVM:
    def __init__(self, method="linear") -> None:
        self.method = method
        self.model = SVC(kernel=self.method)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.x_train_ref = x_train
        self.y_train_ref = y_train

    def test(self, x_test, y_test):
        self.y_test = y_test
        self.y_pred = self.model.predict(x_test)
        self.accuracy = metrics.accuracy_score(y_test, self.y_pred)
        self.precision = metrics.precision_score(
            y_test, self.y_pred, average="weighted"
        )
        self.recall = metrics.recall_score(y_test, self.y_pred, average="weighted")
        self.f1_score = metrics.f1_score(y_test, self.y_pred, average="weighted")

        self.metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }

    def print_metrics(self):
        print(f"=== RESUMO METRICAS SVM {self.method} ===")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 32)

    def gen_and_plot_confusion_matrix(self):
        self.cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        self.disp = metrics.ConfusionMatrixDisplay(confusion_matrix=self.cm)
        self.disp.plot(cmap="Blues")
        plt.title(f"Matriz de Confusão - Kernel {self.method}")
        plt.show()

    def save_confusion_matrix(self):
        self.disp.plot(cmap="Blues")
        plt.title(f"Matriz de Confusão - Kernel {self.method}")
        plt.savefig(f"./images/Matriz_de_Confusão-Kernel_{self.method}.jpg")
        plt.close()

    def plot_decision_boundary(self):
        if self.x_train_ref.shape[1] != 2:
            print(
                f"ERRO: O plot requer 2 features. Dados atuais: {self.x_train_ref.shape[1]}"
            )
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        DecisionBoundaryDisplay.from_estimator(
            self.model,
            self.x_train_ref,
            response_method="predict",
            cmap="coolwarm",
            plot_method="pcolormesh",
            shading="auto",
            alpha=0.6,
            ax=ax,
        )

        if hasattr(self.x_train_ref, "iloc"):
            x_axis = self.x_train_ref.iloc[:, 0]
            y_axis = self.x_train_ref.iloc[:, 1]
        else:
            x_axis = self.x_train_ref[:, 0]
            y_axis = self.x_train_ref[:, 1]

        ax.scatter(
            x_axis, y_axis, c=self.y_train_ref, edgecolors="k", cmap="coolwarm", s=50
        )

        plt.title(f"Decision Boundary (Kernel: {self.method})")

        if hasattr(self.x_train_ref, "columns"):
            plt.xlabel(self.x_train_ref.columns[0])
            plt.ylabel(self.x_train_ref.columns[1])

        plt.savefig(f"./images/Decision_Boundary-Kernel_{self.method}.jpg")
        plt.show()
