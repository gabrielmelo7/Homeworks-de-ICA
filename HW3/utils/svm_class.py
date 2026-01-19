from sklearn.svm import SVC
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
