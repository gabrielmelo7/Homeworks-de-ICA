from utils.svm_class import SVM
import pandas as pd
from utils.train_test_split import splitter


def main():
    # Carregando os dados
    dataset = pd.read_csv("./data/data_yeojohnson_zscore.csv")

    db_features = ["CO", "Proximity_to_Industrial_Areas"]
    target = "Air Quality"
    df_filtered = dataset[db_features + [target]].copy()

    # Separando em teste e treino
    y_train, x_train, y_test, x_test = splitter(dataset)

    # Definindo os modelos
    linear_model = SVM(method="linear")
    poly_model = SVM(method="poly")

    # Treinando os modelos
    linear_model.train(x_train, y_train)
    poly_model.train(x_train, y_train)

    # Testando os modelos
    linear_model.test(x_test, y_test)
    poly_model.test(x_test, y_test)

    # Imprimindo as métricas
    linear_model.print_metrics()
    poly_model.print_metrics()

    # Plotando e salvando as matrizes de confusao
    linear_model.gen_and_plot_confusion_matrix()
    linear_model.save_confusion_matrix()

    poly_model.gen_and_plot_confusion_matrix()
    poly_model.save_confusion_matrix()

    # Treinando com 2 features para vizualização do decision boundary
    y_train_db, x_train_db, y_test_db, x_test_db = splitter(df_filtered)

    linear_model.train(x_train_db, y_train_db)
    poly_model.train(x_train_db, y_train_db)
    linear_model.test(x_test_db, y_test_db)
    poly_model.test(x_test_db, y_test_db)
    linear_model.plot_decision_boundary()
    poly_model.plot_decision_boundary()


if __name__ == "__main__":
    main()
