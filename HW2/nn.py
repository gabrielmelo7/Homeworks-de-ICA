import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.train_test_split import train_test_split


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """Inicializa uma rede neural que possui 1 hidden layer"""

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        # Input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        # Hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Feed forward do input ate a predicao"""
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.predicted_output = self.final_input

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        """Backpropagation para mudar os pesos"""
        m = X.shape[0]

        output_error = self.predicted_output - y

        # Gradientes para a camada de Saída
        d_weights_ho = (1 / m) * np.dot(self.hidden_output.T, output_error)
        d_bias_o = (1 / m) * np.sum(output_error, axis=0, keepdims=True)

        # Propagando o erro para a hidden layer
        hidden_error = np.dot(output_error, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)

        # Gradientes para a camada oculta
        d_weights_ih = (1 / m) * np.dot(X.T, hidden_delta)
        d_bias_h = (1 / m) * np.sum(hidden_delta, axis=0, keepdims=True)

        # Atualizando os pesos
        self.W2 -= learning_rate * d_weights_ho
        self.b2 -= learning_rate * d_bias_o
        self.W1 -= learning_rate * d_weights_ih
        self.b1 -= learning_rate * d_bias_h

    def train(self, X, y, epochs, learning_rate):
        loss_history = []

        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

            # Calculando o mean squarred error
            loss = np.mean(np.square(y - self.predicted_output))
            loss_history.append(loss)

        return loss_history


def cross_validation_search(X, y, k_folds=5, epochs=5000, learning_rate=0.01):
    hidden_sizes_to_test = [1, 2, 4, 8, 16]

    # Misturando os dados aleatoriamente antes de dividir
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Tamanho de cada fold
    fold_size = int(len(X) / k_folds)

    results = {}

    print(f"Iniciando Cross-Validation com {k_folds} folds...")

    for size in hidden_sizes_to_test:
        fold_errors = []

        for k in range(k_folds):
            # Criando índices para separa treino e validação
            start = k * fold_size
            end = (k + 1) * fold_size

            # Dados de Validação
            X_val = X[start:end]
            y_val = y[start:end]

            # Dados de Treino
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            nn = NeuralNetwork(
                input_size=input_dim, hidden_size=size, output_size=output_dim
            )
            nn.train(X_train, y_train, epochs, learning_rate)

            pred = nn.forward(X_val)
            mse = np.mean(np.square(y_val - pred))
            fold_errors.append(mse)

        # Média dos erros para este tamanho de neurônio
        avg_error = np.mean(fold_errors)
        results[size] = avg_error

        if epochs % 1000 == 0:
            print(f"Neurônios: {size} | Média MSE: {avg_error:.3f}")

    # Encontrar o melhor
    best_size = min(results, key=results.get)
    print(
        f"\nMelhor configuração: {best_size} neurônios (MSE: {results[best_size]:.3f})"
    )

    return best_size


def main():
    data_df = pd.read_csv("data/data_yeojohnson_zscore.csv")

    map_values = {"Hazardous": 4, "Poor": 3, "Moderate": 2, "Good": 1}

    data_df["Air Quality"].replace(map_values, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(data_df, "SO2")

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    best_hidden_size = cross_validation_search(x_train, y_train)

    input_dim = x_train.shape[1]  # Número de features
    hidden_dim = best_hidden_size
    output_dim = 1
    epochs = [1000, 2000, 5000, 10000, 20000]
    lr = 0.01

    nn = NeuralNetwork(input_dim, hidden_dim, output_dim)

    for epoch in epochs:
        print(f"Iniciando Treinamento com {epoch} epocas")
        history = nn.train(x_train, y_train, epochs=epoch, learning_rate=lr)

        print("Avaliando no conjunto de teste")
        predictions = nn.forward(x_test)
        test_loss = np.mean(np.square(y_test - predictions))
        print(f"MSE Final no Teste: {test_loss:.3f}")

        # Plotando o erro
        plt.plot(history)
        plt.title("Evolução do Erro (Loss)")
        plt.xlabel("Épocas")
        plt.ylabel("MSE")
        plt.savefig(f"./plots/MSE_Evolution_Com_{epoch}_Epochs")
        plt.show()


if __name__ == "__main__":
    main()
