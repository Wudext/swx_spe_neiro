import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from models.train_data import prepare_train_data
from data.parce_data import load_sep_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ActivationFunctions:
    @staticmethod
    def sigmoid(x, derivative=False):
        """Сигмоидная функция активации"""
        sigmoid_x = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x

    @staticmethod
    def relu(x, derivative=False):
        """ReLU функция активации"""
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def tanh(x, derivative=False):
        """Гиперболический тангенс"""
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    @staticmethod
    def softmax(x, derivative=False):
        """Softmax функция активации"""
        # Для численной стабильности вычитаем максимальное значение
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        # Производная softmax сложна для прямой реализации, поэтому
        # обычно используется в сочетании с функцией потерь
        if derivative:
            return softmax_x
        return softmax_x


class CustomActivation(nn.Module):
    """Кастомные функции активации с автоматическим выбором устройства"""

    def __init__(self, activation_type):
        super().__init__()
        self.activation_type = activation_type

        # Инициализация параметров для адаптивных функций
        if activation_type == "adaptive":
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if self.activation_type == "sigmoid":
            return torch.sigmoid(x)

        elif self.activation_type == "relu":
            return F.relu(x)

        elif self.activation_type == "tanh":
            return torch.tanh(x)

        elif self.activation_type == "softmax":
            return F.softmax(x, dim=1)

        elif self.activation_type == "swish":
            return x * torch.sigmoid(x)

        elif self.activation_type == "adaptive":
            return self.a * x * torch.sigmoid(self.b * x)

        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")


class TorchNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, activation_types, device="auto"):
        """
        Инициализация нейронной сети с поддержкой GPU

        Параметры:
        layer_sizes (list): Размеры слоев [input, hidden1, ..., output]
        activation_types (list): Типы активации для каждого слоя
        device (str): 'cuda', 'cpu' или 'auto' для автоматического выбора
        """
        super().__init__()

        # Автоматический выбор устройства
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cuda" if device == "cuda" else "cpu"
        )
        print(self.device)
        # Создание слоев
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            # Линейный слой
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Инициализация весов
            nn.init.kaiming_normal_(
                self.layers[-1].weight,
                mode="fan_in",
                nonlinearity="relu" if i < len(layer_sizes) - 2 else "linear",
            )

            # Функция активации (кроме последнего слоя)
            if i < len(layer_sizes) - 2:
                self.activations.append(CustomActivation(activation_types[i]))

        # Перемещение модели на выбранное устройство
        self.to(self.device)

    def forward(self, x):
        # Перемещение входных данных на нужное устройство
        if not x.is_cuda and self.device.type == "cuda":
            x = x.to(self.device)

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation(x)

        # Последний слой без активации
        x = self.layers[-1](x)
        return x

    def train(
        self, X_train, y_train, epochs=1000, lr=0.01, batch_size=32, verbose=True
    ):
        """
        Обучение модели с автоматическим выбором оптимательных параметров

        Параметры:
        X_train (Tensor): Обучающие данные
        y_train (Tensor): Целевые значения
        epochs (int): Количество эпох
        lr (float): Скорость обучения
        batch_size (int): Размер батча
        verbose (bool): Вывод информации о процессе
        """
        # Оптимизатор и функция потерь
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.MSELoss() if y_train.ndim == 1 else nn.CrossEntropyLoss()

        # Конвертация данных в тензоры
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).to(self.device)

        # Цикл обучения
        history = []
        for epoch in range(epochs):
            # Мини-батчи
            permutation = torch.randperm(X_train.size()[0])
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]

                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Оптимизация
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

            # Логирование
            history.append(loss.item())
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        return history

    def predict(self, X):
        """Предсказание с автоматическим выбором устройства"""
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            return self(X).cpu().numpy()


class ActivationFunctions:
    @staticmethod
    def sigmoid(x, derivative=False):
        """Сигмоидная функция активации с защитой от переполнения"""
        # Ограничиваем x для предотвращения переполнения
        x_safe = np.clip(x, -500, 500)
        sigmoid_x = 1 / (1 + np.exp(-x_safe))
        if derivative:
            return sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x

    @staticmethod
    def relu(x, derivative=False):
        """ReLU функция активации"""
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def tanh(x, derivative=False):
        """Гиперболический тангенс с защитой от переполнения"""
        # Ограничиваем x для предотвращения переполнения
        x_safe = np.clip(x, -500, 500)
        if derivative:
            return 1 - np.tanh(x_safe) ** 2
        return np.tanh(x_safe)

    @staticmethod
    def softmax(x, derivative=False):
        """Softmax функция активации с защитой от переполнения"""
        # Вычитаем максимальное значение для численной стабильности
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        # Ограничиваем для предотвращения переполнения
        x_safe = np.clip(x_shifted, -500, 500)
        exp_x = np.exp(x_safe)
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        if derivative:
            return softmax_x
        return softmax_x


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, l2_lambda=0.001):
        """
        Инициализация нейронной сети с улучшенной инициализацией весов

        Параметры:
        layer_sizes (list): Список с количеством нейронов в каждом слое
                        [входной_слой, скрытый_слой_1, ..., выходной_слой]
        activation_functions (list): Список функций активации для каждого слоя
                                    (кроме входного)
        """
        super().__init__()
        self.l2_lambda = l2_lambda

        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.num_layers = len(layer_sizes)

        # Проверка соответствия количества функций активации количеству слоев
        if len(activation_functions) != self.num_layers - 1:
            raise ValueError(
                "Количество функций активации должно быть на 1 меньше, чем количество слоев"
            )

        # Инициализация весов и смещений
        self.weights = []
        self.biases = []

        # Инициализация весов с учетом типа функции активации
        for i in range(self.num_layers - 1):
            # Выбор метода инициализации в зависимости от функции активации
            if activation_functions[i] == "relu":
                # He инициализация для ReLU
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier/Glorot инициализация для других функций
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))

            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            )
            # Инициализация смещений нулями
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        """
        Прямое распространение через сеть

        Параметры:
        X (numpy.ndarray): Входные данные размера (n_samples, input_size)

        Возвращает:
        list: Список активаций на каждом слое
        """
        activations = [X]  # Список для хранения активаций на каждом слое
        weighted_inputs = []  # Список для хранения взвешенных входов (до активации)

        # Проход через все слои
        activation = X
        for i in range(self.num_layers - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            weighted_inputs.append(z)

            # Применение соответствующей функции активации
            activation_func = getattr(ActivationFunctions, self.activation_functions[i])
            activation = activation_func(z)
            activations.append(activation)

        return activations, weighted_inputs

    def backward(self, X, y, learning_rate=0.01, clip_value=5.0):
        """
        Обратное распространение ошибки с ограничением градиентов

        Параметры:
        X (numpy.ndarray): Входные данные
        y (numpy.ndarray): Целевые значения
        learning_rate (float): Скорость обучения
        clip_value (float): Максимальное значение для ограничения градиентов

        Возвращает:
        float: Ошибка на текущей итерации
        """
        m = X.shape[0]  # Количество примеров

        # Прямое распространение
        activations, weighted_inputs = self.forward(X)

        # Вычисление ошибки на выходном слое
        output_error = activations[-1] - y

        # Инициализация градиентов
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Вычисление градиента для выходного слоя
        delta = output_error
        if self.activation_functions[-1] != "softmax":
            activation_func = getattr(
                ActivationFunctions, self.activation_functions[-1]
            )
            delta *= activation_func(weighted_inputs[-1], derivative=True)

        nabla_w[-1] = np.dot(activations[-2].T, delta) / m
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Обратное распространение ошибки через скрытые слои
        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l + 1].T)
            activation_func = getattr(
                ActivationFunctions, self.activation_functions[-l]
            )
            delta *= activation_func(weighted_inputs[-l], derivative=True)

            nabla_w[-l] = np.dot(activations[-l - 1].T, delta) / m
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True) / m

        # Ограничение градиентов для предотвращения взрывного роста
        for i in range(len(nabla_w)):
            np.clip(nabla_w[i], -clip_value, clip_value, out=nabla_w[i])
            np.clip(nabla_b[i], -clip_value, clip_value, out=nabla_b[i])

        # Обновление весов и смещений
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * nabla_w[i]
            self.biases[i] -= learning_rate * nabla_b[i]

        # Вычисление ошибки (среднеквадратичная ошибка)
        error = np.mean(np.square(output_error))
        return error

    def train(
        self,
        X_train,
        y_train,
        epochs=1000,
        optimizer_type="adam",
        lr=0.01,
        momentum=0.9,
        weight_decay=0.001,
        batch_size=32,
        verbose=True,
    ):
        """
        Обучение модели с выбором оптимизатора

        Параметры:
        X_train (Tensor): Обучающие данные
        y_train (Tensor): Целевые значения
        optimizer_type (str): Тип оптимизатора ['sgd', 'adam', 'rmsprop']
        lr (float): Скорость обучения
        momentum (float): Параметр импульса (для SGD/RMSprop)
        weight_decay (float): Коэффициент L2 регуляризации
        batch_size (int): Размер батча
        verbose (bool): Вывод информации о процессе
        """

        # Инициализация оптимизатора
        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Функция потерь
        criterion = nn.MSELoss() if y_train.ndim == 1 else nn.CrossEntropyLoss()

        # Конвертация данных
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).to(self.device)

        # Цикл обучения
        history = []
        for epoch in range(epochs):
            # Перемешивание данных
            permutation = torch.randperm(X_train.size()[0])

            # Мини-батчи
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]

                # Обнуление градиентов
                optimizer.zero_grad()

                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Обрезка градиентов
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Шаг оптимизации
                optimizer.step()

            # Логирование
            history.append(loss.item())
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        return history

    # def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=None, verbose=True):
    #     """
    #     Обучение нейронной сети

    #     Параметры:
    #     X (numpy.ndarray): Входные данные
    #     y (numpy.ndarray): Целевые значения
    #     epochs (int): Количество эпох обучения
    #     learning_rate (float): Скорость обучения
    #     batch_size (int): Размер мини-батча (если None, используется весь набор данных)
    #     verbose (bool): Выводить ли информацию о процессе обучения

    #     Возвращает:
    #     list: История ошибок по эпохам
    #     """
    #     m = X.shape[0]  # Количество примеров
    #     errors = []

    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=learning_rate,
    #         weight_decay=self.l2_lambda  # Добавляем L2 регуляризацию
    #     )

    #     for epoch in range(epochs):
    #         # Если указан размер батча, разбиваем данные на мини-батчи
    #         if batch_size:
    #             # Перемешиваем данные
    #             indices = np.random.permutation(m)
    #             X_shuffled = X[indices]
    #             y_shuffled = y[indices]

    #             # Обучение на мини-батчах
    #             for i in range(0, m, batch_size):
    #                 end = min(i + batch_size, m)
    #                 X_batch = X_shuffled[i:end]
    #                 y_batch = y_shuffled[i:end]

    #                 error = self.backward(X_batch, y_batch, learning_rate)
    #         else:
    #             # Обучение на всем наборе данных
    #             error = self.backward(X, y, learning_rate)

    #         errors.append(error)

    #         # Вывод информации о процессе обучения
    #         if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
    #             print(f"Эпоха {epoch}/{epochs}, ошибка: {error:.6f}")

    #     return errors

    def predict(self, X):
        """
        Предсказание на основе обученной модели

        Параметры:
        X (numpy.ndarray): Входные данные

        Возвращает:
        numpy.ndarray: Предсказанные значения
        """
        activations, _ = self.forward(X)
        return activations[-1]


class SolarFlareNet(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super(SolarFlareNet, self).__init__()

        # Архитектура сети
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.layers(x)

class SolarResNet(nn.Module):
    def __init__(self, input_size=4, output_size=2, hidden_size=128, num_blocks=8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Стек residual блоков
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        
        self.final = nn.Linear(hidden_size, output_size)
        
        # Инициализация выходного слоя
        nn.init.xavier_normal_(self.final.weight)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.final(x)

def load_and_preprocess_data():
    # Загрузка и предобработка данных
    data = load_sep_data("data/SEPs_1996_2023corr-1.xlsx")
    data = data.dropna()
    X, y = prepare_train_data(data)
    
    # Нормализация
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    return X, y, scaler_X, scaler_y

def train_model():
    # Загрузка данных
    X, y, scaler_X, scaler_y = load_and_preprocess_data()

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Конвертация в тензоры
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Создание DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarFlareNet().to(device)

    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=1e-5
    )  # L2 регуляризация

    # Планировщик обучения
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Обучение
    best_loss = float("inf")
    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Клиппинг градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        # Валидация
        model.eval()
        with torch.no_grad():
            test_inputs = X_test_t.to(device)
            test_labels = y_test_t.to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)

        scheduler.step(test_loss)

        # Сохранение лучшей модели
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")

        # Логирование
        if epoch % 30 == 0:
            print(f"Epoch {epoch+1}/300")
            print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
            print(f"Test Loss: {test_loss:.4f}\n")

    print(f"Final Test Loss: {best_loss:.4f}")
    return model, scaler_X, scaler_y


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказания")
    plt.title("Фактические vs Предсказанные значения")
    plt.show()

torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    model, scaler_X, scaler_y = train_model()
