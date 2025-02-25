import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from models.train_data import prepare_train_data
from data.parce_data import load_sep_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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


class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.3, activation='ReLU'):
        super().__init__()
        self.activation = getattr(nn, activation)()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )
        self.relu = nn.ReLU()

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity  # Skip connection
        return self.relu(out)

class SolarResNet(nn.Module):
    def __init__(self, input_size=4, output_size=2, hidden_size=128, num_blocks=8, activation='relu'):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()
        )

        # Стек residual блоков
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, activation=activation) for _ in range(num_blocks)]
        )

        self.final = nn.Linear(hidden_size, output_size)

        # Инициализация выходного слоя
        nn.init.xavier_normal_(self.final.weight)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.final(x)

    def train_resnet(self, X, y, epochs: int = 500, lr=0.001, weight_decay=1e-5, batch_size=64):
        # Загрузка данных (используем предыдущую реализацию)
        # X, y, scaler_X, scaler_y = load_and_preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Конвертация в тензоры
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Инициализация модели
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.to(device)

        # Оптимизатор с L2-регуляризацией
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Планировщик скорости обучения
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

        # Обучение
        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            # Валидация
            model.eval()
            with torch.no_grad():
                test_inputs = torch.FloatTensor(X_test).to(device)
                test_labels = torch.FloatTensor(y_test).to(device)
                test_loss = nn.MSELoss()(model(test_inputs), test_labels)

            scheduler.step(test_loss)

            # Логирование
            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {test_loss:.4f}"
                )

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), "best_resnet.pth")

        print(f"Best Test Loss: {best_loss:.4f}")


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

torch.manual_seed(42)
np.random.seed(42)

def predict_and_evaluate(model, X_test, y_test, scaler_X, scaler_y):
    # Преобразование тестовых данных
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Предсказание модели
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    
    # Обратное преобразование масштабирования
    predictions_real = scaler_y.inverse_transform(predictions)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    # Экспоненцирование для P10
    predictions_real[:, 1] = 10**predictions_real[:, 1]
    y_test_real[:, 1] = 10**y_test_real[:, 1]
    
    return y_test_real, predictions_real

def visualize_results(y_true, y_pred):
    # Настройка стиля графиков
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))
    
    # Визуализация для Dt1
    ax[0,0].scatter(y_true[:,0], y_pred[:,0], alpha=0.6)
    ax[0,0].plot([0, max(y_true[:,0])], [0, max(y_true[:,0])], 'r--')
    ax[0,0].set_xlabel('Real Dt1 (min)')
    ax[0,0].set_ylabel('Predicted Dt1 (min)')
    ax[0,0].set_title(f'Dt1 Prediction (R² = {r2_score(y_true[:,0], y_pred[:,0]):.2f})')
    
    # Визуализация для P10
    ax[0,1].scatter(y_true[:,1], y_pred[:,1], alpha=0.6)
    ax[0,1].plot([1, max(y_true[:,1])], [1, max(y_true[:,1])], 'r--')
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('Real P10 (pfu)')
    ax[0,1].set_ylabel('Predicted P10 (pfu)')
    ax[0,1].set_title(f'P10 Prediction (R² = {r2_score(y_true[:,1], y_pred[:,1]):.2f})')

    # Распределение ошибок для Dt1
    errors_dt = y_pred[:,0] - y_true[:,0]
    sns.histplot(errors_dt, kde=True, ax=ax[1,0])
    ax[1,0].set_xlabel('Prediction Error (Dt1)')
    ax[1,0].set_title(f'MAE: {mean_absolute_error(y_true[:,0], y_pred[:,0]):.1f} min')

    # Распределение ошибок для P10
    errors_p10 = np.log10(y_pred[:,1]) - np.log10(y_true[:,1])
    sns.histplot(errors_p10, kde=True, ax=ax[1,1])
    ax[1,1].set_xlabel('Log10 Prediction Error (P10)')
    ax[1,1].set_title(f'Log RMSE: {np.sqrt(mean_squared_error(np.log10(y_true[:,1]), np.log10(y_pred[:,1]))):.2f}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = SolarResNet()
    X, y, scaler_X, scaler_y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.train_resnet(X_train, y_train, epochs=5000)
    
    # Предсказание и визуализация
    y_test_real, predictions_real = predict_and_evaluate(model, X_test, y_test, scaler_X, scaler_y)
    visualize_results(y_test_real, predictions_real)
    
    # Вывод метрик
    print("Dt1 Metrics:")
    print(f"MAE: {mean_absolute_error(y_test_real[:,0], predictions_real[:,0]):.1f} min")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_real[:,0], predictions_real[:,0])):.1f} min")
    print(f"R²: {r2_score(y_test_real[:,0], predictions_real[:,0]):.2f}\n")
    
    print("P10 Metrics:")
    print(f"MAE: {mean_absolute_error(y_test_real[:,1], predictions_real[:,1]):.1f} pfu")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_real[:,1], predictions_real[:,1])):.1f} pfu")
    print(f"R²: {r2_score(y_test_real[:,1], predictions_real[:,1]):.2f}")
