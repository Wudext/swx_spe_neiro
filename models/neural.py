import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.train_data import prepare_train_data
from data.parce_data import load_sep_data

class StabilizedNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size=1,
                 activation='leaky_relu', output_activation='linear',
                 lr=0.0001, dropout=0.0, grad_clip=10.0, eps=1e-8):
        self.weights = []
        self.biases = []
        self.activations = []
        self.lr = lr
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.eps = eps
        self.train_mean = None
        self.train_std = None
        
        # Инициализация весов
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layers)):
            fan_in = layers[i-1]
            std = np.sqrt(2.0 / fan_in) if activation == 'relu' else np.sqrt(1.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, layers[i]) * std)
            self.biases.append(np.zeros(layers[i]))
            self.activations.append(output_activation if i == len(layers)-1 else activation)

    def fit(self, X, y, epochs=100, batch_size=32, val_split=0.2, verbose=True):
        # Предобработка данных
        self.train_mean = np.nanmean(X, axis=0)
        self.train_std = np.nanstd(X, axis=0) + self.eps
        X = self._normalize(X)
        y = self._normalize_target(y)
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )
        
        # Цикл обучения
        for epoch in range(epochs):
            # Обучение на тренировочных данных
            train_loss = self._train_epoch(X_train, y_train, batch_size)
            
            # Валидация
            val_loss = self.evaluate(X_val, y_val)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        return self

    def _train_epoch(self, X, y, batch_size):
        losses = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            mask = ~np.isnan(X_batch)
            X_batch = np.nan_to_num(X_batch)
            
            # Прямое распространение
            pred = self.forward(X_batch, mask)
            
            # Обратное распространение
            self.backward(X_batch, y_batch, mask)
            self._update_weights()
            
            # Расчет потерь
            loss = self._calculate_loss(pred, y_batch)
            losses.append(loss)
        
        return np.mean(losses)

    def forward(self, X, mask=None, training=True):
        if mask is not None:
            X = X * mask  # Применение маски
        
        self.layer_inputs = []
        self.layer_outputs = []
        self.dropout_masks = []
        
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            a = self._activation(z, self.activations[i])
            
            if training and i < len(self.weights)-1:
                dropout_mask = (np.random.rand(*a.shape) > self.dropout) / (1 - self.dropout)
                a *= dropout_mask
                self.dropout_masks.append(dropout_mask)
            
            self.layer_inputs.append(z)
            self.layer_outputs.append(a)
        
        return a

    def backward(self, X, y, mask=None):
        m = X.shape[0]
        y = y.reshape(self.layer_outputs[-1].shape)
        
        # Расчет ошибки
        error = np.clip(self.layer_outputs[-1] - y, -1e3, 1e3)
        delta = error * self._activation_derivative(self.layer_inputs[-1], self.activations[-1])
        
        self.dW = []
        self.db = []
        
        # Обратное распространение
        for i in reversed(range(len(self.weights))):
            a_prev = X if i == 0 else self.layer_outputs[i-1]
            
            dW = np.clip(np.dot(a_prev.T, delta) / m, -self.grad_clip, self.grad_clip)
            db = np.clip(np.sum(delta, axis=0) / m, -self.grad_clip, self.grad_clip)
            
            self.dW.append(dW)
            self.db.append(db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = np.clip(delta, -1e3, 1e3)
                delta *= self._activation_derivative(self.layer_inputs[i-1], self.activations[i-1])
                
                if self.dropout and i <= len(self.dropout_masks):
                    delta *= self.dropout_masks[i-1]
        
        if mask is not None:
            input_grad_mask = np.mean(mask, axis=0, keepdims=True).T
            self.dW[-1] *= input_grad_mask
        
        self.dW = self.dW[::-1]
        self.db = self.db[::-1]

    def _update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.dW[i]
            self.biases[i] -= self.lr * self.db[i]

    def predict(self, X):
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
            
        mask = ~np.isnan(X)
        X_processed = self._normalize(np.nan_to_num(X))
        return self.forward(X_processed, mask, training=False)

    def evaluate(self, X, y):
        pred = self.predict(X)
        y = self._normalize_target(y)
        return mean_squared_error(y, pred)

    def _normalize(self, X):
        return (X - self.train_mean) / self.train_std

    def _normalize_target(self, y):
        return (y - np.mean(y)) / (np.std(y) + self.eps)

    def _activation(self, z, activation):
        z = np.clip(z, -100, 100)
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -15, 15)))
        elif activation == 'tanh':
            return np.tanh(z)
        else:
            return z

    def _activation_derivative(self, z, activation):
        z = np.clip(z, -100, 100)
        if activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'sigmoid':
            s = self._activation(z, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:
            return np.ones_like(z)

    def _calculate_loss(self, pred, y):
        return mean_squared_error(y, pred)

    def save(self, filename):
        save_dict = {
            'train_mean': self.train_mean.astype(np.float32),
            'train_std': self.train_std.astype(np.float32),
            'lr': np.float32(self.lr),
            'dropout': np.float32(self.dropout),
            'grad_clip': np.float32(self.grad_clip),
            'n_layers': np.int32(len(self.weights))
        }

        # Сохраняем веса и смещения каждого слоя отдельно
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'weights_{i}'] = w.astype(np.float32)
            save_dict[f'biases_{i}'] = b.astype(np.float32)

        np.savez(filename, **save_dict)

    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        n_layers = int(data['n_layers'])
        
        # Восстанавливаем архитектуру
        hidden_sizes = [data[f'weights_{i}'].shape[1] for i in range(1, n_layers-1)]
        
        model = cls(
            input_size=data['weights_0'].shape[0],
            hidden_sizes=hidden_sizes,
            output_size=data[f'weights_{n_layers-1}'].shape[1],
            lr=float(data['lr']),
            dropout=float(data['dropout']),
            grad_clip=float(data['grad_clip'])
        )
        
        # Загружаем веса и смещения
        model.weights = [data[f'weights_{i}'].astype(np.float32) for i in range(n_layers)]
        model.biases = [data[f'biases_{i}'].astype(np.float32) for i in range(n_layers)]
        
        model.train_mean = data['train_mean'].astype(np.float32)
        model.train_std = data['train_std'].astype(np.float32)
        
        return model

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказания")
    plt.title("Фактические vs Предсказанные значения")
    plt.show()

if __name__ == "__main__":
    data = load_sep_data('data/SEPs_1996_2023corr-1.xlsx')
    X, y = prepare_train_data(data, drop_na=False)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

    model = StabilizedNeuralNetwork(
        input_size=4,
        hidden_sizes=[256, 128, 64],
        output_size=2,
        lr=0.01,
        dropout=0.2,
        grad_clip=5.0
    )
    model.fit(X_train, y_train, epochs=5000, batch_size=64)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²-Score: {r2:.4f}")

    plot_predictions(y_test, y_pred)
