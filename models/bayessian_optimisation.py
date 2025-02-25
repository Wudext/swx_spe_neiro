import optuna
import torch
import numpy as np
from sklearn.model_selection import KFold
from models.neural import SolarResNet, load_and_preprocess_data  # Предполагаемая структура проекта
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate

def objective(trial):
    # Определение пространства поиска
    config = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_blocks': trial.suggest_int('num_blocks', 3, 8),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'activation': trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'GELU'])
    }
    
    # Кросс-валидация
    kf = KFold(n_splits=3, shuffle=True)
    val_losses = []
    
    X, y, _, _ = load_and_preprocess_data()
    
    for train_idx, val_idx in kf.split(X):
        # Подготовка данных
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Инициализация модели
        model = SolarResNet(
            input_size=4,
            output_size=2,
            hidden_size=config['hidden_size'],
            num_blocks=config['num_blocks'],
            activation=config['activation']
        )
        
        # Обучение
        loss = model.train_resnet(
            X_train, y_train,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            batch_size=config['batch_size']
        )
        
        # Валидация
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val))
            val_loss = torch.nn.MSELoss()(val_pred, torch.FloatTensor(y_val)).item()
            val_losses.append(val_loss)
    
    return np.mean(val_losses)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner()
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

# Вывод лучших параметров
print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

plot_optimization_history(study).show()
plot_parallel_coordinate(study).show()
