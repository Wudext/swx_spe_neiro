from data.parce_data import load_sep_data
import pandas as pd
import numpy as np
import re


def convert_scientific_notation(s):
    """
    Преобразует строки вида '1.3-06' в корректный научный формат '1.3e-06'
    Сложность: O(1) на элемент
    """
    if isinstance(s, str):
        s = re.sub(r"(\d\.?\d*)[-+]0?(\d+)", r"\1e-\2", s)
    return s


def prepare_train_data(
    df: pd.DataFrame, drop_na: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготавливает данные для обучения модели прогнозирования солнечной активности.

    Параметры:
        df (pd.DataFrame): Исходный DataFrame из load_sep_data()
        drop_na (bool): Удалять строки с пропущенными значениями (по умолчанию True)

    Возвращает:
        tuple: (X_train, Y_train) - кортеж с признаками и целевыми переменными

    Сложность: O(n), n - количество строк
    """
    # Выбор признаков и целей
    features = ["Xmax, W/m2", "Lat, °", "Long, °", "V0 CME, km/s"]
    targets = ["Dt1, min", "P10, pfu"]

    # Проверка наличия колонок
    missing_feat = [col for col in features if col not in df.columns]
    missing_targ = [col for col in targets if col not in df.columns]

    if missing_feat:
        raise KeyError(f"Отсутствуют признаки: {missing_feat}")
    if missing_targ:
        raise KeyError(f"Отсутствуют цели: {missing_targ}")

    all_columns = features + targets
    df_clean = df.copy()
    df_clean[all_columns] = df_clean[all_columns].astype(str)
    df_clean.apply(convert_scientific_notation)
    df_clean.replace(to_replace="-", value=0.0, inplace=True)

    # Формирование выходных данных
    X_train = df_clean[features].astype("float32")
    Y_train = df_clean[targets].astype("float32")

    X_train['Xmax, W/m2'] = np.log10(X_train['Xmax, W/m2'] + 1e-10)
    Y_train['P10, pfu'] = np.log10(Y_train['P10, pfu'] + 1)

    return X_train, Y_train


# Пример использования
df = load_sep_data("data/SEPs_1996_2023corr-1.xlsx")
X, Y = prepare_train_data(df)
print(f"Размерность X: {X.shape}, Размерность Y: {Y.shape}")
