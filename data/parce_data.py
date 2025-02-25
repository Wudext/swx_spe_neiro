import pandas as pd

import pandas as pd

import pandas as pd


def load_sep_data(file_path: str) -> pd.DataFrame:
    """
    Универсальный загрузчик данных с обработкой разных форматов даты
    Сложность: O(n), n - количество строк
    """
    # Чтение сырых данных с явным указанием типов
    df = pd.read_excel(
        file_path,
        sheet_name="Table 1",
        dtype={"Date": str, "Time": str, "AR": str},
        usecols=lambda x: "Unnamed" not in x,
    )

    # Препроцессинг даты и времени
    df["Date"] = (
        df["Date"].str.replace(".", "-", regex=False).str.split(" ").str[0].str.strip()
    )
    df["Time"] = df["Time"].str.split(".").str[0].str.strip()
    datetime_str = df["Date"] + " " + df["Time"]

    # Обработка AR
    df["AR"] = pd.to_numeric(df["AR"], errors="coerce").astype("Int32")
    # Парсинг datetime с обработкой ошибок
    df["datetime"] = pd.to_datetime(
        datetime_str, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Валидация и логирование ошибок
    na_count = df["datetime"].isna().sum()
    if na_count > 0:
        print(f"Предупреждение: {na_count} некорректных временных меток")
        print("Примеры проблемных строк:")
        print(df[df["datetime"].isna()][["Date", "Time"]].head(3))

    # Удаление исходных колонок
    df.drop(columns=["Date", "Time"], inplace=True)

    # Обработка числовых колонок
    num_cols = ["Lat, °", "Long, °", "V0 CME, km/s", "P10, pfu", "P100, pfu"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Оптимизация типов
    type_map = {"X-ray flare": "category", "AR": "Int32", "GLE, %": "float32"}
    df = df.astype(type_map)
    cols = ["datetime"] + [col for col in df.columns if col != "datetime"]
    df = df[cols]

    return df


# Пример использования
if __name__ == "__main__":
    df = load_sep_data("data/SEPs_1996_2023corr-1.xlsx")
    print(df.info())
    print(df.head())
