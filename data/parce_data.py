# import pandas as pd
# import re
# from datetime import datetime
# import numpy as np

# def load_sep_data(file_path: str) -> pd.DataFrame:
#     """Загружает и обрабатывает данные из файла."""
#     df = pd.read_excel(
#         file_path,
#         sheet_name="AllPages",
#         na_filter=False
#     )
#     # Очистка названий колонок
#     df.columns = df.columns.str.replace(r'\n|\s+', '', regex=True)
#     df = df.drop([name for name in df.columns if "Unnamed" in name], axis=1)
#     print(df.info())
#     print(df.head())

#     # Обработка времени события
#     df[['Event date', 'Event extra']] = df["Eventname"].apply(lambda x: pd.Series(parse_event_name(x)))
#     df['Tо'] = df.apply(lambda row: parse_to(row['Tо'], row['Event date']), axis=1)

#     processed_rows = []

#     for _, row in df.iterrows():
#         pairs = split_tmax_jmax(row['Tmax'], row['Jmax'], row['Event date'])
#         for tmax_proc, jmax_proc in pairs:
#             new_row = row.copy()
#             new_row['Tmax_proc'] = tmax_proc
#             new_row['Jmax_proc'] = jmax_proc
#             processed_rows.append(new_row)

#     return pd.DataFrame(processed_rows)
#     # return df

# def parse_event_name(event_name: str) -> tuple:
#     """
#     Извлекает дату события и дополнительный номер из колонки Event name.
#     Поддерживает форматы: "19971104-308" и "2010.08.03-215".
#     """
#     event_name = str(event_name).replace('\n', '')

#     # Формат с точками: "2010.08.03-215"
#     if '.' in event_name:
#         match = re.match(r'(\d{4})\.(\d{2})\.(\d{2})-(\d+)', event_name)
#         if match:
#             year, month, day, extra = match.groups()
#             return datetime(int(year), int(month), int(day)), extra

#     # Формат без точек: "19971104-308"
#     match = re.match(r'(\d{4})(\d{2})(\d{2})-(\d+)', event_name)
#     if match:
#         year, month, day, extra = match.groups()
#         return datetime(int(year), int(month), int(day)), extra

#     return None, None

# def parse_to(to: str, event_date: datetime) -> datetime:
#     """
#     Обрабатывает колонку Tо (например, "07h" → 07:00:00).
#     """
#     to = str(to).replace('\n', '')
#     hours = re.search(r'(\d{1,2})h', to)
#     return event_date + pd.Timedelta(hours=int(hours.group(1))) if hours else None

# def split_tmax_jmax(tmax: str, jmax: str, base_date: datetime) -> list:
#     """
#     Разделяет Tmax и Jmax на отдельные значения по '\n' и возвращает список пар.
#     Пример:
#     Tmax = "04d11\n05d0220" → [datetime1, datetime2]
#     Jmax = "66\n17.5" → [66, 17.5]
#     """
#     tmax_parts = str(tmax).split('\n')
#     jmax_parts = str(jmax).split('\n')

#     # Обработка Tmax
#     tmax_dates = []
#     for part in tmax_parts:
#         days_match = re.search(r'(\d+)d', part)
#         hours_match = re.search(r'(\d+)h', part.split('d')[-1])
#         days = int(days_match.group(1)) if days_match else 0
#         hours = int(hours_match.group(1)) if hours_match else 0
#         tmax_dates.append(base_date + pd.Timedelta(days=days, hours=hours))

#     # Выравнивание длин списков
#     max_len = max(len(tmax_dates), len(jmax_parts))
#     tmax_dates += [None] * (max_len - len(tmax_dates))
#     jmax_parts += [None] * (max_len - len(jmax_parts))

#     return list(zip(tmax_dates, jmax_parts))

# # Пример использования
# if __name__ == "__main__":
#     df_23 = load_sep_data("data/23 цикл.xlsx")
#     df_24 = load_sep_data("data/24 цикл.xlsx")

#     combined_df = pd.concat([df_23, df_24], ignore_index=True)
#     print(combined_df[['Event date', 'Event extra', 'Tо', 'Tmax', 'Jmax']])

import pandas as pd
from datetime import timedelta
import re


def process_23_cycle(file_path: str) -> pd.DataFrame:
    """Обработка данных 23 солнечного цикла"""
    df = pd.read_excel(file_path, sheet_name="AllPages")
    df.columns = df.columns.str.replace(r"\n|\s+", "", regex=True)
    df = df.drop([name for name in df.columns if "Unnamed" in name], axis=1)
    print(df.info())

    # 1. Парсинг Event name
    df["Event_date"] = (
        df["Eventname"]
        .astype(str)
        .apply(
            lambda x: pd.to_datetime(x.split("-")[0], format="%Y%m%d", errors="coerce")
        )
    )

    # 2. Обработка T0
    def parse_t0(event_date, t0_str):
        try:
            return event_date + timedelta(hours=int(re.sub(r"\D", "", t0_str)))
        except:
            return pd.NaT

    df["T0_datetime"] = df.apply(lambda x: parse_t0(x["Event_date"], x["Tо"]), axis=1)

    # 3. Парсинг Tmax
    def parse_tmax(event_date, tmax_str):
        results = []
        for part in str(tmax_str).split("\n"):
            match = re.match(r"(\d{1,2})d(\d{1,2})h?", part.strip())
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                results.append(event_date.replace(day=days) + timedelta(hours=hours))
        return results if results else [pd.NaT]

    df["Tmax_parsed"] = df.apply(
        lambda x: parse_tmax(x["Event_date"], x["Tmax"]), axis=1
    )

    def parse_jmax(jmax_str):
        results = []
        for part in str(jmax_str).split("\n"):
            results.append(part)

        return results

    df["Jmax_parsed"] = df.apply(lambda x: parse_jmax(x["Jmax"]), axis=1)

    def split_rows(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Усовершенствованная функция для разделения строк с обработкой float
        """
        # Преобразуем одиночные значения в списки
        for col in columns:
            df[col] = df[col].apply(
                lambda x: [x] if not isinstance(x, (list, tuple)) else x
            )

        # Создаем временный столбец с кортежами значений
        df["_temp_"] = df.apply(
            lambda row: list(zip(*[row[col] for col in columns])), axis=1
        )

        # Разделяем строки и преобразуем кортежи в отдельные колонки
        df = df.explode("_temp_", ignore_index=True)
        df[columns] = pd.DataFrame(df["_temp_"].tolist(), index=df.index)

        return df.drop(columns=["_temp_"])

    df = split_rows(df, ["Tmax_parsed", "Jmax_parsed"])

    def parse_peak_time(event_date: pd.Timestamp, time_str: str) -> list:
        """Парсинг колонки 'Time of peak intensity' с обработкой специальных символов"""
        results = []

        # Разделение на отдельные временные метки
        parts = re.split(r"[●▲Ø○■□SC]+", str(time_str))

        for part in parts:
            # Очистка и нормализация строки
            clean_part = re.sub(r"[^\d<dhms]", "", part.strip()).lower()
            if not clean_part:
                continue

            # Обработка форматов с '<'
            if "<" in clean_part:
                clean_part = clean_part.replace("<", "")
                # Пропускаем метки с '<' как отдельный кейс (по ТЗ не требуется смещение)

            # Основные шаблоны
            patterns = [
                (r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", 3),  # 04d05h58m
                (r"(\d{1,2})d(\d{1,2})h", 2),  # 24d13h
                (r"(\d{1,2})h(\d{1,2})m", 2),  # 05h58m
                (r"(\d{1,2})d", 1),  # 13d
                (r"(\d{1,2})h", 1),  # 11h
                (r"(\d{1,2})m", 1),  # 58m
            ]

            matched = False
            for pattern, groups in patterns:
                match = re.match(pattern, clean_part)
                if match:
                    try:
                        days = int(match.group(1)) if groups >= 1 else 0
                        hours = int(match.group(2)) if groups >= 2 else 0
                        minutes = int(match.group(3)) if groups >= 3 else 0

                        new_date = event_date.replace(day=days) + timedelta(
                            hours=hours, minutes=minutes
                        )
                        results.append(new_date)
                        matched = True
                        break
                    except (ValueError, AttributeError):
                        continue

            if not matched:
                results.append(pd.NaT)

        return results if results else [pd.NaT]

    def extract_first_peak_time(peak_time):
        """Извлекает первое значение из списка временных меток"""
        if isinstance(peak_time, list):
            return peak_time[0] if len(peak_time) > 0 else pd.NaT
        return peak_time

    df["Peak_time"] = df.apply(
        lambda x: parse_peak_time(x["Event_date"], x["Timeofpeakintensity"]), axis=1
    )
    df["Peak_time"] = df["Peak_time"].apply(extract_first_peak_time)

    # 6. Разделение класса вспышек
    df[["Flare_class", "Flare_importance"]] = df["Classofflare"].str.split(
        "/", n=1, expand=True
    )
    df = df.drop(["Eventname", "Tо", "Tmax", "Jmax"], axis=1)

    return df


def process_24_cycle(file_path: str) -> pd.DataFrame:
    """Обработка данных 24 солнечного цикла"""
    df = pd.read_excel(file_path, sheet_name="AllPages")
    df.columns = df.columns.str.replace(r"\n|\s+", "", regex=True)
    df = df.drop([name for name in df.columns if "Unnamed" in name], axis=1)

    # 1. Парсинг Event name
    df["Event_date"] = (
        df["Eventname"]
        .astype(str)
        .apply(
            lambda x: pd.to_datetime(
                x.replace(".", "").split("-")[0], format="%Y%m%d", errors="coerce"
            )
        )
    )

    # 2. Обработка T0
    def parse_t0(event_date, t0_str):
        try:
            return event_date + timedelta(hours=int(re.sub(r"\D", "", t0_str)))
        except:
            return pd.NaT

    df["T0_datetime"] = df.apply(lambda x: parse_t0(x["Event_date"], x["Tо"]), axis=1)

    # 3. Парсинг Tmax
    def parse_tmax(event_date: pd.Timestamp, tmax_str: str) -> list:
        """Парсинг Tmax с поддержкой форматов XXdXXh и XXh"""
        results = []
        for part in str(tmax_str).strip().split("\n"):
            # Удаление всех нецифровых символов кроме d и h
            cleaned = re.sub(r"[^\dh]", "", part)

            # Случай 1: XXdXXh
            if "d" in cleaned:
                try:
                    days_str, hours_str = re.split(r"d|D", cleaned)
                    days = int(days_str) if days_str else 0
                    hours = int(hours_str.replace("h", "")) if hours_str else 0
                    new_date = event_date.replace(day=days) + pd.Timedelta(hours=hours)
                    results.append(new_date)
                except:
                    results.append(pd.NaT)

            # Случай 2: XXh
            elif "h" in cleaned:
                try:
                    hours = int(cleaned.replace("h", ""))
                    new_date = event_date + pd.Timedelta(hours=hours)
                    results.append(new_date)
                except:
                    results.append(pd.NaT)

        return results if results else [pd.NaT]

    def parse_jmax(jmax_str):
        results = []
        for part in str(jmax_str).split("\n"):
            results.append(part)

        return results

    def split_tmax_jmax(df: pd.DataFrame) -> pd.DataFrame:
        """Разделение строк с множественными значениями Tmax и Jmax"""
        # Нормализация данных
        for col in ["Tmax_parsed", "Jmax_parsed"]:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])

        # Создание временного столбца с кортежами
        df["_temp_"] = df.apply(
            lambda row: list(zip(row["Tmax_parsed"], row["Jmax_parsed"])), axis=1
        )

        # Разделение строк
        df = df.explode("_temp_", ignore_index=True)

        # Восстановление колонок
        df[["Tmax_parsed", "Jmax_parsed"]] = pd.DataFrame(
            df["_temp_"].tolist(), index=df.index
        )

        return df.drop(columns=["_temp_"])

    df["Tmax_parsed"] = df.apply(
        lambda x: parse_tmax(x["Event_date"], x["Tmax"]), axis=1
    )
    df["Jmax_parsed"] = df.apply(lambda x: parse_jmax(x["Jmax"]), axis=1)
    df = split_tmax_jmax(df)

    # 5. Время пиковой интенсивности
    def parse_peak_time(event_date: pd.Timestamp, time_str: str) -> list:
        """Парсинг колонки 'Time of peak intensity' с обработкой специальных символов"""
        results = []

        # Разделение на отдельные временные метки
        parts = re.split(r"[●▲Ø○■□SC]+", str(time_str))

        for part in parts:
            # Очистка и нормализация строки
            clean_part = re.sub(r"[^\d<dhms]", "", part.strip()).lower()
            if not clean_part:
                continue

            # Обработка форматов с '<'
            if "<" in clean_part:
                clean_part = clean_part.replace("<", "")
                # Пропускаем метки с '<' как отдельный кейс (по ТЗ не требуется смещение)

            # Основные шаблоны
            patterns = [
                (r"(\d{1,2})d(\d{1,2})h(\d{1,2})m", 3),  # 04d05h58m
                (r"(\d{1,2})d(\d{1,2})h", 2),  # 24d13h
                (r"(\d{1,2})h(\d{1,2})m", 2),  # 05h58m
                (r"(\d{1,2})d", 1),  # 13d
                (r"(\d{1,2})h", 1),  # 11h
                (r"(\d{1,2})m", 1),  # 58m
            ]

            matched = False
            for pattern, groups in patterns:
                match = re.match(pattern, clean_part)
                if match:
                    try:
                        days = int(match.group(1)) if groups >= 1 else 0
                        hours = int(match.group(2)) if groups >= 2 else 0
                        minutes = int(match.group(3)) if groups >= 3 else 0

                        new_date = event_date.replace(day=days) + timedelta(
                            hours=hours, minutes=minutes
                        )
                        results.append(new_date)
                        matched = True
                        break
                    except (ValueError, AttributeError):
                        continue

            if not matched:
                results.append(pd.NaT)

        return results if results else [pd.NaT]

    def extract_first_peak_time(peak_time):
        """Извлекает первое значение из списка временных меток"""
        if isinstance(peak_time, list):
            return peak_time[0] if len(peak_time) > 0 else pd.NaT
        return peak_time

    # Применение функции к DataFrame
    df["Peak_time"] = df.apply(
        lambda x: parse_peak_time(x["Event_date"], x["Timeofpeakintensity"]), axis=1
    )
    df["Peak_time"] = df["Peak_time"].apply(extract_first_peak_time)

    # 6. Разделение класса вспышек
    df[["Flare_class", "Flare_importance"]] = df["Classofflare"].str.split(
        "/", n=1, expand=True
    )

    return df


df_23 = process_23_cycle("data/23 цикл.xlsx")
df_24 = process_24_cycle("data/24 цикл.xlsx")

print(df_23[["Event_date", "T0_datetime", "Tmax_parsed", "Jmax_parsed", "Peak_time"]])
print(df_24[["Event_date", "T0_datetime", "Tmax_parsed", "Jmax_parsed", "Peak_time"]])
