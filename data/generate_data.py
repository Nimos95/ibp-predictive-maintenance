"""
Генерация синтетических данных по ИБП (источникам бесперебойного питания)
за 2024-2025 гг. с имитацией отказов для задач предиктивного обслуживания.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Параметры генерации ---
NUM_DEVICES = 20
DEVICE_IDS = [f"IBP-{i:02d}" for i in range(1, NUM_DEVICES + 1)]
START_DATE = datetime(2024, 1, 1, 0, 0, 0)
END_DATE = datetime(2025, 12, 31, 23, 0, 0)
NUM_FAILING_DEVICES = 5
FAILURE_LEAD_WEEKS_MIN = 2  # недель до отказа начало деградации
FAILURE_LEAD_WEEKS_MAX = 4
TARGET_HORIZON_DAYS = 7  # целевая метка: отказ в ближайшие 7 дней
NOISE_STD = 0.02  # относительное СКО шума (2% от диапазона)
RANDOM_SEED = 42

# Типы устройств: доли и диапазоны базовой температуры, нагрузки и возраста
# new 20%, normal 50%, old 20%, problematic 10%
DEVICE_TYPE_RANGES = {
    "new":         {"age": (0, 6),   "temp": (23, 27), "load": (20, 35)},
    "normal":      {"age": (6, 18),   "temp": (27, 32), "load": (35, 50)},
    "old":         {"age": (18, 30), "temp": (32, 37), "load": (50, 65)},
    "problematic": {"age": (24, 36), "temp": (35, 40), "load": (60, 75)},
}
DEVICE_TYPE_COUNTS = {"new": 4, "normal": 10, "old": 4, "problematic": 2}  # 20 устройств

np.random.seed(RANDOM_SEED)


def generate_hourly_timestamps(start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Генерирует почасовые метки времени от start до end включительно."""
    return pd.date_range(start=start, end=end, freq="h")


def assign_device_types() -> dict:
    """Назначает каждому устройству тип (new / normal / old / problematic) с заданными долями."""
    type_list = []
    for dtype, count in DEVICE_TYPE_COUNTS.items():
        type_list.extend([dtype] * count)
    np.random.shuffle(type_list)
    return dict(zip(DEVICE_IDS, type_list))


def pick_failure_times(timestamps: pd.DatetimeIndex) -> dict:
    """
    Выбирает 5 устройств и для каждого — момент отказа (час).
    Отказ не в первые 2 месяца и не в последний месяц периода.
    """
    total_hours = len(timestamps)
    # окно для отказа: после 2 месяцев, до последнего месяца
    margin_start = 2 * 30 * 24
    margin_end = total_hours - 30 * 24
    failing_devices = np.random.choice(DEVICE_IDS, size=NUM_FAILING_DEVICES, replace=False)
    failure_hour_index = {}
    for dev in failing_devices:
        # случайный час в допустимом окне
        fail_idx = np.random.randint(margin_start, margin_end)
        failure_hour_index[dev] = fail_idx
    return failure_hour_index


def degradation_factor(hours_to_failure: float, lead_weeks: float) -> float:
    """
    Возвращает коэффициент деградации от 0 (норма) до 1 (момент отказа).
    Линейная деградация в окне [lead_weeks] недель до отказа.
    """
    lead_hours = lead_weeks * 7 * 24
    if hours_to_failure > lead_hours or hours_to_failure < 0:
        return 0.0
    return 1.0 - (hours_to_failure / lead_hours)


def generate_device_series(
    device_id: str,
    timestamps: pd.DatetimeIndex,
    failure_hour_index: dict,
    device_type: str,
) -> tuple:
    """
    Генерирует почасовой ряд для одного ИБП.
    Тип устройства задаёт базовые уровень температуры, нагрузки и возраст батареи.
    Для «ломающихся» устройств за 2–4 недели до отказа параметры ухудшаются.
    """
    n = len(timestamps)
    fail_idx = failure_hour_index.get(device_id)
    lead_weeks = np.random.uniform(FAILURE_LEAD_WEEKS_MIN, FAILURE_LEAD_WEEKS_MAX)

    # Базовые уровни по типу устройства (одна реализация на устройство).
    # Для new/normal — смещение в нижнюю часть диапазона (~60–70% записей в зоне «норма»).
    ranges = DEVICE_TYPE_RANGES[device_type]
    if device_type in ("new", "normal"):
        def sample_in_range(lo, hi):
            return lo + (hi - lo) * np.random.beta(1, 10)  # смещение к минимуму для целевого распределения зон
    else:
        def sample_in_range(lo, hi):
            return np.random.uniform(lo, hi)
    base_temp = sample_in_range(*ranges["temp"])
    base_load = sample_in_range(*ranges["load"])
    base_age = sample_in_range(*ranges["age"])
    # Возраст растёт за период: new +6 мес, normal +5 мес (целевое распределение зон), old/problematic +12 мес
    age_delta = 6.0 if device_type == "new" else (5.0 if device_type == "normal" else 12.0)
    age_end = min(36.0, base_age + age_delta)

    vin_min, vin_max = 200.0, 240.0
    vout_min, vout_max = 215.0, 225.0
    load_min, load_max = 20.0, 90.0
    runtime_min, runtime_max = 5.0, 40.0
    cycles_max = 500.0
    age_max = 36.0
    temp_abs_min, temp_abs_max = 20.0, 50.0

    temp = np.zeros(n)
    voltage_in = np.zeros(n)
    voltage_out = np.zeros(n)
    load_percent = np.zeros(n)
    battery_runtime = np.zeros(n)
    battery_cycles = np.zeros(n)
    battery_age = np.zeros(n)

    for i in range(n):
        # Коэффициент деградации для «ломающихся» устройств
        if fail_idx is not None:
            hours_to_fail = (fail_idx - i) * 1.0
            d = degradation_factor(hours_to_fail, lead_weeks)
        else:
            d = 0.0

        # Температура: base_temp + деградация +15°C к моменту отказа, шум sigma=1.5
        temp_center = base_temp + d * 15.0
        temp[i] = np.clip(np.random.normal(temp_center, 1.5), temp_abs_min, temp_abs_max)

        # Напряжение входа — слабая деградация (падение)
        vin_center = vin_max - d * (vin_max - vin_min) * 0.15
        voltage_in[i] = np.clip(np.random.normal(vin_center, 2), vin_min, vin_max)

        # Напряжение выхода — при деградации сдвиг вниз
        vout_center = vout_max - d * (vout_max - vout_min) * 0.2
        voltage_out[i] = np.clip(np.random.normal(vout_center, 1), vout_min, vout_max)

        # Нагрузка: base_load + деградация +25% к моменту отказа, шум sigma=3
        load_center = base_load + d * 25.0
        load_percent[i] = np.clip(np.random.normal(load_center, 3.0), load_min, load_max)

        # Время работы от батареи: при деградации падает
        runtime_center = runtime_max - d * (runtime_max - runtime_min) * 0.85 - (1 - d) * 10
        runtime_center = max(runtime_min, runtime_center)
        battery_runtime[i] = np.clip(np.random.normal(runtime_center, 2), runtime_min, runtime_max)

        # Циклы — плавный рост по времени с вариацией
        progress = i / max(n - 1, 1)
        battery_cycles[i] = np.clip(np.random.normal(progress * cycles_max * 0.6, 20), 0, cycles_max)

        # Возраст: линейно от base_age до age_end за период + случайная вариация
        age_linear = base_age + progress * (age_end - base_age)
        battery_age[i] = np.clip(age_linear + np.random.normal(0, 0.5), 0, age_max)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "device_id": device_id,
            "temperature": temp,
            "voltage_in": voltage_in,
            "voltage_out": voltage_out,
            "load_percent": load_percent,
            "battery_runtime_minutes": battery_runtime,
            "battery_cycles": battery_cycles,
            "battery_age_months": battery_age,
        }
    )
    return df, fail_idx


def add_noise(df: pd.DataFrame, noise_std: float) -> pd.DataFrame:
    """Добавляет нормальный шум к числовым колонкам (кроме timestamp и device_id)."""
    numeric_cols = ["temperature", "voltage_in", "voltage_out", "load_percent",
                    "battery_runtime_minutes", "battery_cycles", "battery_age_months"]
    out = df.copy()
    for col in numeric_cols:
        scale = out[col].std()
        if scale < 1e-6:
            scale = 1.0
        noise = np.random.normal(0, scale * noise_std, size=len(out))
        out[col] = out[col] + noise
    return out


def add_target(df: pd.DataFrame, failure_hour_index: dict) -> pd.DataFrame:
    """
    Добавляет колонку target: 1 если отказ в ближайшие 7 дней, иначе 0.
    """
    target = np.zeros(len(df), dtype=int)
    for dev, fail_idx in failure_hour_index.items():
        mask = df["device_id"] == dev
        pos = np.where(mask)[0]
        for k, idx in enumerate(pos):
            hour_idx = k  # k-я строка устройства = k-й час
            if 0 < (fail_idx - hour_idx) <= TARGET_HORIZON_DAYS * 24:
                target[idx] = 1
    df["target"] = target
    return df


def main():
    print("Генерация почасовых меток времени (2024–2025)...")
    timestamps = generate_hourly_timestamps(START_DATE, END_DATE)
    n_hours = len(timestamps)
    print(f"Всего часов: {n_hours}")

    device_types = assign_device_types()
    print("Типы устройств:", device_types)

    failure_hour_index = pick_failure_times(timestamps)
    print("Устройства с имитацией отказа:", list(failure_hour_index.keys()))

    frames = []
    for device_id in DEVICE_IDS:
        df_dev, _ = generate_device_series(
            device_id, timestamps, failure_hour_index, device_types[device_id]
        )
        frames.append(df_dev)

    df = pd.concat(frames, ignore_index=True)
    print("Добавление шума...")
    df = add_noise(df, NOISE_STD)

    # Ограничиваем значения допустимыми диапазонами и округляем
    df["temperature"] = df["temperature"].clip(20, 50).round(2)
    df["voltage_in"] = df["voltage_in"].clip(200, 240).round(2)
    df["voltage_out"] = df["voltage_out"].clip(215, 225).round(2)
    df["load_percent"] = df["load_percent"].clip(20, 90).round(2)
    df["battery_runtime_minutes"] = df["battery_runtime_minutes"].clip(5, 40).round(2)
    df["battery_cycles"] = df["battery_cycles"].clip(0, 500).round(2)
    df["battery_age_months"] = df["battery_age_months"].clip(0, 36).round(2)

    print("Расчёт целевой переменной (отказ в ближайшие 7 дней)...")
    df = add_target(df, failure_hour_index)

    out_path = os.path.join(os.path.dirname(__file__), "ups_synthetic_2024_2025.csv")
    df.to_csv(out_path, index=False, sep=",")
    print(f"Данные сохранены: {out_path}")
    print(f"Строк: {len(df)}, целевых отказов (target=1): {df['target'].sum()}")


if __name__ == "__main__":
    main()
