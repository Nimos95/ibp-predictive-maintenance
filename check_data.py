import pandas as pd
import numpy as np

# Загрузи данные
df = pd.read_csv('data/ups_synthetic_2024_2025.csv')

print("=== ПРОВЕРКА ДАННЫХ ===")
print(f"Всего записей: {len(df)}")
print(f"ИБП: {df['device_id'].nunique()}")
print(f"Отказов (target=1): {df['target'].sum()}")
print(f"Доля отказов: {df['target'].mean():.3%}")

# Распределение по зонам (по твоей формуле из app.py)
# Нормализуем как в приложении
mins = {
    "temperature": df["temperature"].min(),
    "load": df["load_percent"].min(),
    "battery_age": df["battery_age_months"].min(),
}
maxs = {
    "temperature": df["temperature"].max(),
    "load": df["load_percent"].max(),
    "battery_age": df["battery_age_months"].max(),
}

def normalize(x, min_val, max_val):
    if max_val - min_val == 0:
        return 0.5
    return (x - min_val) / (max_val - min_val)

# Расчет риска
df['risk'] = df.apply(lambda r: 
    0.4 * normalize(r['temperature'], mins['temperature'], maxs['temperature']) +
    0.3 * normalize(r['load_percent'], mins['load'], maxs['load']) +
    0.3 * normalize(r['battery_age_months'], mins['battery_age'], maxs['battery_age']), 
    axis=1
)
df['risk_pct'] = df['risk'] * 100

print("\n=== РАСПРЕДЕЛЕНИЕ ПО ЗОНАМ РИСКА (ВСЕ ДАННЫЕ) ===")
normal = (df['risk_pct'] < 30).sum()
at_risk = ((df['risk_pct'] >= 30) & (df['risk_pct'] <= 70)).sum()
critical = (df['risk_pct'] > 70).sum()
total = len(df)

print(f"Норма (<30%): {normal} ({normal/total:.1%})")
print(f"Желтая (30-70%): {at_risk} ({at_risk/total:.1%})")
print(f"Красная (>70%): {critical} ({critical/total:.1%})")

print("\n=== ПОСЛЕДНЕЕ СОСТОЯНИЕ КАЖДОГО ИБП ===")
latest = df.sort_values('timestamp').groupby('device_id').last().reset_index()
normal_l = (latest['risk_pct'] < 30).sum()
at_risk_l = ((latest['risk_pct'] >= 30) & (latest['risk_pct'] <= 70)).sum()
critical_l = (latest['risk_pct'] > 70).sum()

print(f"Норма (<30%): {normal_l}")
print(f"Желтая (30-70%): {at_risk_l}")
print(f"Красная (>70%): {critical_l}")

print("\n=== СТАТИСТИКА ПО КАЖДОМУ ИБП (последнее состояние) ===")
for _, row in latest.iterrows():
    status = "🟢" if row['risk_pct'] < 30 else "🟡" if row['risk_pct'] <= 70 else "🔴"
    print(f"{status} {row['device_id']}: риск {row['risk_pct']:.1f}%, t={row['temperature']:.1f}°C, load={row['load_percent']:.1f}%, age={row['battery_age_months']:.1f}мес")