"""
Streamlit-приложение для мониторинга парка ИБП.
Загрузка CSV, расчёт риска по правилам, дашборд и детали по устройствам.

Запуск: streamlit run app.py
"""

import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# При запуске через "python app.py" контекста Streamlit нет — выходим без вызова st.*
try:
    from streamlit.runtime.scriptrunner_utils import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("Запустите приложение командой: streamlit run app.py")
        sys.exit(0)
except Exception:
    pass

# --- Настройка страницы ---
st.set_page_config(
    page_title="Мониторинг ИБП",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Расчёт риска по экспертным порогам ---
def compute_risk_row_fixed(row):
    """Риск по фиксированным порогам: 0.4*temp + 0.3*load + 0.3*age (в долях 0–1)."""
    # Температура: 20-50°C, норма до 30°C, критично после 45°C
    if row["temperature"] <= 30:
        norm_temp = 0.0
    elif row["temperature"] >= 45:
        norm_temp = 1.0
    else:
        norm_temp = (row["temperature"] - 30) / (45 - 30)

    # Нагрузка: 20-90%, норма до 40%, критично после 80%
    if row["load"] <= 40:
        norm_load = 0.0
    elif row["load"] >= 80:
        norm_load = 1.0
    else:
        norm_load = (row["load"] - 40) / (80 - 40)

    # Возраст батареи: 0-36 мес, норма до 12 мес, критично после 30 мес
    if row["battery_age"] <= 12:
        norm_age = 0.0
    elif row["battery_age"] >= 30:
        norm_age = 1.0
    else:
        norm_age = (row["battery_age"] - 12) / (30 - 12)

    risk = 0.4 * norm_temp + 0.3 * norm_load + 0.3 * norm_age
    return min(1.0, max(0.0, risk))

def prepare_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Приводит колонки к единым именам и добавляет риск."""
    if df is None:
        return None
    df = df.copy()
    # Единые имена колонок (поддержка разных CSV)
    col_map = {
        "load_percent": "load",
        "battery_age_months": "battery_age",
        "battery_runtime_minutes": "battery_runtime",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "load" not in df.columns and "load_percent" in df.columns:
        df["load"] = df["load_percent"]
    if "battery_age" not in df.columns and "battery_age_months" in df.columns:
        df["battery_age"] = df["battery_age_months"]
    if "battery_runtime" not in df.columns and "battery_runtime_minutes" in df.columns:
        df["battery_runtime"] = df["battery_runtime_minutes"]

    # Время
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.NaT

    df["risk"] = df.apply(compute_risk_row_fixed, axis=1)
    df["risk_pct"] = (df["risk"] * 100).round(1)
    return df

def get_current_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Последнее состояние по каждому device_id (последняя запись по времени)."""
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return df.groupby("device_id", as_index=False).last()
    return (
        df.sort_values("timestamp")
        .groupby("device_id", as_index=False)
        .last()
    )

def risk_status(risk_pct):
    """Возвращает статус: normal / at_risk / critical."""
    if risk_pct < 30:
        return "normal"
    if risk_pct <= 70:
        return "at_risk"
    return "critical"

def risk_color(risk_pct):
    """Цвет для строки таблицы."""
    if risk_pct < 30:
        return "background-color: rgba(0, 200, 83, 0.25);"
    if risk_pct <= 70:
        return "background-color: rgba(255, 193, 7, 0.35);"
    return "background-color: rgba(244, 67, 54, 0.35);"

# --- Загрузка файла и кэш ---
@st.cache_data(ttl=300)
def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)

# --- Боковая панель: загрузка и навигация ---
with st.sidebar:
    st.title("🔋 Мониторинг ИБП")
    uploaded = st.file_uploader(
        "Загрузите CSV с данными ИБП",
        type=["csv"],
        help="Файл должен содержать колонки: device_id, temperature, нагрузка (load или load_percent), battery_age (или battery_age_months), при необходимости timestamp, battery_runtime (или battery_runtime_minutes).",
    )
    st.caption("Инструкция: выберите CSV-файл с почасовыми или дневными данными по устройствам. После загрузки станут доступны все страницы дашборда.")
    st.divider()
    page = st.radio(
        "Страница",
        ["Общий дашборд", "Детали ИБП", "Прогнозы"],
        label_visibility="collapsed",
    )

if uploaded is None:
    st.info("👈 Загрузите CSV-файл с данными ИБП в боковой панели, чтобы начать работу.")
    st.markdown("""
    **Инструкция по загрузке файла:**
    - Нажмите **«Browse files»** или перетащите файл в область загрузки в левой панели.
    - Поддерживается формат **CSV** с разделителем запятая.
    - В файле должны быть колонки: **device_id**, **temperature**, нагрузка (**load** или **load_percent**), **battery_age** (или **battery_age_months**).
    - Для графиков за последние 30 дней желательны колонки **timestamp** и **battery_runtime** (или **battery_runtime_minutes**).
    - Пример: можно использовать сгенерированный файл `data/ups_synthetic_2024_2025.csv`.
    """)
    st.stop()
    sys.exit(0)

df_raw = load_csv(uploaded)
if df_raw is None or df_raw.empty:
    st.error("Не удалось прочитать файл или файл пуст.")
    st.stop()
    sys.exit(1)

df = prepare_data(df_raw)
if df is None:
    st.error("Ошибка подготовки данных.")
    st.stop()
    sys.exit(1)
current = get_current_snapshot(df)
current["status"] = current["risk_pct"].apply(risk_status)

# --- Страница 1: Общий дашборд ---
if page == "Общий дашборд":
    st.header("Мониторинг парка ИБП")

    total = len(current)
    normal = (current["risk_pct"] < 30).sum()
    at_risk = ((current["risk_pct"] >= 30) & (current["risk_pct"] <= 70)).sum()
    critical = (current["risk_pct"] > 70).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего ИБП", total)
    col2.metric("В норме (риск <30%)", normal)
    col3.metric("Под угрозой (30–70%)", at_risk)
    col4.metric("Критично (>70%)", critical)

    st.subheader("Таблица ИБП")
    status_filter = st.selectbox(
        "Фильтр по статусу",
        ["Все", "В норме", "Под угрозой", "Критично"],
        key="filter_status",
    )
    filter_map = {
        "Все": None,
        "В норме": "normal",
        "Под угрозой": "at_risk",
        "Критично": "critical",
    }
    filtered = current
    if filter_map[status_filter]:
        filtered = current[current["status"] == filter_map[status_filter]]

    display_cols = ["device_id", "temperature", "load", "battery_age", "risk_pct"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    table_df = filtered[display_cols].copy()
    table_df = table_df.rename(columns={
        "risk_pct": "риск отказа, %",
        "device_id": "device_id",
        "temperature": "температура",
        "load": "нагрузка",
        "battery_age": "возраст батареи",
    })

    def style_rows(row):
        r = row.get("риск отказа, %") if "риск отказа, %" in row.index else row.get("risk_pct")
        if pd.isna(r):
            return [""] * len(row)
        return [risk_color(float(r))] * len(row)

    st.dataframe(
        table_df.style.apply(style_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )

# --- Страница 2: Детали ИБП ---
elif page == "Детали ИБП":
    st.header("Детали ИБП")
    device_ids = sorted(current["device_id"].unique().tolist())
    device_id = st.selectbox("Выберите device_id", device_ids, key="detail_device")

    if device_id:
        dev_current = current[current["device_id"] == device_id].iloc[0]
        dev_df = df[df["device_id"] == device_id].copy()
        if "timestamp" in dev_df.columns and dev_df["timestamp"].notna().any():
            dev_df = dev_df.sort_values("timestamp")
            last_30_days = dev_df["timestamp"].max() - pd.Timedelta(days=30)
            dev_30 = dev_df[dev_df["timestamp"] >= last_30_days]
        else:
            dev_30 = dev_df.tail(720)  # условно последние 30 дней по записям

        st.subheader(f"Текущие параметры: {device_id}")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Температура, °C", f"{dev_current['temperature']:.1f}")
        with c2:
            st.metric("Нагрузка, %", f"{dev_current['load']:.1f}")
        with c3:
            st.metric("Возраст батареи", f"{dev_current['battery_age']:.1f}")
        with c4:
            st.metric("Риск отказа, %", f"{dev_current['risk_pct']:.1f}")
        with c5:
            if "battery_runtime" in dev_current:
                st.metric("Время работы от АКБ, мин", f"{dev_current['battery_runtime']:.1f}")

        st.subheader("Графики за последние 30 дней")
        if dev_30.empty:
            st.warning("Нет данных за последние 30 дней для выбранного устройства.")
        else:
            if "timestamp" in dev_30.columns and dev_30["timestamp"].notna().any():
                x = dev_30["timestamp"]
            else:
                x = dev_30.index

            fig_temp = px.line(x=x, y=dev_30["temperature"], labels={"x": "Дата", "y": "Температура, °C"}, title="Температура")
            fig_temp.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)

            fig_load = px.line(x=x, y=dev_30["load"], labels={"x": "Дата", "y": "Нагрузка, %"}, title="Нагрузка")
            fig_load.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_load, use_container_width=True)

            if "battery_runtime" in dev_30.columns:
                fig_rt = px.line(x=x, y=dev_30["battery_runtime"], labels={"x": "Дата", "y": "Время работы от АКБ"}, title="Battery runtime")
                fig_rt.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_rt, use_container_width=True)
            else:
                st.caption("Колонка battery_runtime отсутствует в данных — график не построен.")

        risk_pct = float(dev_current["risk_pct"])
        st.subheader("Риск и рекомендация")
        st.metric("Текущий риск отказа", f"{risk_pct:.1f}%")
        if risk_pct < 30:
            st.success("Рекомендация: устройство в норме. Плановый осмотр по графику.")
        elif risk_pct <= 70:
            st.warning("Рекомендация: усилить мониторинг, запланировать проверку батареи и нагрузки в ближайшее время.")
        else:
            st.error("Рекомендация: критический риск. Необходима срочная диагностика и при необходимости замена батареи или снижение нагрузки.")

# --- Страница 3: Прогнозы ---
else:
    st.header("Прогнозы")
    high_risk = current[current["risk_pct"] > 50].sort_values("risk_pct", ascending=False)
    st.subheader("ИБП с риском >50%")
    if high_risk.empty:
        st.info("Нет устройств с риском выше 50%.")
    else:
        cols_show = ["device_id", "temperature", "load", "battery_age", "risk_pct"]
        cols_show = [c for c in cols_show if c in high_risk.columns]
        st.dataframe(high_risk[cols_show].rename(columns={"risk_pct": "риск, %"}), use_container_width=True, hide_index=True)

    st.subheader("Количество ИБП по зонам риска")
    zone_counts = pd.DataFrame({
        "Зона": ["В норме (<30%)", "Под угрозой (30–70%)", "Критично (>70%)"],
        "Количество": [
            (current["risk_pct"] < 30).sum(),
            ((current["risk_pct"] >= 30) & (current["risk_pct"] <= 70)).sum(),
            (current["risk_pct"] > 70).sum(),
        ],
    })
    fig_zones = px.bar(zone_counts, x="Зона", y="Количество", text="Количество", color="Зона", color_discrete_map={
        "В норме (<30%)": "#4CAF50",
        "Под угрозой (30–70%)": "#FFC107",
        "Критично (>70%)": "#F44336",
    })
    fig_zones.update_layout(height=400, showlegend=False, xaxis_tickangle=-25)
    fig_zones.update_traces(textposition="outside")
    st.plotly_chart(fig_zones, use_container_width=True)
