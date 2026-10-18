"""แอปพยากรณ์สภาพอากาศ จังหวัดพระนครศรีอยุธยา ล่วงหน้า 7 วัน ด้วยการเรียนรู้ของเครื่อง (Streamlit)

ตัวแปรที่พยากรณ์ : อุณหภูมิสูงสุด/ต่ำสุด, ปริมาณฝน (รายวันและสะสม), PM2.5 เฉลี่ยรายวัน
แหล่งข้อมูล      : Open-Meteo Weather API และ Air Quality API (ไม่ต้องใช้ API key)
โมเดล           : HistGradientBoostingRegressor (scikit-learn) แบบ direct multi-horizon
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_START = dt.date(2022, 8, 1)  # ข้อมูลคุณภาพอากาศระดับโลกของ Open-Meteo เริ่มต้นช่วงกลางปี 2022
TZ = "Asia/Bangkok"

WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]
PM25 = "pm25"
FEATURE_VARS = WEATHER_VARS + [PM25]

TARGETS = {
    "temperature_2m_max": "อุณหภูมิสูงสุด (°C)",
    "temperature_2m_min": "อุณหภูมิต่ำสุด (°C)",
    "precipitation_sum": "ปริมาณฝน (มม.)",
    PM25: "PM2.5 (µg/m³)",
}
RAIN_LABEL = TARGETS["precipitation_sum"]
RAIN_CUM_LABEL = "ฝนสะสมพยากรณ์ (มม.)"
PM_LABEL = TARGETS[PM25]

HORIZONS = 7
LAGS = [0, 1, 2, 3, 6, 13]
MIN_ROWS = 200  # จำนวนข้อมูลขั้นต่ำที่ใช้ฝึกโมเดลของตัวแปรหนึ่ง ๆ
PM25_LIMIT = 37.5  # มาตรฐาน PM2.5 เฉลี่ย 24 ชั่วโมงของไทย (µg/m³)

LOCATIONS = {
    "พระนครศรีอยุธยา (ตัวเมือง)": (14.3532, 100.5689),
    "กำหนดพิกัดเอง": None,
}

# เกณฑ์ดัชนีคุณภาพอากาศ PM2.5 (24 ชม.) — ควรตรวจสอบกับกรมควบคุมมลพิษก่อนใช้งานจริง
PM25_LEVELS = [
    (15.0, "ดีมาก"),
    (25.0, "ดี"),
    (37.5, "ปานกลาง"),
    (75.0, "เริ่มมีผลกระทบต่อสุขภาพ"),
    (float("inf"), "มีผลกระทบต่อสุขภาพ"),
]


def pm25_category(value: float) -> str:
    for upper, name in PM25_LEVELS:
        if value <= upper:
            return name
    return PM25_LEVELS[-1][1]


# ---------------------------------------------------------------- ข้อมูล
def _to_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def load_weather(lat: float, lon: float, years: int) -> pd.DataFrame:
    today = dt.date.today()
    common = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(WEATHER_VARS),
        "timezone": TZ,
    }
    hist = requests.get(
        ARCHIVE_URL,
        params={
            **common,
            "start_date": (today - dt.timedelta(days=365 * years)).isoformat(),
            "end_date": (today - dt.timedelta(days=6)).isoformat(),
        },
        timeout=60,
    )
    hist.raise_for_status()
    recent = requests.get(
        FORECAST_URL,
        params={**common, "past_days": 14, "forecast_days": 1},
        timeout=60,
    )
    recent.raise_for_status()

    df = pd.concat([_to_frame(hist.json()), _to_frame(recent.json())])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[df.index < pd.Timestamp(today)]  # ตัดวันนี้ (ข้อมูลยังไม่ครบวัน)


def load_pm25(lat: float, lon: float, years: int) -> pd.Series:
    """PM2.5 รายชั่วโมง -> ค่าเฉลี่ยรายวัน (ต้องมีข้อมูลอย่างน้อย 18 ชั่วโมงต่อวัน)"""
    today = dt.date.today()
    start = max(today - dt.timedelta(days=365 * years), AIR_START)
    r = requests.get(
        AIR_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
            "start_date": start.isoformat(),
            "end_date": (today - dt.timedelta(days=1)).isoformat(),
            "timezone": TZ,
        },
        timeout=90,
    )
    r.raise_for_status()
    hourly = pd.DataFrame(r.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    series = hourly.set_index("time")["pm2_5"]
    grouped = series.resample("D")
    return grouped.mean().where(grouped.count() >= 18).rename(PM25)


def load_data(lat: float, lon: float, years: int) -> pd.DataFrame:
    df = load_weather(lat, lon, years)
    try:
        df = df.join(load_pm25(lat, lon, years), how="left")
    except (requests.RequestException, KeyError, ValueError):
        df[PM25] = np.nan  # ถ้าดึง PM2.5 ไม่ได้ ระบบยังพยากรณ์ตัวแปรอื่นต่อได้
    # เติมช่องว่างสั้น ๆ (ไม่เติมปริมาณฝน เพราะการประมาณค่าจะทำให้ฝนคลาดเคลื่อน)
    cols = [c for c in df.columns if c != "precipitation_sum"]
    df[cols] = df[cols].interpolate(limit=3)
    return df


# ---------------------------------------------------------------- โมเดล
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    for col in FEATURE_VARS:
        for lag in LAGS:
            feats[f"{col}_lag{lag}"] = df[col].shift(lag)
    for col in ("temperature_2m_mean", "precipitation_sum", PM25):
        feats[f"{col}_roll7"] = df[col].rolling(7, min_periods=5).mean()
        feats[f"{col}_roll30"] = df[col].rolling(30, min_periods=20).mean()
    doy = df.index.dayofyear.to_numpy()
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return feats


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )


def build_forecast(df: pd.DataFrame):
    """ฝึกโมเดลแยกตามตัวแปรและจำนวนวันล่วงหน้า แล้วพยากรณ์จากข้อมูลวันล่าสุด"""
    X = make_features(df)
    x_latest = X.iloc[[-1]]
    last_date = df.index[-1]

    days = [(last_date + pd.Timedelta(days=h)).date() for h in range(1, HORIZONS + 1)]
    forecast = {d: {} for d in days}
    metrics = []

    for target, label in TARGETS.items():
        if df[target].notna().sum() < MIN_ROWS:
            continue  # ข้อมูลไม่พอสำหรับตัวแปรนี้
        for h in range(1, HORIZONS + 1):
            data = X.assign(y=df[target].shift(-h)).dropna(subset=["y"])
            split = int(len(data) * 0.8)  # แบ่งตามลำดับเวลา ไม่สลับข้อมูล
            train, test = data.iloc[:split], data.iloc[split:]
            cols = [c for c in data.columns if c != "y"]

            model = _new_model().fit(train[cols], train["y"])
            mae = mean_absolute_error(test["y"], model.predict(test[cols]))
            base = mean_absolute_error(test["y"], test[f"{target}_lag0"].fillna(train["y"].mean()))
            metrics.append(
                {"ตัวแปร": label, "ล่วงหน้า (วัน)": h, "MAE โมเดล": mae, "MAE ค่าอ้างอิง": base}
            )

            final = _new_model().fit(data[cols], data["y"])  # ฝึกซ้ำด้วยข้อมูลทั้งหมด
            pred = float(final.predict(x_latest[cols])[0])
            if target in ("precipitation_sum", PM25):
                pred = max(pred, 0.0)
            forecast[days[h - 1]][label] = pred
            forecast[days[h - 1]][f"{label} ±MAE"] = mae

    fc = pd.DataFrame(forecast).T
    fc.index.name = "วันที่"
    if RAIN_LABEL in fc:
        fc[RAIN_CUM_LABEL] = fc[RAIN_LABEL].cumsum()
    return fc, pd.DataFrame(metrics), last_date


def rainfall_summary(df: pd.DataFrame, last_date: pd.Timestamp) -> dict:
    """ฝนสะสมจริง: 7 วัน, 30 วัน, ตั้งแต่ต้นเดือน (เทียบค่าเฉลี่ยช่วงเดียวกันของปีก่อน ๆ) และตั้งแต่ต้นปี"""
    rain = df["precipitation_sum"]
    month_start = last_date.replace(day=1)
    offset = pd.Timedelta(days=last_date.day - 1)

    prev = []
    for year in sorted(set(rain.index.year) - {last_date.year}):
        s = pd.Timestamp(year, last_date.month, 1)
        window = rain[s : s + offset]
        if len(window) == last_date.day and window.notna().all():
            prev.append(window.sum())

    return {
        "last7": rain.tail(7).sum(),
        "last30": rain.tail(30).sum(),
        "mtd": rain[rain.index >= month_start].sum(),
        "mtd_avg": float(np.mean(prev)) if prev else None,
        "ytd": rain[rain.index >= pd.Timestamp(last_date.year, 1, 1)].sum(),
    }


@st.cache_data(ttl=3600, show_spinner="กำลังดึงข้อมูลและฝึกโมเดล...")
def run_pipeline(lat: float, lon: float, years: int):
    df = load_data(lat, lon, years)
    fc, metrics, last_date = build_forecast(df)
    return df, fc, metrics, last_date, rainfall_summary(df, last_date)


# ---------------------------------------------------------------- หน้าจอ
def main():
    st.set_page_config(page_title="พยากรณ์อากาศ จ.พระนครศรีอยุธยา", page_icon="🌦️", layout="wide")
    st.title("🌦️ พยากรณ์สภาพอากาศ จังหวัดพระนครศรีอยุธยา ล่วงหน้า 7 วัน")
    st.caption("อุณหภูมิ | ปริมาณฝนและฝนสะสม | PM2.5 — พยากรณ์ด้วยการเรียนรู้ของเครื่อง")

    with st.sidebar:
        st.header("ตั้งค่า")
        place = st.selectbox("พื้นที่", list(LOCATIONS))
        if LOCATIONS[place] is None:
            lat = st.number_input("ละติจูด", -90.0, 90.0, 14.3532, format="%.4f")
            lon = st.number_input("ลองจิจูด", -180.0, 180.0, 100.5689, format="%.4f")
        else:
            lat, lon = LOCATIONS[place]
            st.caption(f"พิกัด: {lat:.4f}, {lon:.4f}")
        years = st.slider("ข้อมูลฝึกโมเดลย้อนหลัง (ปี)", 3, 10, 5)
        st.caption("ข้อมูล PM2.5 มีย้อนหลังได้ตั้งแต่ ส.ค. 2565 เท่านั้น")

    try:
        df, fc, metrics, last_date, rain = run_pipeline(lat, lon, years)
    except requests.RequestException as exc:
        st.error(f"ไม่สามารถดึงข้อมูลจาก Open-Meteo ได้: {exc}")
        st.stop()

    st.caption(f"ข้อมูลล่าสุดที่ใช้: {last_date:%d/%m/%Y} | จำนวนข้อมูลฝึก: {len(df):,} วัน")
    has_pm = PM_LABEL in fc

    # ---- สรุปพรุ่งนี้
    st.subheader("สรุปวันพรุ่งนี้")
    first = fc.iloc[0]
    cols = st.columns(4)
    cols[0].metric("อุณหภูมิสูงสุด", f"{first[TARGETS['temperature_2m_max']]:.1f} °C")
    cols[1].metric("อุณหภูมิต่ำสุด", f"{first[TARGETS['temperature_2m_min']]:.1f} °C")
    cols[2].metric("ปริมาณฝน", f"{first[RAIN_LABEL]:.1f} มม.")
    if has_pm:
        cols[3].metric("PM2.5", f"{first[PM_LABEL]:.1f} µg/m³", pm25_category(first[PM_LABEL]), delta_color="off")
    else:
        cols[3].metric("PM2.5", "ไม่มีข้อมูล")

    # ---- ตารางพยากรณ์
    st.subheader("ผลการพยากรณ์ 7 วัน")
    show = fc.copy()
    if has_pm:
        show["ระดับ PM2.5"] = show[PM_LABEL].map(pm25_category)
    st.dataframe(show.round(1))

    # ---- กราฟ
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**อุณหภูมิ (°C)**")
        st.line_chart(fc[[TARGETS["temperature_2m_max"], TARGETS["temperature_2m_min"]]])
    with c2:
        st.markdown("**ปริมาณฝนรายวัน (มม.)**")
        st.bar_chart(fc[[RAIN_LABEL]])

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**ฝนสะสมพยากรณ์ (มม.)**")
        st.line_chart(fc[[RAIN_CUM_LABEL]])
    with c4:
        st.markdown(f"**PM2.5 (µg/m³)** — เส้นอ้างอิง {PM25_LIMIT} µg/m³")
        if has_pm:
            chart = fc[[PM_LABEL]].copy()
            chart[f"เกณฑ์ {PM25_LIMIT}"] = PM25_LIMIT
            st.line_chart(chart)
        else:
            st.warning("ไม่สามารถดึงข้อมูล PM2.5 ได้ในขณะนี้")

    # ---- ฝนสะสมจริง
    st.subheader("ปริมาณฝนสะสมจากข้อมูลจริง")
    r = st.columns(4)
    r[0].metric("7 วันที่ผ่านมา", f"{rain['last7']:.1f} มม.")
    r[1].metric("30 วันที่ผ่านมา", f"{rain['last30']:.1f} มม.")
    if rain["mtd_avg"] is not None:
        r[2].metric(
            "ตั้งแต่ต้นเดือน",
            f"{rain['mtd']:.1f} มม.",
            f"{rain['mtd'] - rain['mtd_avg']:+.1f} มม. จากค่าเฉลี่ยช่วงเดียวกัน",
            delta_color="off",
        )
    else:
        r[2].metric("ตั้งแต่ต้นเดือน", f"{rain['mtd']:.1f} มม.")
    r[3].metric("ตั้งแต่ต้นปี", f"{rain['ytd']:.1f} มม.")

    st.subheader("ข้อมูลจริงย้อนหลัง 30 วัน")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**ปริมาณฝนรายวัน (มม.)**")
        st.bar_chart(df[["precipitation_sum"]].tail(30))
    with h2:
        st.markdown("**PM2.5 เฉลี่ยรายวัน (µg/m³)**")
        st.line_chart(df[[PM25]].tail(30))

    with st.expander("ความแม่นยำของโมเดล (ชุดทดสอบ 20% ล่าสุดของข้อมูล)"):
        m = metrics.copy()
        m["ดีกว่าค่าอ้างอิง (%)"] = (1 - m["MAE โมเดล"] / m["MAE ค่าอ้างอิง"]) * 100
        st.dataframe(m.round(2), hide_index=True)
        st.caption(
            "MAE = ค่าคลาดเคลื่อนสัมบูรณ์เฉลี่ย | ค่าอ้างอิง = สมมติว่าวันข้างหน้าเหมือนวันล่าสุด (persistence)"
        )

    st.info(
        "ข้อควรทราบ: (1) PM2.5 จาก Open-Meteo เป็นค่าจากแบบจำลองคุณภาพอากาศ (CAMS) "
        "ไม่ใช่ค่าที่วัดจากสถานีตรวจวัดภาคพื้นดิน จึงอาจแตกต่างจากข้อมูลของกรมควบคุมมลพิษ "
        "(2) แอปนี้เป็นโมเดลทางสถิติ ใช้เพื่อการศึกษาและประกอบการตัดสินใจเบื้องต้นเท่านั้น "
        "ไม่ทดแทนการพยากรณ์อย่างเป็นทางการของกรมอุตุนิยมวิทยา"
    )


if __name__ == "__main__":
    main()
