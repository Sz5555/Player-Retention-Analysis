import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def simulate_dau_series(player_df, days=30, anomaly_dates=None):
    """
    Simulate a 30-day DAU time series with realistic patterns.
    Injects artificial anomalies on specified dates.
    Returns dataframe with: date, dau, is_anomaly, z_score, severity
    """
    if anomaly_dates is None:
        anomaly_dates = {}

    base_dau = 350
    trend = np.linspace(0, -30, days)

    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    dau_values = []
    for i in range(days):
        val = base_dau + trend[i]
        if i % 7 >= 5:
            val += 40
        val += np.random.normal(0, 15)
        val = max(0, int(val))
        dau_values.append(val)

    for day_idx, multiplier in anomaly_dates.items():
        if 0 <= day_idx < days:
            dau_values[day_idx] = int(dau_values[day_idx] * multiplier)

    df = pd.DataFrame({"date": dates, "dau": dau_values})

    window = 7
    df["rolling_mean"] = df["dau"].rolling(window=window, center=True).mean()
    df["rolling_std"] = df["dau"].rolling(window=window, center=True).std()
    df["rolling_std"] = df["rolling_std"].fillna(df["rolling_std"].mean())

    df["z_score"] = (df["dau"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_score"] = df["z_score"].fillna(0)

    def classify(z):
        if abs(z) < 2:
            return "normal"
        elif abs(z) < 3:
            return "warning"
        else:
            return "critical"

    df["severity"] = df["z_score"].apply(classify)
    df["is_anomaly"] = df["severity"] != "normal"

    return df


def get_anomaly_summary(df):
    """Return only the detected anomaly rows with interpretation."""
    anomalies = df[df["is_anomaly"]].copy()
    if anomalies.empty:
        return anomalies

    anomalies["direction"] = anomalies["z_score"].apply(
        lambda z: "spike_up" if z > 0 else "drop"
    )
    anomalies["interpretation"] = anomalies.apply(
        lambda row: (
            f"DAU 突然下降 {abs(row['z_score']):.1f} 个标准差，可能原因：服务器故障、恶性 Bug、竞品上线"
            if row["direction"] == "drop"
            else f"DAU 异常上升 {row['z_score']:.1f} 个标准差，可能原因：营销活动、病毒传播、节假日效应"
        ),
        axis=1,
    )
    return anomalies[["date", "dau", "z_score", "severity", "direction", "interpretation"]]
