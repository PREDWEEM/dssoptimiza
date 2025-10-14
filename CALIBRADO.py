# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Modelo encadenado de edades sucesivas S1–S4
# ===============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="PREDWEEM · Encadenado S1–S4", layout="wide")
st.title("🌱 PREDWEEM — Modelo encadenado de edades sucesivas (S1–S4)")

# === Carga de datos ===
up = st.file_uploader("📂 Subí tu CSV con columnas (fecha, EMERREL)", type=["csv"])
if up is None:
    st.stop()

df = pd.read_csv(up)
df.columns = [c.strip().lower() for c in df.columns]
df.rename(columns={"fecha": "date", "emerrel": "emerg"}, inplace=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna().sort_values("date")
s = pd.Series(df["emerg"].values, index=df["date"])

# === Parámetros del modelo ===
st.sidebar.header("⚙️ Parámetros de simulación")

lags = {
    "S2": st.sidebar.number_input("Desfase S1→S2 (días)", 5, 30, 10),
    "S3": st.sidebar.number_input("Desfase S2→S3 (días)", 5, 30, 10),
    "S4": st.sidebar.number_input("Desfase S3→S4 (días)", 5, 30, 10)
}

controls = {
    "S1": st.sidebar.slider("Eficiencia control S1", 0.0, 1.0, 0.5, 0.05),
    "S2": st.sidebar.slider("Eficiencia control S2", 0.0, 1.0, 0.3, 0.05),
    "S3": st.sidebar.slider("Eficiencia control S3", 0.0, 1.0, 0.2, 0.05)
}

# === Simulación encadenada ===
dates = s.index
S1 = s.copy()

S2 = S1.shift(lags["S2"]).fillna(0) * (1 - controls["S1"])
S3 = S2.shift(lags["S3"]).fillna(0) * (1 - controls["S2"])
S4 = S3.shift(lags["S4"]).fillna(0) * (1 - controls["S3"])

df_states = pd.DataFrame({
    "Fecha": dates,
    "S1": S1,
    "S2": S2,
    "S3": S3,
    "S4": S4
})

# === Visualización ===
fig = go.Figure()
fig.add_trace(go.Bar(x=dates, y=S1, name="S1 — Emergencia"))
fig.add_trace(go.Bar(x=dates, y=S2, name="S2 — Joven"))
fig.add_trace(go.Bar(x=dates, y=S3, name="S3 — Intermedia"))
fig.add_trace(go.Bar(x=dates, y=S4, name="S4 — Adulta"))

fig.update_layout(
    barmode="stack",
    title="Dinámica encadenada de cohortes (S1–S4)",
    xaxis_title="Fecha",
    yaxis_title="Plantas·m²·día⁻¹",
    legend_title="Estados sucesivos",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# === Diagnóstico ===
total = df_states[["S1","S2","S3","S4"]].sum()
st.subheader("Resumen")
st.write("Total de plantas por estado (ajustado por controles):")
st.dataframe(total)

# === Exportación ===
csv = df_states.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar resultados CSV", csv, "cohortes_encadenadas.csv", "text/csv")

