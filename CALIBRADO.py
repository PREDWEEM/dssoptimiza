# -*- coding: utf-8 -*-
# PREDWEEM — Simulador calibrado (encadenado con límite biológico)
# ----------------------------------------------------------
# Características:
# - Estados S1–S4 encadenados (flujo poblacional diario)
# - Controles aplicados dentro de sus ventanas agronómicas
# - Pérdida hiperbólica limitada (pérdida ≤ x)
# - Timeline de tratamientos y gráfico de pérdida vs densidad
# ----------------------------------------------------------

import io, re, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ========= parámetros calibrados =========
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

# ========= función de pérdida corregida =========
def loss_fun(x):
    """
    Función de pérdida hiperbólica con límite biológico:
    - Nunca supera el valor de densidad efectiva (x)
    - Saturación controlada por LMAX
    - Devuelve pérdida (%) si x está en pl·m²
    """
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)  # asegura que pérdida ≤ x

# ========= estado UI =========
st.set_page_config(page_title="PREDWEEM · Encadenado + Límite biológico", layout="wide")
st.title("🌱 PREDWEEM · Simulador encadenado (S₁–S₄) con límite biológico de pérdida")

# ========= constantes =========
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14
PREEM_R_MAX_AFTER_SOW_DAYS = 10

# ========= lectura CSV/Excel =========
st.sidebar.header("Datos de entrada")
up = st.sidebar.file_uploader("Archivo CSV/Excel", type=["csv", "xlsx", "xls"])
if up is None:
    st.info("Subí un archivo de datos para comenzar.")
    st.stop()

if up.name.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(up)
else:
    df = pd.read_csv(up, sep=None, engine="python")

cols = list(df.columns)
c_fecha = st.sidebar.selectbox("Columna de fecha", cols, index=0)
c_valor = st.sidebar.selectbox("Columna EMERREL", cols, index=1 if len(cols) > 1 else 0)

df["fecha"] = pd.to_datetime(df[c_fecha], dayfirst=True, errors="coerce")
df["EMERREL"] = pd.to_numeric(df[c_valor], errors="coerce")
df = df.dropna(subset=["fecha", "EMERREL"]).sort_values("fecha").reset_index(drop=True)

st.success(f"Archivo leído correctamente: {len(df)} registros válidos.")

# ========= fecha de siembra =========
years = df["fecha"].dt.year
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min, sow_max = dt.date(year_ref, 5, 1), dt.date(year_ref, 8, 1)
sow_date = st.sidebar.date_input("📅 Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)

# ========= parámetros Ciec =========
Ca = st.sidebar.number_input("Ca (pl/m²)", 50, 700, 250, 10)
Cs = st.sidebar.number_input("Cs (pl/m²)", 50, 700, 250, 10)
LAIhc = st.sidebar.number_input("LAIhc", 0.5, 10.0, 3.5, 0.1)
Ciec = np.clip((Ca / Cs) * (LAIhc / LAIhc), 0, 1)
one_minus_Ciec = 1 - Ciec

# ========= tope A2 =========
MAX_PLANTS_CAP = st.sidebar.selectbox("Tope densidad efectiva (pl·m²)", [250, 125, 62], index=0)
st.caption(f"AUC(EMERREL) ≙ A2={MAX_PLANTS_CAP} pl·m²")

# ========= helpers =========
def auc_time(f, y):
    t = (f - f.iloc[0]).dt.days.to_numpy()
    y = np.nan_to_num(y)
    return np.trapz(y, t)

def cap_cumulative(series, cap):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y)
    cum = 0.0
    for i, val in enumerate(y):
        if cum < cap:
            val = min(val, cap - cum)
            out[i] = val
            cum += val
        else:
            out[i] = 0.0
    return out

def weights_residual(start, days, fechas):
    if start is None or days <= 0: return np.zeros_like(fechas, float)
    mask = (fechas >= start) & (fechas < start + timedelta(days=days))
    return mask.astype(float)

# ========= simulación encadenada =========
def simulate_chained(df, sow_date, factor, MAX_CAP):
    ts = pd.to_datetime(df["fecha"])
    fechas = ts.dt.date.values
    ms = (fechas >= sow_date)
    births = df["EMERREL"].to_numpy(float)
    births = np.where(ms, births * factor, 0.0)
    births = cap_cumulative(births, MAX_CAP)
    births *= one_minus_Ciec

    n = len(births)
    S1b = np.zeros(n); S2b = np.zeros(n); S3b = np.zeros(n); S4b = np.zeros(n)
    S1c = np.zeros(n); S2c = np.zeros(n); S3c = np.zeros(n); S4c = np.zeros(n)

    # Controles de ejemplo (podés parametrizarlos)
    preR_date = sow_date - timedelta(days=20)
    preemR_date = sow_date + timedelta(days=5)
    postR_date = sow_date + timedelta(days=25)
    gram_date = sow_date + timedelta(days=5)
    preR_eff, preemR_eff, postR_eff, gram_eff = 0.7, 0.7, 0.7, 0.6

    w_preR = weights_residual(preR_date, 45, fechas)
    w_preemR = weights_residual(preemR_date, 45, fechas)
    w_postR = weights_residual(postR_date, 45, fechas)
    w_gram = weights_residual(gram_date, 10, fechas)

    D1, D2, D3 = 6, 21, 32
    for t in range(n):
        # baseline
        inflow = births[t]
        tr12b = S1b[t-1]/D1 if t>0 else 0
        tr23b = S2b[t-1]/D2 if t>0 else 0
        tr34b = S3b[t-1]/D3 if t>0 else 0
        S1b[t] = (S1b[t-1] if t>0 else 0) + inflow - tr12b
        S2b[t] = (S2b[t-1] if t>0 else 0) + tr12b - tr23b
        S3b[t] = (S3b[t-1] if t>0 else 0) + tr23b - tr34b
        S4b[t] = (S4b[t-1] if t>0 else 0) + tr34b

        # control
        red1 = (1 - preR_eff*w_preR[t])*(1 - preemR_eff*w_preemR[t])*(1 - postR_eff*w_postR[t])*(1 - gram_eff*w_gram[t])
        red2 = (1 - preR_eff*w_preR[t])*(1 - preemR_eff*w_preemR[t])*(1 - postR_eff*w_postR[t])*(1 - gram_eff*w_gram[t])
        red3 = (1 - postR_eff*w_postR[t])*(1 - gram_eff*w_gram[t])
        red4 = (1 - postR_eff*w_postR[t])
        inflow_c = births[t]
        S1c[t] = (S1c[t-1] if t>0 else 0)*red1 + inflow_c - ((S1c[t-1]*red1)/D1 if t>0 else 0)
        S2c[t] = (S2c[t-1] if t>0 else 0)*red2 + ((S1c[t-1]*red1)/D1 if t>0 else 0) - ((S2c[t-1]*red2)/D2 if t>0 else 0)
        S3c[t] = (S3c[t-1] if t>0 else 0)*red3 + ((S2c[t-1]*red2)/D2 if t>0 else 0) - ((S3c[t-1]*red3)/D3 if t>0 else 0)
        S4c[t] = (S4c[t-1] if t>0 else 0)*red4 + ((S3c[t-1]*red3)/D3 if t>0 else 0)

    eff_b = S1b*W_S["S1"] + S2b*W_S["S2"] + S3b*W_S["S3"] + S4b*W_S["S4"]
    eff_c = S1c*W_S["S1"] + S2c*W_S["S2"] + S3c*W_S["S3"] + S4c*W_S["S4"]

    X2 = eff_b[ms].sum(); X3 = eff_c[ms].sum()
    AUC = auc_time(df["fecha"], df["EMERREL"])
    factor_area = MAX_CAP / AUC if AUC > 0 else 1.0
    A2_sup = min(MAX_CAP, MAX_CAP*(auc_time(df["fecha"], eff_b/factor_area)/AUC))
    A2_ctrl = min(MAX_CAP, MAX_CAP*(auc_time(df["fecha"], eff_c/factor_area)/AUC))

    return ts, eff_b, eff_c, X2, X3, A2_sup, A2_ctrl

# ========= ejecutar simulación =========
AUC = auc_time(df["fecha"], df["EMERREL"])
factor = MAX_PLANTS_CAP / AUC if AUC > 0 else 1.0
ts, eff_b, eff_c, X2, X3, A2_sup, A2_ctrl = simulate_chained(df, sow_date, factor, MAX_PLANTS_CAP)

# ========= gráfico principal =========
st.subheader("📊 Dinámica temporal (EMERREL + aportes)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["fecha"], y=df["EMERREL"], name="EMERREL cruda", line=dict(color="gray")))
fig.add_trace(go.Scatter(x=ts, y=eff_b, name="Efectivo sin control", line=dict(color="red")))
fig.add_trace(go.Scatter(x=ts, y=eff_c, name="Efectivo con control", line=dict(color="blue", dash="dot")))
fig.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL / pl·m²·día⁻¹")
st.plotly_chart(fig, use_container_width=True)

# ========= resumen numérico =========
st.markdown(f"""
**x₂ (sin control):** {X2:.2f} pl·m²  
**x₃ (con control):** {X3:.2f} pl·m²  
**A2_sup:** {A2_sup:.2f} pl·m² · **A2_ctrl:** {A2_ctrl:.2f} pl·m²  
**Pérdida(x₂):** {loss_fun(X2):.2f}% · **Pérdida(x₃):** {loss_fun(X3):.2f}%
""")

# ========= gráfico de pérdida =========
st.subheader("📉 Pérdida de rendimiento (%) vs densidad efectiva (x)")
x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
y_curve = np.clip(loss_fun(x_curve), 0, MAX_PLANTS_CAP)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida (hiperbólica)", line=dict(color="firebrick", width=2)))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines", name="Referencia y=x", line=dict(color="gray", dash="dot")))
fig_loss.add_trace(go.Scatter(x=[X2], y=[loss_fun(X2)], mode="markers+text", text=["x₂"], textposition="top center", name="Sin control"))
fig_loss.add_trace(go.Scatter(x=[X3], y=[loss_fun(X3)], mode="markers+text", text=["x₃"], textposition="top right", name="Con control"))
fig_loss.update_layout(xaxis_title="Densidad efectiva (pl·m²)", yaxis_title="Pérdida de rendimiento (%)",
                       title="Curva de pérdida (limitada biológicamente)")
st.plotly_chart(fig_loss, use_container_width=True)

# ========= parámetros calibrados =========
with st.expander("Parámetros calibrados"):
    st.markdown(f"- w₁..w₄ = {W_S['S1']}, {W_S['S2']}, {W_S['S3']}, {W_S['S4']}\n"
                f"- α = {ALPHA} · Lmax = {LMAX}\n"
                f"- Función: loss(x) = α·x / (1 + α·x/Lmax), limitada a x")

