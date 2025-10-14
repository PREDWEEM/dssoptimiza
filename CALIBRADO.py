
# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Calibrado con Supresión (1−Ciec)
# Modelo encadenado S₁→S₂→S₃→S₄ sobre EMERREL_SUP
# Representación semanal (W-MON)
# ===============================================================

import io, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="PREDWEEM · Calibrado con Supresión", layout="wide")
st.title("🌱 PREDWEEM — Calibrado con Supresión (1−Ciec) · pl·m²·sem⁻¹")

# ---------- PARÁMETROS FIJOS ----------
ALPHA = 0.503
LMAX  = 125.91
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}
A2_CAP = 250  # tope superior gráfico

# ---------- FUNCIÓN DE PÉRDIDA ----------
def loss_fun(x):
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)

# ---------- CARGA DE DATOS ----------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("Archivo CSV o Excel (fecha, EMERREL)", type=["csv", "xlsx", "xls"])
dayfirst = st.sidebar.checkbox("Fecha dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores en %", value=True)
is_cum = st.sidebar.checkbox("Serie acumulada (EMERAC)", value=False)

if up is None:
    st.info("Subí un archivo para comenzar.")
    st.stop()

if up.name.lower().endswith((".xlsx", ".xls")):
    df0 = pd.read_excel(up)
else:
    raw = up.read()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="replace")
    try:
        df0 = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
    except Exception:
        df0 = pd.read_csv(io.StringIO(txt), sep=";", engine="python")

df0.columns = [c.strip().lower() for c in df0.columns]
c_fecha = next((c for c in df0.columns if "fec" in c), df0.columns[0])
c_valor = next((c for c in df0.columns if "emer" in c), df0.columns[1] if len(df0.columns) > 1 else df0.columns[0])

df0["fecha"] = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
df0["valor"] = pd.to_numeric(df0[c_valor], errors="coerce")
df = df0.dropna().sort_values("fecha").reset_index(drop=True)

idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
serie = df.set_index("fecha")["valor"].reindex(idx)
if is_cum:
    serie = serie.diff().fillna(0).clip(lower=0)
if is_percent:
    serie = serie / 100.0
serie = serie.fillna(0)

df_plot = pd.DataFrame({"fecha": idx, "EMERREL": serie.values})
st.success(f"✅ Archivo leído correctamente: {len(df_plot)} días")

# ---------- PARÁMETROS AGRONÓMICOS ----------
sow_date = st.sidebar.date_input("📅 Fecha de siembra", value=dt.date(idx[0].year, 7, 1))
preR_eff = st.sidebar.slider("Eficiencia Presiembra (%)", 0, 100, 70, 5) / 100
preemR_eff = st.sidebar.slider("Eficiencia Preemergente (%)", 0, 100, 60, 5) / 100
postR_eff = st.sidebar.slider("Eficiencia Postemergente (%)", 0, 100, 50, 5) / 100
gram_eff = st.sidebar.slider("Eficiencia Graminicida (%)", 0, 100, 60, 5) / 100

lag_S1S2 = st.sidebar.slider("Desfase S₁→S₂ (días)", 5, 30, 10)
lag_S2S3 = st.sidebar.slider("Desfase S₂→S₃ (días)", 5, 30, 15)
lag_S3S4 = st.sidebar.slider("Desfase S₃→S₄ (días)", 5, 30, 20)

# ---------- FUNCIÓN SUPRESIÓN DEL CULTIVO ----------
def ciec_curve(dates, sow_date):
    # Supresión simulada simple tipo logística (placeholder, adaptable a LAI real)
    t = np.arange(len(dates))
    g = 1 / (1 + np.exp(-0.15 * (t - 45)))  # crecimiento de cobertura
    return np.clip(g, 0, 1)

df_plot["Ciec"] = ciec_curve(df_plot["fecha"], sow_date)

# ---------- EMERGENCIA AFECTADA POR SUPRESIÓN ----------
df_plot["EMERREL_SUP"] = df_plot["EMERREL"] * (1 - df_plot["Ciec"])

# ---------- APLICACIÓN DE CONTROLES ----------
def mask_window(dates, start, dur):
    end = start + timedelta(days=dur)
    return ((dates >= start) & (dates < end)).astype(float)

dates_d = df_plot["fecha"].dt.date.values
preR_date = sow_date - timedelta(days=20)
preemR_date = sow_date + timedelta(days=5)
postR_date = sow_date + timedelta(days=25)
gram_date = sow_date + timedelta(days=5)

mask_preR  = mask_window(dates_d, preR_date, 45)
mask_preemR = mask_window(dates_d, preemR_date, 45)
mask_postR  = mask_window(dates_d, postR_date, 45)
mask_gram   = mask_window(dates_d, gram_date, 10)

# ---------- MODELO ENCADENADO SOBRE EMERREL_SUP ----------
emer_sup = df_plot["EMERREL_SUP"].values
n = len(emer_sup)
S1 = np.zeros(n)
S2 = np.zeros(n)
S3 = np.zeros(n)
S4 = np.zeros(n)

for t in range(n):
    if dates_d[t] >= sow_date:
        S1[t] = emer_sup[t]
    else:
        S1[t] = 0.0

for t in range(n):
    S2[t] = emer_sup[t - lag_S1S2] if t >= lag_S1S2 else 0
    S3[t] = emer_sup[t - (lag_S1S2 + lag_S2S3)] if t >= (lag_S1S2 + lag_S2S3) else 0
    S4[t] = emer_sup[t - (lag_S1S2 + lag_S2S3 + lag_S3S4)] if t >= (lag_S1S2 + lag_S2S3 + lag_S3S4) else 0

# controles acumulados (actúan multiplicativamente)
for t in range(n):
    surv_preR   = 1 - preR_eff * mask_preR[t]
    surv_preemR = 1 - preemR_eff * mask_preemR[t]
    surv_postR  = 1 - postR_eff * mask_postR[t]
    surv_grami  = 1 - gram_eff * mask_gram[t]
    S2[t] *= surv_preR * surv_preemR
    S3[t] *= surv_preR * surv_preemR * surv_postR
    S4[t] *= surv_preR * surv_preemR * surv_postR * surv_grami

# ---------- COMPILACIÓN Y SALIDA SEMANAL ----------
df_plot["S1"] = S1
df_plot["S2"] = S2
df_plot["S3"] = S3
df_plot["S4"] = S4
df_plot["EFF"] = (W_S["S1"]*S1 + W_S["S2"]*S2 + W_S["S3"]*S3 + W_S["S4"]*S4)

df_week = df_plot.resample("W-MON", on="fecha").sum().reset_index()

# ---------- GRÁFICO PRINCIPAL (EMERREL_SUP + APORTES) ----------
st.subheader("📊 Figura 1 — EMERREL_SUP + aportes (S₁–S₄) · Serie semanal (W-MON)")
fig = go.Figure()

# EMERREL_SUP superpuesta al frente
fig.add_trace(go.Bar(
    x=df_week["fecha"], y=df_week["EMERREL_SUP"],
    name="EMERREL_SUP",
    marker_color="rgba(90,90,90,0.45)",
))

# Aportes de estados (curvas apiladas)
fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["S1"], name="S₁ Emergencia",
                         mode="lines", line=dict(width=2, color="#1b9e77")))
fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["S2"], name="S₂ Joven",
                         mode="lines", line=dict(width=2, color="#d95f02")))
fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["S3"], name="S₃ Intermedia",
                         mode="lines", line=dict(width=2, color="#7570b3")))
fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["S4"], name="S₄ Adulta",
                         mode="lines", line=dict(width=2, color="#e7298a")))

fig.add_hline(y=A2_CAP, line=dict(color="black", dash="dot"),
              annotation_text="A₂ = 250 pl·m²", annotation_position="top right")

fig.update_layout(
    xaxis_title="Semana (W-MON)",
    yaxis_title="Plantas·m²·semana⁻¹",
    legend_title="Series",
    height=540,
    barmode="overlay",
    bargap=0.2
)
st.plotly_chart(fig, use_container_width=True)

# ---------- PÉRDIDA ----------
densidad_efectiva = df_plot["EFF"].sum()
perdida = loss_fun(densidad_efectiva)
st.markdown(f"🌾 **Densidad efectiva:** {densidad_efectiva:.2f} pl·m²  —  **Pérdida estimada:** {perdida:.2f}%")

x_curve = np.linspace(0, max(densidad_efectiva, 1e-6), 300)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida limitada"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines",
                              line=dict(dash="dot", color="gray"), name="y = x"))
fig_loss.add_trace(go.Scatter(x=[densidad_efectiva], y=[perdida], mode="markers+text",
                              text=[f"{perdida:.2f}%"], textposition="top center"))
fig_loss.update_layout(xaxis_title="Densidad efectiva (pl·m²)",
                       yaxis_title="Pérdida de rendimiento (%)")
st.plotly_chart(fig_loss, use_container_width=True)

# ---------- EXPORTACIÓN ----------
csv = df_week.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar resultados CSV (semanal)", csv,
                   "calibrado_supresion_semana.csv", "text/csv")
