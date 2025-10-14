# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Calibrado con Supresión (1−Ciec)
# Gráfico 1: EMERREL + Densidad equivalente semanal (A2=250)
# ===============================================================

import io, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="PREDWEEM · Calibrado con Supresión", layout="wide")
st.title("🌾 PREDWEEM — EMERREL + Densidad equivalente semanal (1−Ciec)")

# ---------- PARÁMETROS CALIBRADOS ----------
ALPHA = 0.503
LMAX  = 125.91
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}
A2_CAP = 250.0  # tope de densidad acumulada (pl·m²)

def loss_fun(x):
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)  # pérdida ≤ densidad efectiva

# ---------- CARGA DE DATOS ----------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("CSV o Excel (fecha, EMERREL)", type=["csv", "xlsx", "xls"])
dayfirst = st.sidebar.checkbox("Fecha dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores EMERREL en %", value=True)
is_cum = st.sidebar.checkbox("Serie acumulada (EMERAC)", value=False)

if up is None:
    st.info("Subí un archivo para comenzar.")
    st.stop()

# lectura flexible CSV / Excel
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
c_valor = next((c for c in df0.columns if "emer" in c), df0.columns[1] if len(df0.columns)>1 else df0.columns[0])

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
st.success(f"✅ Archivo leído correctamente ({len(df_plot)} días)")

# ---------- PARÁMETROS AGRONÓMICOS ----------
year_ref = int(df_plot["fecha"].dt.year.mode().iloc[0])
sow_default = dt.date(year_ref, 7, 1)
sow_date = st.sidebar.date_input("📅 Fecha de siembra", value=sow_default)

# controles (eficiencia y duración)
st.sidebar.header("🧪 Controles herbicidas")
ef_preR   = st.sidebar.slider("Presiembra residual (%)", 0, 100, 70, 1) / 100
ef_preemR = st.sidebar.slider("Preemergente residual (%)", 0, 100, 70, 1) / 100
ef_postR  = st.sidebar.slider("Post residual (%)", 0, 100, 60, 1) / 100
ef_gram   = st.sidebar.slider("Graminicida post (%)", 0, 100, 60, 1) / 100

preR_days   = st.sidebar.slider("Duración Presiembra (días)", 10, 90, 45, 1)
preemR_days = st.sidebar.slider("Duración Preemergente (días)", 10, 90, 45, 1)
postR_days  = st.sidebar.slider("Duración Post residual (días)", 10, 90, 45, 1)
gram_days   = st.sidebar.slider("Duración Graminicida (días)", 5, 30, 10, 1)

# desfases encadenados
st.sidebar.header("⏱️ Desfases biológicos")
lag12 = st.sidebar.slider("Δ(S₁→S₂)", 5, 30, 10, 1)
lag23 = st.sidebar.slider("Δ(S₂→S₃)", 5, 30, 15, 1)
lag34 = st.sidebar.slider("Δ(S₃→S₄)", 5, 30, 20, 1)

# ---------- SUPRESIÓN DEL CULTIVO ----------
def ciec_curve(dates, sow_date):
    """Curva logística simulada de supresión del cultivo (0–1)."""
    sow = pd.to_datetime(sow_date)
    t = (pd.to_datetime(dates) - sow).dt.days.astype(float)
    t = np.maximum(t, 0)
    g = 1 / (1 + np.exp(-0.12 * (t - 45)))  # crecimiento de cobertura
    return np.clip(g, 0, 1)

df_plot["Ciec"] = ciec_curve(df_plot["fecha"], sow_date)
df_plot["EMERREL_SUP"] = df_plot["EMERREL"] * (1 - df_plot["Ciec"])

# ---------- VENTANAS DE APLICACIÓN ----------
dates_d = df_plot["fecha"].dt.date.values
def window_mask(dates, start, dur):
    end = start + timedelta(days=int(dur))
    return ((dates >= start) & (dates < end)).astype(float)

preR_date   = sow_date - timedelta(days=20)
preemR_date = sow_date + timedelta(days=5)
postR_date  = sow_date + timedelta(days=25)
gram_date   = sow_date + timedelta(days=5)

m_preR   = window_mask(dates_d, preR_date, preR_days)
m_preemR = window_mask(dates_d, preemR_date, preemR_days)
m_postR  = window_mask(dates_d, postR_date, postR_days)
m_gram   = window_mask(dates_d, gram_date, gram_days)

# ---------- CÁLCULO ENCADENADO (S1–S4 sobre EMERREL_SUP) ----------
emer_sup = df_plot["EMERREL_SUP"].values
n = len(emer_sup)

S1 = np.where(dates_d >= sow_date, emer_sup, 0.0)
S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)

for t in range(n):
    if t >= lag12:                S2[t] = S1[t - lag12]
    if t >= lag12 + lag23:        S3[t] = S2[t - lag23]
    if t >= lag12 + lag23 + lag34:S4[t] = S3[t - lag34]

# controles multiplicativos
surv_S1 = (1 - ef_preR*m_preR) * (1 - ef_preemR*m_preemR) * (1 - ef_postR*m_postR) * (1 - ef_gram*m_gram)
surv_S2 = (1 - ef_preR*m_preR) * (1 - ef_preemR*m_preemR) * (1 - ef_postR*m_postR)
surv_S3 = (1 - ef_postR*m_postR) * (1 - ef_gram*m_gram)
surv_S4 = (1 - ef_postR*m_postR)

S1c = S1 * surv_S1
S2c = S2 * surv_S2
S3c = S3 * surv_S3
S4c = S4 * surv_S4

# Densidad efectiva diaria (ponderada por pesos)
eq_daily = (W_S["S1"]*S1c + W_S["S2"]*S2c + W_S["S3"]*S3c + W_S["S4"]*S4c)

# ---------- TOPE A2 ACUMULADO ----------
eq_cap = np.zeros(n)
acc = 0.0
for i in range(n):
    if dates_d[i] >= sow_date:
        rem = max(0, A2_CAP - acc)
        take = min(eq_daily[i], rem)
        eq_cap[i] = take
        acc += take
    else:
        eq_cap[i] = 0

df_plot["EQ_CAP"] = eq_cap

# ---------- AGREGACIÓN SEMANAL (W-MON) ----------
df_week = df_plot.resample("W-MON", on="fecha").sum().reset_index()

# ---------- GRÁFICO PRINCIPAL ----------
st.subheader("📈 EMERREL + EMERREL×(1−Ciec) + Densidad semanal (A₂=250)")
fig = go.Figure()

# EMERREL base
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["EMERREL"],
                         mode="lines", name="EMERREL", line=dict(color="gray")))

# EMERREL suprimida
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["EMERREL_SUP"],
                         mode="lines", name="EMERREL×(1−Ciec)",
                         line=dict(color="rgba(60,60,60,0.45)", width=3)))

# Densidad equivalente semanal (pl·m²·sem⁻¹)
fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["EQ_CAP"],
                         mode="lines+markers", name="Densidad equivalente semanal (cap)",
                         yaxis="y2", line=dict(color="#0072B2", width=2)))

fig.add_hline(y=A2_CAP, line=dict(color="black", dash="dot"),
              annotation_text="A₂ = 250 pl·m²", annotation_position="top right")

fig.update_layout(
    xaxis_title="Tiempo",
    yaxis_title="EMERREL",
    yaxis2=dict(title="Plantas·m²·semana⁻¹", overlaying="y", side="right", range=[0,100]),
    legend_title="Series",
    height=520
)
st.plotly_chart(fig, use_container_width=True)

# ---------- PÉRDIDA ----------
densidad_efectiva = df_plot["EQ_CAP"].sum()
perdida = loss_fun(densidad_efectiva)
st.markdown(f"🌾 **Densidad efectiva (cap):** {densidad_efectiva:.2f} pl·m² · **Pérdida estimada:** {perdida:.2f}%")

# Curva de pérdida
x_curve = np.linspace(0, max(A2_CAP, densidad_efectiva, 1.0), 400)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida limitada"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines", name="y=x", line=dict(dash="dot")))
fig_loss.add_trace(go.Scatter(x=[densidad_efectiva], y=[perdida],
                              mode="markers+text", text=[f"{perdida:.2f}%"],
                              textposition="top center"))
fig_loss.update_layout(xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
st.plotly_chart(fig_loss, use_container_width=True)

# ---------- EXPORTACIÓN ----------
csv = df_week[["fecha","EQ_CAP"]].to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar densidad equivalente semanal (CSV)", csv,
                   "densidad_equivalente_semanal.csv", "text/csv")

