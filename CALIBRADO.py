# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Calibrado Encadenado (S₁→S₂→S₃→S₄)
# Mantiene estructura original, solo cambia la lógica de estados
# ===============================================================

import io, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="PREDWEEM · Calibrado Encadenado", layout="wide")
st.title("🌱 PREDWEEM — Calibrado Encadenado (S₁→S₂→S₃→S₄)")

# ---------- PARÁMETROS FIJOS ----------
ALPHA = 0.503
LMAX  = 125.91
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

# tasas de transición encadenadas (fijas)
P12 = 1/10
P23 = 1/15
P34 = 1/20

# ---------- FUNCIÓN DE PÉRDIDA ----------
def loss_fun(x):
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)  # pérdida ≤ densidad efectiva

# ---------- CARGA DE DATOS ----------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("Archivo CSV o Excel (columnas: fecha, EMERREL / EMERAC)", type=["csv", "xlsx", "xls"])
dayfirst = st.sidebar.checkbox("Fecha dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores en % (no 0–1)", value=True)
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

preR_date = sow_date - timedelta(days=20)
preemR_date = sow_date + timedelta(days=5)
postR_date = sow_date + timedelta(days=25)
gram_date = sow_date + timedelta(days=5)

def mask_window(dates, start, dur):
    end = start + timedelta(days=dur)
    return ((dates >= start) & (dates < end)).astype(float)

dates_d = df_plot["fecha"].dt.date.values
mask_preR  = mask_window(dates_d, preR_date, 45)
mask_preemR = mask_window(dates_d, preemR_date, 45)
mask_postR  = mask_window(dates_d, postR_date, 45)
mask_gram   = mask_window(dates_d, gram_date, 10)

# ---------- MODELO ENCADENADO (reemplaza al anterior) ----------
emerrel_series = df_plot["EMERREL"]
n = len(emerrel_series)
S1 = np.zeros(n)
S2 = np.zeros(n)
S3 = np.zeros(n)
S4 = np.zeros(n)

# fase inicial
for t in range(n):
    if dates_d[t] >= sow_date:
        S1[t] = emerrel_series.iloc[t]
    else:
        S1[t] = 0.0

# propagación encadenada
for t in range(1, n):
    surv_preR   = 1 - preR_eff * mask_preR[t]
    surv_preemR = 1 - preemR_eff * mask_preemR[t]
    surv_postR  = 1 - postR_eff * mask_postR[t]
    surv_grami  = 1 - gram_eff * mask_gram[t]

    S2[t] = S2[t-1] + P12 * (S1[t-1] * surv_preR * surv_preemR)
    S3[t] = S3[t-1] + P23 * (S2[t-1] * surv_preemR)
    S4[t] = S4[t-1] + P34 * (S3[t-1] * surv_postR)
    S4[t] *= surv_grami

S1_arr, S2_arr, S3_arr, S4_arr = map(pd.Series, (S1, S2, S3, S4))
df_plot["S1"], df_plot["S2"], df_plot["S3"], df_plot["S4"] = S1_arr, S2_arr, S3_arr, S4_arr
df_plot["EFF"] = (W_S["S1"]*S1_arr + W_S["S2"]*S2_arr +
                  W_S["S3"]*S3_arr + W_S["S4"]*S4_arr)

# ---------- GRÁFICO PRINCIPAL ----------
st.subheader("📊 Dinámica temporal (S₁–S₄)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["S1"], name="S₁ Emergencia"))
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["S2"], name="S₂ Joven"))
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["S3"], name="S₃ Intermedia"))
fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["S4"], name="S₄ Adulta"))
fig.update_layout(xaxis_title="Fecha", yaxis_title="Plantas·m²·día⁻¹", legend_title="Estados")
st.plotly_chart(fig, use_container_width=True)

# ---------- PÉRDIDA ----------
densidad_efectiva = df_plot["EFF"].sum()
perdida = loss_fun(densidad_efectiva)
st.markdown(f"🌾 **Densidad efectiva:** {densidad_efectiva:.2f} pl·m²  —  **Pérdida estimada:** {perdida:.2f}%")

x_curve = np.linspace(0, max(densidad_efectiva,1e-6), 300)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida limitada"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines", line=dict(dash="dot", color="gray"), name="y=x"))
fig_loss.add_trace(go.Scatter(x=[densidad_efectiva], y=[perdida], mode="markers+text", text=[f"{perdida:.2f}%"], textposition="top center"))
fig_loss.update_layout(xaxis_title="Densidad efectiva (pl·m²)", yaxis_title="Pérdida de rendimiento (%)")
st.plotly_chart(fig_loss, use_container_width=True)

# ---------- EXPORTACIÓN ----------
csv = df_plot.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar resultados CSV", csv, "calibrado_encadenado.csv", "text/csv")
