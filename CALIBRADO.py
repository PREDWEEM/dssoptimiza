# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Gráfico 1 estilo original
# Lógica encadenada S1→S2→S3→S4 (sobre EMERREL suprimida por (1−Ciec))
# Salida: UNA SOLA SERIE semanal en eje derecho (pl·m²·sem⁻¹)
# ===============================================================

import io, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ---------------- Config ----------------
st.set_page_config(page_title="PREDWEEM · Gráfico 1 (encadenado)", layout="wide")
st.title("📊 PREDWEEM — EMERREL + Densidad equivalente semanal (encadenado)")

# --- Parámetros calibrados ---
ALPHA = 0.503
LMAX  = 125.91
# pesos por estado (calibrados)
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}
A2_CAP = 250.0  # tope acumulado A2 (pl·m²)

def loss_fun(x):
    """Pérdida (%) limitada para no superar x (densidad efectiva)."""
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)

# ---------------- Lectura de datos ----------------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("CSV o Excel (recomendado: columnas 'fecha', 'EMERREL')", type=["csv","xlsx","xls"])
dayfirst = st.sidebar.checkbox("Fecha dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores EMERREL en %", value=True)
is_cum = st.sidebar.checkbox("Serie acumulada (EMERAC)", value=False)

if up is None:
    st.info("Subí un archivo para continuar.")
    st.stop()

# lectura robusta CSV/Excel
if up.name.lower().endswith((".xlsx",".xls")):
    df0 = pd.read_excel(up)
else:
    raw = up.read()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="replace")
    # intentar detectar separador
    try:
        df0 = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
    except Exception:
        df0 = pd.read_csv(io.StringIO(txt), sep=";", engine="python")

# normalizar columnas
df0.columns = [c.strip().lower() for c in df0.columns]
c_fecha = next((c for c in df0.columns if "fec" in c), df0.columns[0])
c_valor = next((c for c in df0.columns if "emer" in c), df0.columns[1] if len(df0.columns)>1 else df0.columns[0])

df0["fecha"] = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
df0["valor"] = pd.to_numeric(df0[c_valor], errors="coerce")
df = df0.dropna().sort_values("fecha").reset_index(drop=True)

if df.empty:
    st.error("No se pudieron parsear fechas/valores.")
    st.stop()

# índice diario continuo
idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
emer = df.set_index("fecha")["valor"].reindex(idx)

# EMERREL: si viene acumulada → diferenciar; si viene en % → llevar a 0–1
if is_cum:
    emer = emer.diff().fillna(0).clip(lower=0)
if is_percent:
    emer = emer / 100.0
emer = emer.fillna(0.0)

df_plot = pd.DataFrame({"fecha": idx, "EMERREL": emer.values})
st.success(f"✅ Serie diaria: {len(df_plot)} días")

# ---------------- Parámetros agronómicos ----------------
st.sidebar.header("🌱 Supresión del cultivo y ventanas")
# (Podés reemplazar por tu función/LAI real; acá dejamos un perfil logístico simple)
def ciec_curve(dates, sow_date):
    t = (dates - pd.to_datetime(sow_date)).days.values.astype(float)
    # antes de siembra Ciec = 0
    t = np.maximum(t, 0.0)
    g = 1.0 / (1.0 + np.exp(-0.12 * (t - 45)))  # cierre ~ 45 días
    return np.clip(g, 0, 1)

# siembra
year_ref = int(df_plot["fecha"].dt.year.mode().iloc[0])
sow_default = dt.date(year_ref, 7, 1)
sow_date = st.sidebar.date_input("📅 Fecha de siembra", value=sow_default)

# ventanas de control (ajustables)
preR_days   = st.sidebar.slider("Presiembra residual (días de protección)", 10, 120, 45, 1)
preemR_days = st.sidebar.slider("Preemergente residual (días de protección)", 10, 120, 45, 1)
postR_days  = st.sidebar.slider("Post residual (días de protección)", 10, 120, 45, 1)
gram_days   = st.sidebar.slider("Graminicida post (ventana +días)", 5, 20, 10, 1)

# eficiencias (0–100%)
st.sidebar.header("🧪 Eficiencias (%)")
ef_preR   = st.sidebar.slider("Presiembra residual", 0, 100, 70, 1) / 100.0
ef_preemR = st.sidebar.slider("Preemergente residual", 0, 100, 70, 1) / 100.0
ef_postR  = st.sidebar.slider("Post residual", 0, 100, 70, 1) / 100.0
ef_gram   = st.sidebar.slider("Graminicida post", 0, 100, 65, 1) / 100.0

# desfases biológicos encadenados (S1→S2→S3→S4)
st.sidebar.header("⏱️ Desfases S₁→S₂→S₃→S₄ (días)")
lag12 = st.sidebar.slider("Δ(S₁→S₂)", 5, 30, 10, 1)
lag23 = st.sidebar.slider("Δ(S₂→S₃)", 5, 30, 15, 1)
lag34 = st.sidebar.slider("Δ(S₃→S₄)", 5, 30, 20, 1)

# ---------------- EMERREL suprimida por el cultivo ----------------
df_plot["Ciec"] = ciec_curve(df_plot["fecha"], sow_date)
df_plot["EMERREL_SUP"] = df_plot["EMERREL"] * (1.0 - df_plot["Ciec"])

# ---------------- Ventanas de aplicación ----------------
dates_d = df_plot["fecha"].dt.date.values

def window_mask(dates_d, start_date, days):
    if start_date is None or days is None or days <= 0:
        return np.zeros_like(dates_d, dtype=float)
    d0 = start_date
    d1 = start_date + timedelta(days=int(days))
    return ((dates_d >= d0) & (dates_d < d1)).astype(float)

# reglas (pueden ajustarse en tu versión final)
preR_date   = sow_date - timedelta(days=20)       # debe ser ≤ siembra−14 (regla dura)
preemR_date = sow_date + timedelta(days=5)        # en [siembra..siembra+10]
postR_date  = sow_date + timedelta(days=25)       # ≥ siembra+20
gram_date   = sow_date + timedelta(days=5)        # ventana 0..+10

m_preR   = window_mask(dates_d, preR_date, preR_days)
m_preemR = window_mask(dates_d, preemR_date, preemR_days)
m_postR  = window_mask(dates_d, postR_date, postR_days)
m_gram   = window_mask(dates_d, gram_date, gram_days)

# ---------------- Encadenado S1→S2→S3→S4 sobre EMERREL_SUP ----------------
emer_sup = df_plot["EMERREL_SUP"].values
n = len(emer_sup)

# S1: nacimientos desde siembra
S1 = np.where(dates_d >= sow_date, emer_sup, 0.0)

# S2..S4: llegan por desfase desde S1..S3 respectivamente
S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
for t in range(n):
    if t >= lag12:                S2[t] = S1[t - lag12]
    if t >= lag12 + lag23:        S3[t] = S2[t - lag23]
    if t >= lag12 + lag23 + lag34:S4[t] = S3[t - lag34]

# Controles por ventanas (supervivencias multiplicativas) por estado:
# presiembra/preemergente → S1–S2; post residual → S1–S4; graminicida → S1–S3
surv_S1 = (1 - ef_preR*m_preR) * (1 - ef_preemR*m_preemR) * (1 - ef_postR*m_postR) * (1 - ef_gram*m_gram)
surv_S2 = (1 - ef_preR*m_preR) * (1 - ef_preemR*m_preemR) * (1 - ef_postR*m_postR)
surv_S3 = (1 - ef_postR*m_postR) * (1 - ef_gram*m_gram)
surv_S4 = (1 - ef_postR*m_postR)

S1c = S1 * surv_S1
S2c = S2 * surv_S2
S3c = S3 * surv_S3
S4c = S4 * surv_S4

# Densidad equivalente diaria (suma ponderada por pesos calibrados)
eq_daily = (W_S["S1"]*S1c + W_S["S2"]*S2c + W_S["S3"]*S3c + W_S["S4"]*S4c)

# ---------------- Tope A2 acumulado (cap) ----------------
eq_daily_cap = np.zeros(n, dtype=float)
acc = 0.0
for i in range(n):
    if dates_d[i] >= sow_date:
        rem = max(0.0, A2_CAP - acc)
        take = min(eq_daily[i], rem)
        eq_daily_cap[i] = take
        acc += take
    else:
        eq_daily_cap[i] = 0.0

# ---------------- Agregación semanal (W-MON) ----------------
df_daily = pd.DataFrame({
    "fecha": df_plot["fecha"],
    "EMERREL": df_plot["EMERREL"],
    "EMERREL_SUP": df_plot["EMERREL_SUP"],
    "EQ_CAP": eq_daily_cap
})
df_week = df_daily.set_index("fecha").resample("W-MON").sum().reset_index()

# ---------------- Gráfico 1 (estilo original) ----------------
st.subheader(f"📈 EMERREL (izq) y Plantas·m²·semana (der, 0–100) · Tope={int(A2_CAP)}")
fig = go.Figure()

# EMERREL (línea, eje izquierdo)
fig.add_trace(go.Scatter(
    x=df_daily["fecha"], y=df_daily["EMERREL"],
    mode="lines", name="EMERREL (cruda)"
))

# EMERREL_SUP (semitransparente, por delante, eje izquierdo)
fig.add_trace(go.Scatter(
    x=df_daily["fecha"], y=df_daily["EMERREL_SUP"],
    mode="lines", name="EMERREL×(1−Ciec)",
    line=dict(width=3, color="rgba(60,60,60,0.45)")
))

# Serie semanal de densidad equivalente (pl·m²·sem⁻¹), eje derecho
fig.add_trace(go.Scatter(
    x=df_week["fecha"], y=df_week["EQ_CAP"],
    mode="lines+markers", name="Densidad eq. semanal (cap)",
    yaxis="y2"
))

fig.update_layout(
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_title="Tiempo",
    yaxis_title="EMERREL",
    yaxis2=dict(
        overlaying="y", side="right",
        title="Plantas·m²·sem⁻¹ (cap)",
        range=[0, 100], tick0=0, dtick=20, showgrid=False
    ),
    legend_title="Series"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Métricas y pérdida ----------------
x_eff = float(eq_daily_cap.sum())  # densidad efectiva (cap)
loss_pct = float(loss_fun(x_eff))
st.markdown(
    f"**Densidad efectiva (cap):** **{x_eff:.1f}** pl·m²  ·  **Pérdida estimada:** **{loss_pct:.2f}%**"
)

# Curva pérdida (limitada por y=x) para referencia
x_curve = np.linspace(0.0, max(A2_CAP, x_eff, 1.0), 400)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida limitada"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines", name="y=x", line=dict(dash="dot")))
fig_loss.add_trace(go.Scatter(x=[x_eff], y=[loss_pct], mode="markers+text",
                              text=[f"{loss_pct:.2f}%"], textposition="top center",
                              name="Escenario"))
fig_loss.update_layout(xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
st.plotly_chart(fig_loss, use_container_width=True)

# ---------------- Exportación ----------------
csv = df_week[["fecha","EQ_CAP"]].to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar densidad equivalente semanal (CSV)", csv,
                   "densidad_equivalente_semanal.csv", "text/csv")

