# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Encadenado S1–S4 con ventanas de control + timeline
# ===============================================================

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta

# ------------------ CONFIGURACIÓN ------------------
st.set_page_config(page_title="PREDWEEM · Encadenado S1–S4", layout="wide")
st.title("🌱 PREDWEEM — Modelo encadenado de edades sucesivas (S₁–S₄)")

# ------------------ PARÁMETROS (predefinidos) ------------------
# Pérdida calibrada
ALPHA = 0.503
LMAX  = 125.91

# Tasas de transición (encadenado)
P12 = 1/10   # S1→S2 (~10 días)
P23 = 1/15   # S2→S3 (~15 días)
P34 = 1/20   # S3→S4 (~20 días)

# Ventanas de aplicación (predefinidas)
DUR_PRE_R   = 45     # días de residualidad presiembra
DUR_PREEM_R = 45     # días residualidad preemergente
DUR_POST_R  = 45     # días residualidad postemergente
DUR_GRAMI   = 11     # ventana de graminicida (día app +10)

# Eficiencias (predefinidas)
EFF_PRE_R   = 0.70   # 70%
EFF_PREEM_R = 0.60   # 60%
EFF_POST_R  = 0.50   # 50%
EFF_GRAMI   = 0.60   # 60%

# Reglas rígidas de calendario
PRE_R_MAX_BEFORE_SOW   = 14   # presiembra ≤ siembra−14
PREEM_R_MAX_AFTER_SOW  = 10   # preemergente: [siembra..siembra+10]
POST_R_MIN_AFTER_SOW   = 20   # post: ≥ siembra+20
GRAMI_MAX_AFTER_SOW    = 10   # grami: 0..+10

# ------------------ FUNCIÓN DE PÉRDIDA (limitada) ------------------
def loss_fun(x):
    """Pérdida (%) hiperbólica limitada biológicamente: pérdida ≤ x."""
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)

# ------------------ CARGA DE DATOS ------------------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("CSV o Excel (columnas: fecha, EMERREL)", type=["csv","xlsx","xls"])
dayfirst = st.sidebar.checkbox("Fecha dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores en % (no 0–1)", value=True)
is_cum = st.sidebar.checkbox("Serie acumulada (EMERAC)", value=False)

if up is None:
    st.info("Subí un archivo para comenzar.")
    st.stop()

# lectura robusta
if up.name.lower().endswith((".xlsx",".xls")):
    df0 = pd.read_excel(up)
else:
    raw_bytes = up.read()
    try:
        txt = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw_bytes.decode("latin-1", errors="replace")
    try:
        df0 = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
    except Exception:
        # fallback común en CSV latino
        try:
            df0 = pd.read_csv(io.StringIO(txt), sep=";", engine="python")
        except Exception:
            df0 = pd.read_csv(io.StringIO(txt), sep=",", engine="python")

df0.columns = [c.strip().lower() for c in df0.columns]
# heurística suave para elegir columnas
c_fecha = next((c for c in df0.columns if "fec" in c), df0.columns[0])
c_valor = next((c for c in df0.columns if "emer" in c), df0.columns[1] if len(df0.columns)>1 else df0.columns[0])

df0["fecha"] = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = pd.to_numeric(df0[c_valor], errors="coerce")
df = pd.DataFrame({"fecha": df0["fecha"], "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)

if df.empty:
    st.error("No se pudieron parsear fechas/valores válidos.")
    st.stop()

# serie diaria completa
idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
serie = df.set_index("fecha")["valor"].reindex(idx).fillna(0.0)

# EMERREL diaria
if is_cum:     # EMERAC → diaria
    serie = serie.diff().fillna(0.0).clip(lower=0.0)
if is_percent: # % → fracción
    serie = serie / 100.0

df_daily = pd.DataFrame({"fecha": idx, "EMERREL": serie.values})
st.success(f"✅ Datos listos: {len(df_daily)} días.")

# ------------------ SIEMBRA y FECHAS CLAVE ------------------
year_ref = int(df_daily["fecha"].dt.year.mode().iloc[0])
sow_default = date(year_ref, 7, 1)
sow_date = st.sidebar.date_input("📅 Fecha de siembra", value=sow_default)

# Fechas predefinidas (se ajustan a reglas rígidas)
preR_date   = sow_date - timedelta(days=max(PRE_R_MAX_BEFORE_SOW, 20))  # como mínimo 14 días antes (usamos 20 d)
preemR_date = sow_date + timedelta(days=min(5, PREEM_R_MAX_AFTER_SOW))  # dentro de [0..+10]
postR_date  = sow_date + timedelta(days=max(POST_R_MIN_AFTER_SOW, 25))  # ≥ +20
gram_date   = sow_date + timedelta(days=min(5, GRAMI_MAX_AFTER_SOW))    # 0..+10

# ------------------ MÁSCARAS DE CONTROL ------------------
dates_d = df_daily["fecha"].dt.date.values

def mask_window(dates_arr, start_date, duration_days):
    end_date = start_date + timedelta(days=duration_days)
    return ((dates_arr >= start_date) & (dates_arr < end_date)).astype(float)

# respeta ventanas normativas estrictas
mask_preR   = mask_window(dates_d, preR_date,   DUR_PRE_R)   * (dates_d <= (sow_date - timedelta(days=PRE_R_MAX_BEFORE_SOW)))
mask_preem  = mask_window(dates_d, preemR_date, DUR_PREEM_R) * ((dates_d >= sow_date) & (dates_d <= sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW)))
mask_post   = mask_window(dates_d, postR_date,  DUR_POST_R)  * (dates_d >= (sow_date + timedelta(days=POST_R_MIN_AFTER_SOW)))
mask_grami  = mask_window(dates_d, gram_date,   DUR_GRAMI)   * ((dates_d >= sow_date) & (dates_d <= sow_date + timedelta(days=GRAMI_MAX_AFTER_SOW)))

# ------------------ MODELO ENCADENADO (diario) ------------------
EMERREL = df_daily["EMERREL"].values
n = len(EMERREL)

# flujos diarios por estado
S1 = np.zeros(n)
S2 = np.zeros(n)
S3 = np.zeros(n)
S4 = np.zeros(n)

for t in range(n):
    # S1 nace de la emergencia diaria posterior a siembra
    S1[t] = EMERREL[t] if dates_d[t] >= sow_date else 0.0

# Avance encadenado con controles que detienen la transición
for t in range(1, n):
    # supervivencias (1 - eficiencia·máscara) en el día t
    surv_preR   = 1.0 - EFF_PRE_R   * mask_preR[t]
    surv_preem  = 1.0 - EFF_PREEM_R * mask_preem[t]
    surv_post   = 1.0 - EFF_POST_R  * mask_post[t]
    surv_grami  = 1.0 - EFF_GRAMI   * mask_grami[t]

    # 1) S1 aporta a S2, pero solo lo sobreviviente a preR & preem
    S2[t] = S2[t-1] + P12 * (S1[t-1] * surv_preR * surv_preem)

    # 2) S2 aporta a S3, solo lo sobreviviente (preem aplica también sobre S2 joven)
    S3[t] = S3[t-1] + P23 * (S2[t-1] * surv_preem)

    # 3) S3 aporta a S4, solo lo sobreviviente (post actúa antes del pase a S4)
    S4[t] = S4[t-1] + P34 * (S3[t-1] * surv_post)

    # 4) graminicida actúa sobre S4 ya formado ese día
    S4[t] *= surv_grami

df_states = pd.DataFrame({
    "Fecha": df_daily["fecha"].dt.date,
    "S1": S1, "S2": S2, "S3": S3, "S4": S4
})
df_states["Total"] = df_states[["S1","S2","S3","S4"]].sum(axis=1)

densidad_efectiva = float(df_states["Total"].sum())
perdida_pct = float(loss_fun(densidad_efectiva))

# ------------------ GRÁFICO PRINCIPAL + TIMELINE ------------------
st.subheader("📊 Dinámica temporal y tratamientos (timeline)")

fig = go.Figure()

# Curvas por estado
fig.add_trace(go.Scatter(x=df_states["Fecha"], y=df_states["S1"], name="S₁ Emergencia"))
fig.add_trace(go.Scatter(x=df_states["Fecha"], y=df_states["S2"], name="S₂ Joven"))
fig.add_trace(go.Scatter(x=df_states["Fecha"], y=df_states["S3"], name="S₃ Intermedia"))
fig.add_trace(go.Scatter(x=df_states["Fecha"], y=df_states["S4"], name="S₄ Adulta"))

fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Plantas·m²·día⁻¹",
    legend_title="Estados",
    height=520,
    margin=dict(l=10, r=10, t=50, b=60),
    title="Estados encadenados (S₁→S₂→S₃→S₄) con controles"
)

# --- Timeline (barras horizontales abajo del gráfico) ---
COLOR = {
    "preR":   "rgba(0,128,0,0.75)",
    "preemR": "rgba(0,180,0,0.75)",
    "postR":  "rgba(255,140,0,0.85)",
    "grami":  "rgba(30,144,255,0.85)",
}

def add_band(fig, start_d, dur, color, label):
    if dur <= 0: return
    x0 = pd.Timestamp(start_d)
    x1 = x0 + pd.Timedelta(days=dur)
    fig.add_shape(
        type="rect", xref="x", yref="paper",
        x0=x0, x1=x1, y0=-0.16, y1=-0.06,
        line=dict(width=0), fillcolor=color
    )
    fig.add_annotation(
        x=x0 + (x1-x0)/2, y=-0.11, xref="x", yref="paper",
        text=label, showarrow=False, font=dict(size=11, color="white")
    )

# Dibujar cada tratamiento respetando ventanas
add_band(fig, preR_date,   DUR_PRE_R,   COLOR["preR"],   "Presiembra +R")
add_band(fig, preemR_date, DUR_PREEM_R, COLOR["preemR"], "Preemergente +R")
add_band(fig, postR_date,  DUR_POST_R,  COLOR["postR"],  "Postemergente +R")
add_band(fig, gram_date,   DUR_GRAMI,   COLOR["grami"],  "Graminicida")

# marco de la banda de timeline
fig.add_shape(type="rect", xref="paper", yref="paper",
              x0=0, x1=1, y0=-0.18, y1=-0.02,
              line=dict(color="rgba(0,0,0,0.15)", width=1),
              fillcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True)

# ------------------ PÉRDIDA DE RINDE ------------------
st.subheader("📉 Pérdida de rendimiento (hiperbólica limitada)")
x_curve = np.linspace(0, max(densidad_efectiva, 1e-6), 300)
y_curve = loss_fun(x_curve)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida limitada"))
fig_loss.add_trace(go.Scatter(x=[densidad_efectiva], y=[perdida_pct],
                              mode="markers+text", text=[f"{perdida_pct:.2f}%"],
                              textposition="top center", name="Escenario"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines",
                              name="y = x (límite biológico)", line=dict(dash="dot")))
fig_loss.update_layout(
    xaxis_title="Densidad efectiva (pl·m²)",
    yaxis_title="Pérdida de rendimiento (%)",
    height=420
)
st.plotly_chart(fig_loss, use_container_width=True)

st.success(f"🌾 Densidad efectiva total: {densidad_efectiva:.2f} pl·m² — Pérdida estimada: {perdida_pct:.2f}% (respetando pérdida ≤ densidad)")

# ------------------ TABLA y DESCARGA ------------------
st.subheader("📄 Tabla de resultados")
st.dataframe(df_states, use_container_width=True)

csv = df_states.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar resultados CSV", csv, "estados_encadenados.csv", "text/csv")
