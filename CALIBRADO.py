# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM · Modelo encadenado S1–S4 con ventanas de control
#   - S1→S2→S3→S4 encadenado (flujo diario)
#   - Ventanas agronómicas preR, preemR, postR y graminicida
#   - Escalado por AUC a pl·m² con tope A2 (62/125/250)
#   - Pérdida hiperbólica limitada (pérdida ≤ x)
#   - Timeline de tratamientos (barras horizontales)
# ===============================================================

import io, re, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------ parámetros calibrados ------------------
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

def loss_fun(x):
    """Pérdida (%) hiperbólica limitada: nunca supera x (pl·m²)."""
    x = np.asarray(x, dtype=float)
    raw = ALPHA * x / (1.0 + (ALPHA * x / LMAX))
    return np.minimum(raw, x)

# ------------------ app ------------------
st.set_page_config(page_title="PREDWEEM · Encadenado S1–S4", layout="wide")
st.title("🌱 PREDWEEM — Modelo encadenado S1–S4 con ventanas de control")

# ------------------ carga de datos ------------------
st.sidebar.header("📂 Datos de entrada")
up = st.sidebar.file_uploader("Archivo CSV o Excel con columnas (fecha, EMERREL / EMERAC)", type=["csv", "xlsx", "xls"])
if up is None:
    st.info("Subí un archivo CSV o Excel para comenzar.")
    st.stop()

dayfirst   = st.sidebar.checkbox("Fecha en formato dd/mm/aaaa", value=True)
is_percent = st.sidebar.checkbox("Valores en % (no 0–1)", value=True)
is_cum     = st.sidebar.checkbox("La serie es acumulada (EMERAC)", value=False)

# === LECTURA SEGURA ===
import chardet, io

# Detecta tipo de archivo
if up.name.lower().endswith((".xlsx", ".xls")):
    df0 = pd.read_excel(up)
else:
    raw_bytes = up.read()
    # Detecta codificación probable
    enc_guess = chardet.detect(raw_bytes)["encoding"] or "utf-8"
    raw_text = raw_bytes.decode(enc_guess, errors="replace")

    # Intenta detección automática de separador
    try:
        df0 = pd.read_csv(io.StringIO(raw_text), sep=None, engine="python")
    except Exception:
        # fallback común para CSV exportados desde Excel en español
        df0 = pd.read_csv(io.StringIO(raw_text), sep=";", engine="python")

# === Limpieza de columnas ===
df0.columns = [str(c).strip().lower() for c in df0.columns]

# Detección heurística de columnas
c_fecha = next((c for c in df0.columns if "fec" in c), df0.columns[0])
c_valor = next((c for c in df0.columns if "emer" in c), df0.columns[1] if len(df0.columns) > 1 else df0.columns[0])

# Parseo y ordenamiento
df0["fecha"] = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = pd.to_numeric(df0[c_valor], errors="coerce")
df = pd.DataFrame({"fecha": df0["fecha"], "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)

if df.empty:
    st.error("⚠️ No se encontraron datos válidos después de procesar el archivo.")
    st.stop()

# === Reindexación diaria ===
idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
serie = df.set_index("fecha")["valor"].reindex(idx)

# Si es acumulada, derivar emergencias diarias
if is_cum:
    serie = serie.diff().fillna(0.0).clip(lower=0.0)
else:
    serie = serie.fillna(0.0)

# Si está en %, convertir a proporción diaria
if is_percent:
    serie = serie / 100.0

serie = serie.clip(lower=0.0)
df_daily = pd.DataFrame({"fecha": idx, "EMERREL": serie.values})

st.success(f"✅ Archivo leído correctamente ({len(df_daily)} días entre {df_daily['fecha'].min().date()} y {df_daily['fecha'].max().date()}).")

# ------------------ tope A2 y siembra ------------------
st.sidebar.header("🌾 Escenario y siembra")
MAX_CAP = float(st.sidebar.selectbox("Tope A2 (pl·m²)", [250, 125, 62], index=0))

years = df_daily["fecha"].dt.year
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min, sow_max = dt.date(year_ref, 5, 1), dt.date(year_ref, 8, 1)
sow_date = st.sidebar.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)

# ------------------ lags S1→S4 y controles ------------------
st.sidebar.header("⚙️ Transiciones (días)")
D1 = st.sidebar.number_input("Duración/retardo S1→S2", 1, 60, 6, 1)
D2 = st.sidebar.number_input("Duración/retardo S2→S3", 1, 90, 21, 1)
D3 = st.sidebar.number_input("Duración/retardo S3→S4", 1, 120, 32, 1)

st.sidebar.header("🧪 Controles y ventanas")
# Reglas fijas:
# preR: ≤ siembra−14 (S1–S2)
# preemR: [siembra..siembra+10] (S1–S2)
# postR: ≥ siembra+20 (S1–S4)
# graminicida: [siembra..siembra+10] (S1–S3)

# activación
use_preR  = st.sidebar.checkbox("Presiembra + residual (S1–S2)", value=False)
use_preem = st.sidebar.checkbox("Preemergente + residual (S1–S2)", value=False)
use_postR = st.sidebar.checkbox("Post + residual (S1–S4)", value=False)
use_gram  = st.sidebar.checkbox("Graminicida post (S1–S3)", value=False)

# parámetros
def _clamp_date(d, lo, hi): 
    return min(max(d, lo), hi)

min_d, max_d = df_daily["fecha"].min().date(), df_daily["fecha"].max().date()

if use_preR:
    preR_days = st.sidebar.number_input("preR: días de residualidad", 1, 120, 45, 1)
    latest_preR = sow_date - timedelta(days=14)
    d0_preR = st.sidebar.date_input("preR: fecha de inicio", value=max(min_d, latest_preR - timedelta(days=preR_days)),
                                    min_value=min_d, max_value=min(latest_preR, max_d))
else:
    preR_days, d0_preR = 0, None

if use_preem:
    preem_days = st.sidebar.number_input("preemR: días de residualidad", 1, 120, 45, 1)
    win_lo, win_hi = sow_date, min(max_d, sow_date + timedelta(days=10))
    d0_preem = st.sidebar.date_input("preemR: fecha de inicio", value=win_lo, min_value=win_lo, max_value=win_hi)
else:
    preem_days, d0_preem = 0, None

if use_postR:
    postR_days = st.sidebar.number_input("postR: días de residualidad", 1, 120, 45, 1)
    win_lo = max(min_d, sow_date + timedelta(days=20))
    d0_postR = st.sidebar.date_input("postR: fecha de inicio", value=win_lo, min_value=win_lo, max_value=max_d)
else:
    postR_days, d0_postR = 0, None

if use_gram:
    gram_days = st.sidebar.number_input("graminicida: ventana (días)", 1, 30, 10, 1)
    win_lo, win_hi = sow_date, min(max_d, sow_date + timedelta(days=10))
    d0_gram = st.sidebar.date_input("graminicida: fecha de inicio", value=win_lo, min_value=win_lo, max_value=win_hi)
else:
    gram_days, d0_gram = 0, None

st.sidebar.header("🎯 Eficiencias (%)")
ef_preR   = st.sidebar.slider("preR (S1–S2)", 0, 100, 70, 1) if use_preR else 0
ef_preem  = st.sidebar.slider("preemR (S1–S2)", 0, 100, 70, 1) if use_preem else 0
ef_postR  = st.sidebar.slider("postR (S1–S4)", 0, 100, 70, 1) if use_postR else 0
ef_gram   = st.sidebar.slider("graminicida (S1–S3)", 0, 100, 60, 1) if use_gram else 0

# ------------------ utilidades ------------------
def auc_time(fechas: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fechas)
    if mask is not None:
        f = f[mask]; y = y[mask]
    if len(f) < 2: return 0.0
    t = (f - f.iloc[0]).dt.days.to_numpy()
    y = np.nan_to_num(y, nan=0.0)
    return float(np.trapz(y, t))

def cap_cumulative(series, cap):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y)
    cum = 0.0
    for i, val in enumerate(y):
        if cum < cap:
            val = min(max(val, 0.0), cap - cum)
            out[i] = val
            cum += val
        else:
            out[i] = 0.0
    return out

def weights_residual(start_date, days, fechas_d):
    if (start_date is None) or (int(days) <= 0): 
        return np.zeros_like(fechas_d, float)
    d0 = start_date
    d1 = start_date + timedelta(days=int(days))
    return ((fechas_d >= d0) & (fechas_d < d1)).astype(float)

# ------------------ escalado por AUC a pl·m² ------------------
ts = pd.to_datetime(df_daily["fecha"])
fechas_d = ts.dt.date.values

mask_since_sow = (fechas_d >= sow_date)
auc_cruda = auc_time(ts, df_daily["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda <= 0:
    st.error("AUC(EMERREL) desde siembra es 0. No se puede escalar.")
    st.stop()

factor = MAX_CAP / auc_cruda
births_pl = np.where(mask_since_sow, df_daily["EMERREL"].to_numpy(float) * factor, 0.0)
births_pl = cap_cumulative(births_pl, MAX_CAP)  # tope A2 estrictamente aplicado

# ------------------ pesos diarios (ventanas) ------------------
w_preR   = weights_residual(d0_preR,   preR_days,  fechas_d)
w_preem  = weights_residual(d0_preem,  preem_days, fechas_d)
w_postR  = weights_residual(d0_postR,  postR_days, fechas_d)
w_gram   = weights_residual(d0_gram,   gram_days,  fechas_d)

# eficiencias como factores de supervivencia (1 - e%)
rS1 = (1 - (ef_preR/100.0)*w_preR) * (1 - (ef_preem/100.0)*w_preem) * (1 - (ef_postR/100.0)*w_postR) * (1 - (ef_gram/100.0)*w_gram)
rS2 = (1 - (ef_preR/100.0)*w_preR) * (1 - (ef_preem/100.0)*w_preem) * (1 - (ef_postR/100.0)*w_postR) * (1 - (ef_gram/100.0)*w_gram)
rS3 = (1 - (ef_postR/100.0)*w_postR) * (1 - (ef_gram/100.0)*w_gram)
rS4 = (1 - (ef_postR/100.0)*w_postR)

# ------------------ simulación encadenada diaria ------------------
n = len(births_pl)
S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)         # baseline (sin control)
C1 = np.zeros(n); C2 = np.zeros(n); C3 = np.zeros(n); C4 = np.zeros(n)         # con control

D1 = int(D1); D2 = int(D2); D3 = int(D3)

for t in range(n):
    # --- baseline (solo para referencia x2) ---
    inflow = births_pl[t]
    tr12 = (S1[t-1]/D1) if t>0 else 0.0
    tr23 = (S2[t-1]/D2) if t>0 else 0.0
    tr34 = (S3[t-1]/D3) if t>0 else 0.0
    S1[t] = (S1[t-1] if t>0 else 0.0) + inflow - tr12
    S2[t] = (S2[t-1] if t>0 else 0.0) + tr12 - tr23
    S3[t] = (S3[t-1] if t>0 else 0.0) + tr23 - tr34
    S4[t] = (S4[t-1] if t>0 else 0.0) + tr34

    # --- encadenado con control (clave: control antes de transicionar) ---
    # stock previo controlado
    c1_prev = (C1[t-1] if t>0 else 0.0)
    c2_prev = (C2[t-1] if t>0 else 0.0)
    c3_prev = (C3[t-1] if t>0 else 0.0)
    c4_prev = (C4[t-1] if t>0 else 0.0)

    # entra cohorte nueva a S1 y se controla ese mismo día
    c1_now = (c1_prev + inflow) * rS1[t]
    # transiciones desde stocks ya controlados
    t12 = c1_now / D1
    c1_now = c1_now - t12

    c2_now = (c2_prev + t12) * rS2[t]
    t23 = c2_now / D2
    c2_now = c2_now - t23

    c3_now = (c3_prev + t23) * rS3[t]
    t34 = c3_now / D3
    c3_now = c3_now - t34

    c4_now = (c4_prev + t34) * rS4[t]

    C1[t], C2[t], C3[t], C4[t] = c1_now, c2_now, c3_now, c4_now

# aportes efectivos (ponderados)
eff_b = S1*W_S["S1"] + S2*W_S["S2"] + S3*W_S["S3"] + S4*W_S["S4"]
eff_c = C1*W_S["S1"] + C2*W_S["S2"] + C3*W_S["S3"] + C4*W_S["S4"]

X2 = float(eff_b[mask_since_sow].sum())
X3 = float(eff_c[mask_since_sow].sum())
loss2 = float(loss_fun(X2))
loss3 = float(loss_fun(X3))

# ------------------ A2 por AUC (consistente con escalado) ------------------
sup_equiv  = np.divide(eff_b, factor, out=np.zeros_like(eff_b), where=(factor>0))
ctrl_equiv = np.divide(eff_c, factor, out=np.zeros_like(eff_c), where=(factor>0))
auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
auc_sup_ctrl = auc_time(ts, ctrl_equiv, mask=mask_since_sow)
A2_sup  = min(MAX_CAP, MAX_CAP * (auc_sup / auc_cruda))
A2_ctrl = min(MAX_CAP, MAX_CAP * (auc_sup_ctrl / auc_cruda))

# ------------------ gráficos ------------------
st.subheader("📊 Dinámica encadenada de estados (pl·m²·día⁻¹)")
fig_states = go.Figure()
fig_states.add_trace(go.Bar(x=ts, y=C1, name="S1"))
fig_states.add_trace(go.Bar(x=ts, y=C2, name="S2"))
fig_states.add_trace(go.Bar(x=ts, y=C3, name="S3"))
fig_states.add_trace(go.Bar(x=ts, y=C4, name="S4"))
fig_states.update_layout(barmode="stack", xaxis_title="Fecha", yaxis_title="pl·m²·día⁻¹",
                         legend_title="Estados", height=480,
                         margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_states, use_container_width=True)

# Línea EMERREL y efectivos
st.subheader("📈 EMERREL y aportes efectivos")
fig_eff = go.Figure()
fig_eff.add_trace(go.Scatter(x=ts, y=df_daily["EMERREL"], name="EMERREL cruda", mode="lines"))
fig_eff.add_trace(go.Scatter(x=ts, y=eff_b, name="Efectivo sin control", mode="lines"))
fig_eff.add_trace(go.Scatter(x=ts, y=eff_c, name="Efectivo con control", mode="lines", line=dict(dash="dot")))
fig_eff.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL / pl·m²·día⁻¹",
                      height=420, margin=dict(l=10, r=10, t=40, b=60))
st.plotly_chart(fig_eff, use_container_width=True)

# Timeline de tratamientos (barras horizontales abajo del gráfico de EMERREL)
def add_timeline(fig, intervals, lanes, band_height=0.18):
    y0_band = -band_height
    lane_h = band_height / max(1, len(lanes))
    lane_pos = {k: (y0_band + i*lane_h, y0_band + (i+1)*lane_h) for i, (k, _) in enumerate(lanes)}
    colors = {
        "preR": "rgba(0,128,0,0.75)", "preemR": "rgba(0,180,0,0.75)",
        "postR": "rgba(255,140,0,0.85)", "gram": "rgba(30,144,255,0.85)"
    }
    # etiquetas carril
    for i,(k,label) in enumerate(lanes):
        y0,y1 = lane_pos[k]
        fig.add_annotation(xref="paper", yref="paper", x=0.001, y=(y0+y1)/2,
                           text=label, showarrow=False, font=dict(size=10),
                           bgcolor="rgba(255,255,255,0.6)")
    # barras
    for ini, fin, kind in intervals:
        if kind not in lane_pos: continue
        y0,y1 = lane_pos[kind]
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=ini, x1=fin, y0=y0, y1=y1,
                      line=dict(width=0), fillcolor=colors.get(kind,"rgba(120,120,120,0.7)"))
    # marco
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0, x1=1, y0=y0_band, y1=0, line=dict(color="rgba(0,0,0,0.15)", width=1),
                  fillcolor="rgba(0,0,0,0)")

intervals=[]
lanes=[("preR","preR"),("preemR","preemR"),("gram","Grami"),("postR","postR")]
if use_preR and d0_preR:    intervals.append( (pd.to_datetime(d0_preR),  pd.to_datetime(d0_preR)+pd.Timedelta(days=int(preR_days)), "preR") )
if use_preem and d0_preem:  intervals.append( (pd.to_datetime(d0_preem), pd.to_datetime(d0_preem)+pd.Timedelta(days=int(preem_days)), "preemR") )
if use_gram and d0_gram:    intervals.append( (pd.to_datetime(d0_gram),  pd.to_datetime(d0_gram)+pd.Timedelta(days=int(gram_days)), "gram") )
if use_postR and d0_postR:  intervals.append( (pd.to_datetime(d0_postR), pd.to_datetime(d0_postR)+pd.Timedelta(days=int(postR_days)), "postR") )
add_timeline(fig_eff, intervals, lanes, band_height=0.18)
st.plotly_chart(fig_eff, use_container_width=True)

# ------------------ resumen y pérdidas ------------------
st.subheader("📋 Resumen")
c1,c2,c3,c4 = st.columns(4)
c1.metric("x₂ (sin control)", f"{X2:,.2f} pl·m²")
c2.metric("x₃ (con control)", f"{X3:,.2f} pl·m²")
c3.metric("A2 (sup)", f"{A2_sup:,.2f} pl·m²")
c4.metric("A2 (ctrl)", f"{A2_ctrl:,.2f} pl·m²")
st.markdown(f"**Pérdida(x₂):** {loss2:.2f}%  ·  **Pérdida(x₃):** {loss3:.2f}%")

# ------------------ curva de pérdida ------------------
st.subheader("📉 Pérdida (%) vs densidad efectiva (x)")
x_curve = np.linspace(0.0, MAX_CAP, 400)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida (limitada)"))
fig_loss.add_trace(go.Scatter(x=x_curve, y=x_curve, mode="lines", name="Referencia y=x", line=dict(dash="dot")))
fig_loss.add_trace(go.Scatter(x=[X2], y=[loss2], mode="markers+text", text=["x₂"], textposition="top center", name="Sin control"))
fig_loss.add_trace(go.Scatter(x=[X3], y=[loss3], mode="markers+text", text=["x₃"], textposition="top right",  name="Con control"))
fig_loss.update_layout(xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)", height=420)
st.plotly_chart(fig_loss, use_container_width=True)

# ------------------ exportación ------------------
out = pd.DataFrame({
    "fecha": ts, "S1_ctrl": C1, "S2_ctrl": C2, "S3_ctrl": C3, "S4_ctrl": C4,
    "eff_sin_ctrl": eff_b, "eff_con_ctrl": eff_c
})
st.download_button("💾 Descargar resultados (CSV)", out.to_csv(index=False).encode("utf-8"),
                   "predweem_encadenado.csv", "text/csv")

with st.expander("🧪 Parámetros calibrados (solo lectura)"):
    st.markdown(f"- w₁..w₄ = {W_S['S1']}, {W_S['S2']}, {W_S['3'] if '3' in W_S else W_S['S3']}, {W_S['S4']}\n"
                f"- α = {ALPHA} · Lmax = {LMAX}\n"
                f"- loss(x) = α·x/(1+α·x/Lmax), limitada a x")
