# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM v3.3-opt — Simulador calibrado + Optimización de escenarios
# ===============================================================
# - Estructura funcional por edades sucesivas (S1–S4)
# - Tratamientos herbicidas como barras horizontales + "Overlapping"
# - Optimizadores: Grid / Random / Recocido simulado
# ===============================================================

import io, math, random, itertools, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ================== PARÁMETROS CALIBRADOS ==================
ALPHA = 0.503
LMAX = 125.91
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

# Topes A2 (seleccionable)
TOPES_A2 = [250, 125, 62]

# Ventanas y reglas duras
NR_DAYS_DEFAULT = 10            # selectivo no residual (pre) en ventana mock +10d
POST_GRAM_FORWARD_DAYS = 11     # graminicida post: día app + 10
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14
PREEM_R_MAX_AFTER_SOW_DAYS = 10

# ================== UTILIDADES ==================
def hyper_loss(x):
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

def auc_time(fecha: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, dtype=float)
    if mask is not None:
        f = f[mask]
        y_arr = y_arr[mask]
    if len(f) < 2:
        return 0.0
    t = f.view("int64") / 1e9 / 86400.0  # días
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, t))

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y)
    cum = 0.0
    for i in range(len(y)):
        if bool(active_mask[i]):
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val
            cum += val
        else:
            out[i] = 0.0
    return out

def roll_sum_shift(s: pd.Series, win: int, shift_days: int) -> pd.Series:
    return s.rolling(window=win, min_periods=0).sum().shift(shift_days)

# ================== CONFIG STREAMLIT ==================
st.set_page_config(page_title="PREDWEEM v3.3-opt", layout="wide")
st.title("🌾 PREDWEEM v3.3 — Simulador calibrado + Optimización")

# ================== SIDEBAR — Topes y entrada ==================
with st.sidebar:
    st.header("Escenario")
    MAX_PLANTS_CAP = float(st.selectbox("Tope A2 (pl·m²)", TOPES_A2, index=1))

with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV con (fecha, EMERREL o EMERAC)", type=["csv"])
    if up is None:
        st.info("Subí un CSV para continuar.")
        st.stop()
    sep = st.selectbox("Delimitador", [",", ";", "\\t"], index=0)
    dec = st.selectbox("Decimal", [".", ","], index=0)
    dayfirst = st.checkbox("Fecha dd/mm/yyyy", value=True)
    is_cumulative = st.checkbox("Mi serie es acumulada (EMERAC)", value=False)
    as_percent = st.checkbox("Valores en % (no 0–1)", value=True)

raw = up.read().decode("utf-8", errors="ignore")
_sep = {"\\t": "\t"}.get(sep, sep)
df0 = pd.read_csv(io.StringIO(raw), sep=_sep, decimal=dec)
cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, index=0)
c_valor = st.selectbox("Columna de valor", cols, index=1 if len(cols) > 1 else 0)

df = pd.DataFrame({
    "fecha": pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce"),
    "valor": pd.to_numeric(df0[c_valor], errors="coerce")
}).dropna().sort_values("fecha").reset_index(drop=True)

if df.empty:
    st.error("Tras el parseo no quedaron filas válidas.")
    st.stop()

emerrel = df["valor"].astype(float)
if as_percent:
    emerrel = emerrel / 100.0
if is_cumulative:
    emerrel = emerrel.diff().fillna(0.0).clip(lower=0.0)
emerrel = emerrel.clip(lower=0.0)
df_plot = pd.DataFrame({"fecha": pd.to_datetime(df["fecha"]), "EMERREL": emerrel})

ts = pd.to_datetime(df_plot["fecha"])
years = ts.dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1)
sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia (Ciec simplificado)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}**")
    sow_date = st.date_input("Fecha de siembra", value=max(sow_min, ts.min().date()),
                             min_value=sow_min, max_value=sow_max)
    t_lag = st.number_input("Días a emergencia del cultivo", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de canopia", 10, 120, 45, 1)

# Ciec (simplificado, sigmoide básica entre lag y cierre)
days_since_sow = np.array([(d.date() - sow_date).days for d in ts], dtype=float)
def _sigmoid_between(x, a, b):
    if b <= a: b = a + 1
    mid = 0.5 * (a + b)
    r = 4.0 / max(1.0, (b - a))
    return 1.0 / (1.0 + np.exp(-r * (x - mid)))

Ciec = np.where(days_since_sow < t_lag, 0.0, _sigmoid_between(days_since_sow, t_lag, t_close))
Ciec = np.clip(Ciec, 0.0, 1.0)
one_minus_Ciec = 1.0 - Ciec

# ================== EDADES SUCESIVAS S1–S4 ==================
mask_since_sow = (ts.dt.date >= sow_date)
births = np.where(mask_since_sow, df_plot["EMERREL"].astype(float), 0.0)
s = pd.Series(births, index=ts)
S1 = roll_sum_shift(s, 6, 1).fillna(0.0)
S2 = roll_sum_shift(s, 21, 7).fillna(0.0)
S3 = roll_sum_shift(s, 32, 28).fillna(0.0)
S4 = s.cumsum().shift(60).fillna(0.0)

auc_cruda = auc_time(ts, df_plot["EMERREL"], mask=mask_since_sow)
if auc_cruda <= 0:
    st.error("AUC(EMERREL) desde siembra es 0 — no se puede escalar.")
    st.stop()

factor_area = MAX_PLANTS_CAP / auc_cruda
S1_pl = S1.to_numpy(float) * one_minus_Ciec * W_S["S1"] * factor_area
S2_pl = S2.to_numpy(float) * one_minus_Ciec * W_S["S2"] * factor_area
S3_pl = S3.to_numpy(float) * one_minus_Ciec * W_S["S3"] * factor_area
S4_pl = S4.to_numpy(float) * one_minus_Ciec * W_S["S4"] * factor_area
ms = mask_since_sow.to_numpy()
S1_pl = np.where(ms, S1_pl, 0.0)
S2_pl = np.where(ms, S2_pl, 0.0)
S3_pl = np.where(ms, S3_pl, 0.0)
S4_pl = np.where(ms, S4_pl, 0.0)

base_pl_daily = df_plot["EMERREL"].to_numpy(float) * factor_area
base_pl_daily = np.where(ms, base_pl_daily, 0.0)
base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, ms)

supresion = S1_pl + S2_pl + S3_pl + S4_pl
supresion_cap = np.minimum(supresion, base_pl_daily_cap)

# ================== TRATAMIENTOS (UI manual) ==================
with st.sidebar:
    st.header("Tratamientos (manual)")
    pre_glifo = st.checkbox("Pre · glifosato (1 día)", value=False)
    pre_glifo_date = st.date_input("Fecha pre glifo", value=ts.min().date(),
                                   min_value=ts.min().date(), max_value=ts.max().date(),
                                   disabled=not pre_glifo)
    pre_selNR = st.checkbox("Pre · selectivo NR (10 días)", value=False)
    pre_selNR_date = st.date_input("Fecha pre selectivo NR", value=ts.min().date(),
                                   min_value=ts.min().date(), max_value=ts.max().date(),
                                   disabled=not pre_selNR)
    preR = st.checkbox("Presiembra · selectivo + residual", value=False,
                       help="Solo ≤ siembra−14, actúa S1–S2")
    preR_days = st.slider("Duración presiembraR (d)", 15, 120, 45, 1, disabled=not preR)
    preR_max_date = sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    preR_date = st.date_input("Fecha presiembraR", value=min(ts.min().date(), preR_max_date),
                              min_value=ts.min().date(), max_value=min(ts.max().date(), preR_max_date),
                              disabled=not preR)

with st.sidebar:
    preemR = st.checkbox("Preemergente · selectivo + residual", value=False,
                         help="[siembra..siembra+10], actúa S1–S2")
    preemR_days = st.slider("Duración preemR (d)", 15, 120, 45, 1, disabled=not preemR)
    preem_min = sow_date
    preem_max = min(ts.max().date(), sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preemR_date = st.date_input("Fecha preemR", value=preem_min, min_value=preem_min,
                                max_value=preem_max, disabled=not preemR)

with st.sidebar:
    post_gram = st.checkbox("Post · graminicida (NR +10d)", value=False)
    post_gram_date = st.date_input("Fecha post graminicida", value=max(ts.min().date(), sow_date),
                                   min_value=ts.min().date(), max_value=ts.max().date(),
                                   disabled=not post_gram)
    post_selR = st.checkbox("Post · selectivo + residual", value=False,
                            help="≥ siembra+20, actúa S1–S4")
    post_min_postR = max(ts.min().date(), sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha post residual", value=post_min_postR,
                                   min_value=post_min_postR, max_value=ts.max().date(),
                                   disabled=not post_selR)
    post_res_dias = st.slider("Duración postR (d)", 30, 120, 45, 1, disabled=not post_selR)

# ================== EFICIENCIAS ==================
with st.sidebar:
    st.header("Eficiencias (%)")
    ef_pre_glifo = st.slider("Pre glifo (1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR = st.slider("Pre selectivo NR (10d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR = st.slider("PresiembraR", 0, 100, 70, 1) if preR else 0
    ef_preemR = st.slider("PreemR", 0, 100, 70, 1) if preemR else 0
    ef_post_gram = st.slider("Post graminicida", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR = st.slider("Post residual", 0, 100, 70, 1) if post_selR else 0

# ================== PESOS TEMPORALES (para aplicar control) ==================
fechas_d = ts.dt.date.values

def weights_one_day(date_val):
    if not date_val:
        return np.zeros_like(fechas_d, float)
    d0 = date_val
    return ((fechas_d >= d0) & (fechas_d < (d0 + timedelta(days=1)))).astype(float)

def weights_residual(start_date, dias):
    w = np.zeros_like(fechas_d, float)
    if (not start_date) or (not dias) or (int(dias) <= 0):
        return w
    d0 = start_date
    d1 = start_date + timedelta(days=int(dias))
    mask = (fechas_d >= d0) & (fechas_d < d1)
    w[mask] = 1.0
    return w

# ================== APLICAR CONTROLES ==================
ctrl_S1 = np.ones_like(fechas_d, float)
ctrl_S2 = np.ones_like(fechas_d, float)
ctrl_S3 = np.ones_like(fechas_d, float)
ctrl_S4 = np.ones_like(fechas_d, float)

def apply_efficiency_per_state(weights, eff_pct, states_sel):
    if eff_pct <= 0 or (not states_sel):
        return
    reduc = np.clip(1.0 - (eff_pct / 100.0) * np.clip(weights, 0.0, 1.0), 0.0, 1.0)
    if "S1" in states_sel: np.multiply(ctrl_S1, reduc, out=ctrl_S1)
    if "S2" in states_sel: np.multiply(ctrl_S2, reduc, out=ctrl_S2)
    if "S3" in states_sel: np.multiply(ctrl_S3, reduc, out=ctrl_S3)
    if "S4" in states_sel: np.multiply(ctrl_S4, reduc, out=ctrl_S4)

# Validar reglas
warnings = []
def _chk_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debe ser ≤ siembra ({sow_date}).")
def _chk_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debe ser ≥ siembra ({sow_date}).")

if pre_glifo: _chk_pre(pre_glifo_date, "Pre glifo")
if pre_selNR: _chk_pre(pre_selNR_date, "Pre selectivo NR")
if preR and preR_date > (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
    warnings.append(f"PresiembraR ≤ siembra−{PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW}")
if preemR and (preemR_date < sow_date or preemR_date > sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
    warnings.append(f"PreemR ∈ [siembra..siembra+{PREEM_R_MAX_AFTER_SOW_DAYS}]")
if post_gram: _chk_post(post_gram_date, "Post graminicida")
if post_selR and post_selR_date < sow_date + timedelta(days=20):
    warnings.append("Post residual ≥ siembra+20")

for w in warnings:
    st.warning(w)

# Aplicar controles (manual)
if preR:        apply_efficiency_per_state(weights_residual(preR_date, preR_days), ef_preR, ["S1", "S2"])
if preemR:      apply_efficiency_per_state(weights_residual(preemR_date, preemR_days), ef_preemR, ["S1", "S2"])
if pre_selNR:   apply_efficiency_per_state(weights_residual(pre_selNR_date, NR_DAYS_DEFAULT), ef_pre_selNR, ["S1","S2","S3","S4"])
if pre_glifo:   apply_efficiency_per_state(weights_one_day(pre_glifo_date), ef_pre_glifo, ["S1","S2","S3","S4"])
if post_gram:   apply_efficiency_per_state(weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS), ef_post_gram, ["S1","S2","S3"])
if post_selR:   apply_efficiency_per_state(weights_residual(post_selR_date, post_res_dias), ef_post_selR, ["S1","S2","S3","S4"])

S1_ctrl = S1_pl * ctrl_S1
S2_ctrl = S2_pl * ctrl_S2
S3_ctrl = S3_pl * ctrl_S3
S4_ctrl = S4_pl * ctrl_S4
tot_ctrl_daily = S1_ctrl + S2_ctrl + S3_ctrl + S4_ctrl
sup_ctrl_cap = np.minimum(tot_ctrl_daily, supresion_cap)

X2 = float(np.nansum(supresion_cap[ms]))
X3 = float(np.nansum(sup_ctrl_cap[ms]))
loss_x2 = float(hyper_loss(X2))
loss_x3 = float(hyper_loss(X3))

# ================== FIGURA 1 — EMERREL + BARRAS TRATAMIENTOS ==================
def draw_treatment_bars(fig_obj, intervals):
    """Dibuja barras horizontales y overlapping."""
    color_map = {
        "pre_glifo": "#808080",
        "pre_selNR": "#404040",
        "preR": "#008000",
        "preemR": "#00C000",
        "postR": "#FF8C00",
        "post_gram": "#1E90FF",
    }
    y_levels = {
        "pre_glifo": 1.02,
        "pre_selNR": 1.04,
        "preR": 1.06,
        "preemR": 1.08,
        "postR": 1.10,
        "post_gram": 1.12,
    }
    # Barras principales
    for ini, fin, kind in intervals:
        col = color_map.get(kind, "gray")
        y = y_levels.get(kind, 1.05)
        fig_obj.add_trace(go.Scatter(
            x=[ini, fin],
            y=[y, y],
            mode="lines",
            line=dict(color=col, width=10),
            name=kind,
            hovertext=f"{kind}<br>{ini.date()} → {fin.date()}",
            hoverinfo="text",
            showlegend=True
        ))
    # Overlaps
    for i, (a0, a1, ak) in enumerate(intervals):
        for j in range(i + 1, len(intervals)):
            b0, b1, bk = intervals[j]
            o0 = max(a0, b0); o1 = min(a1, b1)
            if o0 < o1:
                fig_obj.add_trace(go.Scatter(
                    x=[o0, o1],
                    y=[1.14, 1.14],
                    mode="lines",
                    line=dict(color="purple", width=10),
                    name="Overlapping",
                    hovertext=f"Overlapping {ak} + {bk}",
                    hoverinfo="text",
                    showlegend=True
                ))

def collect_intervals_from_ui():
    intervals = []
    if pre_glifo:
        ini = pd.to_datetime(pre_glifo_date)
        fin = ini + pd.Timedelta(days=1)
        intervals.append((ini, fin, "pre_glifo"))
    if pre_selNR:
        ini = pd.to_datetime(pre_selNR_date)
        fin = ini + pd.Timedelta(days=NR_DAYS_DEFAULT)
        intervals.append((ini, fin, "pre_selNR"))
    if preR:
        ini = pd.to_datetime(preR_date)
        fin = ini + pd.Timedelta(days=int(preR_days))
        intervals.append((ini, fin, "preR"))
    if preemR:
        ini = pd.to_datetime(preemR_date)
        fin = ini + pd.Timedelta(days=int(preemR_days))
        intervals.append((ini, fin, "preemR"))
    if post_selR:
        ini = pd.to_datetime(post_selR_date)
        fin = ini + pd.Timedelta(days=int(post_res_dias))
        intervals.append((ini, fin, "postR"))
    if post_gram:
        ini = pd.to_datetime(post_gram_date)
        fin = ini + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)
        intervals.append((ini, fin, "post_gram"))
    return intervals

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL"))
fig1.add_trace(go.Scatter(x=ts, y=supresion_cap, mode="lines", name="Plantas·m² (cap)", yaxis="y2"))
fig1.update_layout(
    title=f"Figura 1 — EMERREL + Tratamientos (barras horizontales) · Tope A2={int(MAX_PLANTS_CAP)}",
    xaxis_title="Fecha",
    yaxis_title="EMERREL",
    yaxis2=dict(overlaying="y", side="right", title="Plantas·m²·día⁻¹ (cap)", showgrid=False),
    margin=dict(l=10, r=10, t=50, b=10),
)

intervals_ui = collect_intervals_from_ui()
draw_treatment_bars(fig1, intervals_ui)
st.plotly_chart(fig1, use_container_width=True)

st.caption(
    f"AUC(EMERREL) desde siembra = {auc_cruda:.3f} → factor a plantas = {factor_area:.4f} · "
    f"x₂={X2:.1f} (loss={loss_x2:.2f}%) · x₃={X3:.1f} (loss={loss_x3:.2f}%)"
)

# ================== FIGURA 2 — PÉRDIDA (%) vs x ==================
x_curve = np.linspace(0, MAX_PLANTS_CAP, 400)
y_curve = hyper_loss(x_curve)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida (%)"))
fig2.add_trace(go.Scatter(x=[X2], y=[hyper_loss(X2)], mode="markers+text",
                          text=[f"x₂={X2:.1f}"], textposition="top center", name="sin control"))
fig2.add_trace(go.Scatter(x=[X3], y=[hyper_loss(X3)], mode="markers+text",
                          text=[f"x₃={X3:.1f}"], textposition="top center", name="con control"))
fig2.update_layout(
    title="Figura 2 — Pérdida de rendimiento (%) vs x",
    xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig2, use_container_width=True)

# ================== FIGURA 3 — DINÁMICA S1–S4 ==================
fig3 = go.Figure()
for name, arr in [("S1", S1_pl), ("S2", S2_pl), ("S3", S3_pl), ("S4", S4_pl)]:
    fig3.add_trace(go.Scatter(x=ts, y=arr, mode="lines", name=name))
fig3.update_layout(
    title="Figura 3 — Dinámica temporal por edades sucesivas (S1–S4)",
    xaxis_title="Fecha", yaxis_title="pl·m²·día⁻¹",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig3, use_container_width=True)

# ================== OPTIMIZACIÓN ==================
st.header("🧠 Optimización de cronogramas")

with st.sidebar:
    st.header("Búsqueda de fechas")
    sow_from = st.date_input("Siembra: desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_to   = st.date_input("Siembra: hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step = st.number_input("Paso siembra (días)", 1, 30, 7, 1)

    use_preR_opt = st.checkbox("PresiembraR (S1–S2)", value=True)
    use_preemR_opt = st.checkbox("PreemR (S1–S2)", value=True)
    use_postR_opt = st.checkbox("PostR (S1–S4)", value=True)
    use_postG_opt = st.checkbox("Post graminicida (S1–S3)", value=True)

    ef_preR_opt = st.slider("Efic. presiembraR (%)", 0, 100, 70, 1) if use_preR_opt else 0
    ef_preemR_opt = st.slider("Efic. preemR (%)", 0, 100, 70, 1) if use_preemR_opt else 0
    ef_postR_opt = st.slider("Efic. postR (%)", 0, 100, 70, 1) if use_postR_opt else 0
    ef_postG_opt = st.slider("Efic. post graminicida (%)", 0, 100, 65, 1) if use_postG_opt else 0

    preR_min_back = st.number_input("PresiembraR: hasta X días antes", 14, 120, 45, 1)
    preR_step = st.number_input("Paso presiembraR (d)", 1, 30, 7, 1)
    preem_step = st.number_input("Paso preemR (d)", 1, 10, 3, 1)
    post_fw_max = st.number_input("Post: hasta +X días de siembra", 20, 180, 60, 1)
    post_step = st.number_input("Paso post (d)", 1, 30, 7, 1)
    res_min, res_max = st.slider("Residualidad [min..max] (d)", 15, 120, (30, 60), 5)
    res_step = st.number_input("Paso residualidad (d)", 1, 30, 5, 1)

    optimizer = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100)
    if optimizer == "Recocido simulado":
        sa_iters = st.number_input("Iteraciones SA", 100, 50000, 5000, 100)
        sa_T0 = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("Enfriamiento γ", 0.80, 0.9999, 0.995, 0.0001)

if sow_from > sow_to:
    st.error("Rango de siembra inválido.")
    st.stop()

ts_all = ts.copy()
fechas_d_all = ts_all.dt.date.values
emerrel_all = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()

def recompute_env(sow_d: dt.date):
    mask_since = (ts_all.dt.date >= sow_d)
    births = np.where(mask_since.to_numpy(), emerrel_all, 0.0)
    s = pd.Series(births, index=ts_all)

    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).reindex(ts_all).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).reindex(ts_all).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).reindex(ts_all).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).reindex(ts_all).to_numpy(float)

    # Ciec con la misma sigmoide
    days_s = np.array([(d.date() - sow_d).days for d in ts_all], dtype=float)
    C = np.where(days_s < t_lag, 0.0, _sigmoid_between(days_s, t_lag, t_close))
    one_minus = 1.0 - np.clip(C, 0.0, 1.0)

    auc_cruda_loc = auc_time(ts_all, emerrel_all, mask=mask_since)
    if auc_cruda_loc <= 0:
        return None

    factor_a = MAX_PLANTS_CAP / auc_cruda_loc
    S1_plx = np.where(mask_since, S1 * one_minus * W_S["S1"] * factor_a, 0.0)
    S2_plx = np.where(mask_since, S2 * one_minus * W_S["S2"] * factor_a, 0.0)
    S3_plx = np.where(mask_since, S3 * one_minus * W_S["S3"] * factor_a, 0.0)
    S4_plx = np.where(mask_since, S4 * one_minus * W_S["S4"] * factor_a, 0.0)

    base_pl = np.where(mask_since, emerrel_all * factor_a, 0.0)
    base_cap = cap_cumulative(base_pl, MAX_PLANTS_CAP, mask_since.to_numpy())
    sup_cap = np.minimum(S1_plx + S2_plx + S3_plx + S4_plx, base_cap)

    return {
        "mask_since": mask_since.to_numpy(),
        "factor_area": factor_a,
        "S_pl": (S1_plx, S2_plx, S3_plx, S4_plx),
        "sup_cap": sup_cap,
        "ts": ts_all,
        "fechas_d": fechas_d_all
    }

def daterange(start_date, end_date, step_days):
    out = []
    cur = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    while cur <= end:
        out.append(cur)
        cur += pd.Timedelta(days=int(step_days))
    return out

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_min_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur += pd.Timedelta(days=int(preR_step))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur += pd.Timedelta(days=int(preem_step))
    return out

def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_fw_max))
    if end < start: return []
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur += pd.Timedelta(days=int(post_step))
    return out

res_days = list(range(int(res_min), int(res_max) + 1, int(res_step)))
if int(res_max) not in res_days:
    res_days.append(int(res_max))

def act(kind, date_val, R=None, eff=0):
    return {"kind": kind, "date": pd.to_datetime(date_val).date(),
            "days": int(R) if R is not None else (POST_GRAM_FORWARD_DAYS if kind=="post_gram" else 1),
            "eff": int(eff)}

def evaluate(sd: dt.date, schedule: list):
    sow = pd.to_datetime(sd)
    # Reglas duras
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR" and d < (sow + pd.Timedelta(days=20)): return None
        if a["kind"] == "preR" and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None

    env = recompute_env(sd)
    if env is None:
        return None

    S1p, S2p, S3p, S4p = env["S_pl"]
    sup_cap = env["sup_cap"]
    fechas_d_local = env["fechas_d"]
    mask_since = env["mask_since"]

    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def apply(weights, eff, states):
        if eff <= 0: return
        reduc = np.clip(1.0 - (eff/100.0) * np.clip(weights, 0.0, 1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    for a in schedule:
        d0 = a["date"]
        d1 = a["date"] + pd.Timedelta(days=int(a["days"]))
        w = ((fechas_d_local >= d0) & (fechas_d_local < d1)).astype(float)
        if a["kind"] == "post_gram":
            apply(w, a["eff"], ["S1","S2","S3"])
        elif a["kind"] == "postR":
            apply(w, a["eff"], ["S1","S2","S3","S4"])
        elif a["kind"] in ("preR", "preemR", "pre_glifo", "pre_selNR"):
            # en este optimizador solo usamos preR/preemR/postR/post_gram
            states = ["S1","S2"]
            apply(w, a["eff"], states)

    tot_ctrl = S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4
    ctrl_cap = np.minimum(tot_ctrl, sup_cap)
    X3loc = float(np.nansum(ctrl_cap[mask_since]))
    loss3 = float(hyper_loss(X3loc))
    return {"sow": sd, "loss_pct": loss3, "x3": X3loc, "schedule": schedule}

sow_candidates = daterange(sow_from, sow_to, sow_step)
def build_grid():
    scenarios = []
    for sd in sow_candidates:
        groups = []
        if use_preR_opt:
            groups.append([act("preR", d, R, ef_preR_opt) for d in pre_sow_dates(sd) for R in res_days])
        if use_preemR_opt:
            groups.append([act("preemR", d, R, ef_preemR_opt) for d in preem_dates(sd) for R in res_days])
        if use_postR_opt:
            groups.append([act("postR", d, R, ef_postR_opt) for d in post_dates(sd) for R in res_days])
        if use_postG_opt:
            groups.append([act("post_gram", d, None, ef_postG_opt) for d in post_dates(sd)])

        combos = [[]]
        for r in range(1, len(groups) + 1):
            for subset in itertools.combinations(range(len(groups)), r):
                for p in itertools.product(*[groups[i] for i in subset]):
                    combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random():
    sd = random.choice(sow_candidates)
    sch = []
    if use_preR_opt and random.random() < 0.7:
        cand = pre_sow_dates(sd)
        if cand: sch.append(act("preR", random.choice(cand), random.choice(res_days), ef_preR_opt))
    if use_preemR_opt and random.random() < 0.7:
        cand = preem_dates(sd)
        if cand: sch.append(act("preemR", random.choice(cand), random.choice(res_days), ef_preemR_opt))
    if use_postR_opt and random.random() < 0.7:
        cand = post_dates(sd)
        if cand: sch.append(act("postR", random.choice(cand), random.choice(res_days), ef_postR_opt))
    if use_postG_opt and random.random() < 0.7:
        cand = post_dates(sd)
        if cand: sch.append(act("post_gram", random.choice(cand), None, ef_postG_opt))
    return (pd.to_datetime(sd).date(), sch)

results = []
run_col1, run_col2 = st.columns([1,1])
with run_col1:
    do_run = st.button("▶️ Ejecutar optimización", use_container_width=True)
with run_col2:
    st.caption("La función objetivo minimiza la pérdida (%) evaluando x₃ con cap A2.")

if do_run:
    st.info("Optimizando…")
    if optimizer == "Grid (combinatorio)":
        scenarios = build_grid()
        if len(scenarios) > max_evals:
            random.seed(123)
            scenarios = random.sample(scenarios, k=int(max_evals))
        prog = st.progress(0.0)
        n = len(scenarios)
        step = max(1, n // 100)
        for i, (sd, sch) in enumerate(scenarios, 1):
            r = evaluate(sd, sch)
            if r is not None:
                results.append(r)
            if i % step == 0 or i == n:
                prog.progress(min(1.0, i / n))
        prog.empty()
    elif optimizer == "Búsqueda aleatoria":
        N = int(max_evals)
        prog = st.progress(0.0)
        for i in range(1, N + 1):
            sd, sch = sample_random()
            r = evaluate(sd, sch)
            if r is not None:
                results.append(r)
            if i % max(1, N // 100) == 0 or i == N:
                prog.progress(min(1.0, i / N))
        prog.empty()
    else:
        # Recocido simulado
        cur = sample_random()
        cur_eval = evaluate(*cur)
        tries = 0
        while cur_eval is None and tries < 200:
            cur = sample_random()
            cur_eval = evaluate(*cur)
            tries += 1
        if cur_eval is None:
            st.error("No se encontró estado inicial válido.")
        else:
            best = cur_eval
            cur_loss = cur_eval["loss_pct"]
            T = float(sa_T0)
            prog = st.progress(0.0)
            for it in range(1, int(sa_iters) + 1):
                cand = sample_random()
                cand_eval = evaluate(*cand)
                if cand_eval is not None:
                    d = cand_eval["loss_pct"] - cur_loss
                    if d <= 0 or random.random() < math.exp(-d / max(1e-9, T)):
                        cur_eval = cand_eval
                        cur_loss = cand_eval["loss_pct"]
                        if cur_loss < best["loss_pct"]:
                            best = cand_eval
                    results.append(cur_eval)
                T *= float(sa_cooling)
                if it % max(1, int(sa_iters) // 100) == 0 or it == int(sa_iters):
                    prog.progress(min(1.0, it / float(sa_iters)))
            results.append(best)
            prog.empty()

# ================== REPORTE MEJOR ESCENARIO ==================
if results:
    best = sorted(results, key=lambda r: r["loss_pct"])[0]
    st.subheader("🏆 Mejor escenario encontrado")
    st.markdown(
        f"- **Siembra:** **{best['sow']}**  \n"
        f"- **Pérdida estimada:** **{best['loss_pct']:.2f}%**  \n"
        f"- **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"- **Nº intervenciones:** {len(best['schedule'])}"
    )

    # Tabla del cronograma
    rows = []
    for a in best["schedule"]:
        ini = pd.to_datetime(a["date"])
        fin = ini + pd.Timedelta(days=int(a["days"]))
        rows.append({
            "Intervención": a["kind"],
            "Inicio": str(ini.date()),
            "Fin": str(fin.date()),
            "Duración (d)": int(a["days"]),
            "Eficiencia (%)": int(a["eff"]),
        })
    df_best = pd.DataFrame(rows)
    if not df_best.empty:
        st.dataframe(df_best, use_container_width=True)
        st.download_button(
            "Descargar cronograma óptimo (CSV)",
            df_best.to_csv(index=False).encode("utf-8"),
            "mejor_cronograma.csv", "text/csv"
        )

    # Visual: Figura 1 para el mejor
    envb = recompute_env(best["sow"])
    if envb is not None:
        S1p, S2p, S3p, S4p = envb["S_pl"]
        sup_cap_b = envb["sup_cap"]
        fechas_d_b = envb["fechas_d"]

        # Aplicar schedule óptimo
        c1 = np.ones_like(fechas_d_b, float)
        c2 = np.ones_like(fechas_d_b, float)
        c3 = np.ones_like(fechas_d_b, float)
        c4 = np.ones_like(fechas_d_b, float)

        def _apply(weights, eff, states):
            if eff <= 0: return
            reduc = np.clip(1.0 - (eff/100.0) * np.clip(weights, 0.0, 1.0), 0.0, 1.0)
            if "S1" in states: np.multiply(c1, reduc, out=c1)
            if "S2" in states: np.multiply(c2, reduc, out=c2)
            if "S3" in states: np.multiply(c3, reduc, out=c3)
            if "S4" in states: np.multiply(c4, reduc, out=c4)

        intervals_best = []
        for a in best["schedule"]:
            d0 = a["date"]; d1 = a["date"] + pd.Timedelta(days=int(a["days"]))
            w = ((fechas_d_b >= d0) & (fechas_d_b < d1)).astype(float)
            if a["kind"] == "post_gram": _apply(w, a["eff"], ["S1","S2","S3"])
            elif a["kind"] == "postR":   _apply(w, a["eff"], ["S1","S2","S3","S4"])
            elif a["kind"] == "preR":    _apply(w, a["eff"], ["S1","S2"])
            elif a["kind"] == "preemR":  _apply(w, a["eff"], ["S1","S2"])
            intervals_best.append((pd.to_datetime(d0), pd.to_datetime(d1), a["kind"]))

        tot_ctrl_b = S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4
        ctrl_cap_b = np.minimum(tot_ctrl_b, sup_cap_b)

        fig_best = go.Figure()
        fig_best.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL"))
        fig_best.add_trace(go.Scatter(x=envb["ts"], y=ctrl_cap_b, mode="lines", name="Plantas·m² (cap)", yaxis="y2"))
        fig_best.update_layout(
            title="Figura 1 — Mejor escenario (barras horizontales + overlapping)",
            xaxis_title="Fecha",
            yaxis_title="EMERREL",
            yaxis2=dict(overlaying="y", side="right", title="Plantas·m²·día⁻¹ (cap)", showgrid=False),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        draw_treatment_bars(fig_best, intervals_best)
        st.plotly_chart(fig_best, use_container_width=True)

st.success("✅ Listo: simulación + barras horizontales + overlapping + optimización.")

