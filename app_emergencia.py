# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v3.18 — (1−Ciec) + AUC + Cohortes SECUENCIALES · PCC configurable por fechas
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ------------------ FUNCIÓN DE PÉRDIDA ------------------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ------------------ ESTADO UI ------------------
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM v3.18 · (1−Ciec) + AUC + Cohortes SECUENCIALES · PCC por fechas"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ------------------ LECTURA CSV ------------------
def sniff_sep_dec(text):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get)
    dec_guess = "." if sample.count(".") >= sample.count(",") else ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw(up):
    return up.read()

def parse_csv(raw, sep_opt, dec_opt):
    head = raw[:8000].decode("utf-8", errors="ignore")
    sep_guess, dec_guess = sniff_sep_dec(head)
    sep = sep_guess if sep_opt == "auto" else sep_opt
    dec = dec_guess if dec_opt == "auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec)
    return df, {"sep": sep, "dec": dec}

def clean_numeric_series(s, decimal="."):
    if s.dtype.kind in "if": return s
    t = s.astype(str).str.replace("%", "", regex=False)
    if decimal == ",": t = t.str.replace(".", "").str.replace(",", ".")
    return pd.to_numeric(t, errors="coerce")

def auc_time(fecha, y, mask=None):
    f = pd.to_datetime(fecha); y_arr = np.asarray(y, float)
    if mask is not None: f = f[mask]; y_arr = y_arr[mask]
    if len(f) < 2: return 0.0
    tdays = ((f - f[0]).dt.days).astype(float)
    return float(np.trapz(y_arr, tdays))

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, float)
    out = np.zeros_like(y); cum = 0.0
    for i in range(len(y)):
        if active_mask[i]:
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val; cum += val
    return out

# ------------------ CARGA DE DATOS ------------------
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria/acumulada)", type=["csv"])
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], 0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], 0)
    dayfirst = st.checkbox("Fecha dd/mm/yyyy", True)
    is_cumulative = st.checkbox("Mi CSV es acumulado", False)
    as_percent = st.checkbox("Valores en %", True)

if up is None:
    st.info("Subí un CSV para continuar."); st.stop()

raw = read_raw(up)
df0, meta = parse_csv(raw, sep_opt, dec_opt)
cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, 0)
c_valor = st.selectbox("Columna de valor", cols, 1 if len(cols) > 1 else 0)

fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst)
vals = clean_numeric_series(df0[c_valor], decimal=meta["dec"])
df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
emerrel = df["valor"].astype(float)
if as_percent: emerrel /= 100.0
if is_cumulative: emerrel = emerrel.diff().fillna(0.0).clip(lower=0)
df_plot = pd.DataFrame({"fecha": df["fecha"], "EMERREL": emerrel})

# ------------------ CANOPY ------------------
year_ref = int(df_plot["fecha"].dt.year.mode().iloc[0])
sow_min = dt.date(year_ref, 5, 1); sow_max = dt.date(year_ref, 8, 1)
with st.sidebar:
    st.header("Siembra y Canopia")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    t_lag = st.number_input("Días a emergencia cultivo", 0, 60, 7)
    t_close = st.number_input("Días a cierre surco", 10, 120, 45)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0)
    lai_max = st.number_input("LAI máx", 0.0, 8.0, 3.0)
    k_beer = st.number_input("k Beer–Lambert", 0.1, 1.2, 0.6)

def compute_canopy(fechas, sow_date, t_lag, t_close, cov_max, lai_max, k_beer):
    days = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas])
    def logistic_between(days, start, end, y_max):
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))
    fc_dyn = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100))
    fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
    LAI = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
    LAI = np.clip(LAI, 0.0, lai_max)
    return fc_dyn, LAI

FC, LAI = compute_canopy(df_plot["fecha"], sow_date, t_lag, t_close, cov_max, lai_max, k_beer)
Ciec = np.clip((LAI / 6.0), 0.0, 1.0)
one_minus_Ciec = 1 - Ciec
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = ts.dt.date >= sow_date

# ------------------ PCC CONFIGURABLE POR FECHAS ------------------
# ===============================================================
# 🌾 PERIODO CRÍTICO DE COMPETENCIA (PCC) — fechas calendario robustas
# ===============================================================
with st.sidebar:
    st.header("Periodo Crítico de Competencia (PCC)")

    usar_pcc = st.checkbox("Integrar densidad efectiva solo dentro del PCC", value=True)

    # Calcular año y límites de las fechas en el CSV
    if len(ts) == 0:
        st.warning("No hay datos de fechas para definir el PCC.")
        st.stop()

    year_ref = int(ts.dt.year.mode().iloc[0])

    # Limites absolutos (convertidos a date)
    ts_min_date = pd.to_datetime(ts.min()).date()
    ts_max_date = pd.to_datetime(ts.max()).date()

    # Fechas por defecto (dentro del rango)
    default_ini = dt.date(year_ref, 10, 10)
    default_fin = dt.date(year_ref, 11, 4)
    if default_ini < ts_min_date: default_ini = ts_min_date
    if default_fin > ts_max_date: default_fin = ts_max_date

    # --- Entradas de usuario ---
    pcc_ini_date = st.date_input(
        "Inicio del PCC",
        value=default_ini,
        min_value=ts_min_date,
        max_value=ts_max_date,
        disabled=not usar_pcc,
        key="pcc_ini"
    )

    pcc_fin_date = st.date_input(
        "Fin del PCC",
        value=default_fin,
        min_value=ts_min_date,
        max_value=ts_max_date,
        disabled=not usar_pcc,
        key="pcc_fin"
    )

    # --- Validación y resumen ---
    if pcc_ini_date > pcc_fin_date:
        st.error("⚠️ La fecha de inicio del PCC no puede ser posterior a la fecha de fin.")
        st.stop()

    if usar_pcc:
        st.caption(
            f"PCC activo: del **{pcc_ini_date}** al **{pcc_fin_date}** "
            f"({(pcc_fin_date - pcc_ini_date).days} días)"
        )
    else:
        st.caption("Integración sobre **todo el ciclo de cultivo**.")

# Máscara temporal compatible
mask_pcc = (ts.dt.date >= pcc_ini_date) & (ts.dt.date <= pcc_fin_date)


# ===============================================================
# 🌾 MANEJO DE TRATAMIENTOS + CÁLCULOS DE X₂/X₃ + GRÁFICOS PRINCIPALES
# ===============================================================
NR_DAYS_DEFAULT = 10
FC_S = {"S1":0.1,"S2":0.3,"S3":0.6,"S4":1.0}

def build_states(emerrel, sow_date):
    ts = pd.to_datetime(df_plot["fecha"])
    mask_since = ts.dt.date >= sow_date
    births = np.where(mask_since, emerrel, 0.0)
    T12, T23, T34 = 10, 15, 20
    S1,S2,S3,S4 = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
    for i in range(len(births)):
        if i-T12>=0: S2[i]+=births[i-T12]; S1[i-T12]-=births[i-T12]
        if i-(T12+T23)>=0: S3[i]+=births[i-(T12+T23)]; S2[i-(T12+T23)]-=births[i-(T12+T23)]
        if i-(T12+T23+T34)>=0: S4[i]+=births[i-(T12+T23+T34)]; S3[i-(T12+T23+T34)]-=births[i-(T12+T23+T34)]
    S1,S2,S3,S4=[np.clip(x,0,None) for x in [S1,S2,S3,S4]]
    return S1,S2,S3,S4,mask_since

S1,S2,S3,S4,ms = build_states(emerrel,sow_date)
auc_cruda = auc_time(ts,emerrel,mask=ms)
factor_area_to_plants = 250/auc_cruda if auc_cruda>0 else None

# ---------- A2 cap ----------
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope A₂ (pl·m²)",[250,125,62],index=0))

if auc_cruda and auc_cruda>0:
    factor_area_to_plants = MAX_PLANTS_CAP/auc_cruda

if factor_area_to_plants:
    S1_pl = np.where(ms,S1*one_minus_Ciec*FC_S["S1"]*factor_area_to_plants,0)
    S2_pl = np.where(ms,S2*one_minus_Ciec*FC_S["S2"]*factor_area_to_plants,0)
    S3_pl = np.where(ms,S3*one_minus_Ciec*FC_S["S3"]*factor_area_to_plants,0)
    S4_pl = np.where(ms,S4*one_minus_Ciec*FC_S["S4"]*factor_area_to_plants,0)
    base_pl_daily = np.where(ms,emerrel*factor_area_to_plants,0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily,MAX_PLANTS_CAP,ms)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=None; base_pl_daily_cap=None

# ---------- GRAFICO EMERREL + PCC ----------
st.subheader("📊 EMERREL + PCC")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts,y=emerrel,name="EMERREL"))
if usar_pcc:
    fig.add_vrect(x0=pcc_ini_date,x1=pcc_fin_date,fillcolor="rgba(255,215,0,0.25)",
                  line_width=0,annotation_text="PCC",annotation_position="top left")
fig.update_layout(title="Emergencia y PCC",xaxis_title="Fecha",yaxis_title="EMERREL (0–1)")
st.plotly_chart(fig,use_container_width=True)

# ===============================================================
# 🧠 OPTIMIZACIÓN COMPLETA (Grid / Aleatoria / Recocido) — PCC por fechas
# ===============================================================

import itertools, random, math as _math

# --- Constantes (por si no están definidas arriba) ---
EPS_REMAIN  = 1e-9
EPS_EXCLUDE = 0.99
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10

fechas_d = ts.dt.date.values  # array de fechas (date) para enmascarados

# ---------- Panel de optimización ----------
st.markdown("---")
st.header("🧠 Optimización de siembra + calendario de controles (PCC por fechas)")

with st.sidebar:
    st.header("Optimización — variables habilitadas")
    sow_from = st.date_input("Siembra desde", value=sow_date, key="opt_sow_from")
    sow_to   = st.date_input("Siembra hasta",  value=min(sow_date + dt.timedelta(days=20), ts.max().date()), key="opt_sow_to")
    sow_step = st.number_input("Paso siembra (días)", 1, 30, 2, 1)

    use_preR_opt     = st.checkbox("Usar presiembra + residual (≤ siembra−14; S1–S2)", True, key="opt_use_preR")
    use_preemR_opt   = st.checkbox("Usar preemergente + residual (siembra..siembra+10; S1–S2)", True, key="opt_use_preem")
    use_post_selR_opt= st.checkbox("Usar post + residual (≥ siembra+20; S1–S4)", True, key="opt_use_postR")
    use_post_gram_opt= st.checkbox(f"Usar graminicida post (+{POST_GRAM_FORWARD_DAYS-1}d; S1–S3)", True, key="opt_use_gram")

    ef_preR_opt      = st.slider("Eficiencia presiembraR (%)", 0, 100, 90, 1, disabled=not use_preR_opt, key="opt_ef_preR") if use_preR_opt else 0
    ef_preemR_opt    = st.slider("Eficiencia preemR (%)",     0, 100, 90, 1, disabled=not use_preemR_opt, key="opt_ef_preem") if use_preemR_opt else 0
    ef_post_selR_opt = st.slider("Eficiencia postR (%)",      0, 100, 90, 1, disabled=not use_post_selR_opt, key="opt_ef_postR") if use_post_selR_opt else 0
    ef_post_gram_opt = st.slider("Eficiencia gram post (%)",  0, 100, 90, 1, disabled=not use_post_gram_opt, key="opt_ef_gram") if use_post_gram_opt else 0

    preR_back   = st.number_input("PresiembraR: hasta X días antes de siembra", 14, 180, 21, 1, key="opt_preR_back")
    preR_step   = st.number_input("Paso fechas PRESIEMBRA (días)", 1, 30, 2, 1, key="opt_preR_step")
    preem_step  = st.number_input("Paso fechas PREEMERGENTE (días)", 1, 10, 2, 1, key="opt_preem_step")

    post_max_d  = st.number_input("POST: máx días después siembra", 20, 180, 60, 1, key="opt_post_maxd")
    post_step   = st.number_input("Paso fechas POST (días)", 1, 30, 2, 1, key="opt_post_step")

    res_min, res_max = st.slider("Residualidad (min–max) [días]", 15, 120, (40, 45), 5, key="opt_res_range")
    res_step = st.number_input("Paso residualidad (días)", 1, 30, 5, 1, key="opt_res_step")

    optimizer = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0, key="opt_algo")
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100, key="opt_maxevals")
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1, key="opt_topk")

    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 50000, 5000, 100, key="opt_sa_iters")
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1, key="opt_sa_T0")
        sa_cooling = st.number_input("Enfriamiento γ", 0.80, 0.9999, 0.995, 0.0001, key="opt_sa_cool")

    # Controles de ejecución
    c1b, c2b = st.columns(2)
    with c1b:
        start_clicked = st.button("▶️ Iniciar optimización", use_container_width=True, disabled=st.session_state.opt_running, key="opt_start")
    with c2b:
        stop_clicked  = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running, key="opt_stop")
    if start_clicked:
        st.session_state.opt_stop = False
        st.session_state.opt_running = True
    if stop_clicked:
        st.session_state.opt_stop = True

# ---------- Validaciones simples ----------
if sow_from > sow_to:
    st.error("Rango de siembra inválido (desde > hasta)."); st.stop()
if res_min >= res_max:
    st.error("Residualidad: el mínimo debe ser menor que el máximo."); st.stop()
if factor_area_to_plants is None:
    st.info("Necesitás AUC(EMERREL) > 0 para optimizar."); st.stop()

# ---------- Utilidades de calendario ----------
def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur=cur+pd.Timedelta(days=int(step_days))
    return out

sow_candidates = daterange(sow_from, sow_to, sow_step)

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preR_step))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd); end = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preem_step))
    return out

def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_max_d))
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(post_step))
    return out

# ---------- Discretización de residualidades ----------
res_days = list(range(int(res_min), int(res_max) + 1, int(res_step)))
if int(res_max) not in res_days: res_days.append(int(res_max))

# ---------- Acciones atómicas ----------
def act_presiembraR(date_val, R, eff): return {"kind":"preR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_preemR(date_val, R, eff):     return {"kind":"preemR",  "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_post_selR(date_val, R, eff):  return {"kind":"postR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}
def act_post_gram(date_val, eff):     return {"kind":"post_gram","date":pd.to_datetime(date_val).date(), "days":POST_GRAM_FORWARD_DAYS, "eff":eff, "states":["S1","S2","S3"]}

# ---------- Recomputos rápidos por fecha de siembra ----------
def compute_ciec_for(sd):
    days = np.array([(pd.Timestamp(d).date() - sd).days for d in ts])
    def logistic_between(days, start, end, y_max):
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        fc = y_max/(1.0+np.exp(-r*(days-t_mid)))
        return np.clip(fc,0.0,1.0)
    # Reusar configuración visual (t_lag, t_close, cov_max, lai_max, k_beer)
    fc_dyn = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100))
    LAI_loc = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
    LAI_loc = np.clip(LAI_loc, 0.0, lai_max)
    Ciec_loc = np.clip(LAI_loc/6.0, 0.0, 1.0)  # Ca/Cs=1 para optimización local
    return 1.0 - Ciec_loc

def recompute_for_sow(sd):
    mask_since_loc = (ts.dt.date >= sd).to_numpy()
    one_minus = compute_ciec_for(sd)
    births = np.where(mask_since_loc, emerrel, 0.0)

    # Estados secuenciales (T12/T23/T34 fijos para la optimización)
    T12, T23, T34 = 10, 15, 20
    S1x, S2x, S3x, S4x = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
    for i in range(len(births)):
        if i-T12>=0: S2x[i]+=births[i-T12]; S1x[i-T12]-=births[i-T12]
        if i-(T12+T23)>=0: S3x[i]+=births[i-(T12+T23)]; S2x[i-(T12+T23)]-=births[i-(T12+T23)]
        if i-(T12+T23+T34)>=0: S4x[i]+=births[i-(T12+T23+T34)]; S3x[i-(T12+T23+T34)]-=births[i-(T12+T23+T34)]
    S1x,S2x,S3x,S4x=[np.clip(x,0,None) for x in [S1x,S2x,S3x,S4x]]

    total_states = S1x+S2x+S3x+S4x
    emeac = np.cumsum(births)
    scale = np.minimum(np.divide(emeac, np.clip(total_states,1e-9,None)), 1.0)
    S1x*=scale; S2x*=scale; S3x*=scale; S4x*=scale

    auc_cruda_loc = auc_time(ts, emerrel, mask=mask_since_loc)
    if not (auc_cruda_loc and auc_cruda_loc > 0): return None
    factor_area = MAX_PLANTS_CAP / auc_cruda_loc

    S1p = np.where(mask_since_loc, S1x * one_minus * FC_S["S1"] * factor_area, 0.0)
    S2p = np.where(mask_since_loc, S2x * one_minus * FC_S["S2"] * factor_area, 0.0)
    S3p = np.where(mask_since_loc, S3x * one_minus * FC_S["S3"] * factor_area, 0.0)
    S4p = np.where(mask_since_loc, S4x * one_minus * FC_S["S4"] * factor_area, 0.0)

    base_pl_d = np.where(mask_since_loc, emerrel * factor_area, 0.0)
    base_pl_d_cap = cap_cumulative(base_pl_d, MAX_PLANTS_CAP, mask_since_loc)
    sup_cap = np.minimum(S1p+S2p+S3p+S4p, base_pl_d_cap)

    return {"mask_since": mask_since_loc, "factor_area": factor_area, "auc_cruda": auc_cruda_loc,
            "S_pl": (S1p, S2p, S3p, S4p), "sup_cap": sup_cap, "ts": ts, "fechas_d": fechas_d}

# ---------- Evaluación de una solución (siembra + agenda) ----------
def evaluate(sd, schedule):
    sow = pd.to_datetime(sd)
    sow_plus20 = sow + pd.Timedelta(days=20)

    # Reglas duras de fechas
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR" and d < sow_plus20: return None
        if a["kind"] == "preR"  and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None

    env = recompute_for_sow(sd)
    if env is None: return None

    mask_since = env["mask_since"]; factor_area = env["factor_area"]
    S1p, S2p, S3p, S4p = env["S_pl"]; sup_cap = env["sup_cap"]
    ts_local, fechas_d_local = env["ts"], env["fechas_d"]

    # Controles multiplicativos
    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def _remaining(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1p * c1 * w)
        if "S2" in states: rem += np.sum(S2p * c2 * w)
        if "S3" in states: rem += np.sum(S3p * c3 * w)
        if "S4" in states: rem += np.sum(S4p * c4 * w)
        return float(rem)

    def _apply(w, eff, states):
        if eff <= 0: return
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    eff_pre = eff_pre2 = eff_all = 0.0
    def _eff(prev, this): return 1.0 - (1.0 - prev)*(1.0 - this)

    order = {"preR":0,"preemR":1,"postR":2,"post_gram":3}
    for a in sorted(schedule, key=lambda a: order.get(a["kind"], 9)):
        ini = pd.to_datetime(a["date"]).date()
        fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
        w = ((fechas_d_local >= ini) & (fechas_d_local < fin)).astype(float)

        if a["kind"] == "preR":
            if _remaining(w, ["S1","S2"]) > EPS_REMAIN and a["eff"] > 0:
                _apply(w, a["eff"], ["S1","S2"])
                eff_pre = _eff(0.0, a["eff"]/100.0)

        elif a["kind"] == "preemR":
            if eff_pre < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining(w, ["S1","S2"]) > EPS_REMAIN:
                    _apply(w, a["eff"], ["S1","S2"])
                    eff_pre2 = _eff(eff_pre, a["eff"]/100.0)
                else:
                    eff_pre2 = eff_pre
            else:
                eff_pre2 = eff_pre

        elif a["kind"] == "postR":
            if eff_pre2 < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining(w, ["S1","S2","S3","S4"]) > EPS_REMAIN:
                    _apply(w, a["eff"], ["S1","S2","S3","S4"])
                    eff_all = _eff(eff_pre2, a["eff"]/100.0)
                else:
                    eff_all = eff_pre2
            else:
                eff_all = eff_pre2

        elif a["kind"] == "post_gram":
            if eff_all < EPS_EXCLUDE and a["eff"] > 0 and _remaining(w, ["S1","S2","S3"]) > EPS_REMAIN:
                _apply(w, a["eff"], ["S1","S2","S3"])

    tot_ctrl = S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    # --- Ventana de evaluación PCC por fechas ---
    if usar_pcc:
        mask_eval = (ts_local >= pcc_ini_date) & (ts_local <= pcc_fin_date)
    else:
        mask_eval = mask_since

    X2loc = float(np.nansum(sup_cap[mask_eval]))
    X3loc = float(np.nansum(plantas_ctrl_cap[mask_eval]))
    loss3 = _loss(X3loc)

    # A2 consistente con la ventana
    sup_equiv  = np.divide(sup_cap,          factor_area, out=np.zeros_like(sup_cap),          where=(factor_area>0))
    ctrl_equiv = np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0))
    auc_sup      = auc_time(ts_local, sup_equiv,  mask=mask_eval)
    auc_sup_ctrl = auc_time(ts_local, ctrl_equiv, mask=mask_eval)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/ max(1e-9, env["auc_cruda"])))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/ max(1e-9, env["auc_cruda"])))

    return {"sow": sd, "loss_pct": float(loss3), "x2": X2loc, "x3": X3loc,
            "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

# ---------- Generación de escenarios ----------
def build_all_scenarios():
    scenarios=[]
    for sd in sow_candidates:
        bundles=[]
        if use_preR_opt:
            bundles.append([act_presiembraR(d, R, ef_preR_opt) for d in pre_sow_dates(sd) for R in res_days])
        if use_preemR_opt:
            bundles.append([act_preemR(d, R, ef_preemR_opt) for d in preem_dates(sd) for R in res_days])
        if use_post_selR_opt:
            bundles.append([act_post_selR(d, R, ef_post_selR_opt) for d in post_dates(sd) for R in res_days])
        if use_post_gram_opt:
            bundles.append([act_post_gram(d, ef_post_gram_opt) for d in post_dates(sd)])

        combos=[[]]
        for r in range(1, len(bundles)+1):
            for subset in itertools.combinations(range(len(bundles)), r):
                for p in itertools.product(*[bundles[i] for i in subset]):
                    combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random_scenario():
    sd = random.choice(sow_candidates)
    schedule=[]
    if use_preR_opt and random.random()<0.7:
        cand = pre_sow_dates(sd)
        if cand: schedule.append(act_presiembraR(random.choice(cand), random.choice(res_days), ef_preR_opt))
    if use_preemR_opt and random.random()<0.7:
        cand = preem_dates(sd)
        if cand: schedule.append(act_preemR(random.choice(cand), random.choice(res_days), ef_preemR_opt))
    if use_post_selR_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(act_post_selR(random.choice(cand), random.choice(res_days), ef_post_selR_opt))
    if use_post_gram_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(act_post_gram(random.choice(cand), ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

# ---------- Ejecutar optimización ----------
status_ph = st.empty()
prog_ph = st.empty()
results = []

if st.session_state.opt_running:
    status_ph.info("Optimizando…")

    if optimizer == "Grid (combinatorio)":
        scenarios = build_all_scenarios()
        total = len(scenarios)
        st.caption(f"Se evaluarán {total:,} configuraciones.")
        if total > max_evals:
            random.seed(123)
            scenarios = random.sample(scenarios, k=int(max_evals))
            st.caption(f"Se muestrean {len(scenarios):,} configuraciones (límite).")
        prog = prog_ph.progress(0.0); n=len(scenarios); step = max(1, n//100)
        for i,(sd,sch) in enumerate(scenarios,1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenida. Progreso: {i-1:,}/{n:,}")
                break
            r = evaluate(sd, sch)
            if r is not None: results.append(r)
            if i % step == 0 or i == n: prog.progress(min(1.0, i/n))
        prog_ph.empty()

    elif optimizer == "Búsqueda aleatoria":
        N = int(max_evals); prog = prog_ph.progress(0.0)
        for i in range(1, N+1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenida. Progreso: {i-1:,}/{N:,}")
                break
            sd, sch = sample_random_scenario()
            r = evaluate(sd, sch)
            if r is not None: results.append(r)
            if i % max(1, N//100) == 0 or i == N: prog.progress(min(1.0, i/N))
        prog_ph.empty()

    else:  # Recocido simulado
        cur = sample_random_scenario()
        cur_eval = evaluate(*cur)
        tries=0
        while cur_eval is None and tries<200:
            cur = sample_random_scenario(); cur_eval = evaluate(*cur); tries+=1
        if cur_eval is None:
            status_ph.error("No fue posible iniciar el recocido con un estado válido.")
        else:
            best_eval = cur_eval; cur_loss = cur_eval["loss_pct"]; T = float(sa_T0)
            prog = prog_ph.progress(0.0)
            for it in range(1, int(sa_iters)+1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Detenida en iteración {it-1:,}/{int(sa_iters):,}.")
                    break
                cand = sample_random_scenario()
                cand_eval = evaluate(*cand)
                if cand_eval is not None:
                    d = cand_eval["loss_pct"] - cur_loss
                    if d <= 0 or random.random() < _math.exp(-d / max(1e-9, T)):
                        cur, cur_eval, cur_loss = cand, cand_eval, cand_eval["loss_pct"]
                        results.append(cur_eval)
                        if cur_loss < best_eval["loss_pct"]:
                            best_eval = cur_eval
                T *= float(sa_cooling)
                if it % max(1, int(sa_iters)//100) == 0 or it == int(sa_iters):
                    prog.progress(min(1.0, it/float(sa_iters)))
            results.append(best_eval)
            prog_ph.empty()

    st.session_state.opt_running = False
    st.session_state.opt_stop = False
    status_ph.success("Optimización finalizada.")
else:
    status_ph.info("Listo para optimizar. Ajustá parámetros y presioná **Iniciar optimización**.")

# ---------- Utilidades de reporte ----------
def schedule_df(sch):
    rows=[]
    for a in sch:
        ini = pd.to_datetime(a["date"])
        fin = ini + pd.Timedelta(days=int(a["days"]))
        rows.append({
            "Intervención": a["kind"], "Inicio": str(ini.date()),
            "Fin": str(fin.date()), "Duración (d)": int(a["days"]),
            "Eficiencia (%)": int(a["eff"]), "Estados": ",".join(a["states"])
        })
    return pd.DataFrame(rows)

def recompute_apply_best(best):
    env = recompute_for_sow(pd.to_datetime(best["sow"]).date())
    if env is None: return None
    ts_b, fechas_b = env["ts"], env["fechas_d"]
    S1p,S2p,S3p,S4p = env["S_pl"]; sup_cap_b = env["sup_cap"]

    c1 = np.ones_like(fechas_b, float)
    c2 = np.ones_like(fechas_b, float)
    c3 = np.ones_like(fechas_b, float)
    c4 = np.ones_like(fechas_b, float)

    def _remaining(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1p * c1 * w)
        if "S2" in states: rem += np.sum(S2p * c2 * w)
        if "S3" in states: rem += np.sum(S3p * c3 * w)
        if "S4" in states: rem += np.sum(S4p * c4 * w)
        return float(rem)

    def _apply(w, eff, states):
        if eff <= 0: return
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    order = {"preR":0,"preemR":1,"postR":2,"post_gram":3}
    for a in sorted(best["schedule"], key=lambda a: order.get(a["kind"], 9)):
        ini = pd.to_datetime(a["date"]).date()
        fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
        w = ((fechas_b >= ini) & (fechas_b < fin)).astype(float)
        if _remaining(w, a["states"]) > EPS_REMAIN and a["eff"] > 0:
            _apply(w, a["eff"], a["states"])

    total_ctrl = S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4
    plantas_ctrl_cap_b = np.minimum(total_ctrl, sup_cap_b)

    df_daily_b = pd.DataFrame({
        "fecha": ts_b,
        "sin_ctrl": sup_cap_b,
        "con_ctrl": plantas_ctrl_cap_b
    })
    df_week_b = df_daily_b.set_index("fecha").resample("W-MON").sum().reset_index()

    return {"ts_b": ts_b, "week_b": df_week_b, "sup_cap_b": sup_cap_b,
            "ctrl_cap_b": plantas_ctrl_cap_b}

# ---------- Reporte del mejor escenario ----------
if results:
    results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))
    best = results_sorted[0]

    st.subheader("🏆 Mejor Escenario")
    st.markdown(
        f"**Siembra:** **{best['sow']}**  \n"
        f"**Pérdida esperada:** **{best['loss_pct']:.2f}%**  \n"
        f"**x₂:** {best['x2']:.1f} · **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"**A₂ (sin/ctrl):** {best['A2_sup']:.1f} / {best['A2_ctrl']:.1f} pl·m²"
    )

    df_best = schedule_df(best["schedule"])
    if len(df_best):
        st.dataframe(df_best, use_container_width=True)
        st.download_button("Descargar cronograma (CSV)", df_best.to_csv(index=False).encode("utf-8"),
                           "mejor_cronograma.csv", "text/csv")

    envb = recompute_apply_best(best)
    if envb is not None:
        ts_b = envb["ts_b"]; df_week_b = envb["week_b"]
        fig_best = go.Figure()
        fig_best.add_trace(go.Scatter(x=ts, y=emerrel, name="EMERREL (izq)", mode="lines"))
        fig_best.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["sin_ctrl"], name="pl·m²·sem sin ctrl", mode="lines+markers", yaxis="y2"))
        fig_best.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["con_ctrl"], name="pl·m²·sem con ctrl", mode="lines+markers", yaxis="y2"))

        # Sombreado PCC (por fechas absolutas seleccionadas)
        if usar_pcc:
            fig_best.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, line_width=0,
                               fillcolor="rgba(255,215,0,0.25)", opacity=0.25,
                               annotation_text="PCC", annotation_position="top left",
                               annotation=dict(font_size=12, font_color="black"))

        fig_best.update_layout(
            title="Mejor escenario — EMERREL + plantas·m²·semana",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="EMERREL"),
            yaxis2=dict(overlaying="y", side="right", title="pl·m²·sem"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_best, use_container_width=True)

        # Pérdida vs x (en la ventana seleccionada)
        if usar_pcc:
            mask_eval_b = (ts >= pcc_ini_date) & (ts <= pcc_fin_date)
        else:
            mask_eval_b = (ts.dt.date >= pd.to_datetime(best["sow"]).date())

        X2_b = float(np.nansum(envb["sup_cap_b"][mask_eval_b]))
        X3_b = float(np.nansum(envb["ctrl_cap_b"][mask_eval_b]))
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400); y_curve = _loss(x_curve)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida (%) vs x"))
        fig_loss.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", name="x₂", text=[f"x₂={X2_b:.1f}"], textposition="top center"))
        fig_loss.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", name="x₃", text=[f"x₃={X3_b:.1f}"], textposition="top right"))
        fig_loss.update_layout(title="Pérdida de rendimiento (%) vs densidad efectiva", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
        st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Aún no hay resultados de optimización para mostrar.")

# ===============================================================
# 🌾 FLUJO DE INDIVIDUOS HACIA EL PCC — VISUALIZACIÓN EXPLICATIVA
# ===============================================================
st.subheader("🌱 Flujo de individuos hacia el Periodo Crítico de Competencia (PCC)")

if factor_area_to_plants is not None:
    # Si en la Parte A definiste 'plantas_supresion_cap' y 'plantas_supresion_ctrl_cap':
    if 'plantas_supresion_cap' in locals() and 'plantas_supresion_ctrl_cap' in locals():
        acum_total = np.cumsum(np.where(ts.dt.date >= sow_date, plantas_supresion_cap, 0.0))
        acum_ctrl  = np.cumsum(np.where(ts.dt.date >= sow_date, plantas_supresion_ctrl_cap, 0.0))
    else:
        # Fallback mínimo (si no están disponibles esas series)
        base_pl_daily = np.where(ts.dt.date >= sow_date, emerrel * factor_area_to_plants, 0.0)
        acum_total = np.cumsum(base_pl_daily)
        acum_ctrl  = np.cumsum(0.7 * base_pl_daily)  # proxy: 70% sobreviven (solo visual)

    idx_ini_pcc = np.argmax(ts >= pcc_ini_date) if usar_pcc else 0
    total_al_pcc = acum_total[idx_ini_pcc] if idx_ini_pcc < len(acum_total) else acum_total[-1]
    ctrl_al_pcc  = acum_ctrl[idx_ini_pcc]  if idx_ini_pcc  < len(acum_ctrl)  else acum_ctrl[-1]

    fig_flujo = go.Figure()
    fig_flujo.add_trace(go.Scatter(x=ts, y=acum_total, mode="lines", name="Acumulado sin control", line=dict(width=2)))
    fig_flujo.add_trace(go.Scatter(x=ts, y=acum_ctrl,  mode="lines", name="Acumulado con control", line=dict(width=2)))

    if usar_pcc:
        fig_flujo.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date,
                            fillcolor="rgba(255,215,0,0.25)", line_width=0,
                            annotation_text="PCC", annotation_position="top left")
        fig_flujo.add_trace(go.Scatter(
            x=[pcc_ini_date], y=[total_al_pcc], mode="markers+text", text=["Llegan vivas (sin ctrl)"],
            textposition="top center", marker=dict(size=10)
        ))
        fig_flujo.add_trace(go.Scatter(
            x=[pcc_ini_date], y=[ctrl_al_pcc], mode="markers+text", text=["Llegan vivas (con ctrl)"],
            textposition="bottom center", marker=dict(size=10)
        ))

    fig_flujo.update_layout(
        title="Flujo acumulado de individuos desde la siembra hasta el PCC",
        xaxis_title="Fecha", yaxis_title="Acumulado de malezas (pl·m²)",
        legend_title="Condición", margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_flujo, use_container_width=True)
    st.caption(
        "El área bajo la curva representa la cantidad acumulada de malezas desde la siembra. "
        "Los puntos en el inicio del PCC indican cuántas **llegan vivas** al periodo crítico."
    )
else:
    st.info("No se pudo generar el gráfico de flujo: AUC(EMERREL) no válido.")
















