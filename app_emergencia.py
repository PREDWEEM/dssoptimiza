# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Cohortes SECUENCIALES · Optimización
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import itertools, random, math as _math

# ------------------ FUNCIÓN DE PÉRDIDA ------------------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ------------------ ESTADO UI ------------------
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · (1−Ciec) + AUC + Cohortes SECUENCIALES · Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ------------------ CONSTANTES ------------------
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10
EPS_REMAIN = 1e-9
EPS_EXCLUDE = 0.99

# ------------------ LECTURA CSV ------------------
def sniff_sep_dec(text):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get)
    dec_guess = "." if sample.count(".") >= sample.count(",") else ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw_from_url(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r: return r.read()

def read_raw(up, url):
    if up is not None: return up.read()
    if url: return read_raw_from_url(url)
    raise ValueError("No hay fuente de datos.")

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

def _to_days(ts):
    f = pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]")
    return ((f - f[0]).astype("timedelta64[D]") / np.timedelta64(1, "D")).astype(float)

def auc_time(fecha, y, mask=None):
    f = pd.to_datetime(fecha); y_arr = np.asarray(y, float)
    if mask is not None: f = f[mask]; y_arr = y_arr[mask]
    if len(f) < 2: return 0.0
    tdays = _to_days(f)
    y_arr = np.nan_to_num(y_arr)
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

# ------------------ CARGA ------------------
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria/acumulada)", type=["csv"])
    url = st.text_input("…o URL raw de GitHub", "")
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], 0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], 0)
    dayfirst = st.checkbox("Fecha dd/mm/yyyy", True)
    is_cumulative = st.checkbox("Mi CSV es acumulado", False)
    as_percent = st.checkbox("Valores en %", True)

if up is None and not url:
    st.info("Subí un CSV o pegá una URL para continuar."); st.stop()

raw = read_raw(up, url)
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
df_plot = pd.DataFrame({"fecha": pd.to_datetime(df["fecha"]), "EMERREL": emerrel})

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
Ciec = np.clip((LAI / 6.0) * (250 / 250), 0.0, 1.0)
one_minus_Ciec = 1 - Ciec

# ------------------ SECUENCIAL S1–S4 ------------------
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = ts.dt.date >= sow_date
births = np.where(mask_since_sow, emerrel, 0.0)
T12, T23, T34 = 10, 15, 20
S1, S2, S3, S4 = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
for i in range(len(births)):
    if i-T12>=0: S2[i]+=births[i-T12]; S1[i-T12]-=births[i-T12]
    if i-(T12+T23)>=0: S3[i]+=births[i-(T12+T23)]; S2[i-(T12+T23)]-=births[i-(T12+T23)]
    if i-(T12+T23+T34)>=0: S4[i]+=births[i-(T12+T23+T34)]; S3[i-(T12+T23+T34)]-=births[i-(T12+T23+T34)]
S1,S2,S3,S4 = [np.clip(x,0,None) for x in [S1,S2,S3,S4]]
emeac = np.cumsum(births)
total_states = S1+S2+S3+S4
scale = np.minimum(np.divide(emeac, np.clip(total_states,1e-9,None)),1.0)
S1*=scale; S2*=scale; S3*=scale; S4*=scale

FC_S = {"S1":0.1,"S2":0.3,"S3":0.6,"S4":1.0}
auc_cruda = auc_time(ts, emerrel, mask=mask_since_sow)
factor_area_to_plants = 250/auc_cruda if auc_cruda>0 else None

# ------------------ PCC CONFIG ------------------
with st.sidebar:
    st.header("Periodo Crítico de Competencia (PCC)")
    usar_pcc = st.checkbox("Integrar densidad efectiva solo dentro del PCC", value=False)
    pcc_ini_dias = st.number_input("Inicio PCC (días después siembra)", 0, 120, 50)
    pcc_fin_dias = st.number_input("Fin PCC (días después siembra)", 1, 180, 80)
pcc_ini_date = pd.Timestamp(sow_date + dt.timedelta(days=int(pcc_ini_dias)))
pcc_fin_date = pd.Timestamp(sow_date + dt.timedelta(days=int(pcc_fin_dias)))
mask_pcc = (ts >= pcc_ini_date) & (ts <= pcc_fin_date)

# ------------------ GRAFICO PRINCIPAL ------------------
st.subheader("📊 EMERREL + PCC")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=emerrel, name="EMERREL"))
if usar_pcc:
    fig.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, line_width=0,
                  fillcolor="rgba(255,215,0,0.25)", opacity=0.25,
                  annotation_text="PCC", annotation_position="top left",
                  annotation=dict(font_size=12, font_color="black"))
fig.update_layout(title="Emergencia y PCC", xaxis_title="Fecha", yaxis_title="EMERREL (0–1)")
st.plotly_chart(fig, use_container_width=True)


# ===============================================================
# 🧪 MANEJO + CÁLCULOS DE x₂/x₃ + GRÁFICOS + OPTIMIZACIÓN (PCC-aware)
# ===============================================================

# ---------- Parámetros A2 / cap ----------
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva A₂ (pl·m²)", [250, 125, 62], index=0))

# Si el Bloque 1 fijó factor_area_to_plants con 250, lo re-ajustamos al cap elegido
if auc_cruda and auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
else:
    factor_area_to_plants = None

# ---------- Aportes por estado (en plantas·m²/día) ----------
fechas_d = ts.dt.date.values
ms = (ts.dt.date >= sow_date).to_numpy()

if factor_area_to_plants is not None:
    S1_pl = np.where(ms, S1 * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants, 0.0)
    S2_pl = np.where(ms, S2 * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants, 0.0)
    S3_pl = np.where(ms, S3 * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants, 0.0)
    S4_pl = np.where(ms, S4 * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants, 0.0)

    base_pl_daily = np.where(ms, emerrel * factor_area_to_plants, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, ms)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=None
    base_pl_daily_cap = None

# ---------- UI de manejo ----------
with st.sidebar:
    st.header("Manejo (manual)")
    pre_glifo   = st.checkbox("Pre: glifosato (no residual, 1d)", False)
    pre_glifo_date = st.date_input("Fecha glifosato", value=ts.min().date(),
                                   min_value=ts.min().date(), max_value=ts.max().date(), disabled=not pre_glifo)

    pre_selNR   = st.checkbox("Pre: selectivo no residual (NR, 10d)", False)
    pre_selNR_date = st.date_input("Fecha selectivo NR", value=ts.min().date(),
                                   min_value=ts.min().date(), max_value=ts.max().date(), disabled=not pre_selNR)

    preR        = st.checkbox("Presiembra + residual (≤ siembra−14; S1–S2)", False)
    preR_days   = st.slider("Residualidad presiembra (d)", 15, 120, 14, 1, disabled=not preR)
    preR_last   = sow_date - dt.timedelta(days=14)
    preR_date   = st.date_input("Fecha presiembraR", value=min(ts.min().date(), preR_last),
                                min_value=ts.min().date(), max_value=min(ts.max().date(), preR_last),
                                disabled=not preR)

    preemR      = st.checkbox("Preemergente + residual (siembra..siembra+10; S1–S2)", False)
    preemR_days = st.slider("Residualidad preemergente (d)", 15, 120, 45, 1, disabled=not preemR)
    preem_min   = sow_date
    preem_max   = min(ts.max().date(), sow_date + dt.timedelta(days=10))
    preemR_date = st.date_input("Fecha preemR", value=preem_min, min_value=preem_min, max_value=preem_max, disabled=not preemR)

    post_selR   = st.checkbox("Post + residual (≥ siembra+20; S1–S4)", False)
    post_res_d  = st.slider("Residualidad post (d)", 30, 120, 45, 1, disabled=not post_selR)
    post_min    = max(ts.min().date(), sow_date + dt.timedelta(days=20))
    post_selR_date = st.date_input("Fecha postR", value=post_min, min_value=post_min, max_value=ts.max().date(), disabled=not post_selR)

    post_gram   = st.checkbox("Post: graminicida (NR, día 0..+10; S1–S3)", False)
    post_gram_date = st.date_input("Fecha graminicida", value=max(ts.min().date(), sow_date),
                                   min_value=ts.min().date(), max_value=ts.max().date(), disabled=not post_gram)

with st.sidebar:
    st.header("Eficiencias (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider("Selectivo NR (pre, 10d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Presiembra + residual", 0, 100, 90, 1) if preR else 0
    ef_preemR      = st.slider("Preemergente + residual", 0, 100, 90, 1) if preemR else 0
    ef_post_selR   = st.slider("Post + residual", 0, 100, 90, 1) if post_selR else 0
    ef_post_gram   = st.slider("Graminicida post", 0, 100, 90, 1) if post_gram else 0

with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2)/half_life
    else:
        lam_exp = None

def weights_one_day(date_val):
    if not date_val: return np.zeros_like(fechas_d, float)
    return ((fechas_d == date_val)).astype(float)

def weights_residual(start_date, dias):
    w = np.zeros_like(fechas_d, float)
    if (not start_date) or (not dias) or (int(dias) <= 0): return w
    d0 = start_date; d1 = start_date + dt.timedelta(days=int(dias))
    mask = (fechas_d >= d0) & (fechas_d < d1)
    if not mask.any(): return w
    idxs = np.where(mask)[0]
    t_rel = np.arange(len(idxs), dtype=float)
    if decaimiento_tipo == "Ninguno":
        w[idxs] = 1.0
    elif decaimiento_tipo == "Lineal":
        L = max(1, len(idxs)); w[idxs] = 1.0 - (t_rel / max(1.0, L - 1))
    else:
        w[idxs] = np.exp(-lam_exp * t_rel) if lam_exp is not None else 1.0
    return w

# ---------- Aplicación de controles con gateo y jerarquía ----------
if factor_area_to_plants is not None:
    c1 = np.ones_like(fechas_d, float)
    c2 = np.ones_like(fechas_d, float)
    c3 = np.ones_like(fechas_d, float)
    c4 = np.ones_like(fechas_d, float)

    def apply_eff(weights, eff_pct, states):
        if eff_pct <= 0: return
        reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    def remaining_in_window(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1_pl * c1 * w)
        if "S2" in states: rem += np.sum(S2_pl * c2 * w)
        if "S3" in states: rem += np.sum(S3_pl * c3 * w)
        if "S4" in states: rem += np.sum(S4_pl * c4 * w)
        return float(rem)

    def eff_acc(prev_eff, this_eff):  # independencia probabilística
        return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    eff_pre = 0.0
    eff_pre2 = 0.0
    eff_all = 0.0

    # 1) Presiembra residual (S1–S2)
    if preR:
        w_preR = weights_residual(preR_date, preR_days)
        if remaining_in_window(w_preR, ["S1","S2"]) > 1e-9 and ef_preR > 0:
            apply_eff(w_preR, ef_preR, ["S1","S2"])
            eff_pre = eff_acc(0.0, ef_preR/100.0)

    # 2) Preemergente residual (S1–S2) si acumulado < 99%
    if preemR and (eff_pre < 0.99):
        w_preem = weights_residual(preemR_date, preemR_days)
        if remaining_in_window(w_preem, ["S1","S2"]) > 1e-9 and ef_preemR > 0:
            apply_eff(w_preem, ef_preemR, ["S1","S2"])
            eff_pre2 = eff_acc(eff_pre, ef_preemR/100.0)
        else:
            eff_pre2 = eff_pre
    else:
        eff_pre2 = eff_pre

    # 3) Pre no residuales (afectan masa pero no la jerarquía)
    if pre_selNR and ef_pre_selNR > 0:
        w = weights_residual(pre_selNR_date, 10)
        if remaining_in_window(w, ["S1","S2","S3","S4"]) > 1e-9:
            apply_eff(w, ef_pre_selNR, ["S1","S2","S3","S4"])
    if pre_glifo and ef_pre_glifo > 0:
        w = weights_one_day(pre_glifo_date)
        if remaining_in_window(w, ["S1","S2","S3","S4"]) > 1e-9:
            apply_eff(w, ef_pre_glifo, ["S1","S2","S3","S4"])

    # 4) Post residual (S1–S4) si acumulado < 99%
    if post_selR and (eff_pre2 < 0.99):
        w_post = weights_residual(post_selR_date, post_res_d)
        if remaining_in_window(w_post, ["S1","S2","S3","S4"]) > 1e-9 and ef_post_selR > 0:
            apply_eff(w_post, ef_post_selR, ["S1","S2","S3","S4"])
            eff_all = eff_acc(eff_pre2, ef_post_selR/100.0)
        else:
            eff_all = eff_pre2
    else:
        eff_all = eff_pre2

    # 5) Graminicida post (S1–S3) si acumulado < 99%
    if post_gram and (eff_all < 0.99) and ef_post_gram > 0:
        w = weights_residual(post_gram_date, 11)
        if remaining_in_window(w, ["S1","S2","S3"]) > 1e-9:
            apply_eff(w, ef_post_gram, ["S1","S2","S3"])

    # Series con control
    S1_pl_ctrl = S1_pl * c1
    S2_pl_ctrl = S2_pl * c2
    S3_pl_ctrl = S3_pl * c3
    S4_pl_ctrl = S4_pl * c4

    plantas_supresion      = S1_pl + S2_pl + S3_pl + S4_pl
    plantas_supresion_ctrl = S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl

    # Cap por A₂
    plantas_supresion_cap      = np.minimum(plantas_supresion,      base_pl_daily_cap)
    plantas_supresion_ctrl_cap = np.minimum(plantas_supresion_ctrl, plantas_supresion_cap)

else:
    plantas_supresion_cap = plantas_supresion_ctrl_cap = np.full(len(ts), np.nan)

# ---------- x₂ / x₃ con o sin PCC ----------
if factor_area_to_plants is not None:
    if usar_pcc:
        mask_eval = mask_pcc
        periodo_txt = f"PCC: {pcc_ini_date.date()} → {pcc_fin_date.date()}"
    else:
        mask_eval = ms
        periodo_txt = "Todo el ciclo (desde siembra)"

    X2 = float(np.nansum(plantas_supresion_cap[mask_eval]))
    X3 = float(np.nansum(plantas_supresion_ctrl_cap[mask_eval]))

    # A2 por AUC consistente con la ventana elegida
    sup_equiv  = np.divide(plantas_supresion_cap,     factor_area_to_plants, out=np.zeros_like(plantas_supresion_cap),     where=(factor_area_to_plants>0))
    supc_equiv = np.divide(plantas_supresion_ctrl_cap, factor_area_to_plants, out=np.zeros_like(plantas_supresion_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_eval)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_eval)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / max(1e-9, auc_cruda)))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / max(1e-9, auc_cruda)))
else:
    X2 = X3 = A2_sup_final = A2_ctrl_final = float("nan")
    periodo_txt = "Sin datos válidos"

# ---------- Gráfico EMERREL + plantas semanales + Ciec + PCC ----------
st.subheader("📊 EMERREL + Plantas·m²·semana (cap A₂) + PCC")
df_daily_cap = pd.DataFrame({"fecha": ts,
                             "sin_ctrl": plantas_supresion_cap,
                             "con_ctrl": plantas_supresion_ctrl_cap})
df_week = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()

fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=ts, y=emerrel, name="EMERREL (izq)", mode="lines"))
fig_main.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["sin_ctrl"], name="pl·m²·sem sin control", mode="lines+markers", yaxis="y2"))
fig_main.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["con_ctrl"], name="pl·m²·sem con control", mode="lines+markers", yaxis="y2"))
fig_main.add_trace(go.Scatter(x=ts, y=1.0-one_minus_Ciec, name="Ciec (der2)", mode="lines", yaxis="y3"))

# Sombreado PCC
if usar_pcc:
    fig_main.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, line_width=0,
                       fillcolor="rgba(255,215,0,0.25)", opacity=0.25,
                       annotation_text="PCC", annotation_position="top left",
                       annotation=dict(font_size=12, font_color="black"))

fig_main.update_layout(
    title=f"Tope A₂={int(MAX_PLANTS_CAP)} — Ventana: {periodo_txt}",
    xaxis=dict(title="Fecha"),
    yaxis=dict(title="EMERREL (0–1)"),
    yaxis2=dict(overlaying="y", side="right", title="pl·m²·sem", range=[0, max(100, 1.2*max(1.0, np.nanmax(df_week[["sin_ctrl","con_ctrl"]].to_numpy())))], tick0=0),
    yaxis3=dict(overlaying="y", side="right", position=0.97, title="Ciec (0–1)", range=[0,1]),
    margin=dict(l=10, r=10, t=50, b=10)
)
st.plotly_chart(fig_main, use_container_width=True)

st.subheader("Densidad efectiva y A₂")
st.markdown(
    f"**x₂ (sin control):** {X2:,.1f} pl·m²  \n"
    f"**x₃ (con control):** {X3:,.1f} pl·m²  \n"
    f"**A₂ (sin ctrl):** {A2_sup_final:.1f} pl·m² · **A₂ (con ctrl):** {A2_ctrl_final:.1f} pl·m²"
)
st.caption(f"Ventana usada para integrar: **{periodo_txt}**")

# ===============================================================
# 🧠 OPTIMIZACIÓN (Grid / Aleatorio / Recocido) — PCC-aware
# ===============================================================
st.markdown("---")
st.header("🧠 Optimización de siembra + calendario de controles")

with st.sidebar:
    st.header("Optimización — variables habilitadas")
    sow_from = st.date_input("Siembra desde", value=sow_date)
    sow_to   = st.date_input("Siembra hasta",  value=min(sow_date + dt.timedelta(days=20), ts.max().date()))
    sow_step = st.number_input("Paso siembra (días)", 1, 30, 2, 1)

    use_preR_opt     = st.checkbox("Usar presiembra + residual", True)
    use_preemR_opt   = st.checkbox("Usar preemergente + residual", True)
    use_post_selR_opt= st.checkbox("Usar post + residual", True)
    use_post_gram_opt= st.checkbox("Usar graminicida post", True)

    ef_preR_opt      = st.slider("Eficiencia presiembraR (%)", 0, 100, 90, 1) if use_preR_opt else 0
    ef_preemR_opt    = st.slider("Eficiencia preemR (%)",     0, 100, 90, 1) if use_preemR_opt else 0
    ef_post_selR_opt = st.slider("Eficiencia postR (%)",      0, 100, 90, 1) if use_post_selR_opt else 0
    ef_post_gram_opt = st.slider("Eficiencia gram post (%)",  0, 100, 90, 1) if use_post_gram_opt else 0

    preR_back   = st.number_input("PresiembraR: hasta X días antes siembra", 14, 120, 14, 1)
    preem_step  = st.number_input("Paso fechas preemergente (días)", 1, 10, 2, 1)
    post_max_d  = st.number_input("Post: máx días después siembra", 20, 180, 60, 1)
    post_step   = st.number_input("Paso fechas post (días)", 1, 30, 2, 1)
    res_min, res_max = st.slider("Residualidad (min–max) [días]", 15, 120, (40, 45), 5)
    res_step = st.number_input("Paso residualidad", 1, 30, 5, 1)

    optimizer = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100)
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1)

    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 50000, 5000, 100)
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("Enfriamiento γ", 0.80, 0.9999, 0.995, 0.0001)

    c1b, c2b = st.columns(2)
    with c1b:
        start_clicked = st.button("▶️ Iniciar optimización", use_container_width=True, disabled=st.session_state.opt_running)
    with c2b:
        stop_clicked  = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running)
    if start_clicked:
        st.session_state.opt_stop = False
        st.session_state.opt_running = True
    if stop_clicked:
        st.session_state.opt_stop = True

# --------- Helpers de optimización ---------
def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur=cur+pd.Timedelta(days=int(step_days))
    return out

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=14)
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(res_step))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd); end = pd.to_datetime(sd) + pd.Timedelta(days=10)
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

res_days = list(range(int(res_min), int(res_max)+1, int(res_step)))
if int(res_max) not in res_days: res_days.append(int(res_max))

sow_candidates = daterange(sow_from, sow_to, sow_step)

def act_presiembraR(date_val, R, eff): return {"kind":"preR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_preemR(date_val, R, eff):     return {"kind":"preemR",  "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_post_selR(date_val, R, eff):  return {"kind":"postR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}
def act_post_gram(date_val, eff):     return {"kind":"post_gram","date":pd.to_datetime(date_val).date(), "days":11, "eff":eff, "states":["S1","S2","S3"]}

def compute_ciec_for(sd):
    days = np.array([(pd.Timestamp(d).date() - sd).days for d in ts])
    def logistic_between(days, start, end, y_max):
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        fc = y_max/(1.0+np.exp(-r*(days-t_mid)))
        return np.clip(fc,0.0,1.0)
    fc_dyn = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100))
    LAI_loc = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
    LAI_loc = np.clip(LAI_loc, 0.0, lai_max)
    Ciec_loc = np.clip(LAI_loc/6.0, 0.0, 1.0)  # Ca/Cs=1 para optimización local
    return 1.0 - Ciec_loc

def recompute_for_sow(sd):
    mask_since_loc = (ts.dt.date >= sd).to_numpy()
    one_minus = compute_ciec_for(sd)
    births = np.where(mask_since_loc, emerrel, 0.0)

    # Reconstrucción de estados secuenciales con T12/23/34 iguales a Bloque 1
    S1x, S2x, S3x, S4x = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
    for i in range(len(births)):
        if i-10>=0: S2x[i]+=births[i-10]; S1x[i-10]-=births[i-10]
        if i-(10+15)>=0: S3x[i]+=births[i-(10+15)]; S2x[i-(10+15)]-=births[i-(10+15)]
        if i-(10+15+20)>=0: S4x[i]+=births[i-(10+15+20)]; S3x[i-(10+15+20)]-=births[i-(10+15+20)]
    S1x,S2x,S3x,S4x = [np.clip(x,0,None) for x in [S1x,S2x,S3x,S4x]]
    tot = S1x+S2x+S3x+S4x
    scale = np.minimum(np.divide(np.cumsum(births), np.clip(tot,1e-9,None)), 1.0)
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

def evaluate(sd, schedule):
    sow = pd.to_datetime(sd)
    sow_plus20 = sow + pd.Timedelta(days=20)

    # Reglas duras
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR" and d < sow_plus20: return None
        if a["kind"] == "preR"  and d > (sow - pd.Timedelta(days=14)): return None
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=10))): return None

    env = recompute_for_sow(sd)
    if env is None: return None

    mask_since = env["mask_since"]; factor_area = env["factor_area"]
    S1p, S2p, S3p, S4p = env["S_pl"]; sup_cap = env["sup_cap"]
    ts_local, fechas_d_local = env["ts"], env["fechas_d"]

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
            if _remaining(w, ["S1","S2"]) > 1e-9 and a["eff"] > 0:
                _apply(w, a["eff"], ["S1","S2"])
                eff_pre = _eff(0.0, a["eff"]/100.0)

        elif a["kind"] == "preemR":
            if eff_pre < 0.99 and a["eff"] > 0:
                if _remaining(w, ["S1","S2"]) > 1e-9:
                    _apply(w, a["eff"], ["S1","S2"])
                    eff_pre2 = _eff(eff_pre, a["eff"]/100.0)
                else:
                    eff_pre2 = eff_pre
            else:
                eff_pre2 = eff_pre

        elif a["kind"] == "postR":
            if eff_pre2 < 0.99 and a["eff"] > 0:
                if _remaining(w, ["S1","S2","S3","S4"]) > 1e-9:
                    _apply(w, a["eff"], ["S1","S2","S3","S4"])
                    eff_all = _eff(eff_pre2, a["eff"]/100.0)
                else:
                    eff_all = eff_pre2
            else:
                eff_all = eff_pre2

        elif a["kind"] == "post_gram":
            if eff_all < 0.99 and a["eff"] > 0 and _remaining(w, ["S1","S2","S3"]) > 1e-9:
                _apply(w, a["eff"], ["S1","S2","S3"])

    tot_ctrl = S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    # Máscara de evaluación PCC-aware
    if usar_pcc:
        pcc_i = pd.Timestamp(sd + dt.timedelta(days=int(pcc_ini_dias)))
        pcc_f = pd.Timestamp(sd + dt.timedelta(days=int(pcc_fin_dias)))
        mask_eval = (ts_local >= pcc_i) & (ts_local <= pcc_f)
    else:
        mask_eval = mask_since

    X2loc = float(np.nansum(sup_cap[mask_eval]))
    X3loc = float(np.nansum(plantas_ctrl_cap[mask_eval]))
    loss3 = _loss(X3loc)

    # A2 consistente con la ventana
    auc_sup      = auc_time(ts_local, np.divide(sup_cap,          factor_area, out=np.zeros_like(sup_cap),          where=(factor_area>0)), mask=mask_eval)
    auc_sup_ctrl = auc_time(ts_local, np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0)), mask=mask_eval)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/ max(1e-9, env["auc_cruda"])))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/ max(1e-9, env["auc_cruda"])))

    return {"sow": sd, "loss_pct": float(loss3), "x2": X2loc, "x3": X3loc, "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

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

# --------- Ejecutar optimización ---------
status_ph = st.empty()
prog_ph = st.empty()
results = []

if factor_area_to_plants is None:
    st.info("Necesitás AUC(EMERREL) > 0 para optimizar.")
else:
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

        else:  # Recocido
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

# --------- Reporte del mejor escenario ---------
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
        if _remaining(w, a["states"]) > 1e-9 and a["eff"] > 0:
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

# Mostrar mejores
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
        # Sombreado PCC sobre el mejor
        if usar_pcc:
            pcc_i_b = pd.Timestamp(pd.to_datetime(best["sow"]).date() + dt.timedelta(days=int(pcc_ini_dias)))
            pcc_f_b = pd.Timestamp(pd.to_datetime(best["sow"]).date() + dt.timedelta(days=int(pcc_fin_dias)))
            fig_best.add_vrect(x0=pcc_i_b, x1=pcc_f_b, line_width=0,
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

        # Pérdida vs x
        X2_b = float(np.nansum(envb["sup_cap_b"][(ts >= (pcc_i_b if usar_pcc else ts.min())) & (ts <= (pcc_f_b if usar_pcc else ts.max()))]))
        X3_b = float(np.nansum(envb["ctrl_cap_b"][(ts >= (pcc_i_b if usar_pcc else ts.min())) & (ts <= (pcc_f_b if usar_pcc else ts.max()))]))
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400); y_curve = _loss(x_curve)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida (%) vs x"))
        fig_loss.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", name="x₂", text=[f"x₂={X2_b:.1f}"], textposition="top center"))
        fig_loss.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", name="x₃", text=[f"x₃={X3_b:.1f}"], textposition="top right"))
        fig_loss.update_layout(title="Pérdida de rendimiento (%) vs densidad efectiva", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
        st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Aún no hay resultados de optimización para mostrar.")



