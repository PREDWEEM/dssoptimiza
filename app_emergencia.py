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





