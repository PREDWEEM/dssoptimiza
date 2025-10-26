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
with st.sidebar:
    st.header("Periodo Crítico de Competencia (PCC)")
    usar_pcc = st.checkbox("Integrar densidad efectiva solo dentro del PCC", value=True)

    default_ini = pd.Timestamp(year=year_ref, month=10, day=10)
    default_fin = pd.Timestamp(year=year_ref, month=11, day=4)

    pcc_ini_date = pd.to_datetime(
        st.date_input("Inicio del PCC", value=default_ini.date(),
                      min_value=ts.min().date(), max_value=ts.max().date(),
                      disabled=not usar_pcc)
    )
    pcc_fin_date = pd.to_datetime(
        st.date_input("Fin del PCC", value=default_fin.date(),
                      min_value=ts.min().date(), max_value=ts.max().date(),
                      disabled=not usar_pcc)
    )

    if usar_pcc:
        st.caption(
            f"PCC activo: del **{pcc_ini_date.date()}** al **{pcc_fin_date.date()}** "
            f"({(pcc_fin_date - pcc_ini_date).days} días)"
        )
    else:
        st.caption("Integración sobre **todo el ciclo de cultivo**.")

mask_pcc = (ts >= pcc_ini_date) & (ts <= pcc_fin_date)


















