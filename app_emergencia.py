# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Cohortes SECUENCIALES · Optimización
# ===============================================================
# Reglas agronómicas:
# - Presiembra selectivo residual (preR): SOLO ≤ siembra−14 (actúa S1–S2)
# - Preemergente selectivo residual (preemR): [siembra, siembra+10] (S1–S2)
# - Post-residual (postR): ≥ siembra+20 (actúa S1–S3)
# - Graminicida post: ventana siembra..siembra+10 (S1–S4) y ≥14 días después del postR
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta
import itertools, random, math as _math

# ------------------ FUNCIÓN DE PÉRDIDA (hiperbólica) ------------------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ------------------ ESTADO UI ------------------
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop"    not in st.session_state: st.session_state.opt_stop = False

APP_TITLE = "🌾 PREDWEEM · (1−Ciec) + AUC + Cohortes SECUENCIALES · Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ------------------ CONSTANTES ------------------
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 10
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10
EPS_REMAIN  = 1e-9
EPS_EXCLUDE = 0.99
PCC_REF_MONTH = 10
PCC_REF_DAY   = 10   # 🔧 fecha de referencia para la pérdida (10-oct)

# ------------------ UTILIDADES DE LECTURA ------------------
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "." if sample.count(".") >= sample.count(",") else ","
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample): dec_guess = ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw_from_url(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r: return r.read()

def read_raw(up, url):
    if up is not None: return up.read()
    if url: return read_raw_from_url(url)
    raise ValueError("No hay fuente de datos.")

def parse_csv(raw, sep_opt, dec_opt, encoding="utf-8"):
    head = raw[:8000].decode("utf-8", errors="ignore")
    sep_guess, dec_guess = sniff_sep_dec(head)
    sep = sep_guess if sep_opt == "auto" else ("," if sep_opt=="," else (";" if sep_opt==";" else "\t"))
    dec = dec_guess if dec_opt == "auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python")
    return df, {"sep": sep, "dec": dec, "enc": encoding}

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if": return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%","",regex=False)
    if decimal == ",": t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else:              t = t.str.replace(",","",regex=False)
    return pd.to_numeric(t, errors="coerce")

def _to_days(ts: pd.Series) -> np.ndarray:
    f = pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]")
    t_ns = f.astype("int64")
    return ((t_ns - t_ns[0]) / 1e9 / 86400.0).astype(float)

def auc_time(fecha: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fecha); y_arr = np.asarray(y, dtype=float)
    if mask is not None: f = f[mask]; y_arr = y_arr[mask]
    if len(f) < 2: return 0.0
    tdays = _to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, tdays))

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y); cum = 0.0
    for i in range(len(y)):
        if bool(active_mask[i]):
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val; cum += val
        else: out[i] = 0.0
    return out

# ------------------ CARGA CSV ------------------
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria o EMERAC)", type=["csv"])
    url = st.text_input("…o URL raw de GitHub", placeholder="https://raw.githubusercontent.com/usuario/repo/main/emer.csv")
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha: dd/mm/yyyy", True)
    is_cumulative = st.checkbox("Mi CSV es acumulado (EMERAC)", False)
    as_percent = st.checkbox("Valores en % (no 0–1)", True)
    dedup = st.selectbox("Si hay fechas duplicadas…", ["sumar","promediar","primera"], 0)
    fill_gaps = st.checkbox("Rellenar días faltantes con 0", False)

if up is None and not url:
    st.info("Subí un CSV o pegá una URL para continuar."); st.stop()

try:
    raw = read_raw(up, url)
    if not raw or len(raw) == 0: st.error("El archivo/URL está vacío."); st.stop()
    df0, meta = parse_csv(raw, sep_opt, dec_opt)
    if df0.empty: st.error("El CSV no tiene filas."); st.stop()
    st.success(f"CSV leído. sep='{meta['sep']}' dec='{meta['dec']}' enc='{meta['enc']}'")
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}"); st.stop()

cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, index=0)
c_valor = st.selectbox("Columna de valor", cols, index=1 if len(cols)>1 else 0)

fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = clean_numeric_series(df0[c_valor])
df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
if is_cumulative: df["valor"] = df["valor"].diff().fillna(0)
if as_percent:    df["valor"] /= 100.0

emerrel = df["valor"].clip(lower=0)
df_plot = pd.DataFrame({"fecha": df["fecha"], "EMERREL": emerrel})

# ------------------ SIEMBRA & CANOPIA ------------------
years = df_plot["fecha"].dt.year
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1)
sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag   = st.number_input("Días a emergencia del cultivo", 0, 60, 7)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 80)
    cov_max = st.number_input("Cobertura máxima (%)", 10, 100, 80)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.0)
    k_beer  = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6)
    use_ciec = st.checkbox("Incluir Ciec", True)
    Ca = st.number_input("Ca", 50, 700, 250)
    Cs = st.number_input("Cs", 50, 700, 250)
    LAIhc = st.number_input("LAIhc- Habilidad competitiva", 3.0, 6.0, 3.0)

def compute_canopy(fechas, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer):
    days_since = np.array([(pd.Timestamp(f).date()-sow_date).days for f in fechas])
    def logistic(days, start, end, y_max):
        if end<=start: end=start+1
        t_mid=0.5*(start+end); r=4.0/max(1.0,(end-start))
        return y_max/(1+np.exp(-r*(days-t_mid)))
    if mode_canopy=="Cobertura dinámica (%)":
        fc=np.where(days_since<t_lag,0,logistic(days_since,t_lag,t_close,cov_max/100))
        fc=np.clip(fc,0,1); LAI=-np.log(np.clip(1-fc,1e-9,1))/k_beer; LAI=np.clip(LAI,0,lai_max)
    else:
        LAI=np.where(days_since<t_lag,0,logistic(days_since,t_lag,t_close,lai_max))
        LAI=np.clip(LAI,0,lai_max); fc=1-np.exp(-k_beer*LAI)
    return fc,LAI

FC,LAI=compute_canopy(df_plot["fecha"],sow_date,mode_canopy,int(t_lag),int(t_close),float(cov_max),float(lai_max),float(k_beer))
if use_ciec:
    Ciec=(LAI/LAIhc)*(Ca/Cs); Ciec=np.clip(Ciec,0,1)
else:
    Ciec=np.zeros_like(LAI)
one_minus_Ciec=1-Ciec

# ------------------ PARÁMETROS DE ESCENARIO ------------------
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva (pl·m²)", [250, 125, 62], index=0))
st.caption(f"AUC(EMERREL cruda) ≙ A2 **= {int(MAX_PLANTS_CAP)} pl·m²**. Cohortes S1..S4 **SECUENCIALES**.")

# ------------------ ESTADOS SECUENCIALES (S1→S4) ------------------
ts = pd.to_datetime(df_plot["fecha"])
sow_ref = st.session_state.get("sow_date_cache", None)
if sow_ref is not None:
    mask_since_sow = ts.dt.date >= sow_ref
else:
    mask_since_sow = pd.Series(True, index=ts.index)
st.session_state["sow_date_cache"] = sow_date

births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where((ts.dt.date >= sow_date).to_numpy(), births, 0.0)

T12 = st.sidebar.number_input("Duración S1→S2 (días)", 1, 60, 5, 1)
T23 = st.sidebar.number_input("Duración S2→S3 (días)", 1, 60, 20, 1)
T34 = st.sidebar.number_input("Duración S3→S4 (días)", 1, 60, 26, 1)

S1 = births.copy(); S2 = np.zeros_like(births); S3 = np.zeros_like(births); S4 = np.zeros_like(births)
for i in range(len(births)):
    if i - int(T12) >= 0:
        moved = births[i - int(T12)]; S1[i - int(T12)] -= moved; S2[i] += moved
    if i - (int(T12) + int(T23)) >= 0:
        moved = births[i - (int(T12) + int(T23))]; S2[i - (int(T12) + int(T23))] -= moved; S3[i] += moved
    if i - (int(T12) + int(T23) + int(T34)) >= 0:
        moved = births[i - (int(T12) + int(T23) + int(T34))]; S3[i - (int(T12) + int(T23) + int(T34))] -= moved; S4[i] += moved

S1 = np.clip(S1, 0.0, None); S2 = np.clip(S2, 0.0, None); S3 = np.clip(S3, 0.0, None); S4 = np.clip(S4, 0.0, None)
total_states = S1 + S2 + S3 + S4
emeac = np.cumsum(births)
scale = np.divide(np.clip(emeac, 1e-9, None), np.clip(total_states, 1e-9, None))
scale = np.minimum(scale, 1.0)
S1 *= scale; S2 *= scale; S3 *= scale; S4 *= scale

FC_S = {"S1": 0.1, "S2": 0.3, "S3": 0.6, "S4": 1.0}

auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=(ts.dt.date >= sow_date))
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."
st.caption(conv_caption)

# ------------------ MANEJO DE HERBICIDAS (UI) ------------------
sched_rows = []
def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

fechas_d = ts.dt.date.values
min_date = ts.min().date()
max_date = ts.max().date()

# PRE-SIEMBRA
with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    pre_glifo = st.checkbox("Herbicida total (glifosato)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=min_date, min_value=min_date,
                                   max_value=max_date, disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=min_date, min_value=min_date,
                                   max_value=max_date, disabled=not pre_selNR)

    preR = st.checkbox("Selectivo + residual (presiembra)", value=False,
                       help="Solo permitido hasta siembra−14 días. Actúa S1.")
    preR_days = st.slider("Residualidad presiembra (días)", 15, 120, 14, 1, disabled=not preR)
    preR_max = sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    preR_date = st.date_input("Fecha selectivo + residual (presiembra)",
                              value=min(min_date, preR_max),
                              min_value=min_date,
                              max_value=min(preR_max, max_date),
                              disabled=not preR)

# PRE-EMERGENTE
with st.sidebar:
    st.header("Manejo pre-emergente (manual)")
    preemR = st.checkbox("Selectivo + residual (preemergente)", value=False,
                         help="Ventana [siembra, siembra+10]. Actúa S1.")
    preemR_days = st.slider("Residualidad preemergente (días)", 15, 120, 45, 1, disabled=not preemR)
    preem_min = sow_date
    preem_max = min(max_date, sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preemR_date = st.date_input("Fecha selectivo + residual (preemergente)",
                                value=preem_min, min_value=preem_min,
                                max_value=preem_max, disabled=not preemR)

# POST-EMERGENTE
with st.sidebar:
    st.header("Manejo post-emergencia (manual)")
    post_selR = st.checkbox("Selectivo + residual (post)", value=False, help="≥ siembra + 20 días. Actúa S1–S2.")
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)",
                                   value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

    post_gram = st.checkbox("Selectivo graminicida (post)", value=False, help="Actúa S1–S3. ≥ 14 d después del postR.")
    default_gram_date = (post_selR_date + timedelta(days=14)) if post_selR else max(min_date, sow_date)
    post_gram_date = st.date_input("Fecha graminicida (post)",
                                   value=default_gram_date, min_value=min_date, max_value=max_date, disabled=not post_gram)

# VALIDACIONES
warnings = []
def check_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debería ser ≤ siembra ({sow_date}).")
def check_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debería ser ≥ siembra ({sow_date}).")

if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if preR and preR_date > (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
    warnings.append(f"Presiembra residual ≤ siembra−{PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW}.")
if preemR and (preemR_date < sow_date or preemR_date > sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
    warnings.append(f"Preemergente residual entre siembra y siembra+{PREEM_R_MAX_AFTER_SOW_DAYS}.")
if post_selR and post_selR_date < sow_date + timedelta(days=20):
    warnings.append(f"Post residual ≥ {(sow_date + timedelta(days=20))}.")
if post_gram and post_selR and pd.to_datetime(post_gram_date) < (pd.to_datetime(post_selR_date) + pd.Timedelta(days=14)):
    warnings.append("El graminicida debe aplicarse ≥ 14 d después del postR.")
for w in warnings: st.warning(w)

# CRONOGRAMA RESUMIDO (opcional visual)
if pre_glifo: add_sched("Pre·Glifosato (NSr, 1 d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched("Pre·Selectivo NR",        pre_selNR_date, NR_DAYS_DEFAULT, f"NR {NR_DAYS_DEFAULT} d")
if preR:      add_sched("Pre-siembra + residual",  preR_date,       preR_days,      f"Protege {preR_days} d (S1)")
if preemR:    add_sched("Pre-emergente + residual",preemR_date,     preemR_days,    f"Protege {preemR_days} d (S1)")
if post_selR: add_sched("Post + residual",         post_selR_date,  post_res_dias,  f"Protege {post_res_dias} d (S1–S2)")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post graminicida (NR,+10 d)","Inicio": str(ini), "Fin": str(fin),"Nota": "S1–S3; ≥ 14 d postR"})
sched = pd.DataFrame(sched_rows)

# EFICIENCIAS
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1 d)",           0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo NR (pre,{NR_DAYS_DEFAULT} d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Presiembra residual",            0, 100, 90, 1) if preR else 0
    ef_preemR      = st.slider("Pre-emergente residual",         0, 100, 90, 1) if preemR else 0
    ef_post_selR   = st.slider("Post residual",                  0, 100, 90, 1) if post_selR else 0
    ef_post_gram   = st.slider("Graminicida (+10 d)",            0, 100, 90, 1) if post_gram else 0

# DECAIMIENTO
with st.sidebar:
    st.header("Decaimiento de residuales")
    decaimiento_tipo = st.selectbox("Tipo", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1); lam_exp = math.log(2)/max(1e-6,half_life)
    else: lam_exp = None

def weights_one_day(date_val):
    if not date_val: return np.zeros_like(fechas_d,float)
    d0=date_val
    return ((fechas_d>=d0)&(fechas_d<(d0+timedelta(days=1)))).astype(float)

def weights_residual(start_date,dias):
    w=np.zeros_like(fechas_d,float)
    if (not start_date) or (not dias) or (int(dias)<=0): return w
    d0=start_date; d1=start_date+timedelta(days=int(dias))
    mask=(fechas_d>=d0)&(fechas_d<d1)
    if not mask.any(): return w
    idxs=np.where(mask)[0]; t_rel=np.arange(len(idxs),dtype=float)
    if decaimiento_tipo=="Ninguno": w[idxs]=1.0
    elif decaimiento_tipo=="Lineal":
        L=max(1,len(idxs)); w[idxs]=1.0-(t_rel/max(1.0,L-1))
    else:
        w[idxs]=np.exp(-lam_exp*t_rel) if lam_exp is not None else 1.0
    return w

# ------------------ APLICACIÓN DE EFICIENCIAS + SUPRESIÓN ------------------
if factor_area_to_plants is not None:
    ms=(ts.dt.date>=sow_date).to_numpy()
    one_minus=(1.0-(Ciec if use_ciec else 0.0)).astype(float)

    S1_pl=np.where(ms,S1*one_minus*FC_S["S1"]*factor_area_to_plants,0.0)
    S2_pl=np.where(ms,S2*one_minus*FC_S["S2"]*factor_area_to_plants,0.0)
    S3_pl=np.where(ms,S3*one_minus*FC_S["S3"]*factor_area_to_plants,0.0)
    S4_pl=np.where(ms,S4*one_minus*FC_S["S4"]*factor_area_to_plants,0.0)

    ctrl_S1=ctrl_S2=ctrl_S3=ctrl_S4=np.ones_like(fechas_d,float)

    def apply_efficiency_per_state(weights,eff_pct,states_sel):
        if eff_pct<=0 or (not states_sel): return
        reduc=np.clip(1.0-(eff_pct/100.0)*np.clip(weights,0.0,1.0),0.0,1.0)
        if "S1" in states_sel: np.multiply(ctrl_S1,reduc,out=ctrl_S1)
        if "S2" in states_sel: np.multiply(ctrl_S2,reduc,out=ctrl_S2)
        if "S3" in states_sel: np.multiply(ctrl_S3,reduc,out=ctrl_S3)
        if "S4" in states_sel: np.multiply(ctrl_S4,reduc,out=ctrl_S4)

    def _remaining_in_window(weights,states_sel):
        w=np.clip(weights,0.0,1.0); rem=0.0
        if "S1" in states_sel: rem+=np.sum(S1_pl*ctrl_S1*w)
        if "S2" in states_sel: rem+=np.sum(S2_pl*ctrl_S2*w)
        if "S3" in states_sel: rem+=np.sum(S3_pl*ctrl_S3*w)
        if "S4" in states_sel: rem+=np.sum(S4_pl*ctrl_S4*w)
        return float(rem)

    def _eff_from_to(prev_eff,this_eff): return 1.0-(1.0-prev_eff)*(1.0-this_eff)

    # --- jerarquía ---
    eff_accum_pre=eff_accum_pre2=eff_accum_all=0.0
    if preR:
        w_preR=weights_residual(preR_date,preR_days)
        if _remaining_in_window(w_preR,["S1"])>EPS_REMAIN and ef_preR>0:
            apply_efficiency_per_state(w_preR,ef_preR,["S1"]); eff_accum_pre=_eff_from_to(0.0,ef_preR/100.0)

    if preemR and eff_accum_pre<EPS_EXCLUDE:
        w_preem=weights_residual(preemR_date,preemR_days)
        if _remaining_in_window(w_preem,["S1"])>EPS_REMAIN and ef_preemR>0:
            apply_efficiency_per_state(w_preem,ef_preemR,["S1"]); eff_accum_pre2=_eff_from_to(eff_accum_pre,ef_preemR/100.0)
        else: eff_accum_pre2=eff_accum_pre
    else: eff_accum_pre2=eff_accum_pre

    if pre_selNR:
        w=weights_residual(pre_selNR_date,NR_DAYS_DEFAULT)
        if _remaining_in_window(w,["S1","S2","S3","S4"])>EPS_REMAIN and ef_pre_selNR>0:
            apply_efficiency_per_state(w,ef_pre_selNR,["S1","S2","S3","S4"])
    if pre_glifo:
        w=weights_one_day(pre_glifo_date)
        if _remaining_in_window(w,["S1","S2","S3","S4"])>EPS_REMAIN and ef_pre_glifo>0:
            apply_efficiency_per_state(w,ef_pre_glifo,["S1","S2","S3","S4"])

    if post_selR and eff_accum_pre2<EPS_EXCLUDE:
        w_postR=weights_residual(post_selR_date,post_res_dias)
        if _remaining_in_window(w_postR,["S1","S2"])>EPS_REMAIN and ef_post_selR>0:
            apply_efficiency_per_state(w_postR,ef_post_selR,["S1","S2"]); eff_accum_all=_eff_from_to(eff_accum_pre2,ef_post_selR/100.0)
        else: eff_accum_all=eff_accum_pre2
    else: eff_accum_all=eff_accum_pre2

    if post_gram and eff_accum_all<EPS_EXCLUDE:
        allow_after_postR=True
        if post_selR: allow_after_postR=(pd.to_datetime(post_gram_date)>=(pd.to_datetime(post_selR_date)+pd.Timedelta(days=14)))
        if allow_after_postR:
            w_gram=weights_residual(post_gram_date,POST_GRAM_FORWARD_DAYS)
            if _remaining_in_window(w_gram,["S1","S2","S3"])>EPS_REMAIN and ef_post_gram>0:
                apply_efficiency_per_state(w_gram,ef_post_gram,["S1","S2","S3"])

    S1_pl_ctrl=S1_pl*ctrl_S1; S2_pl_ctrl=S2_pl*ctrl_S2; S3_pl_ctrl=S3_pl*ctrl_S3; S4_pl_ctrl=S4_pl*ctrl_S4
    plantas_supresion=(S1_pl+S2_pl+S3_pl+S4_pl)
    plantas_supresion_ctrl=(S1_pl_ctrl+S2_pl_ctrl+S3_pl_ctrl+S4_pl_ctrl)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=S1_pl_ctrl=S2_pl_ctrl=S3_pl_ctrl=S4_pl_ctrl=plantas_supresion=plantas_supresion_ctrl=np.full(len(ts),np.nan)

# ------------------ CAP A2 ------------------
if factor_area_to_plants is not None:
    base_pl_daily=df_plot["EMERREL"].to_numpy(float)*factor_area_to_plants
    base_pl_daily=np.where((ts.dt.date>=sow_date).to_numpy(),base_pl_daily,0.0)
    base_pl_daily_cap=cap_cumulative(base_pl_daily,MAX_PLANTS_CAP,(ts.dt.date>=sow_date).to_numpy())
    plantas_supresion_cap=np.minimum(plantas_supresion,base_pl_daily_cap)
    plantas_supresion_ctrl_cap=np.minimum(plantas_supresion_ctrl,plantas_supresion_cap)

# ===============================================================
# 🔧 Helpers imprescindibles (deben existir antes del optimizador)
# ===============================================================
_needed_globals = [
    "ts","df_plot","MAX_PLANTS_CAP","FC_S","compute_canopy","auc_time","cap_cumulative",
    "use_ciec","LAIhc","Ca","Cs","mode_canopy","t_lag","t_close","cov_max","lai_max","k_beer"
]
_missing = [k for k in _needed_globals if k not in globals()]
if _missing:
    st.error(f"Faltan variables/funciones previas: {', '.join(_missing)}."); st.stop()

if "compute_ciec_for" not in globals():
    def compute_ciec_for(sow_d: dt.date):
        FCx, LAIx = compute_canopy(ts, sow_d, mode_canopy, int(t_lag), int(t_close), float(cov_max), float(lai_max), float(k_beer))
        if use_ciec:
            Ciec_loc = np.clip((LAIx/max(1e-6,float(LAIhc)))*((float(Ca) if Ca>0 else 1e-6)/(float(Cs) if Cs>0 else 1e-6)),0.0,1.0)
        else:
            Ciec_loc = np.zeros_like(LAIx,float)
        return np.clip(1.0 - Ciec_loc, 0.0, 1.0)

if "recompute_for_sow" not in globals():
    def recompute_for_sow(sow_d: dt.date, T12: int, T23: int, T34: int):
        mask_since = (ts.dt.date >= sow_d)
        births = np.where(mask_since.to_numpy(), df_plot["EMERREL"].to_numpy(float), 0.0)
        one_minus = compute_ciec_for(sow_d)
        S1 = births.copy(); S2 = np.zeros_like(births); S3 = np.zeros_like(births); S4 = np.zeros_like(births)
        for i in range(len(births)):
            if i - int(T12) >= 0:
                mv = births[i - int(T12)]; S1[i - int(T12)] -= mv; S2[i] += mv
            if i - (int(T12)+int(T23)) >= 0:
                mv = births[i - (int(T12)+int(T23))]; S2[i - (int(T12)+int(T23))] -= mv; S3[i] += mv
            if i - (int(T12)+int(T23)+int(T34)) >= 0:
                mv = births[i - (int(T12)+int(T23)+int(T34))]; S3[i - (int(T12)+int(T23)+int(T34))] -= mv; S4[i] += mv
        S1 = np.clip(S1,0,None); S2 = np.clip(S2,0,None); S3 = np.clip(S3,0,None); S4 = np.clip(S4,0,None)
        total = S1+S2+S3+S4
        emeac = np.cumsum(births)
        scale = np.minimum(np.divide(np.clip(emeac,1e-9,None), np.clip(total,1e-9,None)), 1.0)
        S1*=scale; S2*=scale; S3*=scale; S4*=scale
        auc_cruda_loc = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since)
        if not np.isfinite(auc_cruda_loc) or auc_cruda_loc<=1e-9: return None
        factor_area = MAX_PLANTS_CAP/auc_cruda_loc
        S1_pl = np.where(mask_since, S1*one_minus*FC_S["S1"]*factor_area, 0.0)
        S2_pl = np.where(mask_since, S2*one_minus*FC_S["S2"]*factor_area, 0.0)
        S3_pl = np.where(mask_since, S3*one_minus*FC_S["S3"]*factor_area, 0.0)
        S4_pl = np.where(mask_since, S4*one_minus*FC_S["S4"]*factor_area, 0.0)
        base_pl_daily = np.where(mask_since, df_plot["EMERREL"].to_numpy(float)*factor_area, 0.0)
        base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since.to_numpy())
        sup_cap = np.minimum(S1_pl+S2_pl+S3_pl+S4_pl, base_pl_daily_cap)
        return {
            "mask_since": mask_since.to_numpy(),
            "factor_area": factor_area,
            "auc_cruda": auc_cruda_loc,
            "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
            "sup_cap": sup_cap,
            "ts": ts,
            "fechas_d": ts.dt.date.values
        }

for _name in ["compute_ciec_for","recompute_for_sow"]:
    if _name not in globals() or not callable(globals()[_name]):
        st.error(f"No se encontró helper: `{_name}`."); st.stop()

# ===============================================================
# 🧩 BLOQUE 7 — OPTIMIZACIÓN
# ===============================================================
st.markdown("---")
st.header("🧠 Optimización")

for k, v in {"opt_running": False, "opt_stop": False}.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.header("Optimización (variables habilitadas)")
    sow_min = dt.date(int(ts.min().year), 5, 1)
    sow_max = dt.date(int(ts.min().year), 8, 1)
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="opt_sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta", value=sow_max, min_value=sow_min, max_value=sow_max, key="opt_sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 2, 1, key="opt_sow_step")

    use_preR_opt      = st.checkbox("Incluir presiembra + residual", value=True, key="opt_use_preR")
    use_preemR_opt    = st.checkbox("Incluir preemergente + residual", value=True, key="opt_use_preemR")
    use_post_selR_opt = st.checkbox("Incluir post + residual", value=True, key="opt_use_postR")
    use_post_gram_opt = st.checkbox("Incluir graminicida post", value=True, key="opt_use_gram")

    ef_preR_opt      = st.slider("Eficiencia presiembraR (%)", 0, 100, 90, 1, key="opt_ef_preR")     if use_preR_opt else 0
    ef_preemR_opt    = st.slider("Eficiencia preemergenteR (%)", 0, 100, 90, 1, key="opt_ef_preemR")  if use_preemR_opt else 0
    ef_post_selR_opt = st.slider("Eficiencia post residual (%)", 0, 100, 90, 1, key="opt_ef_postR")   if use_post_selR_opt else 0
    ef_post_gram_opt = st.slider("Eficiencia graminicida post (%)",0, 100, 90, 1, key="opt_ef_gram")  if use_post_gram_opt else 0

    st.markdown("### ⏱️ Residualidades por tipo")
    res_min_preR,   res_max_preR   = st.slider("Presiembra (min–max)",     15, 120, (30, 45), 5, key="opt_res_preR")
    res_step_preR                  = st.number_input("Paso presiembra (d)", 1, 30,   5, 1, key="opt_step_preR")
    res_min_preemR, res_max_preemR = st.slider("Preemergente (min–max)",   15, 120, (40, 50), 5, key="opt_res_preemR")
    res_step_preemR                = st.number_input("Paso preemergente (d)",1, 30,  5, 1, key="opt_step_preemR")
    res_min_postR,  res_max_postR  = st.slider("Post (min–max)",           15, 120, (20, 25), 5, key="opt_res_postR")
    res_step_postR                 = st.number_input("Paso post (días)",     1, 30,  5, 1, key="opt_step_postR")

    preR_min_back  = st.number_input("Presiembra: hasta X días antes de siembra", 14, 120, 14, 1, key="opt_preR_back")
    preR_step_days = st.number_input("Paso fechas presiembra (días)",             1,  30,  2,  1, key="opt_preR_step")
    preem_step_days= st.number_input("Paso fechas preemergente (días)",           1,  10,  2,  1, key="opt_preem_step")
    post_days_fw   = st.number_input("Post: días después de siembra (máx.)",     20, 100, 90, 1, key="opt_post_fw")
    post_step_days = st.number_input("Paso fechas post (días)",                   1,  30,  4,  1, key="opt_post_step")

    st.header("Objetivo prioritario (ventana)")
    use_window_obj = st.checkbox("Activar prioridad de ventana", value=True)
    year_any = int(ts.min().year) if len(ts) else int(dt.date.today().year)
    win_min  = dt.date(year_any, 1, 1); win_max  = dt.date(year_any, 12, 31)
    win_start = st.date_input("Inicio ventana", value=dt.date(year_any, 10, 10), min_value=win_min, max_value=win_max, key="obj_win_start")
    win_end   = st.date_input("Fin ventana",    value=dt.date(year_any, 11, 10), min_value=win_min, max_value=win_max, key="obj_win_end")
    weight_factor = st.number_input("Multiplicador dentro de ventana", 0.1, 50.0, 1.00, 0.5, key="obj_weight")

    optimizer  = st.selectbox("Optimizador", ["Búsqueda aleatoria","Grid (combinatorio)","Recocido simulado"], index=0, key="opt_optimizer")
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100, key="opt_maxevals")
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1, key="opt_topk")

    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 50000, 5000, 100, key="opt_sa_iters")
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1, key="opt_sa_T0")
        sa_cooling = st.number_input("Factor de enfriamiento (γ)", 0.80, 0.9999, 0.995, 0.0001, key="opt_sa_cool")

    st.subheader("Ejecución")
    c1, c2 = st.columns(2)
    with c1: start_clicked = st.button("▶️ Iniciar",  use_container_width=True, key="btn_opt_start", disabled=st.session_state.opt_running)
    with c2: stop_clicked  = st.button("⏹️ Detener", use_container_width=True, key="btn_opt_stop",  disabled=not st.session_state.opt_running)
    if start_clicked: st.session_state.opt_stop = False; st.session_state.opt_running = True
    if stop_clicked:  st.session_state.opt_stop = True

if sow_search_from > sow_search_to: st.error("Rango de siembra inválido (desde > hasta)."); st.stop()

def _make_residual_list(rmin, rmax, rstep):
    L = list(range(int(rmin), int(rmax)+1, int(rstep)))
    if int(rmax) not in L: L.append(int(rmax))
    return L
res_days_preR   = _make_residual_list(res_min_preR,   res_max_preR,   res_step_preR)
res_days_preemR = _make_residual_list(res_min_preemR, res_max_preemR, res_step_preemR)
res_days_postR  = _make_residual_list(res_min_postR,  res_max_postR,  res_step_postR)

def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur+=pd.Timedelta(days=int(step_days))
    return out

AUC_THR = 1e-9
def _auc_since_sow(sd: dt.date) -> float:
    mask = (ts.dt.date >= sd)
    if not mask.any(): return 0.0
    return auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask)
def is_valid_sow(sd: dt.date, thr: float=AUC_THR) -> bool:
    try:
        val = _auc_since_sow(sd)
        return (val is not None) and (val > thr)
    except Exception:
        return False

_sow_all = daterange(sow_search_from, sow_search_to, sow_step_days)
sow_candidates = [pd.to_datetime(d).date() for d in _sow_all if is_valid_sow(pd.to_datetime(d).date())]
if len(sow_candidates)==0:
    st.warning("No hay fechas de siembra con AUC(EMERREL) > 0 en el rango."); 
else:
    try:
        aucs = [_auc_since_sow(sd) for sd in sow_candidates]
        st.caption(f"Fechas de siembra válidas: {len(sow_candidates)} · AUC(min)={min(aucs):.6f}")
    except Exception: pass

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_min_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    return list(pd.date_range(start, end, freq=f"{int(preR_step_days)}D"))
def preem_dates(sd):
    start = pd.to_datetime(sd); end = start + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    return list(pd.date_range(start, end, freq=f"{int(preem_step_days)}D"))
def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_days_fw))
    if end < start: return []
    return list(pd.date_range(start, end, freq=f"{int(post_step_days)}D"))

def act_presiembraR(date_val, R, eff):  return {"kind":"preR","date":pd.to_datetime(date_val).date(),"days":int(R),"eff":eff,"states":["S1"]}
def act_preemR(date_val, R, eff):      return {"kind":"preemR","date":pd.to_datetime(date_val).date(),"days":int(R),"eff":eff,"states":["S1"]}
def act_post_selR(date_val, R, eff):   return {"kind":"postR","date":pd.to_datetime(date_val).date(),"days":int(R),"eff":eff,"states":["S1","S2"]}
def act_post_gram(date_val, eff):      return {"kind":"post_gram","date":pd.to_datetime(date_val).date(),"days":POST_GRAM_FORWARD_DAYS,"eff":eff,"states":["S1","S2","S3"]}

def build_all_scenarios():
    scenarios=[]
    for sd in sow_candidates:
        grp=[]
        if use_preR_opt:      grp.append([act_presiembraR(d,R,ef_preR_opt)   for d in pre_sow_dates(sd)  for R in res_days_preR])
        if use_preemR_opt:    grp.append([act_preemR(d,R,ef_preemR_opt)      for d in preem_dates(sd)    for R in res_days_preemR])
        if use_post_selR_opt: grp.append([act_post_selR(d,R,ef_post_selR_opt)for d in post_dates(sd)     for R in res_days_postR])
        if use_post_gram_opt: grp.append([act_post_gram(d,ef_post_gram_opt)  for d in post_dates(sd)])
        combos=[[]]
        for r in range(1,len(grp)+1):
            for subset in itertools.combinations(range(len(grp)), r):
                for p in itertools.product(*[grp[i] for i in subset]): combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random_scenario():
    sd=None
    for _ in range(20):
        cand_sd = random.choice(sow_candidates) if sow_candidates else None
        if cand_sd is None: break
        if is_valid_sow(cand_sd): sd=cand_sd; break
    if sd is None: return (pd.to_datetime(dt.date(int(ts.min().year),1,1)).date(), [])
    schedule=[]
    if use_preR_opt      and random.random()<0.7: 
        cand=pre_sow_dates(sd);   if cand: schedule.append(act_presiembraR(random.choice(cand), random.choice(res_days_preR), ef_preR_opt))
    if use_preemR_opt    and random.random()<0.7:
        cand=preem_dates(sd);     if cand: schedule.append(act_preemR(random.choice(cand), random.choice(res_days_preemR), ef_preemR_opt))
    if use_post_selR_opt and random.random()<0.7:
        cand=post_dates(sd);      if cand: schedule.append(act_post_selR(random.choice(cand), random.choice(res_days_postR), ef_post_selR_opt))
    if use_post_gram_opt and random.random()<0.7:
        cand=post_dates(sd);      if cand: schedule.append(act_post_gram(random.choice(cand), ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

# ---------- Objetivo por ventana (pesos) ----------
def _in_window_md(d: dt.date, m1: int, d1: int, m2: int, d2: int) -> bool:
    md=(d.month,d.day); start=(m1,d1); end=(m2,d2)
    if start<=end: return start<=md<=end
    return md>=start or md<=end

def build_objective_weights(fechas_d: np.ndarray, use_window: bool, win_start: dt.date, win_end: dt.date, w_factor: float) -> np.ndarray:
    n=len(fechas_d)
    if (not use_window) or (w_factor is None) or (float(w_factor)<=1.0) or n==0:
        return np.ones(n,dtype=float)
    m1,d1=int(win_start.month),int(win_start.day); m2,d2=int(win_end.month),int(win_end.day)
    mask=np.fromiter((_in_window_md(d,m1,d1,m2,d2) for d in fechas_d), count=n, dtype=bool)
    w=np.ones(n,dtype=float); w[mask]=float(w_factor); return w

status_ph = st.empty(); prog_ph = st.empty(); results=[]

def evaluate(sd: dt.date, schedule: list):
    sow = pd.to_datetime(sd); sow_plus_20 = sow + pd.Timedelta(days=20)
    # Reglas duras
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR"   and d < sow_plus_20: return None
        if a["kind"] == "preR"    and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if a["kind"] == "preemR"  and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None
        if a["kind"] == "post_gram":
            postR_dates = [pd.to_datetime(x["date"]) for x in schedule if x["kind"] == "postR"]
            if postR_dates and min((d - pr).days for pr in postR_dates) < 14: return None

    env = recompute_for_sow(sd, int(T12), int(T23), int(T34))
    if env is None: return None
    mask_since    = env["mask_since"]; factor_area   = env["factor_area"]; auc_cruda_loc = env["auc_cruda"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]; sup_cap = env["sup_cap"]; ts_local = env["ts"]; fechas_d = env["fechas_d"]

    c1 = np.ones_like(fechas_d,float); c2 = np.ones_like(fechas_d,float)
    c3 = np.ones_like(fechas_d,float); c4 = np.ones_like(fechas_d,float)
    order={"preR":0,"preemR":1,"postR":2,"post_gram":3}
    def _apply_eval(w, eff, states):
        if eff<=0: return
        reduc=np.clip(1.0-(eff/100.0)*np.clip(w,0.0,1.0),0.0,1.0)
        if "S1" in states: np.multiply(c1,reduc,out=c1)
        if "S2" in states: np.multiply(c2,reduc,out=c2)
        if "S3" in states: np.multiply(c3,reduc,out=c3)
        if "S4" in states: np.multiply(c4,reduc,out=c4)
    for a in sorted(schedule, key=lambda a: order.get(a["kind"], 9)):
        ini=pd.to_datetime(a["date"]).date(); fin=(pd.to_datetime(a["date"])+pd.Timedelta(days=int(a["days"]))).date()
        w=((fechas_d>=ini)&(fechas_d<fin)).astype(float); _apply_eval(w,a["eff"],a["states"])
    plantas_ctrl_cap=np.minimum(S1_pl*c1+S2_pl*c2+S3_pl*c3+S4_pl*c4, sup_cap)

    # 🔹 NUEVO: pérdida al 10-oct
    year_ref_loc = int(ts_local.dt.year.mode().iloc[0])
    target_date = dt.date(year_ref_loc, PCC_REF_MONTH, PCC_REF_DAY)
    idx_ref = int(np.argmin(np.abs(np.array([(f - target_date).days for f in fechas_d]))))
    X2loc = float(sup_cap[idx_ref])
    X3loc = float(S1_pl[idx_ref]*c1[idx_ref] + S2_pl[idx_ref]*c2[idx_ref] + S3_pl[idx_ref]*c3[idx_ref] + S4_pl[idx_ref]*c4[idx_ref])
    loss3 = _loss(X3loc)

    # A2 auxiliares
    sup_equiv  = np.divide(sup_cap, factor_area, out=np.zeros_like(sup_cap), where=(factor_area>0))
    ctrl_equiv = np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0))
    auc_sup      = auc_time(ts_local, sup_equiv,  mask=mask_since)
    auc_sup_ctrl = auc_time(ts_local, ctrl_equiv, mask=mask_since)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/auc_cruda_loc))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/auc_cruda_loc))

    return {"sow": sd, "loss_pct": float(loss3), "x2": X2loc, "x3": X3loc, "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

if factor_area_to_plants is None or not np.isfinite(auc_cruda):
    st.info("Necesitás AUC(EMERREL cruda) > 0 para optimizar.")
else:
    if st.session_state.opt_running:
        status_ph.info("Optimizando…")
        if st.session_state.get("opt_optimizer","Búsqueda aleatoria") == "Grid (combinatorio)":
            scenarios = build_all_scenarios()
            total = len(scenarios); st.caption(f"Se evaluarán {total:,} configuraciones.")
            if total > max_evals:
                random.seed(123); scenarios = random.sample(scenarios, k=int(max_evals))
                st.caption(f"Se muestrean {len(scenarios):,} configuraciones (límite).")
            prog = prog_ph.progress(0.0); n=len(scenarios); step=max(1,n//100)
            for i,(sd,sch) in enumerate(scenarios,1):
                if st.session_state.opt_stop: status_ph.warning(f"Detenida en {i-1:,}/{n:,}."); break
                r=evaluate(sd,sch); 
                if r is not None: results.append(r)
                if i%step==0 or i==n: prog.progress(min(1.0,i/n))
            prog_ph.empty()
        elif st.session_state.get("opt_optimizer","Búsqueda aleatoria") == "Búsqueda aleatoria":
            N=int(max_evals); prog=prog_ph.progress(0.0)
            for i in range(1,N+1):
                if st.session_state.opt_stop: status_ph.warning(f"Detenida en {i-1:,}/{N:,}."); break
                sd,sch=sample_random_scenario(); r=evaluate(sd,sch)
                if r is not None: results.append(r)
                if i%max(1,N//100)==0 or i==N: prog.progress(min(1.0,i/N))
            prog_ph.empty()
        else: # SA
            cur = sample_random_scenario(); cur_eval=evaluate(*cur); tries=0
            while cur_eval is None and tries<200: cur=sample_random_scenario(); cur_eval=evaluate(*cur); tries+=1
            if cur_eval is None: status_ph.error("No fue posible estado inicial válido.")
            else:
                best_eval=cur_eval; cur_loss=cur_eval["loss_pct"]; T=float(sa_T0); prog=prog_ph.progress(0.0)
                for it in range(1,int(sa_iters)+1):
                    if st.session_state.opt_stop: status_ph.warning(f"Detenida en {it-1:,}/{int(sa_iters):,}."); break
                    cand=sample_random_scenario(); cand_eval=evaluate(*cand)
                    if cand_eval is not None:
                        d=cand_eval["loss_pct"]-cur_loss
                        if d<=0 or random.random()<_math.exp(-d/max(1e-9,T)):
                            cur,cur_eval,cur_loss=cand,cand_eval,cand_eval["loss_pct"]; results.append(cur_eval)
                            if cur_loss < best_eval["loss_pct"]: best_eval=cur_eval
                    T*=float(sa_cooling)
                    if it%max(1,int(sa_iters)//100)==0 or it==int(sa_iters): prog.progress(min(1.0,it/float(sa_iters)))
                results.append(best_eval); prog_ph.empty()
        st.session_state.opt_running=False; st.session_state.opt_stop=False; status_ph.success("Optimización finalizada.")
    else:
        status_ph.info("Listo para optimizar. Ajustá parámetros y presioná **Iniciar**.")

# ===============================================================
# 🧩 BLOQUE 8 — REPORTE Y GRÁFICOS DEL MEJOR ESCENARIO
# ===============================================================
if results:
    results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))
    best = results_sorted[0]

    # Tabla y descarga del cronograma
    def schedule_df(sch):
        rows=[]
        for a in sch:
            ini = pd.to_datetime(a["date"]); fin = ini + pd.Timedelta(days=int(a["days"]))
            rows.append({"Intervención": a["kind"], "Inicio": str(ini.date()), "Fin": str(fin.date()),
                         "Duración (d)": int(a["days"]), "Eficiencia (%)": int(a["eff"]), "Estados": ",".join(a["states"])})
        return pd.DataFrame(rows)
    df_best = schedule_df(best["schedule"])
    if len(df_best):
        st.dataframe(df_best, use_container_width=True)
        st.download_button("Descargar mejor cronograma (CSV)", df_best.to_csv(index=False).encode("utf-8"), "mejor_cronograma.csv", "text/csv")

    # Recomputo y controles
    envb = recompute_for_sow(pd.to_datetime(best["sow"]).date(), int(T12), int(T23), int(T34))
    if envb is None:
        st.info("No se pudieron recomputar series para el mejor escenario.")
    else:
        ts_b = envb["ts"]; fechas_d_b = envb["fechas_d"]; mask_since_b = envb["mask_since"]
        S1p, S2p, S3p, S4p = envb["S_pl"]; sup_cap_b = envb["sup_cap"]

        c1 = np.ones_like(fechas_d_b,float); c2 = np.ones_like(fechas_d_b,float)
        c3 = np.ones_like(fechas_d_b,float); c4 = np.ones_like(fechas_d_b,float)
        for a in best["schedule"]:
            ini = pd.to_datetime(a["date"]).date(); fin = (pd.to_datetime(a["date"])+pd.Timedelta(days=int(a["days"]))).date()
            w = ((fechas_d_b>=ini)&(fechas_d_b<fin)).astype(float)
            reduc=np.clip(1.0-(a["eff"]/100.0)*np.clip(w,0.0,1.0),0.0,1.0)
            if "S1" in a["states"]: np.multiply(c1,reduc,out=c1)
            if "S2" in a["states"]: np.multiply(c2,reduc,out=c2)
            if "S3" in a["states"]: np.multiply(c3,reduc,out=c3)
            if "S4" in a["states"]: np.multiply(c4,reduc,out=c4)

        total_ctrl_daily = (S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4)
        eps=1e-12; scale = np.where(total_ctrl_daily>eps, np.minimum(1.0, sup_cap_b/total_ctrl_daily),0.0)
        S1_ctrl_cap_b = S1p*c1*scale; S2_ctrl_cap_b = S2p*c2*scale; S3_ctrl_cap_b = S3p*c3*scale; S4_ctrl_cap_b = S4p*c4*scale

        # ---- Línea y referencia 10-Oct ----
        year_ref_b = int(ts_b.dt.year.mode().iloc[0])
        target_date = dt.date(year_ref_b, PCC_REF_MONTH, PCC_REF_DAY)
        ref_line_date = pd.to_datetime(target_date)
        ref_line_style = dict(line_width=2, line_dash="dot", line_color="red")
        ref_annot = dict(x=ref_line_date, y=0.98, xref="x", yref="paper", text="10-Oct", showarrow=False,
                         font=dict(size=12, color="red", family="Arial", weight="bold"), bgcolor="rgba(255,255,255,0.7)")

        # ---- Pérdida al 10-Oct ----
        idx_ref = int(np.argmin(np.abs(np.array([(f - target_date).days for f in fechas_d_b]))))
        X2_b = float(sup_cap_b[idx_ref])
        X3_b = float(S1_ctrl_cap_b[idx_ref] + S2_ctrl_cap_b[idx_ref] + S3_ctrl_cap_b[idx_ref] + S4_ctrl_cap_b[idx_ref])
        loss_b = _loss(X3_b)

        # ======== Resumen ========
        st.subheader("🏆 Mejor escenario")
        st.markdown(
            f"**Siembra:** **{best['sow']}**  \n"
            f"**Pérdida estimada (10-oct):** **{loss_b:.2f}%**  \n"
            f"**x₂ (sin control):** {X2_b:.1f} · **x₃ (con control):** {X3_b:.1f} pl·m²  \n"
            f"**A2_sup:** {best['A2_sup']:.1f} · **A2_ctrl:** {best['A2_ctrl']:.1f} pl·m²"
        )

        # ======== Gráfico A ========
        df_daily_b = pd.DataFrame({
            "fecha": ts_b,
            "pl_sin_ctrl_cap": np.where(mask_since_b, sup_cap_b, 0.0),
            "pl_con_ctrl_cap": np.where(mask_since_b, S1_ctrl_cap_b + S2_ctrl_cap_b + S3_ctrl_cap_b + S4_ctrl_cap_b, 0.0),
        })
        df_week_b = df_daily_b.set_index("fecha").resample("W-MON").sum().reset_index()
        fig_best1 = go.Figure()
        fig_best1.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_sin_ctrl_cap"], name="Aporte semanal (sin control)", yaxis="y2", mode="lines+markers"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_con_ctrl_cap"], name="Aporte semanal (con control)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
        one_minus_best = compute_ciec_for(pd.to_datetime(best["sow"]).date()); Ciec_best = 1.0 - one_minus_best
        fig_best1.add_trace(go.Scatter(x=ts_b, y=Ciec_best, mode="lines", name="Ciec (mejor)", yaxis="y3"))
        fig_best1.update_layout(margin=dict(l=10,r=10,t=40,b=10), title="EMERREL y plantas·m²·semana · Mejor escenario",
                                xaxis_title="Tiempo", yaxis_title="EMERREL",
                                yaxis2=dict(overlaying="y", side="right", title="pl·m²·sem⁻¹"),
                                yaxis3=dict(overlaying="y", side="right", title="Ciec", position=0.97, range=[0,1]))
        # Franjas por intervención (opcional, mismo color/opacity en cada tipo)
        color_map = {"preR":"rgba(255,165,0,0.35)","preemR":"rgba(46,204,113,0.35)","postR":"rgba(30,144,255,0.35)","post_gram":"rgba(255,99,132,0.35)"}
        FRANJA_OPACITY = 0.35
        for a in best["schedule"]:
            x0 = pd.to_datetime(a["date"]); x1 = x0 + pd.Timedelta(days=int(a["days"]))
            color = color_map.get(a["kind"], "rgba(128,128,128,0.35)")
            fig_best1.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=color, opacity=FRANJA_OPACITY)
            fig_best1.add_annotation(x=x0+(x1-x0)/2, y=0.86, xref="x", yref="paper", text=a["kind"], showarrow=False,
                                     bgcolor=color.replace("0.35","0.90"), font=dict(color="white"))
        # Línea 10-Oct
        fig_best1.add_vline(x=ref_line_date, **ref_line_style); fig_best1.add_annotation(**ref_annot)
        st.plotly_chart(fig_best1, use_container_width=True)
        st.markdown("<div style='font-size:14px;line-height:1.5em;'><b>🟧 PreR</b> · <b>🟩 PreemR</b> · <b>🟦 PostR</b> · <b>🟥 Gram</b></div>", unsafe_allow_html=True)

        # ======== Gráfico B (pérdida al 10-oct) ========
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400); y_curve = _loss(x_curve)
        fig2_best = go.Figure()
        fig2_best.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida (%) al 10-oct"))
        fig2_best.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", name="x₂ (sin control, 10-oct)", text=[f"x₂={X2_b:.1f}"], textposition="top center"))
        fig2_best.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", name="x₃ (con control, 10-oct)", text=[f"x₃={X3_b:.1f}"], textposition="top right"))
        fig2_best.add_vline(x=X3_b, line_width=2, line_dash="dot", line_color="red")
        fig2_best.add_annotation(x=X3_b, y=_loss(X3_b), text="10-Oct (x₃)", showarrow=True, arrowhead=2, ax=30, ay=-40, font=dict(color="red"))
        fig2_best.update_layout(title="Pérdida de rendimiento (%) vs densidad efectiva al 10-octubre",
                                xaxis_title="Densidad efectiva (pl·m² al 10-oct)", yaxis_title="Pérdida estimada (%)",
                                margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig2_best, use_container_width=True)

        # ======== Gráfico C (dinámica estados, con 10-Oct) ========
        df_states_week_b = (pd.DataFrame({"fecha": ts_b,"S1": S1_ctrl_cap_b,"S2": S2_ctrl_cap_b,"S3": S3_ctrl_cap_b,"S4": S4_ctrl_cap_b})
                            .set_index("fecha").resample("W-MON").sum().reset_index())
        fig_states = go.Figure()
        for col in ["S1","S2","S3","S4"]:
            fig_states.add_trace(go.Scatter(x=df_states_week_b["fecha"], y=df_states_week_b[col], mode="lines", name=col, stackgroup="one"))
        fig_states.update_layout(title="Aportes semanales por estado (con control + cap)", xaxis_title="Tiempo", yaxis_title="pl·m²·sem⁻¹", margin=dict(l=20,r=20,t=50,b=40))
        fig_states.add_vline(x=ref_line_date, **ref_line_style); fig_states.add_annotation(**ref_annot)
        st.plotly_chart(fig_states, use_container_width=True)













