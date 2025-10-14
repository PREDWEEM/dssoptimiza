# -*- coding: utf-8 -*-
# PREDWEEM — Simulador calibrado con mortalidad encadenada S1→S4
# - Si se controla S1 al 100%, no hay transición posible hacia S2–S4.
# - Si se controla S1 y S2, no hay transición hacia S3–S4, etc.
# - Timeline de tratamientos (barras horizontales) sin aviso de overlapping.
#
# Parámetros calibrados:
#   Pesos por estado: w1=0.369, w2=0.393, w3=1.150, w4=1.769
#   Pérdida: loss(x) = α*x / (1 + (α*x/Lmax)), α=0.503, Lmax=125.91
#   Tope A2 seleccionable: 250 / 125 / 62 pl·m²

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta

# ================== PARÁMETROS CALIBRADOS ==================
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

def _loss(x):
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

# ================== STREAMLIT UI ==================
APP_TITLE = "PREDWEEM · Simulador (encadenado) + Timeline"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14
PREEM_R_MAX_AFTER_SOW_DAYS = 10

# ================== Utilidades ==================
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample): dec_guess = ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw_from_url(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r: return r.read()

def read_raw(up, url):
    if up is not None:
        b = up.read()
        return b if isinstance(b, (bytes, bytearray)) else str(b).encode("utf-8", errors="ignore")
    if url: return read_raw_from_url(url)
    raise ValueError("No hay fuente de datos.")

def parse_csv(raw, sep_opt, dec_opt, encoding="utf-8", on_bad="warn"):
    head_bytes = raw[:8000] if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8", errors="ignore")[:8000]
    head = head_bytes.decode("utf-8", errors="ignore")
    sep_guess, dec_guess = sniff_sep_dec(head)
    sep = sep_guess if sep_opt == "auto" else ("," if sep_opt=="," else (";" if sep_opt==";" else "\t"))
    dec = dec_guess if dec_opt == "auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python", on_bad_lines=on_bad)
    return df, {"sep": sep, "dec": dec, "enc": encoding}

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if": return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%","",regex=False)
    if decimal == ",": t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else: t = t.str.replace(",","",regex=False)
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
        else:
            out[i] = 0.0
    return out

# ================== Entrada de datos ==================
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva (pl·m²)", options=[250,125,62], index=0))

with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV o Excel", type=["csv","xlsx","xls"])
    url = st.text_input("…o URL raw (opcional)", placeholder="https://raw.githubusercontent.com/usuario/repo/main/emer.csv")
    sep_opt = st.selectbox("Delimitador (CSV)", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha: dd/mm/yyyy", value=True)
    is_cumulative = st.checkbox("Serie acumulada (EMERAC)", value=False)
    as_percent = st.checkbox("Valores en % (no 0–1)", value=True)
    dedup = st.selectbox("Si hay fechas duplicadas…", ["sumar", "promediar", "primera"], index=0)
    fill_gaps = st.checkbox("Rellenar días faltantes con 0", value=False)

if up is None and not url:
    st.info("Subí un archivo o pegá una URL para continuar."); st.stop()

try:
    if up is not None and up.name.lower().endswith((".xlsx",".xls")):
        df0 = pd.read_excel(up); meta = {"sep":"(Excel)","dec":"auto","enc":"n/a"}
    else:
        raw = read_raw(up, url)
        if not raw or len(raw) == 0: st.error("El archivo/URL está vacío."); st.stop()
        df0, meta = parse_csv(raw, sep_opt, dec_opt)
    if df0.empty: st.error("El archivo no tiene filas."); st.stop()
    st.success(f"Entrada leída. sep='{meta['sep']}' dec='{meta['dec']}' enc='{meta['enc']}'")
except (URLError, HTTPError) as e:
    st.error(f"No se pudo acceder a la URL: {e}"); st.stop()
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}"); st.stop()

# Selección de columnas y depuración
cols = list(df0.columns)
with st.expander("Seleccionar columnas y depurar datos", expanded=True):
    c_fecha = st.selectbox("Columna de fecha", cols, index=0)
    c_valor = st.selectbox("Columna de valor (EMERREL diaria o EMERAC)", cols, index=1 if len(cols)>1 else 0)

    fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
    sample_str = df0[c_valor].astype(str).head(200).str.cat(sep=" ")
    dec_for_col = "," if (sample_str.count(",")>sample_str.count(".") and re.search(r",\d", sample_str)) else "."
    vals = clean_numeric_series(df0[c_valor], decimal=dec_for_col)

    df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
    if df.empty: st.error("Tras el parseo no quedaron filas válidas."); st.stop()

    if df["fecha"].duplicated().any():
        if dedup == "sumar":
            df = df.groupby("fecha").sum(numeric_only=True).rename_axis("fecha").reset_index()
        elif dedup == "promediar":
            df = df.groupby("fecha").mean(numeric_only=True).rename_axis("fecha").reset_index()
        else:
            df = df.drop_duplicates(subset=["fecha"], keep="first")

    if fill_gaps and len(df) > 1:
        full_idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
        df = df.set_index("fecha").reindex(full_idx).rename_axis("fecha").reset_index()
        df["valor"] = df["valor"].fillna(0.0)

    emerrel = df["valor"].astype(float)
    if as_percent: emerrel = emerrel / 100.0
    if is_cumulative: emerrel = emerrel.diff().fillna(0.0).clip(lower=0.0)
    emerrel = emerrel.clip(lower=0.0)
    df_plot = pd.DataFrame({"fecha": pd.to_datetime(df["fecha"]), "EMERREL": emerrel})

# ================== Siembra & Canopia ==================
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1); sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia (para Ciec)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}**")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de agosto."); st.stop()

def compute_canopy(fechas: pd.Series, sow_date: dt.date, mode_canopy: str,
                   t_lag: int, t_close: int, cov_max: float, lai_max: float, k_beer: float):
    days_since_sow = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas], dtype=float)
    def logistic_between(days, start, end, y_max):
        if end <= start: end = start + 1
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))
    if mode_canopy == "Cobertura dinámica (%)":
        fc_dyn = np.where(days_since_sow < t_lag, 0.0, logistic_between(days_since_sow, t_lag, t_close, cov_max/100.0))
        fc_dyn = np.clip(fc_dyn,0.0,1.0)
        LAI = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
        LAI = np.clip(LAI,0.0,lai_max)
    else:
        LAI = np.where(days_since_sow < t_lag, 0.0, logistic_between(days_since_sow, t_lag, t_close, lai_max))
        LAI = np.clip(LAI,0.0,lai_max)
        fc_dyn = 1 - np.exp(-k_beer*LAI)
        fc_dyn = np.clip(fc_dyn,0.0,1.0)
    return fc_dyn, LAI

FC, LAI = compute_canopy(df_plot["fecha"], sow_date, mode_canopy, int(t_lag), int(t_close),
                         float(cov_max), float(lai_max), float(k_beer))
if use_ciec:
    Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
    Cs_safe = float(Cs) if float(Cs) > 0 else 1e-6
    Ciec = (LAI / max(1e-6, float(LAIhc))) * (Ca_safe / Cs_safe)
    Ciec = np.clip(Ciec, 0.0, 1.0)
else:
    Ciec = np.zeros_like(LAI, dtype=float)

df_ciec = pd.DataFrame({"fecha": df_plot["fecha"], "Ciec": Ciec})
one_minus_Ciec = np.clip((1.0 - Ciec).astype(float), 0.0, 1.0)

# ================== Construcción de S1..S4 base (sin control) ==================
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)

births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where(mask_since_sow.to_numpy(), births, 0.0)
births_series = pd.Series(births, index=ts)

def roll_sum_shift(s: pd.Series, win: int, shift_days: int) -> pd.Series:
    return s.rolling(window=win, min_periods=0).sum().shift(shift_days)

# Cohortes aproximadas por edad
S1_raw = roll_sum_shift(births_series, 6, 1).fillna(0.0).reindex(ts).to_numpy(float)
S2_raw = roll_sum_shift(births_series, 21, 7).fillna(0.0).reindex(ts).to_numpy(float)
S3_raw = roll_sum_shift(births_series, 32, 28).fillna(0.0).reindex(ts).to_numpy(float)
S4_raw = births_series.cumsum().shift(60).fillna(0.0).reindex(ts).to_numpy(float)

# ================== Escalado por área (AUC → tope) ==================
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ================== UI de manejo (manual) ==================
sched_rows = []
def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

fechas_d = ts.dt.date.values
min_date = ts.min().date(); max_date = ts.max().date()

with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    pre_glifo = st.checkbox("Herbicida total (glifosato)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selNR)

    preR = st.checkbox("Selectivo + residual (presiembra)", value=False,
                       help="Solo permitido hasta siembra−14 días. (Actúa S1–S2).")
    preR_days = st.slider("Residualidad presiembra (días)", 15, 120, 45, 1, disabled=not preR)
    preR_max = (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW))
    preR_date = st.date_input("Fecha selectivo + residual (presiembra)",
                              value=min(min_date, preR_max),
                              min_value=min_date,
                              max_value=min(preR_max, max_date),
                              disabled=not preR)

with st.sidebar:
    st.header("Manejo preemergente (manual)")
    preemR = st.checkbox("Selectivo + residual (preemergente)", value=False,
                         help="Ventana [siembra..siembra+10]. (Actúa S1–S2).")
    preemR_days = st.slider("Residualidad preemergente (días)", 15, 120, 45, 1, disabled=not preemR)
    preem_min = sow_date
    preem_max = min(max_date, sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preemR_date = st.date_input("Fecha selectivo + residual (preemergente)",
                                value=preem_min,
                                min_value=preem_min,
                                max_value=preem_max,
                                disabled=not preemR)

with st.sidebar:
    st.header("Manejo post-emergencia (manual)")
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False)
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False,
                            help="≥ siembra + 20 días.")
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)", value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

# Validaciones (reglas)
warnings = []
def check_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debería ser ≤ fecha de siembra ({sow_date}).")
def check_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debería ser ≥ fecha de siembra ({sow_date}).")
if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if preR:
    if preR_date > (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
        warnings.append(f"Presiembra residual debe ser ≤ siembra−{PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW} ({sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)}).")
if preemR:
    if preemR_date < sow_date or preemR_date > sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS):
        warnings.append(f"Preemergente residual debe estar entre siembra y siembra+{PREEM_R_MAX_AFTER_SOW_DAYS}.")
if post_gram:  check_post(post_gram_date, "Graminicida (post)")
if post_selR and post_selR_date and post_selR_date < sow_date + timedelta(days=20):
    warnings.append(f"Selectivo + residual (post): debe ser ≥ {sow_date + timedelta(days=20)}.")
for w in warnings: st.warning(w)

# Cronograma para timeline (sidebar → tabla interna)
if pre_glifo: add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, 1, "Barbecho")
if pre_selNR: add_sched("Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR {NR_DAYS_DEFAULT}d")
if preR:      add_sched("Pre-SIEMBRA · selectivo + residual", preR_date, preR_days, f"Protege {preR_days}d (S1–S2)")
if preemR:    add_sched("PREEMERGENTE · selectivo + residual", preemR_date, preemR_days, f"Protege {preemR_days}d (S1–S2)")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post · graminicida (NR, +10d)", "Inicio": str(ini), "Fin": str(fin), "Nota": "Ventana día de app + 10 días"})
if post_selR: add_sched("Post · selectivo + residual", post_selR_date, post_res_dias, f"Protege {post_res_dias}d")
sched = pd.DataFrame(sched_rows)

# ================== Eficiencias ==================
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 70, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 70, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 70, 1) if post_selR else 0

# ================== Ventanas de efecto (arrays de fechas.date) ==================
fechas_d = ts.dt.date.values

def weights_one_day(date_val):
    w = np.zeros_like(fechas_d, float)
    if not date_val: return w
    d0 = date_val
    mask = (fechas_d == d0)
    w[mask] = 1.0
    return w

def weights_residual(start_date, dias):
    w = np.zeros_like(fechas_d, float)
    if (not start_date) or (not dias) or (int(dias) <= 0): return w
    d0 = start_date; d1 = start_date + timedelta(days=int(dias))
    mask = (fechas_d >= d0) & (fechas_d < d1)
    if not mask.any(): return w
    w[mask] = 1.0
    return w

# ================== Mortalidad encadenada S1→S4 ==================
# Factores de supervivencia por estado por día (1 = no afecta; 0 = mata 100%)
ctrl_S1 = np.ones_like(fechas_d, float)
ctrl_S2 = np.ones_like(fechas_d, float)
ctrl_S3 = np.ones_like(fechas_d, float)
ctrl_S4 = np.ones_like(fechas_d, float)

def apply_eff(weights, eff_pct, states):
    if eff_pct <= 0: return
    reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
    if "S1" in states: np.multiply(ctrl_S1, reduc, out=ctrl_S1)
    if "S2" in states: np.multiply(ctrl_S2, reduc, out=ctrl_S2)
    if "S3" in states: np.multiply(ctrl_S3, reduc, out=ctrl_S3)
    if "S4" in states: np.multiply(ctrl_S4, reduc, out=ctrl_S4)

# Aplicación (respetando reglas de ventana)
if preR:      apply_eff(weights_residual(preR_date,   preR_days),   ef_preR,      ["S1","S2"])
if preemR:    apply_eff(weights_residual(preemR_date, preemR_days), ef_preemR,    ["S1","S2"])
if pre_selNR: apply_eff(weights_residual(pre_selNR_date, NR_DAYS_DEFAULT), ef_pre_selNR, ["S1","S2","S3","S4"])
if pre_glifo: apply_eff(weights_one_day(pre_glifo_date), ef_pre_glifo, ["S1","S2","S3","S4"])
if post_gram: apply_eff(weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS), ef_post_gram, ["S1","S2","S3"])
if post_selR: apply_eff(weights_residual(post_selR_date, post_res_dias), ef_post_selR, ["S1","S2","S3","S4"])

# Escalado a plantas y competencia cultivo
if factor_area_to_plants is not None:
    S1_pl0 = S1_raw * one_minus_Ciec * W_S["S1"] * factor_area_to_plants
    S2_pl0 = S2_raw * one_minus_Ciec * W_S["S2"] * factor_area_to_plants
    S3_pl0 = S3_raw * one_minus_Ciec * W_S["S3"] * factor_area_to_plants
    S4_pl0 = S4_raw * one_minus_Ciec * W_S["S4"] * factor_area_to_plants
else:
    S1_pl0 = S2_pl0 = S3_pl0 = S4_pl0 = np.zeros_like(fechas_d, float)

# FACTOR ENCADENADO: si S1=0 ⇒ S2=S3=S4=0; si S1>0 y S2=0 ⇒ S3=S4=0; etc.
F1 = ctrl_S1
F2 = ctrl_S1 * ctrl_S2
F3 = ctrl_S1 * ctrl_S2 * ctrl_S3
F4 = ctrl_S1 * ctrl_S2 * ctrl_S3 * ctrl_S4

# Contribuciones sin control (solo competencia y pesos)
contrib_0 = np.where(mask_since_sow.to_numpy(), (S1_pl0 + S2_pl0 + S3_pl0 + S4_pl0), 0.0)

# Con control encadenado
S1_ctl = S1_pl0 * F1
S2_ctl = S2_pl0 * F2
S3_ctl = S3_pl0 * F3
S4_ctl = S4_pl0 * F4
contrib_ctl = np.where(mask_since_sow.to_numpy(), (S1_ctl + S2_ctl + S3_ctl + S4_ctl), 0.0)

# Tope A2 diario por acumulación
if factor_area_to_plants is not None:
    base_pl_daily = df_plot["EMERREL"].to_numpy(float) * factor_area_to_plants
    base_pl_daily = np.where(mask_since_sow.to_numpy(), base_pl_daily, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())

    sup_cap      = np.minimum(contrib_0,   base_pl_daily_cap)
    sup_ctrl_cap = np.minimum(contrib_ctl, base_pl_daily_cap)
else:
    sup_cap = sup_ctrl_cap = np.full(len(ts), np.nan)

# Agregación semanal (W-MON)
df_week = pd.DataFrame({"fecha": ts,
                        "sin_ctrl": np.where(mask_since_sow.to_numpy(), sup_cap, 0.0),
                        "con_ctrl": np.where(mask_since_sow.to_numpy(), sup_ctrl_cap, 0.0)}
                      ).set_index("fecha").resample("W-MON").sum().reset_index()

# ===== A2 por AUC =====
if factor_area_to_plants is not None and auc_cruda > 0:
    sup_equiv  = np.divide(sup_cap,      factor_area_to_plants, out=np.zeros_like(sup_cap),      where=(factor_area_to_plants>0))
    supc_equiv = np.divide(sup_ctrl_cap, factor_area_to_plants, out=np.zeros_like(sup_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

# ===== pérdidas vs x =====
X2 = float(np.nansum(sup_cap[mask_since_sow]))
X3 = float(np.nansum(sup_ctrl_cap[mask_since_sow]))
loss_x2_pct = float(_loss(X2)) if np.isfinite(X2) else float("nan")
loss_x3_pct = float(_loss(X3)) if np.isfinite(X3) else float("nan")

# ================== Timeline de tratamientos ==================
COLOR_TREAT = {
    "glifo":     "rgba(70, 70, 70, 0.7)",
    "preR":      "rgba(0, 128, 0, 0.7)",
    "preemR":    "rgba(0, 180, 0, 0.7)",
    "postR":     "rgba(255, 140, 0, 0.8)",
    "post_gram": "rgba(30, 144, 255, 0.8)",
    "preNR":     "rgba(128, 0, 128, 0.7)"
}
LANES = [("glifo","Glifosato"),
         ("preNR","Pre NR"),
         ("preR","PreR"),
         ("preemR","PreemR"),
         ("post_gram","Post grami"),
         ("postR","PostR")]

def build_manual_intervals():
    out=[]
    if pre_glifo and pre_glifo_date:
        ini = pd.to_datetime(pre_glifo_date); fin = ini + pd.Timedelta(days=1); out.append((ini, fin, "glifo"))
    if pre_selNR and pre_selNR_date:
        ini = pd.to_datetime(pre_selNR_date); fin = ini + pd.Timedelta(days=NR_DAYS_DEFAULT); out.append((ini, fin, "preNR"))
    if preR and preR_date:
        ini = pd.to_datetime(preR_date); fin = ini + pd.Timedelta(days=int(preR_days)); out.append((ini, fin, "preR"))
    if preemR and preemR_date:
        ini = pd.to_datetime(preemR_date); fin = ini + pd.Timedelta(days=int(preemR_days)); out.append((ini, fin, "preemR"))
    if post_gram and post_gram_date:
        ini = pd.to_datetime(post_gram_date); fin = ini + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS); out.append((ini, fin, "post_gram"))
    if post_selR and post_selR_date:
        ini = pd.to_datetime(post_selR_date); fin = ini + pd.Timedelta(days=int(post_res_dias)); out.append((ini, fin, "postR"))
    return out

def add_timeline(fig: go.Figure, intervals, lanes=LANES, band_height=0.16, gap=0.005):
    y0_band = -band_height
    lane_height = (band_height - gap*(len(lanes)+1)) / max(1, len(lanes))
    lane_pos={}
    for i,(k,label) in enumerate(lanes):
        y0 = y0_band + gap*(i+1) + lane_height*i
        y1 = y0 + lane_height
        lane_pos[k]=(y0,y1)
        fig.add_annotation(xref="paper", yref="paper", x=0.002, y=(y0+y1)/2,
                           text=label, showarrow=False, font=dict(size=10),
                           align="left", bgcolor="rgba(255,255,255,0.6)")
    for (ini,fin,kind) in intervals:
        if kind not in lane_pos: continue
        y0,y1 = lane_pos[kind]
        fig.add_shape(type="rect", xref="x", yref="paper", x0=ini, x1=fin, y0=y0, y1=y1,
                      line=dict(width=0), fillcolor=COLOR_TREAT.get(kind, "rgba(120,120,120,0.7)"))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=y0_band, y1=0,
                  line=dict(color="rgba(0,0,0,0.15)", width=1), fillcolor="rgba(0,0,0,0)")

# ================== Gráfico principal ==================
st.subheader(f"📊 Gráfico: EMERREL + aportes (cap A2={int(MAX_PLANTS_CAP)}) — Semanal (W-MON)")

with st.sidebar:
    st.header("Opciones visuales")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve  = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_timeline    = st.checkbox("Mostrar timeline (abajo)", value=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)",
                         line=dict(width=2, color="darkgray")))

layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=50),
                     title=f"EMERREL (izq) y Plantas·m²·semana (der, 0–100) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")

if show_plants_axis:
    fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["sin_ctrl"], mode="lines+markers",
                             name="Aporte semanal (sin control)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["con_ctrl"], mode="lines+markers",
                             name="Aporte semanal (con control)", yaxis="y2", line=dict(dash="dot")))
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0, 100], tick0=0, dtick=20, showgrid=False)

if show_ciec_curve:
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines",
                             name="Ciec", yaxis="y3"))
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0,1]))

if show_timeline:
    intervals = build_manual_intervals()
    add_timeline(fig, intervals, lanes=LANES, band_height=0.16)

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)

st.caption(conv_caption + f" · A2_sup={A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f} · A2_ctrl={A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}")

# ================== Panel numérico ==================
st.subheader(f"Densidad efectiva (x) y A2 (cap={int(MAX_PLANTS_CAP)})")
st.markdown(
    f"""
**x₂ — Sin control (cap):** **{X2:,.1f}** pl·m²  
**x₃ — Con control (cap):** **{X3:,.1f}** pl·m²  
**A2 (sup, cap):** **{A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f}** pl·m²  
**A2 (ctrl, cap):** **{A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}** pl·m²  
**Pérdida(x₂):** {loss_x2_pct:.2f}% · **Pérdida(x₃):** {loss_x3_pct:.2f}%
"""
)

# ================== Parámetros (lectura) ==================
with st.expander("Parámetros calibrados (solo lectura)", expanded=False):
    st.markdown(
        f"""
- **w₁..w₄** = {W_S['S1']}, {W_S['S2']}, {W_S['S3']}, {W_S['S4']}  
- **α** = {ALPHA} · **Lmax** = {LMAX}  
- **loss(x) = α·x / (1 + α·x/Lmax)**
"""
    )

# ================== FUNCIÓN DE SIMULACIÓN (encadenada simplificada) ==================
def simulate_chained(
    emerrel_series,
    one_minus_ciec,
    sow_date,
    k1, k2, k3, k4,
    T1=6, T2=21, T3=32,
    factor_area_to_plants=None
):
    """
    Versión compacta para uso en optimización.
    Replica la dinámica encadenada del modelo principal (S1→S4 bloqueante).
    """
    ts_local = pd.to_datetime(emerrel_series.index)
    dates_d = np.array([d.date() for d in ts_local])
    mask = (dates_d >= sow_date)
    births = emerrel_series.to_numpy(float)
    births = np.where(mask, births * one_minus_ciec, 0.0)

    # Densidad efectiva base (sin control)
    if factor_area_to_plants is not None:
        base_pl = births * factor_area_to_plants
    else:
        base_pl = births

    # Simulación encadenada
    n = len(base_pl)
    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for t in range(n):
        inflow = base_pl[t]
        if t == 0:
            s1p = s2p = s3p = s4p = 0.0
        else:
            s1p, s2p, s3p, s4p = S1[t-1], S2[t-1], S3[t-1], S4[t-1]
            s1p *= (1.0 - k1[t-1])
            s2p *= (1.0 - k2[t-1])
            s3p *= (1.0 - k3[t-1])
            s4p *= (1.0 - k4[t-1])
        s1p += inflow
        s1p *= (1.0 - k1[t])
        q12 = s1p / max(1, T1)
        s1p -= q12
        s2p += q12; s2p *= (1.0 - k2[t])
        q23 = s2p / max(1, T2)
        s2p -= q23
        s3p += q23; s3p *= (1.0 - k3[t])
        q34 = s3p / max(1, T3)
        s3p -= q34
        s4p += q34; s4p *= (1.0 - k4[t])
        S1[t], S2[t], S3[t], S4[t] = s1p, s2p, s3p, s4p

    contrib_ctl = (W_S["S1"]*S1 + W_S["S2"]*S2 + W_S["S3"]*S3 + W_S["S4"]*S4)
    return {"contrib_ctl": contrib_ctl, "mask": mask}

# ================== OPTIMIZACIÓN DE ESCENARIOS (completa) ==================
st.header("⚙️ Optimización de escenario de manejo")

with st.expander("Configuración del optimizador", expanded=False):
    opt_method = st.selectbox("Método de búsqueda", ["Grid Search", "Random Search", "Recocido Simulado"], index=1)
    opt_objective = st.selectbox("Objetivo", ["Minimizar pérdida (%)", "Maximizar A2 controlado"], index=0)
    n_iter = st.slider("Número de iteraciones (solo Random/SA)", 10, 500, 100, 10)
    temp0 = st.number_input("Temperatura inicial (solo SA)", 0.1, 10.0, 1.5, 0.1)
    cooling = st.number_input("Factor de enfriamiento (solo SA)", 0.80, 0.999, 0.95, 0.01)
    run_opt = st.button("🚀 Ejecutar optimización")

# ---------------- Definición del objetivo ----------------
def simulate_loss(preR_d, preemR_d, postR_d, gram_d, ef_preR, ef_preemR, ef_postR, ef_gram):
    """
    Corre la simulación encadenada con fechas y eficiencias dadas.
    Devuelve la pérdida (%) o el A2_ctrl según objetivo.
    """
    fechas_d = ts.dt.date.values

    def wres(d, dur):
        w = np.zeros_like(fechas_d, float)
        if not d: return w
        d0 = d; d1 = d + timedelta(days=int(dur))
        mask = (fechas_d >= d0) & (fechas_d < d1)
        w[mask] = 1.0
        return w

    # Mortalidades (eficiencias dinámicas)
    k1 = (ef_preR/100)*wres(preR_d, preR_days) + (ef_preemR/100)*wres(preemR_d, preemR_days) + (ef_gram/100)*wres(gram_d, POST_GRAM_FORWARD_DAYS) + (ef_postR/100)*wres(postR_d, post_res_dias)
    k2 = (ef_preR/100)*wres(preR_d, preR_days) + (ef_preemR/100)*wres(preemR_d, preemR_days) + (ef_gram/100)*wres(gram_d, POST_GRAM_FORWARD_DAYS) + (ef_postR/100)*wres(postR_d, post_res_dias)
    k3 = (ef_gram/100)*wres(gram_d, POST_GRAM_FORWARD_DAYS) + (ef_postR/100)*wres(postR_d, post_res_dias)
    k4 = (ef_postR/100)*wres(postR_d, post_res_dias)

    sim = simulate_chained(
        emerrel_series=df_plot.set_index("fecha")["EMERREL"],
        one_minus_ciec=one_minus_Ciec,
        sow_date=sow_date,
        k1=k1, k2=k2, k3=k3, k4=k4,
        T1=6, T2=21, T3=32,
        factor_area_to_plants=factor_area_to_plants
    )

    contrib_ctl = sim["contrib_ctl"]
    mask = sim["mask"]
    if factor_area_to_plants is None or not np.isfinite(np.sum(contrib_ctl)):
        return np.inf if "Minimizar" in opt_objective else -np.inf

    X_ctrl = float(np.nansum(contrib_ctl[mask]))
    loss_pct = float(_loss(X_ctrl))
    if "Minimizar" in opt_objective:
        return loss_pct
    else:
        return X_ctrl

# ---------------- Métodos de búsqueda ----------------
def grid_search():
    results = []
    preR_opts = [sow_date - timedelta(days=x) for x in [25,20,15]]
    preem_opts = [sow_date + timedelta(days=x) for x in [0,5,10]]
    gram_opts = [sow_date + timedelta(days=x) for x in [0,5,10]]
    post_opts = [sow_date + timedelta(days=x) for x in [20,30,40,50]]
    effs = [60,70,80,90]
    for preR_d in preR_opts:
        for preemR_d in preem_opts:
            for gram_d in gram_opts:
                for postR_d in post_opts:
                    for e1 in effs:
                        val = simulate_loss(preR_d, preemR_d, postR_d, gram_d, e1, e1, e1, e1)
                        results.append((val, preR_d, preemR_d, gram_d, postR_d, e1))
    return results

def random_search(n=100):
    results = []
    for _ in range(n):
        preR_d = sow_date - timedelta(days=np.random.randint(15,40))
        preemR_d = sow_date + timedelta(days=np.random.randint(0,10))
        gram_d = sow_date + timedelta(days=np.random.randint(0,10))
        postR_d = sow_date + timedelta(days=np.random.randint(20,60))
        e1 = np.random.uniform(50,95)
        val = simulate_loss(preR_d, preemR_d, postR_d, gram_d, e1, e1, e1, e1)
        results.append((val, preR_d, preemR_d, gram_d, postR_d, e1))
    return results

def simulated_annealing(n=100, T0=1.5, cool=0.95):
    current = (sow_date - timedelta(days=20), sow_date+timedelta(days=5),
               sow_date+timedelta(days=5), sow_date+timedelta(days=30), 80)
    best_val = simulate_loss(*current)
    best = current
    T = T0
    for i in range(n):
        candidate = (
            current[0] + timedelta(days=np.random.randint(-3,3)),
            current[1] + timedelta(days=np.random.randint(-3,3)),
            current[2] + timedelta(days=np.random.randint(-3,3)),
            current[3] + timedelta(days=np.random.randint(-5,5)),
            np.clip(current[4] + np.random.uniform(-5,5), 50, 95)
        )
        val = simulate_loss(*candidate)
        delta = val - best_val if "Minimizar" in opt_objective else best_val - val
        if delta < 0 or np.random.rand() < math.exp(-abs(delta)/max(1e-6,T)):
            current = candidate
            if ("Minimizar" in opt_objective and val < best_val) or ("Maximizar" in opt_objective and val > best_val):
                best_val, best = val, candidate
        T *= cool
    return [(best_val, *best)]

# ---------------- Ejecución ----------------
if run_opt:
    st.info(f"Ejecutando optimización ({opt_method})...")
    if opt_method == "Grid Search":
        results = grid_search()
    elif opt_method == "Random Search":
        results = random_search(n_iter)
    else:
        results = simulated_annealing(n_iter, temp0, cooling)

    if not results:
        st.error("No se obtuvieron resultados válidos.")
    else:
        df_res = pd.DataFrame(results, columns=["valor","preR_d","preemR_d","gram_d","postR_d","eficiencia"])
        if "Minimizar" in opt_objective:
            best = df_res.loc[df_res["valor"].idxmin()]
        else:
            best = df_res.loc[df_res["valor"].idxmax()]

        st.success(f"🧭 Mejor escenario encontrado:\n"
                   f"PresiembraR={best.preR_d} · PreemR={best.preemR_d} · Graminicida={best.gram_d} · PostR={best.postR_d}\n"
                   f"Eficiencia={best.eficiencia:.1f}% · "
                   f"{'Pérdida mínima' if 'Minimizar' in opt_objective else 'A2 máxima'}={best.valor:.2f}")

        # ===== Gráfico de pérdida de rinde =====
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
        y_curve = _loss(x_curve)

        st.subheader("📉 Curva de pérdida de rendimiento (%) vs densidad efectiva (x)")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo calibrado"))
        if "Minimizar" in opt_objective:
            fig_loss.add_trace(go.Scatter(x=[best.valor], y=[best.valor], mode="markers+text",
                                          name="Escenario óptimo", text=[f"{best.valor:.2f}%"],
                                          textposition="top center"))
        fig_loss.update_layout(
            xaxis_title="Densidad efectiva (x, pl·m²)",
            yaxis_title="Pérdida de rendimiento (%)",
            title="Función de pérdida calibrada"
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # ===== Líneas verticales en EMERREL =====
        st.subheader("🌿 Fechas óptimas sobre EMERREL")
        fig_treat = go.Figure()
        fig_treat.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["EMERREL"], mode="lines", name="EMERREL"))
        for label, date in [("PresiembraR", best.preR_d), ("PreemR", best.preemR_d),
                            ("Graminicida", best.gram_d), ("PostR", best.postR_d)]:
            fig_treat.add_vline(x=date, line_dash="dot", line_color="red")
            fig_treat.add_annotation(x=date, y=1.05*df_plot["EMERREL"].max(), text=label,
                                     showarrow=False, yref="y", bgcolor="rgba(255,0,0,0.2)")
        fig_treat.update_layout(
            title="EMERREL con tratamientos óptimos",
            xaxis_title="Fecha", yaxis_title="EMERREL (fracción)"
        )
        st.plotly_chart(fig_treat, use_container_width=True)







