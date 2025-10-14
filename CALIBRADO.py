# -*- coding: utf-8 -*-
# PREDWEEM — Modelo encadenado S1→S2→S3→S4 + Optimización
# - Si S1 se controla al 100% en una fecha, no hay transición a S2 (ni a S3/S4).
# - Igual para S2→S3 y S3→S4.
# - Mantiene parámetros calibrados, tope A2, (1−Ciec) como supresión de aporte (no mata plantas).
# - Timeline de tratamientos como barras horizontales (sin etiqueta de solapamiento).

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta
import itertools, random, math as _math

# ================== PARÁMETROS CALIBRADOS ==================
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

def loss_fn(x):
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

# Duraciones medias (días) por estado para transición diaria
# (coherentes con tus ventanas previas de rolling)
TAU = {"S1": 6.0, "S2": 21.0, "S3": 32.0}  # S4 = sumidero (sin salida)

# ================== ESTADO UI / APP ==================
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · Modelo encadenado S1→S4 + Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# Constantes
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10

# ===== Helpers lectura =====
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

# ================== Tope A2 ==================
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox(
        "Tope de densidad efectiva (pl·m²)",
        options=[250, 125, 62], index=0
    ))
st.caption(f"AUC(EMERREL cruda) ≙ tope A2 **= {int(MAX_PLANTS_CAP)} pl·m²**.")

# ================== Carga de datos ==================
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])
    url = st.text_input("…o URL raw (opcional)", placeholder="https://raw.githubusercontent.com/usuario/repo/main/emer.csv")
    sep_opt = st.selectbox("Delimitador (CSV)", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha: dd/mm/yyyy", value=True)
    is_cumulative = st.checkbox("Mi serie es acumulada (EMERAC)", value=False)
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

# ================== Selección de columnas ==================
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

# ============= Siembra & Canopia (Ciec) =============
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
one_minus_Ciec = np.clip(1.0 - Ciec, 0.0, 1.0)
df_ciec = pd.DataFrame({"fecha": df_plot["fecha"], "Ciec": Ciec})

# ============= Escalado por AUC a plantas y tope de nacimientos =============
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# Nacimientos diarios (pl/m²·día) escalados y CAP sobre la suma acumulada
if factor_area_to_plants is not None:
    births_daily_raw = df_plot["EMERREL"].to_numpy(float) * factor_area_to_plants
    births_daily_raw = np.where(mask_since_sow.to_numpy(), births_daily_raw, 0.0)
    births_daily = cap_cumulative(births_daily_raw, MAX_PLANTS_CAP, mask_since_sow.to_numpy())
else:
    births_daily = np.zeros(len(ts))

# ================= Manejo (UI manual) =================
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
                                value=preem_min, min_value=preem_min,
                                max_value=preem_max, disabled=not preemR)

with st.sidebar:
    st.header("Manejo post-emergencia (manual)")
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False)
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False, help="≥ siembra + 20 días.")
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)", value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

warnings = []
def check_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debería ser ≤ fecha de siembra ({sow_date}).")
def check_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debería ser ≥ fecha de siembra ({sow_date}).")
if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if preR and preR_date > (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
    warnings.append(f"Presiembra residual debe ser ≤ siembra−{PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW}.")
if preemR and (preemR_date < sow_date or preemR_date > sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
    warnings.append(f"Preemergente residual debe estar entre siembra y siembra+{PREEM_R_MAX_AFTER_SOW_DAYS}.")
if post_gram:  check_post(post_gram_date, "Graminicida (post)")
if post_selR and post_selR_date and post_selR_date < sow_date + timedelta(days=20):
    warnings.append(f"Selectivo + residual (post): debe ser ≥ {sow_date + timedelta(days=20)}.")
for w in warnings: st.warning(w)

# cronograma legible (sidebar)
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

# ================= Eficiencias =================
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 70, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 70, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 70, 1) if post_selR else 0

# ================= Decaimiento opcional =================
with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None
if decaimiento_tipo != "Exponencial": lam_exp = None

# ===== Pesos temporales de tratamientos =====
def weights_interval(start_date, dias):
    w = np.zeros_like(fechas_d, float)
    if (not start_date) or (not dias) or (int(dias) <= 0): return w
    d0 = start_date; d1 = start_date + timedelta(days=int(dias))
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

def weights_one_day(date_val):
    w = np.zeros_like(fechas_d, float)
    if not date_val: return w
    d0 = date_val; d1 = d0 + timedelta(days=1)
    w[(fechas_d >= d0) & (fechas_d < d1)] = 1.0
    return w

# Precomputo de kill diario por estado (k_t en [0,1])
k_S1 = np.zeros_like(fechas_d, float)
k_S2 = np.zeros_like(fechas_d, float)
k_S3 = np.zeros_like(fechas_d, float)
k_S4 = np.zeros_like(fechas_d, float)

def apply_kill(weights, eff_pct, states):
    if eff_pct <= 0 or weights is None: return
    k = (eff_pct/100.0) * np.clip(weights, 0.0, 1.0)
    if "S1" in states: k_S1[:] = 1.0 - (1.0 - k_S1) * (1.0 - k)   # combina multiplicativamente
    if "S2" in states: k_S2[:] = 1.0 - (1.0 - k_S2) * (1.0 - k)
    if "S3" in states: k_S3[:] = 1.0 - (1.0 - k_S3) * (1.0 - k)
    if "S4" in states: k_S4[:] = 1.0 - (1.0 - k_S4) * (1.0 - k)

# Aplicar tratamientos manuales → k_S*
if preR:      apply_kill(weights_interval(preR_date,    preR_days),    ef_preR,      ["S1","S2"])
if preemR:    apply_kill(weights_interval(preemR_date,  preemR_days),  ef_preemR,    ["S1","S2"])
if pre_selNR: apply_kill(weights_interval(pre_selNR_date, NR_DAYS_DEFAULT), ef_pre_selNR, ["S1","S2","S3","S4"])
if pre_glifo: apply_kill(weights_one_day(pre_glifo_date), ef_pre_glifo, ["S1","S2","S3","S4"])
if post_gram: apply_kill(weights_interval(post_gram_date, POST_GRAM_FORWARD_DAYS), ef_post_gram, ["S1","S2","S3"])
if post_selR: apply_kill(weights_interval(post_selR_date, post_res_dias), ef_post_selR, ["S1","S2","S3","S4"])

# ================= DINÁMICA ENCADENADA (S1→S2→S3→S4) =================
N1 = np.zeros(len(ts)); N2 = np.zeros(len(ts)); N3 = np.zeros(len(ts)); N4 = np.zeros(len(ts))
r12 = 1.0/TAU["S1"]; r23 = 1.0/TAU["S2"]; r34 = 1.0/TAU["S3"]

for t in range(len(ts)):
    # supervivencia del día t antes de transición
    s1 = 1.0 - k_S1[t]
    s2 = 1.0 - k_S2[t]
    s3 = 1.0 - k_S3[t]
    s4 = 1.0 - k_S4[t]

    if t == 0:
        N1_t = births_daily[t] * s1
        move12 = r12 * N1_t
        N1[0] = N1_t - move12

        N2_t = 0.0 + move12
        N2_t *= s2
        move23 = r23 * N2_t
        N2[0] = N2_t - move23

        N3_t = 0.0 + move23
        N3_t *= s3
        move34 = r34 * N3_t
        N3[0] = N3_t - move34

        N4_t = 0.0 + move34
        N4[0] = N4_t * s4
    else:
        # S1 en t
        N1_prev = N1[t-1]
        N1_t = (N1_prev + births_daily[t]) * s1
        move12 = r12 * N1_t
        N1[t] = N1_t - move12

        # S2 en t
        N2_prev = N2[t-1]
        N2_t = (N2_prev + move12) * s2
        move23 = r23 * N2_t
        N2[t] = N2_t - move23

        # S3 en t
        N3_prev = N3[t-1]
        N3_t = (N3_prev + move23) * s3
        move34 = r34 * N3_t
        N3[t] = N3_t - move34

        # S4 en t
        N4_prev = N4[t-1]
        N4_t = (N4_prev + move34) * s4
        N4[t] = N4_t

# Aporte efectivo diario (pl·m²·día) ponderado y suprimido por (1−Ciec)
one_minus = one_minus_Ciec.astype(float)
contrib_daily = (W_S["S1"]*N1 + W_S["S2"]*N2 + W_S["S3"]*N3 + W_S["S4"]*N4) * np.clip(one_minus,0,1)

# Series cap/semana para gráfico de la derecha
df_daily = pd.DataFrame({
    "fecha": ts,
    "pl_contrib_diaria": contrib_daily
})
df_week = df_daily.set_index("fecha").resample("W-MON").sum().reset_index()
sem_x = df_week["fecha"]
plm2sem = df_week["pl_contrib_diaria"].to_numpy()

# x (densidad efectiva integrada) y A2 por AUC relativo
X3 = float(np.nansum(contrib_daily[mask_since_sow]))
# Para x2 “sin control” encadenado, simulamos con k=0 (rápido: reutilizamos N con k=0)
def simulate_no_control():
    N1n = np.zeros(len(ts)); N2n = np.zeros(len(ts)); N3n = np.zeros(len(ts)); N4n = np.zeros(len(ts))
    for t in range(len(ts)):
        s1=s2=s3=s4=1.0
        if t==0:
            N1_t = births_daily[t]*s1; move12=r12*N1_t; N1n[0]=N1_t-move12
            N2_t = 0.0+move12; N2_t*=s2; move23=r23*N2_t; N2n[0]=N2_t-move23
            N3_t = 0.0+move23; N3_t*=s3; move34=r34*N3_t; N3n[0]=N3_t-move34
            N4_t = 0.0+move34; N4n[0]=N4_t*s4
        else:
            N1_t = (N1n[t-1] + births_daily[t])*s1; move12=r12*N1_t; N1n[t]=N1_t-move12
            N2_t = (N2n[t-1] + move12)*s2; move23=r23*N2_t; N2n[t]=N2_t-move23
            N3_t = (N3n[t-1] + move23)*s3; move34=r34*N3_t; N3n[t]=N3_t-move34
            N4_t = (N4n[t-1] + move34)*s4; N4n[t]=N4_t
    return (W_S["S1"]*N1n + W_S["S2"]*N2n + W_S["S3"]*N3n + W_S["S4"]*N4n) * one_minus
contrib_daily_noc = simulate_no_control()
X2 = float(np.nansum(contrib_daily_noc[mask_since_sow]))

# A2 (relativo al AUC de EMERREL cruda)
if factor_area_to_plants is not None and auc_cruda > 0:
    # invertimos: contrib -> “supresión equivalente” de EMERREL = contrib / factor
    sup_equiv  = np.divide(contrib_daily_noc, factor_area_to_plants, out=np.zeros_like(contrib_daily_noc), where=(factor_area_to_plants>0))
    ctrl_equiv = np.divide(contrib_daily,      factor_area_to_plants, out=np.zeros_like(contrib_daily),      where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, ctrl_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

loss_x2_pct = float(loss_fn(X2)) if np.isfinite(X2) else float("nan")
loss_x3_pct = float(loss_fn(X3)) if np.isfinite(X3) else float("nan")

# ===== Timeline (barras horizontales abajo) =====
COLOR_TREAT = {
    "glifo":     "rgba(70, 70, 70, 0.7)",
    "preR":      "rgba(0, 128, 0, 0.75)",
    "preemR":    "rgba(0, 180, 0, 0.75)",
    "postR":     "rgba(255, 140, 0, 0.85)",
    "post_gram": "rgba(30, 144, 255, 0.85)",
    "preNR":     "rgba(128, 0, 128, 0.75)"
}
LANES = [("glifo", "Glifosato"),
         ("preNR", "Pre NR"),
         ("preR", "PreR"),
         ("preemR", "PreemR"),
         ("post_gram", "Post grami"),
         ("postR", "PostR")]

def build_manual_intervals():
    out=[]
    if pre_glifo and pre_glifo_date:
        ini = pd.to_datetime(pre_glifo_date); out.append((ini, ini+pd.Timedelta(days=1), "glifo"))
    if pre_selNR and pre_selNR_date:
        ini = pd.to_datetime(pre_selNR_date); out.append((ini, ini+pd.Timedelta(days=NR_DAYS_DEFAULT), "preNR"))
    if preR and preR_date:
        ini = pd.to_datetime(preR_date); out.append((ini, ini+pd.Timedelta(days=int(preR_days)), "preR"))
    if preemR and preemR_date:
        ini = pd.to_datetime(preemR_date); out.append((ini, ini+pd.Timedelta(days=int(preemR_days)), "preemR"))
    if post_gram and post_gram_date:
        ini = pd.to_datetime(post_gram_date); out.append((ini, ini+pd.Timedelta(days=POST_GRAM_FORWARD_DAYS), "post_gram"))
    if post_selR and post_selR_date:
        ini = pd.to_datetime(post_selR_date); out.append((ini, ini+pd.Timedelta(days=int(post_res_dias)), "postR"))
    return out

def add_timeline(fig: go.Figure, intervals, lanes=LANES, band_height=0.16, gap=0.005):
    y0_band = -band_height
    lane_h = (band_height - gap*(len(lanes)+1)) / max(1, len(lanes))
    lane_pos = {}
    for i,(k,label) in enumerate(lanes):
        y0 = y0_band + gap*(i+1) + lane_h*i
        y1 = y0 + lane_h
        lane_pos[k] = (y0,y1)
        fig.add_annotation(xref="paper", yref="paper", x=0.002, y=(y0+y1)/2,
                           text=label, showarrow=False, font=dict(size=10),
                           align="left", bgcolor="rgba(255,255,255,0.6)")
    for (ini,fin,kind) in intervals:
        if kind not in lane_pos: continue
        y0,y1 = lane_pos[kind]
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=ini, x1=fin, y0=y0, y1=y1,
                      line=dict(width=0), fillcolor=COLOR_TREAT.get(kind, "rgba(120,120,120,0.7)"))
    # marco sutil
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=y0_band, y1=0,
                  line=dict(color="rgba(0,0,0,0.15)", width=1), fillcolor="rgba(0,0,0,0)")

# ===== Gráfico 1 =====
st.subheader(f"📊 EMERREL + aporte semanal (modelo encadenado) · Tope A2={int(MAX_PLANTS_CAP)}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))

with st.sidebar:
    st.header("Opciones visuales")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve  = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_timeline    = st.checkbox("Mostrar timeline de tratamientos (abajo)", value=True)

layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=50),
                     title=f"EMERREL (izq) y Plantas·m²·semana (der) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")

if show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right", title="Plantas·m²·sem⁻¹",
                                   position=1.0, showgrid=False)
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem, name="Aporte semanal (encadenado)", yaxis="y2",
                             mode="lines+markers"))

if show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0,1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3"))

if show_timeline:
    add_timeline(fig, build_manual_intervals(), lanes=LANES, band_height=0.16)

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f} · A2_ctrl={A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}")

# Panel x y A2
st.subheader(f"Densidad efectiva (x) y A2 (por AUC)")
st.markdown(
    f"""
**x₂ — Sin control:** **{X2:,.1f}** pl·m²  
**x₃ — Con control:** **{X3:,.1f}** pl·m²  
**A2 (sup):** **{A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f}** pl·m²  
**A2 (ctrl):** **{A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}** pl·m²  
**Pérdida(x₂):** {loss_x2_pct:.2f}% · **Pérdida(x₃):** {loss_x3_pct:.2f}%
"""
)

# =================== (Opcional) Mostrar calibración ===================
with st.expander("Parámetros calibrados (solo lectura)", expanded=False):
    st.markdown(
        f"""
- **w₁..w₄** = {W_S['S1']}, {W_S['S2']}, {W_S['S3']}, {W_S['S4']}  
- **α** = {ALPHA} · **Lmax** = {LMAX}  
- **loss(x) = α·x / (1 + α·x/Lmax)**  
- **τ₁, τ₂, τ₃** = {TAU['S1']}, {TAU['S2']}, {TAU['S3']} días (transición S1→S2→S3→S4)
"""
    )

# =================== Optimización (estructura básica) ===================
st.markdown("---")
st.header("🧠 Optimización (borrador con modelo encadenado)")

with st.sidebar:
    st.header("Optimización (rango de siembra)")
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 7, 1)

    optimizer = st.selectbox("Optimizador", ["Búsqueda aleatoria"], index=0)
    max_evals = st.number_input("Máx. evaluaciones", 100, 20000, 1000, 100)

    st.subheader("Ejecución")
    c1, c2 = st.columns(2)
    with c1:
        start_clicked = st.button("▶️ Iniciar", use_container_width=True, disabled=st.session_state.opt_running)
    with c2:
        stop_clicked  = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running)
    if start_clicked:
        st.session_state.opt_stop = False
        st.session_state.opt_running = True
    if stop_clicked:
        st.session_state.opt_stop = True

def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur=cur+pd.Timedelta(days=int(step_days))
    return out

def compute_one_minus_for(sd):
    FCx, LAIx = compute_canopy(ts, sd, mode_canopy, int(t_lag), int(t_close), float(cov_max), float(lai_max), float(k_beer))
    if use_ciec:
        Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
        Cs_safe = float(Cs) if float(Cs) > 0 else 1e-6
        Ciec_loc = np.clip((LAIx / max(1e-6, float(LAIhc))) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec_loc = np.zeros_like(LAIx, float)
    return np.clip(1.0 - Ciec_loc, 0.0, 1.0)

def simulate_chain(sd, k1, k2, k3, k4):
    # Re-simula nacimientos recortando a partir de siembra (misma escala)
    mask = (ts.dt.date >= sd)
    births = np.where(mask.to_numpy(), births_daily_raw if factor_area_to_plants is not None else df_plot["EMERREL"].to_numpy(float), 0.0)
    if factor_area_to_plants is not None:
        births = births * 1.0  # ya está en pl/m²·día
        births = cap_cumulative(births, MAX_PLANTS_CAP, mask.to_numpy())
    N1 = np.zeros(len(ts)); N2 = np.zeros(len(ts)); N3 = np.zeros(len(ts)); N4 = np.zeros(len(ts))
    r12 = 1.0/TAU["S1"]; r23 = 1.0/TAU["S2"]; r34 = 1.0/TAU["S3"]
    for t in range(len(ts)):
        s1 = 1.0 - k1[t]; s2 = 1.0 - k2[t]; s3 = 1.0 - k3[t]; s4 = 1.0 - k4[t]
        if t == 0:
            N1_t = births[t]*s1; move12=r12*N1_t; N1[0]=N1_t-move12
            N2_t = 0.0+move12; N2_t*=s2; move23=r23*N2_t; N2[0]=N2_t-move23
            N3_t = 0.0+move23; N3_t*=s3; move34=r34*N3_t; N3[0]=N3_t-move34
            N4_t = 0.0+move34; N4[0]=N4_t*s4
        else:
            N1_t = (N1[t-1] + births[t])*s1; move12=r12*N1_t; N1[t]=N1_t-move12
            N2_t = (N2[t-1] + move12)*s2; move23=r23*N2_t; N2[t]=N2_t-move23
            N3_t = (N3[t-1] + move23)*s3; move34=r34*N3_t; N3[t]=N3_t-move34
            N4_t = (N4[t-1] + move34)*s4; N4[t]=N4_t
    one_minus_loc = compute_one_minus_for(sd)
    contrib = (W_S["S1"]*N1 + W_S["S2"]*N2 + W_S["S3"]*N3 + W_S["S4"]*N4) * np.clip(one_minus_loc,0,1)
    X = float(np.nansum(contrib[mask]))
    return X, contrib

if st.session_state.opt_running:
    status_ph = st.empty(); prog_ph = st.empty()
    status_ph.info("Optimizando…")
    sow_candidates = daterange(sow_search_from, sow_search_to, sow_step_days)
    # Usamos los mismos k_S* actuales (de la UI manual) para la evaluación
    best = None
    N = int(max_evals)
    prog = prog_ph.progress(0.0)
    for i in range(1, N+1):
        if st.session_state.opt_stop:
            status_ph.warning(f"Detenida. Progreso: {i-1:,}/{N:,}")
            break
        sd = random.choice(sow_candidates).date()
        Xcand, _ = simulate_chain(sd, k_S1, k_S2, k_S3, k_S4)
        loss_cand = float(loss_fn(Xcand))
        if (best is None) or (loss_cand < best["loss"]):
            best = {"sow": sd, "loss": loss_cand, "x": Xcand}
        if i % max(1, N//100) == 0 or i == N:
            prog.progress(min(1.0, i/N))
    st.session_state.opt_running = False
    st.session_state.opt_stop = False
    prog_ph.empty()
    if best:
        status_ph.success("Optimización finalizada.")
        st.subheader("🏆 Mejor siembra (con tratamientos manuales dados)")
        st.markdown(f"**Siembra:** **{best['sow']}**  \n**x (encadenado):** {best['x']:.1f} pl·m²  \n**Pérdida estimada:** **{best['loss']:.2f}%**")
    else:
        status_ph.error("No se encontró solución candidata.")
else:
    st.info("Listo para optimizar. Ajustá parámetros y presioná **Iniciar**.")

