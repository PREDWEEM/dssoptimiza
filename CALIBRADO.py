# -*- coding: utf-8 -*-
# PREDWEEM — Simulador calibrado (ENCADENADO) + Visualización + Optimización liviana
# - Estados encadenados S1->S2->S3->S4 con bloqueo biológico real.
# - Tratamientos como barras horizontales (timeline) debajo del gráfico.
# - Sin etiqueta de "Overlapping".
# - Conversión por AUC al tope A2 y cap diario.

import io, re, math, datetime as dt, itertools, random
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ========================= PARÁMETROS CALIBRADOS =========================
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}   # pesos por estado
# Tiempos de residencia (días) para el encadenamiento:
T1, T2, T3 = 6, 21, 32  # S1, S2, S3; S4 = > T1+T2+T3 (sin salida)
# Tratamientos:
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14
PREEM_R_MAX_AFTER_SOW_DAYS = 10

def _loss(x):
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

# ========================= STREAMLIT UI STATE =========================
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · Encadenado S1→S4 + Timeline"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ========================= HELPERS I/O =========================
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample):
        dec_guess = ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw_from_url(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r: return r.read()

def read_raw(up, url):
    if up is not None:
        b = up.read()
        return b if isinstance(b, (bytes, bytearray)) else str(b).encode("utf-8", errors="ignore")
    if url:
        return read_raw_from_url(url)
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

# ========================= SIDEBAR: ESCENARIO Y DATOS =========================
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox(
        "Tope de densidad efectiva (pl·m²)", options=[250, 125, 62], index=0,
        help="Tope único de densidad efectiva y A2 (calibrado)."
    ))

st.caption(f"AUC(EMERREL cruda) ≙ tope A2 **= {int(MAX_PLANTS_CAP)} pl·m²**.")

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
        df0 = pd.read_excel(up)
        meta = {"sep":"(Excel)","dec":"auto","enc":"n/a"}
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

# ========================= SIEMBRA, CANOPIA, CIEC =========================
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

one_minus_Ciec = np.clip((1.0 - Ciec).astype(float), 0.0, 1.0)
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)

# ========================= CONVERSIÓN A PLANTAS y CAP =========================
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ========================= TRATAMIENTOS (UI) =========================
fechas_d = ts.dt.date.values
min_date = ts.min().date(); max_date = ts.max().date()

def add_sched(rows, nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

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
                                value=preem_min, min_value=preem_min, max_value=preem_max, disabled=not preemR)

with st.sidebar:
    st.header("Manejo post-emergencia (manual)")
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False)
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False,
                            help="≥ siembra + 20 días.")
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)", value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

# Validaciones
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

# Cronograma tabla (sidebar)
sched_rows = []
if pre_glifo: add_sched(sched_rows, "Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched(sched_rows, "Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR {NR_DAYS_DEFAULT}d")
if preR:      add_sched(sched_rows, "Pre-SIEMBRA · selectivo + residual", preR_date, preR_days, f"Protege {preR_days}d (S1–S2)")
if preemR:    add_sched(sched_rows, "PREEMERGENTE · selectivo + residual", preemR_date, preemR_days, f"Protege {preemR_days}d (S1–S2)")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post · graminicida (NR, +10d)", "Inicio": str(ini), "Fin": str(fin), "Nota": "Ventana día + 10"})
if post_selR: add_sched(sched_rows, "Post · selectivo + residual", post_selR_date, post_res_dias, f"Protege {post_res_dias}d")
sched = pd.DataFrame(sched_rows)

# Eficiencias
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 70, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 70, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 70, 1) if post_selR else 0

with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None
if decaimiento_tipo != "Exponencial": lam_exp = None

# ========================= PESOS DE TRATAMIENTO POR DÍA =========================
def weights_one_day(date_val):
    if not date_val: return np.zeros_like(fechas_d, float)
    d0 = date_val
    return ((fechas_d >= d0) & (fechas_d < (d0 + timedelta(days=1)))).astype(float)

def weights_residual(start_date, dias):
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

def combine_kill(*terms):
    """Productoria de (1 - eff*weight) → 1 - k (k = kill total del día)."""
    base = np.ones_like(terms[0]) if terms else np.ones_like(fechas_d, float)
    for t in terms:
        base *= (1.0 - np.clip(t, 0.0, 1.0))
    return np.clip(1.0 - base, 0.0, 1.0)

# Ventanas por tratamiento
w_glifo   = weights_one_day(pre_glifo_date) if pre_glifo else np.zeros_like(fechas_d, float)
w_preNR   = weights_residual(pre_selNR_date, NR_DAYS_DEFAULT) if pre_selNR else np.zeros_like(fechas_d, float)
w_preR    = weights_residual(preR_date, preR_days) if preR else np.zeros_like(fechas_d, float)
w_preemR  = weights_residual(preemR_date, preemR_days) if preemR else np.zeros_like(fechas_d, float)
w_postG   = weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS) if post_gram else np.zeros_like(fechas_d, float)
w_postR   = weights_residual(post_selR_date, post_res_dias) if post_selR else np.zeros_like(fechas_d, float)

# Kill diario por estado (0..1) — encadenado respeta estados:
k1 = combine_kill(ef_pre_glifo/100*w_glifo, ef_pre_selNR/100*w_preNR, ef_preR/100*w_preR,  ef_preemR/100*w_preemR, ef_post_gram/100*w_postG, ef_post_selR/100*w_postR)
k2 = combine_kill(ef_pre_glifo/100*w_glifo, ef_pre_selNR/100*w_preNR, ef_preR/100*w_preR,  ef_preemR/100*w_preemR,                     ef_post_selR/100*w_postR)
k3 = combine_kill(ef_pre_glifo/100*w_glifo, ef_pre_selNR/100*w_preNR,                                          ef_post_gram/100*w_postG, ef_post_selR/100*w_postR)
k4 = combine_kill(ef_pre_glifo/100*w_glifo, ef_pre_selNR/100*w_preNR,                                                                  ef_post_selR/100*w_postR)

# ========================= SIMULACIÓN ENCADENADA =========================
def simulate_chained(emerrel_series, one_minus_ciec, sow_date, k1, k2, k3, k4,
                     T1=6, T2=21, T3=32, factor_area_to_plants=None):
    """Devuelve dict con stocks S1..S4 y contribución diaria (pl/m²·día) para SIN control y CON control."""
    ts_local = pd.to_datetime(emerrel_series.index)
    dates_d  = ts_local.dt.date.values
    mask     = (dates_d >= sow_date)
    b_raw    = emerrel_series.to_numpy(float)
    b_after_comp = np.where(mask, b_raw * one_minus_ciec, 0.0)  # competencia del cultivo sobre ingresos S1

    # Convertir a plantas base (para cap posterior y referencia)
    if factor_area_to_plants is None:
        base_pl_daily = np.full_like(b_after_comp, np.nan, dtype=float)
    else:
        base_pl_daily = np.where(mask, b_raw * factor_area_to_plants, 0.0)

    n = len(b_after_comp)
    # SIN control (solo competencia del cultivo en el ingreso)
    S1_0 = np.zeros(n); S2_0 = np.zeros(n); S3_0 = np.zeros(n); S4_0 = np.zeros(n)
    # CON control (kill diario por estado)
    S1 = np.zeros(n);   S2 = np.zeros(n);   S3 = np.zeros(n);   S4 = np.zeros(n)

    for t in range(n):
        inflow = b_after_comp[t]  # ingreso a S1
        # --- SIN control ---
        if t == 0:
            s1_prev=s2_prev=s3_prev=s4_prev = 0.0
        else:
            s1_prev, s2_prev, s3_prev, s4_prev = S1_0[t-1], S2_0[t-1], S3_0[t-1], S4_0[t-1]
        # progresión uniforme por día
        flow12 = s1_prev / max(1, T1)
        flow23 = s2_prev / max(1, T2)
        flow34 = s3_prev / max(1, T3)
        S1_0[t] = s1_prev + inflow - flow12
        S2_0[t] = s2_prev + flow12  - flow23
        S3_0[t] = s3_prev + flow23  - flow34
        S4_0[t] = s4_prev + flow34  # S4 sin salida

        # --- CON control ---
        if t == 0:
            s1p=s2p=s3p=s4p = 0.0
        else:
            s1p, s2p, s3p, s4p = S1[t-1], S2[t-1], S3[t-1], S4[t-1]
        # aplica kill del día previo al avance:
        s1p = s1p * (1.0 - k1[t-1]) if t>0 else s1p
        s2p = s2p * (1.0 - k2[t-1]) if t>0 else s2p
        s3p = s3p * (1.0 - k3[t-1]) if t>0 else s3p
        s4p = s4p * (1.0 - k4[t-1]) if t>0 else s4p

        # entra nuevo reclutamiento hoy y se le aplica kill del día t
        s1p = s1p + inflow
        s1p = s1p * (1.0 - k1[t])  # lo muerto en S1 NO progresa ⇒ bloqueo natural

        # progresión con bloqueo: solo lo vivo progresa
        q12 = s1p / max(1, T1)
        s1p = s1p - q12
        s2p = s2p + q12
        s2p = s2p * (1.0 - k2[t])  # kill en S2 antes de pasar a S3
        q23 = s2p / max(1, T2)
        s2p = s2p - q23
        s3p = s3p + q23
        s3p = s3p * (1.0 - k3[t])
        q34 = s3p / max(1, T3)
        s3p = s3p - q34
        s4p = (s4p + q34) * (1.0 - k4[t])  # kill en S4 (permite mortalidad de bancos tardíos)

        S1[t], S2[t], S3[t], S4[t] = s1p, s2p, s3p, s4p

    # contribuciones diarias (pl/m²·día) por pesos
    contrib_0   = W_S["S1"]*S1_0 + W_S["S2"]*S2_0 + W_S["S3"]*S3_0 + W_S["S4"]*S4_0
    contrib_ctl = W_S["S1"]*S1   + W_S["S2"]*S2   + W_S["S3"]*S3   + W_S["S4"]*S4

    return {
        "S0": (S1_0, S2_0, S3_0, S4_0),
        "S":  (S1,   S2,   S3,   S4),
        "contrib_0": contrib_0,
        "contrib_ctl": contrib_ctl,
        "base_pl_daily": base_pl_daily,
        "mask": mask
    }

# ========================= CORRER SIMULACIÓN =========================
if factor_area_to_plants is None:
    st.info("Subí datos con AUC>0 desde siembra para simular/optimizar.")
    st.stop()

sim = simulate_chained(
    emerrel_series=df_plot.set_index("fecha")["EMERREL"],
    one_minus_ciec=one_minus_Ciec,
    sow_date=sow_date,
    k1=k1, k2=k2, k3=k3, k4=k4,
    T1=T1, T2=T2, T3=T3,
    factor_area_to_plants=factor_area_to_plants
)

S1_0,S2_0,S3_0,S4_0 = sim["S0"]
S1c,S2c,S3c,S4c     = sim["S"]
contrib0 = sim["contrib_0"]
contribC = sim["contrib_ctl"]
base_pl_daily = sim["base_pl_daily"]
mask = sim["mask"]

# Cap diario por A2 y reescalado de componentes (como en versión previa)
base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask)
sup_cap_0   = np.minimum(contrib0, base_pl_daily_cap)
sup_cap_ctl = np.minimum(contribC, base_pl_daily_cap)

# Reparto proporcional del cap en la versión controlada entre estados
total_ctl = (W_S["S1"]*S1c + W_S["S2"]*S2c + W_S["S3"]*S3c + W_S["S4"]*S4c)
eps = 1e-12
scale = np.where(total_ctl>eps, np.minimum(1.0, sup_cap_ctl/total_ctl), 0.0)
S1c_cap = W_S["S1"]*S1c*scale; S2c_cap = W_S["S2"]*S2c*scale; S3c_cap = W_S["S3"]*S3c*scale; S4c_cap = W_S["S4"]*S4c*scale
sup_cap_ctl = S1c_cap + S2c_cap + S3c_cap + S4c_cap  # por consistencia numérica

# Agregación semanal
df_daily_cap = pd.DataFrame({"fecha": ts, "sin_ctrl": np.where(mask, sup_cap_0, 0.0), "con_ctrl": np.where(mask, sup_cap_ctl, 0.0)})
df_week = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()

# A2 por AUC
sup_equiv  = np.divide(sup_cap_0,   factor_area_to_plants, out=np.zeros_like(sup_cap_0),   where=(factor_area_to_plants>0))
ctl_equiv  = np.divide(sup_cap_ctl, factor_area_to_plants, out=np.zeros_like(sup_cap_ctl), where=(factor_area_to_plants>0))
auc_sup      = auc_time(ts, sup_equiv, mask=mask)
auc_sup_ctrl = auc_time(ts, ctl_equiv, mask=mask)
A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/auc_cruda)) if auc_cruda>0 else float("nan")
A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/auc_cruda)) if auc_cruda>0 else float("nan")

# x2/x3 y pérdidas
X2 = float(np.nansum(sup_cap_0[mask])); X3 = float(np.nansum(sup_cap_ctl[mask]))
loss_x2_pct = float(_loss(X2)); loss_x3_pct = float(_loss(X3))

# ========================= TIMELINE (barras horizontales) =========================
COLOR_TREAT = {
    "glifo":     "rgba(70, 70, 70, 0.7)",
    "preR":      "rgba(0, 128, 0, 0.7)",
    "preemR":    "rgba(0, 180, 0, 0.7)",
    "postR":     "rgba(255, 140, 0, 0.8)",
    "post_gram": "rgba(30, 144, 255, 0.8)",
    "preNR":     "rgba(128, 0, 128, 0.7)"
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
        ini=pd.to_datetime(pre_glifo_date); out.append((ini, ini+pd.Timedelta(days=1), "glifo"))
    if pre_selNR and pre_selNR_date:
        ini=pd.to_datetime(pre_selNR_date); out.append((ini, ini+pd.Timedelta(days=NR_DAYS_DEFAULT), "preNR"))
    if preR and preR_date:
        ini=pd.to_datetime(preR_date); out.append((ini, ini+pd.Timedelta(days=int(preR_days)), "preR"))
    if preemR and preemR_date:
        ini=pd.to_datetime(preemR_date); out.append((ini, ini+pd.Timedelta(days=int(preemR_days)), "preemR"))
    if post_gram and post_gram_date:
        ini=pd.to_datetime(post_gram_date); out.append((ini, ini+pd.Timedelta(days=POST_GRAM_FORWARD_DAYS), "post_gram"))
    if post_selR and post_selR_date:
        ini=pd.to_datetime(post_selR_date); out.append((ini, ini+pd.Timedelta(days=int(post_res_dias)), "postR"))
    return out

def add_timeline(fig: go.Figure, intervals, lanes=LANES, band_height=0.16, gap=0.005):
    # banda inferior: y en [0 - band_height, 0]
    y0_band = -band_height
    lane_height = (band_height - gap*(len(lanes)+1)) / max(1, len(lanes))
    lane_pos = {}
    for i, (k, label) in enumerate(lanes):
        y0 = y0_band + gap*(i+1) + lane_height*i
        y1 = y0 + lane_height
        lane_pos[k] = (y0, y1)
        fig.add_annotation(xref="paper", yref="paper", x=0.002, y=(y0+y1)/2,
                           text=label, showarrow=False, font=dict(size=10),
                           align="left", bgcolor="rgba(255,255,255,0.6)")
    for (ini, fin, kind) in intervals:
        if kind not in lane_pos: continue
        y0, y1 = lane_pos[kind]
        fig.add_shape(type="rect", xref="x", yref="paper", x0=ini, x1=fin, y0=y0, y1=y1,
                      line=dict(width=0), fillcolor=COLOR_TREAT.get(kind, "rgba(120,120,120,0.7)"))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=y0_band, y1=0,
                  line=dict(color="rgba(0,0,0,0.15)", width=1), fillcolor="rgba(0,0,0,0)")

# ========================= GRÁFICO PRINCIPAL =========================
with st.sidebar:
    st.header("Opciones visuales")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve  = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_timeline    = st.checkbox("Mostrar timeline de tratamientos (abajo)", value=True)

st.subheader(f"📊 EMERREL + aportes (cap A2={int(MAX_PLANTS_CAP)}) — Serie semanal (W-MON)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
layout_kwargs = dict(
    margin=dict(l=10, r=10, t=40, b=50),
    title=f"EMERREL (izq) y Plantas·m²·semana (der, 0–100) · Tope={int(MAX_PLANTS_CAP)}",
    xaxis_title="Tiempo", yaxis_title="EMERREL"
)

if show_plants_axis:
    fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["sin_ctrl"], name="Aporte semanal (sin control, cap)", yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_week["fecha"], y=df_week["con_ctrl"], name="Aporte semanal (con control, cap)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right", title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0,100], tick0=0, dtick=20, showgrid=False)

if show_ciec_curve:
    fig.add_trace(go.Scatter(x=ts, y=1.0-one_minus_Ciec, mode="lines", name="Ciec", yaxis="y3"))
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0,1]))

if show_timeline:
    add_timeline(fig, build_manual_intervals(), lanes=LANES, band_height=0.16)

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={A2_sup_final:.1f} · A2_ctrl={A2_ctrl_final:.1f}")

# ========================= PANEL x y A2 =========================
st.subheader(f"Densidad efectiva (x) y A2 (por AUC, cap={int(MAX_PLANTS_CAP)})")
st.markdown(
    f"""
**x₂ — Sin control (cap):** **{X2:,.1f}** pl·m²  
**x₃ — Con control (cap):** **{X3:,.1f}** pl·m²  
**A2 (sup, cap):** **{A2_sup_final:.1f}** pl·m²  
**A2 (ctrl, cap):** **{A2_ctrl_final:.1f}** pl·m²  
**Pérdida(x₂):** {loss_x2_pct:.2f}% · **Pérdida(x₃):** {loss_x3_pct:.2f}%
"""
)

# ========================= OPTIMIZACIÓN (LIVIANA) =========================
st.markdown("---")
st.header("🧠 Optimización (búsqueda aleatoria)")

with st.sidebar:
    st.header("Optimización")
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 7, 1)

    use_preR_opt     = st.checkbox("Incluir presiembra + residual (≤ siembra−14; S1–S2)", value=True)
    use_preemR_opt   = st.checkbox("Incluir preemergente + residual (siembra..siembra+10; S1–S2)", value=True)
    use_post_selR_opt= st.checkbox("Incluir post + residual (≥ siembra + 20; S1–S4)", value=True)
    use_post_gram_opt= st.checkbox(f"Incluir graminicida post (+{POST_GRAM_FORWARD_DAYS-1}d; S1–S3)", value=True)

    ef_preR_opt     = st.slider("Eficiencia presiembraR (%)", 0, 100, 70, 1)   if use_preR_opt else 0
    ef_preemR_opt   = st.slider("Eficiencia preemergenteR (%)", 0, 100, 70, 1) if use_preemR_opt else 0
    ef_post_selR_opt= st.slider("Eficiencia post residual (%)", 0, 100, 70, 1) if use_post_selR_opt else 0
    ef_post_gram_opt= st.slider("Eficiencia graminicida post (%)", 0, 100, 65, 1) if use_post_gram_opt else 0

    preR_min_back  = st.number_input("PresiembraR: buscar hasta X días antes", 14, 120, 45, 1)
    preR_step_days = st.number_input("Paso fechas PRESIEMBRA (días)", 1, 30, 7, 1)
    preem_step_days= st.number_input("Paso fechas PREEMERGENTE (días)", 1, 10, 3, 1)
    post_days_fw   = st.number_input("Post: días después de siembra (máximo)", 20, 180, 60, 1)
    post_step_days = st.number_input("Paso fechas POST (días)", 1, 30, 7, 1)

    res_min, res_max = st.slider("Residualidad (min–max) [días]", min_value=15, max_value=120, value=(30, 60), step=5)
    res_step = st.number_input("Paso de residualidad (días)", min_value=1, max_value=30, value=5, step=1)

    max_evals   = st.number_input("Máx. evaluaciones aleatorias", 100, 50000, 2000, 100)

def daterange(start_date, end_date, step_days):
    cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end:
        yield cur
        cur=cur+pd.Timedelta(days=int(step_days))

sow_candidates = list(daterange(sow_search_from, sow_search_to, sow_step_days))
def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_min_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(preR_step_days))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(preem_step_days))
    return out

def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_days_fw))
    if end < start: return []
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(post_step_days))
    return out

res_days = list(range(int(res_min), int(res_max) + 1, int(res_step)))
if int(res_max) not in res_days: res_days.append(int(res_max))

def sample_random_scenario():
    sd = random.choice(sow_candidates)
    schedule = []
    if use_preR_opt and random.random()<0.7:
        cand = pre_sow_dates(sd)
        if cand: schedule.append(("preR", random.choice(cand), random.choice(res_days), ef_preR_opt))
    if use_preemR_opt and random.random()<0.7:
        cand = preem_dates(sd)
        if cand: schedule.append(("preemR", random.choice(cand), random.choice(res_days), ef_preemR_opt))
    if use_post_selR_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(("postR", random.choice(cand), random.choice(res_days), ef_post_selR_opt))
    if use_post_gram_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(("post_gram", random.choice(cand), POST_GRAM_FORWARD_DAYS, ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

def evaluate_scenario(sd, schedule):
    # Reglas duras
    sow = pd.to_datetime(sd)
    sow_plus_20 = sow + pd.Timedelta(days=20)
    for kind, datev, days, eff in schedule:
        d = pd.to_datetime(datev)
        if kind == "postR" and d < sow_plus_20: return None
        if kind == "preR" and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if kind == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None

    # construir kill arrays
    w_gl = weights_one_day(datev) if any(k=="glifo" for k,_,_,_ in schedule) else np.zeros_like(fechas_d, float)
    # (no usamos glifo en aleatoria para simplificar)
    w_preNRo=np.zeros_like(fechas_d,float)
    w_preRo = np.zeros_like(fechas_d,float)
    w_preEo = np.zeros_like(fechas_d,float)
    w_postGo= np.zeros_like(fechas_d,float)
    w_postRo= np.zeros_like(fechas_d,float)
    ef_preNRo=0

    for kind, datev, days, eff in schedule:
        if kind=="preR":   w_preRo  += weights_residual(datev, days)
        if kind=="preemR": w_preEo  += weights_residual(datev, days)
        if kind=="post_gram": w_postGo += weights_residual(datev, days)
        if kind=="postR":  w_postRo += weights_residual(datev, days)

    k1l = combine_kill(ef_preNRo/100*w_preNRo, eff/100*w_preRo, eff/100*w_preEo, ef_post_gram_opt/100*w_postGo, ef_post_selR_opt/100*w_postRo)
    k2l = combine_kill(ef_preNRo/100*w_preNRo, eff/100*w_preRo, eff/100*w_preEo,                             ef_post_selR_opt/100*w_postRo)
    k3l = combine_kill(ef_preNRo/100*w_preNRo,                                          ef_post_gram_opt/100*w_postGo, ef_post_selR_opt/100*w_postRo)
    k4l = combine_kill(ef_preNRo/100*w_preNRo,                                                                     ef_post_selR_opt/100*w_postRo)

    siml = simulate_chained(df_plot.set_index("fecha")["EMERREL"], one_minus_Ciec, sd, k1l, k2l, k3l, k4l, T1,T2,T3, factor_area_to_plants)
    sup0 = np.minimum(siml["contrib_0"], cap_cumulative(siml["base_pl_daily"], MAX_PLANTS_CAP, siml["mask"]))
    supc = np.minimum(siml["contrib_ctl"], cap_cumulative(siml["base_pl_daily"], MAX_PLANTS_CAP, siml["mask"]))
    X2l = float(np.nansum(sup0[siml["mask"]])); X3l=float(np.nansum(supc[siml["mask"]]))
    return {"sow": sd, "x2": X2l, "x3": X3l, "loss_pct": float(_loss(X3l)), "schedule": schedule}

c1, c2 = st.columns(2)
with c1:
    start_opt = st.button("▶️ Ejecutar optimización", use_container_width=True)
with c2:
    top_k = st.number_input("Top-k a mostrar", 1, 20, 5, 1)

if start_opt:
    results=[]
    prog = st.progress(0.0)
    for i in range(1, int(max_evals)+1):
        sd, sch = sample_random_scenario()
        r = evaluate_scenario(sd, sch)
        if r is not None: results.append(r)
        if i % max(1, int(max_evals)//100) == 0 or i==int(max_evals):
            prog.progress(min(1.0, i/int(max_evals)))
    if not results:
        st.warning("No se encontraron escenarios válidos con los parámetros actuales.")
    else:
        results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))[:int(top_k)]
        st.subheader("🏆 Mejores escenarios (aleatorio)")
        for j, r in enumerate(results_sorted, 1):
            st.markdown(f"**#{j}** — Siembra: **{r['sow']}** · Pérdida: **{r['loss_pct']:.2f}%** · x₂={r['x2']:.1f} · x₃={r['x3']:.1f}")
        # descargar top-k
        df_best = pd.DataFrame([{
            "rank": j,
            "sow": r["sow"],
            "loss_pct": r["loss_pct"],
            "x2": r["x2"], "x3": r["x3"],
            "schedule": "; ".join([f"{k}@{pd.to_datetime(d).date()}+{int(dd)}d({int(eff)}%)" for (k,d,dd,eff) in r["schedule"]])
        } for j,r in enumerate(results_sorted,1)])
        st.download_button("Descargar top-k (CSV)", df_best.to_csv(index=False).encode("utf-8"),
                           "mejores_escenarios.csv", "text/csv")

# ========================= PARÁMETROS CALIBRADOS (INFO) =========================
with st.expander("Parámetros calibrados (solo lectura)", expanded=False):
    st.markdown(
        f"- **w₁..w₄** = {W_S['S1']}, {W_S['S2']}, {W_S['S3']}, {W_S['S4']}  \n"
        f"- **α** = {ALPHA} · **Lmax** = {LMAX}  \n"
        f"- **loss(x) = α·x / (1 + α·x/Lmax)**  \n"
        f"- **T₁,T₂,T₃** = {T1},{T2},{T3} días"
    )

