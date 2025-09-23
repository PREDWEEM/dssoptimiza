
# -*- coding: utf-8 -*-
# PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Cohortes · Optimización
# Módulo principal de la app Streamlit
#
# Notas clave:
# - Se modela supresión (EMERREL × (1−Ciec)) y control por tratamientos.
# - Cohortes S1..S4 (edad relativa a la emergencia).
# - Tope único de densidad efectiva (A2) configurable.
# - Restricciones de ventana para tratamientos (pre, preemergente y post).
# - Optimizadores: Grid / Búsqueda aleatoria / Recocido simulado.
# - Máx. evaluaciones controlado desde la UI.
#
# IMPORTANTE:
# - Las constantes y los imports van en la Sección 2.
# - La configuración de Streamlit (set_page_config, title) también va en Sección 2,
#   después de los imports (no se puede invocar st.* antes de importar streamlit).

__app_name__ = "PREDWEEM"
__version__ = "2025.03"
APP_TITLE = "PREDWEEM · Supresión (1−Ciec) + Control (AUC) + Cohortes · Optimización"

import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta
import itertools, random, math as _math

# -------------------------- Estado UI optimización --------------------------
if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_stop" not in st.session_state:
    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · Supresión (1−Ciec) + Control (AUC) + Cohortes · Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ========================== Selector de escenario ==========================
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox(
        "Tope de densidad efectiva (pl·m²)",
        options=[250, 125, 62],
        index=0,
        help="Define el tope único de densidad efectiva y A2."
    ))
st.caption(
    f"AUC(EMERREL cruda) ≙ tope A2 **= {int(MAX_PLANTS_CAP)} pl·m²**. "
    "Cohortes S1..S4 (edad desde emergencia). Salidas en pl·m²·sem⁻¹ con cap acumulativo y reescalado proporcional por estado; todo desde siembra (t=0)."
)

# ========================== Constantes y helpers ==========================
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11  # NO TOCAR

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
    if up is not None: return up.read()
    if url: return read_raw_from_url(url)
    raise ValueError("No hay fuente de datos.")

def parse_csv(raw, sep_opt, dec_opt, encoding="utf-8", on_bad="warn"):
    head = raw[:8000].decode("utf-8", errors="ignore")
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

# ======================= Canopia (sin ICIC): FC y LAI =================
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

# ========================= Sidebar: datos base =========================
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria o EMERAC)", type=["csv"])
    url = st.text_input("…o URL raw de GitHub", placeholder="https://raw.githubusercontent.com/usuario/repo/main/emer.csv")
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha: día/mes/año (dd/mm/yyyy)", value=True)
    is_cumulative = st.checkbox("Mi CSV es acumulado (EMERAC)", value=False)
    as_percent = st.checkbox("Valores en % (no 0–1)", value=True)
    dedup = st.selectbox("Si hay fechas duplicadas…", ["sumar", "promediar", "primera"], index=0)
    fill_gaps = st.checkbox("Rellenar días faltantes con 0", value=False)

if up is None and not url:
    st.info("Subí un CSV o pegá una URL para continuar.")
    st.stop()

# ========================= Carga y parseo CSV ==========================
try:
    raw = read_raw(up, url)
    if not raw or len(raw) == 0: st.error("El archivo/URL está vacío."); st.stop()
    df0, meta = parse_csv(raw, sep_opt, dec_opt)
    if df0.empty: st.error("El CSV no tiene filas."); st.stop()
    st.success(f"CSV leído. sep='{meta['sep']}' dec='{meta['dec']}' enc='{meta['enc']}'")
except (URLError, HTTPError) as e:
    st.error(f"No se pudo acceder a la URL: {e}"); st.stop()
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}"); st.stop()

# ===================== Selección de columnas ===========================
cols = list(df0.columns)
with st.expander("Seleccionar columnas y depurar datos", expanded=True):
    c_fecha = st.selectbox("Columna de fecha", cols, index=0)
    c_valor = st.selectbox("Columna de valor (EMERREL diaria o EMERAC)", cols, index=1 if len(cols)>1 else 0)

    fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
    sample_str = df0[c_valor].astype(str).head(200).str.cat(sep=" ")
    dec_for_col = "," if (sample_str.count(",")>sample_str.count(".") and re.search(r",\d", sample_str)) else "."
    vals = clean_numeric_series(df0[c_valor], decimal=dec_for_col)

    df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
    if df.empty: st.error("Tras el parseo no quedaron filas válidas (fechas/valores NaN)."); st.stop()

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

# ==================== Siembra & parámetros de canopia ==================
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1); sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia (para Ciec)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}** (1 de mayo al 1 de agosto)")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

# ========================= Sidebar: Ciec y TOPE ========================
with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

with st.sidebar:
    st.header("Tope A2 / densidad efectiva")
    st.markdown(f"**Tope seleccionado**: `{int(MAX_PLANTS_CAP)} pl·m²`")
    st.caption("Se usa en la equivalencia por área, el cap acumulativo, A2/A2_ctrl y ejes.")

# ========================= Periodo crítico (PC) ========================
with st.sidebar:
    st.header("Periodo crítico")
    st.caption("Ventana **11 de septiembre → 15 de noviembre** (solo resalta).")
    use_pc = st.checkbox("Resaltar periodo crítico", value=False)
    ref_pc = st.selectbox("Referencia de edad", ["Punto medio", "11-Sep", "15-Nov"], index=0)

year_pc = int(sow_date.year if sow_date else (years.mode().iloc[0] if len(years) else dt.date.today().year))
PC_START = pd.to_datetime(f"{year_pc}-09-11")
PC_END   = pd.to_datetime(f"{year_pc}-11-15")

with st.sidebar:
    st.header("Etiquetas y escalas")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_nonres_bands = st.checkbox("Marcar bandas de efecto", value=True)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de agosto."); st.stop()

# ============================ FC/LAI + Ciec ===========================
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

# ===================== Cohortes S1..S4 =====================
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)
births = df_plot["EMERREL"].astype(float).to_numpy()
births = np.clip(births, 0.0, None)
births = np.where(mask_since_sow.to_numpy(), births, 0.0)
births_series = pd.Series(births, index=pd.to_datetime(df_plot["fecha"]))
def roll_sum_shift(s: pd.Series, win: int, shift_days: int) -> pd.Series:
    return s.rolling(window=win, min_periods=0).sum().shift(shift_days)
S1_coh = roll_sum_shift(births_series, 6, 1).fillna(0.0)
S2_coh = roll_sum_shift(births_series, 21, 7).fillna(0.0)
S3_coh = roll_sum_shift(births_series, 32, 28).fillna(0.0)
S4_coh = births_series.cumsum().shift(60).fillna(0.0)

FC_S = {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0}
S1_arr = S1_coh.reindex(pd.to_datetime(df_plot["fecha"])).to_numpy(float)
S2_arr = S2_coh.reindex(pd.to_datetime(df_plot["fecha"])).to_numpy(float)
S3_arr = S3_coh.reindex(pd.to_datetime(df_plot["fecha"])).to_numpy(float)
S4_arr = S4_coh.reindex(pd.to_datetime(df_plot["fecha"])).to_numpy(float)

# ================== AUC y equivalencia por área =============
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f} pl·m² por EMERREL·día)"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# =================== Manejo (manual) ===================

# Constantes de ventanas reglamentarias
PRE_R_MAX_BEFORE_SOW_DAYS = 14  # PRESIEMBRA R: sólo ≤ siembra - 14
PREEM_R_MAX_AFTER_SOW_DAYS = 7  # PREEMERGENTE R: 0..+7 desde siembra

sched_rows = []
def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    min_date = ts.min().date(); max_date = ts.max().date()

    # Pre totales / selectivos (como antes)
    pre_glifo = st.checkbox("Herbicida total (glifosato)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selNR)

    # NUEVO: Selectivo + residual (PRESIEMBRA) — sólo ≤ siembra-14
    pre_selR  = st.checkbox("Selectivo + residual (PRESIEMBRA, ≤ siembra−14)", value=False)
    pre_res_dias = st.slider("Residualidad PRESIEMBRA (días)", 30, 60, 45, 1, disabled=not pre_selR)
    pre_selR_date = st.date_input("Fecha selectivo + residual (PRESIEMBRA)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selR)

    st.header("Manejo preemergencia / postemergencia (manual)")
    # NUEVO: Selectivo + residual PREEMERGENTE — 0..+7
    preem_selR = st.checkbox("Selectivo + residual (PREEMERGENTE, 0..+7)", value=False)
    preem_res_dias = st.slider("Residualidad PREEMERGENTE (días)", 30, 60, 45, 1, disabled=not preem_selR)
    preem_min = sow_date
    preem_max = min(max_date, sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preem_selR_date = st.date_input(
        "Fecha selectivo + residual (PREEMERGENTE)",
        value=preem_min, min_value=preem_min, max_value=preem_max, disabled=not preem_selR
    )

    # Post (como antes)
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False)
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (POST, ≥ siembra+20)", value=False)
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (POST)", value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad POST (días)", 30, 60, 45, 1, disabled=not post_selR)

# Validaciones y advertencias
warnings = []

def check_pre(date_val, name):
    if date_val and date_val > sow_date:
        warnings.append(f"{name}: debería ser ≤ fecha de siembra ({sow_date}).")

def check_pre_seleri_window(date_val, name):
    # Sólo ≤ siembra - 14
    if date_val:
        if date_val > (sow_date - timedelta(days=PRE_R_MAX_BEFORE_SOW_DAYS)):
            warnings.append(f"{name}: inválida. Debe ser ≤ {sow_date - timedelta(days=PRE_R_MAX_BEFORE_SOW_DAYS)} (no se permite en {{−14, siembra}}).")

def check_preem_window(date_val, name):
    # Debe cumplir sow ≤ fecha ≤ sow+7
    if date_val:
        if not (sow_date <= date_val <= sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
            warnings.append(f"{name}: inválida. Debe estar entre {sow_date} y {sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)} (0..+7).")

def check_post(date_val, name):
    if date_val and date_val < sow_date:
        warnings.append(f"{name}: debería ser ≥ fecha de siembra ({sow_date}).")

# Aplicar checks
if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if pre_selR:   check_pre_seleri_window(pre_selR_date, "Selectivo + residual (PRESIEMBRA)")
if preem_selR: check_preem_window(preem_selR_date, "Selectivo + residual (PREEMERGENTE)")
if post_gram:  check_post(post_gram_date, "Graminicida (post)")
if post_selR and post_selR_date and post_selR_date < sow_date + timedelta(days=20):
    warnings.append(f"Selectivo + residual (POST): debe ser ≥ {sow_date + timedelta(days=20)}.")
for w in warnings: st.warning(w)

# Cargar cronograma (para mostrar bandas)
if pre_glifo: add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched(f"Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR por defecto {NR_DAYS_DEFAULT}d")
if pre_selR:  add_sched("Pre · selectivo + residual (PRESIEMBRA)", pre_selR_date, pre_res_dias, f"Protege {pre_res_dias}d")
if preem_selR: add_sched("Preem · selectivo + residual (0..+7)", preem_selR_date, preem_res_dias, f"Protege {preem_res_dias}d")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post · graminicida (NR, +10d)", "Inicio": str(ini), "Fin": str(fin), "Nota": "Ventana día de app + 10 días"})
if post_selR: add_sched("Post · selectivo + residual (≥+20)", post_selR_date, post_res_dias, f"Protege {post_res_dias}d")
sched = pd.DataFrame(sched_rows)

with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo    = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR    = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_pre_selR     = st.slider("Selectivo + residual (PRESIEMBRA)", 0, 100, 70, 1) if pre_selR else 0
    ef_preem_selR   = st.slider("Selectivo + residual (PREEMERGENTE)", 0, 100, 70, 1) if preem_selR else 0
    ef_post_gram    = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR    = st.slider("Selectivo + residual (POST, 30–60d)", 0, 100, 70, 1) if post_selR else 0

with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None
if decaimiento_tipo != "Exponencial": lam_exp = None

# =================== Estados objetivo por tratamiento ==================
with st.sidebar:
    st.header("Estados objetivo por tratamiento")
    st.caption("Los **residuales (pre, preemergente y post)** actúan sobre **S1–S4** durante su periodo.")
    default_glifo, default_selNR, default_gram = ["S1","S2","S3","S4"], ["S1","S2","S3","S4"], ["S1","S2","S3"]
    states_glifo = st.multiselect("Glifosato (pre)", ["S1","S2","S3","S4"], default_glifo, disabled=not pre_glifo)
    states_preNR = st.multiselect("Selectivo NR (pre)", ["S1","S2","S3","S4"], default_selNR, disabled=not pre_selNR)
    st.markdown("**PRESIEMBRA R / PREEMERGENTE R / POST R:** estados fijados a **S1–S4**")
    states_preR     = ["S1","S2","S3","S4"]
    states_preemR   = ["S1","S2","S3","S4"]
    states_gram     = st.multiselect("Graminicida (post)", ["S1","S2","S3","S4"], default_gram, disabled=not post_gram)
    states_postR    = ["S1","S2","S3","S4"]
if pre_selNR and (len(states_preNR) == 0): states_preNR = ["S1","S2","S3","S4"]

# =================== Ventanas de efecto (manual) ======================
fechas_d = ts.dt.date.values
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

# ==================== Aportes por estado (pl·m²·día⁻¹) =================
if factor_area_to_plants is not None:
    S1_pl = S1_arr * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants
    S2_pl = S2_arr * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants
    S3_pl = S3_arr * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants
    S4_pl = S4_arr * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants
    ms = mask_since_sow.to_numpy()
    S1_pl = np.where(ms, S1_pl, 0.0); S2_pl = np.where(ms, S2_pl, 0.0)
    S3_pl = np.where(ms, S3_pl, 0.0); S4_pl = np.where(ms, S4_pl, 0.0)

    ctrl_S1 = np.ones_like(fechas_d, float); ctrl_S2 = np.ones_like(fechas_d, float)
    ctrl_S3 = np.ones_like(fechas_d, float); ctrl_S4 = np.ones_like(fechas_d, float)

    def apply_efficiency_per_state(weights, eff_pct, states_sel):
        if eff_pct <= 0 or (not states_sel): return
        reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states_sel: np.multiply(ctrl_S1, reduc, out=ctrl_S1)
        if "S2" in states_sel: np.multiply(ctrl_S2, reduc, out=ctrl_S2)
        if "S3" in states_sel: np.multiply(ctrl_S3, reduc, out=ctrl_S3)
        if "S4" in states_sel: np.multiply(ctrl_S4, reduc, out=ctrl_S4)

    if pre_glifo:    apply_efficiency_per_state(weights_one_day(pre_glifo_date), ef_pre_glifo, states_glifo)
    if pre_selNR:    apply_efficiency_per_state(weights_residual(pre_selNR_date, NR_DAYS_DEFAULT), ef_pre_selNR, states_preNR)
    if pre_selR:     apply_efficiency_per_state(weights_residual(pre_selR_date,  pre_res_dias),   ef_pre_selR,  states_preR)      # PRESIEMBRA R
    if preem_selR:   apply_efficiency_per_state(weights_residual(preem_selR_date, preem_res_dias), ef_preem_selR, states_preemR) # PREEMERGENTE R (0..+7)
    if post_gram:    apply_efficiency_per_state(weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS), ef_post_gram, states_gram)
    if post_selR:    apply_efficiency_per_state(weights_residual(post_selR_date, post_res_dias), ef_post_selR, states_postR)

    S1_pl_ctrl = np.where(ms, S1_pl * ctrl_S1, 0.0)
    S2_pl_ctrl = np.where(ms, S2_pl * ctrl_S2, 0.0)
    S3_pl_ctrl = np.where(ms, S3_pl * ctrl_S3, 0.0)
    S4_pl_ctrl = np.where(ms, S4_pl * ctrl_S4, 0.0)
    plantas_supresion      = (S1_pl + S2_pl + S3_pl + S4_pl)
    plantas_supresion_ctrl = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=S1_pl_ctrl=S2_pl_ctrl=S3_pl_ctrl=S4_pl_ctrl=plantas_supresion=plantas_supresion_ctrl=np.full(len(ts), np.nan)

# ==================== Tope A2 estricto ====================
if factor_area_to_plants is not None:
    base_pl_daily = df_plot["EMERREL"].to_numpy(float) * factor_area_to_plants
    base_pl_daily = np.where(mask_since_sow.to_numpy(), base_pl_daily, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())
    plantas_supresion_cap      = np.minimum(plantas_supresion, base_pl_daily_cap)
    plantas_supresion_ctrl_cap = np.minimum(plantas_supresion_ctrl, plantas_supresion_cap)
else:
    base_pl_daily = base_pl_daily_cap = plantas_supresion_cap = plantas_supresion_ctrl_cap = np.full(len(ts), np.nan)

# ========= Reescalado proporcional por estado =========
if factor_area_to_plants is not None:
    total_ctrl_daily = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
    eps = 1e-12
    scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, plantas_supresion_ctrl_cap / total_ctrl_daily), 0.0)
    S1_pl_ctrl_cap = S1_pl_ctrl * scale
    S2_pl_ctrl_cap = S2_pl_ctrl * scale
    S3_pl_ctrl_cap = S3_pl_ctrl * scale
    S4_pl_ctrl_cap = S4_pl_ctrl * scale
    plantas_supresion_ctrl_cap = S1_pl_ctrl_cap + S2_pl_ctrl_cap + S3_pl_ctrl_cap + S4_pl_ctrl_cap
else:
    S1_pl_ctrl_cap=S2_pl_ctrl_cap=S3_pl_ctrl_cap=S4_pl_ctrl_cap=np.full(len(ts), np.nan)

# ============ Agregación SEMANAL ============
df_daily_cap = pd.DataFrame({
    "fecha": pd.to_datetime(ts),
    "pl_sin_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_cap, 0.0),
    "pl_con_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_ctrl_cap, 0.0),
    "pl_base_cap":     np.where(mask_since_sow.to_numpy(), base_pl_daily_cap, 0.0),
})
df_week_cap = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()
sem_x = df_week_cap["fecha"]
plm2sem_sin_ctrl_cap = df_week_cap["pl_sin_ctrl_cap"].to_numpy()
plm2sem_con_ctrl_cap = df_week_cap["pl_con_ctrl_cap"].to_numpy()

# ============= A2 por AUC =============
if factor_area_to_plants is not None and auc_cruda > 0:
    sup_equiv  = np.divide(plantas_supresion_cap,     factor_area_to_plants, out=np.zeros_like(plantas_supresion_cap),     where=(factor_area_to_plants>0))
    supc_equiv = np.divide(plantas_supresion_ctrl_cap, factor_area_to_plants, out=np.zeros_like(plantas_supresion_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

# ======== x y pérdidas ========
def perdida_rinde_pct(x): x = np.asarray(x, float); return 0.375 * x / (1.0 + (0.375 * x / 76.639))
if factor_area_to_plants is not None:
    X2 = float(np.nansum(plantas_supresion_cap[mask_since_sow]))
    X3 = float(np.nansum(plantas_supresion_ctrl_cap[mask_since_sow]))
    loss_x2_pct = float(perdida_rinde_pct(X2)) if np.isfinite(X2) else float("nan")
    loss_x3_pct = float(perdida_rinde_pct(X3)) if np.isfinite(X3) else float("nan")
else:
    X2 = X3 = float("nan"); loss_x2_pct = loss_x3_pct = float("nan")

# ============================== Gráfico 1 (manual) ==============================
st.subheader(f"📊 Gráfico 1: EMERREL + aportes (cap A2={int(MAX_PLANTS_CAP)}) — Serie semanal (W-MON)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)",
                         hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL: %{y:.4f}<extra></extra>"))
layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=10),
                     title=f"EMERREL + aportes (izq) y Plantas·m²·semana (der, 0–100) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")
if factor_area_to_plants is not None and show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0, 100], tick0=0, dtick=20, showgrid=False)
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_sin_ctrl_cap, name="Aporte semanal (sin control, cap)",
                             yaxis="y2", mode="lines+markers",
                             hovertemplate="Lunes: %{x|%Y-%m-%d}<br>pl·m²·sem⁻¹: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_con_ctrl_cap, name="Aporte semanal (con control, cap)",
                             yaxis="y2", mode="lines+markers", line=dict(dash="dot"),
                             hovertemplate="Lunes: %{x|%Y-%m-%d}<br>pl·m²·sem⁻¹: %{y:.2f}<extra></extra>"))

def _add_label(center_ts, text, bgcolor, y=0.94):
    fig.add_annotation(x=center_ts, y=y, xref="x", yref="paper",
        text=text, showarrow=False, bgcolor=bgcolor, opacity=0.9,
        bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2)

def add_residual_band(start_date, days, label, color_rgba="rgba(255,160,122,1)", alpha=0.15):
    if start_date is None or days is None: return
    try:
        d_int = int(days)
        if d_int <= 0: return
        x0 = pd.to_datetime(start_date); x1 = x0 + pd.Timedelta(days=d_int)
        if x1 <= x0: return
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=color_rgba, opacity=alpha)
        label_bg = re.sub(r"rgba\(([^)]+),\s*1(\.0*)?\)", r"rgba(\1,0.85)", color_rgba)
        _add_label(x0 + (x1 - x0)/2, label, label_bg)
    except Exception:
        return

def add_one_day_band(date_val, label, color="Gold"):
    if date_val is None: return
    try:
        x0 = pd.to_datetime(date_val); x1 = x0 + pd.Timedelta(days=1)
        if x1 <= x0: return
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=color, opacity=0.25)
        _add_label(x0 + (x1 - x0)/2, label, "rgba(255,215,0,0.85)")
    except Exception:
        return

if show_nonres_bands:
    if pre_glifo:  add_one_day_band(pre_glifo_date, "Glifo (1d)")
    if pre_selNR:  add_residual_band(pre_selNR_date, NR_DAYS_DEFAULT, f"Sel. NR ({NR_DAYS_DEFAULT}d)", "rgba(219,112,147,1)")
    if pre_selR:   add_residual_band(pre_selR_date, pre_res_dias, "Residual PRESIEMBRA", "rgba(255,160,122,1)")
    if preem_selR: add_residual_band(preem_selR_date, preem_res_dias, "Residual PREEMERGENTE (0..+7)", "rgba(255,200,122,1)")
    if post_gram:  add_residual_band(post_gram_date, POST_GRAM_FORWARD_DAYS, "Graminicida (+10d)", "rgba(144,238,144,1)", alpha=0.25)
    if post_selR:  add_residual_band(post_selR_date, post_res_dias, "Residual POST (≥+20)", "rgba(173,216,230,1)")

if use_pc:
    fig.add_vrect(x0=PC_START, x1=PC_END, line_width=0, fillcolor="MediumPurple", opacity=0.12)
    fig.add_annotation(x=PC_START + (PC_END-PC_START)/2, y=1.04, xref="x", yref="paper",
                       text="Periodo crítico", showarrow=False, bgcolor="rgba(147,112,219,0.85)",
                       bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2)

if use_ciec and show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3",
                             hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Ciec: %{y:.2f}<extra></extra>"))

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f} · A2_ctrl={A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}")

# ======================= A2 / x en UI ======================
st.subheader(f"Densidad efectiva (x) y A2 (por AUC, cap={int(MAX_PLANTS_CAP)})")
st.markdown(
    f"""
**x₂ — Sin control (cap):** **{X2:,.1f}** pl·m²  
**x₃ — Con control (cap):** **{X3:,.1f}** pl·m²  
**A2 (sup, cap):** **{A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f}** pl·m²  
**A2 (ctrl, cap):** **{A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}** pl·m²
"""
)
# ================= Figura 2 y 3 (como antes) + Diagnóstico breve =================
if np.isfinite(X2) or np.isfinite(X3):
    x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
    y_curve = 0.375 * x_curve / (1.0 + (0.375 * x_curve / 76.639))
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x",
                                  hovertemplate="x = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"))
    if np.isfinite(X2):
        fig_loss.add_trace(go.Scatter(x=[X2], y=[loss_x2_pct], mode="markers+text", name="x₂: sin control (cap)",
                                      text=[f"x₂ = {X2:.1f}"], textposition="top center",
                                      marker=dict(size=10, symbol="diamond"),
                                      hovertemplate="x₂ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"))
    if np.isfinite(X3):
        fig_loss.add_trace(go.Scatter(x=[X3], y=[loss_x3_pct], mode="markers+text", name="x₃: con control (cap)",
                                      text=[f"x₃ = {X3:.1f}"], textposition="top right",
                                      marker=dict(size=11, symbol="star"),
                                      hovertemplate="x₃ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"))
    fig_loss.update_layout(title=f"Figura 2 — Pérdida de rendimiento (%) vs. x (cap A2={int(MAX_PLANTS_CAP)})",
                           xaxis_title="x (pl·m²) — integral de aportes (cohortes, cap) desde siembra",
                           yaxis_title="Pérdida de rendimiento (%)",
                           margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Figura 2: no hay valores finitos de x para graficar (revisar datos/siembra).")

st.subheader("Figura 3 — Composición porcentual por estado en el Periodo Crítico (PC)")
mask_pc_days = (ts >= PC_START) & (ts <= PC_END)
if factor_area_to_plants is None or not np.isfinite(factor_area_to_plants):
    st.info("AUC cruda = 0 → no se puede escalar a plantas·m²; no es posible calcular aportes en PC.")
else:
    mspc = (mask_since_sow & mask_pc_days).to_numpy()
    a_S1 = float(np.nansum(S1_pl_ctrl_cap[mspc] if 'S1_pl_ctrl_cap' in locals() else 0.0))
    a_S2 = float(np.nansum(S2_pl_ctrl_cap[mspc] if 'S2_pl_ctrl_cap' in locals() else 0.0))
    a_S3 = float(np.nansum(S3_pl_ctrl_cap[mspc] if 'S3_pl_ctrl_cap' in locals() else 0.0))
    a_S4 = float(np.nansum(S4_pl_ctrl_cap[mspc] if 'S4_pl_ctrl_cap' in locals() else 0.0))
    tot  = a_S1 + a_S2 + a_S3 + a_S4
    labels = ["S1 (FC=0.0)", "S2 (FC=0.3)", "S3 (FC=0.6)", "S4 (FC=1.0)"]
    if np.isfinite(tot) and tot > 0:
        pct = 100.0 * np.array([a_S1, a_S2, a_S3, a_S4], float) / tot
        st.markdown(f"**Ventana PC:** {PC_START.date()} → {PC_END.date()}  \n**Total (S1–S4) en PC:** **{tot:,.1f}** pl·m²")
        df_pc_pct = pd.DataFrame({"Estado": labels, "% del total PC": pct}).sort_values("% del total PC", ascending=False).reset_index(drop=True)
        st.dataframe(df_pc_pct, use_container_width=True)
        st.download_button("Descargar composición porcentual en PC (CSV)",
                           df_pc_pct.to_csv(index=False).encode("utf-8"),
                           "composicion_porcentual_estados_PC.csv","text/csv", key="dl_pct_estados_pc")
        fig_pc_donut = go.Figure(data=[go.Pie(labels=labels, values=pct, hole=0.5, textinfo="label+percent",
                                              hovertemplate="%{label}<br>%: %{value:.2f}%<extra></extra>")])
        fig_pc_donut.update_layout(title="Composición porcentual por estado en el Periodo Crítico (donut)",
                                   margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pc_donut, use_container_width=True)
    else:
        st.info("Figura 3: total en PC es 0 o no finito; no se puede calcular porcentaje.")

st.code(json.dumps({
    "siembra": str(sow_date),
    "tope_A2_plm2": int(MAX_PLANTS_CAP),
    "AUC_EMERREL_cruda_desde_siembra_dias": float(auc_cruda),
}, ensure_ascii=False, indent=2))

# ==================================================================================
# ======================= OPTIMIZACIÓN (extendida) =================================
# ==================================================================================
st.markdown("---")
st.header("🧠 Optimización")

with st.sidebar:
    st.header("Optimización (variables habilitadas)")
    st.caption("Se exploran siembras y cronogramas pre/post residuales, preemergente y graminicida.")

    # Rango de siembra
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 7, 1)

    # Tratamientos a considerar
    use_pre_selR_opt    = st.checkbox("Incluir PRESIEMBRA + residual (≤ siembra−14)", value=True)
    use_preem_selR_opt  = st.checkbox("Incluir PREEMERGENTE + residual (0..+7)", value=True)
    use_post_selR_opt   = st.checkbox("Incluir POST + residual (≥ siembra + 20 días)", value=True)
    use_post_gram_opt   = st.checkbox(f"Incluir graminicida post (+{POST_GRAM_FORWARD_DAYS-1}d)", value=True)

    # Eficiencias (constantes en búsqueda)
    ef_pre_selR_opt   = st.slider("Eficiencia PRESIEMBRA R (%)", 0, 100, 70, 1)  if use_pre_selR_opt else 0
    ef_preem_selR_opt = st.slider("Eficiencia PREEMERGENTE R (%)", 0, 100, 70, 1) if use_preem_selR_opt else 0
    ef_post_selR_opt  = st.slider("Eficiencia POST R (%)", 0, 100, 70, 1) if use_post_selR_opt else 0
    ef_post_gram_opt  = st.slider("Eficiencia graminicida post (%)", 0, 100, 65, 1) if use_post_gram_opt else 0

    # Ventanas / resolución
    pre_days_back  = st.number_input("PRESIEMBRA R: días antes de siembra (máx búsqueda)", 14, 180, 30, 1)  # buscaremos ≤ siembra−14 (ver abajo)
    pre_step_days  = st.number_input("Paso fechas PRESIEMBRA (días)", 1, 30, 7, 1)

    preem_step_days = st.number_input("Paso fechas PREEMERGENTE (días)", 1, 7, 1, 1)

    post_days_fw   = st.number_input("POST: días después de siembra (máximo)", 20, 180, 60, 1)
    post_step_days = st.number_input("Paso fechas POST (días)", 1, 30, 7, 1)

    res_min, res_max = st.slider("Residualidad (min–max) [días]", min_value=15, max_value=120, value=(30, 60), step=5)
    res_step = st.number_input("Paso de residualidad (días)", min_value=1, max_value=30, value=5, step=1)

    # Optimizador
    optimizer = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)

    # Límites y salida (máx. 100000)
    max_evals   = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100)
    top_k_show  = st.number_input("Top-k a mostrar", 1, 20, 5, 1)

    # Parámetros específicos de Recocido
    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 50000, 5000, 100)
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("Factor de enfriamiento (γ)", 0.80, 0.9999, 0.995, 0.0001)

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

# Validaciones rápidas
if sow_search_from > sow_search_to:
    st.error("Rango de siembra inválido (desde > hasta)."); st.stop()
if res_min >= res_max:
    st.error("Residualidad: el mínimo debe ser menor que el máximo."); st.stop()
if res_step <= 0:
    st.error("El paso de residualidad debe ser > 0."); st.stop()

# Helpers de optimización ------------------------------------------------
def daterange(start_date, end_date, step_days):
    out = []
    cur = pd.to_datetime(start_date); end = pd.to_datetime(end_date)
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(step_days))
    return out

# Datos base optimización (como antes)
ts_all   = pd.to_datetime(df_plot["fecha"])
fechas_d_all = ts_all.dt.date.values
emerrel_all  = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()

mode_canopy_opt = mode_canopy
t_lag_opt, t_close_opt = int(t_lag), int(t_close)
cov_max_opt, lai_max_opt, k_beer_opt = float(cov_max), float(lai_max), float(k_beer)
use_ciec_opt, Ca_opt, Cs_opt, LAIhc_opt = use_ciec, float(Ca), float(Cs), float(LAIhc)

def recompute_for_sow(sow_d: dt.date):
    mask_since = (ts_all.dt.date >= sow_d)
    FCx, LAIx = compute_canopy(ts_all, sow_d, mode_canopy_opt, t_lag_opt, t_close_opt, cov_max_opt, lai_max_opt, k_beer_opt)
    if use_ciec_opt:
        Ca_safe = Ca_opt if Ca_opt > 0 else 1e-6
        Cs_safe = Cs_opt if Cs_opt > 0 else 1e-6
        Ciec_loc = np.clip((LAIx / max(1e-6, LAIhc_opt)) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec_loc = np.zeros_like(LAIx, float)
    one_minus_Ciec_loc = np.clip(1.0 - Ciec_loc, 0.0, 1.0)

    births = np.where(mask_since.to_numpy(), emerrel_all, 0.0)
    s = pd.Series(births, index=ts_all)
    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).reindex(ts_all).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).reindex(ts_all).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).reindex(ts_all).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).reindex(ts_all).to_numpy(float)

    auc_cruda_loc = auc_time(ts_all, emerrel_all, mask=mask_since)
    if auc_cruda_loc <= 0:
        return None

    factor_area = MAX_PLANTS_CAP / auc_cruda_loc
    S1_pl = np.where(mask_since, S1 * one_minus_Ciec_loc * 0.0 * factor_area, 0.0)
    S2_pl = np.where(mask_since, S2 * one_minus_Ciec_loc * 0.3 * factor_area, 0.0)
    S3_pl = np.where(mask_since, S3 * one_minus_Ciec_loc * 0.6 * factor_area, 0.0)
    S4_pl = np.where(mask_since, S4 * one_minus_Ciec_loc * 1.0 * factor_area, 0.0)

    base_pl_daily = np.where(mask_since, emerrel_all * factor_area, 0.0)
    base_pl_daily_cap_loc = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since.to_numpy())
    sup_cap = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_pl_daily_cap_loc)

    return {
        "mask_since": mask_since.to_numpy(),
        "factor_area": factor_area,
        "auc_cruda": auc_cruda_loc,
        "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
        "sup_cap": sup_cap,
        "ts": ts_all,
        "fechas_d": fechas_d_all
    }

# ------------ Definición de acciones (incluye nuevas) ------------
def act_pre_selR(date_val, R, eff):
    return {"kind":"pre_selR",  "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}  # PRESIEMBRA R

def act_preem_selR(date_val, R, eff):
    return {"kind":"preem_selR","date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}  # PREEMERGENTE R (0..+7)

def act_post_selR(date_val, R, eff):
    return {"kind":"post_selR", "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}

def act_post_gram(date_val, eff):
    return {"kind":"post_gram", "date":pd.to_datetime(date_val).date(), "days":POST_GRAM_FORWARD_DAYS, "eff":eff, "states":["S1","S2","S3"]}

def perdida_rinde_pct_local(x): x = np.asarray(x, float); return 0.375 * x / (1.0 + (0.375 * x / 76.639))

# Restricciones duras en evaluate()
def evaluate(sd: dt.date, schedule: list):
    # Regla 1: post_selR >= siembra + 20 días
    sow_plus_20 = pd.to_datetime(sd) + pd.Timedelta(days=20)
    # Regla 2: pre_selR sólo ≤ siembra − 14
    sow_minus_14 = pd.to_datetime(sd) - pd.Timedelta(days=PRE_R_MAX_BEFORE_SOW_DAYS)
    # Regla 3: preem_selR en [siembra, siembra+7]
    sow_plus_7 = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)

    for a in schedule:
        ad = pd.to_datetime(a["date"])
        if a["kind"] == "post_selR" and ad < sow_plus_20:   return None
        if a["kind"] == "pre_selR"  and ad > sow_minus_14:  return None
        if a["kind"] == "preem_selR" and not (pd.to_datetime(sd) <= ad <= sow_plus_7): return None

    env = recompute_for_sow(sd)
    if env is None: return None
    mask_since = env["mask_since"]; factor_area = env["factor_area"]; auc_cruda_loc = env["auc_cruda"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]; sup_cap = env["sup_cap"]
    ts_local, fechas_d_local = env["ts"], env["fechas_d"]

    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def apply(weights, eff, states):
        if eff <= 0 or not len(states): return
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    for a in schedule:
        d0, d1 = a["date"], a["date"] + pd.Timedelta(days=int(a["days"]))
        w = ((fechas_d_local >= d0) & (fechas_d_local < d1)).astype(float)
        apply(w, a["eff"], a["states"])

    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    X2loc = float(np.nansum(sup_cap[mask_since]))
    X3loc = float(np.nansum(plantas_ctrl_cap[mask_since]))
    loss3 = float(perdida_rinde_pct_local(X3loc))

    sup_equiv  = np.divide(sup_cap,          factor_area, out=np.zeros_like(sup_cap),          where=(factor_area>0))
    ctrl_equiv = np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0))
    auc_sup      = auc_time(ts_local, sup_equiv,  mask=mask_since)
    auc_sup_ctrl = auc_time(ts_local, ctrl_equiv, mask=mask_since)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/auc_cruda_loc))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/auc_cruda_loc))

    return {"sow": sd, "loss_pct": loss3, "x2": X2loc, "x3": X3loc, "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

# ---------- Espacios de fechas por tratamiento (respetando ventanas) ----------
sow_candidates = daterange(sow_search_from, sow_search_to, sow_step_days)

def pre_dates_for(sow_d):
    # PRESIEMBRA R: sólo ≤ siembra−14 → generamos desde (siembra−pre_days_back) hasta (siembra−14)
    start = pd.to_datetime(sow_d) - pd.Timedelta(days=int(pre_days_back))
    end   = pd.to_datetime(sow_d) - pd.Timedelta(days=PRE_R_MAX_BEFORE_SOW_DAYS)
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(pre_step_days))
    return out

def preem_dates_for(sow_d):
    # PREEMERGENTE R: en [siembra, siembra+7]
    start = pd.to_datetime(sow_d)
    end   = pd.to_datetime(sow_d) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(preem_step_days))
    return out

def post_dates_for(sow_d):
    start = pd.to_datetime(sow_d) + pd.Timedelta(days=20)  # regla (≥ siembra + 20)
    end   = pd.to_datetime(sow_d) + pd.Timedelta(days=int(post_days_fw))
    if end < start: return []
    cur, out = start, []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=int(post_step_days))
    return out

res_days = list(range(int(res_min), int(res_max) + 1, int(res_step)))
if int(res_max) not in res_days: res_days.append(int(res_max))

status_ph = st.empty()
prog_ph = st.empty()
results = []

def build_all_scenarios():
    scenarios = []
    for sd in sow_candidates:
        pre_dates    = pre_dates_for(sd)
        preem_dates  = preem_dates_for(sd)
        post_dates   = post_dates_for(sd)
        groups = []
        if use_pre_selR_opt and pre_dates:
            groups.append([act_pre_selR(d, R, ef_pre_selR_opt) for d in pre_dates for R in res_days])
        if use_preem_selR_opt and preem_dates:
            groups.append([act_preem_selR(d, R, ef_preem_selR_opt) for d in preem_dates for R in res_days])
        if use_post_selR_opt and post_dates:
            groups.append([act_post_selR(d, R, ef_post_selR_opt) for d in post_dates for R in res_days])
        if use_post_gram_opt and post_dates:
            groups.append([act_post_gram(d, ef_post_gram_opt) for d in post_dates])
        combos = [[]]
        for r in range(1, len(groups)+1):
            for subset in itertools.combinations(range(len(groups)), r):
                prod = itertools.product(*[groups[i] for i in subset])
                combos.extend([list(p) for p in prod])
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random_scenario():
    sd = random.choice(sow_candidates)
    pre_dates   = pre_dates_for(sd)
    preem_dates = preem_dates_for(sd)
    post_dates  = post_dates_for(sd)
    schedule = []
    if use_pre_selR_opt and pre_dates and random.random()<0.7:
        d = random.choice(pre_dates); R = random.choice(res_days)
        schedule.append(act_pre_selR(d, R, ef_pre_selR_opt))
    if use_preem_selR_opt and preem_dates and random.random()<0.7:
        d = random.choice(preem_dates); R = random.choice(res_days)
        schedule.append(act_preem_selR(d, R, ef_preem_selR_opt))
    if use_post_selR_opt and post_dates and random.random()<0.7:
        d = random.choice(post_dates); R = random.choice(res_days)
        schedule.append(act_post_selR(d, R, ef_post_selR_opt))
    if use_post_gram_opt and post_dates and random.random()<0.7:
        d = random.choice(post_dates)
        schedule.append(act_post_gram(d, ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

def neighbors_state(current):
    sd, sch = current
    idx = sow_candidates.index(pd.Timestamp(sd))
    moves = []
    # Mover fecha de siembra
    for delta in [-1, 1]:
        j = idx + delta
        if 0 <= j < len(sow_candidates):
            sd2 = sow_candidates[j].date()
            pre_d2   = pre_dates_for(sd2)
            preem_d2 = preem_dates_for(sd2)
            post_d2  = post_dates_for(sd2)
            sch2 = []
            for a in sch:
                if a["kind"] == "pre_selR" and pre_d2:
                    best = min(pre_d2, key=lambda d: abs((pd.to_datetime(a["date"]) - d).days))
                    sch2.append(act_pre_selR(best, a["days"], a["eff"]))
                elif a["kind"] == "preem_selR" and preem_d2:
                    best = min(preem_d2, key=lambda d: abs((pd.to_datetime(a["date"]) - d).days))
                    sch2.append(act_preem_selR(best, a["days"], a["eff"]))
                elif a["kind"] == "post_selR" and post_d2:
                    best = min(post_d2, key=lambda d: abs((pd.to_datetime(a["date"]) - d).days))
                    sch2.append(act_post_selR(best, a["days"], a["eff"]))
                elif a["kind"] == "post_gram" and post_d2:
                    best = min(post_d2, key=lambda d: abs((pd.to_datetime(a["date"]) - d).days))
                    sch2.append(act_post_gram(best, a["eff"]))
            moves.append((sd2, sch2))

    # Mutaciones locales
    pre_d   = pre_dates_for(pd.Timestamp(sd))
    preem_d = preem_dates_for(pd.Timestamp(sd))
    post_d  = post_dates_for(pd.Timestamp(sd))

    # PRESIEMBRA R
    if use_pre_selR_opt and pre_d:
        sch2 = [x for x in sch if x["kind"]!="pre_selR"]
        if any(x["kind"]=="pre_selR" for x in sch):
            action = random.choice(["date","dur","drop"])
            if action=="date":
                d = random.choice(pre_d); R = [x for x in sch if x["kind"]=="pre_selR"][0]["days"]
                moves.append((sd, sch2+[act_pre_selR(d, R, ef_pre_selR_opt)]))
            elif action=="dur":
                R = random.choice(res_days); d = [x for x in sch if x["kind"]=="pre_selR"][0]["date"]
                moves.append((sd, sch2+[act_pre_selR(d, R, ef_pre_selR_opt)]))
            else:
                moves.append((sd, sch2))
        else:
            d = random.choice(pre_d); R = random.choice(res_days)
            moves.append((sd, sch+[act_pre_selR(d, R, ef_pre_selR_opt)]))

    # PREEMERGENTE R
    if use_preem_selR_opt and preem_d:
        sch2 = [x for x in sch if x["kind"]!="preem_selR"]
        if any(x["kind"]=="preem_selR" for x in sch):
            action = random.choice(["date","dur","drop"])
            if action=="date":
                d = random.choice(preem_d); R = [x for x in sch if x["kind"]=="preem_selR"][0]["days"]
                moves.append((sd, sch2+[act_preem_selR(d, R, ef_preem_selR_opt)]))
            elif action=="dur":
                R = random.choice(res_days); d = [x for x in sch if x["kind"]=="preem_selR"][0]["date"]
                moves.append((sd, sch2+[act_preem_selR(d, R, ef_preem_selR_opt)]))
            else:
                moves.append((sd, sch2))
        else:
            d = random.choice(preem_d); R = random.choice(res_days)
            moves.append((sd, sch+[act_preem_selR(d, R, ef_preem_selR_opt)]))

    # POST R
    if use_post_selR_opt and post_d:
        sch2 = [x for x in sch if x["kind"]!="post_selR"]
        if any(x["kind"]=="post_selR" for x in sch):
            action = random.choice(["date","dur","drop"])
            if action=="date":
                d = random.choice(post_d); R = [x for x in sch if x["kind"]=="post_selR"][0]["days"]
                moves.append((sd, sch2+[act_post_selR(d, R, ef_post_selR_opt)]))
            elif action=="dur":
                R = random.choice(res_days); d = [x for x in sch if x["kind"]=="post_selR"][0]["date"]
                moves.append((sd, sch2+[act_post_selR(d, R, ef_post_selR_opt)]))
            else:
                moves.append((sd, sch2))
        else:
            d = random.choice(post_d); R = random.choice(res_days)
            moves.append((sd, sch+[act_post_selR(d, R, ef_post_selR_opt)]))

    # POST graminicida
    if use_post_gram_opt and post_d:
        sch2 = [x for x in sch if x["kind"]!="post_gram"]
        if any(x["kind"]=="post_gram" for x in sch):
            action = random.choice(["date","drop"])
            if action=="date":
                d = random.choice(post_d)
                moves.append((sd, sch2+[act_post_gram(d, ef_post_gram_opt)]))
            else:
                moves.append((sd, sch2))
        else:
            d = random.choice(post_d)
            moves.append((sd, sch+[act_post_gram(d, ef_post_gram_opt)]))

    random.shuffle(moves)
    return moves

# ------------- Bucle del optimizador -------------
if factor_area_to_plants is None or not np.isfinite(auc_cruda):
    st.info("Primero necesitás AUC(EMERREL cruda) > 0 desde alguna siembra evaluada para escalar a plantas·m².")
else:
    if st.session_state.opt_running:
        status_ph.info("Optimizando…")
        if optimizer == "Grid (combinatorio)":
            scenarios = build_all_scenarios()
            total = len(scenarios)
            st.caption(f"Se evaluarán {total:,} configuraciones (siembra + cronogramas)")
            if total > max_evals:
                random.seed(123)
                scenarios = random.sample(scenarios, k=int(max_evals))
                st.caption(f"Se muestrean {len(scenarios):,} configs (límite máx. evaluaciones)")
            prog = prog_ph.progress(0.0); n = len(scenarios); step = max(1, n//100)
            for i,(sd,sch) in enumerate(scenarios,1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Optimización detenida. Progreso: {i-1:,}/{n:,}")
                    break
                r = evaluate(sd, sch)
                if r is not None: results.append(r)
                if i % step == 0 or i == n: prog.progress(min(1.0, i/n))
            prog_ph.empty()

        elif optimizer == "Búsqueda aleatoria":
            N = int(max_evals)
            prog = prog_ph.progress(0.0)
            for i in range(1, N+1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Optimización detenida. Progreso: {i-1:,}/{N:,}")
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
                status_ph.error("No fue posible encontrar un estado inicial válido.")
            else:
                best_eval = cur_eval
                cur_loss = cur_eval["loss_pct"]
                T = float(sa_T0)
                prog = prog_ph.progress(0.0)
                for it in range(1, int(sa_iters)+1):
                    if st.session_state.opt_stop:
                        status_ph.warning(f"Optimización detenida en iteración {it-1:,}/{int(sa_iters):,}.")
                        break
                    neighs = neighbors_state(cur)
                    if not neighs:
                        neighs = [sample_random_scenario()]
                    cand = random.choice(neighs)
                    cand_eval = evaluate(*cand)
                    if cand_eval is not None:
                        cand_loss = cand_eval["loss_pct"]
                        d = cand_loss - cur_loss
                        if d <= 0 or random.random() < _math.exp(-d / max(1e-9, T)):
                            cur, cur_eval, cur_loss = cand, cand_eval, cand_loss
                            results.append(cur_eval)
                            if cand_loss < best_eval["loss_pct"]:
                                best_eval = cand_eval
                        else:
                            results.append(cand_eval)
                    T = T * float(sa_cooling)
                    if it % max(1, int(sa_iters)//100) == 0 or it == int(sa_iters):
                        prog.progress(min(1.0, it/float(sa_iters)))
                results.append(best_eval)
                prog_ph.empty()

        st.session_state.opt_running = False
        st.session_state.opt_stop = False
        status_ph.success("Optimización finalizada.")
    else:
        status_ph.info("Listo para optimizar. Ajustá los parámetros y presioná **Iniciar**.")

# ---------------------------- Reporte de resultados ----------------------------
if results:
    results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))
    best = results_sorted[0]

    st.subheader("🏆 Mejor escenario")
    st.markdown(
        f"**Siembra:** **{best['sow']}**  \n"
        f"**Pérdida estimada:** **{best['loss_pct']:.2f}%**  \n"
        f"**x₂:** {best['x2']:.1f} · **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"**A2_sup:** {best['A2_sup']:.1f} · **A2_ctrl:** {best['A2_ctrl']:.1f} pl·m²"
    )

    def schedule_df(sch):
        rows=[]
        for a in sch:
            ini = pd.to_datetime(a["date"]); fin = ini + pd.Timedelta(days=int(a["days"]))
            label = {"pre_selR":"PRESIEMBRA R","preem_selR":"PREEMERGENTE R","post_selR":"POST R","post_gram":"POST grami"}.get(a["kind"], a["kind"])
            rows.append({
                "Intervención": label,
                "Inicio": str(ini.date()),
                "Fin": str(fin.date()),
                "Duración (d)": int(a["days"]),
                "Eficiencia (%)": int(a["eff"]),
                "Estados": ",".join(a["states"])
            })
        return pd.DataFrame(rows)

    df_best = schedule_df(best["schedule"])
    if len(df_best):
        st.dataframe(df_best, use_container_width=True)
        st.download_button("Descargar mejor cronograma (CSV)", df_best.to_csv(index=False).encode("utf-8"),
                           "mejor_cronograma.csv", "text/csv", key="dl_mejor_crono")

    def row_of(r):
        key = " | ".join([f"{ {'pre_selR':'PRESIEMBRA R','preem_selR':'PREEMERGENTE R','post_selR':'POST R','post_gram':'POST grami'}.get(a['kind'],a['kind']) }@{a['date']}+{int(a['days'])}d({int(a['eff'])}%)" for a in r["schedule"]]) if r["schedule"] else "Sin manejo"
        return [str(r["sow"]), key, r["loss_pct"], r["x2"], r["x3"], r["A2_sup"], r["A2_ctrl"]]

    top_k = int(top_k_show)
    top_rows = [row_of(r) for r in results_sorted[:top_k]]
    df_top = pd.DataFrame(top_rows, columns=["Siembra", "Escenario", "Pérdida (%)", "x₂", "x₃", "A2_sup", "A2_ctrl"])
    if optimizer == "Grid (combinatorio)":
        st.caption("⚠️ Si se muestreó por límite de evaluaciones o cortaste, pueden ser resultados parciales.")
    st.subheader(f"Top-{top_k} escenarios")
    st.dataframe(df_top, use_container_width=True)
    st.download_button("Descargar Top-k (CSV)", df_top.to_csv(index=False).encode("utf-8"),
                       "topk_siembra_cronogramas.csv", "text/csv", key="dl_topk")
else:
    st.info("Aún no hay resultados de optimización para mostrar.")
# =============================================================================
# Sección — Gráficos del mejor escenario (post-optimizador)
# Requiere: `best` (dict), `recompute_for_sow` (func), `emerrel_all`, `compute_canopy`,
#           y parámetros de canopia + flags de UI ya definidos en tu script.
# =============================================================================
def _apply_schedule_to_states(fechas_d, schedule):
    """Devuelve multiplicadores c1..c4 aplicados por el cronograma (peso=1 en ventana)."""
    c1 = np.ones_like(fechas_d, float)
    c2 = np.ones_like(fechas_d, float)
    c3 = np.ones_like(fechas_d, float)
    c4 = np.ones_like(fechas_d, float)

    def _apply(weights_mask, eff, states):
        if eff <= 0 or not states:
            return
        # Reducción multiplicativa dentro de la ventana (peso=1)
        reduc = np.ones_like(weights_mask, float)
        reduc[weights_mask] = 1.0 - (eff / 100.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    for a in (schedule or []):
        ini = pd.to_datetime(a["date"]).date()
        fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
        mask = (fechas_d >= ini) & (fechas_d < fin)
        _apply(mask, float(a.get("eff", 0)), a.get("states", []))

    return c1, c2, c3, c4


def _weekly_sum(df_daily, ycols):
    """Resample semanal (W-MON) sumando columnas `ycols`."""
    return (
        df_daily.set_index("fecha")[ycols]
        .resample("W-MON").sum()
        .reset_index()
    )


def _loss_pct(x):
    x = float(x)
    return 0.375 * x / (1.0 + (0.375 * x / 76.639))


def draw_best_scenario(best):
    try:
        # 1) Recomputar entorno con la siembra óptima
        sow_best = pd.to_datetime(best["sow"]).date()
        env = recompute_for_sow(sow_best)
        if env is None:
            st.info("No se pudieron recomputar series para el mejor escenario.")
            return

        ts_b         = env["ts"]
        fechas_d_b   = env["fechas_d"]
        mask_since_b = env["mask_since"]
        factor_area  = env["factor_area"]
        auc_cruda_b  = env["auc_cruda"]
        S1p, S2p, S3p, S4p = env["S_pl"]
        sup_cap_b    = env["sup_cap"]  # sin control, con cap estricto

        # 2) Aplicar cronograma del mejor escenario a S1..S4
        c1, c2, c3, c4 = _apply_schedule_to_states(fechas_d_b, best.get("schedule", []))
        total_ctrl_daily = (S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4)

        # Reescala proporcional para respetar el cap por día
        eps = 1e-12
        scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, sup_cap_b / total_ctrl_daily), 0.0)
        S1_ctrl_cap_b = S1p * c1 * scale
        S2_ctrl_cap_b = S2p * c2 * scale
        S3_ctrl_cap_b = S3p * c3 * scale
        S4_ctrl_cap_b = S4p * c4 * scale
        plantas_ctrl_cap_b = S1_ctrl_cap_b + S2_ctrl_cap_b + S3_ctrl_cap_b + S4_ctrl_cap_b

        # Serie base (cap) para referencia
        base_pl_daily_b = np.where(mask_since_b, emerrel_all * factor_area, 0.0)
        base_pl_daily_cap_b = cap_cumulative(base_pl_daily_b, MAX_PLANTS_CAP, mask_since_b)

        # 3) Tablas diarias y semanales
        df_daily_b = pd.DataFrame({
            "fecha": ts_b,
            "sin_ctrl_cap": np.where(mask_since_b, sup_cap_b, 0.0),
            "con_ctrl_cap": np.where(mask_since_b, plantas_ctrl_cap_b, 0.0),
            "base_cap":     np.where(mask_since_b, base_pl_daily_cap_b, 0.0),
            "S1": np.where(mask_since_b, S1_ctrl_cap_b, 0.0),
            "S2": np.where(mask_since_b, S2_ctrl_cap_b, 0.0),
            "S3": np.where(mask_since_b, S3_ctrl_cap_b, 0.0),
            "S4": np.where(mask_since_b, S4_ctrl_cap_b, 0.0),
        })
        df_week_b = _weekly_sum(df_daily_b, ["sin_ctrl_cap", "con_ctrl_cap"])
        df_states_week_b = _weekly_sum(df_daily_b, ["S1", "S2", "S3", "S4"])

        # 4) Ciec (opcional)
        Ciec_b = None
        if "use_ciec" in globals() and use_ciec:
            FC_b, LAI_b = compute_canopy(
                ts_b, sow_best, mode_canopy, int(t_lag), int(t_close),
                float(cov_max), float(lai_max), float(k_beer)
            )
            Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
            Cs_safe = float(Cs) if float(Cs) > 0 else 1e-6
            Ciec_b = np.clip((LAI_b / max(1e-6, float(LAIhc))) * (Ca_safe / Cs_safe), 0.0, 1.0)

        # 5) Gráfico 1 — EMERREL + aportes semanales (mejor escenario)
        st.subheader(f"📊 Gráfico 1 — Mejor escenario (siembra {sow_best})")
        fig_best1 = go.Figure()
        # EMERREL cruda (línea izquierda)
        fig_best1.add_trace(go.Scatter(
            x=ts_b, y=emerrel_all, mode="lines", name="EMERREL (cruda)",
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL: %{y:.4f}<extra></extra>"
        ))
        layout_kwargs_best = dict(
            margin=dict(l=10, r=10, t=40, b=10),
            title=f"EMERREL (izq) y Plantas·m²·semana (der) · Tope={int(MAX_PLANTS_CAP)} · (mejor)",
            xaxis_title="Tiempo", yaxis_title="EMERREL",
        )
        if "show_plants_axis" in globals() and show_plants_axis:
            layout_kwargs_best["yaxis2"] = dict(
                overlaying="y", side="right",
                title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                position=1.0, range=[0, 100], tick0=0, dtick=20, showgrid=False
            )
            fig_best1.add_trace(go.Scatter(
                x=df_week_b["fecha"], y=df_week_b["sin_ctrl_cap"],
                name="Aporte semanal (sin control, cap) — mejor",
                yaxis="y2", mode="lines+markers",
                hovertemplate="Lunes: %{x|%Y-%m-%d}<br>pl·m²·sem⁻¹: %{y:.2f}<extra></extra>"
            ))
            fig_best1.add_trace(go.Scatter(
                x=df_week_b["fecha"], y=df_week_b["con_ctrl_cap"],
                name="Aporte semanal (con control, cap) — mejor",
                yaxis="y2", mode="lines+markers", line=dict(dash="dot"),
                hovertemplate="Lunes: %{x|%Y-%m-%d}<br>pl·m²·sem⁻¹: %{y:.2f}<extra></extra>"
            ))
        if (Ciec_b is not None) and ("show_ciec_curve" in globals()) and show_ciec_curve:
            fig_best1.update_layout(yaxis3=dict(
                overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]
            ))
            fig_best1.add_trace(go.Scatter(
                x=ts_b, y=Ciec_b, mode="lines", name="Ciec (mejor)", yaxis="y3",
                hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Ciec: %{y:.2f}<extra></extra>"
            ))
        # Bandas del cronograma óptimo
        if len(best.get("schedule", [])) > 0:
            for a in best["schedule"]:
                x0 = pd.to_datetime(a["date"])
                x1 = x0 + pd.Timedelta(days=int(a["days"]))
                fig_best1.add_vrect(
                    x0=x0, x1=x1, line_width=0, fillcolor="rgba(30,144,255,0.18)", opacity=0.18
                )
                fig_best1.add_annotation(
                    x=x0 + (x1-x0)/2, y=0.86, xref="x", yref="paper",
                    text=a["kind"], showarrow=False, bgcolor="rgba(30,144,255,0.85)",
                    bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2
                )
        fig_best1.update_layout(**layout_kwargs_best)
        st.plotly_chart(fig_best1, use_container_width=True)
        st.caption(
            f"AUC(EMERREL cruda desde siembra óptima) = {auc_cruda_b:.4f} → "
            f"{int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area:.4f}). Series semanales según cronograma óptimo."
        )

        # 6) Figura 2 — Pérdida (%) vs x con puntos x2/x3 del mejor
        X2_b = float(np.nansum(sup_cap_b[mask_since_b]))
        X3_b = float(np.nansum(plantas_ctrl_cap_b[mask_since_b]))
        loss_x2_b = _loss_pct(X2_b)
        loss_x3_b = _loss_pct(X3_b)

        st.subheader(f"Figura 2 — Pérdida (%) vs x · Mejor escenario (siembra {sow_best})")
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
        y_curve = 0.375 * x_curve / (1.0 + (0.375 * x_curve / 76.639))
        fig2_best = go.Figure()
        fig2_best.add_trace(go.Scatter(
            x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x",
            hovertemplate="x = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
        ))
        fig2_best.add_trace(go.Scatter(
            x=[X2_b], y=[loss_x2_b], mode="markers+text", name="x₂: sin control (cap)",
            text=[f"x₂ = {X2_b:.1f}"], textposition="top center",
            marker=dict(size=10, symbol="diamond"),
            hovertemplate="x₂ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
        ))
        fig2_best.add_trace(go.Scatter(
            x=[X3_b], y=[loss_x3_b], mode="markers+text", name="x₃: con control (cap)",
            text=[f"x₃ = {X3_b:.1f}"], textposition="top right",
            marker=dict(size=11, symbol="star"),
            hovertemplate="x₃ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
        ))
        fig2_best.update_layout(
            title=f"Pérdida de rendimiento (%) vs. x (cap A2={int(MAX_PLANTS_CAP)})",
            xaxis_title="x (pl·m²) — integral de aportes (cohortes, cap) desde siembra",
            yaxis_title="Pérdida de rendimiento (%)",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2_best, use_container_width=True)

        # 7) Figura 3 — Donut PC con % por estado (mejor)
        year_pc_best = sow_best.year
        PC_START_best = pd.to_datetime(f"{year_pc_best}-09-11")
        PC_END_best   = pd.to_datetime(f"{year_pc_best}-11-15")
        st.subheader(f"Figura 3 — Composición porcentual por estado en el PC (siembra {sow_best})")
        st.caption(f"Ventana PC: {PC_START_best.date()} → {PC_END_best.date()}")

        mask_pc_days_best = (ts_b >= PC_START_best) & (ts_b <= PC_END_best)
        mspc_b = (mask_since_b & mask_pc_days_best.to_numpy())
        a_S1 = float(np.nansum(S1_ctrl_cap_b[mspc_b]))
        a_S2 = float(np.nansum(S2_ctrl_cap_b[mspc_b]))
        a_S3 = float(np.nansum(S3_ctrl_cap_b[mspc_b]))
        a_S4 = float(np.nansum(S4_ctrl_cap_b[mspc_b]))
        tot_b = a_S1 + a_S2 + a_S3 + a_S4

        if np.isfinite(tot_b) and tot_b > 0:
            labels = ["S1 (FC=0.0)", "S2 (FC=0.3)", "S3 (FC=0.6)", "S4 (FC=1.0)"]
            pct_b = 100.0 * np.array([a_S1, a_S2, a_S3, a_S4], float) / tot_b
            df_pc_best = pd.DataFrame({"Estado": labels, "% del total PC": pct_b}) \
                            .sort_values("% del total PC", ascending=False).reset_index(drop=True)
            st.dataframe(df_pc_best, use_container_width=True)
            st.download_button(
                "Descargar composición (mejor) en PC (CSV)",
                df_pc_best.to_csv(index=False).encode("utf-8"),
                "composicion_PC_mejor_escenario.csv",
                "text/csv",
                key="dl_pct_estados_pc_best"
            )
            fig3_best = go.Figure(data=[go.Pie(
                labels=labels, values=pct_b, hole=0.5, textinfo="label+percent",
                hovertemplate="%{label}<br>%: %{value:.2f}%<extra></extra>"
            )])
            fig3_best.update_layout(
                title="Composición porcentual por estado en el Periodo Crítico (mejor escenario)",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig3_best, use_container_width=True)
        else:
            st.info("En la ventana del PC del mejor escenario, el total es 0 o no finito (no se puede armar el donut).")

        # 8) Figura 4 — Aportes semanales por estado (stack) · mejor
        st.subheader("Figura 4 — Dinámica temporal de S1–S4 (mejor escenario)")
        fig_states = go.Figure()
        fig_states.add_trace(go.Scatter(
            x=df_states_week_b["fecha"], y=df_states_week_b["S1"], mode="lines",
            name="S1 (FC=0.0)", stackgroup="one",
            hovertemplate="Lunes: %{x|%Y-%m-%d}<br>S1: %{y:.2f} pl·m²·sem⁻¹<extra></extra>"
        ))
        fig_states.add_trace(go.Scatter(
            x=df_states_week_b["fecha"], y=df_states_week_b["S2"], mode="lines",
            name="S2 (FC=0.3)", stackgroup="one",
            hovertemplate="Lunes: %{x|%Y-%m-%d}<br>S2: %{y:.2f} pl·m²·sem⁻¹<extra></extra>"
        ))
        fig_states.add_trace(go.Scatter(
            x=df_states_week_b["fecha"], y=df_states_week_b["S3"], mode="lines",
            name="S3 (FC=0.6)", stackgroup="one",
            hovertemplate="Lunes: %{x|%Y-%m-%d}<br>S3: %{y:.2f} pl·m²·sem⁻¹<extra></extra>"
        ))
        fig_states.add_trace(go.Scatter(
            x=df_states_week_b["fecha"], y=df_states_week_b["S4"], mode="lines",
            name="S4 (FC=1.0)", stackgroup="one",
            hovertemplate="Lunes: %{x|%Y-%m-%d}<br>S4: %{y:.2f} pl·m²·sem⁻¹<extra></extra>"
        ))
        fig_states.update_layout(
            title="Aportes semanales por estado (con control + cap) · pl·m²·sem⁻¹",
            xaxis_title="Tiempo (semana iniciada en Lunes)",
            yaxis_title="pl·m²·sem⁻¹",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_states, use_container_width=True)

    except Exception as e:
        st.warning(f"No fue posible dibujar los gráficos del mejor escenario: {e}")


# --- Llamada (colocá esto donde ya existe `best` tras ordenar `results`) ---
try:
    if 'best' in locals() and isinstance(best, dict):
        draw_best_scenario(best)
except Exception as _e:
    st.warning(f"No fue posible invocar draw_best_scenario: {_e}")






