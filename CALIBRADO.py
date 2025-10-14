# -*- coding: utf-8 -*-
# PREDWEEM — Simulador calibrado (encadenado) + Timeline + Pérdida de rinde
# ----------------------------------------------------------
# Cambios clave:
# - Modelo encadenado diario S1→S2→S3→S4 (flujo poblacional):
#   * S1 recibe nacimientos (EMERREL escalada).
#   * Cada día transfiere una fracción 1/D_i a la siguiente caja (tiempo medio de residencia).
#   * Los controles reducen cada estado en su ventana y ANTES de transferir.
#   * Si S1 queda en 0 por control, NO fluye nada a S2 (ni a S3/S4).
# - Ventanas de aplicación estrictas:
#   * presiembra+residual: <= siembra−14; actúa S1–S2
#   * preemergente+residual: [siembra..siembra+10]; actúa S1–S2
#   * graminicida post (NR): [siembra..siembra+10]; actúa S1–S3
#   * post+residual: >= siembra+20; actúa S1–S4
# - Timeline de tratamientos (sin etiqueta de overlapping).
# - Curva de pérdida de rinde + puntos x2/x3.
#
# Parámetros calibrados fijos:
# - Pesos por estado: w1=0.369, w2=0.393, w3=1.150, w4=1.769
# - Pérdida: loss(x) = α*x / (1 + α*x/Lmax), α=0.503, Lmax=125.91
# - Tope A2 seleccionable: 250/125/62 pl·m²

import io, re, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ========= parámetros calibrados =========
ALPHA = 0.503
LMAX  = 125.91
W_S   = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

def loss_fun(x):
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

# ========= estado UI =========
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · Simulador (encadenado) + Timeline"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ========= constantes y reglas =========
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10

# ========= utilidades de lectura =========
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

# ========= tope A2 =========
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox(
        "Tope de densidad efectiva (pl·m²)",
        options=[250, 125, 62], index=0,
        help="Tope único de densidad efectiva y A2 (calibrado)."
    ))
st.caption(
    f"AUC(EMERREL cruda) ≙ tope A2 **= {int(MAX_PLANTS_CAP)} pl·m²**. "
    "Estados S1..S4 encadenados. Salidas en pl·m²·sem⁻¹ con cap."
)

# ========= entrada de datos =========
with st.sidebar:
    st.header("Datos de entrada")
    up  = st.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])
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
        if not raw or len(raw)==0: st.error("El archivo/URL está vacío."); st.stop()
        df0, meta = parse_csv(raw, sep_opt, dec_opt)
    if df0.empty: st.error("El archivo no tiene filas."); st.stop()
    st.success(f"Entrada leída. sep='{meta['sep']}' dec='{meta['dec']}' enc='{meta['enc']}'")
except (URLError, HTTPError) as e:
    st.error(f"No se pudo acceder a la URL: {e}"); st.stop()
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}"); st.stop()

# ========= selección/depuración =========
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

# ========= siembra & canopia =========
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

# ========= máscaras/fechas =========
ts = pd.to_datetime(df_plot["fecha"])            # datetime64[ns]
dates_d = ts.dt.date.values                      # np.array(datetime.date)
mask_since_sow = (dates_d >= sow_date)

# ========= equivalencia por área (A2) =========
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ========= controles (UI) =========
with st.sidebar:
    st.header("Manejo (manual)")
    pre_glifo = st.checkbox("Herbicida total (glifosato, pre)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=dates_d[0], min_value=dates_d[0], max_value=dates_d[-1], disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre, NR 10d)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=dates_d[0], min_value=dates_d[0], max_value=dates_d[-1], disabled=not pre_selNR)

    preR = st.checkbox("Selectivo + residual (presiembra)", value=False,
                       help="Solo permitido hasta siembra−14 días. (Actúa S1–S2)")
    preR_days = st.slider("Residualidad presiembra (días)", 15, 120, 45, 1, disabled=not preR)
    preR_max = (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW))
    preR_date = st.date_input("Fecha selectivo + residual (presiembra)",
                              value=min(dates_d[0], preR_max),
                              min_value=dates_d[0],
                              max_value=min(preR_max, dates_d[-1]),
                              disabled=not preR)

with st.sidebar:
    st.header("Manejo preemergente / post")
    preemR = st.checkbox("Selectivo + residual (preemergente)", value=False,
                         help="Ventana [siembra..siembra+10]. (Actúa S1–S2)")
    preemR_days = st.slider("Residualidad preemergente (días)", 15, 120, 45, 1, disabled=not preemR)
    preem_min = sow_date
    preem_max = min(dates_d[-1], sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preemR_date = st.date_input("Fecha selectivo + residual (preemergente)",
                                value=preem_min, min_value=preem_min, max_value=preem_max, disabled=not preemR)

    post_gram = st.checkbox("Selectivo graminicida (post, +10d)", value=False,
                            help="Ventana [siembra..siembra+10]. (Actúa S1–S3)")
    post_gram_date = st.date_input("Fecha graminicida (post)",
                                   value=max(dates_d[0], sow_date),
                                   min_value=dates_d[0], max_value=dates_d[-1],
                                   disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False,
                            help="≥ siembra + 20 días. (Actúa S1–S4)")
    post_min_postR = max(dates_d[0], sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)",
                                   value=post_min_postR, min_value=post_min_postR, max_value=dates_d[-1], disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

# Eficiencias
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 70, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 70, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 70, 1) if post_selR else 0

# Validaciones “duras” de ventanas
warnings = []
def check_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debería ser ≤ fecha de siembra ({sow_date}).")
def check_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debería ser ≥ fecha de siembra ({sow_date}).")
if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if preR and preR_date > (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
    warnings.append(f"Presiembra residual debe ser ≤ siembra−{PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW} ({sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)}).")
if preemR and (preemR_date < sow_date or preemR_date > sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
    warnings.append(f"Preemergente residual debe estar entre siembra y siembra+{PREEM_R_MAX_AFTER_SOW_DAYS}.")
if post_gram:  check_post(post_gram_date, "Graminicida (post)")
if post_selR and post_selR_date and post_selR_date < sow_date + timedelta(days=20):
    warnings.append(f"Selectivo + residual (post): debe ser ≥ {sow_date + timedelta(days=20)}.")
for w in warnings: st.warning(w)

# ========= decaimiento en residuales =========
with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None
if decaimiento_tipo != "Exponencial": lam_exp = None

# ========= helpers de pesos (ventanas -> arrays diarios) =========
def weights_one_day(d0: date, dates_d: np.ndarray):
    if not d0: return np.zeros_like(dates_d, float)
    return (dates_d == d0).astype(float)

def weights_residual(start_date: date, dias: int, dates_d: np.ndarray):
    w = np.zeros_like(dates_d, float)
    if (not start_date) or (not dias) or (int(dias) <= 0): return w
    d0 = start_date; d1 = start_date + timedelta(days=int(dias))
    mask = (dates_d >= d0) & (dates_d < d1)
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

# ========= simulación encadenada =========
def simulate_chained(emerrel_series: pd.Series,
                     sow_date: date,
                     one_minus_ciec: np.ndarray,
                     factor_area_to_plants: float,
                     MAX_PLANTS_CAP: float,
                     D1: int = 6, D2: int = 21, D3: int = 32):
    """
    Modelo encadenado diario S1→S2→S3→S4 (dwell-time promedio D1..D3).
    - Nacimientos (base) = EMERREL * factor_area_to_plants, solo desde siembra.
    - Se aplica Ciec multiplicativo a nacimientos (competencia del cultivo) ANTES de entrar a S1.
    - Tope A2 se implementa sobre nacimientos (cap acumulado).
    - Controles: multiplicativos por estado y día, antes de transferencias.
    Devuelve dict con series diarias y semanales (baseline vs control).
    """
    # ---------------- base temporal ----------------
    ts_local = emerrel_series.index.to_series()
    dates_local = ts_local.dt.date.values
    n = len(ts_local)
    ms = (dates_local >= sow_date)

    # ---------------- nacimientos escalados ----------------
    births_base = emerrel_series.to_numpy(float) * (factor_area_to_plants if factor_area_to_plants else 0.0)
    births_base = np.where(ms, births_base, 0.0)

    # cap acumulado en nacimientos
    births_cap = cap_cumulative(births_base, MAX_PLANTS_CAP, ms)

    # efecto de canopia (1 - Ciec) sobre nacimientos que entran a S1
    births_eff = births_cap * np.clip(one_minus_ciec, 0.0, 1.0)

    # ---------------- controles por estado (pesos diarios) ----------------
    # Construimos arrays de 0..1 que representan la "intensidad" por día; luego ef% se aplica fuera.

    w_pre_glifo   = weights_one_day(pre_glifo_date, dates_local)    if pre_glifo else np.zeros(n)
    w_preNR       = weights_residual(pre_selNR_date, NR_DAYS_DEFAULT, dates_local) if pre_selNR else np.zeros(n)

    w_preR        = weights_residual(preR_date, preR_days, dates_local) if preR else np.zeros(n)
    # fuerza ventana presiembraR: <= siembra-14
    if preR:
        mask_ok = (dates_local <= (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)))
        w_preR = w_preR * mask_ok.astype(float)

    w_preemR      = weights_residual(preemR_date, preemR_days, dates_local) if preemR else np.zeros(n)
    # ventana [siembra..siembra+10]
    if preemR:
        mask_ok = (dates_local >= sow_date) & (dates_local <= (sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)))
        w_preemR = w_preemR * mask_ok.astype(float)

    w_post_gram   = weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS, dates_local) if post_gram else np.zeros(n)
    if post_gram:
        mask_ok = (dates_local >= sow_date) & (dates_local <= (sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)))
        w_post_gram = w_post_gram * mask_ok.astype(float)

    w_postR       = weights_residual(post_selR_date, post_res_dias, dates_local) if post_selR else np.zeros(n)
    if post_selR:
        mask_ok = (dates_local >= (sow_date + timedelta(days=20)))
        w_postR = w_postR * mask_ok.astype(float)

    # ---------------- eficiencias combinadas por estado ----------------
    # Reducción multiplicativa: pop_new = pop_old * Π(1 - eff_i * w_i)
    # NR (pre selectivo no residual) → todos los estados
    red_S1 = np.ones(n)
    red_S2 = np.ones(n)
    red_S3 = np.ones(n)
    red_S4 = np.ones(n)

    def apply_reduction(red, eff_pct, w):
        if eff_pct <= 0: return red
        return red * np.clip(1.0 - (eff_pct/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)

    # glifo (pre, 1 día) → todos
    if pre_glifo:  # día puntual
        red_S1 = apply_reduction(red_S1, ef_pre_glifo, w_pre_glifo)
        red_S2 = apply_reduction(red_S2, ef_pre_glifo, w_pre_glifo)
        red_S3 = apply_reduction(red_S3, ef_pre_glifo, w_pre_glifo)
        red_S4 = apply_reduction(red_S4, ef_pre_glifo, w_pre_glifo)

    # pre NR (10d) → todos
    if pre_selNR:
        red_S1 = apply_reduction(red_S1, ef_pre_selNR, w_preNR)
        red_S2 = apply_reduction(red_S2, ef_pre_selNR, w_preNR)
        red_S3 = apply_reduction(red_S3, ef_pre_selNR, w_preNR)
        red_S4 = apply_reduction(red_S4, ef_pre_selNR, w_preNR)

    # presiembra residual → S1–S2
    if preR:
        red_S1 = apply_reduction(red_S1, ef_preR, w_preR)
        red_S2 = apply_reduction(red_S2, ef_preR, w_preR)

    # preemergente residual → S1–S2
    if preemR:
        red_S1 = apply_reduction(red_S1, ef_preemR, w_preemR)
        red_S2 = apply_reduction(red_S2, ef_preemR, w_preemR)

    # graminicida post → S1–S3
    if post_gram:
        red_S1 = apply_reduction(red_S1, ef_post_gram, w_post_gram)
        red_S2 = apply_reduction(red_S2, ef_post_gram, w_post_gram)
        red_S3 = apply_reduction(red_S3, ef_post_gram, w_post_gram)

    # post residual → S1–S4
    if post_selR:
        red_S1 = apply_reduction(red_S1, ef_post_selR, w_postR)
        red_S2 = apply_reduction(red_S2, ef_post_selR, w_postR)
        red_S3 = apply_reduction(red_S3, ef_post_selR, w_postR)
        red_S4 = apply_reduction(red_S4, ef_post_selR, w_postR)

    # ---------------- integración diaria (baseline y control) ----------------
    # baseline: sin controles (para x2)
    S1_b = np.zeros(n); S2_b = np.zeros(n); S3_b = np.zeros(n); S4_b = np.zeros(n)
    # control: con reducciones (para x3)
    S1_c = np.zeros(n); S2_c = np.zeros(n); S3_c = np.zeros(n); S4_c = np.zeros(n)

    D1 = max(1, int(D1)); D2 = max(1, int(D2)); D3 = max(1, int(D3))

    for t in range(n):
        # --- Baseline ---
        # entrada a S1 = nacimientos efectivos del día (sin control)
        in_S1_b = births_eff[t]
        # mortalidad/transferencias: transferencia promedio diaria = stock/D
        tr12_b = S1_b[t-1]/D1 if t>0 else 0.0
        tr23_b = S2_b[t-1]/D2 if t>0 else 0.0
        tr34_b = S3_b[t-1]/D3 if t>0 else 0.0

        S1_b[t] = (S1_b[t-1] if t>0 else 0.0) + in_S1_b - tr12_b
        S2_b[t] = (S2_b[t-1] if t>0 else 0.0) + tr12_b - tr23_b
        S3_b[t] = (S3_b[t-1] if t>0 else 0.0) + tr23_b - tr34_b
        S4_b[t] = (S4_b[t-1] if t>0 else 0.0) + tr34_b  # S4 acumula

        # --- Control ---
        in_S1_c = births_eff[t]  # los nacimientos ven Ciec (ya aplicado arriba)
        # aplicar control a stocks previos ANTES de transferir
        S1_prev = (S1_c[t-1] if t>0 else 0.0) * red_S1[t]
        S2_prev = (S2_c[t-1] if t>0 else 0.0) * red_S2[t]
        S3_prev = (S3_c[t-1] if t>0 else 0.0) * red_S3[t]
        S4_prev = (S4_c[t-1] if t>0 else 0.0) * red_S4[t]

        tr12_c = S1_prev/D1
        tr23_c = S2_prev/D2
        tr34_c = S3_prev/D3

        S1_c[t] = S1_prev + in_S1_c - tr12_c
        S2_c[t] = S2_prev + tr12_c - tr23_c
        S3_c[t] = S3_prev + tr23_c - tr34_c
        S4_c[t] = S4_prev + tr34_c

        # No negativos numéricos
        if S1_b[t] < 0: S1_b[t]=0.0
        if S2_b[t] < 0: S2_b[t]=0.0
        if S3_b[t] < 0: S3_b[t]=0.0
        if S4_b[t] < 0: S4_b[t]=0.0
        if S1_c[t] < 0: S1_c[t]=0.0
        if S2_c[t] < 0: S2_c[t]=0.0
        if S3_c[t] < 0: S3_c[t]=0.0
        if S4_c[t] < 0: S4_c[t]=0.0

    # contribución efectiva diaria (ponderación W_S)
    eff_b = S1_b*W_S["S1"] + S2_b*W_S["S2"] + S3_b*W_S["S3"] + S4_b*W_S["S4"]
    eff_c = S1_c*W_S["S1"] + S2_c*W_S["S2"] + S3_c*W_S["S3"] + S4_c*W_S["S4"]

    # x2/x3 (sólo desde siembra)
    X2 = float(np.nansum(eff_b[ms]))
    X3 = float(np.nansum(eff_c[ms]))

    # A2 por AUC usando equivalencia de área
    if factor_area_to_plants and factor_area_to_plants>0 and auc_cruda>0:
        sup_equiv   = np.divide(eff_b, factor_area_to_plants, out=np.zeros_like(eff_b), where=True)
        ctrl_equiv  = np.divide(eff_c, factor_area_to_plants, out=np.zeros_like(eff_c), where=True)
        AUC_sup     = auc_time(ts_local, sup_equiv,  mask=ms)
        AUC_ctrl    = auc_time(ts_local, ctrl_equiv, mask=ms)
        A2_sup      = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(AUC_sup / auc_cruda))
        A2_ctrl     = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(AUC_ctrl / auc_cruda))
    else:
        A2_sup = A2_ctrl = float("nan")

    # Semanal (W-MON)
    df_daily = pd.DataFrame({
        "fecha": ts_local.values,
        "eff_b": eff_b,
        "eff_c": eff_c
    })
    df_week = (
        df_daily.set_index("fecha")
        .resample("W-MON").sum()
        .reset_index()
    )

    return {
        "ts": ts_local,
        "dates": dates_local,
        "S_b": (S1_b,S2_b,S3_b,S4_b),
        "S_c": (S1_c,S2_c,S3_c,S4_c),
        "eff_b": eff_b,
        "eff_c": eff_c,
        "X2": X2, "X3": X3,
        "A2_sup": A2_sup, "A2_ctrl": A2_ctrl,
        "week": df_week
    }

# ========= ejecutar simulación =========
sim = None
if factor_area_to_plants is None:
    st.info("Necesitás AUC(EMERREL cruda) > 0 (desde siembra) para simular."); st.stop()
else:
    sim = simulate_chained(
        emerrel_series=df_plot.set_index("fecha")["EMERREL"],
        sow_date=sow_date,
        one_minus_ciec=one_minus_Ciec,
        factor_area_to_plants=factor_area_to_plants,
        MAX_PLANTS_CAP=MAX_PLANTS_CAP,
        D1=6, D2=21, D3=32
    )

# ========= timeline =========
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
    out = []
    if pre_glifo and pre_glifo_date:
        ini = pd.to_datetime(pre_glifo_date); fin = ini + pd.Timedelta(days=1)
        out.append((ini, fin, "glifo"))
    if pre_selNR and pre_selNR_date:
        ini = pd.to_datetime(pre_selNR_date); fin = ini + pd.Timedelta(days=NR_DAYS_DEFAULT)
        out.append((ini, fin, "preNR"))
    if preR and preR_date:
        ini = pd.to_datetime(preR_date); fin = ini + pd.Timedelta(days=int(preR_days))
        out.append((ini, fin, "preR"))
    if preemR and preemR_date:
        ini = pd.to_datetime(preemR_date); fin = ini + pd.Timedelta(days=int(preemR_days))
        out.append((ini, fin, "preemR"))
    if post_gram and post_gram_date:
        ini = pd.to_datetime(post_gram_date); fin = ini + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)
        out.append((ini, fin, "post_gram"))
    if post_selR and post_selR_date:
        ini = pd.to_datetime(post_selR_date); fin = ini + pd.Timedelta(days=int(post_res_dias))
        out.append((ini, fin, "postR"))
    return out

def add_timeline(fig: go.Figure, intervals, lanes=LANES, band_height=0.16, gap=0.005):
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
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=ini, x1=fin, y0=y0, y1=y1,
                      line=dict(width=0),
                      fillcolor=COLOR_TREAT.get(kind, "rgba(120,120,120,0.7)"))
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0, x1=1, y0=y0_band, y1=0,
                  line=dict(color="rgba(0,0,0,0.15)", width=1),
                  fillcolor="rgba(0,0,0,0)")

# ========= gráficos =========
with st.sidebar:
    st.header("Opciones visuales")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_timeline = st.checkbox("Mostrar timeline de tratamientos (abajo)", value=True)

# Serie semanal (W-MON)
df_week = sim["week"]
sem_x = df_week["fecha"]
plm2sem_sin_ctrl = df_week["eff_b"].to_numpy()
plm2sem_con_ctrl = df_week["eff_c"].to_numpy()

st.subheader(f"📊 EMERREL + aportes (encadenado) — Serie semanal (W-MON)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=50),
                     title=f"EMERREL (izq) y Plantas·m²·semana (der) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")

if show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (encadenado)",
                                   position=1.0, showgrid=False)
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_sin_ctrl, name="Aporte semanal (sin control)", yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_con_ctrl, name="Aporte semanal (con control)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
if show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3"))

if show_timeline:
    manual_intervals = build_manual_intervals()
    add_timeline(fig, manual_intervals, lanes=LANES, band_height=0.16)

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={sim['A2_sup'] if np.isfinite(sim['A2_sup']) else float('nan'):.1f} · A2_ctrl={sim['A2_ctrl'] if np.isfinite(sim['A2_ctrl']) else float('nan'):.1f}")

# Panel x y A2
X2 = sim["X2"]; X3 = sim["X3"]
st.subheader(f"Densidad efectiva (x) y A2 (por AUC, cap={int(MAX_PLANTS_CAP)})")
st.markdown(
    f"""
**x₂ — Sin control:** **{X2:,.1f}** pl·m²  
**x₃ — Con control:** **{X3:,.1f}** pl·m²  
**A2 (sup):** **{sim['A2_sup'] if np.isfinite(sim['A2_sup']) else float('nan'):.1f}** pl·m²  
**A2 (ctrl):** **{sim['A2_ctrl'] if np.isfinite(sim['A2_ctrl']) else float('nan'):.1f}** pl·m²  
**Pérdida(x₂):** {loss_fun(X2):.2f}% · **Pérdida(x₃):** {loss_fun(X3):.2f}%
"""
)

# ========= Gráfico de pérdida (%) vs x =========
st.subheader("📉 Pérdida de rendimiento (%) vs x")
x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
y_curve = loss_fun(x_curve)
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x"))
fig_loss.add_trace(go.Scatter(x=[X2], y=[loss_fun(X2)], mode="markers+text",
                              name="x₂ (sin ctrl)", text=[f"x₂={X2:.1f}"], textposition="top center"))
fig_loss.add_trace(go.Scatter(x=[X3], y=[loss_fun(X3)], mode="markers+text",
                              name="x₃ (con ctrl)", text=[f"x₃={X3:.1f}"], textposition="top right"))
fig_loss.update_layout(xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
st.plotly_chart(fig_loss, use_container_width=True)

# ========= (opcional) mostrar series S1–S4 encadenadas (semanal) =========
with st.expander("Estados S1–S4 (encadenados) — Serie semanal", expanded=False):
    S1c, S2c, S3c, S4c = sim["S_c"]
    df_states_week = (
        pd.DataFrame({"fecha": sim["ts"].values, "S1": S1c, "S2": S2c, "S3": S3c, "S4": S4c})
        .set_index("fecha").resample("W-MON").sum().reset_index()
    )
    fig_states = go.Figure()
    for col in ["S1","S2","S3","S4"]:
        fig_states.add_trace(go.Scatter(x=df_states_week["fecha"], y=df_states_week[col],
                                        mode="lines", name=col, stackgroup="one"))
    fig_states.update_layout(title="Aportes semanales por estado (con control)",
                             xaxis_title="Tiempo", yaxis_title="pl·m²·sem⁻¹")
    st.plotly_chart(fig_states, use_container_width=True)

# ========= parámetros calibrados (solo lectura) =========
with st.expander("Parámetros calibrados (solo lectura)", expanded=False):
    st.markdown(
        f"""
- **w₁..w₄** = {W_S['S1']}, {W_S['S2']}, {W_S['S3']}, {W_S['S4']}  
- **α** = {ALPHA} · **Lmax** = {LMAX}  
- **loss(x) = α·x / (1 + α·x/Lmax)**
"""
    )

