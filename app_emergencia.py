# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Cohortes SECUENCIALES · Optimización
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import itertools, random     # ✅ necesarios para combinaciones y búsqueda aleatoria
from datetime import timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ------------------ FUNCIÓN DE PÉRDIDA ------------------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ------------------ ESTADO UI ------------------
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

APP_TITLE = "🌾 PREDWEEM — (1−Ciec) + AUC + Cohortes SECUENCIALES · Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ------------------ CONSTANTES ------------------
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 15
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10
EPS_REMAIN = 1e-9
EPS_EXCLUDE = 0.99  # 99%: umbral de exclusión jerárquica
# ======================= I/O CSV =======================
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

# ======================= ESCENARIO =======================
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva (pl·m²)", [250, 125, 62], index=0))
st.caption(f"AUC(EMERREL cruda) ≙ A2 **= {int(MAX_PLANTS_CAP)} pl·m²**. Cohortes S1..S4 **SECUENCIALES**.")

# ======================= CARGA CSV =======================
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
except (URLError, HTTPError) as e:
    st.error(f"No se pudo acceder a la URL: {e}"); st.stop()
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}"); st.stop()

# ======================= PARSEO COLUMNAS =======================
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

# ======================= SIEMBRA & CANOPIA =======================
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1); sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia (para Ciec)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}**")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag   = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.0, 0.1)
    k_beer  = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (competitividad material)", 3.0, 6.0, 3.0, 0.1)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de julio."); st.stop()

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

# ======================= ESTADOS S1→S4 (secuenciales) =======================
ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)

births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where(mask_since_sow.to_numpy(), births, 0.0)

T12 = st.sidebar.number_input("Duración S1→S2 (días)", 1, 60, 10, 1)
T23 = st.sidebar.number_input("Duración S2→S3 (días)", 1, 60, 15, 1)
T34 = st.sidebar.number_input("Duración S3→S4 (días)", 1, 60, 20, 1)
globals()["T12"] = int(T12); globals()["T23"] = int(T23); globals()["T34"] = int(T34)

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
S1_arr, S2_arr, S3_arr, S4_arr = S1, S2, S3, S4

# ======================= ESCALADO A PLANTAS =======================
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ======================= MANEJO (UI MANUAL) =======================
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
                       help="Solo permitido hasta siembra−14 días. Actúa S1–S2.")
    preR_days = st.slider("Residualidad presiembra (días)", 15, 120, 14, 1, disabled=not preR)
    preR_max = (sow_date - timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW))
    preR_date = st.date_input("Fecha selectivo + residual (presiembra)",
                              value=min(min_date, preR_max),
                              min_value=min_date,
                              max_value=min(preR_max, max_date),
                              disabled=not preR)

with st.sidebar:
    st.header("Manejo preemergente (manual)")
    preemR = st.checkbox("Selectivo + residual (preemergente)", value=False,
                         help="Ventana [siembra, siembra+10]. Actúa S1–S2.")
    preemR_days = st.slider("Residualidad preemergente (días)", 15, 120, 45, 1, disabled=not preemR)
    preem_min = sow_date
    preem_max = min(max_date, sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))
    preemR_date = st.date_input("Fecha selectivo + residual (preemergente)",
                                value=preem_min, min_value=preem_min, max_value=preem_max, disabled=not preemR)

with st.sidebar:
    st.header("Manejo post-emergencia (manual)")
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False,
                            help="Ahora controla S1–S4; ventana día de app +10d.")
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False,
                            help="Regla: ≥ siembra + 20 días. **Actúa S1–S2** (actualizado).")
    post_min_postR = max(min_date, sow_date + timedelta(days=20))
    post_selR_date = st.date_input("Fecha selectivo + residual (post)", value=post_min_postR, min_value=post_min_postR, max_value=max_date, disabled=not post_selR)
    post_res_dias = st.slider("Residualidad post (días)", 30, 120, 45, 1, disabled=not post_selR)

# Validaciones
warnings = []
def check_pre(date_val, name):
    if date_val and date_val > sow_date: warnings.append(f"{name}: debería ser ≤ siembra ({sow_date}).")
def check_post(date_val, name):
    if date_val and date_val < sow_date: warnings.append(f"{name}: debería ser ≥ siembra ({sow_date}).")
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

# Cronograma legible
if pre_glifo: add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched("Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR {NR_DAYS_DEFAULT}d")
if preR:      add_sched("Pre-SIEMBRA · selectivo + residual", preR_date, preR_days, f"Protege {preR_days}d (S1–S2)")
if preemR:    add_sched("PREEMERGENTE · selectivo + residual", preemR_date, preemR_days, f"Protege {preemR_days}d (S1–S2)")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post · graminicida (NR, +10d)", "Inicio": str(ini), "Fin": str(fin), "Nota": "S1–S4 (actualizado)"})
if post_selR: add_sched("Post · selectivo + residual", post_selR_date, post_res_dias, f"Protege {post_res_dias}d (S1–S2)")

sched = pd.DataFrame(sched_rows)

# ======================= EFICIENCIAS =======================
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 90, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 90, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 90, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 90, 1) if post_selR else 0

# ======================= Decaimiento opcional =======================
with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días)", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None
if decaimiento_tipo != "Exponencial": lam_exp = None

# ===== Ventanas de efecto =====
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

# ======================= APORTES/CONTROL =======================
if factor_area_to_plants is not None:
    ms = mask_since_sow.to_numpy()
    S1_pl = np.where(ms, S1_arr * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants, 0.0)
    S2_pl = np.where(ms, S2_arr * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants, 0.0)
    S3_pl = np.where(ms, S3_arr * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants, 0.0)
    S4_pl = np.where(ms, S4_arr * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants, 0.0)

    ctrl_S1 = np.ones_like(fechas_d, float); ctrl_S2 = np.ones_like(fechas_d, float)
    ctrl_S3 = np.ones_like(fechas_d, float); ctrl_S4 = np.ones_like(fechas_d, float)

    def apply_efficiency_per_state(weights, eff_pct, states_sel):
        if eff_pct <= 0 or (not states_sel): return
        reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states_sel: np.multiply(ctrl_S1, reduc, out=ctrl_S1)
        if "S2" in states_sel: np.multiply(ctrl_S2, reduc, out=ctrl_S2)
        if "S3" in states_sel: np.multiply(ctrl_S3, reduc, out=ctrl_S3)
        if "S4" in states_sel: np.multiply(ctrl_S4, reduc, out=ctrl_S4)

    def _remaining_in_window(weights, states_sel, c1, c2, c3, c4):
        w = np.clip(weights, 0.0, 1.0)
        rem = 0.0
        for s in states_sel:
            if s == "S1": rem += np.sum(S1_pl * c1 * w)
            elif s == "S2": rem += np.sum(S2_pl * c2 * w)
            elif s == "S3": rem += np.sum(S3_pl * c3 * w)
            elif s == "S4": rem += np.sum(S4_pl * c4 * w)
        return float(rem)

    def _eff_from_to(prev_eff, this_eff):
        return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    eff_accum_pre = 0.0
    eff_accum_pre2 = 0.0
    eff_accum_all = 0.0

    # 1) Presiembra residual (S1–S2)
    if preR:
        w_preR = weights_residual(preR_date, preR_days)
        if _remaining_in_window(w_preR, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_preR > 0:
            apply_efficiency_per_state(w_preR, ef_preR, ["S1","S2"])
            eff_accum_pre = _eff_from_to(0.0, ef_preR/100.0)

    # 2) Preemergente residual (S1–S2)
    if preemR and (eff_accum_pre < EPS_EXCLUDE):
        w_preem = weights_residual(preemR_date, preemR_days)
        if _remaining_in_window(w_preem, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_preemR > 0:
            apply_efficiency_per_state(w_preem, ef_preemR, ["S1","S2"])
            eff_accum_pre2 = _eff_from_to(eff_accum_pre, ef_preemR/100.0)
        else:
            eff_accum_pre2 = eff_accum_pre
    else:
        eff_accum_pre2 = eff_accum_pre

    # 3) NR pre (opcional, no jerárquico)
    if pre_selNR:
        w = weights_residual(pre_selNR_date, NR_DAYS_DEFAULT)
        if _remaining_in_window(w, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_pre_selNR > 0:
            apply_efficiency_per_state(w, ef_pre_selNR, ["S1","S2","S3","S4"])
    if pre_glifo:
        w = weights_one_day(pre_glifo_date)
        if _remaining_in_window(w, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_pre_glifo > 0:
            apply_efficiency_per_state(w, ef_pre_glifo, ["S1","S2","S3","S4"])

    # 4) Post residual — **actualizado: actúa S1–S2**
    if post_selR and (eff_accum_pre2 < EPS_EXCLUDE):
        w_postR = weights_residual(post_selR_date, post_res_dias)
        if _remaining_in_window(w_postR, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_post_selR > 0:
            apply_efficiency_per_state(w_postR, ef_post_selR, ["S1","S2"])
            eff_accum_all = _eff_from_to(eff_accum_pre2, ef_post_selR/100.0)
        else:
            eff_accum_all = eff_accum_pre2
    else:
        eff_accum_all = eff_accum_pre2

    # 5) Graminicida post — **actualizado: actúa S1–S4** y va DESPUÉS de postR
    if post_gram and (eff_accum_all < EPS_EXCLUDE):
        w_gram = weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS)
        if _remaining_in_window(w_gram, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_post_gram > 0:
            apply_efficiency_per_state(w_gram, ef_post_gram, ["S1","S2","S3","S4"])

    S1_pl_ctrl = S1_pl * ctrl_S1
    S2_pl_ctrl = S2_pl * ctrl_S2
    S3_pl_ctrl = S3_pl * ctrl_S3
    S4_pl_ctrl = S4_pl * ctrl_S4

    plantas_supresion      = (S1_pl + S2_pl + S3_pl + S4_pl)
    plantas_supresion_ctrl = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=S1_pl_ctrl=S2_pl_ctrl=S3_pl_ctrl=S4_pl_ctrl=plantas_supresion=plantas_supresion_ctrl=np.full(len(ts), np.nan)

# ======================= CAP A2 & REESCALADO =======================
if factor_area_to_plants is not None:
    base_pl_daily = df_plot["EMERREL"].to_numpy(float) * factor_area_to_plants
    base_pl_daily = np.where(mask_since_sow.to_numpy(), base_pl_daily, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())

    plantas_supresion_cap      = np.minimum(plantas_supresion, base_pl_daily_cap)
    plantas_supresion_ctrl_cap = np.minimum(plantas_supresion_ctrl, plantas_supresion_cap)

    total_ctrl_daily = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
    eps = 1e-12
    scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, plantas_supresion_ctrl_cap / total_ctrl_daily), 0.0)
    S1_pl_ctrl_cap = S1_pl_ctrl * scale
    S2_pl_ctrl_cap = S2_pl_ctrl * scale
    S3_pl_ctrl_cap = S3_pl_ctrl * scale
    S4_pl_ctrl_cap = S4_pl_ctrl * scale
    plantas_supresion_ctrl_cap = S1_pl_ctrl_cap + S2_pl_ctrl_cap + S3_pl_ctrl_cap + S4_pl_ctrl_cap
else:
    base_pl_daily = base_pl_daily_cap = plantas_supresion_cap = plantas_supresion_ctrl_cap = np.full(len(ts), np.nan)

# ======================= AGREGACIÓN SEMANAL & A2/x =======================
df_daily_cap = pd.DataFrame({
    "fecha": ts,
    "pl_sin_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_cap, 0.0),
    "pl_con_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_ctrl_cap, 0.0),
})
df_week_cap = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()
sem_x = df_week_cap["fecha"]
plm2sem_sin_ctrl_cap = df_week_cap["pl_sin_ctrl_cap"].to_numpy()
plm2sem_con_ctrl_cap = df_week_cap["pl_con_ctrl_cap"].to_numpy()

if factor_area_to_plants is not None and auc_cruda > 0:
    sup_equiv  = np.divide(plantas_supresion_cap,     factor_area_to_plants, out=np.zeros_like(plantas_supresion_cap),     where=(factor_area_to_plants>0))
    supc_equiv = np.divide(plantas_supresion_ctrl_cap, factor_area_to_plants, out=np.zeros_like(plantas_supresion_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

if factor_area_to_plants is not None:
    X2 = float(np.nansum(plantas_supresion_cap[mask_since_sow]))
    X3 = float(np.nansum(plantas_supresion_ctrl_cap[mask_since_sow]))
else:
    X2 = X3 = float("nan")

# ======================= GRÁFICO PRINCIPAL =======================
st.subheader(f"📊 Gráfico 1: EMERREL + aportes (cap A2={int(MAX_PLANTS_CAP)}) — Serie semanal (W-MON)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=10),
                     title=f"EMERREL (izq) y Plantas·m²·semana (der, 0–100) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")

with st.sidebar:
    st.header("Opciones visuales")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·sem⁻¹ (eje derecho)", value=True)
    show_ciec_curve = st.checkbox("Mostrar curva Ciec (0–1)", value=True)

if factor_area_to_plants is not None and show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0, 100], tick0=0, dtick=20, showgrid=False)
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_sin_ctrl_cap, name="Aporte semanal (sin control, cap)", yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_con_ctrl_cap, name="Aporte semanal (con control, cap)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
if show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3"))

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f} · A2_ctrl={A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}")

st.subheader(f"Densidad efectiva (x) y A2 (por AUC, cap={int(MAX_PLANTS_CAP)})")
st.markdown(
    f"""
**x₂ — Sin control (cap):** **{X2:,.1f}** pl·m²  
**x₃ — Con control (cap):** **{X3:,.1f}** pl·m²  
**A2 (sup, cap):** **{A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f}** pl·m²  
**A2 (ctrl, cap):** **{A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}** pl·m²
"""
)

# ===============================================================
# 🌾 CONTROLES DE OPTIMIZACIÓN — SIDEBAR STREAMLIT
# ===============================================================

st.sidebar.header("⚙️ Parámetros de Optimización")

# ----- Tipo de optimizador -----
optimizer = st.sidebar.selectbox(
    "Método de optimización",
    ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"],
    index=0,
    key="optimizer"
)

max_evals = st.sidebar.number_input(
    "Máx. combinaciones / iteraciones",
    min_value=10, max_value=5000, value=500, step=10
)

# ----- Parámetros de Recocido Simulado -----
with st.sidebar.expander("🔥 Parámetros de recocido simulado"):
    sa_T0 = st.number_input("Temperatura inicial (T0)", min_value=0.001, value=1.0, step=0.1)
    sa_cooling = st.number_input("Factor de enfriamiento (0–1)", min_value=0.80, max_value=0.999, value=0.95, step=0.01)
    sa_iters = st.number_input("Iteraciones totales", min_value=50, max_value=5000, value=500, step=10)

# ===============================================================
# 🎯 PARÁMETROS DE CONTROL Y RESIDUALIDAD
# ===============================================================

st.sidebar.subheader("🧪 Eficiencias y residualidades")

# ----- Presiembra -----
use_preR_opt = st.sidebar.checkbox("Usar presiembra residual", value=True)
ef_preR_opt = st.sidebar.slider("Eficiencia presiembra (%)", 0, 100, 90, 5)
res_days_preR = st.sidebar.multiselect(
    "Duración presiembra (días de residualidad)",
    [10, 15, 20, 25, 30],
    default=[20]
)

# ----- Preemergente -----
use_preemR_opt = st.sidebar.checkbox("Usar preemergente residual", value=True)
ef_preemR_opt = st.sidebar.slider("Eficiencia preemergente (%)", 0, 100, 85, 5)
res_days_preemR = st.sidebar.multiselect(
    "Duración preemergente (días de residualidad)",
    [10, 15, 20, 25, 30],
    default=[15]
)

# ----- Post-residual -----
use_post_selR_opt = st.sidebar.checkbox("Usar postemergente residual", value=True)
ef_post_selR_opt = st.sidebar.slider("Eficiencia post-residual (%)", 0, 100, 80, 5)
res_days_postR = st.sidebar.multiselect(
    "Duración post-residual (días de residualidad)",
    [10, 15, 20, 25, 30, 40],
    default=[20]
)

# ----- Graminicida post -----
use_post_gram_opt = st.sidebar.checkbox("Usar graminicida postemergente", value=True)
ef_post_gram_opt = st.sidebar.slider("Eficiencia graminicida (%)", 0, 100, 95, 5)

# ===============================================================
# 🚀 CONTROLES DE EJECUCIÓN
# ===============================================================
st.sidebar.subheader("▶️ Ejecución del optimizador")

col1, col2 = st.sidebar.columns(2)

start_clicked = col1.button("▶️ Iniciar", use_container_width=True)
stop_clicked  = col2.button("🛑 Detener", use_container_width=True)

if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_stop" not in st.session_state:
    st.session_state.opt_stop = False

if start_clicked and not st.session_state.opt_running:
    st.session_state.opt_running = True
    st.session_state.opt_stop = False
elif stop_clicked and st.session_state.opt_running:
    st.session_state.opt_stop = True










# ===============================================================
# 🌾 OPTIMIZADOR COMPLETO — SIN RESTRICCIÓN DE SOLAPAMIENTO
# ===============================================================

def evaluate(sd: dt.date, schedule: list):
    """Evalúa un cronograma de controles, permitiendo solapamientos."""
    sow = pd.to_datetime(sd)
    sow_plus_20 = sow + pd.Timedelta(days=20)

    valid_actions = []
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR" and d < sow_plus_20:
            continue
        if a["kind"] == "preR" and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)):
            continue
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))):
            continue
        valid_actions.append(a)

    if not valid_actions:
        return None

    env = recompute_for_sow(sd, int(T12), int(T23), int(T34))
    if env is None:
        return None

    mask_since = env["mask_since"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]
    sup_cap = env["sup_cap"]
    ts_local = env["ts"]
    fechas_d_local = env["fechas_d"]

    # Factores de control por estado
    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def _apply_ctrl(w, eff, states):
        reduc = np.clip(1.0 - (eff / 100.0) * w, 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    order = {"preR": 0, "preemR": 1, "postR": 2, "post_gram": 3}
    valid_actions.sort(key=lambda x: (order.get(x["kind"], 9), x["date"]))

    for a in valid_actions:
        d0 = a["date"]
        d1 = a["date"] + dt.timedelta(days=int(a["days"]))
        w = ((fechas_d_local >= d0) & (fechas_d_local < d1)).astype(float)
        if np.any(w):
            _apply_ctrl(w, a["eff"], a["states"])

    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    X2loc = float(np.nansum(sup_cap[mask_since]))
    X3loc = float(np.nansum(plantas_ctrl_cap[mask_since]))
    loss3 = _loss(X3loc)

    auc_sup  = auc_time(ts_local, sup_cap,          mask=mask_since)
    auc_ctrl = auc_time(ts_local, plantas_ctrl_cap, mask=mask_since)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup  / env["auc_cruda"]))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_ctrl / env["auc_cruda"]))

    return {"sow": sd, "loss_pct": loss3, "x2": X2loc, "x3": X3loc,
            "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": valid_actions}


def build_all_scenarios():
    """Genera todas las combinaciones de escenarios posibles."""
    scenarios = []
    for sd in sow_candidates:
        grp = []
        if use_preR_opt:
            grp.append([act_presiembraR(d, R, ef_preR_opt)
                        for d in pre_sow_dates(sd)
                        for R in res_days_preR])
        if use_preemR_opt:
            grp.append([act_preemR(d, R, ef_preemR_opt)
                        for d in preem_dates(sd)
                        for R in res_days_preemR])
        if use_post_selR_opt:
            grp.append([act_post_selR(d, R, ef_post_selR_opt)
                        for d in post_dates(sd)
                        for R in res_days_postR])
        if use_post_gram_opt:
            grp.append([act_post_gram(d, ef_post_gram_opt)
                        for d in post_dates(sd)])
        combos = [[]]
        for r in range(1, len(grp)+1):
            for subset in itertools.combinations(range(len(grp)), r):
                for p in itertools.product(*[grp[i] for i in subset]):
                    combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    st.caption(f"Escenarios generados: {len(scenarios):,}")
    return scenarios


def sample_random_scenario():
    """Genera un escenario aleatorio (válido o vacío)."""
    for _ in range(200):
        sd = random.choice(sow_candidates)
        schedule = []
        if use_preR_opt and random.random() < 0.7:
            cand = pre_sow_dates(sd)
            if cand:
                schedule.append(act_presiembraR(random.choice(cand),
                                                random.choice(res_days_preR),
                                                ef_preR_opt))
        if use_preemR_opt and random.random() < 0.7:
            cand = preem_dates(sd)
            if cand:
                schedule.append(act_preemR(random.choice(cand),
                                           random.choice(res_days_preemR),
                                           ef_preemR_opt))
        if use_post_selR_opt and random.random() < 0.7:
            cand = post_dates(sd)
            if cand:
                schedule.append(act_post_selR(random.choice(cand),
                                              random.choice(res_days_postR),
                                              ef_post_selR_opt))
        if use_post_gram_opt and random.random() < 0.7:
            cand = post_dates(sd)
            if cand:
                schedule.append(act_post_gram(random.choice(cand),
                                              ef_post_gram_opt))
        if schedule:
            return (pd.to_datetime(sd).date(), schedule)
    return (pd.to_datetime(sd).date(), [])


# ===============================================================
# EJECUCIÓN DEL OPTIMIZADOR
# ===============================================================
status_ph = st.empty()
prog_ph = st.empty()
results = []

optimizer = st.session_state.get("optimizer", "Grid (combinatorio)")

if factor_area_to_plants is None or not np.isfinite(auc_cruda):
    st.info("Necesitás AUC(EMERREL cruda) > 0 para optimizar.")
else:
    if st.session_state.opt_running:
        status_ph.info("Optimizando…")

        if optimizer == "Grid (combinatorio)":
            scenarios = build_all_scenarios()
            total = len(scenarios)
            st.caption(f"Se evaluarán {total:,} configuraciones")
            if total > max_evals:
                random.seed(123)
                scenarios = random.sample(scenarios, k=int(max_evals))
                st.caption(f"Se muestrean {len(scenarios):,} configs (límite)")
            prog = prog_ph.progress(0.0)
            for i, (sd, sch) in enumerate(scenarios, 1):
                if st.session_state.opt_stop: break
                r = evaluate(sd, sch)
                if r: results.append(r)
                if i % max(1, len(scenarios)//100) == 0:
                    prog.progress(i/len(scenarios))
            prog_ph.empty()

        elif optimizer == "Búsqueda aleatoria":
            N = int(max_evals)
            prog = prog_ph.progress(0.0)
            for i in range(1, N+1):
                if st.session_state.opt_stop: break
                sd, sch = sample_random_scenario()
                r = evaluate(sd, sch)
                if r: results.append(r)
                if i % max(1, N//100) == 0:
                    prog.progress(i/N)
            prog_ph.empty()

        else:  # Recocido simulado
            cur = sample_random_scenario()
            cur_eval = evaluate(*cur)
            while cur_eval is None:
                cur = sample_random_scenario()
                cur_eval = evaluate(*cur)
            best_eval = cur_eval
            cur_loss = cur_eval["loss_pct"]
            T = float(sa_T0)
            prog = prog_ph.progress(0.0)
            for it in range(1, int(sa_iters)+1):
                if st.session_state.opt_stop: break
                cand = sample_random_scenario()
                cand_eval = evaluate(*cand)
                if cand_eval:
                    d = cand_eval["loss_pct"] - cur_loss
                    if d <= 0 or random.random() < math.exp(-d / max(1e-9, T)):
                        cur, cur_eval, cur_loss = cand, cand_eval, cand_eval["loss_pct"]
                        results.append(cur_eval)
                        if cur_loss < best_eval["loss_pct"]:
                            best_eval = cur_eval
                T *= float(sa_cooling)
                if it % max(1, int(sa_iters)//100) == 0:
                    prog.progress(it/int(sa_iters))
            results.append(best_eval)
            prog_ph.empty()

        st.session_state.opt_running = False
        st.session_state.opt_stop = False
        status_ph.success("Optimización finalizada.")
    else:
        status_ph.info("Listo para optimizar. Presioná ▶️ Iniciar.")

# ===============================================================
# VISUALIZACIÓN DEL MEJOR ESCENARIO
# ===============================================================
if results:
    results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))
    best = results_sorted[0]

    st.subheader("🏆 Mejor escenario encontrado")
    st.markdown(
        f"**Siembra:** {best['sow']}  \n"
        f"**Pérdida estimada:** {best['loss_pct']:.2f}%  \n"
        f"**x₂:** {best['x2']:.1f} · **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"**A2_sup:** {best['A2_sup']:.1f} · **A2_ctrl:** {best['A2_ctrl']:.1f} pl·m²"
    )

    df_best = pd.DataFrame([
        {"Intervención": a["kind"], "Inicio": str(a["date"]),
         "Fin": str(pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))),
         "Duración (d)": a["days"], "Eficiencia (%)": a["eff"],
         "Estados": ",".join(a["states"])}
        for a in best["schedule"]
    ])
    st.dataframe(df_best, use_container_width=True)

    # Recalcular series del mejor escenario
    envb = recompute_for_sow(pd.to_datetime(best["sow"]).date(), int(T12), int(T23), int(T34))
    if envb:
        ts_b = envb["ts"]
        fechas_d_b = envb["fechas_d"]
        S1p, S2p, S3p, S4p = envb["S_pl"]
        sup_cap_b = envb["sup_cap"]

        c1 = np.ones_like(fechas_d_b); c2 = np.ones_like(fechas_d_b)
        c3 = np.ones_like(fechas_d_b); c4 = np.ones_like(fechas_d_b)

        def _apply_eval(w, eff, states):
            reduc = np.clip(1.0 - (eff / 100.0) * w, 0.0, 1.0)
            if "S1" in states: np.multiply(c1, reduc, out=c1)
            if "S2" in states: np.multiply(c2, reduc, out=c2)
            if "S3" in states: np.multiply(c3, reduc, out=c3)
            if "S4" in states: np.multiply(c4, reduc, out=c4)

        for a in sorted(best["schedule"], key=lambda x: (order.get(x["kind"], 9), x["date"])):
            ini = pd.to_datetime(a["date"]).date()
            fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
            w = ((fechas_d_b >= ini) & (fechas_d_b < fin)).astype(float)
            if np.any(w):
                _apply_eval(w, a["eff"], a["states"])

        total_ctrl = (S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4)
        plantas_ctrl_cap_b = np.minimum(total_ctrl, sup_cap_b)

        # --- Gráfico 1: EMERREL + (1−Ciec) + tratamientos ---
        figA = go.Figure()
        figA.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL cruda"))
        figA.add_trace(go.Scatter(x=ts_b, y=sup_cap_b, name="Sin control", yaxis="y2"))
        figA.add_trace(go.Scatter(x=ts_b, y=plantas_ctrl_cap_b, name="Con control", yaxis="y2", line=dict(dash="dot")))
        one_minus_best = compute_ciec_for(pd.to_datetime(best["sow"]).date())
        figA.add_trace(go.Scatter(x=ts_b, y=1 - one_minus_best, mode="lines", name="Ciec", yaxis="y3"))
        for a in best["schedule"]:
            x0 = pd.to_datetime(a["date"]); x1 = x0 + pd.Timedelta(days=int(a["days"]))
            figA.add_vrect(x0=x0, x1=x1, fillcolor="rgba(30,144,255,0.15)", opacity=0.15, line_width=0)
            figA.add_annotation(x=x0+(x1-x0)/2, y=0.9, xref="x", yref="paper", text=a["kind"],
                                showarrow=False, bgcolor="rgba(30,144,255,0.7)")
        figA.update_layout(title="🌾 EMERREL + (1−Ciec) + Tratamientos",
                           xaxis_title="Fecha", yaxis_title="EMERREL",
                           yaxis2=dict(overlaying="y", side="right", title="pl·m²"),
                           yaxis3=dict(overlaying="y", side="right", title="Ciec", range=[0,1]))
        st.plotly_chart(figA, use_container_width=True)

        # --- Gráfico 2: Pérdida ---
        X2_b, X3_b = best["x2"], best["x3"]
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
        figB = go.Figure()
        figB.add_trace(go.Scatter(x=x_curve, y=_loss(x_curve), mode="lines", name="Función pérdida"))
        figB.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", text=["x₂"], textposition="top center"))
        figB.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", text=["x₃"], textposition="top right"))
        figB.update_layout(title="Pérdida vs densidad efectiva", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
        st.plotly_chart(figB, use_container_width=True)

        # --- Gráfico 3: Estados semanales ---
        df_states = (pd.DataFrame({"fecha": ts_b, "S1": S1p*c1, "S2": S2p*c2, "S3": S3p*c3, "S4": S4p*c4})
                     .set_index("fecha").resample("W-MON").sum().reset_index())
        figC = go.Figure()
        for s in ["S1","S2","S3","S4"]:
            figC.add_trace(go.Scatter(x=df_states["fecha"], y=df_states[s], name=s, stackgroup="one"))
        figC.update_layout(title="Dinámica semanal S1–S4 (con control)",
                           xaxis_title="Fecha", yaxis_title="pl·m²·sem⁻¹")
        st.plotly_chart(figC, use_container_width=True)
else:
    st.info("Aún no hay resultados de optimización para mostrar.")























