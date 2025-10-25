# -*- coding: utf-8 -*-
# ===============================================================
# PREDWEEM v3.14 — (1−Ciec) * Sens(PCC) + AUC + Cohortes SECUENCIALES · Vista
# ===============================================================
# Parámetros conservados (confirmados):
#   - Pérdida: α = 0.503 ; Lmax = 125.91
#   - Pesos por estado (aporte relativo): S1=0.1, S2=0.3, S3=0.6, S4=1.0
# Reglas:
# - Presiembra selectivo residual (preR): SOLO ≤ siembra−14 (actúa S1–S2)
# - Preemergente selectivo residual (preemR): [siembra, siembra+10] (S1–S2)
# - Post-residual (postR): ≥ siembra+20 (S1–S4)
# - Graminicida post: ventana siembra..siembra+10 (S1–S3)
# Lógica nueva (v3.14):
# - Sensibilidad al PCC: (1−Ciec) se multiplica por sens_factor(t) dentro de [PCC_INI, PCC_FIN]
# - Descomposición de pérdida: aporta dentro PCC vs fuera PCC
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta

APP_TITLE = "PREDWEEM v3.14 · PCC-sensible + AUC + Cohortes SECUENCIALES"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# ------------------ Pérdida ------------------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ------------------ Estado UI global ------------------
if "opt_running" not in st.session_state: st.session_state.opt_running = False
if "opt_stop"    not in st.session_state: st.session_state.opt_stop = False

# ------------------ Constantes ------------------
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW  = 14
PREEM_R_MAX_AFTER_SOW_DAYS        = 10
EPS_REMAIN = 1e-9
EPS_EXCLUDE = 0.99

# ------------------ Utils CSV ------------------
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

# ------------------ Escenario A2 ------------------
with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva (pl·m²)", [250, 125, 62], index=0))
st.caption(f"AUC(EMERREL cruda) ≙ A2 **= {int(MAX_PLANTS_CAP)} pl·m²**. Cohortes S1..S4 **SECUENCIALES**.")

# ------------------ Carga CSV ------------------
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

# ------------------ Parseo columnas ------------------
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

# ------------------ Siembra & Canopia (Ciec) ------------------
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
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer  = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

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

# ------------------ PCC: sensibilidad ------------------
with st.sidebar:
    st.header("Periodo Crítico de Competencia (PCC)")
    # Defaults razonables (pueden ajustarse según cultivo)
    pcc_def_ini = (sow_date + timedelta(days=20))
    pcc_def_fin = (sow_date + timedelta(days=60))
    PCC_INI = st.date_input("PCC inicio", value=pcc_def_ini, min_value=df_plot["fecha"].min().date(), max_value=df_plot["fecha"].max().date())
    PCC_FIN = st.date_input("PCC fin",    value=pcc_def_fin, min_value=df_plot["fecha"].min().date(), max_value=df_plot["fecha"].max().date())
    SENS_FACTOR = st.slider("Factor de sensibilidad en PCC", 0.20, 2.00, 1.00, 0.05,
                            help="Multiplica (1−Ciec) SOLO dentro del PCC. <1 reduce presión; >1 la aumenta.")

if PCC_FIN < PCC_INI:
    st.error("El FIN del PCC no puede ser anterior al INICIO."); st.stop()

# Vector de sensibilidad por fecha
ts = pd.to_datetime(df_plot["fecha"])
fechas_d = ts.dt.date.values
sens_factor = np.ones(len(fechas_d), dtype=float)
mask_pcc = (fechas_d >= PCC_INI) & (fechas_d <= PCC_FIN)
sens_factor[mask_pcc] = float(SENS_FACTOR)

# (1−Ciec) modulado por PCC
one_minus_Ciec_sens = np.clip(one_minus_Ciec * sens_factor, 0.0, None)

# ------------------ Estados secuenciales (S1→S4) ------------------
mask_since_sow = (ts.dt.date >= sow_date)
births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where(mask_since_sow.to_numpy(), births, 0.0)

T12 = st.sidebar.number_input("Duración S1→S2 (días)", 1, 60, 10, 1)
T23 = st.sidebar.number_input("Duración S2→S3 (días)", 1, 60, 15, 1)
T34 = st.sidebar.number_input("Duración S3→S4 (días)", 1, 60, 20, 1)

globals()["T12"] = int(T12); globals()["T23"] = int(T23); globals()["T34"] = int(T34)

S1 = births.copy()
S2 = np.zeros_like(births)
S3 = np.zeros_like(births)
S4 = np.zeros_like(births)

for i in range(len(births)):
    if i - int(T12) >= 0:
        moved = births[i - int(T12)]
        S1[i - int(T12)] -= moved
        S2[i] += moved
    if i - (int(T12) + int(T23)) >= 0:
        moved = births[i - (int(T12) + int(T23))]
        S2[i - (int(T12) + int(T23))] -= moved
        S3[i] += moved
    if i - (int(T12) + int(T23) + int(T34)) >= 0:
        moved = births[i - (int(T12) + int(T23) + int(T34))]
        S3[i - (int(T12) + int(T23) + int(T34))] -= moved
        S4[i] += moved

S1 = np.clip(S1, 0.0, None)
S2 = np.clip(S2, 0.0, None)
S3 = np.clip(S3, 0.0, None)
S4 = np.clip(S4, 0.0, None)

total_states = S1 + S2 + S3 + S4
emeac = np.cumsum(births)
scale = np.divide(np.clip(emeac, 1e-9, None), np.clip(total_states, 1e-9, None))
scale = np.minimum(scale, 1.0)
S1 *= scale; S2 *= scale; S3 *= scale; S4 *= scale

# Pesos por estado (confirmados)
FC_S = {"S1": 0.1, "S2": 0.3, "S3": 0.6, "S4": 1.0}
S1_arr, S2_arr, S3_arr, S4_arr = S1, S2, S3, S4

# ------------------ Escalado a plantas ------------------
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ------------------ UI de manejo manual ------------------
sched_rows = []
def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

min_date = ts.min().date(); max_date = ts.max().date()

with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    pre_glifo = st.checkbox("Herbicida total (glifosato)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selNR)

    preR = st.checkbox("Selectivo + residual (presiembra)", value=False,
                       help="Solo permitido hasta siembra−14 días. Actúa S1–S2.")
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
                         help="Ventana [siembra, siembra+10]. Actúa S1–S2.")
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
                            help="Regla: ≥ siembra + 20 días. Actúa S1–S4.")
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

# cronograma legible
if pre_glifo:  add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR:  add_sched("Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR {NR_DAYS_DEFAULT}d")
if preR:       add_sched("Pre-SIEMBRA · selectivo + residual", preR_date, preR_days, f"Protege {preR_days}d (S1–S2)")
if preemR:     add_sched("PREEMERGENTE · selectivo + residual", preemR_date, preemR_days, f"Protege {preemR_days}d (S1–S2)")
if post_gram:
    ini = pd.to_datetime(post_gram_date).date()
    fin = (pd.to_datetime(post_gram_date) + pd.Timedelta(days=POST_GRAM_FORWARD_DAYS)).date()
    sched_rows.append({"Intervención": "Post · graminicida (NR, +10d)", "Inicio": str(ini), "Fin": str(fin), "Nota": "Ventana día de app + 10 días"})
if post_selR:  add_sched("Post · selectivo + residual", post_selR_date, post_res_dias, f"Protege {post_res_dias}d")
sched = pd.DataFrame(sched_rows)

# ------------------ Eficiencias ------------------
with st.sidebar:
    st.header("Eficiencia de control (%)")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1d)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_preR        = st.slider("Selectivo + residual (presiembra)", 0, 100, 70, 1) if preR else 0
    ef_preemR      = st.slider("Selectivo + residual (preemergente)", 0, 100, 70, 1) if preemR else 0
    ef_post_gram   = st.slider(f"Graminicida (post, +10d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post)", 0, 100, 70, 1) if post_selR else 0

# ------------------ Decaimiento en residuales ------------------
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

# ------------------ Aportes por estado (sens PCC) ------------------
if factor_area_to_plants is not None:
    ms = mask_since_sow.to_numpy()
    S1_pl = np.where(ms, S1_arr * one_minus_Ciec_sens * FC_S["S1"] * factor_area_to_plants, 0.0)
    S2_pl = np.where(ms, S2_arr * one_minus_Ciec_sens * FC_S["S2"] * factor_area_to_plants, 0.0)
    S3_pl = np.where(ms, S3_arr * one_minus_Ciec_sens * FC_S["S3"] * factor_area_to_plants, 0.0)
    S4_pl = np.where(ms, S4_arr * one_minus_Ciec_sens * FC_S["S4"] * factor_area_to_plants, 0.0)

    # controles (inicialmente 1.0)
    ctrl_S1 = np.ones_like(fechas_d, float); ctrl_S2 = np.ones_like(fechas_d, float)
    ctrl_S3 = np.ones_like(fechas_d, float); ctrl_S4 = np.ones_like(fechas_d, float)

    def apply_efficiency_per_state(weights, eff_pct, states_sel):
        if eff_pct <= 0 or (not states_sel): return
        reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states_sel: np.multiply(ctrl_S1, reduc, out=ctrl_S1)
        if "S2" in states_sel: np.multiply(ctrl_S2, reduc, out=ctrl_S2)
        if "S3" in states_sel: np.multiply(ctrl_S3, reduc, out=ctrl_S3)
        if "S4" in states_sel: np.multiply(ctrl_S4, reduc, out=ctrl_S4)

    # Gateo (remanente) + Jerarquía (preR→preemR→postR→gram)
    def _remaining_in_window(weights, states_sel, c1, c2, c3, c4):
        w = np.clip(weights, 0.0, 1.0); rem = 0.0
        for s in states_sel:
            if s == "S1": rem += np.sum(S1_pl * c1 * w)
            elif s == "S2": rem += np.sum(S2_pl * c2 * w)
            elif s == "S3": rem += np.sum(S3_pl * c3 * w)
            elif s == "S4": rem += np.sum(S4_pl * c4 * w)
        return float(rem)

    def _eff_from_to(prev_eff, this_eff):  # independencia
        return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    eff_accum_pre = 0.0
    eff_accum_pre2 = 0.0
    eff_accum_all = 0.0

    # 1) Presiembra R (S1–S2)
    if 'preR' in sched["Intervención"].astype(str).str.lower().str.cat(sep=' ').lower() if len(sched) else False:
        pass  # solo informativo; la aplicación real la hacemos según toggles de UI

    # Aplicamos según toggles de UI:
    if preR:
        w_preR = weights_residual(preR_date, preR_days)
        if _remaining_in_window(w_preR, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_preR > 0:
            apply_efficiency_per_state(w_preR, ef_preR, ["S1","S2"])
            eff_accum_pre = _eff_from_to(0.0, ef_preR/100.0)

    # 2) Preemergente (S1–S2)
    if preemR and (eff_accum_pre < EPS_EXCLUDE):
        w_preem = weights_residual(preemR_date, preemR_days)
        if _remaining_in_window(w_preem, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_preemR > 0:
            apply_efficiency_per_state(w_preem, ef_preemR, ["S1","S2"])
            eff_accum_pre2 = _eff_from_to(eff_accum_pre, ef_preemR/100.0)
        else:
            eff_accum_pre2 = eff_accum_pre
    else:
        eff_accum_pre2 = eff_accum_pre

    # 3) Pre no residual (NR) y Glifo (NR)
    if pre_selNR:
        w = weights_residual(pre_selNR_date, NR_DAYS_DEFAULT)
        if _remaining_in_window(w, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_pre_selNR > 0:
            apply_efficiency_per_state(w, ef_pre_selNR, ["S1","S2","S3","S4"])
    if pre_glifo:
        w = weights_one_day(pre_glifo_date)
        if _remaining_in_window(w, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_pre_glifo > 0:
            apply_efficiency_per_state(w, ef_pre_glifo, ["S1","S2","S3","S4"])

    # 4) Post residual (S1–S4)
    if post_selR and (eff_accum_pre2 < EPS_EXCLUDE):
        w_postR = weights_residual(post_selR_date, post_res_dias)
        if _remaining_in_window(w_postR, ["S1","S2","S3","S4"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_post_selR > 0:
            apply_efficiency_per_state(w_postR, ef_post_selR, ["S1","S2","S3","S4"])
            eff_accum_all = _eff_from_to(eff_accum_pre2, ef_post_selR/100.0)
        else:
            eff_accum_all = eff_accum_pre2
    else:
        eff_accum_all = eff_accum_pre2

    # 5) Graminicida (S1–S3)
    if post_gram and (eff_accum_all < EPS_EXCLUDE):
        w_gram = weights_residual(post_gram_date, POST_GRAM_FORWARD_DAYS)
        if _remaining_in_window(w_gram, ["S1","S2","S3"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4) > EPS_REMAIN and ef_post_gram > 0:
            apply_efficiency_per_state(w_gram, ef_post_gram, ["S1","S2","S3"])

    # Series con control aplicado
    S1_pl_ctrl = S1_pl * ctrl_S1
    S2_pl_ctrl = S2_pl * ctrl_S2
    S3_pl_ctrl = S3_pl * ctrl_S3
    S4_pl_ctrl = S4_pl * ctrl_S4

    plantas_supresion      = (S1_pl + S2_pl + S3_pl + S4_pl)
    plantas_supresion_ctrl = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=S1_pl_ctrl=S2_pl_ctrl=S3_pl_ctrl=S4_pl_ctrl=plantas_supresion=plantas_supresion_ctrl=np.full(len(ts), np.nan)

# ------------------ CAP por A2 ------------------
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

# ------------------ Agregación semanal ------------------
df_daily_cap = pd.DataFrame({
    "fecha": ts,
    "pl_sin_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_cap, 0.0),
    "pl_con_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_ctrl_cap, 0.0),
})
df_week_cap = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()
sem_x = df_week_cap["fecha"]
plm2sem_sin_ctrl_cap = df_week_cap["pl_sin_ctrl_cap"].to_numpy()
plm2sem_con_ctrl_cap = df_week_cap["pl_con_ctrl_cap"].to_numpy()

# ------------------ A2 por AUC ------------------
if factor_area_to_plants is not None and auc_cruda > 0:
    sup_equiv  = np.divide(plantas_supresion_cap,     factor_area_to_plants, out=np.zeros_like(plantas_supresion_cap),     where=(factor_area_to_plants>0))
    supc_equiv = np.divide(plantas_supresion_ctrl_cap, factor_area_to_plants, out=np.zeros_like(plantas_supresion_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

# ------------------ x y pérdida ------------------
if factor_area_to_plants is not None:
    X2 = float(np.nansum(plantas_supresion_cap[mask_since_sow]))
    X3 = float(np.nansum(plantas_supresion_ctrl_cap[mask_since_sow]))
else:
    X2 = X3 = float("nan")

# ------------------ GRÁFICO principal ------------------
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
    show_pcc_band = st.checkbox("Sombrear PCC en el gráfico", value=True)

if show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0, 100], tick0=0, dtick=20, showgrid=False)
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_sin_ctrl_cap, name="Aporte semanal (sin control, cap)", yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sem_x, y=plm2sem_con_ctrl_cap, name="Aporte semanal (con control, cap)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
if show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3"))
if show_pcc_band:
    fig.add_vrect(x0=pd.to_datetime(PCC_INI), x1=pd.to_datetime(PCC_FIN), line_width=0,
                  fillcolor="rgba(255, 140, 0, 0.18)", opacity=0.18)

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

# ------------------ Descomposición por PCC ------------------
st.markdown("### 🟧 Descomposición de pérdida dentro vs fuera del PCC")
if factor_area_to_plants is None:
    st.info("Cargá datos válidos para ver la descomposición PCC.")
else:
    inside = mask_pcc & mask_since_sow.to_numpy()
    outside = (~mask_pcc) & mask_since_sow.to_numpy()

    X3_in  = float(np.nansum(plantas_supresion_ctrl_cap[inside]))
    X3_out = float(np.nansum(plantas_supresion_ctrl_cap[outside]))
    X2_in  = float(np.nansum(plantas_supresion_cap[inside]))
    X2_out = float(np.nansum(plantas_supresion_cap[outside]))

    loss_in  = float(_loss(X3_in))
    loss_out = float(_loss(X3_out))
    loss_tot = float(_loss(X3_in + X3_out))

    df_pcc = pd.DataFrame({
        "Segmento": ["Dentro PCC", "Fuera PCC", "Total"],
        "x₂ (sin ctrl)": [X2_in, X2_out, X2_in + X2_out],
        "x₃ (con ctrl)": [X3_in, X3_out, X3_in + X3_out],
        "Pérdida % (modelo)": [loss_in, loss_out, loss_tot]
    })

    st.dataframe(df_pcc.style.format({"x₂ (sin ctrl)":"{:.1f}","x₃ (con ctrl)":"{:.1f}","Pérdida % (modelo)":"{:.2f}"}), use_container_width=True)

    fig_pcc = go.Figure()
    fig_pcc.add_trace(go.Bar(name="x₃ (con ctrl)", x=df_pcc["Segmento"], y=df_pcc["x₃ (con ctrl)"]))
    fig_pcc.add_trace(go.Scatter(name="Pérdida %", x=df_pcc["Segmento"], y=df_pcc["Pérdida % (modelo)"],
                                 mode="lines+markers", yaxis="y2"))
    fig_pcc.update_layout(
        title="Contribución de x₃ y Pérdida (%) por segmento PCC",
        yaxis_title="x₃ (pl·m²)",
        yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_pcc, use_container_width=True)

st.markdown("---")
st.header("🧠 Optimización")
# (El bloque de optimización completo continúa en el BLOQUE 2)












