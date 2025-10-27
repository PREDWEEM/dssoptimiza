# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Cohortes SECUENCIALES · Optimización
# ===============================================================
# Reglas actualizadas:
# - Presiembra selectivo residual (preR): SOLO ≤ siembra−14 (actúa S1–S2)
# - Preemergente selectivo residual (preemR): [siembra, siembra+10] (S1–S2)
# - Post-emergente selectivo residual (postR): ≥ siembra+20 (S1–S2)
# - Graminicida post: ventana siembra..siembra+10 (S1–S4)
# Jerarquía: preR → preemR → postR → graminicida
# Gateo jerárquico por remanente (<99% acumulado)
# ===============================================================

import io, re, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="PREDWEEM · (1−Ciec) + AUC + Cohortes SECUENCIALES", layout="wide")
st.title("🌾 PREDWEEM — (1−Ciec) + AUC + Cohortes SECUENCIALES · Optimización")

# ---------- PARÁMETROS FIJOS ----------
NR_DAYS_DEFAULT = 10
POST_GRAM_FORWARD_DAYS = 11
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14
PREEM_R_MAX_AFTER_SOW_DAYS = 10
EPS_REMAIN = 1e-9
EPS_EXCLUDE = 0.99

# ---------- FUNCIÓN DE PÉRDIDA ----------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ---------- FUNCIONES DE UTILIDAD ----------
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
    y_arr = np.nan_to_num(y_arr, nan=0.0)
    return float(np.trapz(y_arr, tdays))

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y); cum = 0.0
    for i in range(len(y)):
        if bool(active_mask[i]):
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val; cum += val
    return out

# ---------- PARÁMETROS GENERALES ----------
with st.sidebar:
    st.header("Configuración general")
    MAX_PLANTS_CAP = float(st.selectbox("Tope de densidad efectiva (pl·m²)", [250, 125, 62], index=0))
st.caption(f"AUC(EMERREL cruda) ≙ A2={int(MAX_PLANTS_CAP)} pl·m² · Cohortes S1–S4 SECUENCIALES")

# ---------- CARGA DE DATOS ----------
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria o EMERAC)", type=["csv"])
    url = st.text_input("…o URL raw de GitHub")
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], 0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], 0)
    dayfirst = st.checkbox("Fecha: dd/mm/yyyy", True)
    is_cumulative = st.checkbox("Mi CSV es acumulado (EMERAC)", False)
    as_percent = st.checkbox("Valores en % (no 0–1)", True)

if up is None and not url:
    st.info("Subí un CSV o pegá una URL para continuar.")
    st.stop()

try:
    raw = read_raw(up, url)
    df0, meta = parse_csv(raw, sep_opt, dec_opt)
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}")
    st.stop()

cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, 0)
c_valor = st.selectbox("Columna de valor", cols, 1 if len(cols)>1 else 0)

fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = clean_numeric_series(df0[c_valor])
df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)

emerrel = df["valor"].astype(float)
if as_percent: emerrel /= 100.0
if is_cumulative: emerrel = emerrel.diff().fillna(0).clip(lower=0)
emerrel = emerrel.clip(lower=0.0)
df_plot = pd.DataFrame({"fecha": df["fecha"], "EMERREL": emerrel})

# ---------- CANOPIA ----------
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min, sow_max = dt.date(year_ref, 5, 1), dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra y canopia")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], 0)
    t_lag = st.number_input("Días a emergencia", 0, 60, 7)
    t_close = st.number_input("Días a cierre", 10, 120, 45)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.0)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6)
    use_ciec = st.checkbox("Incluir Ciec", True)
    Ca = st.number_input("Densidad Ca", 50, 700, 250)
    Cs = st.number_input("Densidad Cs", 50, 700, 250)
    LAIhc = st.number_input("LAIhc", 3.0, 6.0, 3.0)

def compute_canopy(fechas, sow, mode, lag, close, cov_max, lai_max, k):
    days = np.array([(pd.Timestamp(d).date() - sow).days for d in fechas])
    def logistic_between(days, start, end, y_max):
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1+np.exp(-r*(days-t_mid)))
    if mode == "Cobertura dinámica (%)":
        fc = np.where(days<lag, 0, logistic_between(days, lag, close, cov_max/100))
        fc = np.clip(fc,0,1); LAI = -np.log(np.clip(1-fc,1e-9,1))/max(1e-6,k)
        LAI = np.clip(LAI,0,lai_max)
    else:
        LAI = np.where(days<lag, 0, logistic_between(days, lag, close, lai_max))
        LAI = np.clip(LAI,0,lai_max); fc = 1 - np.exp(-k*LAI)
    return fc, LAI

FC, LAI = compute_canopy(df_plot["fecha"], sow_date, mode_canopy, int(t_lag), int(t_close), float(cov_max), float(lai_max), float(k_beer))
if use_ciec:
    Ciec = np.clip((LAI / LAIhc) * (Ca / Cs), 0, 1)
else:
    Ciec = np.zeros_like(LAI)
one_minus_Ciec = np.clip(1 - Ciec, 0, 1)


# ===============================================================
# 📗 ESTADOS FENOLÓGICOS (S1–S4) + SUPRESIÓN Y CONTROL
# ===============================================================

ts = pd.to_datetime(df_plot["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)

# EMERREL diario (0–1)
births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where(mask_since_sow.to_numpy(), births, 0.0)

# ---- Duraciones promedio entre estados ----
T12 = st.sidebar.number_input("Duración S1→S2 (días)", 1, 60, 10)
T23 = st.sidebar.number_input("Duración S2→S3 (días)", 1, 60, 15)
T34 = st.sidebar.number_input("Duración S3→S4 (días)", 1, 60, 20)
globals()["T12"], globals()["T23"], globals()["T34"] = int(T12), int(T23), int(T34)

# ---- Simulación compartimental secuencial ----
S1, S2, S3, S4 = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
for i in range(len(births)):
    if i - int(T12) >= 0:
        moved = births[i - int(T12)]; S1[i - int(T12)] -= moved; S2[i] += moved
    if i - (int(T12) + int(T23)) >= 0:
        moved = births[i - (int(T12)+int(T23))]; S2[i - (int(T12)+int(T23))] -= moved; S3[i] += moved
    if i - (int(T12) + int(T23) + int(T34)) >= 0:
        moved = births[i - (int(T12)+int(T23)+int(T34))]; S3[i - (int(T12)+int(T23)+int(T34))] -= moved; S4[i] += moved

S1, S2, S3, S4 = [np.clip(x, 0, None) for x in (S1, S2, S3, S4)]

# ---- Escalado a EMEAC ----
emeac = np.cumsum(births)
total_states = S1 + S2 + S3 + S4
scale = np.divide(np.clip(emeac, 1e-9, None), np.clip(total_states, 1e-9, None))
scale = np.minimum(scale, 1.0)
S1, S2, S3, S4 = S1*scale, S2*scale, S3*scale, S4*scale

# ---- Pesos por estado (competencia relativa) ----
FC_S = {"S1": 0.1, "S2": 0.3, "S3": 0.6, "S4": 1.0}

# ===============================================================
# 📈 ESCALADO A PLANTAS
# ===============================================================
auc_cruda = np.trapz(df_plot["EMERREL"].to_numpy(), np.arange(len(df_plot)))
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
else:
    st.error("AUC(EMERREL) = 0 → no puede escalar a plantas.")
    st.stop()

st.caption(f"Factor de escala: {factor_area_to_plants:.3f} (AUC→pl·m²)")

# Supresión por (1−Ciec)
S1_pl = S1 * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants
S2_pl = S2 * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants
S3_pl = S3 * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants
S4_pl = S4 * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants

# ===============================================================
# 🧪 FUNCIONES DE APLICACIÓN DE HERBICIDAS
# ===============================================================
def weights_residual(start_date, days):
    if not start_date or int(days) <= 0: return np.zeros_like(ts, float)
    d0, d1 = pd.to_datetime(start_date), pd.to_datetime(start_date) + pd.Timedelta(days=int(days))
    return ((ts >= d0) & (ts < d1)).astype(float)

def weights_one_day(date_val):
    if not date_val: return np.zeros_like(ts, float)
    d0 = pd.to_datetime(date_val)
    return ((ts >= d0) & (ts < d0 + pd.Timedelta(days=1))).astype(float)

# ===============================================================
# 💧 FUNCIONES DE APLICACIÓN Y GATEO (NUEVAS REGLAS)
# ===============================================================
def apply_efficiency_per_state(weights, eff_pct, states_sel, ctrls):
    if eff_pct <= 0 or not states_sel: return
    reduc = np.clip(1.0 - (eff_pct / 100.0) * weights, 0.0, 1.0)
    for s in states_sel: np.multiply(ctrls[s], reduc, out=ctrls[s])

def remaining_in_window(weights, states_sel, ctrls):
    rem = 0.0
    for s in states_sel:
        if s == "S1": rem += np.sum(S1_pl * ctrls[s] * weights)
        elif s == "S2": rem += np.sum(S2_pl * ctrls[s] * weights)
        elif s == "S3": rem += np.sum(S3_pl * ctrls[s] * weights)
        elif s == "S4": rem += np.sum(S4_pl * ctrls[s] * weights)
    return rem

def eff_from_to(prev_eff, this_eff):
    return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

# ===============================================================
# ⚙️ CONTROL DE HERBICIDAS (Manual con reglas actualizadas)
# ===============================================================
ctrls = {s: np.ones_like(ts, float) for s in ["S1", "S2", "S3", "S4"]}

# UI básica de prueba (ejemplo de aplicación manual)
with st.sidebar:
    st.header("Manejo manual de control (prueba)")
    preR = st.checkbox("Presiembra + residual (S1–S2)", value=True)
    preemR = st.checkbox("Preemergente + residual (S1–S2)", value=True)
    postR = st.checkbox("Post-emergente + residual (S1–S2)", value=True)
    gram = st.checkbox("Graminicida (S1–S4)", value=True)

# Ejemplo de ventanas
if preR:
    w_preR = weights_residual(sow_date - timedelta(days=16), 30)
    apply_efficiency_per_state(w_preR, 90, ["S1", "S2"], ctrls)

if preemR:
    w_preem = weights_residual(sow_date, 40)
    apply_efficiency_per_state(w_preem, 90, ["S1", "S2"], ctrls)

if postR:
    w_post = weights_residual(sow_date + timedelta(days=25), 45)
    apply_efficiency_per_state(w_post, 90, ["S1", "S2"], ctrls)

if gram:
    w_gram = weights_residual(sow_date + timedelta(days=35), 10)
    apply_efficiency_per_state(w_gram, 90, ["S1", "S2", "S3", "S4"], ctrls)

# Aplicación final de controles
S1_ctrl, S2_ctrl, S3_ctrl, S4_ctrl = [S1_pl * ctrls["S1"], S2_pl * ctrls["S2"], S3_pl * ctrls["S3"], S4_pl * ctrls["S4"]]
plantas_total = S1_pl + S2_pl + S3_pl + S4_pl
plantas_ctrl = S1_ctrl + S2_ctrl + S3_ctrl + S4_ctrl

# ===============================================================
# 🔢 CAP A2 (Tope máximo)
# ===============================================================
base_pl_daily = df_plot["EMERREL"].to_numpy() * factor_area_to_plants
base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())
plantas_ctrl_cap = np.minimum(plantas_ctrl, base_pl_daily_cap)

A2_sup = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_time(ts, plantas_total, mask_since_sow) / auc_cruda))
A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_time(ts, plantas_ctrl_cap, mask_since_sow) / auc_cruda))
X2 = np.sum(plantas_total[mask_since_sow])
X3 = np.sum(plantas_ctrl_cap[mask_since_sow])
loss_pct = _loss(X3)

st.markdown(f"**A2_sup:** {A2_sup:.1f} · **A2_ctrl:** {A2_ctrl:.1f} · **x₂:** {X2:.1f} · **x₃:** {X3:.1f} · **Pérdida estimada:** {loss_pct:.2f}%")

# ===============================================================
# 📙 PARTE 3 — Optimización + Mejor escenario (gráficos completos)
# ===============================================================

st.markdown("---")
st.header("🧠 Optimización")

# ---------- Helper: (1−Ciec) para una siembra específica ----------
def compute_ciec_for(sow_d: dt.date):
    FCx, LAIx = compute_canopy(
        df_plot["fecha"], sow_d, mode_canopy,
        int(t_lag), int(t_close), float(cov_max), float(lai_max), float(k_beer)
    )
    if use_ciec:
        Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
        Cs_safe = float(Cs) if float(Cs) > 0 else 1e-6
        Ciec_loc = np.clip((LAIx / max(1e-6, float(LAIhc))) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec_loc = np.zeros_like(LAIx, float)
    return np.clip(1.0 - Ciec_loc, 0.0, 1.0)

# ---------- Recompute estados y escalas para una siembra ----------
def recompute_for_sow(sow_d: dt.date, T12: int, T23: int, T34: int):
    ts_all = pd.to_datetime(df_plot["fecha"])
    mask_since = (ts_all.dt.date >= sow_d)
    births = np.where(mask_since.to_numpy(), df_plot["EMERREL"].to_numpy(float), 0.0)
    one_minus = compute_ciec_for(sow_d)

    # Estados S1→S4 (secuenciales)
    S1 = births.copy(); S2 = np.zeros_like(births); S3 = np.zeros_like(births); S4 = np.zeros_like(births)
    for i in range(len(births)):
        if i - int(T12) >= 0:
            moved = births[i - int(T12)]; S1[i - int(T12)] -= moved; S2[i] += moved
        if i - (int(T12) + int(T23)) >= 0:
            moved = births[i - (int(T12)+int(T23))]; S2[i - (int(T12)+int(T23))] -= moved; S3[i] += moved
        if i - (int(T12) + int(T23) + int(T34)) >= 0:
            moved = births[i - (int(T12)+int(T23)+int(T34))]; S3[i - (int(T12)+int(T23)+int(T34))] -= moved; S4[i] += moved

    S1 = np.clip(S1, 0.0, None); S2 = np.clip(S2, 0.0, None); S3 = np.clip(S3, 0.0, None); S4 = np.clip(S4, 0.0, None)

    # Escalado a EMEAC
    emeac = np.cumsum(births)
    total_states = S1 + S2 + S3 + S4
    scale = np.divide(np.clip(emeac, 1e-9, None), np.clip(total_states, 1e-9, None))
    scale = np.minimum(scale, 1.0); S1*=scale; S2*=scale; S3*=scale; S4*=scale

    # Escalado por AUC
    auc_cruda_loc = auc_time(ts_all, df_plot["EMERREL"].to_numpy(float), mask=mask_since)
    if auc_cruda_loc <= 0: return None
    factor_area = MAX_PLANTS_CAP / auc_cruda_loc

    # Aportes ponderados (1−Ciec) y FC_S
    S1_pl = np.where(mask_since, S1 * one_minus * 0.1 * factor_area, 0.0)
    S2_pl = np.where(mask_since, S2 * one_minus * 0.3 * factor_area, 0.0)
    S3_pl = np.where(mask_since, S3 * one_minus * 0.6 * factor_area, 0.0)
    S4_pl = np.where(mask_since, S4 * one_minus * 1.0 * factor_area, 0.0)

    # Tope A2 por día
    base_pl_daily     = np.where(mask_since, df_plot["EMERREL"].to_numpy(float) * factor_area, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since.to_numpy())
    sup_cap = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_pl_daily_cap)

    return {
        "mask_since": mask_since.to_numpy(),
        "factor_area": factor_area,
        "auc_cruda": auc_cruda_loc,
        "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
        "sup_cap": sup_cap,
        "ts": ts_all,
        "fechas_d": ts_all.dt.date.values
    }

# ---------- Acciones con reglas (postR ⇒ S1–S2, gram ⇒ S1–S4) ----------
def act_presiembraR(date_val, R, eff): return {"kind":"preR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_preemR(date_val, R, eff):     return {"kind":"preemR",  "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_post_selR(date_val, R, eff):  return {"kind":"postR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_post_gram(date_val, eff):     return {"kind":"post_gram","date":pd.to_datetime(date_val).date(), "days":POST_GRAM_FORWARD_DAYS, "eff":eff, "states":["S1","S2","S3","S4"]}

# ---------- Evaluación de un cronograma ----------
def evaluate(sd: dt.date, schedule: list):
    sow = pd.to_datetime(sd); sow_plus_20 = sow + pd.Timedelta(days=20)

    # Reglas duras de ventana
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR"  and d < sow_plus_20: return None
        if a["kind"] == "preR"   and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None

    env = recompute_for_sow(sd, int(T12), int(T23), int(T34))
    if env is None: return None
    mask_since = env["mask_since"]; factor_area = env["factor_area"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]; sup_cap = env["sup_cap"]
    ts_local, fechas_d_local = env["ts"], env["fechas_d"]

    # Controles multiplicativos por estado
    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def _remaining_in_window(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1_pl * c1 * w)
        if "S2" in states: rem += np.sum(S2_pl * c2 * w)
        if "S3" in states: rem += np.sum(S3_pl * c3 * w)
        if "S4" in states: rem += np.sum(S4_pl * c4 * w)
        return float(rem)

    def _apply(w, eff, states):
        if eff <= 0: return False
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)
        return True

    eff_accum_pre = eff_accum_pre2 = eff_accum_all = 0.0
    def _eff_from_to(prev_eff, this_eff): return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    # Orden jerárquico fijo
    order = {"preR":0,"preemR":1,"postR":2,"post_gram":3}
    for a in sorted(schedule, key=lambda a: order.get(a["kind"], 9)):
        d0, d1 = a["date"], a["date"] + pd.Timedelta(days=int(a["days"]))
        w = ((fechas_d_local >= d0) & (fechas_d_local < d1)).astype(float)

        if a["kind"] == "preR":
            if _remaining_in_window(w, ["S1","S2"]) > EPS_REMAIN and a["eff"] > 0:
                _apply(w, a["eff"], ["S1","S2"])
                eff_accum_pre = _eff_from_to(0.0, a["eff"]/100.0)

        elif a["kind"] == "preemR":
            if eff_accum_pre < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window(w, ["S1","S2"]) > EPS_REMAIN:
                _apply(w, a["eff"], ["S1","S2"])
                eff_accum_pre2 = _eff_from_to(eff_accum_pre, a["eff"]/100.0)
            else:
                eff_accum_pre2 = eff_accum_pre

        elif a["kind"] == "postR":  # ahora solo S1–S2
            if eff_accum_pre2 < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window(w, ["S1","S2"]) > EPS_REMAIN:
                _apply(w, a["eff"], ["S1","S2"])
                eff_accum_all = _eff_from_to(eff_accum_pre2, a["eff"]/100.0)
            else:
                eff_accum_all = eff_accum_pre2

        elif a["kind"] == "post_gram":  # ahora S1–S4
            if eff_accum_all < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window(w, ["S1","S2","S3","S4"]) > EPS_REMAIN:
                _apply(w, a["eff"], ["S1","S2","S3","S4"])

    # Resultado con cap
    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    X2loc = float(np.nansum(sup_cap[mask_since]))
    X3loc = float(np.nansum(plantas_ctrl_cap[mask_since]))
    loss3 = _loss(X3loc)

    # A2 por AUC
    sup_equiv  = np.divide(sup_cap,          factor_area, out=np.zeros_like(sup_cap),          where=(factor_area>0))
    ctrl_equiv = np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0))
    auc_sup      = auc_time(ts_local, sup_equiv,  mask=mask_since)
    auc_sup_ctrl = auc_time(ts_local, ctrl_equiv, mask=mask_since)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/env["auc_cruda"]))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/env["auc_cruda"]))

    return {"sow": sd, "loss_pct": float(loss3), "x2": X2loc, "x3": X3loc,
            "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

# ---------- Espacio de búsqueda (sidebar) ----------
with st.sidebar:
    st.header("Optimización (variables habilitadas)")
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 2, 1)

    use_preR_opt      = st.checkbox("Incluir presiembra + residual (≤ siembra−14; S1–S2)", value=True)
    use_preemR_opt    = st.checkbox("Incluir preemergente + residual (siembra..siembra+10; S1–S2)", value=True)
    use_post_selR_opt = st.checkbox("Incluir post + residual (≥ siembra + 20; S1–S2)", value=True)
    use_post_gram_opt = st.checkbox(f"Incluir graminicida post (+{POST_GRAM_FORWARD_DAYS-1}d; S1–S4)", value=True)

    ef_preR_opt      = st.slider("Eficiencia presiembraR (%)", 0, 100, 90, 1)   if use_preR_opt else 0
    ef_preemR_opt    = st.slider("Eficiencia preemergenteR (%)", 0, 100, 90, 1) if use_preemR_opt else 0
    ef_post_selR_opt = st.slider("Eficiencia post residual (%)", 0, 100, 90, 1) if use_post_selR_opt else 0
    ef_post_gram_opt = st.slider("Eficiencia graminicida post (%)", 0, 100, 90, 1) if use_post_gram_opt else 0

    st.markdown("### ⏱️ Residualidades por tipo")
    res_min_preR,   res_max_preR   = st.slider("Presiembra residual (min–max)",   15, 120, (30, 45), 5)
    res_step_preR                   = st.number_input("Paso presiembra (días)",     1, 30, 5, 1)
    res_min_preemR, res_max_preemR = st.slider("Preemergente residual (min–max)", 15, 120, (40, 50), 5)
    res_step_preemR                 = st.number_input("Paso preemergente (días)",   1, 30, 5, 1)
    res_min_postR,  res_max_postR  = st.slider("Post residual (min–max)",         15, 120, (45, 60), 5)
    res_step_postR                  = st.number_input("Paso post (días)",           1, 30, 5, 1)

    preR_min_back  = st.number_input("PresiembraR: buscar hasta X días antes de siembra", 14, 120, 14, 1)
    preR_step_days = st.number_input("Paso fechas PRESIEMBRA (días)", 1, 30, 2, 1)
    preem_step_days = st.number_input("Paso fechas PREEMERGENTE (días)", 1, 10, 2, 1)
    post_days_fw   = st.number_input("Post: días después de siembra (máximo)", 20, 180, 60, 1)
    post_step_days = st.number_input("Paso fechas POST (días)", 1, 30, 2, 1)

    optimizer  = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100)
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1)

    if "opt_running" not in st.session_state: st.session_state.opt_running = False
    if "opt_stop" not in st.session_state:    st.session_state.opt_stop = False

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

# ---------- Utilidades del espacio de búsqueda ----------
def _make_residual_list(rmin, rmax, rstep):
    L = list(range(int(rmin), int(rmax)+1, int(rstep)))
    if int(rmax) not in L: L.append(int(rmax))
    return L

res_days_preR   = _make_residual_list(res_min_preR,   res_max_preR,   res_step_preR)
res_days_preemR = _make_residual_list(res_min_preemR, res_max_preemR, res_step_preemR)
res_days_postR  = _make_residual_list(res_min_postR,  res_max_postR,  res_step_postR)

def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur=cur+pd.Timedelta(days=int(step_days))
    return out

sow_candidates = daterange(sow_search_from, sow_search_to, sow_step_days)

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_min_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preR_step_days))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd); end = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preem_step_days))
    return out

def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_days_fw))
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(post_step_days))
    return out

# ---------- Construcción de escenarios ----------
def build_all_scenarios():
    scenarios = []
    for sd in sow_candidates:
        grp = []
        if use_preR_opt:
            grp.append([act_presiembraR(d, R, ef_preR_opt) for d in pre_sow_dates(sd) for R in res_days_preR])
        if use_preemR_opt:
            grp.append([act_preemR(d, R, ef_preemR_opt) for d in preem_dates(sd) for R in res_days_preemR])
        if use_post_selR_opt:
            grp.append([act_post_selR(d, R, ef_post_selR_opt) for d in post_dates(sd) for R in res_days_postR])
        if use_post_gram_opt:
            grp.append([act_post_gram(d, ef_post_gram_opt) for d in post_dates(sd)])
        combos = [[]]
        for r in range(1, len(grp)+1):
            for subset in itertools.combinations(range(len(grp)), r):
                for p in itertools.product(*[grp[i] for i in subset]):
                    combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random_scenario():
    sd = random.choice(sow_candidates)
    schedule = []
    if use_preR_opt and random.random()<0.7:
        cand = pre_sow_dates(sd)
        if cand: schedule.append(act_presiembraR(random.choice(cand), random.choice(res_days_preR), ef_preR_opt))
    if use_preemR_opt and random.random()<0.7:
        cand = preem_dates(sd)
        if cand: schedule.append(act_preemR(random.choice(cand), random.choice(res_days_preemR), ef_preemR_opt))
    if use_post_selR_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(act_post_selR(random.choice(cand), random.choice(res_days_postR), ef_post_selR_opt))
    if use_post_gram_opt and random.random()<0.7:
        cand = post_dates(sd)
        if cand: schedule.append(act_post_gram(random.choice(cand), ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

# ---------- Ejecución del optimizador ----------
status_ph = st.empty()
prog_ph = st.empty()
results = []

if sow_search_from > sow_search_to:
    st.error("Rango de siembra inválido (desde > hasta).")
elif st.session_state.opt_running:
    status_ph.info("Optimizando…")
    if optimizer == "Grid (combinatorio)":
        scenarios = build_all_scenarios()
        total = len(scenarios)
        st.caption(f"Se evaluarán {total:,} configuraciones")
        if total > max_evals:
            random.seed(123)
            scenarios = random.sample(scenarios, k=int(max_evals))
            st.caption(f"Se muestrean {len(scenarios):,} configs (límite)")
        prog = prog_ph.progress(0.0); n = len(scenarios); step = max(1, n//100)
        for i,(sd,sch) in enumerate(scenarios,1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenida. Progreso: {i-1:,}/{n:,}")
                break
            r = evaluate(sd, sch)
            if r is not None: results.append(r)
            if i % step == 0 or i == n: prog.progress(min(1.0, i/n))
        prog_ph.empty()

    elif optimizer == "Búsqueda aleatoria":
        N = int(max_evals); prog = prog_ph.progress(0.0)
        for i in range(1, N+1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenida. Progreso: {i-1:,}/{N:,}")
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
            best_eval = cur_eval; cur_loss = cur_eval["loss_pct"]; T = float(sa_T0)
            prog = prog_ph.progress(0.0)
            for it in range(1, int(sa_iters)+1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Detenida en iteración {it-1:,}/{int(sa_iters):,}.")
                    break
                cand = sample_random_scenario()
                cand_eval = evaluate(*cand)
                if cand_eval is not None:
                    d = cand_eval["loss_pct"] - cur_loss
                    if d <= 0 or random.random() < math.exp(-d / max(1e-9, T)):
                        cur, cur_eval, cur_loss = cand, cand_eval, cand_eval["loss_pct"]
                        results.append(cur_eval)
                        if cur_loss < best_eval["loss_pct"]:
                            best_eval = cur_eval
                T *= float(sa_cooling)
                if it % max(1, int(sa_iters)//100) == 0 or it == int(sa_iters):
                    prog.progress(min(1.0, it/float(sa_iters)))
            results.append(best_eval)
            prog_ph.empty()

    st.session_state.opt_running = False
    st.session_state.opt_stop = False
    status_ph.success("Optimización finalizada.")
else:
    status_ph.info("Listo para optimizar. Ajustá parámetros y presioná **Iniciar**.")

# ---------- Reporte y gráficos del mejor escenario ----------
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

    # Tabla y descarga del cronograma
    def schedule_df(sch):
        rows=[]
        for a in sch:
            ini = pd.to_datetime(a["date"]); fin = ini + pd.Timedelta(days=int(a["days"]))
            rows.append({
                "Intervención": a["kind"],
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
                           "mejor_cronograma.csv", "text/csv")

    # Recomputo completo del mejor para graficar
    envb = recompute_for_sow(pd.to_datetime(best["sow"]).date(), int(T12), int(T23), int(T34))
    if envb is None:
        st.info("No se pudieron recomputar series para el mejor escenario.")
    else:
        ts_b = envb["ts"]; fechas_d_b = envb["fechas_d"]; mask_since_b = envb["mask_since"]
        S1p, S2p, S3p, S4p = envb["S_pl"]; sup_cap_b = envb["sup_cap"]

        # Reaplicar agenda con las mismas reglas
        c1 = np.ones_like(fechas_d_b, float); c2 = np.ones_like(fechas_d_b, float)
        c3 = np.ones_like(fechas_d_b, float); c4 = np.ones_like(fechas_d_b, float)

        def _remaining_in_window_eval(w, states):
            rem = 0.0
            if "S1" in states: rem += np.sum(S1p * c1 * w)
            if "S2" in states: rem += np.sum(S2p * c2 * w)
            if "S3" in states: rem += np.sum(S3p * c3 * w)
            if "S4" in states: rem += np.sum(S4p * c4 * w)
            return float(rem)

        def _apply_eval(w, eff, states):
            if eff <= 0: return False
            reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
            if "S1" in states: np.multiply(c1, reduc, out=c1)
            if "S2" in states: np.multiply(c2, reduc, out=c2)
            if "S3" in states: np.multiply(c3, reduc, out=c3)
            if "S4" in states: np.multiply(c4, reduc, out=c4)
            return True

        eff_accum_pre = eff_accum_pre2 = eff_accum_all = 0.0
        def _eff_from_to(prev_eff, this_eff): return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)
        order = {"preR":0,"preemR":1,"postR":2,"post_gram":3}

        for a in sorted(best["schedule"], key=lambda a: order.get(a["kind"], 9)):
            ini = pd.to_datetime(a["date"]).date()
            fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
            w = ((fechas_d_b >= ini) & (fechas_d_b < fin)).astype(float)

            if a["kind"] == "preR":
                if _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN and a["eff"] > 0:
                    _apply_eval(w, a["eff"], ["S1","S2"])
                    eff_accum_pre = _eff_from_to(0.0, a["eff"]/100.0)

            elif a["kind"] == "preemR":
                if eff_accum_pre < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2"])
                    eff_accum_pre2 = _eff_from_to(eff_accum_pre, a["eff"]/100.0)
                else:
                    eff_accum_pre2 = eff_accum_pre

            elif a["kind"] == "postR":  # S1–S2
                if eff_accum_pre2 < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2"])
                    eff_accum_all = _eff_from_to(eff_accum_pre2, a["eff"]/100.0)
                else:
                    eff_accum_all = eff_accum_pre2

            elif a["kind"] == "post_gram":  # S1–S4
                if eff_accum_all < EPS_EXCLUDE and a["eff"] > 0 and _remaining_in_window_eval(w, ["S1","S2","S3","S4"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2","S3","S4"])

        total_ctrl_daily = (S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4)
        eps = 1e-12
        scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, sup_cap_b / total_ctrl_daily), 0.0)
        S1_ctrl_cap_b = S1p * c1 * scale; S2_ctrl_cap_b = S2p * c2 * scale
        S3_ctrl_cap_b = S3p * c3 * scale; S4_ctrl_cap_b = S4p * c4 * scale

        # ===== Gráfico A: EMERREL + aportes semanales con/sin control + Ciec (óptimo) =====
        df_daily_b = pd.DataFrame({
            "fecha": ts_b,
            "pl_sin_ctrl_cap": np.where(mask_since_b, sup_cap_b, 0.0),
            "pl_con_ctrl_cap": np.where(mask_since_b, S1_ctrl_cap_b+S2_ctrl_cap_b+S3_ctrl_cap_b+S4_ctrl_cap_b, 0.0),
        })
        df_week_b = df_daily_b.set_index("fecha").resample("W-MON").sum().reset_index()

        fig_best1 = go.Figure()
        fig_best1.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_sin_ctrl_cap"], name="Aporte semanal (sin control, cap)", yaxis="y2", mode="lines+markers"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_con_ctrl_cap"], name="Aporte semanal (con control, cap)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))

        one_minus_best = compute_ciec_for(pd.to_datetime(best["sow"]).date())
        Ciec_best = 1.0 - one_minus_best
        fig_best1.add_trace(go.Scatter(x=ts_b, y=Ciec_best, mode="lines", name="Ciec (mejor)", yaxis="y3"))

        fig_best1.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            title="EMERREL y plantas·m²·semana · Mejor escenario",
            xaxis_title="Tiempo", yaxis_title="EMERREL",
            yaxis2=dict(overlaying="y", side="right", title="pl·m²·sem⁻¹", range=[0,100]),
            yaxis3=dict(overlaying="y", side="right", title="Ciec", position=0.97, range=[0,1])
        )

        for a in best["schedule"]:
            x0 = pd.to_datetime(a["date"]); x1 = x0 + pd.Timedelta(days=int(a["days"]))
            fig_best1.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="rgba(30,144,255,0.18)", opacity=0.18)
            fig_best1.add_annotation(x=x0 + (x1-x0)/2, y=0.86, xref="x", yref="paper",
                                     text=a["kind"], showarrow=False, bgcolor="rgba(30,144,255,0.85)")
        st.plotly_chart(fig_best1, use_container_width=True)

        # ===== Gráfico B: Pérdida (%) vs x con marcadores x2 y x3 =====
        X2_b = float(np.nansum(sup_cap_b[mask_since_b]))
        X3_b = float(np.nansum((S1_ctrl_cap_b+S2_ctrl_cap_b+S3_ctrl_cap_b+S4_ctrl_cap_b)[mask_since_b]))
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400); y_curve = _loss(x_curve)
        fig2_best = go.Figure()
        fig2_best.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x"))
        fig2_best.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", name="x₂ (sin ctrl)", text=[f"x₂={X2_b:.1f}"], textposition="top center"))
        fig2_best.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", name="x₃ (con ctrl)", text=[f"x₃={X3_b:.1f}"], textposition="top right"))
        fig2_best.update_layout(title="Pérdida de rendimiento (%) vs x", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
        st.plotly_chart(fig2_best, use_container_width=True)

        # ===== Gráfico C: Dinámica S1–S4 semanal (stacked) con control + cap =====
        df_states_week_b = (
            pd.DataFrame({"fecha": ts_b, "S1": S1_ctrl_cap_b, "S2": S2_ctrl_cap_b, "S3": S3_ctrl_cap_b, "S4": S4_ctrl_cap_b})
            .set_index("fecha").resample("W-MON").sum().reset_index()
        )
        st.subheader("Dinámica temporal de S1–S4 (con control + cap) — Mejor escenario")
        fig_states = go.Figure()
        for col in ["S1","S2","S3","S4"]:
            fig_states.add_trace(go.Scatter(x=df_states_week_b["fecha"], y=df_states_week_b[col], mode="lines", name=col, stackgroup="one"))
        fig_states.update_layout(title="Aportes semanales por estado (con control + cap)", xaxis_title="Tiempo", yaxis_title="pl·m²·sem⁻¹")
        st.plotly_chart(fig_states, use_container_width=True)
else:
    st.info("Aún no hay resultados de optimización para mostrar.")




















