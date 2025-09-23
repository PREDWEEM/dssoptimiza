# -*- coding: utf-8 -*-
# app_emergencia.py — PREDWEEM (Supresión + Control + Cohortes) con Optimización y Mejor Escenario

import io, re, json, math, datetime as dt
from datetime import timedelta, date
import itertools, random, math as _math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# -------------------------- Estado UI optimización --------------------------
if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_stop" not in st.session_state:
    st.session_state.opt_stop = False

APP_TITLE = "PREDWEEM · Supresión (1−Ciec) + Control (AUC) + Cohortes · Optimización"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# -------------------------- Constantes globales --------------------------
NR_DAYS_DEFAULT = 10               # duración por defecto no residual
POST_GRAM_FORWARD_DAYS = 11        # ventana fija día 0 + 10 (NO TOCAR)
PRE_R_MIN_DAYS_BEFORE_SOW = 14     # presiembra residual: como máximo hasta siembra-14
PREEM_R_MAX_AFTER_SOW_DAYS = 10    # preemergente residual: como máximo hasta siembra+10
POST_R_MIN_AFTER_SOW_DAYS = 20     # post residual: mínimo siembra+20

# “banderas” de UI
SHOW_PLANTS_AXIS_MAX = 100

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
    with urlopen(req, timeout=30) as r:
        return r.read()

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
    if decimal == ",":
        t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else:
        t = t.str.replace(",","",regex=False)
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
    "Cohortes S1..S4 (edad desde emergencia)."
)

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

with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

# ========================= Esquema de tratamientos (inline) =========================
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator

TipoDecaimiento = Literal["Ninguno", "Lineal", "Exponencial"]
TipoTrat = Literal[
    "presiembra_residual",
    "preemergente_residual",
    "post_residual",
    "post_graminicida",
    "pre_no_residual",
    "post_no_residual"
]

class Decaimiento(BaseModel):
    tipo: TipoDecaimiento = "Ninguno"
    semivida_dias: Optional[int] = None

class Tratamiento(BaseModel):
    id: str
    tipo: TipoTrat
    fecha_inicio: date
    duracion_dias: int = Field(gt=0)
    eficiencia_pct: float = Field(ge=0, le=100)
    estados: List[Literal["S1","S2","S3","S4"]]
    decaimiento: Optional[Decaimiento] = Decaimiento()
    nota: Optional[str] = None

class CiecCFG(BaseModel):
    usar: bool = True
    Ca: float = 250
    Cs: float = 250
    LAIhc: float = 3.5

class CanopiaCFG(BaseModel):
    modo: Literal["Cobertura","LAI"] = "Cobertura"
    t_lag: int = 7
    t_cierre: int = 45
    cov_max_pct: float = 85
    lai_max: float = 3.5
    k_beer: float = 0.6
    Ciec: CiecCFG = CiecCFG()

class Observaciones(BaseModel):
    x_final_plm2: Optional[float] = None

class Escenario(BaseModel):
    metadata: Optional[dict] = {}
    siembra: date
    tope_A2_plm2: int = 250
    observaciones: Observaciones = Observaciones()
    canopia: CanopiaCFG = CanopiaCFG()
    tratamientos: List[Tratamiento] = []

    @validator("tratamientos")
    def reglas_tratamientos(cls, ts, values):
        sow: date = values.get("siembra")
        if sow is None: return ts
        for t in ts:
            if t.tipo == "presiembra_residual":
                if not (t.fecha_inicio <= sow - timedelta(days=PRE_R_MIN_DAYS_BEFORE_SOW)):
                    raise ValueError(f"{t.id} presiembra_residual: fecha ≤ {sow - timedelta(days=PRE_R_MIN_DAYS_BEFORE_SOW)}")
                if set(t.estados) - {"S1","S2"}:
                    raise ValueError(f"{t.id} presiembra_residual: sólo S1 y S2")

            if t.tipo == "preemergente_residual":
                if not (t.fecha_inicio <= sow + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
                    raise ValueError(f"{t.id} preemergente_residual: fecha ≤ {sow + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)}")
                if set(t.estados) - {"S1","S2"}:
                    raise ValueError(f"{t.id} preemergente_residual: sólo S1 y S2")

            if t.tipo == "post_residual":
                if not (t.fecha_inicio >= sow + timedelta(days=POST_R_MIN_AFTER_SOW_DAYS)):
                    raise ValueError(f"{t.id} post_residual: fecha ≥ {sow + timedelta(days=POST_R_MIN_AFTER_SOW_DAYS)}")

            if t.tipo == "post_graminicida":
                if t.duracion_dias != POST_GRAM_FORWARD_DAYS:
                    raise ValueError(f"{t.id} post_graminicida: duracion_dias debe ser {POST_GRAM_FORWARD_DAYS}")
                if t.fecha_inicio < sow:
                    raise ValueError(f"{t.id} post_graminicida: fecha_inicio ≥ {sow}")
        return ts

# ------------------- Carga de escenario externo (JSON/CSV) -------------------
with st.sidebar:
    st.header("Escenario externo (JSON / CSV)")
    use_external = st.checkbox("Usar escenario externo", value=False)
    ext_json = st.file_uploader("escenario.json (todo en uno)", type=["json"], key="esc_json")
    ext_trat_csv = st.file_uploader("tratamientos.csv (opcional)", type=["csv"], key="esc_trat_csv")
    st.caption("Si subís JSON, no hace falta el CSV. Si subís ambos, el CSV reemplaza la lista de tratamientos del JSON.")

escenario_ext = None
if use_external and (ext_json is not None):
    try:
        data = json.loads(ext_json.read().decode("utf-8"))
        escenario_ext = Escenario(**data)
        st.success("Escenario JSON validado.")
    except Exception as e:
        st.error(f"Error validando escenario JSON: {e}")
        use_external = False

if use_external and escenario_ext is not None and (ext_trat_csv is not None):
    try:
        import io as _io
        dfT = pd.read_csv(_io.BytesIO(ext_trat_csv.read()))
        trats = []
        for _,r in dfT.iterrows():
            estados = str(r["estados"]).split("|") if "estados" in r and pd.notna(r["estados"]) else []
            dec = Decaimiento(
                tipo = r.get("decaimiento_tipo","Ninguno"),
                semivida_dias = int(r["semivida_dias"]) if "semivida_dias" in r and pd.notna(r["semivida_dias"]) else None
            )
            trats.append(Tratamiento(
                id = str(r["id"]),
                tipo = r["tipo"],
                fecha_inicio = pd.to_datetime(r["fecha_inicio"]).date(),
                duracion_dias = int(r["duracion_dias"]),
                eficiencia_pct = float(r["eficiencia_pct"]),
                estados = estados,
                decaimiento = dec,
                nota = r.get("nota", None)
            ))
        escenario_ext.tratamientos = trats
        st.success("Tratamientos CSV cargados y aplicados al escenario.")
    except Exception as e:
        st.error(f"Error leyendo tratamientos.csv: {e}")
        use_external = False

# ------------ Si hay escenario externo, sobreescribo parámetros ------------
if use_external and (escenario_ext is not None):
    try:
        sow_date = escenario_ext.siembra
        MAX_PLANTS_CAP = float(escenario_ext.tope_A2_plm2)
        mode_canopy = "Cobertura dinámica (%)" if escenario_ext.canopia.modo == "Cobertura" else "LAI dinámico"
        t_lag = escenario_ext.canopia.t_lag
        t_close = escenario_ext.canopia.t_cierre
        cov_max = escenario_ext.canopia.cov_max_pct
        lai_max = escenario_ext.canopia.lai_max
        k_beer = escenario_ext.canopia.k_beer
        use_ciec = escenario_ext.canopia.Ciec.usar
        Ca = escenario_ext.canopia.Ciec.Ca
        Cs = escenario_ext.canopia.Ciec.Cs
        LAIhc = escenario_ext.canopia.Ciec.LAIhc
        st.info(f"Escenario externo activo · Siembra={sow_date} · Tope A2={int(MAX_PLANTS_CAP)} pl·m²")
    except Exception as e:
        st.error(f"No se pudieron aplicar parámetros del escenario: {e}")
        use_external = False

# ========================= Periodo crítico (PC) ========================
with st.sidebar:
    st.header("Periodo crítico")
    use_pc = st.checkbox("Resaltar periodo crítico (11-Sep→15-Nov)", value=False)

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
births = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()
births = np.where(mask_since_sow.to_numpy(), births, 0.0)
births_series = pd.Series(births, index=ts)

def roll_sum_shift(s: pd.Series, win: int, shift_days: int) -> pd.Series:
    return s.rolling(window=win, min_periods=0).sum().shift(shift_days)

S1_coh = roll_sum_shift(births_series, 6, 1).fillna(0.0)
S2_coh = roll_sum_shift(births_series, 21, 7).fillna(0.0)
S3_coh = roll_sum_shift(births_series, 32, 28).fillna(0.0)
S4_coh = births_series.cumsum().shift(60).fillna(0.0)

FC_S = {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0}
S1_arr = S1_coh.reindex(ts).to_numpy(float)
S2_arr = S2_coh.reindex(ts).to_numpy(float)
S3_arr = S3_coh.reindex(ts).to_numpy(float)
S4_arr = S4_coh.reindex(ts).to_numpy(float)

# ============ AUC/área y equivalencia =============
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(float), mask=mask_since_sow)
if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda
    conv_caption = f"AUC(EMERREL desde siembra) = {auc_cruda:.4f} → {int(MAX_PLANTS_CAP)} pl·m² (factor={factor_area_to_plants:.4f})"
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# ============ Aportes por estado (pl·m²·día⁻¹, sin control) ============
if factor_area_to_plants is not None:
    ms = mask_since_sow.to_numpy()
    S1_pl = np.where(ms, S1_arr * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants, 0.0)
    S2_pl = np.where(ms, S2_arr * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants, 0.0)
    S3_pl = np.where(ms, S3_arr * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants, 0.0)
    S4_pl = np.where(ms, S4_arr * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants, 0.0)
else:
    S1_pl=S2_pl=S3_pl=S4_pl=np.full(len(ts), np.nan)

# Controles multiplicativos por estado (inicial=1)
fechas_d = ts.dt.date.values
ctrl_S1 = np.ones_like(fechas_d, float)
ctrl_S2 = np.ones_like(fechas_d, float)
ctrl_S3 = np.ones_like(fechas_d, float)
ctrl_S4 = np.ones_like(fechas_d, float)

# ================== Panel manual (opcional si NO usás escenario externo) ==================
with st.sidebar:
    st.header("Manejo manual (opcional)")
    min_date = ts.min().date(); max_date = ts.max().date()

    # Presiembra residual (nueva regla: ≤ siembra-14; S1,S2)
    pre_R_on = st.checkbox("Presiembra residual (S1–S2)", value=False, help="Debe ser ≤ siembra−14")
    pre_R_date = st.date_input("Fecha presiembra residual", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_R_on)
    pre_R_days = st.slider("Residualidad presiembra (días)", 15, 120, 45, 1, disabled=not pre_R_on)
    ef_pre_R   = st.slider("Eficiencia presiembra (%)", 0, 100, 70, 1, disabled=not pre_R_on)

    # Preemergente residual (nueva regla: ≤ siembra+10; S1,S2)
    preem_R_on = st.checkbox("Preemergente residual (S1–S2)", value=False, help="Hasta siembra+10")
    preem_R_date = st.date_input("Fecha preemergente residual", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not preem_R_on)
    preem_R_days = st.slider("Residualidad preemergente (días)", 15, 120, 45, 1, disabled=not preem_R_on)
    ef_preem_R   = st.slider("Eficiencia preemergente (%)", 0, 100, 70, 1, disabled=not preem_R_on)

    # Post residual (≥ siembra+20; S1..S4 por defecto, pero lo podés ajustar si querés)
    post_R_on = st.checkbox("Postemergente residual", value=False, help="≥ siembra+20")
    post_R_date = st.date_input("Fecha post residual", value=max(min_date, sow_date + timedelta(days=20)), min_value=min_date, max_value=max_date, disabled=not post_R_on)
    post_R_days = st.slider("Residualidad post (días)", 15, 120, 45, 1, disabled=not post_R_on)
    ef_post_R   = st.slider("Eficiencia post residual (%)", 0, 100, 70, 1, disabled=not post_R_on)
    states_postR = st.multiselect("Estados afectados (post R)", ["S1","S2","S3","S4"], default=["S1","S2","S3","S4"], disabled=not post_R_on)

    # Post graminicida (ventana fija +10; ≥ siembra)
    post_G_on = st.checkbox("Post graminicida (+10d)", value=False, help="≥ siembra; ventana fija día 0+10")
    post_G_date = st.date_input("Fecha post graminicida", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_G_on)
    ef_post_G   = st.slider("Eficiencia graminicida (%)", 0, 100, 65, 1, disabled=not post_G_on)

# Validaciones y advertencias
warnings = []
if pre_R_on and not (pre_R_date <= sow_date - timedelta(days=PRE_R_MIN_DAYS_BEFORE_SOW)):
    warnings.append(f"Presiembra residual: debe ser ≤ {sow_date - timedelta(days=PRE_R_MIN_DAYS_BEFORE_SOW)}.")
if preem_R_on and not (preem_R_date <= sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
    warnings.append(f"Preemergente residual: debe ser ≤ {sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)}.")
if post_R_on and (post_R_date < sow_date + timedelta(days=POST_R_MIN_AFTER_SOW_DAYS)):
    warnings.append(f"Post residual: debe ser ≥ {sow_date + timedelta(days=POST_R_MIN_AFTER_SOW_DAYS)}.")
if post_G_on and (post_G_date < sow_date):
    warnings.append(f"Post graminicida: debe ser ≥ fecha de siembra ({sow_date}).")
for w in warnings: st.warning(w)

# ---------------- Aplicación de tratamientos (funciones comunes) ----------------
def apply_efficiency_per_state(weights, eff_pct, states_sel, c1, c2, c3, c4):
    if eff_pct <= 0 or (not states_sel): return c1, c2, c3, c4
    reduc = np.clip(1.0 - (eff_pct/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
    if "S1" in states_sel: np.multiply(c1, reduc, out=c1)
    if "S2" in states_sel: np.multiply(c2, reduc, out=c2)
    if "S3" in states_sel: np.multiply(c3, reduc, out=c3)
    if "S4" in states_sel: np.multiply(c4, reduc, out=c4)
    return c1, c2, c3, c4

def weights_residual(start_date, dias, decaimiento_tipo="Ninguno", semivida=None):
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
        lam = math.log(2) / max(1e-6, semivida if semivida else 20)
        w[idxs] = np.exp(-lam * t_rel)
    return w

def weights_one_day(date_val):
    if not date_val: return np.zeros_like(fechas_d, float)
    d0 = date_val
    return ((fechas_d >= d0) & (fechas_d < (d0 + timedelta(days=1)))).astype(float)

# ---------------- Aplicar MANEJO MANUAL (si NO hay escenario externo) ----------------
if not (use_external and (escenario_ext is not None)) and (factor_area_to_plants is not None):
    # Presiembra residual (S1–S2)
    if pre_R_on and (pre_R_date <= sow_date - timedelta(days=PRE_R_MIN_DAYS_BEFORE_SOW)):
        w = weights_residual(pre_R_date, pre_R_days, "Ninguno")
        ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4 = apply_efficiency_per_state(w, ef_pre_R, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4)

    # Preemergente residual (S1–S2)
    if preem_R_on and (preem_R_date <= sow_date + timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
        w = weights_residual(preem_R_date, preem_R_days, "Ninguno")
        ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4 = apply_efficiency_per_state(w, ef_preem_R, ["S1","S2"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4)

    # Post residual (S1..S4 según selección)
    if post_R_on and (post_R_date >= sow_date + timedelta(days=POST_R_MIN_AFTER_SOW_DAYS)):
        w = weights_residual(post_R_date, post_R_days, "Ninguno")
        ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4 = apply_efficiency_per_state(w, ef_post_R, states_postR, ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4)

    # Post graminicida (+10)
    if post_G_on and (post_G_date >= sow_date):
        w = weights_residual(post_G_date, POST_GRAM_FORWARD_DAYS, "Ninguno")
        ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4 = apply_efficiency_per_state(w, ef_post_G, ["S1","S2","S3"], ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4)

# ---------------- Aplicar TRATAMIENTOS DEL ESCENARIO EXTERNO ----------------
def apply_tratamiento_ext(trat, c1, c2, c3, c4, sow):
    # Validaciones ya pasadas por Pydantic; acá sólo construimos ventana & aplicamos
    if trat.tipo in ["presiembra_residual","preemergente_residual","post_residual"]:
        if trat.decaimiento and trat.decaimiento.tipo == "Exponencial":
            w = weights_residual(trat.fecha_inicio, trat.duracion_dias, "Exponencial", trat.decaimiento.semivida_dias)
        elif trat.decaimiento and trat.decaimiento.tipo == "Lineal":
            w = weights_residual(trat.fecha_inicio, trat.duracion_dias, "Lineal")
        else:
            w = weights_residual(trat.fecha_inicio, trat.duracion_dias, "Ninguno")
    elif trat.tipo in ["pre_no_residual","post_no_residual"]:
        w = weights_one_day(trat.fecha_inicio)
    elif trat.tipo == "post_graminicida":
        w = weights_residual(trat.fecha_inicio, POST_GRAM_FORWARD_DAYS, "Ninguno")
    else:
        w = np.zeros_like(fechas_d, float)
    return apply_efficiency_per_state(w, trat.eficiencia_pct, trat.estados, c1, c2, c3, c4)

if (use_external and (escenario_ext is not None)) and (factor_area_to_plants is not None):
    ctrl_S1[:] = 1.0; ctrl_S2[:] = 1.0; ctrl_S3[:] = 1.0; ctrl_S4[:] = 1.0
    for trat in escenario_ext.tratamientos:
        try:
            ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4 = apply_tratamiento_ext(trat, ctrl_S1, ctrl_S2, ctrl_S3, ctrl_S4, escenario_ext.siembra)
        except Exception as e:
            st.warning(f"Tratamiento {trat.id} no aplicado: {e}")

# ==================== Series con y sin control + cap A2 ====================
if factor_area_to_plants is not None:
    plantas_supresion      = (S1_pl + S2_pl + S3_pl + S4_pl)
    S1_pl_ctrl = S1_pl * ctrl_S1
    S2_pl_ctrl = S2_pl * ctrl_S2
    S3_pl_ctrl = S3_pl * ctrl_S3
    S4_pl_ctrl = S4_pl * ctrl_S4
    plantas_supresion_ctrl = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)

    base_pl_daily = np.where(mask_since_sow.to_numpy(), df_plot["EMERREL"].to_numpy(float) * factor_area_to_plants, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())

    plantas_supresion_cap      = np.minimum(plantas_supresion, base_pl_daily_cap)
    plantas_supresion_ctrl_cap = np.minimum(plantas_supresion_ctrl, plantas_supresion_cap)

    # Reescalado proporcional por estado
    total_ctrl_daily = (S1_pl_ctrl + S2_pl_ctrl + S3_pl_ctrl + S4_pl_ctrl)
    eps = 1e-12
    scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, plantas_supresion_ctrl_cap / total_ctrl_daily), 0.0)
    S1_pl_ctrl_cap = S1_pl_ctrl * scale
    S2_pl_ctrl_cap = S2_pl_ctrl * scale
    S3_pl_ctrl_cap = S3_pl_ctrl * scale
    S4_pl_ctrl_cap = S4_pl_ctrl * scale
    plantas_supresion_ctrl_cap = S1_pl_ctrl_cap + S2_pl_ctrl_cap + S3_pl_ctrl_cap + S4_pl_ctrl_cap
else:
    plantas_supresion_cap = plantas_supresion_ctrl_cap = base_pl_daily_cap = np.full(len(ts), np.nan)
    S1_pl_ctrl_cap=S2_pl_ctrl_cap=S3_pl_ctrl_cap=S4_pl_ctrl_cap=np.full(len(ts), np.nan)

# Agregación semanal
df_daily_cap = pd.DataFrame({
    "fecha": ts,
    "pl_sin_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_cap, 0.0),
    "pl_con_ctrl_cap": np.where(mask_since_sow.to_numpy(), plantas_supresion_ctrl_cap, 0.0),
    "pl_base_cap":     np.where(mask_since_sow.to_numpy(), base_pl_daily_cap, 0.0),
})
df_week_cap = df_daily_cap.set_index("fecha").resample("W-MON").sum().reset_index()

# A2 por AUC
if factor_area_to_plants is not None and auc_cruda > 0:
    sup_equiv  = np.divide(plantas_supresion_cap,     factor_area_to_plants, out=np.zeros_like(plantas_supresion_cap),     where=(factor_area_to_plants>0))
    supc_equiv = np.divide(plantas_supresion_ctrl_cap, factor_area_to_plants, out=np.zeros_like(plantas_supresion_ctrl_cap), where=(factor_area_to_plants>0))
    auc_sup      = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_sup_ctrl = auc_time(ts, supc_equiv, mask=mask_since_sow)
    A2_sup_final  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup      / auc_cruda))
    A2_ctrl_final = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda))
else:
    A2_sup_final = A2_ctrl_final = float("nan")

# x y pérdidas
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
fig.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))

layout_kwargs = dict(margin=dict(l=10, r=10, t=40, b=10),
                     title=f"EMERREL + aportes (izq) y Plantas·m²·semana (der, 0–{SHOW_PLANTS_AXIS_MAX}) · Tope={int(MAX_PLANTS_CAP)}",
                     xaxis_title="Tiempo", yaxis_title="EMERREL")

if factor_area_to_plants is not None and show_plants_axis:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right",
                                   title=f"Plantas·m²·sem⁻¹ (cap A2={int(MAX_PLANTS_CAP)})",
                                   position=1.0, range=[0, SHOW_PLANTS_AXIS_MAX], tick0=0, dtick=20, showgrid=False)
    fig.add_trace(go.Scatter(x=df_week_cap["fecha"], y=df_week_cap["pl_sin_ctrl_cap"], name="Aporte semanal (sin control, cap)",
                             yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_week_cap["fecha"], y=df_week_cap["pl_con_ctrl_cap"], name="Aporte semanal (con control, cap)",
                             yaxis="y2", mode="lines+markers", line=dict(dash="dot")))
if use_pc:
    fig.add_vrect(x0=PC_START, x1=PC_END, line_width=0, fillcolor="MediumPurple", opacity=0.12)

if use_ciec and show_ciec_curve:
    fig.update_layout(yaxis3=dict(overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]))
    fig.add_trace(go.Scatter(x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3"))

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2_sup={A2_sup_final if np.isfinite(A2_sup_final) else float('nan'):.1f} · A2_ctrl={A2_ctrl_final if np.isfinite(A2_ctrl_final) else float('nan'):.1f}")

# ================= Figura 2 (manual): Pérdida (%) vs x =================
st.subheader("Figura 2 — Pérdida de rendimiento (%) vs. x")
if np.isfinite(X2) or np.isfinite(X3):
    x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
    y_curve = 0.375 * x_curve / (1.0 + (0.375 * x_curve / 76.639))
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x"))
    if np.isfinite(X2): fig_loss.add_trace(go.Scatter(x=[X2], y=[loss_x2_pct], mode="markers+text", name="x₂ sin control", text=[f"x₂={X2:.1f}"], textposition="top center"))
    if np.isfinite(X3): fig_loss.add_trace(go.Scatter(x=[X3], y=[loss_x3_pct], mode="markers+text", name="x₃ con control", text=[f"x₃={X3:.1f}"], textposition="top right"))
    fig_loss.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("No hay valores finitos de x para graficar.")

# ================= Figura 3 (manual): Composición porcentual PC (donut) =================
st.subheader("Figura 3 — Composición porcentual por estado en el PC")
mask_pc_days = (ts >= PC_START) & (ts <= PC_END)
if factor_area_to_plants is None or not np.isfinite(factor_area_to_plants):
    st.info("AUC cruda = 0 → no se puede escalar a plantas·m²; no es posible calcular aportes en PC.")
else:
    mspc = (mask_since_sow & mask_pc_days).to_numpy()
    a_S1 = float(np.nansum(S1_pl_ctrl_cap[mspc])) if "S1_pl_ctrl_cap" in locals() else 0.0
    a_S2 = float(np.nansum(S2_pl_ctrl_cap[mspc])) if "S2_pl_ctrl_cap" in locals() else 0.0
    a_S3 = float(np.nansum(S3_pl_ctrl_cap[mspc])) if "S3_pl_ctrl_cap" in locals() else 0.0
    a_S4 = float(np.nansum(S4_pl_ctrl_cap[mspc])) if "S4_pl_ctrl_cap" in locals() else 0.0
    tot  = a_S1 + a_S2 + a_S3 + a_S4
    labels = ["S1 (FC=0.0)", "S2 (FC=0.3)", "S3 (FC=0.6)", "S4 (FC=1.0)"]
    if np.isfinite(tot) and tot > 0:
        pct = 100.0 * np.array([a_S1, a_S2, a_S3, a_S4], float) / tot
        st.markdown(f"**Total (S1–S4) en PC:** **{tot:,.1f}** pl·m²")
        df_pc_pct = pd.DataFrame({"Estado": labels, "% del total PC": pct}).sort_values("% del total PC", ascending=False).reset_index(drop=True)
        st.dataframe(df_pc_pct, use_container_width=True)
        fig_pc_donut = go.Figure(data=[go.Pie(labels=labels, values=pct, hole=0.5, textinfo="label+percent")])
        fig_pc_donut.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pc_donut, use_container_width=True)
    else:
        st.info("Total en PC es 0 o no finito; no se puede calcular porcentaje.")

# ================= Figura 4 — Dinámica semanal S1–S4 (con control + cap) =================
try:
    df_states_daily = pd.DataFrame({
        "fecha": ts,
        "S1": np.where(mask_since_sow.to_numpy(), S1_pl_ctrl_cap, 0.0),
        "S2": np.where(mask_since_sow.to_numpy(), S2_pl_ctrl_cap, 0.0),
        "S3": np.where(mask_since_sow.to_numpy(), S3_pl_ctrl_cap, 0.0),
        "S4": np.where(mask_since_sow.to_numpy(), S4_pl_ctrl_cap, 0.0),
    })
    df_states_week = df_states_daily.set_index("fecha").resample("W-MON").sum().reset_index()
    st.subheader("Figura 4 — Aportes semanales por estado (con control + cap)")
    fig_states = go.Figure()
    for col, name in [("S1","S1 (FC=0.0)"),("S2","S2 (FC=0.3)"),("S3","S3 (FC=0.6)"),("S4","S4 (FC=1.0)")]:
        fig_states.add_trace(go.Scatter(x=df_states_week["fecha"], y=df_states_week[col], mode="lines", name=name, stackgroup="one"))
    fig_states.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_states, use_container_width=True)
except Exception as e:
    st.warning(f"No fue posible dibujar la Figura 4: {e}")




















