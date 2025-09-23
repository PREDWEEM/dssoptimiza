# -*- coding: utf-8 -*-
# calibra.py — Calibración con datos independientes (pérdida de rinde % como objetivo)
# Requisitos: streamlit, numpy, pandas, plotly
# Ejecutar: streamlit run calibra.py

import io, re, math, random, itertools
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ===============================
# Config UI
# ===============================
APP_TITLE = "Calibración — Pérdida de rinde (%) con datos independientes"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ===============================
# Constantes del modelo
# ===============================
MAX_PLANTS_CAP = 250.0         # tope de densidad efectiva A2 (pl·m²)
POST_GRAM_FORWARD_DAYS = 11    # ventana fija post: día 0 + 10
NR_DAYS_DEFAULT = 45           # residualidad por defecto si no viene

# ===============================
# Utils de lectura y limpieza
# ===============================
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample):
        dec_guess = ","
    return sep_guess, dec_guess

def read_csv_bytes(raw: bytes):
    head = raw[:8000].decode("utf-8", "ignore")
    sep, dec = sniff_sep_dec(head)
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python")
    return df

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%","",regex=False)
    if decimal == ",":
        t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else:
        t = t.str.replace(",","",regex=False)
    return pd.to_numeric(t, errors="coerce")

def to_days(ts: pd.Series) -> np.ndarray:
    f = pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]").astype("int64")
    t0 = f[0]
    return ((f - t0)/1e9/86400.0).astype(float)

def auc_time(fecha: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, dtype=float)
    if mask is not None:
        f = f[mask]; y_arr = y_arr[mask]
    if len(f) < 2:
        return 0.0
    tdays = to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, tdays))

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y)
    cum = 0.0
    for i in range(len(y)):
        if bool(active_mask[i]):
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val
            cum += val
        else:
            out[i] = 0.0
    return out

# ===============================
# Curva pérdida de rinde (%)
# ===============================
def loss_pct_from_x(x):
    # Modelo usado en tu app anterior
    x = float(x)
    return 0.375 * x / (1.0 + (0.375 * x / 76.639))

# ===============================
# Curva dosis → eficiencia (Emax)
# ===============================
def emax_eff(dose, Emax, ED50, h):
    try:
        dose = float(dose)
    except:
        return 0.0
    if not np.isfinite(dose) or dose <= 0:
        return 0.0
    return float(Emax * (dose**h) / (ED50**h + dose**h))

# ===============================
# Sidebar — Cargar datos
# ===============================
st.sidebar.header("Datos de entrada")

st.sidebar.subheader("1) Emergencia (EMERREL o EMERAC)")
up_emer = st.sidebar.file_uploader("CSV de emergencia (fecha, valor)", type=["csv"], key="emer_csv")

emer_as_percent = st.sidebar.checkbox("Valores en % (no 0–1)", value=True)
emer_is_cum = st.sidebar.checkbox("Es acumulado (EMERAC)", value=False)
emer_dayfirst = st.sidebar.checkbox("Fecha en formato dd/mm/yyyy", value=True)

st.sidebar.subheader("2) Observaciones independientes")
up_obs = st.sidebar.file_uploader("CSV de observaciones", type=["csv"], key="obs_csv")

if up_emer is None or up_obs is None:
    st.info(
        "Cargá ambos CSV para continuar.\n\n"
        "• **Emergencia**: columnas `fecha, valor`. Si es EMERAC (acumulada), tildá la opción.\n"
        "• **Observaciones**: columnas mínimas `trial_id, sow_date, herb_type, app_date, dose, observed_loss_pct`. "
        "Opcionales: `residual_days, plants_count, eff_override`."
    )
    st.stop()

# ===============================
# Procesar EMERREL/EMERAC
# ===============================
try:
    df0 = read_csv_bytes(up_emer.read())
    df0.columns = [c.strip() for c in df0.columns]
    c_fecha = df0.columns[0]
    c_valor = df0.columns[1] if len(df0.columns) > 1 else df0.columns[0]
    fechas = pd.to_datetime(df0[c_fecha], dayfirst=emer_dayfirst, errors="coerce")
    sample_str = df0[c_valor].astype(str).head(200).str.cat(sep=" ")
    dec_for_col = "," if (sample_str.count(",")>sample_str.count(".") and re.search(r",\d", sample_str)) else "."
    vals = clean_numeric_series(df0[c_valor], decimal=dec_for_col)
    df_emer = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
    if df_emer.empty:
        st.error("El CSV de emergencia quedó vacío tras el parseo.")
        st.stop()
    emerg = df_emer["valor"].astype(float)
    if emer_as_percent:
        emerg = emerg / 100.0
    if emer_is_cum:
        emerg = emerg.diff().fillna(0.0).clip(lower=0.0)
    emerg = emerg.clip(lower=0.0)
    df_emer_plot = pd.DataFrame({"fecha": pd.to_datetime(df_emer["fecha"]), "EMERREL": emerg})
except Exception as e:
    st.error(f"No se pudo leer/parsear emergencia: {e}")
    st.stop()

st.success("Emergencia cargada correctamente.")

# ===============================
# Procesar Observaciones
# ===============================
try:
    df_obs = read_csv_bytes(up_obs.read())
    df_obs.columns = [c.strip().lower() for c in df_obs.columns]
except Exception as e:
    st.error(f"No se pudo leer el CSV de observaciones: {e}")
    st.stop()

req_cols = ["trial_id","sow_date","herb_type","app_date","dose","observed_loss_pct"]
missing = [c for c in req_cols if c not in df_obs.columns]
if missing:
    st.error(f"Faltan columnas obligatorias en observaciones: {missing}")
    st.stop()

df_obs["trial_id"] = df_obs["trial_id"].astype(str)
df_obs["sow_date"] = pd.to_datetime(df_obs["sow_date"], errors="coerce").dt.date
df_obs["app_date"] = pd.to_datetime(df_obs["app_date"], errors="coerce").dt.date
df_obs["herb_type"] = df_obs["herb_type"].astype(str).str.strip().str.lower()
df_obs["dose"] = pd.to_numeric(df_obs["dose"], errors="coerce")
df_obs["observed_loss_pct"] = pd.to_numeric(df_obs["observed_loss_pct"], errors="coerce")

if "residual_days" in df_obs.columns:
    df_obs["residual_days"] = pd.to_numeric(df_obs["residual_days"], errors="coerce")
else:
    df_obs["residual_days"] = np.nan

if "plants_count" in df_obs.columns:
    df_obs["plants_count"] = pd.to_numeric(df_obs["plants_count"], errors="coerce")
else:
    df_obs["plants_count"] = np.nan

if "eff_override" in df_obs.columns:
    df_obs["eff_override"] = pd.to_numeric(df_obs["eff_override"], errors="coerce")
else:
    df_obs["eff_override"] = np.nan

valid_types = {"presiembra","preemergente","post"}
bad = ~df_obs["herb_type"].isin(valid_types)
if bad.any():
    st.error("`herb_type` solo puede ser 'presiembra', 'preemergente' o 'post'.")
    st.stop()

st.success("Observaciones cargadas correctamente.")

# ===============================
# Cohortes S1..S4 (definición)
# ===============================
def build_cohorts(ts: pd.Series, emerrel: np.ndarray, sow_d: date):
    ts = pd.to_datetime(ts)
    mask_since = (ts.dt.date >= sow_d).to_numpy()
    births = np.where(mask_since, emerrel, 0.0)
    s = pd.Series(births, index=ts)

    # Ventanas (idénticas a tu app base)
    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).reindex(ts).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).reindex(ts).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).reindex(ts).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).reindex(ts).to_numpy(float)

    # Factores de cobertura por estado (FC_S)
    FC = {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0}

    return ts, mask_since, S1, S2, S3, S4, FC

# ===============================
# Simulador de una observación → pérdida %
# Reglas:
# - Presiembra residual: solo S1–S2, permitida ≤ siembra−14 (si cae en (-14, 0), se ignora)
# - Preemergente residual: solo S1–S2, permitida [siembra−1, siembra+10]
# - Post graminicida: S1–S3, ventana fuerza [0, +10] desde siembra
# ===============================
def simula_observacion_loss_pct(df_emer_plot, sow_date, preR=None, preEM=None, postG=None):
    ts = pd.to_datetime(df_emer_plot["fecha"])
    emerrel = df_emer_plot["EMERREL"].to_numpy(float)

    # AUC cruda desde siembra → factor área (pl·m² / EMERREL·día)
    mask_since = (ts.dt.date >= sow_date).to_numpy()
    auc_cruda = auc_time(ts, emerrel, mask=mask_since)
    if auc_cruda <= 0:
        return np.nan
    factor_area = MAX_PLANTS_CAP / auc_cruda

    # Cohortes
    ts_all, mask_since_all, S1, S2, S3, S4, FC = build_cohorts(ts, emerrel, sow_date)

    # Aportes sin control (pl·m²·día⁻¹)
    S1_pl = np.where(mask_since_all, S1 * FC["S1"] * factor_area, 0.0)
    S2_pl = np.where(mask_since_all, S2 * FC["S2"] * factor_area, 0.0)
    S3_pl = np.where(mask_since_all, S3 * FC["S3"] * factor_area, 0.0)
    S4_pl = np.where(mask_since_all, S4 * FC["S4"] * factor_area, 0.0)

    base_pl_daily = np.where(mask_since_all, emerrel * factor_area, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_all)
    sup_cap = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_pl_daily_cap)

    # Controles por estado
    c1 = np.ones_like(mask_since_all, float)
    c2 = np.ones_like(mask_since_all, float)
    c3 = np.ones_like(mask_since_all, float)
    c4 = np.ones_like(mask_since_all, float)

    fechas_d = ts.dt.date.values

    def apply_window_reduction(date0: date, days: int, eff_pct: float, states: list):
        if date0 is None or days is None or eff_pct <= 0:
            return
        d0 = date0
        d1 = date0 + timedelta(days=int(days))
        w = ((fechas_d >= d0) & (fechas_d < d1)).astype(float)
        reduc = np.clip(1.0 - (eff_pct/100.0)*w, 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    sow = pd.to_datetime(sow_date)

    # ---- Presiembra residual (solo S1–S2), permitida ≤ siembra−14
    if preR is not None:
        d = pd.to_datetime(preR.get("date")).date() if preR.get("date") else None
        if d is not None:
            if d <= (sow - pd.Timedelta(days=14)).date():
                R = int(preR.get("days") or NR_DAYS_DEFAULT)
                apply_window_reduction(d, R, float(preR.get("eff", 0.0)), ["S1","S2"])
            # else: cae en (-14, 0) ⇒ se ignora

    # ---- Preemergente residual (solo S1–S2), permitida entre [siembra−1, siembra+10]
    if preEM is not None:
        d = pd.to_datetime(preEM.get("date")).date() if preEM.get("date") else None
        if d is not None:
            earliest = (sow - pd.Timedelta(days=1)).date()
            latest   = (sow + pd.Timedelta(days=10)).date()
            if earliest <= d <= latest:
                R = int(preEM.get("days") or NR_DAYS_DEFAULT)
                apply_window_reduction(d, R, float(preEM.get("eff", 0.0)), ["S1","S2"])

    # ---- Post (graminicida) (S1–S3), forzamos ventana [sow, sow+10]
    if postG is not None:
        d = pd.to_datetime(postG.get("date")).date() if postG.get("date") else None
        if d is not None:
            start = max(d, sow.date())
            end   = (sow + pd.Timedelta(days=10)).date()
            if start <= end:
                apply_window_reduction(start, (end - start).days + 1, float(postG.get("eff", 0.0)), ["S1","S2","S3"])

    # Con control + cap + reescalado proporcional por estado
    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    eps = 1e-12
    scale = np.where(tot_ctrl > eps, np.minimum(1.0, sup_cap / tot_ctrl), 0.0)
    plantas_ctrl_cap = (S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4) * scale

    X3 = float(np.nansum(plantas_ctrl_cap[mask_since_all]))
    return float(loss_pct_from_x(X3))

# ===============================
# Dosis → Eficiencia (parámetros por tipo)
# ===============================
st.sidebar.header("Dosis → Eficiencia (Emax)")
tune_emax = st.sidebar.checkbox("Calibrar Emax/ED50", value=False)

st.sidebar.markdown("**Presiembra (residual; S1–S2)**")
preR_Emax = st.sidebar.slider("Emax_presiembra (%)", 40, 100, 80, 1)
preR_ED50 = st.sidebar.number_input("ED50_presiembra (unid. dosis)", 0.1, 500.0, 50.0, 1.0)
preR_h    = st.sidebar.number_input("h_presiembra", 0.5, 4.0, 1.0, 0.1)

st.sidebar.markdown("**Preemergente (residual; S1–S2)**")
preEM_Emax = st.sidebar.slider("Emax_preemergente (%)", 40, 100, 85, 1)
preEM_ED50 = st.sidebar.number_input("ED50_preemergente (unid. dosis)", 0.1, 500.0, 40.0, 1.0)
preEM_h    = st.sidebar.number_input("h_preemergente", 0.5, 4.0, 1.0, 0.1)

st.sidebar.markdown("**Post (graminicida; S1–S3)**")
postG_Emax = st.sidebar.slider("Emax_post (%)", 40, 100, 75, 1)
postG_ED50 = st.sidebar.number_input("ED50_post (unid. dosis)", 0.1, 500.0, 60.0, 1.0)
postG_h    = st.sidebar.number_input("h_post", 0.5, 4.0, 1.0, 0.1)

# ===============================
# Agrupar observaciones por trial_id
# (varias filas → varios tratamientos de la misma parcela)
# ===============================
def group_observations(df):
    grouped = []
    for tid, g in df.groupby("trial_id", sort=False):
        sow = g["sow_date"].dropna().iloc[0]
        loss_obs = g["observed_loss_pct"].dropna().iloc[0] if g["observed_loss_pct"].notna().any() else np.nan
        plants_obs = g["plants_count"].dropna().iloc[0] if g["plants_count"].notna().any() else np.nan
        trts = []
        for _, r in g.iterrows():
            t = r["herb_type"]
            eff = r.get("eff_override", np.nan)
            if not np.isfinite(eff):
                if t == "presiembra":
                    eff = emax_eff(r["dose"], preR_Emax, preR_ED50, preR_h)
                elif t == "preemergente":
                    eff = emax_eff(r["dose"], preEM_Emax, preEM_ED50, preEM_h)
                else:
                    eff = emax_eff(r["dose"], postG_Emax, postG_ED50, postG_h)
            trts.append({
                "herb_type": t,
                "app_date": r["app_date"],
                "residual_days": int(r["residual_days"]) if np.isfinite(r["residual_days"]) else None,
                "eff": float(np.clip(eff, 0.0, 100.0)),
                "dose": float(r["dose"]) if np.isfinite(r["dose"]) else np.nan,
                "eff_override": r.get("eff_override", np.nan)
            })
        grouped.append({
            "trial_id": tid,
            "sow_date": sow,
            "observed_loss_pct": float(loss_obs) if np.isfinite(loss_obs) else np.nan,
            "plants_count": float(plants_obs) if np.isfinite(plants_obs) else np.nan,
            "treatments": trts
        })
    return grouped

obs_list = group_observations(df_obs)

st.subheader("Muestras agrupadas (primeras 5)")
st.json(obs_list[:5])

# ===============================
# Evaluación por observación
# ===============================
def eval_obs_loss_pct(item):
    sow = item["sow_date"]
    preR = None; preEM = None; postG = None
    for t in item["treatments"]:
        if t["herb_type"] == "presiembra":
            preR = {"date": t["app_date"], "days": (t["residual_days"] or NR_DAYS_DEFAULT), "eff": t["eff"]}
        elif t["herb_type"] == "preemergente":
            preEM = {"date": t["app_date"], "days": (t["residual_days"] or NR_DAYS_DEFAULT), "eff": t["eff"]}
        else:
            postG = {"date": t["app_date"], "eff": t["eff"]}
    yhat = simula_observacion_loss_pct(df_emer_plot, sow, preR=preR, preEM=preEM, postG=postG)
    return yhat

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any(): return np.inf
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

# ===============================
# Calibración (opcional) Emax/ED50 por tipo
# ===============================
st.sidebar.header("Calibración")
optimizer = st.sidebar.selectbox("Optimizador", ["Grid simple", "Búsqueda aleatoria"], index=1)
if optimizer == "Grid simple":
    st.sidebar.caption("Espacios (coarse) si la calibración está activada")
    grid_Emax = st.sidebar.multiselect("Emax candidatos (%)", [60,70,80,85,90,95,100], default=[70,80,90])
    grid_ED50 = st.sidebar.multiselect("ED50 candidatos", [10,20,30,40,50,60,80,100], default=[30,50,80])
    max_grid = st.sidebar.number_input("Máx. combinaciones", 100, 50000, 3000, 100)
else:
    n_rand = st.sidebar.number_input("Evaluaciones aleatorias", 100, 100000, 3000, 100)
    rand_Emax_min = st.sidebar.slider("Emax mín (%)", 50, 100, 60, 1)
    rand_Emax_max = st.sidebar.slider("Emax máx (%)", 60, 100, 95, 1)
    rand_ED50_min = st.sidebar.number_input("ED50 mín", 1.0, 200.0, 10.0, 1.0)
    rand_ED50_max = st.sidebar.number_input("ED50 máx", 5.0, 500.0, 120.0, 1.0)

def set_params(eR, dR, eE, dE, eP, dP):
    return dict(preR_Emax=eR, preR_ED50=dR, preEM_Emax=eE, preEM_ED50=dE, postG_Emax=eP, postG_ED50=dP)

def apply_param_set(params):
    preds, obs = [], []
    for it in obs_list:
        # re-evaluamos eficiencias SÓLO si originalmente venían por dosis (sin override)
        trts_new = []
        for t in it["treatments"]:
            eff = t["eff"]
            if not np.isfinite(t.get("eff_override", np.nan)):
                if t["herb_type"] == "presiembra":
                    eff = emax_eff(t.get("dose", np.nan), params["preR_Emax"], params["preR_ED50"], preR_h)
                elif t["herb_type"] == "preemergente":
                    eff = emax_eff(t.get("dose", np.nan), params["preEM_Emax"], params["preEM_ED50"], preEM_h)
                else:
                    eff = emax_eff(t.get("dose", np.nan), params["postG_Emax"], params["postG_ED50"], postG_h)
            trts_new.append({
                "herb_type": t["herb_type"],
                "app_date": t["app_date"],
                "residual_days": t["residual_days"],
                "eff": float(np.clip(eff, 0.0, 100.0)),
                "dose": t.get("dose", np.nan),
                "eff_override": t.get("eff_override", np.nan)
            })
        it2 = {**it, "treatments": trts_new}
        yhat = eval_obs_loss_pct(it2)
        preds.append(yhat)
        obs.append(it["observed_loss_pct"])
    return rmse(obs, preds), np.array(obs, float), np.array(preds, float)

st.subheader("Ejecutar evaluación / calibración")
best = None

if st.button("Calcular / Calibrar", use_container_width=True):
    if not tune_emax:
        rm, yo, yp = apply_param_set(set_params(preR_Emax, preR_ED50, preEM_Emax, preEM_ED50, postG_Emax, postG_ED50))
        best = {"rmse": rm, "obs": yo, "pred": yp,
                "preR_Emax": preR_Emax, "preR_ED50": preR_ED50,
                "preEM_Emax": preEM_Emax, "preEM_ED50": preEM_ED50,
                "postG_Emax": postG_Emax, "postG_ED50": postG_ED50}
    else:
        if optimizer == "Grid simple":
            combos = []
            for a in grid_Emax:
                for b in grid_ED50:
                    for c in grid_Emax:
                        for d in grid_ED50:
                            for e in grid_Emax:
                                for f in grid_ED50:
                                    combos.append(set_params(a,b,c,d,e,f))
            if len(combos) > max_grid:
                random.shuffle(combos)
                combos = combos[:int(max_grid)]

            progress = st.progress(0.0)
            for i, p in enumerate(combos, 1):
                rm, yo, yp = apply_param_set(p)
                if (best is None) or (rm < best["rmse"]):
                    best = {"rmse": rm, "obs": yo, "pred": yp, **p}
                if i % max(1, len(combos)//100) == 0 or i == len(combos):
                    progress.progress(i/len(combos))
            progress.empty()
        else:
            progress = st.progress(0.0)
            for i in range(1, int(n_rand)+1):
                p = set_params(
                    random.uniform(rand_Emax_min, rand_Emax_max),
                    random.uniform(rand_ED50_min, rand_ED50_max),
                    random.uniform(rand_Emax_min, rand_Emax_max),
                    random.uniform(rand_ED50_min, rand_ED50_max),
                    random.uniform(rand_Emax_min, rand_Emax_max),
                    random.uniform(rand_ED50_min, rand_ED50_max),
                )
                rm, yo, yp = apply_param_set(p)
                if (best is None) or (rm < best["rmse"]):
                    best = {"rmse": rm, "obs": yo, "pred": yp, **p}
                if i % max(1, int(n_rand)//100) == 0 or i == int(n_rand):
                    progress.progress(i/int(n_rand))
            progress.empty()

# ===============================
# Reporte
# ===============================
if best is not None:
    st.success(f"RMSE pérdida de rinde (%): **{best['rmse']:.3f}**")
    if tune_emax:
        st.markdown(
            f"**Parámetros óptimos**\n\n"
            f"- Presiembra: Emax={best['preR_Emax']:.1f}% · ED50={best['preR_ED50']:.2f}\n"
            f"- Preemergente: Emax={best['preEM_Emax']:.1f}% · ED50={best['preEM_ED50']:.2f}\n"
            f"- Post: Emax={best['postG_Emax']:.1f}% · ED50={best['postG_ED50']:.2f}"
        )

    # Tabla Obs vs Pred por trial_id
    rows = []
    for it, yobs, ypred in zip(obs_list, best["obs"], best["pred"]):
        rows.append({
            "trial_id": it["trial_id"],
            "sow_date": it["sow_date"],
            "obs_loss_pct": yobs,
            "pred_loss_pct": ypred,
            "plants_count": it.get("plants_count", np.nan)
        })
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True)
    st.download_button("Descargar resultados (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                       "obs_vs_pred_loss_pct.csv", "text/csv")

    # Gráfico Obs vs Pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best["obs"], y=best["pred"], mode="markers", name="Obs vs Pred",
                             hovertemplate="Obs: %{x:.2f}%<br>Pred: %{y:.2f}%<extra></extra>"))
    mmax = float(np.nanmax([np.nanmax(best["obs"]), np.nanmax(best["pred"]), 1.0]))
    xy = np.linspace(0, 1.05*mmax, 50)
    fig.add_trace(go.Scatter(x=xy, y=xy, mode="lines", name="Identidad", line=dict(dash="dash")))
    fig.update_layout(title="Pérdida de rinde (%) — Observado vs Predicho",
                      xaxis_title="Observado (%)", yaxis_title="Predicho (%)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)











