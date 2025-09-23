# -*- coding: utf-8 -*-
# app_calibracion.py — Calibración de PREDWEEM con datos independientes
# - Admite objetivos: x (pl·m²), A2 (pl·m²), o serie semanal (pl·m²·sem⁻¹)
# - Opcional: cronograma de manejo (JSON) en formato del “mejor escenario”
# - Optimizadores: Búsqueda aleatoria / Recocido simulado
# - Salidas: métricas, gráficos y JSON de parámetros calibrados

import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
import random, math as _math

# ============================== UI base ==============================
st.set_page_config(page_title="PREDWEEM · Calibración", layout="wide")
st.title("PREDWEEM · Calibración con datos independientes")

# ============================== Helpers I/O ==============================
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample): dec_guess = ","
    return sep_guess, dec_guess

def parse_csv_file(uploaded, sep_opt="auto", dec_opt="auto", dayfirst=True):
    raw = uploaded.read()
    head = raw[:8000].decode("utf-8", errors="ignore")
    sep_g, dec_g = sniff_sep_dec(head)
    sep = sep_g if sep_opt=="auto" else ("," if sep_opt=="," else (";" if sep_opt==";" else "\t"))
    dec = dec_g if dec_opt=="auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python")
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=dayfirst, errors="coerce")
    return df

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

# ============================== Sidebar: datos & opciones ==============================
with st.sidebar:
    st.header("Datos de entrada")
    up_emer = st.file_uploader("EMERREL (CSV: fecha, valor)", type=["csv"])
    up_obs  = st.file_uploader("Observaciones (CSV)", type=["csv"])
    st.caption("Observaciones admitidas:\n- Serie semanal: columnas `fecha`,`obs_sem` (pl·m²·sem⁻¹)\n- x total: una fila con `x_obs`\n- A2 total: una fila con `A2_obs`")
    up_sched = st.file_uploader("Cronograma (JSON, opcional)", type=["json"])

    sep_opt = st.selectbox("Delimitador CSV", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal CSV", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha dd/mm/aaaa", value=True)

    st.header("Escenario y tope")
    MAX_PLANTS_CAP = float(st.selectbox("Tope A2 (pl·m²)", [250,125,62], index=0))

    st.header("Siembra")
    default_year = dt.date.today().year
    sow_date = st.date_input("Fecha de siembra", value=dt.date(default_year,5,1))

    st.header("Objetivo de calibración")
    target_kind = st.selectbox("Variable a ajustar", ["Serie semanal", "x total", "A2 total"], index=0)
    loss_fn = st.selectbox("Función de error", ["RMSE","MAE","Huber"], index=0)

if up_emer is None or up_obs is None:
    st.info("Cargá EMERREL y Observaciones para continuar.")
    st.stop()

# ============================== Carga EMERREL ==============================
df_e = parse_csv_file(up_emer, sep_opt, dec_opt, dayfirst)
if df_e.empty:
    st.error("El CSV de EMERREL está vacío."); st.stop()

cols_e = list(df_e.columns)
c_f = st.selectbox("Columna fecha (EMERREL)", cols_e, index=0)
c_v = st.selectbox("Columna valor (EMERREL)", cols_e, index=(1 if len(cols_e)>1 else 0))
ts = pd.to_datetime(df_e[c_f], dayfirst=dayfirst, errors="coerce")
val = clean_numeric_series(df_e[c_v], decimal="," if dec_opt=="," else ".").astype(float)
df_emer = pd.DataFrame({"fecha": ts, "EMERREL": val}).dropna().sort_values("fecha").reset_index(drop=True)
if df_emer.empty:
    st.error("Tras el parseo no quedaron filas válidas para EMERREL."); st.stop()

# ============================== Carga Observaciones ==============================
df_obs = parse_csv_file(up_obs, sep_opt, dec_opt, dayfirst)
if df_obs.empty:
    st.error("El CSV de observaciones está vacío."); st.stop()

x_obs = None; A2_obs = None
if target_kind == "Serie semanal":
    if not {"fecha","obs_sem"}.issubset(df_obs.columns):
        st.error("La observación semanal requiere columnas: fecha, obs_sem"); st.stop()
    df_obs["fecha"] = pd.to_datetime(df_obs["fecha"], dayfirst=dayfirst, errors="coerce")
    df_obs = df_obs.dropna().sort_values("fecha").reset_index(drop=True)
elif target_kind == "x total":
    if "x_obs" not in df_obs.columns:
        st.error("x total requiere columna: x_obs"); st.stop()
    x_obs = float(df_obs["x_obs"].iloc[0])
else:
    if "A2_obs" not in df_obs.columns:
        st.error("A2 total requiere columna: A2_obs"); st.stop()
    A2_obs = float(df_obs["A2_obs"].iloc[0])

# ============================== Cronograma (opcional) ==============================
schedule = []
if up_sched is not None:
    try:
        raw_json = up_sched.read().decode("utf-8")
        schedule = json.loads(raw_json)
        if not isinstance(schedule, list): schedule = []
        # normalizar fechas a date
        for a in schedule:
            if "date" in a:
                a["date"] = pd.to_datetime(a["date"]).date()
    except Exception as e:
        st.warning(f"No se pudo leer el JSON del cronograma: {e}")
        schedule = []

# ============================== Parámetros a calibrar ==============================
with st.sidebar:
    st.header("Parámetros a calibrar")
    calib_k_beer   = st.checkbox("Calibrar k (Beer–Lambert)", True)
    calib_lai_max  = st.checkbox("Calibrar LAI máximo", True)
    calib_t_lag    = st.checkbox("Calibrar días a emergencia cultivo (lag)", True)
    calib_t_close  = st.checkbox("Calibrar días a cierre entresurco", True)
    calib_LAIhc    = st.checkbox("Calibrar LAIhc (Ciec)", True)
    calib_eff      = st.checkbox("Calibrar eficiencias del cronograma (multiplicador 0.5–1.2)", value=(len(schedule)>0))

    st.header("Cotas (mín–máx)")
    k_min,k_max        = st.number_input("k min",0.1,1.5,0.3,0.05), st.number_input("k max",0.1,1.5,1.0,0.05)
    lai_min,lai_max    = st.number_input("LAI max min",0.1,10.0,2.0,0.1), st.number_input("LAI max max",0.1,10.0,6.0,0.1)
    lag_min,lag_max    = st.number_input("lag min (d)",0,60,3,1), st.number_input("lag max (d)",0,60,12,1)
    close_min,close_max= st.number_input("cierre min (d)",10,120,35,1), st.number_input("cierre max (d)",10,120,80,1)
    hc_min,hc_max      = st.number_input("LAIhc min",0.5,10.0,2.0,0.1), st.number_input("LAIhc max",0.5,10.0,6.0,0.1)

# ============================== Series base para simular ==============================
ts = pd.to_datetime(df_emer["fecha"])
mask_since_sow = (ts.dt.date >= sow_date)
emerrel = df_emer["EMERREL"].astype(float).clip(lower=0.0).to_numpy()

def compute_canopy(fechas: pd.Series, sow_date: dt.date, t_lag: int, t_close: int, lai_max: float, k_beer: float):
    days = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas], float)
    def logistic_between(days, start, end, y_max):
        if end <= start: end = start + 1
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))
    LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
    LAI = np.clip(LAI,0.0,lai_max)
    fc_dyn = 1 - np.exp(-k_beer*LAI)
    return fc_dyn, LAI

def simulate(params, schedule_in=None, eff_mult=1.0):
    # params: dict con t_lag,t_close,lai_max,k_beer,LAIhc
    t_lag = int(params["t_lag"]); t_close=int(params["t_close"])
    lai_m = float(params["lai_max"]); k_b=float(params["k_beer"]); LAIhc=float(params["LAIhc"])

    FC, LAI = compute_canopy(ts, sow_date, t_lag, t_close, lai_m, k_b)
    Ca, Cs = 250.0, 250.0  # densidades estándar para Ciec
    Ciec = np.clip((LAI / max(1e-6, LAIhc)) * (Ca / Cs), 0.0, 1.0)
    one_minus = np.clip(1.0 - Ciec, 0.0, 1.0)

    births = np.where(mask_since_sow.to_numpy(), emerrel, 0.0)
    s = pd.Series(births, index=ts)
    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).to_numpy(float)

    auc_cruda = auc_time(ts, emerrel, mask=mask_since_sow)
    if auc_cruda <= 0: return None
    factor_area = MAX_PLANTS_CAP / auc_cruda

    # coef FC por estado (como app principal)
    S1_pl = np.where(mask_since_sow, S1 * one_minus * 0.0 * factor_area, 0.0)
    S2_pl = np.where(mask_since_sow, S2 * one_minus * 0.3 * factor_area, 0.0)
    S3_pl = np.where(mask_since_sow, S3 * one_minus * 0.6 * factor_area, 0.0)
    S4_pl = np.where(mask_since_sow, S4 * one_minus * 1.0 * factor_area, 0.0)

    base_pl_daily = np.where(mask_since_sow.to_numpy(), emerrel * factor_area, 0.0)
    base_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since_sow.to_numpy())
    sup_cap = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_cap)

    # manejo
    ctrl = np.ones((4, len(ts)), float)
    if schedule_in:
        fechas_d = ts.dt.date.values
        def weights(d0, days):
            d0 = pd.to_datetime(d0).date() if not isinstance(d0, dt.date) else d0
            d1 = pd.to_datetime(d0) + pd.Timedelta(days=int(days))
            w = ((fechas_d >= d0) & (fechas_d < d1.date())).astype(float)
            return w
        for a in schedule_in:
            eff = float(a.get("eff",0))*eff_mult
            w = weights(a["date"], a["days"])
            reduc = np.clip(1.0 - (eff/100.0)*w, 0.0, 1.0)
            stset = set(a.get("states",["S1","S2","S3","S4"]))
            if "S1" in stset: ctrl[0] *= reduc
            if "S2" in stset: ctrl[1] *= reduc
            if "S3" in stset: ctrl[2] *= reduc
            if "S4" in stset: ctrl[3] *= reduc

    S1c = S1_pl*ctrl[0]; S2c=S2_pl*ctrl[1]; S3c=S3_pl*ctrl[2]; S4c=S4_pl*ctrl[3]
    total_ctrl = S1c+S2c+S3c+S4c
    total_ctrl_cap = np.minimum(total_ctrl, sup_cap)

    # outputs
    weekly = (pd.DataFrame({"fecha": ts, "val": total_ctrl_cap})
                .set_index("fecha").resample("W-MON").sum().reset_index())
    X3 = float(np.nansum(total_ctrl_cap[mask_since_sow]))
    # A2 por AUC
    sup_equiv  = np.divide(sup_cap,        factor_area, out=np.zeros_like(sup_cap),        where=(factor_area>0))
    ctrl_equiv = np.divide(total_ctrl_cap, factor_area, out=np.zeros_like(total_ctrl_cap), where=(factor_area>0))
    auc_sup    = auc_time(ts, sup_equiv,  mask=mask_since_sow)
    auc_ctrl   = auc_time(ts, ctrl_equiv, mask=mask_since_sow)
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_ctrl/auc_cruda))
    return {"weekly": weekly, "x3": X3, "A2_ctrl": A2_ctrl}

# ============================== Pérdida / Costo ==============================
def huber(res, delta=1.0):
    a = np.abs(res)
    return np.where(a<=delta, 0.5*a*a, delta*(a-0.5*delta))

def score(sim, df_obs, kind, loss="RMSE"):
    if sim is None: return np.inf
    if kind == "Serie semanal":
        dfm = pd.merge(df_obs[["fecha","obs_sem"]], sim["weekly"], on="fecha", how="inner")
        if len(dfm)==0: return np.inf
        err = dfm["val"].to_numpy(float) - dfm["obs_sem"].to_numpy(float)
    elif kind == "x total":
        err = np.array([sim["x3"] - float(x_obs)])
    else:
        err = np.array([sim["A2_ctrl"] - float(A2_obs)])
    if loss == "RMSE":
        return float(np.sqrt(np.mean(err**2)))
    elif loss == "MAE":
        return float(np.mean(np.abs(err)))
    else:
        return float(np.mean(huber(err)))

# ============================== Muestreo de parámetros ==============================
def random_params():
    p = {}
    if calib_t_lag:   p["t_lag"]   = random.randint(lag_min, lag_max)
    else:             p["t_lag"]   = int((lag_min+lag_max)//2)
    if calib_t_close: p["t_close"] = random.randint(close_min, close_max)
    else:             p["t_close"] = int((close_min+close_max)//2)
    if calib_lai_max: p["lai_max"] = random.uniform(lai_min, lai_max)
    else:             p["lai_max"] = float((lai_min+lai_max)/2)
    if calib_k_beer:  p["k_beer"]  = random.uniform(k_min, k_max)
    else:             p["k_beer"]  = float((k_min+k_max)/2)
    if calib_LAIhc:   p["LAIhc"]   = random.uniform(hc_min, hc_max)
    else:             p["LAIhc"]   = float((hc_min+hc_max)/2)
    return p

# ============================== Optimizadores ==============================
with st.sidebar:
    st.header("Optimización")
    opt_kind = st.selectbox("Optimizador", ["Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals = st.number_input("Máx. evaluaciones", 100, 50000, 3000, 100)
    sa_iters  = st.number_input("Iteraciones SA (si aplica)", 100, 50000, 3000, 100)
    sa_T0     = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
    sa_cool   = st.number_input("Enfriamiento γ", 0.80, 0.9999, 0.995, 0.0001)

start = st.button("▶️ Calibrar", use_container_width=True)

best = None
if start:
    st.write("Calibrando…")
    prog = st.progress(0.0)
    if opt_kind == "Búsqueda aleatoria":
        best = {"score": np.inf}
        for i in range(1, int(max_evals)+1):
            p = random_params()
            eff_mult = random.uniform(0.5,1.2) if calib_eff else 1.0
            sim = simulate(p, schedule_in=schedule, eff_mult=eff_mult)
            sc = score(sim, df_obs, target_kind, loss_fn)
            if sc < best["score"]:
                best = {"params":p, "eff_mult":eff_mult, "sim":sim, "score":sc, "iter":i}
            if i % max(1,int(max_evals)//100) == 0: prog.progress(i/float(max_evals))
    else:
        cur_p = random_params()
        cur_eff = random.uniform(0.5,1.2) if calib_eff else 1.0
        cur_sim = simulate(cur_p, schedule_in=schedule, eff_mult=cur_eff)
        cur_sc = score(cur_sim, df_obs, target_kind, loss_fn)
        best = {"params":cur_p.copy(), "eff_mult":cur_eff, "sim":cur_sim, "score":cur_sc, "iter":0}
        T = float(sa_T0)
        for it in range(1, int(sa_iters)+1):
            cand = cur_p.copy()
            if calib_t_lag:   cand["t_lag"]   = int(np.clip(cand["t_lag"]   + random.randint(-2,2), lag_min, lag_max))
            if calib_t_close: cand["t_close"] = int(np.clip(cand["t_close"] + random.randint(-3,3), close_min, close_max))
            if calib_lai_max: cand["lai_max"] = float(np.clip(cand["lai_max"] + random.uniform(-0.2,0.2), lai_min, lai_max))
            if calib_k_beer:  cand["k_beer"]  = float(np.clip(cand["k_beer"] + random.uniform(-0.05,0.05), k_min, k_max))
            if calib_LAIhc:   cand["LAIhc"]   = float(np.clip(cand["LAIhc"] + random.uniform(-0.2,0.2), hc_min, hc_max))
            cand_eff = float(np.clip(cur_eff + (random.uniform(-0.05,0.05) if calib_eff else 0.0), 0.5, 1.2))
            sim = simulate(cand, schedule_in=schedule, eff_mult=cand_eff)
            sc = score(sim, df_obs, target_kind, loss_fn)
            d = sc - cur_sc
            if d <= 0 or random.random() < _math.exp(-d / max(1e-9, T)):
                cur_p, cur_eff, cur_sim, cur_sc = cand, cand_eff, sim, sc
                if sc < best["score"]:
                    best = {"params":cand.copy(), "eff_mult":cand_eff, "sim":sim, "score":sc, "iter":it}
            T *= float(sa_cool)
            if it % max(1,int(sa_iters)//100) == 0: prog.progress(it/float(sa_iters))
    prog.progress(1.0)

# ============================== Resultados & Gráficos ==============================
if best is not None and "sim" in best and best["sim"] is not None:
    st.success(f"Listo. Mejor puntaje ({loss_fn}): **{best['score']:.4f}** en iteración {best['iter']}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parámetros calibrados")
        st.json({
            "t_lag": int(best["params"]["t_lag"]),
            "t_close": int(best["params"]["t_close"]),
            "lai_max": round(best["params"]["lai_max"],3),
            "k_beer": round(best["params"]["k_beer"],3),
            "LAIhc": round(best["params"]["LAIhc"],3),
            "eff_multiplier": round(float(best.get("eff_mult",1.0)),3)
        })
    with col2:
        st.download_button(
            "Descargar parámetros calibrados (JSON)",
            data=json.dumps({
                "t_lag": int(best["params"]["t_lag"]),
                "t_close": int(best["params"]["t_close"]),
                "lai_max": float(best["params"]["lai_max"]),
                "k_beer": float(best["params"]["k_beer"]),
                "LAIhc": float(best["params"]["LAIhc"]),
                "eff_multiplier": float(best.get("eff_mult",1.0))
            }, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="parametros_calibrados.json",
            mime="application/json"
        )

    sim = best["sim"]

    if target_kind == "Serie semanal":
        st.subheader("Serie semanal — observado vs simulado")
        df_plot = pd.merge(df_obs[["fecha","obs_sem"]], sim["weekly"], on="fecha", how="outer").sort_values("fecha")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["obs_sem"], mode="lines+markers", name="Obs (pl·m²·sem⁻¹)"))
        fig.add_trace(go.Scatter(x=df_plot["fecha"], y=df_plot["val"],     mode="lines+markers", name="Sim (pl·m²·sem⁻¹)"))
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Fecha (Lunes)", yaxis_title="pl·m²·sem⁻¹")
        st.plotly_chart(fig, use_container_width=True)

        # Paridad
        dfm = pd.merge(df_obs[["fecha","obs_sem"]], sim["weekly"], on="fecha", how="inner")
        st.subheader("Predicho vs Observado (paridad)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dfm["obs_sem"], y=dfm["val"], mode="markers", name="Puntos"))
        lim = float(max(1e-6, dfm[["obs_sem","val"]].to_numpy().max()))
        fig2.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode="lines", name="1:1", line=dict(dash="dash")))
        fig2.update_layout(xaxis_title="Observado", yaxis_title="Simulado", margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    elif target_kind == "x total":
        st.subheader("x (pl·m²) — Observado vs Simulado")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Observado","Simulado"], y=[float(x_obs), float(sim["x3"])]))
        fig.update_layout(yaxis_title="pl·m²", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:  # A2 total
        st.subheader("A2 (pl·m²) — Observado vs Simulado")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Observado","Simulado"], y=[float(A2_obs), float(sim["A2_ctrl"])]))
        fig.update_layout(yaxis_title="pl·m²", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Esperando ejecución de calibración…")
