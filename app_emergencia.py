# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v3.18.4 — Supresión (1−Ciec) + AUC + Cohortes SECUENCIALES
# ===============================================================
import io, math, datetime as dt, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

APP_TITLE = "🌾 PREDWEEM v3.18.4 — (1−Ciec) + AUC + Cohortes · PCC por fechas"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------- Inicialización del estado ----------
if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_stop" not in st.session_state:
    st.session_state.opt_stop = False

# ---------- Funciones base ----------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

def auc_time(fecha, y, mask=None):
    if fecha is None or len(fecha) == 0:
        return 0.0
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, float)
    if mask is not None:
        if len(mask) != len(f):
            mask = np.resize(mask, len(f))
        f = f[mask]; y_arr = y_arr[mask]
    valid = ~np.isnan(y_arr)
    if not np.any(valid) or len(f[valid]) < 2:
        return 0.0
    f_valid = f[valid]; y_valid = y_arr[valid]
    tdays = (f_valid - f_valid.iloc[0]).dt.days.astype(float)
    return float(np.trapz(y_valid, tdays)) if len(tdays) >= 2 else 0.0

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, float)
    out = np.zeros_like(y); cum = 0.0
    for i in range(len(y)):
        if active_mask[i]:
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val; cum += val
    return out

# ---------- Carga CSV ----------
st.sidebar.header("Datos de entrada")
up = st.sidebar.file_uploader("CSV (fecha, EMERREL diaria/acumulada)", type=["csv"])
sep_opt = st.sidebar.selectbox("Delimitador", ["auto", ",", ";", "\\t"], 0)
dec_opt = st.sidebar.selectbox("Decimal", ["auto", ".", ","], 0)
dayfirst = st.sidebar.checkbox("Fecha dd/mm/yyyy", True)
is_cumulative = st.sidebar.checkbox("Mi CSV es acumulado", False)
as_percent = st.sidebar.checkbox("Valores en %", True)
if up is None: st.stop()

def sniff_sep_dec(text):
    counts = {sep: text.count(sep) for sep in [",",";","\t"]}
    sep_guess = max(counts,key=counts.get)
    dec_guess = "." if text.count(".")>=text.count(",") else ","
    return sep_guess,dec_guess

raw = up.read()
head = raw[:8000].decode("utf-8", errors="ignore")
sep_guess, dec_guess = sniff_sep_dec(head)
sep = sep_guess if sep_opt=="auto" else sep_opt
dec = dec_guess if dec_opt=="auto" else dec_opt
df0 = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec)

cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, 0)
c_valor = st.selectbox("Columna de valor", cols, 1 if len(cols)>1 else 0)
fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = pd.to_numeric(df0[c_valor], errors="coerce")
df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)

emerrel = df["valor"].astype(float)
if as_percent: emerrel /= 100.0
if is_cumulative: emerrel = emerrel.diff().fillna(0.0).clip(lower=0)
df_plot = pd.DataFrame({"fecha": df["fecha"], "EMERREL": emerrel})
ts = pd.to_datetime(df_plot["fecha"])

# ---------- Canopy ----------
year_ref = int(ts.dt.year.mode().iloc[0])
sow_min, sow_max = dt.date(year_ref,5,1), dt.date(year_ref,8,1)
with st.sidebar:
    st.header("Siembra y Canopia")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    t_lag = st.number_input("Días a emergencia cultivo", 0, 60, 7)
    t_close = st.number_input("Días a cierre surco", 10, 120, 45)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0)
    lai_max = st.number_input("LAI máx", 0.0, 8.0, 3.0)
    k_beer = st.number_input("k Beer–Lambert", 0.1, 1.2, 0.6)

def compute_canopy(fechas, sow_date, t_lag, t_close, cov_max, lai_max, k_beer):
    days = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas])
    def logistic_between(days, start, end, y_max):
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))
    fc_dyn = np.where(days<t_lag,0.0,logistic_between(days,t_lag,t_close,cov_max/100))
    fc_dyn = np.clip(fc_dyn,0.0,1.0)
    LAI = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
    LAI = np.clip(LAI,0.0,lai_max)
    return fc_dyn, LAI

FC, LAI = compute_canopy(df_plot["fecha"], sow_date, t_lag, t_close, cov_max, lai_max, k_beer)
Ciec = np.clip((LAI / 6.0), 0.0, 1.0)
one_minus_Ciec = 1 - Ciec

# ===============================================================
# 🌿 DEFINICIÓN DE TRATAMIENTOS HERBICIDAS Y CONTROL EFECTIVO
# ===============================================================
with st.sidebar:
    st.header("Tratamientos herbicidas y ventanas")
    preR_eff   = st.slider("Presiembra residual (%)", 0, 100, 70, 5)
    preemR_eff = st.slider("Preemergente residual (%)", 0, 100, 80, 5)
    postR_eff  = st.slider("Post-residual (%)", 0, 100, 85, 5)
    gram_eff   = st.slider("Graminicida post (%)", 0, 100, 90, 5)

    preR_dur   = st.number_input("Duración preR (días)", 5, 30, 14)
    preemR_dur = st.number_input("Duración preemR (días)", 5, 20, 10)
    postR_dur  = st.number_input("Duración postR (días)", 5, 30, 20)
    gram_dur   = st.number_input("Duración graminicida (días)", 5, 15, 10)

# Ventanas fijas agronómicas
preR_start   = sow_date - dt.timedelta(days=16)
preR_end     = sow_date - dt.timedelta(days=1)
preemR_start = sow_date
preemR_end   = sow_date + dt.timedelta(days=10)
postR_start  = sow_date + dt.timedelta(days=20)
gram_start   = sow_date
gram_end     = sow_date + dt.timedelta(days=10)

# Construcción de máscara diaria
dates = pd.to_datetime(df_plot["fecha"])
mask_preR   = (dates.dt.date >= preR_start)   & (dates.dt.date <= preR_end)
mask_preemR = (dates.dt.date >= preemR_start) & (dates.dt.date <= preemR_end)
mask_postR  = (dates.dt.date >= postR_start)
mask_gram   = (dates.dt.date >= gram_start)   & (dates.dt.date <= gram_end)

# Supresión y control base
S1, S2, S3, S4, mask_since_sow = build_states(emerrel, sow_date)
auc_cruda = auc_time(ts, emerrel, mask=mask_since_sow)
factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda if auc_cruda > 0 else None

if factor_area_to_plants:
    base_pl = np.where(mask_since_sow, emerrel * factor_area_to_plants, 0.0)
    base_pl_cap = cap_cumulative(base_pl, MAX_PLANTS_CAP, mask_since_sow)

    # --- Lógica jerárquica ---
    rem = base_pl_cap.copy()
    ctrl_preR   = np.where(mask_preR,   preR_eff / 100, 0)
    ctrl_preemR = np.where(mask_preemR, preemR_eff / 100, 0)
    ctrl_postR  = np.where(mask_postR,  postR_eff / 100, 0)
    ctrl_gram   = np.where(mask_gram,   gram_eff / 100, 0)

    rem_preR   = rem * (1 - ctrl_preR)
    rem_preemR = rem_preR * (1 - ctrl_preemR)
    rem_postR  = rem_preemR * (1 - ctrl_postR)
    rem_gram   = rem_postR * (1 - ctrl_gram)
    rem_final  = np.clip(rem_gram, 0, MAX_PLANTS_CAP)
else:
    st.stop()

# ===============================================================
# 🧠 OPTIMIZACIÓN + VISUALIZACIÓN DE RESULTADOS
# ===============================================================
with st.sidebar:
    st.header("Optimización de control")
    optimizer = st.selectbox("Método", ["Grid","Aleatoria","Recocido"], index=1)
    max_evals = st.number_input("Máx. evaluaciones", 100, 50000, 2000, 100)
    if optimizer == "Recocido":
        sa_iters = st.number_input("Iteraciones", 100, 10000, 2000, 100)
        sa_T0 = st.number_input("T inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("γ enfriamiento", 0.80, 0.9999, 0.995, 0.0001)
    c1, c2 = st.columns(2)
    with c1:
        start_clicked = st.button("▶️ Iniciar", use_container_width=True, disabled=st.session_state.opt_running)
    with c2:
        stop_clicked = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running)
    if start_clicked: st.session_state.opt_running = True; st.session_state.opt_stop = False
    if stop_clicked: st.session_state.opt_stop = True

def evaluate_loss(rem_series):
    """Evalúa pérdida de rinde basada en densidad efectiva dentro del PCC"""
    mask_eff = mask_pcc if usar_pcc else mask_since_sow
    x_eff = np.trapz(rem_series[mask_eff], dx=1)
    return _loss(x_eff)

results = []
if st.session_state.opt_running:
    st.info("Optimizando...")
    for i in range(int(max_evals)):
        if st.session_state.opt_stop: break
        eff_var = np.random.uniform(0.6, 1.0)
        loss = evaluate_loss(rem_final * eff_var)
        results.append({"var": eff_var, "loss": loss})
        if i % max(1, int(max_evals / 100)) == 0:
            st.progress(i / max_evals)
    st.session_state.opt_running = False
    st.success("Optimización finalizada.")

# ---------- RESULTADOS ----------
if results:
    best = sorted(results, key=lambda r: r["loss"])[0]
    best_loss = best["loss"]
    st.metric("Pérdida óptima estimada (%)", f"{best_loss:.2f}")

    # --- Curva de pérdida ---
    x_curve = np.linspace(0, MAX_PLANTS_CAP, 300)
    y_curve = _loss(x_curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Función pérdida"))
    fig.update_layout(title="Pérdida de rendimiento vs densidad efectiva",
                      xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
    st.plotly_chart(fig, use_container_width=True)

# ---------- FLUJO HACIA PCC ----------
st.subheader("🌱 Flujo acumulado de individuos hacia el PCC")
acum_total = np.cumsum(rem_final)
idx_ini = np.argmax(ts.dt.date >= pcc_ini_date) if usar_pcc else 0
val_pcc = acum_total[idx_ini] if idx_ini < len(acum_total) else acum_total[-1]

figf = go.Figure()
figf.add_trace(go.Scatter(x=ts, y=acum_total, mode="lines", name="Acumulado con control"))
if usar_pcc:
    figf.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, fillcolor="rgba(255,215,0,0.25)",
                   line_width=0, annotation_text="PCC", annotation_position="top left")
    figf.add_trace(go.Scatter(x=[pcc_ini_date], y=[val_pcc], mode="markers+text",
                              text=["Llegan vivas"], textposition="top center"))
figf.update_layout(title="Flujo acumulado desde siembra hasta PCC",
                   xaxis_title="Fecha", yaxis_title="Malezas acumuladas (pl·m²)")
st.plotly_chart(figf, use_container_width=True)



























