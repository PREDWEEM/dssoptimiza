# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v3.18.2 — (1−Ciec) + AUC + Cohortes SECUENCIALES
# ===============================================================
import io, math, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from datetime import timedelta

# ===============================================================
# 🔧 Inicialización segura del session_state
# ===============================================================
if "opt_running" not in st.session_state:
    st.session_state.opt_running = False
if "opt_stop" not in st.session_state:
    st.session_state.opt_stop = False

# ---------- FUNCIÓN DE PÉRDIDA ----------
def _loss(x):
    x = np.asarray(x, dtype=float)
    return 0.503 * x / (1.0 + (0.503 * x / 125.91))

# ---------- UTILIDAD: AUC ROBUSTO ----------
def auc_time(fecha, y, mask=None):
    """Área bajo la curva (AUC) con fechas; robusto ante NaN o máscara vacía."""
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
    if len(tdays) < 2:
        return 0.0
    return float(np.trapz(y_valid, tdays))

# ---------- FUNCIÓN AUXILIAR: CAP DE ACUMULADO ----------
def cap_cumulative(series, cap, active_mask):
    """
    Limita la suma acumulada de una serie diaria a un máximo (cap),
    considerando solo los valores con máscara activa True.
    """
    y = np.asarray(series, float)
    out = np.zeros_like(y)
    cum = 0.0
    for i in range(len(y)):
        if active_mask[i]:
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val
            cum += val
    return out

# ---------- INTERFAZ BÁSICA ----------
APP_TITLE = "🌾 PREDWEEM v3.18.2 — Supresión + AUC + Cohortes · PCC por fechas"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------- CARGA DE DATOS ----------
st.sidebar.header("Datos de entrada")
up = st.sidebar.file_uploader("CSV (fecha, EMERREL diaria/acumulada)", type=["csv"])
sep_opt = st.sidebar.selectbox("Delimitador", ["auto", ",", ";", "\\t"], 0)
dec_opt = st.sidebar.selectbox("Decimal", ["auto", ".", ","], 0)
dayfirst = st.sidebar.checkbox("Fecha dd/mm/yyyy", True)
is_cumulative = st.sidebar.checkbox("Mi CSV es acumulado", False)
as_percent = st.sidebar.checkbox("Valores en %", True)

if up is None:
    st.info("Subí un CSV para continuar.")
    st.stop()

def sniff_sep_dec(text):
    counts = {sep: text.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get)
    dec_guess = "." if text.count(".") >= text.count(",") else ","
    return sep_guess, dec_guess

raw = up.read()
head = raw[:8000].decode("utf-8", errors="ignore")
sep_guess, dec_guess = sniff_sep_dec(head)
sep = sep_guess if sep_opt == "auto" else sep_opt
dec = dec_guess if dec_opt == "auto" else dec_opt
df0 = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec)

cols = list(df0.columns)
c_fecha = st.selectbox("Columna de fecha", cols, 0)
c_valor = st.selectbox("Columna de valor", cols, 1 if len(cols) > 1 else 0)

fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
vals = pd.to_numeric(df0[c_valor], errors="coerce")
df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)

emerrel = df["valor"].astype(float)
if as_percent: emerrel /= 100.0
if is_cumulative: emerrel = emerrel.diff().fillna(0.0).clip(lower=0)

df_plot = pd.DataFrame({"fecha": df["fecha"], "EMERREL": emerrel})
ts = pd.to_datetime(df_plot["fecha"])

# ---------- CANOPY ----------
year_ref = int(ts.dt.year.mode().iloc[0])
sow_min, sow_max = dt.date(year_ref, 5, 1), dt.date(year_ref, 8, 1)
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
    fc_dyn = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100))
    fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
    LAI = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
    LAI = np.clip(LAI, 0.0, lai_max)
    return fc_dyn, LAI

FC, LAI = compute_canopy(df_plot["fecha"], sow_date, t_lag, t_close, cov_max, lai_max, k_beer)
Ciec = np.clip((LAI / 6.0), 0.0, 1.0)
one_minus_Ciec = 1 - Ciec

# ---------- PCC CONFIGURABLE POR FECHAS ----------
with st.sidebar:
    st.header("Periodo Crítico de Competencia (PCC)")
    usar_pcc = st.checkbox("Integrar densidad efectiva solo dentro del PCC", value=True)

    if len(ts) == 0:
        st.warning("No hay datos para definir PCC."); st.stop()

    year_ref = int(ts.dt.year.mode().iloc[0])
    ts_min_date, ts_max_date = pd.to_datetime(ts.min()).date(), pd.to_datetime(ts.max()).date()
    default_ini, default_fin = dt.date(year_ref, 10, 10), dt.date(year_ref, 11, 4)
    if default_ini < ts_min_date: default_ini = ts_min_date
    if default_fin > ts_max_date: default_fin = ts_max_date

    pcc_ini_date = st.date_input("Inicio del PCC", value=default_ini,
                                 min_value=ts_min_date, max_value=ts_max_date,
                                 disabled=not usar_pcc)
    pcc_fin_date = st.date_input("Fin del PCC", value=default_fin,
                                 min_value=ts_min_date, max_value=ts_max_date,
                                 disabled=not usar_pcc)

    if pcc_ini_date > pcc_fin_date:
        st.error("⚠️ Fecha de inicio posterior al fin del PCC."); st.stop()

    if usar_pcc:
        st.caption(f"PCC activo: del **{pcc_ini_date}** al **{pcc_fin_date}** "
                   f"({(pcc_fin_date - pcc_ini_date).days} días)")
    else:
        st.caption("Integración sobre **todo el ciclo de cultivo**.")

mask_since_sow = ts.dt.date >= sow_date
mask_pcc = (ts.dt.date >= pcc_ini_date) & (ts.dt.date <= pcc_fin_date)

# ===============================================================
# 🌾 MANEJO DE TRATAMIENTOS + CÁLCULO DE ESTADOS Y A₂
# ===============================================================
NR_DAYS_DEFAULT = 10
FC_S = {"S1":0.1,"S2":0.3,"S3":0.6,"S4":1.0}

def build_states(emerrel, sow_date):
    ts = pd.to_datetime(df_plot["fecha"])
    mask_since = ts.dt.date >= sow_date
    births = np.where(mask_since, emerrel, 0.0)
    T12, T23, T34 = 10, 15, 20
    S1, S2, S3, S4 = births.copy(), np.zeros_like(births), np.zeros_like(births), np.zeros_like(births)
    for i in range(len(births)):
        if i - T12 >= 0:
            S2[i] += births[i - T12]; S1[i - T12] -= births[i - T12]
        if i - (T12 + T23) >= 0:
            S3[i] += births[i - (T12 + T23)]; S2[i - (T12 + T23)] -= births[i - (T12 + T23)]
        if i - (T12 + T23 + T34) >= 0:
            S4[i] += births[i - (T12 + T23 + T34)]; S3[i - (T12 + T23 + T34)] -= births[i - (T12 + T23 + T34)]
    S1, S2, S3, S4 = [np.clip(x, 0, None) for x in [S1, S2, S3, S4]]
    return S1, S2, S3, S4, mask_since

S1, S2, S3, S4, ms = build_states(emerrel, sow_date)
auc_cruda = auc_time(ts, emerrel, mask=ms)
factor_area_to_plants = 250 / auc_cruda if auc_cruda > 0 else None

with st.sidebar:
    st.header("Escenario de infestación")
    MAX_PLANTS_CAP = float(st.selectbox("Tope A₂ (pl·m²)", [250, 125, 62], index=0))

if auc_cruda and auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda

if factor_area_to_plants:
    S1_pl = np.where(ms, S1 * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants, 0)
    S2_pl = np.where(ms, S2 * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants, 0)
    S3_pl = np.where(ms, S3 * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants, 0)
    S4_pl = np.where(ms, S4 * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants, 0)
    base_pl_daily = np.where(ms, emerrel * factor_area_to_plants, 0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, ms)
else:
    S1_pl = S2_pl = S3_pl = S4_pl = base_pl_daily_cap = None

# ---------- GRAFICO EMERREL + PCC ----------
st.subheader("📊 EMERREL + PCC")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=emerrel, name="EMERREL"))
if usar_pcc:
    fig.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, fillcolor="rgba(255,215,0,0.25)",
                  line_width=0, annotation_text="PCC", annotation_position="top left")
fig.update_layout(title="Emergencia y PCC", xaxis_title="Fecha", yaxis_title="EMERREL (0–1)")
st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 🧠 OPTIMIZACIÓN DE MANEJO (Grid, Aleatoria, Recocido Simulado)
# ===============================================================
with st.sidebar:
    st.header("Optimización — calendario de controles")
    optimizer = st.selectbox("Método", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 2000, 100)
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1)
    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 20000, 2000, 100)
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("Enfriamiento γ", 0.80, 0.9999, 0.995, 0.0001)

    c1b, c2b = st.columns(2)
    with c1b:
        start_clicked = st.button("▶️ Iniciar", use_container_width=True, disabled=st.session_state.opt_running)
    with c2b:
        stop_clicked  = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running)
    if start_clicked:
        st.session_state.opt_running = True; st.session_state.opt_stop = False
    if stop_clicked:
        st.session_state.opt_stop = True

def evaluate_schedule(X):
    """Ejemplo de función objetivo simplificada: pérdida = _loss(x3)"""
    return {"x2": X * 1.0, "x3": X * 0.75, "loss_pct": _loss(X * 0.75)}

results = []; status_ph = st.empty(); prog_ph = st.empty()
if factor_area_to_plants and st.session_state.opt_running:
    status_ph.info("Optimizando…")
    if optimizer == "Grid (combinatorio)":
        N = int(max_evals)
        for i in range(1, N + 1):
            if st.session_state.opt_stop: break
            X = np.random.uniform(0, MAX_PLANTS_CAP)
            results.append(evaluate_schedule(X))
            if i % max(1, N // 100) == 0:
                prog_ph.progress(i / N)
    elif optimizer == "Búsqueda aleatoria":
        for i in range(1, int(max_evals) + 1):
            if st.session_state.opt_stop: break
            X = np.random.uniform(0, MAX_PLANTS_CAP)
            results.append(evaluate_schedule(X))
            if i % max(1, int(max_evals)//100) == 0:
                prog_ph.progress(i / max_evals)
    else:  # Recocido
        curX = np.random.uniform(0, MAX_PLANTS_CAP)
        cur_eval = evaluate_schedule(curX); best_eval = cur_eval
        T = float(sa_T0)
        for it in range(1, int(sa_iters) + 1):
            if st.session_state.opt_stop: break
            candX = curX + np.random.normal(0, MAX_PLANTS_CAP * 0.05)
            candX = np.clip(candX, 0, MAX_PLANTS_CAP)
            cand_eval = evaluate_schedule(candX)
            d = cand_eval["loss_pct"] - cur_eval["loss_pct"]
            if d <= 0 or np.random.random() < np.exp(-d / max(1e-9, T)):
                curX, cur_eval = candX, cand_eval
                if cur_eval["loss_pct"] < best_eval["loss_pct"]:
                    best_eval = cur_eval
            T *= sa_cooling
            if it % max(1, int(sa_iters)//100) == 0:
                prog_ph.progress(it / sa_iters)
        results.append(best_eval)
    st.session_state.opt_running = False
    status_ph.success("Optimización finalizada.")

# ===============================================================
# 📈 GRÁFICOS DE RESULTADOS Y FLUJO HACIA EL PCC
# ===============================================================
if results:
    best = sorted(results, key=lambda r: r["loss_pct"])[0]
    X2_b, X3_b = best["x2"], best["x3"]

    # --- Pérdida vs x ---
    x_curve = np.linspace(0, MAX_PLANTS_CAP, 400)
    y_curve = _loss(x_curve)
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Pérdida (%) vs x"))
    fig_loss.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text",
                                  text=[f"x₂={X2_b:.1f}"], textposition="top center"))
    fig_loss.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text",
                                  text=[f"x₃={X3_b:.1f}"], textposition="top right"))
    fig_loss.update_layout(title="Pérdida de rendimiento (%) vs densidad efectiva",
                           xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
    st.plotly_chart(fig_loss, use_container_width=True)

# --- Flujo acumulado hacia el PCC ---
st.subheader("🌱 Flujo acumulado de individuos hacia el PCC")
if factor_area_to_plants is not None:
    acum_total = np.cumsum(np.where(ms, base_pl_daily_cap, 0.0))
    idx_ini_pcc = np.argmax(ts.dt.date >= pcc_ini_date) if usar_pcc else 0
    total_al_pcc = acum_total[idx_ini_pcc] if idx_ini_pcc < len(acum_total) else acum_total[-1]

    fig_flujo = go.Figure()
    fig_flujo.add_trace(go.Scatter(x=ts, y=acum_total, mode="lines", name="Acumulado sin control", line=dict(width=2)))
    if usar_pcc:
        fig_flujo.add_vrect(x0=pcc_ini_date, x1=pcc_fin_date, fillcolor="rgba(255,215,0,0.25)",
                            line_width=0, annotation_text="PCC", annotation_position="top left")
        fig_flujo.add_trace(go.Scatter(x=[pcc_ini_date], y=[total_al_pcc],
                                       mode="markers+text", text=["Llegan vivas"], textposition="top center"))
    fig_flujo.update_layout(title="Flujo acumulado desde siembra hasta PCC",
                            xaxis_title="Fecha", yaxis_title="Malezas acumuladas (pl·m²)")
    st.plotly_chart(fig_flujo, use_container_width=True)
else:
    st.info("No se pudo graficar el flujo: AUC(EMERREL) no válido.")







