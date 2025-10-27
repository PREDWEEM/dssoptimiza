# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Simulador + Optimizador de Manejo (v3.16)
# ===============================================================
# - Supresión (1−Ciec) + Control herbicida (AUC)
# - Cohortes secuenciales (S1–S4)
# - Optimización multi-ventana con tratamientos combinados
# - Jerarquía: preR → preemR → postR → graminicida post
# - Sin restricción de solapamiento (superposición permitida)
# ===============================================================

import io, math, random, itertools, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- CONFIGURACIÓN DE LA APP ----------
st.set_page_config(
    page_title="🌾 PREDWEEM — Optimizador de Control de Malezas",
    layout="wide",
)
st.title("🌾 PREDWEEM — Supresión (1−Ciec) + Control (AUC) + Optimización")

# ===============================================================
# 🔧 PARÁMETROS CALIBRADOS
# ===============================================================

# Parámetros fijos de la función de pérdida hiperbólica
ALPHA = 0.503
LMAX  = 125.91

# Pesos relativos de los estados de emergencia
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}

# Densidad máxima equivalente (tope A2)
MAX_PLANTS_CAP = 250.0  # pl·m²

# ===============================================================
# 📉 FUNCIÓN DE PÉRDIDA HIPERBÓLICA
# ===============================================================

def _loss(x):
    """Función pérdida hiperbólica calibrada."""
    x = np.array(x, dtype=float)
    return (ALPHA * x) / (1 + (ALPHA * x / LMAX))

# ===============================================================
# 🌱 PARÁMETROS AGRONÓMICOS BÁSICOS
# ===============================================================

# Reglas de ventanas
PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW = 14   # preR ≤ siembra−14
PREEM_R_MAX_AFTER_SOW_DAYS = 10         # preemR ≤ siembra+10
POST_R_MIN_AFTER_SOW_DAYS = 20          # postR ≥ siembra+20

# Control de simulación de emergencia (por ejemplo, cohortes)
T12, T23, T34 = 15, 20, 25  # separación media entre S1–S2–S3–S4 (puede ser ajustado)

# ===============================================================
# 📊 FUNCIONES AUXILIARES (AUC Y SUPRESIÓN)
# ===============================================================

def auc_time(x, y, mask=None):
    """Área bajo la curva (AUC) entre dos vectores."""
    if mask is None:
        mask = np.ones_like(y, dtype=bool)
    x = np.array(x)[mask]
    y = np.array(y)[mask]
    if len(x) < 2:
        return 0.0
    return np.trapz(y, x)

def compute_ciec_for(sow_date):
    """
    Calcula (1−Ciec) diario desde siembra para simular supresión del cultivo.
    En apps completas, se reemplaza por modelo dinámico de LAI.
    """
    days = np.arange(0, 120)
    lai_max = 4.0
    k_beer = 0.55
    lai = lai_max / (1 + np.exp(-0.1 * (days - 40)))  # crecimiento logístico
    ciec = 1 - np.exp(-k_beer * lai)
    return ciec

# ===============================================================
# 🧩 SIMULACIÓN BASE DE EMERGENCIA (S1–S4)
# ===============================================================

def recompute_for_sow(sow_date, T12, T23, T34):
    """
    Reconstruye la secuencia temporal de emergencia y supresión
    para una fecha de siembra determinada.
    """
    try:
        ts = pd.date_range(
            start=pd.to_datetime(sow_date) - pd.Timedelta(days=15),
            end=pd.to_datetime(sow_date) + pd.Timedelta(days=120),
            freq="D"
        )
        fechas_d = ts.date
        mask_since = ts >= pd.to_datetime(sow_date)

        # Simulación simple de cohortes emergentes
        S1 = np.exp(-0.5 * ((np.arange(len(ts)) - 10) / 7)**2)
        S2 = np.exp(-0.5 * ((np.arange(len(ts)) - (10+T12)) / 7)**2)
        S3 = np.exp(-0.5 * ((np.arange(len(ts)) - (10+T12+T23)) / 7)**2)
        S4 = np.exp(-0.5 * ((np.arange(len(ts)) - (10+T12+T23+T34)) / 7)**2)

        # Normalización
        total = S1 + S2 + S3 + S4
        total = total / np.max(total)

        # Supresión por cultivo (1−Ciec)
        one_minus_ciec = compute_ciec_for(sow_date)
        one_minus_ciec = np.pad(one_minus_ciec, (0, len(ts)-len(one_minus_ciec)), mode='edge')

        # Ajustar pesos relativos y supresión
        S1_pl = W_S["S1"] * S1 * one_minus_ciec
        S2_pl = W_S["S2"] * S2 * one_minus_ciec
        S3_pl = W_S["S3"] * S3 * one_minus_ciec
        S4_pl = W_S["S4"] * S4 * one_minus_ciec
        sup_cap = (S1_pl + S2_pl + S3_pl + S4_pl) * MAX_PLANTS_CAP

        auc_cruda = auc_time(np.arange(len(ts)), total)

        return {
            "ts": ts,
            "fechas_d": fechas_d,
            "mask_since": mask_since,
            "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
            "sup_cap": sup_cap,
            "auc_cruda": auc_cruda
        }
    except Exception as e:
        st.error(f"Error en recompute_for_sow(): {e}")
        return None

# ===============================================================
# 📂 DATOS INICIALES DE EJEMPLO PARA VISUALIZACIÓN
# ===============================================================

ts = pd.date_range("2025-09-01", periods=100, freq="D")
emerrel = np.clip(np.cumsum(np.random.rand(100)) / 100, 0, 1)
df_plot = pd.DataFrame({"Fecha": ts, "EMERREL": emerrel})
factor_area_to_plants = 1.0
auc_cruda = auc_time(np.arange(len(ts)), emerrel)

# Mostrar resumen en la interfaz
st.caption(f"α = {ALPHA} · Lmax = {LMAX} · AUC cruda = {auc_cruda:.2f}")

# ===============================================================
# 🌿 DEFINICIÓN DE ACCIONES DE CONTROL (HERBICIDAS)
# ===============================================================

def act_presiembraR(date, days, eff):
    """Acción: Presiembra selectivo residual (controla S1–S2)."""
    return {
        "kind": "preR",
        "date": pd.to_datetime(date).date(),
        "days": int(days),
        "eff": float(eff),
        "states": ["S1", "S2"],
        "desc": f"Presiembra residual {eff:.0f}% ({days}d)"
    }

def act_preemR(date, days, eff):
    """Acción: Preemergente selectivo residual (controla S1–S2)."""
    return {
        "kind": "preemR",
        "date": pd.to_datetime(date).date(),
        "days": int(days),
        "eff": float(eff),
        "states": ["S1", "S2"],
        "desc": f"Preemergente residual {eff:.0f}% ({days}d)"
    }

def act_post_selR(date, days, eff):
    """Acción: Postemergente residual (controla S1–S2)."""
    return {
        "kind": "postR",
        "date": pd.to_datetime(date).date(),
        "days": int(days),
        "eff": float(eff),
        "states": ["S1", "S2"],
        "desc": f"Postemergente residual {eff:.0f}% ({days}d)"
    }

def act_post_gram(date, eff):
    """Acción: Graminicida postemergente (controla S1–S4)."""
    return {
        "kind": "post_gram",
        "date": pd.to_datetime(date).date(),
        "days": 10,  # ventana típica corta
        "eff": float(eff),
        "states": ["S1", "S2", "S3", "S4"],
        "desc": f"Graminicida post {eff:.0f}%"
    }

# ===============================================================
# 🌾 VENTANAS DE FECHAS (FUNCIONES AUXILIARES)
# ===============================================================

def pre_sow_dates(sow_date):
    """Fechas posibles de presiembra (siembra−25 → siembra−14)."""
    sow = pd.to_datetime(sow_date)
    return [sow - pd.Timedelta(days=d) for d in range(25, 13, -1)]

def preem_dates(sow_date):
    """Fechas posibles de preemergente (siembra → siembra+10)."""
    sow = pd.to_datetime(sow_date)
    return [sow + pd.Timedelta(days=d) for d in range(0, 11)]

def post_dates(sow_date):
    """Fechas posibles de postemergente (siembra+20 → siembra+40)."""
    sow = pd.to_datetime(sow_date)
    return [sow + pd.Timedelta(days=d) for d in range(20, 41)]
# ===============================================================
# 📅 DEFINICIÓN DE FECHAS DE SIEMBRA CANDIDATAS
# ===============================================================

st.subheader("📅 Fechas de siembra candidatas")

# Si la app guarda la fecha de siembra en sesión (por ejemplo, desde otro módulo)
if "sow_date" in st.session_state:
    sow_date = pd.to_datetime(st.session_state.sow_date)
else:
    # Si no hay una definida, pedirla al usuario o usar fecha actual como base
    sow_date = st.date_input(
        "Fecha base de siembra",
        value=dt.date(2025, 9, 15),
        help="Seleccioná la fecha de siembra base del cultivo."
    )
    sow_date = pd.to_datetime(sow_date)

# Generar un rango de ±5 días alrededor de la siembra base
sow_candidates = [sow_date + pd.Timedelta(days=d) for d in range(-5, 6)]

# Mostrar el rango generado
st.caption(
    f"Fechas de siembra candidatas generadas: "
    f"{sow_candidates[0].date()} → {sow_candidates[-1].date()} "
    f"({len(sow_candidates)} fechas)"
)

# ===============================================================
# ⚙️ CONTROLES DE OPTIMIZACIÓN — SIDEBAR STREAMLIT
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
    min_value=10, max_value=5000, value=500, step=10,
    help="Número máximo de combinaciones o iteraciones evaluadas."
)

# ----- Parámetros específicos de Recocido Simulado -----
with st.sidebar.expander("🔥 Parámetros de recocido simulado"):
    sa_T0 = st.number_input("Temperatura inicial (T₀)", min_value=0.001, value=1.0, step=0.1)
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
# 🌾 BLOQUE 5 — OPTIMIZADOR COMPLETO + VISUALIZACIÓN FINAL
# ===============================================================

import itertools, random, math as _math

# ---------- Construcción de escenarios ----------
def build_all_scenarios():
    scenarios = []
    for sd in sow_candidates:
        groupings = []

        if use_preR_opt:
            groupings.append([act_presiembraR(d, R, ef_preR_opt) for d in pre_sow_dates(sd) for R in res_days_preR])
        if use_preemR_opt:
            groupings.append([act_preemR(d, R, ef_preemR_opt) for d in preem_dates(sd) for R in res_days_preemR])
        if use_post_selR_opt:
            groupings.append([act_post_selR(d, R, ef_post_selR_opt) for d in post_dates(sd) for R in res_days_postR])
        if use_post_gram_opt:
            groupings.append([act_post_gram(d, ef_post_gram_opt) for d in post_dates(sd)])

        combos = [[]]
        for r in range(1, len(groupings) + 1):
            for subset in itertools.combinations(range(len(groupings)), r):
                for prod in itertools.product(*[groupings[i] for i in subset]):
                    combos.append(list(prod))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios


# ---------- Muestreo aleatorio de escenarios ----------
def sample_random_scenario():
    sd = random.choice(sow_candidates)
    schedule = []
    if use_preR_opt and random.random() < 0.7:
        schedule.append(act_presiembraR(random.choice(pre_sow_dates(sd)),
                                        random.choice(res_days_preR),
                                        ef_preR_opt))
    if use_preemR_opt and random.random() < 0.7:
        schedule.append(act_preemR(random.choice(preem_dates(sd)),
                                   random.choice(res_days_preemR),
                                   ef_preemR_opt))
    if use_post_selR_opt and random.random() < 0.7:
        schedule.append(act_post_selR(random.choice(post_dates(sd)),
                                      random.choice(res_days_postR),
                                      ef_post_selR_opt))
    if use_post_gram_opt and random.random() < 0.7:
        schedule.append(act_post_gram(random.choice(post_dates(sd)),
                                      ef_post_gram_opt))
    return pd.to_datetime(sd).date(), schedule


# ---------- Evaluación de un escenario ----------
def evaluate(sd, schedule):
    env = recompute_for_sow(sd, 10, 15, 20)  # usa tus T12, T23, T34 globales
    if env is None:
        return None

    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]
    ts_local = env["ts"]
    mask_since = env["mask_since"]
    factor_area = env["factor_area"]
    sup_cap = env["sup_cap"]

    c1 = np.ones_like(S1_pl)
    c2 = np.ones_like(S2_pl)
    c3 = np.ones_like(S3_pl)
    c4 = np.ones_like(S4_pl)

    order = {"preR": 0, "preemR": 1, "postR": 2, "post_gram": 3}

    def _apply(weights, eff, states):
        if eff <= 0:
            return
        reduc = np.clip(1.0 - (eff / 100.0) * weights, 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    for a in sorted(schedule, key=lambda x: order.get(x["kind"], 9)):
        d0 = a["date"]
        d1 = d0 + pd.Timedelta(days=int(a["days"]))
        w = ((env["fechas_d"] >= d0) & (env["fechas_d"] < d1)).astype(float)
        _apply(w, a["eff"], a["states"])

    total_ctrl = S1_pl * c1 + S2_pl * c2 + S3_pl * c3 + S4_pl * c4
    plantas_ctrl_cap = np.minimum(total_ctrl, sup_cap)

    X2loc = np.sum(sup_cap[mask_since])
    X3loc = np.sum(plantas_ctrl_cap[mask_since])
    loss = _loss(X3loc)

    return {
        "sow": sd,
        "schedule": schedule,
        "x2": X2loc,
        "x3": X3loc,
        "loss_pct": float(loss),
    }


# ===============================================================
# 🚀 EJECUCIÓN DEL OPTIMIZADOR
# ===============================================================

st.subheader("🧠 Optimizador de escenarios")

status_ph = st.empty()
progress_ph = st.empty()
results = []

if st.session_state.opt_running:
    status_ph.info("Optimizando…")

    if optimizer == "Grid (combinatorio)":
        scenarios = build_all_scenarios()
        total = len(scenarios)
        st.caption(f"{total:,} combinaciones generadas")
        if total > max_evals:
            random.seed(42)
            scenarios = random.sample(scenarios, k=max_evals)
            st.caption(f"→ Muestreadas {len(scenarios):,} combinaciones (límite)")
        step = max(1, len(scenarios)//100)
        for i, (sd, sch) in enumerate(scenarios, 1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenido en {i}/{len(scenarios)}")
                break
            r = evaluate(sd, sch)
            if r: results.append(r)
            if i % step == 0: progress_ph.progress(i/len(scenarios))

    elif optimizer == "Búsqueda aleatoria":
        N = int(max_evals)
        for i in range(1, N + 1):
            if st.session_state.opt_stop:
                status_ph.warning(f"Detenido en {i}/{N}")
                break
            sd, sch = sample_random_scenario()
            r = evaluate(sd, sch)
            if r: results.append(r)
            if i % max(1, N//100) == 0: progress_ph.progress(i/N)

    elif optimizer == "Recocido simulado":
        current = sample_random_scenario()
        best_eval = evaluate(*current)
        if not best_eval:
            status_ph.error("No se pudo iniciar el recocido.")
        else:
            cur_loss = best_eval["loss_pct"]
            T = sa_T0
            for it in range(1, int(sa_iters) + 1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Detenido en iter {it}/{sa_iters}")
                    break
                cand = sample_random_scenario()
                cand_eval = evaluate(*cand)
                if not cand_eval: continue
                delta = cand_eval["loss_pct"] - cur_loss
                if delta <= 0 or random.random() < _math.exp(-delta / max(T, 1e-6)):
                    best_eval = cand_eval if cand_eval["loss_pct"] < best_eval["loss_pct"] else best_eval
                    cur_loss = cand_eval["loss_pct"]
                    results.append(cand_eval)
                T *= sa_cooling
                if it % max(1, sa_iters//100) == 0:
                    progress_ph.progress(it / sa_iters)

    st.session_state.opt_running = False
    status_ph.success("✅ Optimización completada")

else:
    st.info("Ajustá parámetros y presioná ▶️ *Iniciar* para comenzar la optimización.")


# ===============================================================
# 🏆 VISUALIZACIÓN DEL MEJOR ESCENARIO
# ===============================================================

if results:
    best = sorted(results, key=lambda r: r["loss_pct"])[0]
    st.subheader("🏆 Mejor escenario encontrado")
    st.markdown(
        f"**Fecha de siembra:** {best['sow']}  \n"
        f"**x₂:** {best['x2']:.1f} · **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"**Pérdida estimada:** {best['loss_pct']:.2f}%"
    )

    df_best = pd.DataFrame([
        {
            "Tipo": a["kind"],
            "Inicio": a["date"],
            "Duración (d)": a["days"],
            "Eficiencia (%)": a["eff"],
            "Estados": ",".join(a["states"])
        }
        for a in best["schedule"]
    ])
    st.dataframe(df_best, use_container_width=True)

    # -------- Gráfico A: EMERREL + controles --------
    env_best = recompute_for_sow(best["sow"], 10, 15, 20)
    ts_b = env_best["ts"]
    mask_b = env_best["mask_since"]
    sup_cap_b = env_best["sup_cap"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], name="EMERREL cruda", mode="lines"))
    fig1.add_trace(go.Scatter(x=ts_b, y=sup_cap_b, name="Plantas sin control", yaxis="y2", mode="lines"))
    fig1.update_layout(
        title="Emergencia y acumulación de plantas (mejor escenario)",
        yaxis_title="EMERREL", yaxis2=dict(overlaying="y", side="right", title="Plantas·m²")
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -------- Gráfico B: Pérdida (%) vs x --------
    x_curve = np.linspace(0, MAX_PLANTS_CAP, 400)
    y_curve = _loss(x_curve)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_curve, y=y_curve, name="Modelo pérdida"))
    fig2.add_trace(go.Scatter(x=[best["x2"]], y=[_loss(best["x2"])], mode="markers+text",
                              text=["x₂"], textposition="top center"))
    fig2.add_trace(go.Scatter(x=[best["x3"]], y=[_loss(best["x3"])], mode="markers+text",
                              text=["x₃"], textposition="top center"))
    fig2.update_layout(title="Curva pérdida vs densidad efectiva", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Aún no hay resultados disponibles. Ejecutá el optimizador primero.")










