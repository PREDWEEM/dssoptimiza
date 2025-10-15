# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Cohortes Secuenciales + Optimización
# + Áreas coloreadas por tratamiento herbicida
# ===============================================================

import io, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ---------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------------
st.set_page_config(page_title="PREDWEEM · Cohortes + Optimización", layout="wide")
st.title("🌱 PREDWEEM — Cohortes secuenciales + Optimización (áreas coloreadas)")

# ---------------------------------------------------------------
# PARÁMETROS CALIBRADOS
# ---------------------------------------------------------------
ALPHA = 0.503
LMAX  = 125.91
W_S = {"S1": 0.369, "S2": 0.393, "S3": 1.150, "S4": 1.769}
A2_CAP = 250.0  # tope de densidad acumulada (pl·m²)

def loss_fun(x):
    """Función de pérdida hiperbólica calibrada"""
    x = np.asarray(x, dtype=float)
    return ALPHA * x / (1.0 + (ALPHA * x / LMAX))

# ---------------------------------------------------------------
# GENERACIÓN DE DATOS DE EJEMPLO
# ---------------------------------------------------------------
days = pd.date_range("2025-09-01", "2025-11-30", freq="D")
sow_date = dt.datetime(2025, 9, 20)

# Emergencia simulada
em_rel = np.clip(np.cumsum(np.random.normal(0.02, 0.01, len(days))), 0, 1)
S1_raw, S2_raw, S3_raw, S4_raw = em_rel*0.3, em_rel*0.25, em_rel*0.25, em_rel*0.2

# ---------------------------------------------------------------
# SECUENCIALIDAD S1→S2→S3→S4 (versión robusta)
# ---------------------------------------------------------------
S1 = np.clip(np.array(S1_raw), 0, None)
S2_cap = np.minimum(np.array(S2_raw), np.cumsum(S1))
S2 = np.clip(S2_cap - S1, 0, None)
S3_cap = np.minimum(np.array(S3_raw), np.cumsum(S1 + S2))
S3 = np.clip(S3_cap - (S1 + S2), 0, None)
S4_cap = np.minimum(np.array(S4_raw), np.cumsum(S1 + S2 + S3))
S4 = np.clip(S4_cap - (S1 + S2 + S3), 0, None)

df = pd.DataFrame({"fecha": days, "S1": S1, "S2": S2, "S3": S3, "S4": S4})
y_max = df[["S1","S2","S3","S4"]].sum(axis=1).max() * 1.1

# ---------------------------------------------------------------
# DEFINICIÓN DE FECHAS DE TRATAMIENTOS (EJEMPLO)
# ---------------------------------------------------------------
t_preR_ini, t_preR_fin = sow_date - timedelta(days=18), sow_date - timedelta(days=14)
t_preemR_ini, t_preemR_fin = sow_date, sow_date + timedelta(days=10)
t_postR_ini, t_postR_fin = sow_date + timedelta(days=20), sow_date + timedelta(days=30)
t_gram_ini, t_gram_fin = sow_date + timedelta(days=5), sow_date + timedelta(days=10)

# ---------------------------------------------------------------
# GRÁFICO PRINCIPAL — EMERGENCIA Y TRATAMIENTOS (ÁREAS COLOREADAS)
# ---------------------------------------------------------------
fig = go.Figure()

# --- Curvas de estados ---
for s, color in zip(["S1","S2","S3","S4"], ["#1f77b4","#2ca02c","#ff7f0e","#d62728"]):
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df[s], name=s, mode="lines", line=dict(color=color)
    ))

# --- Áreas coloreadas por tratamiento ---
fig.add_shape(
    type="rect", x0=t_preR_ini, x1=t_preR_fin, y0=0, y1=y_max,
    fillcolor="rgba(255,165,0,0.15)", line=dict(width=0), layer="below"
)
fig.add_annotation(
    x=t_preR_ini + (t_preR_fin - t_preR_ini)/2, y=y_max*0.97,
    text="Pre-siembra R", showarrow=False, font=dict(size=10, color="orange")
)

fig.add_shape(
    type="rect", x0=t_preemR_ini, x1=t_preemR_fin, y0=0, y1=y_max,
    fillcolor="rgba(255,215,0,0.15)", line=dict(width=0), layer="below"
)
fig.add_annotation(
    x=t_preemR_ini + (t_preemR_fin - t_preemR_ini)/2, y=y_max*0.97,
    text="Pre-emergente R", showarrow=False, font=dict(size=10, color="goldenrod")
)

fig.add_shape(
    type="rect", x0=t_postR_ini, x1=t_postR_fin, y0=0, y1=y_max,
    fillcolor="rgba(0,128,0,0.15)", line=dict(width=0), layer="below"
)
fig.add_annotation(
    x=t_postR_ini + (t_postR_fin - t_postR_ini)/2, y=y_max*0.97,
    text="Post-residual", showarrow=False, font=dict(size=10, color="green")
)

fig.add_shape(
    type="rect", x0=t_gram_ini, x1=t_gram_fin, y0=0, y1=y_max,
    fillcolor="rgba(255,0,0,0.15)", line=dict(width=0), layer="below"
)
fig.add_annotation(
    x=t_gram_ini + (t_gram_fin - t_gram_ini)/2, y=y_max*0.97,
    text="Graminicida", showarrow=False, font=dict(size=10, color="red")
)

# --- Ajuste de layout ---
fig.update_layout(
    title="Emergencia por estado (S1–S4) + Áreas de tratamientos herbicidas",
    yaxis=dict(title="Emergencia relativa / densidad efectiva", range=[0, y_max]),
    shapes_layer="below",
    margin=dict(l=60, r=60, t=60, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# SECCIÓN DE OPTIMIZACIÓN (EJEMPLO DEMO)
# ---------------------------------------------------------------
st.subheader("🔎 Escenario óptimo simulado")

best = {
    "t_preR_ini": t_preR_ini, "t_preR_fin": t_preR_fin,
    "t_preemR_ini": t_preemR_ini, "t_preemR_fin": t_preemR_fin,
    "t_postR_ini": t_postR_ini, "t_postR_fin": t_postR_fin,
    "t_gram_ini": t_gram_ini, "t_gram_fin": t_gram_fin,
}

fig_opt = go.Figure()

# Curva del mejor escenario simulado
fig_opt.add_trace(go.Scatter(
    x=df["fecha"],
    y=(df["S1"]+df["S2"]+df["S3"]+df["S4"]) * (1 - np.random.uniform(0.2,0.3,len(df))),
    name="Densidad efectiva controlada",
    mode="lines",
    line=dict(color="black", width=2)
))

# --- Áreas coloreadas por tratamiento también aquí ---
for (ini, fin, color, label) in [
    (best["t_preR_ini"], best["t_preR_fin"], "rgba(255,165,0,0.15)", "Pre-siembra R"),
    (best["t_preemR_ini"], best["t_preemR_fin"], "rgba(255,215,0,0.15)", "Pre-emergente R"),
    (best["t_postR_ini"], best["t_postR_fin"], "rgba(0,128,0,0.15)", "Post-residual"),
    (best["t_gram_ini"], best["t_gram_fin"], "rgba(255,0,0,0.15)", "Graminicida")
]:
    fig_opt.add_shape(type="rect", x0=ini, x1=fin, y0=0, y1=y_max,
                      fillcolor=color, line=dict(width=0), layer="below")
    fig_opt.add_annotation(x=ini + (fin - ini)/2, y=y_max*0.97,
                           text=label, showarrow=False,
                           font=dict(size=10, color=color.replace("rgba","rgb").split(",0.")[0]+")"))

fig_opt.update_layout(
    title="Escenario óptimo de control (áreas coloreadas)",
    yaxis=dict(title="Emergencia relativa / densidad efectiva", range=[0, y_max]),
    shapes_layer="below",
    margin=dict(l=60, r=60, t=60, b=40)
)
st.plotly_chart(fig_opt, use_container_width=True)

st.success("✅ Script completo: áreas coloreadas por tratamiento herbicida.")
