# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM — Cohortes Secuenciales + Optimización
# + Tratamientos con líneas horizontales y leyenda lateral
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
st.title("🌱 PREDWEEM — Cohortes secuenciales + Optimización")

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
# SECUENCIALIDAD S1→S2→S3→S4
# ---------------------------------------------------------------
S1 = S1_raw.clip(lower=0)
S2_cap = np.minimum(S2_raw, S1.cumsum())
S2 = (S2_cap - S1).clip(lower=0)
S3_cap = np.minimum(S3_raw, (S1 + S2).cumsum())
S3 = (S3_cap - (S1 + S2)).clip(lower=0)
S4_cap = np.minimum(S4_raw, (S1 + S2 + S3).cumsum())
S4 = (S4_cap - (S1 + S2 + S3)).clip(lower=0)

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
# FUNCIONES PARA REPRESENTAR TRATAMIENTOS
# ---------------------------------------------------------------
def add_trat_line(fig, t_ini, t_fin, color, label, y_pos=-5, thickness=5):
    """Dibuja línea horizontal para indicar aplicación de un tratamiento"""
    if pd.isna(t_ini) or pd.isna(t_fin): 
        return
    fig.add_shape(
        type="line", x0=t_ini, x1=t_fin, y0=y_pos, y1=y_pos,
        line=dict(color=color, width=thickness)
    )
    fig.add_annotation(
        x=t_ini + (t_fin - t_ini)/2,
        y=y_pos - 1.2,
        text=label,
        showarrow=False,
        font=dict(size=10, color=color)
    )

def add_trat_legend(fig, y_max, sow_date):
    """Agrega leyenda lateral con líneas de colores"""
    trat_info = [
        ("Pre-siembra R", "orange"),
        ("Pre-emergente R", "goldenrod"),
        ("Post-residual", "green"),
        ("Graminicida", "red"),
    ]
    y_leg_top = y_max * 0.9
    dy = y_max * 0.05
    for i, (label, color) in enumerate(trat_info):
        y_leg = y_leg_top - i * dy
        fig.add_shape(
            type="line",
            x0=sow_date + timedelta(days=80),
            x1=sow_date + timedelta(days=90),
            y0=y_leg, y1=y_leg,
            line=dict(color=color, width=5)
        )
        fig.add_annotation(
            x=sow_date + timedelta(days=92),
            y=y_leg,
            text=label,
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=color)
        )

# ---------------------------------------------------------------
# GRÁFICO PRINCIPAL — EMERGENCIA Y TRATAMIENTOS
# ---------------------------------------------------------------
fig = go.Figure()

# Curvas de estados
for s, color in zip(["S1","S2","S3","S4"], ["#1f77b4","#2ca02c","#ff7f0e","#d62728"]):
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df[s], name=s, mode="lines", line=dict(color=color)
    ))

# Líneas horizontales de tratamientos
add_trat_line(fig, t_preR_ini, t_preR_fin, "orange", "Pre-siembra R", y_pos=-2)
add_trat_line(fig, t_preemR_ini, t_preemR_fin, "goldenrod", "Pre-emergente R", y_pos=-4)
add_trat_line(fig, t_postR_ini, t_postR_fin, "green", "Post-residual", y_pos=-6)
add_trat_line(fig, t_gram_ini, t_gram_fin, "red", "Graminicida", y_pos=-8)

# Leyenda lateral
add_trat_legend(fig, y_max, sow_date)

fig.update_layout(
    title="Emergencia por estado (S1–S4) + Tratamientos herbicidas",
    yaxis=dict(title="Emergencia relativa / densidad efectiva", range=[-12, y_max]),
    shapes_layer="below",
    margin=dict(l=60, r=120, t=60, b=40)
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

# Añadir líneas y leyenda también aquí
add_trat_line(fig_opt, best["t_preR_ini"], best["t_preR_fin"], "orange", "Pre-siembra R", y_pos=-2)
add_trat_line(fig_opt, best["t_preemR_ini"], best["t_preemR_fin"], "goldenrod", "Pre-emergente R", y_pos=-4)
add_trat_line(fig_opt, best["t_postR_ini"], best["t_postR_fin"], "green", "Post-residual", y_pos=-6)
add_trat_line(fig_opt, best["t_gram_ini"], best["t_gram_fin"], "red", "Graminicida", y_pos=-8)
add_trat_legend(fig_opt, y_max, sow_date)

fig_opt.update_layout(
    title="Escenario óptimo de control",
    yaxis=dict(title="Emergencia relativa / densidad efectiva", range=[-12, y_max]),
    shapes_layer="below",
    margin=dict(l=60, r=120, t=60, b=40)
)
st.plotly_chart(fig_opt, use_container_width=True)

st.success("✅ Script completo: Cohortes secuenciales + Optimización + Líneas y leyenda lateral.")
