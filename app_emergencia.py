# -*- coding: utf-8 -*-
# PREDWEEM — Modelo de pérdida de rinde con densidad efectiva y optimización

import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.graph_objects as go
import random

# ================== NUEVAS FUNCIONES ==================
def effective_density(ts, S_states, weights, mask):
    """
    Calcula densidad efectiva x como AUC ponderado de estados (S1–S4).
    """
    contrib = (weights["S1"] * S_states["S1"] +
               weights["S2"] * S_states["S2"] +
               weights["S3"] * S_states["S3"] +
               weights["S4"] * S_states["S4"])
    tdays = (pd.to_datetime(ts).astype("int64") / 86400e9).astype(float)
    return float(np.trapz(contrib[mask], tdays[mask]))

def loss_hyperbolic(x, alpha=0.375, Lmax=76.639):
    """Función hiperbólica (Cousens) de pérdida de rinde (%)"""
    x = np.asarray(x, dtype=float)
    return (alpha * x) / (1.0 + (alpha / Lmax) * x)

# ================== ESTADO UI ==================
st.set_page_config(page_title="PREDWEEM Optimización", layout="wide")
st.title("🌱 PREDWEEM — Cohortes + Densidad efectiva + Optimización")

with st.sidebar:
    st.header("Parámetros del modelo")
    alpha_user = st.number_input("α (sensibilidad inicial)", 0.01, 5.0, 0.375, 0.01)
    Lmax_user  = st.number_input("Lmax (% pérdida máxima)", 10.0, 200.0, 76.639, 1.0)
    max_iter   = st.number_input("Iteraciones optimización", 10, 500, 50, 10)

# ================== CARGA DE DATOS ==================
st.header("Carga de datos de ensayos")

uploaded = st.file_uploader("Subir archivo CSV con columnas: fecha, S1, S2, S3, S4, perdida_obs", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["fecha"])
    st.write("Datos cargados:", df.head())

    # Estados
    ts = df["fecha"]
    S_states = {
        "S1": df["S1"],
        "S2": df["S2"],
        "S3": df["S3"],
        "S4": df["S4"]
    }
    weights = {"S1":0.0,"S2":0.3,"S3":0.6,"S4":1.0}

    # Periodo crítico = todo el rango de fechas (puede adaptarse)
    mask_since = np.ones(len(ts), dtype=bool)

    # Cálculo de x (densidad efectiva) y pérdida predicha
    x_obs = effective_density(ts, S_states, weights, mask_since)
    y_pred = loss_hyperbolic(x_obs, alpha_user, Lmax_user)

    st.subheader("Densidad efectiva calculada")
    st.write(f"x = {x_obs:.2f} pl·m² → pérdida predicha = {y_pred:.2f}%")
    
# ================== FUNCIÓN evaluate ==================
def evaluate(params, ts, S_states, weights, mask, y_obs):
    alpha, Lmax = params
    x = effective_density(ts, S_states, weights, mask)
    yhat = loss_hyperbolic(x, alpha, Lmax)
    return np.mean((y_obs - yhat)**2), x, yhat

# ================== RANDOM SEARCH ==================
def random_search(n_iter, ts, S_states, weights, mask, y_obs):
    best_score = 1e9
    best_params, best_x, best_y = None, None, None
    for _ in range(n_iter):
        alpha = random.uniform(0.01, 1.0)
        Lmax  = random.uniform(20, 150)
        score, x, yhat = evaluate((alpha, Lmax), ts, S_states, weights, mask, y_obs)
        if score < best_score:
            best_score, best_params, best_x, best_y = score, (alpha, Lmax), x, yhat
    return best_params, best_score, best_x, best_y

if uploaded is not None:
    st.header("Optimización")

    y_obs = df["perdida_obs"].mean()  # ejemplo: usar promedio de pérdidas observadas
    best_params, best_score, best_x, best_y = random_search(max_iter, ts, S_states, weights, mask_since, y_obs)

    st.success(f"Mejor α={best_params[0]:.3f}, Lmax={best_params[1]:.1f}, RMSE={np.sqrt(best_score):.2f}")
    st.write(f"Densidad efectiva óptima: x={best_x:.2f} → pérdida predicha={best_y:.2f}%")

    # Gráfico de pérdida vs densidad
    x_curve = np.linspace(0, 200, 300)
    y_curve = loss_hyperbolic(x_curve, best_params[0], best_params[1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Curva ajustada"))
    fig.add_trace(go.Scatter(x=[best_x], y=[best_y], mode="markers+text",
                             text=[f"x={best_x:.1f}, y={best_y:.1f}%"], textposition="top center"))
    fig.update_layout(title="Pérdida de rinde (%) vs Densidad efectiva",
                      xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📂 Esperando carga de archivo CSV...")






















