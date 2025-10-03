# -*- coding: utf-8 -*-
# Streamlit — Calibración completa desde Excel (ensayos, emergencia, tratamientos)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Calibración Hipérbolica (Excel)", layout="wide")
st.title("📈 Calibración desde archivo Excel")

# ----------------------------
# Parámetros fijos
# ----------------------------
W_ESTADOS = {"s1": 0.25, "s2": 0.50, "s3": 0.75, "s4": 1.0}

# Función hiperbólica
def loss_function(x, alpha, Lmax):
    return (alpha * Lmax * x) / (alpha + x)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# ----------------------------
# Cálculo de densidad efectiva
# ----------------------------
def calcular_densidad(emer_df, trat_df, ensayo, Ca, Cs, pc_ini, pc_fin):
    # Filtrar emergencia del ensayo
    sub = emer_df[emer_df["ensayo_id"]==ensayo].copy()
    sub["fecha"] = pd.to_datetime(sub["fecha"])
    sub = sub[(sub["fecha"]>=pd.to_datetime(pc_ini)) & (sub["fecha"]<=pd.to_datetime(pc_fin))]

    # Densidad base = emergencia relativa * Ca/Cs
    sub["dens_base"] = sub["emer_rel"] * (Ca/Cs)

    # TODO: aplicar tratamientos si los hubiera (ahora es un multiplicador simple)
    # Ejemplo: reducir densidad en función de eficacia (se puede extender luego)
    # por ahora lo dejamos directo

    dens_eff = sub["dens_base"].sum()
    return dens_eff

# ----------------------------
# App principal
# ----------------------------
file = st.file_uploader("📂 Subí el archivo Excel (calibra borde.xlsx)", type=["xlsx"])

if file:
    xls = pd.ExcelFile(file)
    st.write("📑 Hojas encontradas:", xls.sheet_names)

    ensayos = pd.read_excel(file, sheet_name="ensayos")
    emergencia = pd.read_excel(file, sheet_name="emergencia")
    tratamientos = pd.read_excel(file, sheet_name="tratamientos")

    st.subheader("👀 Datos cargados")
    st.write("Ensayos:", ensayos.head())
    st.write("Emergencia:", emergencia.head())
    st.write("Tratamientos:", tratamientos.head())

    # Construir dataset con densidad efectiva
    resultados = []
    for _, row in ensayos.iterrows():
        ensayo = row["ensayo_id"]
        dens_eff = calcular_densidad(emergencia, tratamientos,
                                     ensayo, row["Ca"], row["Cs"],
                                     row["pc_ini"], row["pc_fin"])
        resultados.append({
            "ensayo_id": ensayo,
            "dens_eff": dens_eff,
            "loss_obs_pct": row["loss_obs_pct"],
            "dataset": "calibración"  # TODO: decidir según archivo (se puede separar testeo)
        })
    df = pd.DataFrame(resultados)

    st.subheader("📊 Dataset generado")
    st.write(df.head())

    # ==========================
    # Optimización α y Lmax
    # ==========================
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(df["dens_eff"].values, alpha, Lmax)
        return rmse(df["loss_obs_pct"].values, y_pred)

    res = minimize(objective, [1.0, 80.0], bounds=[(1e-6, None),(1e-6,None)])
    alpha_opt, Lmax_opt = res.x

    st.success(f"✅ Parámetros optimizados: α={alpha_opt:.4f}, Lmax={Lmax_opt:.2f}")

    # Métricas
    y_true = df["loss_obs_pct"].values
    y_pred = loss_function(df["dens_eff"].values, alpha_opt, Lmax_opt)
    st.write({
        "RMSE": rmse(y_true,y_pred),
        "MAE": mae(y_true,y_pred),
        "R2": r2(y_true,y_pred)
    })

    # Gráfico
    st.subheader("📉 Curva ajustada vs datos")
    fig, ax = plt.subplots(figsize=(7,6))
    x_grid = np.linspace(0, max(df["dens_eff"])*1.2, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    ax.plot(x_grid, y_grid, 'b-', label=f"Hiperbólica ajustada\nα={alpha_opt:.3f}, Lmax={Lmax_opt:.1f}")
    ax.scatter(df["dens_eff"], df["loss_obs_pct"], c="green", marker="o", label="Obs")
    ax.set_xlabel("Densidad efectiva (x)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

