
# -*- coding: utf-8 -*-
# Streamlit — Calibración desde Excel (calibración + testeo)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Calibración Hipérbolica (Excel)", layout="wide")
st.title("📈 Calibración desde archivos Excel (Calibración + Testeo)")

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

    # Aquí se pueden aplicar tratamientos PRE/POST (pendiente implementar)
    dens_eff = sub["dens_base"].sum()
    return dens_eff

# ----------------------------
# App principal
# ----------------------------
st.sidebar.header("📂 Archivos de entrada")
file_calib = st.sidebar.file_uploader("Archivo de CALIBRACIÓN (ej: calibra borde.xlsx)", type=["xlsx"])
file_test = st.sidebar.file_uploader("Archivo de TESTEO (ej: testeo.xlsx)", type=["xlsx"])

if file_calib and file_test:
    # Leer calibración
    ensayos_c = pd.read_excel(file_calib, sheet_name="ensayos")
    emergencia_c = pd.read_excel(file_calib, sheet_name="emergencia")
    tratamientos_c = pd.read_excel(file_calib, sheet_name="tratamientos")

    # Leer testeo
    ensayos_t = pd.read_excel(file_test, sheet_name="ensayos")
    emergencia_t = pd.read_excel(file_test, sheet_name="emergencia")
    tratamientos_t = pd.read_excel(file_test, sheet_name="tratamientos")

    st.subheader("👀 Datos cargados")
    st.write("Ensayos calibración:", ensayos_c.head())
    st.write("Ensayos testeo:", ensayos_t.head())

    # Construir dataset calibración
    resultados_c = []
    for _, row in ensayos_c.iterrows():
        dens_eff = calcular_densidad(emergencia_c, tratamientos_c,
                                     row["ensayo_id"], row["Ca"], row["Cs"],
                                     row["pc_ini"], row["pc_fin"])
        resultados_c.append({
            "ensayo_id": row["ensayo_id"],
            "dens_eff": dens_eff,
            "loss_obs_pct": row["loss_obs_pct"],
            "dataset": "calibración"
        })
    df_c = pd.DataFrame(resultados_c)

    # Construir dataset testeo
    resultados_t = []
    for _, row in ensayos_t.iterrows():
        dens_eff = calcular_densidad(emergencia_t, tratamientos_t,
                                     row["ensayo_id"], row["Ca"], row["Cs"],
                                     row["pc_ini"], row["pc_fin"])
        resultados_t.append({
            "ensayo_id": row["ensayo_id"],
            "dens_eff": dens_eff,
            "loss_obs_pct": row["loss_obs_pct"],
            "dataset": "testeo"
        })
    df_t = pd.DataFrame(resultados_t)

    # Unir datasets
    df = pd.concat([df_c, df_t], ignore_index=True)

    st.subheader("📊 Dataset generado")
    st.write(df.head())

    # ==========================
    # Optimización α y Lmax (solo calibración)
    # ==========================
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(df_c["dens_eff"].values, alpha, Lmax)
        return rmse(df_c["loss_obs_pct"].values, y_pred)

    res = minimize(objective, [1.0, 80.0], bounds=[(1e-6, None),(1e-6,None)])
    alpha_opt, Lmax_opt = res.x

    st.success(f"✅ Parámetros optimizados: α={alpha_opt:.4f}, Lmax={Lmax_opt:.2f}")

    # ==========================
    # Evaluación calibración y testeo
    # ==========================
    st.subheader("📊 Métricas por conjunto")
    metrics = {}
    for name, subset in [("Calibración", df_c), ("Testeo", df_t)]:
        y_true = subset["loss_obs_pct"].values
        y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
        metrics[name] = {
            "RMSE": rmse(y_true,y_pred),
            "MAE": mae(y_true,y_pred),
            "R2": r2(y_true,y_pred)
        }
    st.write(pd.DataFrame(metrics).T.round(2))

    # ==========================
    # Gráfico
    # ==========================
    st.subheader("📉 Curva ajustada vs datos")
    fig, ax = plt.subplots(figsize=(7,6))
    x_grid = np.linspace(0, max(df["dens_eff"])*1.2, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    ax.plot(x_grid, y_grid, 'b-', label=f"Hiperbólica ajustada\nα={alpha_opt:.3f}, Lmax={Lmax_opt:.1f}")

    ax.scatter(df_c["dens_eff"], df_c["loss_obs_pct"], c="green", marker="o", label="Calibración (obs)")
    ax.scatter(df_t["dens_eff"], df_t["loss_obs_pct"], c="red", marker="s", label="Testeo (obs)")

    ax.set_xlabel("Densidad efectiva (x)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ==========================
    # Descargar resultados
    # ==========================
    df["loss_pred_opt"] = loss_function(df["dens_eff"].values, alpha_opt, Lmax_opt)
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados optimizados (CSV)", csv_out, 
                       "resultados_optimizados.csv", "text/csv")
