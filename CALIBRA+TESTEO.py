# -*- coding: utf-8 -*-
# Streamlit — Optimización de α y Lmax con datos de calibración/testeo
# Lee un CSV con columnas: ensayo_id, dens_eff, loss_obs_pct, dataset
# Ajusta función hiperbólica, calcula métricas y genera gráficos.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Calibración Hipérbolica", layout="centered")

st.title("📈 Calibración de curva de pérdida (α y Lmax)")

# ==========================
# 1. Subir CSV
# ==========================
file = st.file_uploader("📂 Subí el archivo resultados_calibra_testeo.csv", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    if not {"dens_eff","loss_obs_pct","dataset"}.issubset(df.columns):
        st.error("⚠️ El archivo no contiene las columnas requeridas: dens_eff, loss_obs_pct, dataset")
        st.stop()

    calib = df[df["dataset"]=="calibración"]
    test = df[df["dataset"]=="testeo"]

    # ==========================
    # 2. Función hiperbólica y métricas
    # ==========================
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

    # ==========================
    # 3. Optimización de α y Lmax (solo calibración)
    # ==========================
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(calib["dens_eff"].values, alpha, Lmax)
        return rmse(calib["loss_obs_pct"].values, y_pred)

    x0 = [1.0, 80.0]
    bounds = [(1e-6, None), (1e-6, None)]
    res = minimize(objective, x0, bounds=bounds)

    alpha_opt, Lmax_opt = res.x

    st.success(f"✅ Parámetros optimizados: α = {alpha_opt:.4f} · Lmax = {Lmax_opt:.2f}")

    # ==========================
    # 4. Evaluación
    # ==========================
    st.subheader("📊 Métricas de desempeño")

    metrics = {}
    for name, subset in [("Calibración", calib), ("Testeo", test)]:
        if subset.empty: 
            continue
        y_true = subset["loss_obs_pct"].values
        y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
        metrics[name] = {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "R2": r2(y_true, y_pred)
        }

    st.write(pd.DataFrame(metrics).T.round(2))

    # ==========================
    # 5. Gráfico
    # ==========================
    st.subheader("📉 Curva ajustada vs datos observados")

    fig, ax = plt.subplots(figsize=(7,6))
    x_grid = np.linspace(0, max(df["dens_eff"])*1.2, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    ax.plot(x_grid, y_grid, 'b-', label=f"Hiperbólica ajustada\nα={alpha_opt:.3f}, Lmax={Lmax_opt:.1f}")

    ax.scatter(calib["dens_eff"], calib["loss_obs_pct"], c="green", marker="o", label="Calibración (obs)")
    if not test.empty:
        ax.scatter(test["dens_eff"], test["loss_obs_pct"], c="red", marker="s", label="Testeo (obs)")

    ax.set_xlabel("Densidad efectiva (x)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.set_title("Curva de pérdida recalibrada")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ==========================
    # 6. Exportar resultados
    # ==========================
    df["loss_pred_opt"] = loss_function(df["dens_eff"].values, alpha_opt, Lmax_opt)
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados optimizados (CSV)", csv_out, "resultados_optimizados.csv", "text/csv")
