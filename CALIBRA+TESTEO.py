# -*- coding: utf-8 -*-
# Streamlit app — Calibración y Testeo con densidad efectiva en plantas/m²

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Calibración pérdida de rinde", layout="wide")

# =========================
# Funciones auxiliares
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

def loss_function(x, alpha, Lmax):
    return (alpha * Lmax * x) / (alpha + x)

# =========================
# Cálculo densidad efectiva
# =========================
def calcular_densidad_ensayo(row, emer_df, trat_df, W_ESTADOS, infestacion_escenario=250):
    """
    Calcula densidad efectiva de malezas en plantas/m² para un ensayo.

    Parámetros
    ----------
    row : pandas.Series
        Fila de la hoja 'ensayos' (incluye ensayo_id, fechas, MAX_PLANTS_CAP o IC, etc.)
    emer_df : pandas.DataFrame
        Hoja 'emergencia' (ensayo_id, fecha, emer_rel).
    trat_df : pandas.DataFrame
        Hoja 'tratamientos' (ensayo_id, tipo, fecha_aplicacion, eficacia_pct, residual_dias, actua_s1...).
    W_ESTADOS : dict
        Pesos fijos para S1–S4. Ej: {"s1":0.25, "s2":0.5, "s3":0.75, "s4":1.0}
    infestacion_escenario : int
        Nivel de infestación máximo simulado (ej: 250, 125, 50 plantas/m²).

    Retorna
    -------
    dens_eff : float
        Densidad efectiva en plantas/m² durante el PC.
    """

    ensayo_id = row["ensayo_id"]
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    fecha_siembra = pd.to_datetime(row["fecha_siembra"])

    # 1. Capacidad máxima del ensayo (IC o MAX_PLANTS_CAP)
    IC = row.get("MAX_PLANTS_CAP", np.nan)
    if pd.isna(IC):
        IC = infestacion_escenario  # fallback

    # 2. Emergencia del ensayo
    emer_sub = emer_df[emer_df["ensayo_id"] == ensayo_id].copy()
    if emer_sub.empty:
        return np.nan

    emer_sub["fecha"] = pd.to_datetime(emer_sub["fecha"])
    emer_sub = emer_sub.sort_values("fecha")
    emer_sub["emer_rel"] = emer_sub["emer_rel"].astype(float)

    # 3. Acumuladores por estado
    auc_estados = {k: 0.0 for k in W_ESTADOS.keys()}

    # Clasificar en estados y acumular dentro del PC
    for _, r in emer_sub.iterrows():
        edad = (r["fecha"] - fecha_siembra).days
        if 1 <= edad <= 6:
            estado = "s1"
        elif 7 <= edad <= 15:
            estado = "s2"
        elif 16 <= edad <= 25:
            estado = "s3"
        else:
            estado = "s4"

        if pc_ini <= r["fecha"] <= pc_fin:
            auc_estados[estado] += r["emer_rel"]

    # 4. Aplicar tratamientos (eficacia y residualidad)
    trat_sub = trat_df[trat_df["ensayo_id"] == ensayo_id].copy()
    if not trat_sub.empty:
        for _, t in trat_sub.iterrows():
            ef = float(t.get("eficacia_pct", 0) or 0) / 100.0
            if ef <= 0:
                continue
            for est in W_ESTADOS.keys():
                if t.get(f"actua_{est}", 0) == 1:
                    auc_estados[est] *= (1 - ef)

    # 5. Calcular densidad fraccional ponderada por estados
    dens_frac = 0.0
    for est, peso in W_ESTADOS.items():
        dens_frac += auc_estados[est] * peso

    # 6. Escalar a plantas/m² → usando IC (capacidad máxima)
    dens_eff = dens_frac * IC

    return dens_eff

# =========================
# App principal
# =========================
st.title("🌱 Calibración y Testeo de pérdida de rinde")
uploaded = st.file_uploader("Subir archivo Excel (con hojas: ensayos, emergencia, tratamientos)", type=["xlsx"])

escenario = st.sidebar.selectbox(
    "Escenario de infestación (plantas/m²)",
    [250, 125, 50]
)

if uploaded:
    # Leer hojas
    ensayos = pd.read_excel(uploaded, sheet_name="ensayos")
    emergencia = pd.read_excel(uploaded, sheet_name="emergencia")
    tratamientos = pd.read_excel(uploaded, sheet_name="tratamientos")

    # Pesos fijos
    W_ESTADOS = {"s1": 0.25, "s2": 0.5, "s3": 0.75, "s4": 1.0}

    # Construir dataset
    resultados = []
    for _, row in ensayos.iterrows():
        dens_eff = calcular_densidad_ensayo(row, emergencia, tratamientos, W_ESTADOS, escenario)
        resultados.append({
            "ensayo_id": row["ensayo_id"],
            "dataset": row.get("dataset", "calibración"),
            "loss_obs_pct": row.get("loss_obs_pct", np.nan),
            "dens_eff": dens_eff
        })
    df_res = pd.DataFrame(resultados)

    st.subheader("Datos procesados")
    st.dataframe(df_res)

    # Separar calibración y testeo
    calib = df_res[df_res["dataset"] == "calibración"].dropna()
    test = df_res[df_res["dataset"] == "testeo"].dropna()

    # Optimización de parámetros
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(calib["dens_eff"].values, alpha, Lmax)
        return rmse(calib["loss_obs_pct"].values, y_pred)

    res = minimize(objective, [1.0, 80.0], bounds=[(1e-6, None), (1e-6, None)])
    alpha_opt, Lmax_opt = res.x

    st.subheader("Parámetros optimizados")
    st.write(f"α = {alpha_opt:.3f} · Lmax = {Lmax_opt:.2f}")

    # Evaluación
    for name, subset in [("Calibración", calib), ("Testeo", test)]:
        if subset.empty: 
            continue
        y_true = subset["loss_obs_pct"].values
        y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
        st.markdown(f"**{name}:**")
        st.write(f"RMSE = {rmse(y_true, y_pred):.2f}")
        st.write(f"MAE  = {mae(y_true, y_pred):.2f}")
        st.write(f"R²   = {r2(y_true, y_pred):.3f}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(7,6))
    x_grid = np.linspace(0, df_res["dens_eff"].max()*1.1, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    ax.plot(x_grid, y_grid, "b-", label="Curva optimizada")

    ax.scatter(calib["dens_eff"], calib["loss_obs_pct"], c="green", label="Calibración")
    if not test.empty:
        ax.scatter(test["dens_eff"], test["loss_obs_pct"], c="red", marker="s", label="Testeo")

    ax.set_xlabel("Densidad efectiva (plantas/m²)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.set_title(f"Escenario: {escenario} plantas/m²")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Descargar resultados
    df_res["loss_pred_pct"] = loss_function(df_res["dens_eff"], alpha_opt, Lmax_opt)
    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados", data=csv, file_name="resultados_calibrados.csv", mime="text/csv")
