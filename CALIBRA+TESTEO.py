# -*- coding: utf-8 -*-
# Streamlit — Calibración + Testeo con PRE/POST y estados S1–S4
# Versión robusta: normaliza nombres de columnas

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------
# Parámetros calibrados y pesos fijos
# -------------------
ALPHA = 0.9782
LMAX = 83.77
W_ESTADOS = {"s1": 0.25, "s2": 0.5, "s3": 0.75, "s4": 1.0}

# -------------------
# Funciones auxiliares
# -------------------
def normalize_columns(df):
    """Pone todas las columnas en minúsculas y sin espacios"""
    df.columns = df.columns.str.strip().str.lower()
    return df

def loss_pred(x, alpha=ALPHA, Lmax=LMAX):
    return Lmax * ((alpha * x) / (1 + alpha * x))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def eval_metrics(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}

def asignar_estado(dias):
    if dias <= 6: return "s1"
    elif dias <= 12: return "s2"
    elif dias <= 18: return "s3"
    else: return "s4"

def aplicar_tratamientos(df_emer, df_trat, ensayo_id):
    sub = df_trat[df_trat["ensayo_id"] == ensayo_id].dropna(subset=["tipo"])
    if sub.empty:
        return df_emer
    df = df_emer.copy()
    for _, t in sub.iterrows():
        if pd.isna(t["fecha_aplicacion"]): 
            continue
        t_ini = pd.to_datetime(t["fecha_aplicacion"])
        t_fin = t_ini + pd.to_timedelta(t.get("residual_dias", 0), unit="D")
        eficacia = (1 - t["eficacia_pct"] / 100.0)
        for s in ["s1","s2","s3","s4"]:
            if t.get(f"actua_{s}",0) == 1:
                mask = (df["estado"]==s) & (df["fecha"]>=t_ini) & (df["fecha"]<=t_fin)
                df.loc[mask,"emer_rel"] *= eficacia
    return df

def compute_effective_density(ens_df, emer_df, trat_df):
    results = []
    for _, row in ens_df.iterrows():
        ensayo = row["ensayo_id"]
        pc_ini = pd.to_datetime(row["pc_ini"])
        pc_fin = pd.to_datetime(row["pc_fin"])
        Ca, Cs = row["ca"], row["cs"]

        df_emer = emer_df[emer_df["ensayo_id"] == ensayo].copy()
        if df_emer.empty:
            results.append((ensayo, 0))
            continue

        df_emer["fecha"] = pd.to_datetime(df_emer["fecha"])
        df_emer = df_emer[(df_emer["fecha"] >= pc_ini) & (df_emer["fecha"] <= pc_fin)].copy()

        sow_date = pd.to_datetime(row["fecha_siembra"])
        df_emer["dias_desde_siembra"] = (df_emer["fecha"] - sow_date).dt.days
        df_emer["estado"] = df_emer["dias_desde_siembra"].apply(asignar_estado)

        df_emer = aplicar_tratamientos(df_emer, trat_df, ensayo)

        df_emer["peso_estado"] = df_emer["estado"].map(W_ESTADOS)
        df_emer["emer_pond"] = df_emer["emer_rel"] * df_emer["peso_estado"]

        dens_eff = df_emer["emer_pond"].sum() * (Ca / Cs)
        results.append((ensayo, dens_eff))
    return pd.DataFrame(results, columns=["ensayo_id", "dens_eff"])

def plot_obs_vs_pred(df, title):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["loss_obs_pct"], df["loss_pred_pct"], alpha=0.7)
    ax.plot([0,100],[0,100],'r--')
    ax.set_xlabel("Pérdida observada (%)")
    ax.set_ylabel("Pérdida predicha (%)")
    ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

# -------------------
# Interfaz Streamlit
# -------------------
st.set_page_config(page_title="Calibración + Testeo (S1–S4 + PRE/POST)", layout="wide")
st.title("🌱 Calibración y Testeo de Pérdida de Rinde")
st.write("Modelo con estados fijos (S1–S4), tratamientos PRE/POST con residualidad, y curva hiperbólica calibrada.")

st.sidebar.header("📂 Subir archivos")
calibra_file = st.sidebar.file_uploader("Archivo de calibración (.xlsx)", type="xlsx")
testeo_file = st.sidebar.file_uploader("Archivo de testeo (.xlsx)", type="xlsx")

if calibra_file and testeo_file:
    ens_cal = normalize_columns(pd.read_excel(calibra_file, sheet_name="ensayos"))
    emer_cal = normalize_columns(pd.read_excel(calibra_file, sheet_name="emergencia"))
    trat_cal = normalize_columns(pd.read_excel(calibra_file, sheet_name="tratamientos"))

    ens_test = normalize_columns(pd.read_excel(testeo_file, sheet_name="ensayos"))
    emer_test = normalize_columns(pd.read_excel(testeo_file, sheet_name="emergencia"))
    trat_test = normalize_columns(pd.read_excel(testeo_file, sheet_name="tratamientos"))

    # Depuración rápida
    st.write("📑 Columnas en emergencia (test):", emer_test.columns.tolist())

    # Calcular densidad efectiva
    dens_cal = compute_effective_density(ens_cal, emer_cal, trat_cal)
    dens_test = compute_effective_density(ens_test, emer_test, trat_test)

    ens_cal = ens_cal.merge(dens_cal, on="ensayo_id")
    ens_test = ens_test.merge(dens_test, on="ensayo_id")

    # Predicciones
    ens_cal["loss_pred_pct"] = loss_pred(ens_cal["dens_eff"])
    ens_test["loss_pred_pct"] = loss_pred(ens_test["dens_eff"])

    # Métricas
    metrics_cal = eval_metrics(ens_cal["loss_obs_pct"], ens_cal["loss_pred_pct"])
    metrics_test = eval_metrics(ens_test["loss_obs_pct"], ens_test["loss_pred_pct"])

    st.subheader("📊 Resultados")
    st.write("**Métricas de Calibración:**", metrics_cal)
    st.write("**Métricas de Testeo:**", metrics_test)

    # Gráficos
    col1, col2 = st.columns(2)
    with col1: plot_obs_vs_pred(ens_cal, "Calibración (Entrenamiento)")
    with col2: plot_obs_vs_pred(ens_test, "Testeo (Validación)")

    # Descarga
    st.subheader("⬇️ Descargar resultados")
    out = pd.concat([ens_cal.assign(dataset="calibración"), ens_test.assign(dataset="testeo")])
    buffer = BytesIO()
    out.to_csv(buffer, index=False)
    st.download_button("Descargar CSV", data=buffer.getvalue(),
                       file_name="resultados_calibra_testeo.csv",
                       mime="text/csv")
else:
    st.info("Subí ambos archivos Excel (calibración y testeo) desde la barra lateral.")


