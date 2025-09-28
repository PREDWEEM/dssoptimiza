# -*- coding: utf-8 -*-
# calibra_app.py — Calibración PREDWEEM con carga de Excel vía Streamlit

import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize

# === 1. Interfaz de carga ===
st.title("Calibración del modelo PREDWEEM")
st.write("Subí un archivo Excel con hojas: **ensayos**, **tratamientos**, **emergencia**")

uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # === 2. Leer hojas ===
    df_ensayos = pd.read_excel(uploaded_file, sheet_name="ensayos")
    df_trat = pd.read_excel(uploaded_file, sheet_name="tratamientos")
    df_emerg = pd.read_excel(uploaded_file, sheet_name="emergencia")

    st.success("Archivo cargado correctamente ✅")

    # === 3. Funciones auxiliares ===
    def interpolate_emergencia(sub_emerg, fecha_siembra, fecha_pcfin):
        fechas = pd.date_range(start=fecha_siembra, end=fecha_pcfin, freq="D")
        serie = sub_emerg.set_index("fecha").reindex(fechas).interpolate(method="linear").fillna(0)
        serie = serie.rename(columns={"emer_rel": "emer_rel"})
        serie.index.name = "fecha"
        return serie

    def asignar_cohortes(df):
        S1 = df.emer_rel.rolling(6, min_periods=1).sum()
        S2 = df.emer_rel.rolling(21, min_periods=1).sum().shift(7, fill_value=0)
        S3 = df.emer_rel.rolling(32, min_periods=1).sum().shift(28, fill_value=0)
        S4 = df.emer_rel.cumsum().shift(60, fill_value=0)
        return pd.DataFrame({"S1": S1, "S2": S2, "S3": S3, "S4": S4}, index=df.index)

    def aplicar_tratamientos(df_states, ens_id, fecha_siembra):
        sub_trat = df_trat[df_trat.ensayo_id == ens_id].dropna(subset=["tipo"])
        df_ctrl = df_states.copy()
        for _, row in sub_trat.iterrows():
            f_app = pd.to_datetime(row["fecha_aplicacion"])
            ef = row["eficacia_pct"] / 100.0
            res = int(row.get("residual_dias", 0)) if not pd.isna(row.get("residual_dias")) else 0
            mask = (df_ctrl.index >= f_app) & (df_ctrl.index <= f_app + pd.Timedelta(days=res))
            for s in ["S1", "S2", "S3", "S4"]:
                if row.get(f"actua_{s.lower()}", 0) == 1:
                    df_ctrl.loc[mask, s] *= (1 - ef)
        return df_ctrl

    def perdida_modelo(x, alpha, Lmax):
        return (alpha * x) / (1 + (alpha * x / Lmax))

    def simular_ensayo(row, alpha, Lmax):
        ens_id = row.ensayo_id
        f_siembra = pd.to_datetime(row.fecha_siembra)
        f_pcfin = pd.to_datetime(row.pc_fin)

        emerg = df_emerg[df_emerg.ensayo_id == ens_id][["fecha", "emer_rel"]]
        serie = interpolate_emergencia(emerg, f_siembra, f_pcfin)

        states = asignar_cohortes(serie)
        states_ctrl = aplicar_tratamientos(states, ens_id, f_siembra)

        A2_ctrl = states_ctrl.sum(axis=1).max()
        return perdida_modelo(A2_ctrl, alpha, Lmax)

    def objective(params):
        alpha, Lmax = params
        preds, obs = [], []
        for _, row in df_ensayos.iterrows():
            pred = simular_ensayo(row, alpha, Lmax)
            preds.append(pred)
            obs.append(row.loss_obs_pct)
        return np.sqrt(np.mean((np.array(obs) - np.array(preds)) ** 2))

    # === 4. Botón de calibración ===
    if st.button("Ejecutar calibración"):
        x0 = [0.3, 80.0]
        res = minimize(objective, x0, bounds=[(0.01, 2.0), (10, 200)], method="L-BFGS-B")

        alpha_best, Lmax_best = res.x
        st.subheader("Resultados de calibración")
        st.write(f"**α = {alpha_best:.3f}**")
        st.write(f"**Lmax = {Lmax_best:.2f}**")
        st.write(f"**RMSE = {res.fun:.2f}**")

        # Comparación observado vs predicho
        df_results = df_ensayos.copy()
        df_results["predicho"] = [
            simular_ensayo(row, alpha_best, Lmax_best) for _, row in df_ensayos.iterrows()
        ]
        st.write("### Comparación Observado vs Predicho")
        st.dataframe(df_results[["ensayo_id", "loss_obs_pct", "predicho"]])

        st.line_chart(df_results[["loss_obs_pct", "predicho"]].set_index(df_results.ensayo_id))

