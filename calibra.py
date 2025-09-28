# -*- coding: utf-8 -*-
# Script de calibración PREDWEEM con datos independientes
# Autor: GUILLERMO + ChatGPT
# --------------------------------------------------------
# Lee calibra borde.xlsx, procesa ensayos, emergencias y tratamientos,
# simula densidad efectiva controlada (A2_ctrl) y calibra la función de pérdida
# Parámetros a ajustar: alpha, Lmax

import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize

# === 1. Cargar datos ===
file_path = "calibra borde.xlsx"
df_ensayos = pd.read_excel(file_path, sheet_name="ensayos")
df_trat = pd.read_excel(file_path, sheet_name="tratamientos")
df_emerg = pd.read_excel(file_path, sheet_name="emergencia")

# === 2. Funciones auxiliares ===

def interpolate_emergencia(sub_emerg, fecha_siembra, fecha_pcfin):
    """Interpola emergencias a paso diario desde siembra hasta fin de PC."""
    fechas = pd.date_range(start=fecha_siembra, end=fecha_pcfin, freq="D")
    serie = sub_emerg.set_index("fecha").reindex(fechas).interpolate(method="linear").fillna(0)
    serie = serie.rename(columns={"emer_rel": "emer_rel"})
    serie.index.name = "fecha"
    return serie

def asignar_cohortes(df):
    """Divide emergencia en cohortes S1..S4 según edad desde siembra."""
    dias = np.arange(len(df))
    # Definiciones tentativas de cohortes (ajustables)
    S1 = df.emer_rel.rolling(6, min_periods=1).sum()
    S2 = df.emer_rel.rolling(21, min_periods=1).sum().shift(7, fill_value=0)
    S3 = df.emer_rel.rolling(32, min_periods=1).sum().shift(28, fill_value=0)
    S4 = df.emer_rel.cumsum().shift(60, fill_value=0)
    return pd.DataFrame({"S1": S1, "S2": S2, "S3": S3, "S4": S4}, index=df.index)

def aplicar_tratamientos(df_states, ens_id, fecha_siembra):
    """Aplica reducciones de tratamientos sobre cohortes según reglas."""
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
    """Función de pérdida (%)."""
    return (alpha * x) / (1 + (alpha * x / Lmax))

def simular_ensayo(row, df_emerg, alpha, Lmax):
    """Simula un ensayo: calcula A2_ctrl y pérdida predicha."""
    ens_id = row.ensayo_id
    f_siembra = pd.to_datetime(row.fecha_siembra)
    f_pcfin = pd.to_datetime(row.pc_fin)

    # 1. Emergencia interpolada
    emerg = df_emerg[df_emerg.ensayo_id == ens_id][["fecha", "emer_rel"]]
    serie = interpolate_emergencia(emerg, f_siembra, f_pcfin)

    # 2. Cohortes
    states = asignar_cohortes(serie)

    # 3. Tratamientos
    states_ctrl = aplicar_tratamientos(states, ens_id, f_siembra)

    # 4. Densidad efectiva (simple suma, normalizada al tope A2)
    A2_ctrl = states_ctrl.sum(axis=1).max()
    # 5. Pérdida predicha
    loss_pred = perdida_modelo(A2_ctrl, alpha, Lmax)
    return loss_pred

# === 3. Función objetivo de calibración ===
def objective(params):
    alpha, Lmax = params
    preds, obs = [], []
    for _, row in df_ensayos.iterrows():
        pred = simular_ensayo(row, df_emerg, alpha, Lmax)
        preds.append(pred)
        obs.append(row.loss_obs_pct)
    preds = np.array(preds)
    obs = np.array(obs)
    return np.sqrt(np.mean((obs - preds) ** 2))  # RMSE

# === 4. Optimización ===
x0 = [0.3, 80.0]  # valores iniciales para alpha y Lmax
res = minimize(objective, x0, bounds=[(0.01, 2.0), (10, 200)], method="L-BFGS-B")

alpha_best, Lmax_best = res.x
print("Mejores parámetros encontrados:")
print(f"α = {alpha_best:.3f}, Lmax = {Lmax_best:.2f}")
print(f"RMSE = {res.fun:.2f}")

