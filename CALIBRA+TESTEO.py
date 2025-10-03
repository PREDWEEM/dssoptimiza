# -*- coding: utf-8 -*-
# Validación y Testeo del modelo de pérdida de rinde

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------
# Parámetros calibrados
# -------------------
ALPHA = 0.9782
LMAX = 83.77

# -------------------
# Función de pérdida hiperbólica
# -------------------
def loss_pred(x, alpha=ALPHA, Lmax=LMAX):
    return Lmax * ((alpha * x) / (1 + alpha * x))

# -------------------
# Cargar datos
# -------------------
calibra_file = "calibra borde.xlsx"
testeo_file = "testeo.xlsx"

# Hojas
ens_cal = pd.read_excel(calibra_file, sheet_name="ensayos")
emer_cal = pd.read_excel(calibra_file, sheet_name="emergencia")
trat_cal = pd.read_excel(calibra_file, sheet_name="tratamientos")

ens_test = pd.read_excel(testeo_file, sheet_name="ensayos")
emer_test = pd.read_excel(testeo_file, sheet_name="emergencia")
trat_test = pd.read_excel(testeo_file, sheet_name="tratamientos")

# -------------------
# Función para calcular densidad efectiva en PC
# -------------------
def compute_effective_density(ens_df, emer_df):
    """
    Simplificación: suma de emergencias relativas en el PC * Ca/Cs
    (aquí podés insertar la lógica detallada de pesos S1-S4 y tratamientos)
    """
    results = []
    for _, row in ens_df.iterrows():
        ensayo = row["ensayo_id"]
        pc_ini = pd.to_datetime(row["pc_ini"])
        pc_fin = pd.to_datetime(row["pc_fin"])
        Ca, Cs = row["Ca"], row["Cs"]

        # Emergencias del ensayo en PC
        df_emer = emer_df[emer_df["ensayo_id"] == ensayo].copy()
        df_emer["fecha"] = pd.to_datetime(df_emer["fecha"])
        df_emer = df_emer[(df_emer["fecha"] >= pc_ini) & (df_emer["fecha"] <= pc_fin)]

        # Densidad efectiva: suma de emergencia relativa × capacidad competitiva
        dens_eff = df_emer["emer_rel"].sum() * (Ca / Cs)

        results.append((ensayo, dens_eff))
    return pd.DataFrame(results, columns=["ensayo_id", "dens_eff"])

# -------------------
# Calcular densidad efectiva
# -------------------
dens_cal = compute_effective_density(ens_cal, emer_cal)
dens_test = compute_effective_density(ens_test, emer_test)

# Merge con ensayos
ens_cal = ens_cal.merge(dens_cal, on="ensayo_id")
ens_test = ens_test.merge(dens_test, on="ensayo_id")

# -------------------
# Predicciones
# -------------------
ens_cal["loss_pred_pct"] = loss_pred(ens_cal["dens_eff"])
ens_test["loss_pred_pct"] = loss_pred(ens_test["dens_eff"])

# -------------------
# Métricas
# -------------------
def eval_metrics(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

metrics_cal = eval_metrics(ens_cal["loss_obs_pct"], ens_cal["loss_pred_pct"])
metrics_test = eval_metrics(ens_test["loss_obs_pct"], ens_test["loss_pred_pct"])

print("Métricas CALIBRACIÓN:", metrics_cal)
print("Métricas TESTEO:", metrics_test)

# -------------------
# Gráficos
# -------------------
def plot_obs_vs_pred(df, title):
    plt.figure(figsize=(6,6))
    plt.scatter(df["loss_obs_pct"], df["loss_pred_pct"], alpha=0.7)
    plt.plot([0,100], [0,100], 'r--')
    plt.xlabel("Pérdida observada (%)")
    plt.ylabel("Pérdida predicha (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_obs_vs_pred(ens_cal, "Calibración (Entrenamiento)")
plot_obs_vs_pred(ens_test, "Testeo (Validación)")

