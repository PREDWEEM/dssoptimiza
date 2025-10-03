# -*- coding: utf-8 -*-
# Script: calibra_testeo.py
# Calibración y validación del modelo de pérdida de rinde con densidad efectiva en plantas/m²

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# Función hiperbólica de pérdida de rinde
# ==========================================
def loss_function(x, alpha, Lmax):
    return (alpha * Lmax * x) / (alpha + x)

# ==========================================
# Métricas
# ==========================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# ==========================================
# Función para calcular densidad efectiva
# ==========================================
def calcular_densidad_ensayo(row, emer_df, trat_df, W_ESTADOS, infestacion_escenario=250):
    ensayo_id = row["ensayo_id"]
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    fecha_siembra = pd.to_datetime(row["fecha_siembra"])

    # IC o capacidad máxima de infestación
    IC = row.get("MAX_PLANTS_CAP", np.nan)
    if pd.isna(IC):
        IC = infestacion_escenario

    # Emergencia de ese ensayo
    emer_sub = emer_df[emer_df["ensayo_id"] == ensayo_id].copy()
    if emer_sub.empty:
        return np.nan

    emer_sub["fecha"] = pd.to_datetime(emer_sub["fecha"])
    emer_sub = emer_sub.sort_values("fecha")
    emer_sub["emer_rel"] = emer_sub["emer_rel"].astype(float)

    # Acumuladores por estado
    auc_estados = {k: 0.0 for k in W_ESTADOS.keys()}

    # Clasificación en estados
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

    # Aplicar tratamientos
    trat_sub = trat_df[trat_df["ensayo_id"] == ensayo_id].copy()
    if not trat_sub.empty:
        for _, t in trat_sub.iterrows():
            ef = float(t.get("eficacia_pct", 0) or 0) / 100.0
            if ef <= 0:
                continue
            for est in W_ESTADOS.keys():
                if t.get(f"actua_{est}", 0) == 1:
                    auc_estados[est] *= (1 - ef)

    # Densidad fraccional ponderada
    dens_frac = sum(auc_estados[est] * W_ESTADOS[est] for est in W_ESTADOS.keys())

    # Escalar a plantas/m²
    dens_eff = dens_frac * IC
    return dens_eff

# ==========================================
# Pipeline principal
# ==========================================
def main():
    # Leer archivo Excel único
    FILE = "calibra borde.xlsx"  # <-- cambiar si tu archivo se llama distinto
    ensayos = pd.read_excel(FILE, sheet_name="ensayos")
    emergencia = pd.read_excel(FILE, sheet_name="emergencia")
    tratamientos = pd.read_excel(FILE, sheet_name="tratamientos")

    # Pesos fijos para estados
    W_ESTADOS = {"s1": 0.25, "s2": 0.5, "s3": 0.75, "s4": 1.0}

    # Calcular densidad efectiva para cada ensayo
    resultados = []
    for _, row in ensayos.iterrows():
        dens_eff = calcular_densidad_ensayo(row, emergencia, tratamientos, W_ESTADOS)
        resultados.append({
            "ensayo_id": row["ensayo_id"],
            "dataset": row.get("dataset", "calibracion"),
            "loss_obs_pct": row["loss_obs_pct"],
            "dens_eff": dens_eff
        })

    df_res = pd.DataFrame(resultados).dropna()

    # Separar calibración y testeo
    calib = df_res[df_res["dataset"].str.contains("calib", case=False)]
    test = df_res[df_res["dataset"].str.contains("test", case=False)]

    # Optimizar α y Lmax usando calibración
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(calib["dens_eff"].values, alpha, Lmax)
        return rmse(calib["loss_obs_pct"].values, y_pred)

    x0 = [1.0, 80.0]
    bounds = [(1e-6, None), (1e-6, None)]
    res = minimize(objective, x0, bounds=bounds)
    alpha_opt, Lmax_opt = res.x

    print(f"Parámetros óptimos: α = {alpha_opt:.3f}, Lmax = {Lmax_opt:.2f}")

    # Evaluar en calibración y testeo
    for name, subset in [("Calibración", calib), ("Testeo", test)]:
        if subset.empty:
            continue
        y_true = subset["loss_obs_pct"].values
        y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
        print(f"\n{name}:")
        print(f"  RMSE = {rmse(y_true, y_pred):.2f}")
        print(f"  MAE  = {mae(y_true, y_pred):.2f}")
        print(f"  R²   = {r2(y_true, y_pred):.3f}")

    # Guardar resultados
    df_res["loss_pred_pct"] = loss_function(df_res["dens_eff"].values, alpha_opt, Lmax_opt)
    df_res.to_csv("resultados_final.csv", index=False)
    print("\nResultados guardados en resultados_final.csv")

    # Gráfico
    plt.figure(figsize=(7, 6))
    x_grid = np.linspace(0, df_res["dens_eff"].max() * 1.2, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    plt.plot(x_grid, y_grid, "b-", label=f"Hiperbólica ajustada\nα={alpha_opt:.2f}, Lmax={Lmax_opt:.1f}")

    plt.scatter(calib["dens_eff"], calib["loss_obs_pct"], c="green", label="Calibración (obs)")
    if not test.empty:
        plt.scatter(test["dens_eff"], test["loss_obs_pct"], c="red", marker="s", label="Testeo (obs)")

    plt.xlabel("Densidad efectiva (plantas/m²)")
    plt.ylabel("Pérdida de rinde (%)")
    plt.title("Curva de pérdida vs datos observados")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
