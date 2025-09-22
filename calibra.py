import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -----------------------------
# 1. Cargar datos de ensayos
# -----------------------------
df = pd.read_csv("ensayos.csv")
df["loss_obs"] = 100 * (df["rinde_clean"] - df["rinde_obs"]) / df["rinde_clean"]

# -----------------------------
# 2. Definir modelo (ejemplo simplificado)
# -----------------------------
def simular_perdida(fecha_siembra, tratamientos, params):
    # params = [FC_S2, FC_S3, FC_S4, MAX_PLANTS_CAP, Imax, A50]
    FC = {"S1": 0.0, "S2": params[0], "S3": params[1], "S4": params[2]}
    MAX_CAP = params[3]
    Imax, A50 = params[4], params[5]

    # aquí deberías llamar a tu simulador PREDWEEM
    # salida: densidad efectiva (eq·pl·m²) durante PCI
    dens_efectiva = run_predweem(fecha_siembra, tratamientos, FC, MAX_CAP)

    # curva de respuesta hiperbólica
    loss_pred = (Imax * dens_efectiva) / (A50 + dens_efectiva)
    return loss_pred

# -----------------------------
# 3. Función objetivo para calibrar
# -----------------------------
def objective(params):
    losses_pred = []
    for _, row in df.iterrows():
        pred = simular_perdida(row["fecha_siembra"], row["tratamientos"], params)
        losses_pred.append(pred)
    losses_pred = np.array(losses_pred)
    return np.mean((df["loss_obs"].values - losses_pred)**2)  # RMSE²

# -----------------------------
# 4. Optimización
# -----------------------------
# valores iniciales: [FC_S2, FC_S3, FC_S4, MAX_CAP, Imax, A50]
x0 = [0.3, 0.6, 1.0, 500, 80, 150]

result = minimize(objective, x0, bounds=[
    (0,1), (0,1), (0.8,1.2), (100,1000), (50,100), (50,500)
])

print("Parámetros calibrados:", result.x)
print("Error cuadrático medio:", result.fun**0.5)

