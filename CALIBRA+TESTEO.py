import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("resultados_calibra_testeo.csv")

# Separar calibración y testeo
calib = df[df["dataset"]=="calibración"]
test = df[df["dataset"]=="testeo"]

# Función de pérdida hiperbólica
def loss_function(x, alpha, Lmax):
    return (alpha * Lmax * x) / (alpha + x)

# Métricas
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# Función objetivo para optimizar (solo calibración)
def objective(params):
    alpha, Lmax = params
    y_pred = loss_function(calib["dens_eff"].values, alpha, Lmax)
    return rmse(calib["loss_obs_pct"].values, y_pred)

# Optimización inicial
x0 = [1.0, 80.0]
bounds = [(1e-6, None), (1e-6, None)]
res = minimize(objective, x0, bounds=bounds)

alpha_opt, Lmax_opt = res.x
print(f"α óptimo = {alpha_opt:.4f} · Lmax óptimo = {Lmax_opt:.2f}")

# Evaluación y gráficos
plt.figure(figsize=(12,5))

# ---- Gráfico 1: curva hiperbólica ----
plt.subplot(1,2,1)
x_range = np.linspace(0, calib["dens_eff"].max()*1.5, 200)
y_curve = loss_function(x_range, alpha_opt, Lmax_opt)
plt.plot(x_range, y_curve, 'r-', label="Curva ajustada")

plt.scatter(calib["dens_eff"], calib["loss_obs_pct"], c="blue", label="Calibración")
if not test.empty:
    plt.scatter(test["dens_eff"], test["loss_obs_pct"], c="green", marker="s", label="Testeo")

plt.xlabel("Densidad efectiva")
plt.ylabel("Pérdida de rinde (%)")
plt.title("Curva hiperbólica ajustada")
plt.legend()

# ---- Gráfico 2: Observado vs Predicho ----
plt.subplot(1,2,2)

for subset, color, name in [(calib,"blue","Calibración"),(test,"green","Testeo")]:
    if subset.empty: 
        continue
    y_true = subset["loss_obs_pct"].values
    y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
    plt.scatter(y_true, y_pred, c=color, label=name)
    
    print(f"\n{name}:")
    print(f"

