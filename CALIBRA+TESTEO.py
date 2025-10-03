# -*- coding: utf-8 -*-
# Streamlit — Calibración/Testeo con un único Excel (dataset marcado en hoja ensayos)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Calibración Hipérbolica (Excel único)", layout="wide")
st.title("📈 Calibración + Testeo desde un solo Excel")

# ----------------------------
# Parámetros fijos
# ----------------------------
W_ESTADOS = {"s1": 0.25, "s2": 0.50, "s3": 0.75, "s4": 1.0}

def loss_function(x, alpha, Lmax):
    return (alpha * Lmax * x) / (alpha + x)

def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# ----------------------------
# Tratamientos
# ----------------------------
def aplicar_tratamientos(sub, trat_df, ensayo):
    trats = trat_df[trat_df["ensayo_id"]==ensayo].dropna(subset=["tipo"])
    if trats.empty:
        return sub, []
    log = []
    for _, t in trats.iterrows():
        if pd.isna(t["fecha_aplicacion"]): continue
        f_ini = pd.to_datetime(t["fecha_aplicacion"])
        f_fin = f_ini + pd.to_timedelta(int(t["residual_dias"] if not pd.isna(t["residual_dias"]) else 0), unit="D")
        mask = (sub["fecha"]>=f_ini) & (sub["fecha"]<=f_fin if f_fin>f_ini else f_ini)
        if not mask.any(): continue
        estados_afectados = []
        for estado in ["s1","s2","s3","s4"]:
            if t.get(f"actua_{estado}",0)==1:
                col = f"emer_{estado}"
                if col in sub.columns:
                    sub.loc[mask, col] *= (1 - t["eficacia_pct"]/100.0)
                    estados_afectados.append(estado)
        log.append({
            "ensayo_id": ensayo, "tipo": t["tipo"], "fecha": f_ini.date(),
            "eficacia": t["eficacia_pct"], "residual_dias": t["residual_dias"],
            "estados": ",".join(estados_afectados), "dias_afectados": mask.sum()
        })
    return sub, log

# ----------------------------
# Calcular densidad efectiva
# ----------------------------
def calcular_densidad(emer_df, trat_df, ensayo, Ca, Cs, pc_ini, pc_fin):
    sub = emer_df[emer_df["ensayo_id"]==ensayo].copy()
    sub["fecha"] = pd.to_datetime(sub["fecha"])
    sub = sub[(sub["fecha"]>=pd.to_datetime(pc_ini)) & (sub["fecha"]<=pd.to_datetime(pc_fin))]

    # fallback si no están separados los estados
    if not all(f"emer_{s}" in sub.columns for s in ["s1","s2","s3","s4"]):
        for s in ["s1","s2","s3","s4"]:
            sub[f"emer_{s}"] = sub["emer_rel"]/4

    # aplicar tratamientos
    sub, log = aplicar_tratamientos(sub, trat_df, ensayo)

    dens = sum((sub[f"emer_{s}"] * W_ESTADOS[s]).sum() for s in W_ESTADOS)
    dens_eff = dens * (Ca/Cs)
    for l in log: l["dens_eff"] = dens_eff
    return dens_eff, log

# ----------------------------
# App principal
# ----------------------------
file = st.file_uploader("📂 Subí un único Excel (con hojas ensayos, emergencia, tratamientos)", type=["xlsx"])

if file:
    ensayos = pd.read_excel(file, sheet_name="ensayos")
    emergencia = pd.read_excel(file, sheet_name="emergencia")
    tratamientos = pd.read_excel(file, sheet_name="tratamientos")

    resultados, log_all = [], []
    for _, row in ensayos.iterrows():
        dens_eff, log = calcular_densidad(emergencia, tratamientos,
                                          row["ensayo_id"], row["Ca"], row["Cs"],
                                          row["pc_ini"], row["pc_fin"])
        resultados.append({
            "ensayo_id": row["ensayo_id"],
            "dens_eff": dens_eff,
            "loss_obs_pct": row["loss_obs_pct"],
            "dataset": row["dataset"]
        })
        log_all.extend(log)

    df = pd.DataFrame(resultados)
    log_df = pd.DataFrame(log_all)

    st.subheader("📊 Dataset generado")
    st.write(df.head())

    st.subheader("📝 Diagnóstico tratamientos")
    if log_df.empty: st.info("No se aplicaron tratamientos")
    else: st.dataframe(log_df)

    # Optimización solo con calibración
    def objective(params):
        alpha, Lmax = params
        y_pred = loss_function(df[df["dataset"]=="calibración"]["dens_eff"].values, alpha, Lmax)
        return rmse(df[df["dataset"]=="calibración"]["loss_obs_pct"].values, y_pred)

    res = minimize(objective, [1.0, 80.0], bounds=[(1e-6,None),(1e-6,None)])
    alpha_opt, Lmax_opt = res.x
    st.success(f"✅ α={alpha_opt:.4f}, Lmax={Lmax_opt:.2f}")

    # Métricas
    metrics = {}
    for name in ["calibración","testeo"]:
        subset = df[df["dataset"]==name]
        if subset.empty: continue
        y_true = subset["loss_obs_pct"].values
        y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
        metrics[name] = {"RMSE": rmse(y_true,y_pred), "MAE": mae(y_true,y_pred), "R2": r2(y_true,y_pred)}
    st.write(pd.DataFrame(metrics).T.round(2))

    # Gráfico
    fig, ax = plt.subplots(figsize=(7,6))
    x_grid = np.linspace(0, max(df["dens_eff"])*1.2, 200)
    y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
    ax.plot(x_grid, y_grid, 'b-', label=f"Hiperbólica ajustada\nα={alpha_opt:.3f}, Lmax={Lmax_opt:.1f}")
    if not df[df["dataset"]=="calibración"].empty:
        ax.scatter(df[df["dataset"]=="calibración"]["dens_eff"],
                   df[df["dataset"]=="calibración"]["loss_obs_pct"],
                   c="green", marker="o", label="Calibración")
    if not df[df["dataset"]=="testeo"].empty:
        ax.scatter(df[df["dataset"]=="testeo"]["dens_eff"],
                   df[df["dataset"]=="testeo"]["loss_obs_pct"],
                   c="red", marker="s", label="Testeo")
    ax.set_xlabel("Densidad efectiva (x)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

    # Descargar
    df["loss_pred_opt"] = loss_function(df["dens_eff"].values, alpha_opt, Lmax_opt)
    st.download_button("⬇️ Descargar resultados optimizados (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "resultados_optimizados.csv", "text/csv")
