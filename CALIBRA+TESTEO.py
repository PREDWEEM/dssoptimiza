# -*- coding: utf-8 -*-
# Streamlit — Validación + Testeo desde UN solo Excel
# Hojas: ensayos, emergencia, tratamientos

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------ Config ------------------
st.set_page_config(page_title="Validación + Testeo (Excel único)", layout="wide")
st.title("📈 Validación + Testeo con tratamientos y estados (S1–S4)")

# Pesos fijos por estado
W_ESTADOS = {"s1": 0.25, "s2": 0.50, "s3": 0.75, "s4": 1.0}
# Duraciones (días) de cada estado (no solapados por edad)
DUR = {"s1": 7, "s2": 21, "s3": 32, "s4": 60}  # S4 se manejará como >=60 d (dinámico)

# ------------------ Utilidades ------------------
def normalize_columns(df):
    if df is None:
        return None
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("á","a").str.replace("é","e").str.replace("í","i").str.replace("ó","o").str.replace("ú","u")
    )
    return df

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def to_float_safe(s):
    return pd.to_numeric(s, errors="coerce")

def loss_function(x, alpha, Lmax):
    x = np.asarray(x, float)
    return (alpha * Lmax * x) / (alpha + x + 1e-12)

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def daily_series(sub_emerg, start, end):
    """Construye serie diaria (reindexa/interpola) de emergencia relativa [0..1]."""
    idx = pd.date_range(start=start, end=end, freq="D")
    s = (sub_emerg.set_index("fecha")["emer_rel"]
         .reindex(idx)
         .interpolate("linear")
         .fillna(0.0)
    )
    s.index.name = "fecha"
    return s

def non_overlapping_states(emer_daily: pd.Series) -> pd.DataFrame:
    """Estados por edad no solapados: S1=0–6, S2=7–27, S3=28–59, S4>=60 días."""
    v = emer_daily.to_numpy(float)
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])

    def sum_window(i, a, b):  # suma v[j] para j in [i-b, i-a]
        lo, hi = i - b, i - a
        if hi < 0: return 0.0
        lo_c, hi_c = max(0, lo), min(n - 1, hi)
        if lo_c > hi_c: return 0.0
        return c[hi_c + 1] - c[lo_c]

    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for i in range(n):
        S1[i] = sum_window(i, 0, 6)     # 0..6
        S2[i] = sum_window(i, 7, 27)    # 7..27
        S3[i] = sum_window(i, 28, 59)   # 28..59
        j = i - 60                       # >=60
        S4[i] = c[j + 1] if j >= 0 else 0.0
    return pd.DataFrame({"s1": S1, "s2": S2, "s3": S3, "s4": S4}, index=emer_daily.index)

def effective_T4_days(sow_date, pc_ini, pc_fin) -> int:
    start_S4 = max(pc_ini, sow_date + pd.Timedelta(days=60))
    if start_S4 > pc_fin:
        return 1
    return max(1, (pc_fin - start_S4).days + 1)

def aplicar_tratamientos(s_states: pd.DataFrame, idx: pd.DatetimeIndex,
                         trts_sub: pd.DataFrame, log_list: list, ensayo_id: str):
    """Aplica POST (por estados) y PRE (sobre emergencia -> aquí se refleja en estados por ventana temporal)
       usando residualidad. Se implementa como multiplicador por estado y día."""
    if trts_sub is None or trts_sub.empty:
        return s_states

    # Control multiplicativo por estado
    ctrl = {k: np.ones(len(idx), dtype=float) for k in ["s1","s2","s3","s4"]}

    for _, t in trts_sub.iterrows():
        f_ap = to_datetime_safe(t.get("fecha_aplicacion"))
        if pd.isna(f_ap): 
            continue
        e = float(to_float_safe(t.get("eficacia_pct")))
        e = 0.0 if np.isnan(e) else e
        r = int(to_float_safe(t.get("residual_dias")))
        r = 0 if np.isnan(r) else r

        mask = (idx >= f_ap) & (idx <= (f_ap + pd.Timedelta(days=r)))
        a1 = int(to_float_safe(t.get("actua_s1"))) == 1 if "actua_s1" in t.index else True
        a2 = int(to_float_safe(t.get("actua_s2"))) == 1 if "actua_s2" in t.index else True
        a3 = int(to_float_safe(t.get("actua_s3"))) == 1 if "actua_s3" in t.index else True
        a4 = int(to_float_safe(t.get("actua_s4"))) == 1 if "actua_s4" in t.index else True

        for k, act in zip(["s1","s2","s3","s4"], [a1,a2,a3,a4]):
            if act:
                ctrl[k][mask] *= (1.0 - e/100.0)

        log_list.append({
            "ensayo_id": ensayo_id,
            "tipo": str(t.get("tipo","")),
            "fecha_aplicacion": f_ap.date(),
            "eficacia_pct": e,
            "residual_dias": r,
            "estados": ",".join([k for k,act in zip(["s1","s2","s3","s4"],[a1,a2,a3,a4]) if act]),
            "dias_afectados": int(mask.sum())
        })

    S = s_states.copy()
    for k in ["s1","s2","s3","s4"]:
        S[k] = S[k].to_numpy(float) * ctrl[k]
    return S

def auc(idx: pd.DatetimeIndex, values: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    t = (idx - idx[0]).days.astype(float)
    return float(np.trapz(np.asarray(values, float), t))

def calcular_densidad_ensayo(ens_row, emer_df, trts_df, log_list):
    ensayo = str(ens_row["ensayo_id"])
    f_sow  = to_datetime_safe(ens_row["fecha_siembra"])
    pc_ini = to_datetime_safe(ens_row["pc_ini"])
    pc_fin = to_datetime_safe(ens_row["pc_fin"])
    Ca     = float(to_float_safe(ens_row.get("ca", np.nan)))
    Cs     = float(to_float_safe(ens_row.get("cs", np.nan)))
    if np.isnan(Ca): Ca = 250.0
    if np.isnan(Cs): Cs = 250.0

    # Sub-emergencia del ensayo
    sub = emer_df[emer_df["ensayo_id"] == ensayo][["fecha","emer_rel"]].copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_ini) or pd.isna(pc_fin):
        return np.nan, {}

    sub["fecha"] = to_datetime_safe(sub["fecha"])
    sub["emer_rel"] = to_float_safe(sub["emer_rel"]).fillna(0.0)
    # Si viene en %, pasar a fracción
    if sub["emer_rel"].max() > 1.5:
        sub["emer_rel"] = sub["emer_rel"]/100.0
    sub["emer_rel"] = np.clip(sub["emer_rel"], 0.0, 1.0)

    # Serie diaria desde siembra hasta fin de PC
    s = daily_series(sub, f_sow, pc_fin)
    idx = s.index

    # Estados base por edad (no solapados) a partir de emergencia diaria
    states = non_overlapping_states(s)

    # Aplicar tratamientos (POST/S por estados; PRE queda reflejado por fechas tempranas)
    trts_sub = trts_df[trts_df["ensayo_id"] == ensayo].copy() if trts_df is not None else None
    states = aplicar_tratamientos(states, idx, trts_sub, log_list, ensayo)

    # T4 (dinámico dentro del PC) para normalizar por duración
    T4 = effective_T4_days(f_sow, pc_ini, pc_fin)

    # Densidad diaria ponderada por pesos / duraciones
    D = (
        states["s1"].to_numpy(float) * W_ESTADOS["s1"] / DUR["s1"] +
        states["s2"].to_numpy(float) * W_ESTADOS["s2"] / DUR["s2"] +
        states["s3"].to_numpy(float) * W_ESTADOS["s3"] / DUR["s3"] +
        states["s4"].to_numpy(float) * W_ESTADOS["s4"] / max(1, T4)
    )

    # Integrar solo dentro del PC
    mask_pc = (idx >= pc_ini) & (idx <= pc_fin)
    dens_states = auc(idx[mask_pc], D[mask_pc])

    # Ajuste por Ca/Cs
    dens_eff = dens_states * (Ca / max(1e-6, Cs))

    diag = {
        "ensayo_id": ensayo,
        "Ca": Ca, "Cs": Cs,
        "T4_efectivo_dias": T4,
        "AUC_states_PC": dens_states,
        "dens_eff": dens_eff
    }
    return dens_eff, diag

# ------------------ Interfaz ------------------
file = st.file_uploader("📂 Subí el Excel con hojas: ensayos, emergencia, tratamientos", type=["xlsx"])

if not file:
    st.info("Subí el archivo Excel para comenzar.")
    st.stop()

# Leer hojas y normalizar columnas
ensayos = normalize_columns(pd.read_excel(file, sheet_name="ensayos"))
emerg   = normalize_columns(pd.read_excel(file, sheet_name="emergencia"))
trts    = normalize_columns(pd.read_excel(file, sheet_name="tratamientos"))

# Renombrar clave típica: 'calibracion' -> 'calibración'
if "dataset" in ensayos.columns:
    ensayos["dataset"] = ensayos["dataset"].astype(str).str.strip().str.lower()
    ensayos["dataset"] = ensayos["dataset"].replace({"calibracion":"calibración","validacion":"testeo"})

# Validación mínima de columnas críticas
need_ens = {"ensayo_id","fecha_siembra","pc_ini","pc_fin","loss_obs_pct","dataset"}
need_em  = {"ensayo_id","fecha","emer_rel"}
miss_ens = [c for c in need_ens if c not in ensayos.columns]
miss_em  = [c for c in need_em if c not in emerg.columns]
if miss_ens or miss_em:
    st.error(f"Faltan columnas.\n'ensayos' faltan: {miss_ens}\n'emergencia' faltan: {miss_em}")
    st.stop()

# Coerción de tipos
for c in ["fecha_siembra","pc_ini","pc_fin"]:
    ensayos[c] = to_datetime_safe(ensayos[c])
for c in ["ca","cs","loss_obs_pct"]:
    if c in ensayos.columns:
        ensayos[c] = to_float_safe(ensayos[c])
emerg["fecha"] = to_datetime_safe(emerg["fecha"])
emerg["emer_rel"] = to_float_safe(emerg["emer_rel"])

# ------------------ Cálculo densidad por ensayo ------------------
rows = []
diag_list = []
trt_logs = []
for _, row in ensayos.iterrows():
    dens_eff, diag = calcular_densidad_ensayo(row, emerg, trts, trt_logs)
    rows.append({
        "ensayo_id": row["ensayo_id"],
        "dataset": row["dataset"],
        "loss_obs_pct": float(row["loss_obs_pct"]) if not pd.isna(row["loss_obs_pct"]) else np.nan,
        "dens_eff": dens_eff
    })
    if diag: diag_list.append(diag)

df = pd.DataFrame(rows)
diag_df = pd.DataFrame(diag_list)
log_df  = pd.DataFrame(trt_logs)

st.subheader("📊 Dataset armado (dens_eff + observado)")
st.dataframe(df, use_container_width=True)

with st.expander("🛠️ Diagnóstico por ensayo (AUC/estados, T4, Ca/Cs)"):
    if diag_df.empty: st.write("Sin diagnóstico disponible.")
    else: st.dataframe(diag_df, use_container_width=True)

with st.expander("📝 Log de tratamientos aplicados"):
    if log_df.empty: st.write("No se aplicaron tratamientos / no hay registros.")
    else: st.dataframe(log_df, use_container_width=True)

# ------------------ Optimización α y Lmax (solo calibración) ------------------
calib = df[df["dataset"]=="calibración"].dropna(subset=["dens_eff","loss_obs_pct"])
test  = df[df["dataset"]=="testeo"].dropna(subset=["dens_eff","loss_obs_pct"])

if calib.empty:
    st.error("No hay ensayos marcados como 'calibración' con datos completos.")
    st.stop()

def objective(params):
    alpha, Lmax = params
    y_pred = loss_function(calib["dens_eff"].values, alpha, Lmax)
    return rmse(calib["loss_obs_pct"].values, y_pred)

res = minimize(objective, x0=[1.0, 80.0], bounds=[(1e-6,None),(1e-6,None)], method="L-BFGS-B")
alpha_opt, Lmax_opt = map(float, res.x)
st.success(f"✅ Parámetros optimizados: α = {alpha_opt:.4f} · Lmax = {Lmax_opt:.2f}")

# ------------------ Métricas ------------------
def eval_block(name, subset):
    if subset.empty:
        return None
    y_true = subset["loss_obs_pct"].values
    y_pred = loss_function(subset["dens_eff"].values, alpha_opt, Lmax_opt)
    return {"RMSE": rmse(y_true,y_pred), "MAE": mae(y_true,y_pred), "R2": r2(y_true,y_pred)}

metrics = {}
m_cal = eval_block("Calibración", calib)
if m_cal: metrics["Calibración"] = m_cal
m_tst = eval_block("Testeo", test)
if m_tst: metrics["Testeo"] = m_tst

st.subheader("📈 Métricas")
st.write(pd.DataFrame(metrics).T.round(3) if metrics else "Sin métricas calculables.")

# ------------------ Gráfico ------------------
st.subheader("📉 Curva ajustada vs datos")
fig, ax = plt.subplots(figsize=(7,6))
xmax = float(np.nanmax(df["dens_eff"])) if not df["dens_eff"].isna().all() else 1.0
x_grid = np.linspace(0, max(1e-6, xmax)*1.2, 300)
y_grid = loss_function(x_grid, alpha_opt, Lmax_opt)
ax.plot(x_grid, y_grid, label=f"Hiperbólica ajustada\nα={alpha_opt:.3f}, Lmax={Lmax_opt:.1f}")

if not calib.empty:
    ax.scatter(calib["dens_eff"], calib["loss_obs_pct"], c="green", label="Calibración", alpha=0.8)
if not test.empty:
    ax.scatter(test["dens_eff"], test["loss_obs_pct"], c="red", label="Testeo", alpha=0.8)

ax.set_xlabel("Densidad efectiva (x)")
ax.set_ylabel("Pérdida de rinde (%)")
ax.grid(True); ax.legend()
st.pyplot(fig)

# ------------------ Descargas ------------------
st.subheader("⬇️ Descargas")
df["loss_pred_opt"] = loss_function(df["dens_eff"].values, alpha_opt, Lmax_opt)
st.download_button("CSV — Resultados (obs + pred + dens_eff)",
                   df.to_csv(index=False).encode("utf-8"),
                   "resultados_validacion_testeo.csv", "text/csv")

if not diag_df.empty:
    st.download_button("CSV — Diagnóstico por ensayo",
                       diag_df.to_csv(index=False).encode("utf-8"),
                       "diagnostico_por_ensayo.csv", "text/csv")
if not log_df.empty:
    st.download_button("CSV — Log de tratamientos aplicados",
                       log_df.to_csv(index=False).encode("utf-8"),
                       "log_tratamientos.csv", "text/csv")
