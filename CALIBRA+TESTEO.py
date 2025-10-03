# -*- coding: utf-8 -*-
# Streamlit — Calibración/Testeo con IC + Escenarios de infestación (Alto/Medio/Bajo)
# Un único Excel: hojas 'ensayos', 'emergencia', 'tratamientos'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================
# Configuración
# ============================
st.set_page_config(page_title="Calibración con IC + Escenarios", layout="wide")
st.title("📈 Calibración + Testeo — IC por ensayo + escenarios de infestación")

# Pesos fijos por estado (pueden tunearse)
W_ESTADOS = {"s1": 0.25, "s2": 0.50, "s3": 0.75, "s4": 1.00}
# Duración (días) para normalizar cada estado (S4 dinámico)
DUR = {"s1": 7, "s2": 21, "s3": 32}  # S1=0–6, S2=7–27, S3=28–59 (S4 = >=60)

ESCENARIOS = {"Alto (250)": 250.0, "Medio (125)": 125.0, "Bajo (50)": 50.0}

# ============================
# Utilidades
# ============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("á","a").str.replace("é","e").str.replace("í","i").str.replace("ó","o").str.replace("ú","u")
    )
    return out

def to_datetime_safe(s):  return pd.to_datetime(s, errors="coerce")
def to_float_safe(s):     return pd.to_numeric(s, errors="coerce")

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def loss_function(x, alpha, Lmax):
    x = np.asarray(x, float)
    return (alpha * Lmax * x) / (alpha + x + 1e-12)

def auc(idx: pd.DatetimeIndex, values: np.ndarray) -> float:
    if len(idx) < 2: return 0.0
    t = (idx - idx[0]).days.astype(float)
    return float(np.trapz(np.asarray(values, float), t))

def daily_series(sub_emerg: pd.DataFrame, start, end) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    s = (sub_emerg.set_index("fecha")["emer_rel"]
         .reindex(idx).interpolate("linear").fillna(0.0))
    s.index.name = "fecha"
    return s

def non_overlapping_states(emer_daily: pd.Series) -> pd.DataFrame:
    # Estados por edad no solapada: S1=0..6, S2=7..27, S3=28..59, S4>=60
    v = emer_daily.to_numpy(float)
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])

    def sum_window(i, a, b):
        lo, hi = i - b, i - a
        if hi < 0: return 0.0
        lo_c, hi_c = max(0, lo), min(n - 1, hi)
        if lo_c > hi_c: return 0.0
        return c[hi_c + 1] - c[lo_c]

    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for i in range(n):
        S1[i] = sum_window(i, 0, 6)      # 0..6
        S2[i] = sum_window(i, 7, 27)     # 7..27
        S3[i] = sum_window(i, 28, 59)    # 28..59
        j = i - 60                       # >=60
        S4[i] = c[j + 1] if j >= 0 else 0.0

    return pd.DataFrame({"s1": S1, "s2": S2, "s3": S3, "s4": S4}, index=emer_daily.index)

def effective_T4_days(sow_date: pd.Timestamp, pc_ini: pd.Timestamp, pc_fin: pd.Timestamp) -> int:
    start_S4 = max(pc_ini, sow_date + pd.Timedelta(days=60))
    if start_S4 > pc_fin: return 1
    return max(1, (pc_fin - start_S4).days + 1)

# Tratamientos por estado con residualidad
def aplicar_tratamientos(states_df: pd.DataFrame, idx: pd.DatetimeIndex,
                         trts_sub: pd.DataFrame, log_list: list, ensayo_id: str):
    if trts_sub is None or trts_sub.empty: return states_df
    ctrl = {k: np.ones(len(idx), dtype=float) for k in ["s1","s2","s3","s4"]}

    for _, t in trts_sub.iterrows():
        f_ap = to_datetime_safe(t.get("fecha_aplicacion"))
        if pd.isna(f_ap): continue
        e = float(to_float_safe(t.get("eficacia_pct"))); e = 0.0 if np.isnan(e) else e
        r = int(to_float_safe(t.get("residual_dias")));  r = 0 if np.isnan(r) else r

        mask = (idx >= f_ap) & (idx <= (f_ap + pd.Timedelta(days=r)))
        a1 = int(to_float_safe(t.get("actua_s1"))) == 1 if "actua_s1" in t.index else True
        a2 = int(to_float_safe(t.get("actua_s2"))) == 1 if "actua_s2" in t.index else True
        a3 = int(to_float_safe(t.get("actua_s3"))) == 1 if "actua_s3" in t.index else True
        a4 = int(to_float_safe(t.get("actua_s4"))) == 1 if "actua_s4" in t.index else True

        for k, act in zip(["s1","s2","s3","s4"], [a1,a2,a3,a4]):
            if act: ctrl[k][mask] *= (1.0 - e/100.0)

        log_list.append({
            "ensayo_id": ensayo_id,
            "tipo": str(t.get("tipo","")),
            "fecha_aplicacion": f_ap.date(),
            "eficacia_pct": e,
            "residual_dias": r,
            "estados": ",".join([k for k,act in zip(["s1","s2","s3","s4"],[a1,a2,a3,a4]) if act]),
            "dias_afectados": int(mask.sum())
        })

    S = states_df.copy()
    for k in ["s1","s2","s3","s4"]:
        S[k] = S[k].to_numpy(float) * ctrl[k]
    return S

# Densidad efectiva con IC + escenario (sin Ca)
def calcular_densidad_ensayo(ens_row, emer_df, trts_df, log_list, infestacion_escenario: float):
    ensayo = str(ens_row["ensayo_id"])
    f_sow  = to_datetime_safe(ens_row["fecha_siembra"])
    pc_ini = to_datetime_safe(ens_row["pc_ini"])
    pc_fin = to_datetime_safe(ens_row["pc_fin"])
    IC     = float(to_float_safe(ens_row.get("ic", 250.0)))  # IC por ensayo

    sub = emer_df[emer_df["ensayo_id"] == ensayo][["fecha","emer_rel"]].copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_ini) or pd.isna(pc_fin):
        return np.nan, {}

    sub["fecha"] = to_datetime_safe(sub["fecha"])
    sub["emer_rel"] = to_float_safe(sub["emer_rel"]).fillna(0.0)
    if sub["emer_rel"].max() > 1.5: sub["emer_rel"] = sub["emer_rel"] / 100.0
    sub["emer_rel"] = np.clip(sub["emer_rel"], 0.0, 1.0)

    # Serie diaria y estados por edad
    s = daily_series(sub, f_sow, pc_fin)
    idx = s.index
    states = non_overlapping_states(s)

    # Tratamientos
    trts_sub = trts_df[trts_df["ensayo_id"] == ensayo].copy() if trts_df is not None else None
    states = aplicar_tratamientos(states, idx, trts_sub, log_list, ensayo)

    # Normalización por duración y AUC en PC
    T4 = effective_T4_days(f_sow, pc_ini, pc_fin)
    D = (
        states["s1"].to_numpy(float) * W_ESTADOS["s1"] / DUR["s1"] +
        states["s2"].to_numpy(float) * W_ESTADOS["s2"] / DUR["s2"] +
        states["s3"].to_numpy(float) * W_ESTADOS["s3"] / DUR["s3"] +
        states["s4"].to_numpy(float) * W_ESTADOS["s4"] / max(1, T4)
    )
    mask_pc = (idx >= pc_ini) & (idx <= pc_fin)
    auc_states_pc = auc(idx[mask_pc], D[mask_pc])

    # Escalado por escenario/IC (sin Ca)
    dens_eff = auc_states_pc * (float(infestacion_escenario) / max(1e-6, IC))

    diag = {
        "ensayo_id": ensayo,
        "IC": IC,
        "T4_efectivo_dias": T4,
        "AUC_states_PC": auc_states_pc,
        "infestacion_escenario": float(infestacion_escenario),
        "dens_eff": dens_eff
    }
    return dens_eff, diag

# ============================
# Sidebar y carga de archivo
# ============================
st.sidebar.header("📂 Archivo Excel")
file = st.sidebar.file_uploader("Subí un Excel con hojas: ensayos, emergencia, tratamientos", type=["xlsx"])

escenario_opt = st.sidebar.selectbox(
    "🌿 Escenario de infestación",
    ["Alto (250)", "Medio (125)", "Bajo (50)", "Comparar los 3"], index=0
)

if not file:
    st.info("Subí el archivo para comenzar.")
    st.stop()

# Leer hojas
ensayos = normalize_columns(pd.read_excel(file, sheet_name="ensayos"))
emerg   = normalize_columns(pd.read_excel(file, sheet_name="emergencia"))
trts    = normalize_columns(pd.read_excel(file, sheet_name="tratamientos"))

# Validación mínima
need_ens = {"ensayo_id","fecha_siembra","pc_ini","pc_fin","loss_obs_pct","dataset"}
need_em  = {"ensayo_id","fecha","emer_rel"}
miss_ens = [c for c in need_ens if c not in ensayos.columns]
miss_em  = [c for c in need_em if c not in emerg.columns]
if miss_ens or miss_em:
    st.error(f"Faltan columnas requeridas.\n'ensayos' faltan: {miss_ens}\n'emergencia' faltan: {miss_em}")
    st.stop()

# Normalizaciones típicas
ensayos["dataset"] = ensayos["dataset"].astype(str).str.strip().str.lower()
ensayos["dataset"] = ensayos["dataset"].replace({"calibracion":"calibración"})
for c in ["fecha_siembra","pc_ini","pc_fin"]:
    ensayos[c] = to_datetime_safe(ensayos[c])
for c in ["loss_obs_pct","ic"]:
    if c in ensayos.columns: ensayos[c] = to_float_safe(ensayos[c])
if "ic" not in ensayos.columns:
    st.warning("No se encontró columna 'IC' en 'ensayos'. Se usa IC=250 por defecto.")
    ensayos["ic"] = 250.0
ensayos["ic"] = ensayos["ic"].fillna(250.0)

emerg["fecha"] = to_datetime_safe(emerg["fecha"])
emerg["emer_rel"] = to_float_safe(emerg["emer_rel"]).fillna(0.0)

# ============================
# Pipeline por escenario
# ============================
def correr_pipeline_para_escenario(nombre_esc, infestacion_esc):
    rows, diag_list, trt_logs = [], [], []
    for _, row in ensayos.iterrows():
        dens_eff, diag = calcular_densidad_ensayo(
            row, emerg, trts, trt_logs, infestacion_escenario=infestacion_esc
        )
        rows.append({
            "ensayo_id": row["ensayo_id"],
            "dataset": row["dataset"],
            "loss_obs_pct": float(row["loss_obs_pct"]) if not pd.isna(row["loss_obs_pct"]) else np.nan,
            "dens_eff": dens_eff
        })
        if diag: diag_list.append(diag)

    df_e = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_list)
    log_df  = pd.DataFrame(trt_logs)

    calib = df_e[df_e["dataset"]=="calibración"].dropna(subset=["dens_eff","loss_obs_pct"])
    test  = df_e[df_e["dataset"]=="testeo"].dropna(subset=["dens_eff","loss_obs_pct"])

    if calib.empty:
        return nombre_esc, df_e, diag_df, log_df, None, None, {}

    def objective(p):
        a, L = p
        return rmse(calib["loss_obs_pct"].values, loss_function(calib["dens_eff"].values, a, L))

    res = minimize(objective, x0=[1.0, 80.0], bounds=[(1e-6,None),(1e-6,None)], method="L-BFGS-B")
    alpha_opt, Lmax_opt = map(float, res.x)

    def eval_block(sub):
        if sub.empty: return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
        y_true = sub["loss_obs_pct"].values
        y_pred = loss_function(sub["dens_eff"].values, alpha_opt, Lmax_opt)
        return {"RMSE": rmse(y_true,y_pred), "MAE": mae(y_true,y_pred), "R2": r2(y_true,y_pred)}

    metrics = {"Calibración": eval_block(calib), "Testeo": eval_block(test)}
    df_e["loss_pred_opt"] = loss_function(df_e["dens_eff"].values, alpha_opt, Lmax_opt)

    return nombre_esc, df_e, diag_df, log_df, alpha_opt, Lmax_opt, metrics

# ============================
# Ejecutar según selección
# ============================
if escenario_opt != "Comparar los 3":
    nombre = escenario_opt
    infest = ESCENARIOS.get(nombre, 250.0)
    nombre, df_out, diag_out, log_out, a, L, mets = correr_pipeline_para_escenario(nombre, infest)

    st.subheader(f"🔧 Escenario: {nombre} (IC por ensayo)")
    if a is not None:
        st.success(f"α = {a:.4f} · Lmax = {L:.2f}")
    st.subheader("📈 Métricas")
    st.write(pd.DataFrame(mets).T.round(3))

    st.subheader("📊 Resultados")
    st.dataframe(df_out, use_container_width=True)

    # Gráfico
    st.subheader("📉 Curva ajustada vs datos")
    fig, ax = plt.subplots(figsize=(7,6))
    xmax = float(np.nanmax(df_out["dens_eff"])) if not df_out["dens_eff"].isna().all() else 1.0
    x_grid = np.linspace(0, max(1e-6, xmax)*1.2, 300)
    if a is not None:
        ax.plot(x_grid, loss_function(x_grid, a, L), label=f"Hiperbólica — α={a:.3f}, Lmax={L:.1f}")
    ax.scatter(df_out[df_out["dataset"]=="calibración"]["dens_eff"],
               df_out[df_out["dataset"]=="calibración"]["loss_obs_pct"],
               c="green", label="Calibración", alpha=0.85)
    ax.scatter(df_out[df_out["dataset"]=="testeo"]["dens_eff"],
               df_out[df_out["dataset"]=="testeo"]["loss_obs_pct"],
               c="red", label="Testeo", alpha=0.85)
    ax.set_xlabel("Densidad efectiva (x)")
    ax.set_ylabel("Pérdida de rinde (%)")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

    # Descargas
    st.subheader("⬇️ Descargas")
    st.download_button("CSV — Resultados (escenario)",
                       df_out.to_csv(index=False).encode("utf-8"),
                       f"resultados_{nombre.replace(' ','_')}.csv","text/csv")
    if not diag_out.empty:
        st.download_button("CSV — Diagnóstico por ensayo",
                           diag_out.to_csv(index=False).encode("utf-8"),
                           f"diagnostico_{nombre.replace(' ','_')}.csv","text/csv")
    if not log_out.empty:
        st.download_button("CSV — Log de tratamientos",
                           log_out.to_csv(index=False).encode("utf-8"),
                           f"tratamientos_{nombre.replace(' ','_')}.csv","text/csv")

else:
    st.subheader("📊 Comparación de escenarios (Alto/Medio/Bajo) — usando IC por ensayo")
    comparativa = []
    curvas = {}

    for nombre, infest in ESCENARIOS.items():
        nombre, df_out, _, _, a, L, mets = correr_pipeline_para_escenario(nombre, infest)
        fila = {
            "Escenario": nombre,
            "α": a, "Lmax": L,
            "RMSE Cal": mets.get("Calibración",{}).get("RMSE", np.nan),
            "MAE Cal":  mets.get("Calibración",{}).get("MAE",  np.nan),
            "R2 Cal":   mets.get("Calibración",{}).get("R2",   np.nan),
            "RMSE Tes": mets.get("Testeo",{}).get("RMSE", np.nan),
            "MAE Tes":  mets.get("Testeo",{}).get("MAE",  np.nan),
            "R2 Tes":   mets.get("Testeo",{}).get("R2",   np.nan)
        }
        comparativa.append(fila)
        curvas[nombre] = (df_out, a, L)

    st.dataframe(pd.DataFrame(comparativa).round(3), use_container_width=True)

    st.subheader("📉 Curvas por escenario")
    fig, ax = plt.subplots(figsize=(8,6))
    for nombre, (df_out, a, L) in curvas.items():
        xmax = float(np.nanmax(df_out["dens_eff"])) if not df_out["dens_eff"].isna().all() else 1.0
        x_grid = np.linspace(0, max(1e-6, xmax)*1.2, 300)
        if (a is not None) and (L is not None):
            ax.plot(x_grid, loss_function(x_grid, a, L), label=f"{nombre} (α={a:.2f}, Lmax={L:.1f})")
    ax.set_xlabel("Densidad efectiva (x)"); ax.set_ylabel("Pérdida de rinde (%)")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

