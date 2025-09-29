# -*- coding: utf-8 -*-
# app_calibra_estados_normalizada_opt.py
# Streamlit — Calibración con x (S1–S4) normalizada por duración + función HIPERBÓLICA
# + Optimización de pesos w_S1..w_S4
#
# Requiere: pandas, numpy, plotly, streamlit. (Opcional: scipy, xlsxwriter, kaleido)

import io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Calibración + Optimización de pesos", layout="wide")
st.title("Calibración — x (S1–S4) normalizada + Hiperbólica · con OPTIMIZACIÓN de pesos")

uploaded = st.file_uploader("📥 Subí tu Excel con hojas: `ensayos` y `emergencia`", type=["xlsx"])
st.caption(
    "Hojas mínimas: **ensayos** (ensayo_id, loss_obs_pct, MAX_PLANTS_CAP, fecha_siembra, pc_ini, pc_fin[, Ca, Cs]) "
    "y **emergencia** (ensayo_id, fecha, emer_rel en 0–1 o %)."
)

# ================= Sidebar =================
with st.sidebar:
    st.header("Estados / Periodos")
    t4_mode = st.radio("T4 (S4)", ["Dinámico (en PC)", "Fijo (60 días)"], index=0)
    st.caption("S1=0–6d, S2=7–27d, S3=28–59d, S4≥60d. Se normaliza por la duración de cada estado.")

    st.header("Competencia del cultivo")
    use_ciec = st.checkbox("Aplicar (1 − Ciec)", value=True)
    mode_canopy = st.selectbox("Modelo de canopia", ["Cobertura (%)", "LAI dinámico"], index=0)
    t_lag   = st.number_input("Días a emergencia (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 180, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máx", 0.0, 10.0, 3.5, 0.1)
    k_beer  = st.number_input("k Beer–Lambert", 0.10, 1.50, 0.60, 0.05)
    Ca      = st.number_input("Ca (pl·m²)", 50, 700, 250, 10)
    Cs      = st.number_input("Cs (pl·m²)", 50, 700, 250, 10)
    LAIhc   = st.number_input("LAIhc", 0.5, 10.0, 3.5, 0.1)

    st.header("Pesos por estado (iniciales / manual)")
    w_s1 = st.number_input("w_S1", 0.0, 2.0, 0.0, 0.05)
    w_s2 = st.number_input("w_S2", 0.0, 2.0, 0.3, 0.05)
    w_s3 = st.number_input("w_S3", 0.0, 2.0, 0.6, 0.05)
    w_s4 = st.number_input("w_S4", 0.0, 2.0, 1.0, 0.05)
    st.caption("La optimización respeta 0 ≤ w1 ≤ w2 ≤ w3 ≤ w4 ≤ 2.0")

    st.header("Escalado por CAP")
    scale_basis = st.radio(
        "Base para el factor CAP",
        ["AUC total (siembra→fin)", "AUC en PC (conservador)"], index=0,
        help="El factor = CAP / AUC(base). Elegí si querés escalar por el AUC total o solo por el AUC dentro del periodo crítico."
    )

    st.header("Ajuste hiperbólico")
    alpha0 = st.number_input("α inicial", 1e-4, 5.0, 0.05, 0.01, format="%.4f")
    Lmax0  = st.number_input("Lmax inicial (%)", 5.0, 300.0, 80.0, 1.0)
    st.caption("Bounds: α∈[1e-4, 5.0], Lmax∈[10, 300]")

    st.header("Optimizar pesos (búsqueda aleatoria)")
    do_opt = st.checkbox("Activar optimización de pesos", value=False)
    n_weights = st.number_input("N combinaciones de pesos", 10, 5000, 200, 10)
    n_al = st.number_input("Intentos α–Lmax por combinación", 20, 2000, 300, 20,
                           help="Usa aleatorio rápido (sin scipy) para velocidad")
    seed = st.number_input("Seed", 0, 10_000, 2025, 1)
    st.markdown("---")

    run = st.button("🧠 Ejecutar calibración", use_container_width=True)

# ================= Utilidades =================
def daily_series(sub_emerg: pd.DataFrame, start, end) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    s = sub_emerg.set_index("fecha")["emer_rel"].reindex(idx).interpolate("linear").fillna(0.0)
    s.index.name = "fecha"
    return s

def auc(idx: pd.DatetimeIndex, values: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    t = (idx - idx[0]).days.astype(float)
    return float(np.trapz(np.asarray(values, float), t))

def compute_canopy(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                   mode_canopy: str, t_lag: int, t_close: int,
                   cov_max: float, lai_max: float, k_beer: float):
    days = (idx - pd.Timestamp(sow_date)).days.astype(float)

    def logistic_between(days_, start, end, y_max):
        end = start + 1 if end <= start else end
        t_mid = 0.5 * (start + end)
        r = 4.0 / max(1.0, (end - start))
        return y_max / (1.0 + np.exp(-r * (days_ - t_mid)))

    if mode_canopy == "Cobertura (%)":
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max / 100.0))
        fc = np.clip(fc, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc, 1e-9, 1.0)) / max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc = 1.0 - np.exp(-k_beer * LAI)
        fc = np.clip(fc, 0.0, 1.0)
    return fc, LAI

def ciec_series(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer, Ca, Cs, LAIhc):
    _, LAI = compute_canopy(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer)
    Ca_s = Ca if Ca > 0 else 1e-6
    Cs_s = Cs if Cs > 0 else 1e-6
    Ciec = (LAI / max(1e-6, LAIhc)) * (Ca_s / Cs_s)
    return np.clip(Ciec, 0.0, 1.0)

def non_overlapping_states(emer_daily: pd.Series) -> pd.DataFrame:
    """S1 0–6, S2 7–27, S3 28–59, S4 ≥60 (en pl·m²/día) con integral acumulada."""
    v = emer_daily.to_numpy(float)
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])

    def sum_age_window(i, a, b):
        lo, hi = i - b, i - a
        if hi < 0:
            return 0.0
        lo_c, hi_c = max(0, lo), min(n - 1, hi)
        if lo_c > hi_c:
            return 0.0
        return c[hi_c + 1] - c[lo_c]

    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for i in range(n):
        S1[i] = sum_age_window(i, 0, 6)
        S2[i] = sum_age_window(i, 7, 27)
        S3[i] = sum_age_window(i, 28, 59)
        j = i - 60
        S4[i] = c[j + 1] if j >= 0 else 0.0

    return pd.DataFrame({"S1": S1, "S2": S2, "S3": S3, "S4": S4}, index=emer_daily.index)

def effective_T4_days(sow_date: pd.Timestamp, pc_ini: pd.Timestamp, pc_fin: pd.Timestamp) -> int:
    start_S4 = max(pc_ini, sow_date + pd.Timedelta(days=60))
    if start_S4 > pc_fin:
        return 1
    return max(1, (pc_fin - start_S4).days + 1)

def loss_hyperbolic(x, alpha, Lmax):
    x = np.asarray(x, float)
    return (alpha * x) / (1.0 + (alpha * x / Lmax))

def calibrate_hyperbolic_exact(x, y, alpha0=0.05, Lmax0=80.0):
    """Usa scipy si está disponible (más exacto)."""
    try:
        from scipy.optimize import minimize
        def objective(p):
            a, L = p
            yhat = loss_hyperbolic(x, a, L)
            return np.mean((y - yhat) ** 2)
        res = minimize(objective, x0=[alpha0, Lmax0],
                       bounds=[(1e-4, 5.0), (10.0, 300.0)],
                       method="L-BFGS-B")
        a, L = float(res.x[0]), float(res.x[1])
        yhat = loss_hyperbolic(x, a, L)
    except Exception:
        # fallback aleatorio
        rng = np.random.default_rng(123)
        best = None; best_loss = float("inf")
        for _ in range(2000):
            a = 10**rng.uniform(-4, np.log10(5.0))
            L = rng.uniform(10.0, 300.0)
            yhat_ = loss_hyperbolic(x, a, L)
            mse = float(np.mean((y - yhat_) ** 2))
            if mse < best_loss:
                best_loss, best, yhat = mse, (a, L), yhat_
        a, L = best
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
    r2 = (1 - ss_res / ss_tot) if (ss_tot and ss_tot != 0) else float("nan")
    return {"alpha": a, "Lmax": L, "rmse": rmse, "mae": mae, "r2": r2, "yhat": yhat}

def calibrate_hyperbolic_quick(x, y, iters=300, seed=7):
    """Aleatorio puro (rápido) para usar dentro del loop de optimización de pesos."""
    rng = np.random.default_rng(seed)
    best = None; best_loss = float("inf"); yhat = None
    for _ in range(int(iters)):
        a = 10**rng.uniform(-4, np.log10(5.0))
        L = rng.uniform(10.0, 300.0)
        yhat_ = loss_hyperbolic(x, a, L)
        mse = float(np.mean((y - yhat_) ** 2))
        if mse < best_loss:
            best_loss, best, yhat = mse, (a, L), yhat_
    a, L = best
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
    r2 = (1 - ss_res / ss_tot) if (ss_tot and ss_tot != 0) else float("nan")
    return {"alpha": a, "Lmax": L, "rmse": rmse, "mae": mae, "r2": r2, "yhat": yhat}

def build_x_given_weights(ens, emer, w1, w2, w3, w4,
                          use_ciec=True, mode_canopy="Cobertura (%)",
                          t_lag=7, t_close=45, cov_max=85.0, lai_max=3.5, k_beer=0.60,
                          Ca=250, Cs=250, LAIhc=3.5,
                          t4_mode="Dinámico (en PC)", scale_basis="AUC total (siembra→fin)"):
    """Devuelve x por ensayo según pesos dados."""
    L1, L2, L3 = 7, 21, 32
    ens2 = ens.copy()
    ens2["MAX_PLANTS_CAP"] = pd.to_numeric(ens2["MAX_PLANTS_CAP"], errors="coerce").fillna(0)
    ens2["pc_ini"] = ens2.apply(lambda r: r["pc_ini"] if pd.notna(r["pc_ini"]) else r["fecha_siembra"], axis=1)

    xs = []
    for _, row in ens2.iterrows():
        ens_id = row["ensayo_id"]
        f_sow  = pd.to_datetime(row["fecha_siembra"])
        pc_ini = pd.to_datetime(row["pc_ini"])
        pc_fin = pd.to_datetime(row["pc_fin"])
        cap    = float(row["MAX_PLANTS_CAP"])

        if pd.isna(f_sow) or pd.isna(pc_fin) or cap <= 0:
            xs.append(0.0)
            continue

        sub = emer[emer["ensayo_id"] == ens_id][["fecha", "emer_rel"]].dropna().copy()
        if sub.empty:
            xs.append(0.0)
            continue

        sub["emer_rel"] = pd.to_numeric(sub["emer_rel"], errors="coerce").fillna(0.0)
        sub["emer_rel"] = np.clip(sub["emer_rel"], 0.0, None)
        if sub["emer_rel"].max() > 1.5:
            sub["emer_rel"] = sub["emer_rel"] / 100.0
        sub["emer_rel"] = np.clip(sub["emer_rel"], 0.0, 1.0)

        # Serie diaria desde siembra a fin del PC (para cubrir el PC seguro)
        s = daily_series(sub, f_sow, pc_fin)

        # AUC para escalado CAP
        auc_total = auc(s.index, s.values)
        mask_pc = (s.index >= pc_ini) & (s.index <= pc_fin)
        if not mask_pc.any():
            xs.append(0.0)
            continue
        auc_pc = auc(s.index[mask_pc], s.values[mask_pc])

        EPS = 1e-6
        base = max(auc_pc if scale_basis.startswith("AUC en PC") else auc_total, EPS)
        factor = cap / base

        # (1−Ciec) opcional
        if use_ciec:
            Ciec = ciec_series(
                s.index, f_sow, mode_canopy, int(t_lag), int(t_close),
                float(cov_max), float(lai_max), float(k_beer), int(Ca if Ca > 0 else 1),
                int(Cs if Cs > 0 else 1), float(LAIhc)
            )
            daily_eff = s.values * (1.0 - Ciec) * factor
        else:
            daily_eff = s.values * factor

        states = non_overlapping_states(pd.Series(daily_eff, index=s.index))

        # T4 fijo o dinámico
        if t4_mode.startswith("Fijo"):
            T4 = 60
        else:
            T4 = effective_T4_days(f_sow, pc_ini, pc_fin)

        D = (
            states["S1"] * w1 / L1 +
            states["S2"] * w2 / L2 +
            states["S3"] * w3 / L3 +
            states["S4"] * w4 / max(1, T4)
        ).to_numpy()

        x_val_i = float(auc(s.index[mask_pc], D[mask_pc]))
        xs.append(x_val_i)

    return np.array(xs, dtype=float)

def optimize_weights(ens, emer, y_obs, n_weights=200, n_al=300, seed=2025,
                     use_ciec=True, mode_canopy="Cobertura (%)",
                     t_lag=7, t_close=45, cov_max=85.0, lai_max=3.5, k_beer=0.60,
                     Ca=250, Cs=250, LAIhc=3.5,
                     t4_mode="Dinámico (en PC)", scale_basis="AUC total (siembra→fin)"):
    """Búsqueda aleatoria monótona de pesos, con calibración hiperbólica rápida por combinación."""
    rng = np.random.default_rng(int(seed))
    records = []
    best = None
    prog = st.progress(0, text="Buscando combinación de pesos…")
    for it in range(int(n_weights)):
        # muestreo monótono: 0 ≤ w1 ≤ w2 ≤ w3 ≤ w4 ≤ 2
        w1 = rng.uniform(0.0, 2.0)
        w2 = rng.uniform(w1, 2.0)
        w3 = rng.uniform(w2, 2.0)
        w4 = rng.uniform(w3, 2.0)

        x_val = build_x_given_weights(
            ens, emer, w1, w2, w3, w4,
            use_ciec, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer,
            Ca, Cs, LAIhc, t4_mode, scale_basis
        )
        fit = calibrate_hyperbolic_quick(x_val, y_obs, iters=int(n_al), seed=7 + it)
        rec = dict(w_S1=w1, w_S2=w2, w_S3=w3, w_S4=w4,
                   alpha=fit["alpha"], Lmax=fit["Lmax"],
                   RMSE=fit["rmse"], MAE=fit["mae"], R2=fit["r2"])
        records.append(rec)
        if best is None or fit["rmse"] < best["RMSE"]:
            best = rec
        prog.progress(int((it + 1) / int(n_weights) * 100))
    prog.empty()
    df = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    return df, best

# ================= Ejecución =================
if uploaded is None:
    st.info("Subí el Excel para habilitar la calibración.")
    st.stop()

# Leer Excel
try:
    ens = pd.read_excel(uploaded, sheet_name="ensayos")
    emer = pd.read_excel(uploaded, sheet_name="emergencia")
except Exception as e:
    st.error(f"No pude leer el Excel: {e}")
    st.stop()

# Normalizar fechas
for c in ["fecha_siembra", "pc_ini", "pc_fin"]:
    ens[c] = pd.to_datetime(ens[c], errors="coerce")
emer["fecha"] = pd.to_datetime(emer["fecha"], errors="coerce")

# Chequeo de columnas
need_ens = {"ensayo_id", "loss_obs_pct", "MAX_PLANTS_CAP", "fecha_siembra", "pc_ini", "pc_fin"}
need_em = {"ensayo_id", "fecha", "emer_rel"}
if not need_ens.issubset(ens.columns) or not need_em.issubset(emer.columns):
    st.error("Estructura inválida. Revisá columnas requeridas en 'ensayos' y 'emergencia'.")
    st.stop()

# Sanitización de MAX_PLANTS_CAP
ens["MAX_PLANTS_CAP"] = pd.to_numeric(ens["MAX_PLANTS_CAP"], errors="coerce").fillna(0)
bad_cap = ens["MAX_PLANTS_CAP"] <= 0
if bad_cap.any():
    st.warning(f"Ensayos con MAX_PLANTS_CAP ≤ 0 serán omitidos: {ens.loc[bad_cap, 'ensayo_id'].tolist()}")

# Si pc_ini es NaT, usar fecha_siembra como proxy
ens["pc_ini"] = ens.apply(lambda r: r["pc_ini"] if pd.notna(r["pc_ini"]) else r["fecha_siembra"], axis=1)

# Construir x con pesos manuales (sin optimizar) y calibrar
def compute_x_and_fit(ens, emer, w1, w2, w3, w4):
    x_val = build_x_given_weights(
        ens, emer, w1, w2, w3, w4,
        use_ciec, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer,
        Ca, Cs, LAIhc, t4_mode, scale_basis
    )
    y_obs = ens["loss_obs_pct"].astype(float).to_numpy()
    fit = calibrate_hyperbolic_exact(x_val, y_obs, float(alpha0), float(Lmax0))
    return x_val, y_obs, fit

with st.spinner("Calculando x y calibrando (pesos actuales)…"):
    x_val_manual, y_obs, fit_manual = compute_x_and_fit(ens, emer, w_s1, w_s2, w_s3, w_s4)

# Mostrar resultados con pesos manuales
st.subheader("Resultados con pesos actuales (manuales)")
st.write(f"**α = {fit_manual['alpha']:.4f} · Lmax = {fit_manual['Lmax']:.2f}** · RMSE = {fit_manual['rmse']:.2f} · "
         f"MAE = {fit_manual['mae']:.2f} · R² = {fit_manual['r2']:.3f}")

ens_out = ens[["ensayo_id", "loss_obs_pct", "MAX_PLANTS_CAP"]].copy()
ens_out["x_pl_m2_states"] = x_val_manual
ens_out["predicho"] = fit_manual["yhat"]
st.dataframe(ens_out, use_container_width=True)

# Gráfico Obs vs Pred (manual)
fig_m = go.Figure()
fig_m.add_trace(go.Scatter(
    x=ens_out["loss_obs_pct"], y=ens_out["predicho"],
    mode="markers+text", text=ens_out["ensayo_id"], textposition="top center",
    hovertemplate=("Ensayo: %{text}<br>Obs: %{x:.2f}%<br>Pred: %{y:.2f}%<br>" +
                   "x: %{customdata[0]:.2f} pl·m²<extra></extra>"),
    customdata=np.c_[ens_out["x_pl_m2_states"]],
    name="Ensayos"
))
mx_m = max(float(ens_out["loss_obs_pct"].max()), float(ens_out["predicho"].max()), 1.0)
fig_m.add_trace(go.Scatter(x=[0, mx_m], y=[0, mx_m], mode="lines", name="1:1", line=dict(dash="dash")))
fig_m.update_layout(
    title="Observado vs Predicho — (pesos manuales)",
    xaxis_title="Observado (%)", yaxis_title="Predicho (%)",
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_m, use_container_width=True)

# Curva pérdida (manual)
x_grid_m = np.linspace(0.0, max(1.0, float(np.nanmax(x_val_manual)) * 1.25), 400)
y_curve_m = loss_hyperbolic(x_grid_m, fit_manual["alpha"], fit_manual["Lmax"])
fig_m2 = go.Figure()
fig_m2.add_trace(go.Scatter(x=x_grid_m, y=y_curve_m, mode="lines", name="Curva hiperbólica"))
fig_m2.add_trace(go.Scatter(x=x_val_manual, y=y_obs, mode="markers", name="Observado"))
fig_m2.update_layout(
    title=f"Función de pérdida (pesos manuales) — α={fit_manual['alpha']:.4f}, Lmax={fit_manual['Lmax']:.1f}",
    xaxis_title="x (pl·m²) — estados normalizada por duración", yaxis_title="Pérdida de rinde (%)",
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_m2, use_container_width=True)

# ================= Optimización de pesos =================
if do_opt and run:
    with st.spinner("Optimizando pesos (búsqueda aleatoria monótona)…"):
        trials_df, best = optimize_weights(
            ens, emer, y_obs,
            n_weights=int(n_weights), n_al=int(n_al), seed=int(seed),
            use_ciec=use_ciec, mode_canopy=mode_canopy,
            t_lag=int(t_lag), t_close=int(t_close),
            cov_max=float(cov_max), lai_max=float(lai_max), k_beer=float(k_beer),
            Ca=int(Ca), Cs=int(Cs), LAIhc=float(LAIhc),
            t4_mode=t4_mode, scale_basis=scale_basis
        )
    st.success(
        f"Mejores pesos: w_S1={best['w_S1']:.3f}, w_S2={best['w_S2']:.3f}, "
        f"w_S3={best['w_S3']:.3f}, w_S4={best['w_S4']:.3f} · "
        f"α={best['alpha']:.4f}, Lmax={best['Lmax']:.1f} · "
        f"RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}, R²={best['R2']:.3f}"
    )

    # Recalcular x y fit exacto con los mejores pesos (fuera del loop rápido)
    x_val_best = build_x_given_weights(
        ens, emer, best["w_S1"], best["w_S2"], best["w_S3"], best["w_S4"],
        use_ciec, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer,
        Ca, Cs, LAIhc, t4_mode, scale_basis
    )
    fit_best_exact = calibrate_hyperbolic_exact(x_val_best, y_obs, float(alpha0), float(Lmax0))

    out_best = ens[["ensayo_id", "loss_obs_pct", "MAX_PLANTS_CAP"]].copy()
    out_best["x_pl_m2_states"] = x_val_best
    out_best["predicho"] = fit_best_exact["yhat"]

    st.subheader("Mejor combinación — Obs vs Pred")
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=out_best["loss_obs_pct"], y=out_best["predicho"],
        mode="markers+text", text=out_best["ensayo_id"], textposition="top center",
        hovertemplate=("Ensayo: %{text}<br>Obs: %{x:.2f}%<br>Pred: %{y:.2f}%<br>" +
                       "x: %{customdata[0]:.2f} pl·m²<extra></extra>"),
        customdata=np.c_[out_best["x_pl_m2_states"]],
        name="Ensayos"
    ))
    mx_b = max(float(out_best["loss_obs_pct"].max()), float(out_best["predicho"].max()), 1.0)
    fig_b.add_trace(go.Scatter(x=[0, mx_b], y=[0, mx_b], mode="lines", name="1:1", line=dict(dash="dash")))
    fig_b.update_layout(
        title="Observado vs Predicho — (mejores pesos)",
        xaxis_title="Observado (%)", yaxis_title="Predicho (%)",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # Curva pérdida (mejor)
    x_grid_b = np.linspace(0.0, max(1.0, float(np.nanmax(x_val_best)) * 1.25), 400)
    y_curve_b = loss_hyperbolic(x_grid_b, fit_best_exact["alpha"], fit_best_exact["Lmax"])
    fig_b2 = go.Figure()
    fig_b2.add_trace(go.Scatter(x=x_grid_b, y=y_curve_b, mode="lines", name="Curva hiperbólica"))
    fig_b2.add_trace(go.Scatter(x=x_val_best, y=y_obs, mode="markers", name="Observado"))
    fig_b2.update_layout(
        title=f"Función de pérdida (mejores pesos) — α={fit_best_exact['alpha']:.4f}, Lmax={fit_best_exact['Lmax']:.1f}",
        xaxis_title="x (pl·m²) — estados normalizada por duración", yaxis_title="Pérdida de rinde (%)",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_b2, use_container_width=True)

    # ---- Descargas (mejores pesos) ----
    st.subheader("Descargas (mejores pesos)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📥 CSV resultados (best)",
            data=out_best.to_csv(index=False).encode("utf-8"),
            file_name="calibracion_best_pesos.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            out_best.to_excel(writer, sheet_name="resultados", index=False)
            pd.DataFrame({
                "w_S1": [best["w_S1"]], "w_S2": [best["w_S2"]],
                "w_S3": [best["w_S3"]], "w_S4": [best["w_S4"]],
                "alpha": [fit_best_exact["alpha"]], "Lmax": [fit_best_exact["Lmax"]],
                "RMSE": [fit_best_exact["rmse"]], "MAE": [fit_best_exact["mae"]], "R2": [fit_best_exact["r2"]],
                "t4_mode": [t4_mode], "use_ciec": [use_ciec], "scale_basis": [scale_basis],
                "canopy_mode": [mode_canopy], "t_lag": [t_lag], "t_close": [t_close],
                "cov_max": [cov_max], "lai_max": [lai_max], "k_beer": [k_beer],
                "Ca": [Ca], "Cs": [Cs], "LAIhc": [LAIhc]
            }).to_excel(writer, sheet_name="resumen", index=False)
            trials_sheet = "top100_trials"
            st.caption("Para todas las combinaciones evaluadas, usá el botón de abajo (CSV).")
        st.download_button(
            "📊 Excel resultados (best)",
            data=bio.getvalue(),
            file_name="calibracion_best_pesos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c3:
        st.download_button(
            "🧪 CSV (todas las combinaciones evaluadas)",
            data=trials_df.to_csv(index=False).encode("utf-8"),
            file_name="calibracion_trials_pesos.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---- Descargas (pesos manuales) ----
st.subheader("Descargas (pesos manuales)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.download_button(
        "📥 CSV resultados (manual)",
        data=ens_out.to_csv(index=False).encode("utf-8"),
        file_name="calibracion_estados_normalizada_hiperbolica_manual.csv",
        mime="text/csv",
        use_container_width=True
    )
with c2:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        ens_out.to_excel(writer, sheet_name="resultados", index=False)
        pd.DataFrame({
            "alpha": [fit_manual["alpha"]], "Lmax": [fit_manual["Lmax"]],
            "RMSE": [fit_manual["rmse"]], "MAE": [fit_manual["mae"]], "R2": [fit_manual["r2"]],
            "t4_mode": [t4_mode], "use_ciec": [use_ciec], "scale_basis": [scale_basis],
            "w_S1": [w_s1], "w_S2": [w_s2], "w_S3": [w_s3], "w_S4": [w_s4],
            "canopy_mode": [mode_canopy], "t_lag": [t_lag], "t_close": [t_close],
            "cov_max": [cov_max], "lai_max": [lai_max], "k_beer": [k_beer],
            "Ca": [Ca], "Cs": [Cs], "LAIhc": [LAIhc]
        }).to_excel(writer, sheet_name="resumen", index=False)
    st.download_button(
        "📊 Excel resultados (manual)",
        data=bio.getvalue(),
        file_name="calibracion_estados_normalizada_hiperbolica_manual.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
with c3:
    params = {
        "loss_kind": "hyperbolic",
        "alpha": float(fit_manual["alpha"]), "Lmax": float(fit_manual["Lmax"]),
        "use_ciec": bool(use_ciec),
        "state_weights": {"S1": float(w_s1), "S2": float(w_s2), "S3": float(w_s3), "S4": float(w_s4)},
        "state_durations": {"S1": 7, "S2": 21, "S3": 32, "S4": (60 if t4_mode.startswith('Fijo') else "effective_in_PC")},
        "t4_mode": ("fixed" if t4_mode.startswith("Fijo") else "dynamic"),
        "canopy": {
            "mode": mode_canopy, "t_lag": int(t_lag), "t_close": int(t_close),
            "cov_max": float(cov_max), "lai_max": float(lai_max),
            "k_beer": float(k_beer), "Ca": float(Ca), "Cs": float(Cs), "LAIhc": float(LAIhc)
        },
        "scale_basis": ("AUC_PC" if scale_basis.startswith("AUC en PC") else "AUC_total")
    }
    st.download_button(
        "🧾 JSON parámetros (manual)",
        data=json.dumps(params, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="predweem_config_estados_normalizada.json",
        mime="application/json",
        use_container_width=True
    )
with c4:
    st.info("Tip: para exportar gráficos a PNG desde plotly usá `kaleido` (`pip install -U kaleido`).")
