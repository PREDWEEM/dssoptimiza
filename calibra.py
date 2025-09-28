# -*- coding: utf-8 -*-
# calibra_auc_pc_app.py — Calibración PREDWEEM con AUC en el PC
# Autor: GUILLERMO + ChatGPT

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============== UI ==============
st.set_page_config(page_title="Calibración PREDWEEM (AUC en PC)", layout="wide")
st.title("Calibración PREDWEEM — AUC dentro del Período Crítico (PC)")

uploaded = st.file_uploader("Cargar Excel con hojas: ensayos, tratamientos, emergencia", type=["xlsx"])

with st.sidebar:
    st.header("Opciones de calibración")
    loss_kind = st.selectbox(
        "Función de pérdida",
        ["Hiperbólica (α, Lmax)", "Exponencial saturada (α, β, Lmax)"],
        index=0
    )

    st.subheader("Pesos dentro del PC")
    pc_weight_kind = st.selectbox(
        "Perfil de peso",
        ["Uniforme", "Triangular (pico al centro)", "Gaussiana (σ relativo al PC)"],
        index=0
    )
    sigma_rel = st.slider("σ relativo (solo Gaussiana)", 0.05, 0.50, 0.20, 0.01, disabled=(pc_weight_kind!="Gaussiana"))

    st.subheader("Competencia del cultivo (Ciec)")
    use_ciec = st.checkbox("Multiplicar por (1 − Ciec) antes del AUC", value=True)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 180, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 10.0, 3.5, 0.1)
    k_beer = st.number_input("k Beer–Lambert", 0.10, 1.50, 0.60, 0.05)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

    st.subheader("Tratamientos")
    apply_trat = st.checkbox("Aplicar tratamientos (reducción residual simple)", value=True)

    st.subheader("Optimización")
    if loss_kind == "Hiperbólica (α, Lmax)":
        alpha_bounds = (1e-3, 10.0)
        Lmax_bounds  = (5.0, 300.0)
        x0 = [0.375, 80.0]
    else:
        alpha_bounds = (1e-4, 10.0)
        beta_bounds  = (0.3, 3.0)
        Lmax_bounds  = (5.0, 300.0)
        x0 = [0.5, 1.0, 120.0]

    metric_kind = st.selectbox("Métrica a minimizar", ["RMSE", "MAE"], index=0)

# ============== Utils ==============
def as_daily_index(start, end):
    return pd.date_range(start=start, end=end, freq="D")

def interp_daily(sub_emerg: pd.DataFrame, start, end):
    idx = as_daily_index(start, end)
    s = sub_emerg.set_index("fecha")["emer_rel"].reindex(idx).interpolate("linear").fillna(0.0)
    s.index.name = "fecha"
    return s

def compute_canopy(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                   mode_canopy: str, t_lag: int, t_close: int, cov_max: float,
                   lai_max: float, k_beer: float):
    days_since_sow = (idx.date - sow_date.date()).astype("timedelta64[D]").astype(int).astype(float)

    def logistic_between(days, start, end, y_max):
        if end <= start: end = start + 1
        t_mid = 0.5 * (start + end); r = 4.0 / max(1.0, (end - start))
        return y_max / (1.0 + np.exp(-r * (days - t_mid)))

    if mode_canopy == "Cobertura dinámica (%)":
        fc_dyn = np.where(days_since_sow < t_lag, 0.0, logistic_between(days_since_sow, t_lag, t_close, cov_max/100.0))
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc_dyn, 1e-9, 1.0)) / max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days_since_sow < t_lag, 0.0, logistic_between(days_since_sow, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc_dyn = 1.0 - np.exp(-k_beer * LAI)
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)

    return fc_dyn, LAI

def ciec_series(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                mode_canopy: str, t_lag: int, t_close: int, cov_max: float,
                lai_max: float, k_beer: float, Ca: float, Cs: float, LAIhc: float):
    _, LAI = compute_canopy(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer)
    Ca_safe = Ca if Ca > 0 else 1e-6
    Cs_safe = Cs if Cs > 0 else 1e-6
    Ciec = (LAI / max(1e-6, LAIhc)) * (Ca_safe / Cs_safe)
    return np.clip(Ciec, 0.0, 1.0)

def weights_pc(idx: pd.DatetimeIndex, pc_ini: pd.Timestamp, pc_fin: pd.Timestamp,
               kind="Uniforme", sigma_rel=0.20) -> np.ndarray:
    t = idx.values.astype("datetime64[ns]").astype("int64") / 1e9 / 86400.0
    t0 = pd.to_datetime(pc_ini).to_datetime64().astype("int64") / 1e9 / 86400.0
    t1 = pd.to_datetime(pc_fin).to_datetime64().astype("int64") / 1e9 / 86400.0
    mask = (t >= t0) & (t <= t1)
    w = np.zeros_like(t, dtype=float)
    if not mask.any():
        return w
    ti = t[mask]
    if kind == "Uniforme":
        w[mask] = 1.0
    elif kind == "Triangular (pico al centro)":
        mid = 0.5 * (t0 + t1)
        span = max(1e-6, (t1 - t0))
        w[mask] = 1.0 - np.abs((ti - mid) / (span/2.0))
        w[mask] = np.clip(w[mask], 0.0, 1.0)
    else:
        mid = 0.5 * (t0 + t1)
        span = max(1e-6, (t1 - t0))
        sigma = sigma_rel * span
        w[mask] = np.exp(-0.5 * ((ti - mid) / max(1e-6, sigma))**2)
    # Normalizar tal que promedio ≈ 1 en el PC (no cambia la escala de AUC)
    mean_w = w[mask].mean() if mask.any() else 1.0
    if mean_w > 1e-9:
        w[mask] = w[mask] / mean_w
    return w

def auc_with_weights(idx: pd.DatetimeIndex, values: np.ndarray, weights: np.ndarray) -> float:
    # integra valores*pesos respecto al tiempo (días)
    t = idx.values.astype("datetime64[ns]").astype("int64") / 1e9 / 86400.0
    y = np.asarray(values, float) * np.asarray(weights, float)
    # mantener solo tramos con peso>0
    mask = weights > 0
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(y[mask], t[mask]))

def apply_treatments(series: pd.Series, trat_sub: pd.DataFrame) -> pd.Series:
    # Reducción residual simple sobre la serie diaria (aprox)
    s = series.copy()
    if trat_sub is None or trat_sub.empty:
        return s
    for _, tr in trat_sub.iterrows():
        if pd.isna(tr.get("tipo")) or pd.isna(tr.get("fecha_aplicacion")) or pd.isna(tr.get("eficacia_pct")):
            continue
        f_app = pd.to_datetime(tr["fecha_aplicacion"])
        resd = int(tr["residual_dias"]) if not pd.isna(tr.get("residual_dias")) else 0
        eff = float(tr["eficacia_pct"]) / 100.0
        m = (s.index >= f_app) & (s.index <= f_app + pd.Timedelta(days=resd))
        if m.any():
            s.loc[m] = s.loc[m] * (1.0 - eff)
    return s

def loss_hyperbolic(x, alpha, Lmax):
    x = np.asarray(x, float)
    return (alpha * x) / (1.0 + (alpha * x / Lmax))

def loss_exponential(x, alpha, beta, Lmax):
    x = np.asarray(x, float)
    return Lmax * (1.0 - np.exp(-alpha * np.power(np.maximum(0.0, x), beta)))

def fit_line(y_true, y_pred):
    # y_pred ≈ a + b*y_true (mínimos cuadrados)
    y = np.asarray(y_true, float); x = np.asarray(y_true, float)  # fitting vs true for 1:1 overlay clarity
    # recta de regresión predicho = a + b*observado
    X = np.vstack([np.ones_like(y), y]).T
    b_hat = np.linalg.lstsq(X, np.asarray(y_pred, float), rcond=None)[0]
    a, b = float(b_hat[0]), float(b_hat[1])
    # R2 del ajuste lineal
    yhat = a + b*y
    ss_res = np.sum((y_pred - yhat)**2)
    ss_tot = np.sum((y_pred - np.mean(y_pred))**2) if len(y_pred) > 1 else np.nan
    R2 = 1 - ss_res/ss_tot if ss_tot and ss_tot != 0 else np.nan
    return a, b, R2

# ============== Core ==============
if uploaded is None:
    st.info("Subí el Excel para comenzar.")
    st.stop()

try:
    ens = pd.read_excel(uploaded, sheet_name="ensayos")
    trat = pd.read_excel(uploaded, sheet_name="tratamientos")
    emer = pd.read_excel(uploaded, sheet_name="emergencia")
except Exception as e:
    st.error(f"No pude leer el Excel: {e}")
    st.stop()

# Asegurar tipos de fecha
for c in ["fecha_siembra", "pc_ini", "pc_fin"]:
    ens[c] = pd.to_datetime(ens[c], errors="coerce")
emer["fecha"] = pd.to_datetime(emer["fecha"], errors="coerce")
if "ensayo_id" not in ens or "ensayo_id" not in emer or "ensayo_id" not in trat:
    st.error("Falta columna 'ensayo_id' en alguna hoja.")
    st.stop()

# Precomputar x_pc por ensayo
def compute_x_for_row(row):
    ens_id = row["ensayo_id"]
    f_sow = pd.to_datetime(row["fecha_siembra"])
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    max_cap = float(row["MAX_PLANTS_CAP"])

    sub = emer[emer["ensayo_id"] == ens_id][["fecha","emer_rel"]].dropna().copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_fin):
        return 0.0
    s_idx = as_daily_index(f_sow, pc_fin)
    s = sub.set_index("fecha")["emer_rel"].reindex(s_idx).interpolate("linear").fillna(0.0)
    s.index.name = "fecha"

    # Ciec
    if use_ciec:
        Ciec = ciec_series(s.index, f_sow, mode_canopy, int(t_lag), int(t_close), float(cov_max),
                           float(lai_max), float(k_beer), float(Ca), float(Cs), float(LAIhc))
        s = s * (1.0 - Ciec)

    # Tratamientos (reducción simple)
    if apply_trat:
        tsub = trat[trat["ensayo_id"] == ens_id]
        s = apply_treatments(s, tsub)

    # Factor de área: AUC siembra→pc_fin
    w_all = np.ones(len(s), float)
    auc_all = auc_with_weights(s.index, s.values, w_all)
    if auc_all <= 0.0:
        return 0.0
    factor = max_cap / auc_all

    # Pesos dentro del PC
    w_pc = weights_pc(s.index, pc_ini, pc_fin, pc_weight_kind, float(sigma_rel))
    x_pc = factor * auc_with_weights(s.index, s.values, w_pc)
    return float(x_pc)

with st.spinner("Calculando x (AUC-PC) por ensayo…"):
    ens = ens.copy()
    ens["x_pc"] = ens.apply(compute_x_for_row, axis=1)

# Objetivo
obs = ens["loss_obs_pct"].to_numpy(dtype=float)
xvec = ens["x_pc"].to_numpy(dtype=float)

def objective(params):
    if loss_kind == "Hiperbólica (α, Lmax)":
        alpha, Lmax = params
        pred = loss_hyperbolic(xvec, alpha, Lmax)
    else:
        alpha, beta, Lmax = params
        pred = loss_exponential(xvec, alpha, beta, Lmax)

    if metric_kind == "RMSE":
        return float(np.sqrt(np.mean((obs - pred) ** 2)))
    else:
        return float(np.mean(np.abs(obs - pred)))

st.markdown("---")
colA, colB = st.columns([1,1])
with colA:
    if st.button("🧠 Ejecutar calibración", use_container_width=True):
        if loss_kind == "Hiperbólica (α, Lmax)":
            bounds = [alpha_bounds, Lmax_bounds]
        else:
            bounds = [alpha_bounds, (beta_bounds[0], beta_bounds[1]), Lmax_bounds]
        try:
            res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
            st.session_state.calib = {"ok": True, "res": res}
        except Exception as e:
            st.session_state.calib = {"ok": False, "err": str(e)}
with colB:
    if st.button("🔁 Recalcular x (por si cambiaste toggles)", use_container_width=True):
        st.experimental_rerun()

# Mostrar resultados
if "calib" in st.session_state and st.session_state.calib.get("ok"):
    res = st.session_state.calib["res"]
    p = res.x
    if loss_kind == "Hiperbólica (α, Lmax)":
        alpha_b, Lmax_b = p
        pred = loss_hyperbolic(xvec, alpha_b, Lmax_b)
        st.success(f"α = {alpha_b:.4f} · Lmax = {Lmax_b:.2f} · {metric_kind} = {res.fun:.2f}")
        params_dict = {"alpha": alpha_b, "Lmax": Lmax_b}
    else:
        alpha_b, beta_b, Lmax_b = p
        pred = loss_exponential(xvec, alpha_b, beta_b, Lmax_b)
        st.success(f"α = {alpha_b:.4f} · β = {beta_b:.3f} · Lmax = {Lmax_b:.2f} · {metric_kind} = {res.fun:.2f}")
        params_dict = {"alpha": alpha_b, "beta": beta_b, "Lmax": Lmax_b}

    # Métricas adicionales
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    mae  = float(np.mean(np.abs(obs - pred)))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2)) if len(obs) > 1 else float("nan")
    R2 = (1 - ss_res/ss_tot) if (ss_tot and ss_tot != 0) else float("nan")

    st.markdown(f"**RMSE:** {rmse:.2f} · **MAE:** {mae:.2f} · **R²:** {R2:.3f}")

    # Tabla
    df_out = ens.copy()
    df_out["predicho"] = pred
    df_out["x_pc_pl_m2"] = xvec
    for k,v in params_dict.items(): df_out[k] = v
    df_out["RMSE"] = rmse; df_out["MAE"] = mae; df_out["R2"] = R2
    st.dataframe(df_out[["ensayo_id","loss_obs_pct","predicho","x_pc_pl_m2","MAX_PLANTS_CAP"]], use_container_width=True)

    # Gráfico Obs vs Pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_out["loss_obs_pct"], y=df_out["predicho"],
                             mode="markers+text", text=df_out["ensayo_id"],
                             textposition="top center", name="Ensayos"))
    # Línea 1:1
    mx = max(float(df_out["loss_obs_pct"].max()), float(df_out["predicho"].max())) if len(df_out) else 1.0
    fig.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines", name="1:1", line=dict(dash="dash")))
    # Recta de regresión pred ~ a + b*obs
    a,b,R2_line = fit_line(df_out["loss_obs_pct"], df_out["predicho"])
    fig.add_trace(go.Scatter(x=[0,mx], y=[a + b*0, a + b*mx], mode="lines", name=f"Regresión (R²={R2_line:.3f})"))

    ttl = f"Observado vs Predicho — {loss_kind} · Pesos {pc_weight_kind}"
    if use_ciec: ttl += " · con (1−Ciec)"
    fig.update_layout(title=ttl, xaxis_title="Observado (%)", yaxis_title="Predicho (%)", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Descargas
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar CSV", data=csv_bytes, file_name="calibracion_auc_pc_resultados.csv", mime="text/csv", use_container_width=True)
    with c2:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="resultados")
            summary = pd.DataFrame({**params_dict, **{"RMSE":[rmse], "MAE":[mae], "R2":[R2], "loss_kind":[loss_kind], "pc_weight":[pc_weight_kind], "use_ciec":[use_ciec]}})
            summary.to_excel(writer, index=False, sheet_name="resumen")
        st.download_button("📊 Descargar Excel", data=bio.getvalue(),
                           file_name="calibracion_auc_pc_resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3:
        # Exportar PNG del gráfico
        try:
            import kaleido  # para static image
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button("🖼️ Descargar PNG", data=png_bytes, file_name="obs_vs_pred.png", mime="image/png", use_container_width=True)
        except Exception:
            st.info("Instalá `kaleido` para habilitar descarga PNG del gráfico: `pip install -U kaleido`")

else:
    st.info("Configurá los toggles y presioná **🧠 Ejecutar calibración**.")


