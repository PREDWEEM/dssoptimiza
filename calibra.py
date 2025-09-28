# -*- coding: utf-8 -*-
# app_calibra_estados_hiperbolica.py
# Streamlit • Calibración con x ponderada por estados (S1–S4) + función hiperbólica
# Requiere: pandas, numpy, scipy, plotly, xlsxwriter  (y opcionalmente kaleido para PNG)

import io, json, numpy as np, pandas as pd, streamlit as st
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="Calibración PREDWEEM (Estados + Hiperbólica)", layout="wide")
st.title("Calibración — x ponderada por estados (S1–S4) + función hiperbólica")

uploaded = st.file_uploader("📥 Subí tu Excel con hojas: `ensayos` y `emergencia`", type=["xlsx"])
st.caption("Hojas mínimas: **ensayos** (ensayo_id, loss_obs_pct, MAX_PLANTS_CAP, fecha_siembra, pc_ini, pc_fin) "
           "y **emergencia** (ensayo_id, fecha, emer_rel en 0–1 o %).")

# ---------------- Sidebar: opciones ----------------
with st.sidebar:
    st.header("Opciones")
    use_ciec = st.checkbox("Aplicar competencia del cultivo (1 − Ciec)", value=True)
    st.subheader("Parámetros de canopia (si usás Ciec)")
    mode_canopy = st.selectbox("Modelo de canopia", ["Cobertura (%)", "LAI dinámico"], index=0)
    t_lag   = st.number_input("Días a emergencia (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 180, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máx", 0.0, 10.0, 3.5, 0.1)
    k_beer  = st.number_input("k Beer–Lambert", 0.10, 1.50, 0.60, 0.05)
    Ca      = st.number_input("Ca (pl·m²)", 50, 700, 250, 10)
    Cs      = st.number_input("Cs (pl·m²)", 50, 700, 250, 10)
    LAIhc   = st.number_input("LAIhc", 0.5, 10.0, 3.5, 0.1)

    st.subheader("Pesos por estado")
    w_s1 = st.number_input("w_S1", 0.0, 2.0, 0.0, 0.05)
    w_s2 = st.number_input("w_S2", 0.0, 2.0, 0.3, 0.05)
    w_s3 = st.number_input("w_S3", 0.0, 2.0, 0.6, 0.05)
    w_s4 = st.number_input("w_S4", 0.0, 2.0, 1.0, 0.05)

    st.subheader("Iniciales y límites (hiperbólica)")
    alpha0 = st.number_input("α inicial", 1e-4, 5.0, 0.05, 0.01, format="%.4f")
    Lmax0  = st.number_input("Lmax inicial (%)", 5.0, 300.0, 80.0, 1.0)
    st.caption("Bounds fijos: α∈[1e-4, 5.0], Lmax∈[10, 300]")

    run = st.button("🧠 Ejecutar calibración", use_container_width=True)

# ---------------- Utilidades ----------------
def daily_series(sub_emerg: pd.DataFrame, start, end) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    s = sub_emerg.set_index("fecha")["emer_rel"].reindex(idx).interpolate("linear").fillna(0.0)
    s.index.name = "fecha"
    return s

def auc(idx: pd.DatetimeIndex, values: np.ndarray) -> float:
    if len(idx) < 2: return 0.0
    t = idx.values.astype("datetime64[ns]").astype("int64")/86400e9
    y = np.asarray(values, float)
    return float(np.trapz(y, t))

def compute_canopy(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                   mode_canopy: str, t_lag: int, t_close: int,
                   cov_max: float, lai_max: float, k_beer: float):
    days = (idx.date - sow_date.date()).astype("timedelta64[D]").astype(int).astype(float)
    def logistic_between(days_, start, end, y_max):
        end = start+1 if end<=start else end
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0 + np.exp(-r*(days_ - t_mid)))
    if mode_canopy == "Cobertura (%)":
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100.0))
        fc = np.clip(fc, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0-fc, 1e-9, 1.0))/max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc = 1.0 - np.exp(-k_beer*LAI)
        fc = np.clip(fc, 0.0, 1.0)
    return fc, LAI

def ciec_series(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                mode_canopy: str, t_lag: int, t_close: int,
                cov_max: float, lai_max: float, k_beer: float,
                Ca: float, Cs: float, LAIhc: float):
    _, LAI = compute_canopy(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer)
    Ca_s = Ca if Ca>0 else 1e-6
    Cs_s = Cs if Cs>0 else 1e-6
    Ciec = (LAI / max(1e-6, LAIhc)) * (Ca_s / Cs_s)
    return np.clip(Ciec, 0.0, 1.0)

def non_overlapping_states(emer_daily: pd.Series) -> pd.DataFrame:
    """Estados por edad (no solapados): S1 0–6, S2 7–27, S3 28–59, S4 ≥60 (en pl·m²/día)."""
    v = emer_daily.to_numpy(float)
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])
    def sum_age_window(i, a, b):
        lo, hi = i - b, i - a
        if hi < 0: return 0.0
        lo_c, hi_c = max(0, lo), min(n-1, hi)
        if lo_c > hi_c: return 0.0
        return c[hi_c+1] - c[lo_c]
    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for i in range(n):
        S1[i] = sum_age_window(i, 0, 6)
        S2[i] = sum_age_window(i, 7, 27)
        S3[i] = sum_age_window(i, 28, 59)
        j = i - 60
        S4[i] = c[j+1] if j >= 0 else 0.0
    return pd.DataFrame({"S1": S1, "S2": S2, "S3": S3, "S4": S4}, index=emer_daily.index)

def loss_hyperbolic(x, alpha, Lmax):
    x = np.asarray(x, float)
    return (alpha * x) / (1.0 + (alpha * x / Lmax))

def calibrate_hyperbolic(x, y, alpha0=0.05, Lmax0=80.0):
    def objective(p):
        a, L = p
        yhat = loss_hyperbolic(x, a, L)
        return np.mean((y - yhat)**2)
    res = minimize(objective, x0=[alpha0, Lmax0],
                   bounds=[(1e-4, 5.0), (10.0, 300.0)],
                   method="L-BFGS-B")
    a, L = float(res.x[0]), float(res.x[1])
    yhat = loss_hyperbolic(x, a, L)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae  = float(np.mean(np.abs(y - yhat)))
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else float("nan")
    r2 = (1 - ss_res/ss_tot) if (ss_tot and ss_tot != 0) else float("nan")
    return {"alpha": a, "Lmax": L, "rmse": rmse, "mae": mae, "r2": r2, "yhat": yhat}

def compute_x_states_for_row(row, emer_df: pd.DataFrame, use_ciec: bool,
                             mode_canopy: str, t_lag: int, t_close: int, cov_max: float,
                             lai_max: float, k_beer: float, Ca: float, Cs: float, LAIhc: float,
                             w_states: dict):
    ens_id = row["ensayo_id"]
    f_sow  = pd.to_datetime(row["fecha_siembra"])
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    cap    = float(row["MAX_PLANTS_CAP"])

    sub = emer_df[emer_df["ensayo_id"] == ens_id][["fecha","emer_rel"]].dropna().copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_fin):
        return 0.0

    # Convertir % a 0..1 si fuese necesario (heurística)
    if sub["emer_rel"].max() > 1.5:
        sub["emer_rel"] = sub["emer_rel"] / 100.0

    s = daily_series(sub, f_sow, pc_fin)     # EMERREL diaria (0..1)
    auc_total = auc(s.index, s.values)
    if auc_total <= 0:
        return 0.0
    factor = cap / auc_total                 # a pl·m²/día

    if use_ciec:
        Ciec = ciec_series(s.index, f_sow, mode_canopy, int(t_lag), int(t_close),
                           float(cov_max), float(lai_max), float(k_beer),
                           float(Ca), float(Cs), float(LAIhc))
        daily_eff = s.values * (1.0 - Ciec) * factor
    else:
        daily_eff = s.values * factor

    states = non_overlapping_states(pd.Series(daily_eff, index=s.index))
    D = (states["S1"]*w_states["S1"] +
         states["S2"]*w_states["S2"] +
         states["S3"]*w_states["S3"] +
         states["S4"]*w_states["S4"]).to_numpy()

    mask_pc = (s.index >= pc_ini) & (s.index <= pc_fin)
    x_states = auc(s.index[mask_pc], D[mask_pc])
    return float(x_states)

# ---------------- Ejecución ----------------
if not uploaded:
    st.info("Subí el Excel para habilitar la calibración.")
    st.stop()

try:
    ens = pd.read_excel(uploaded, sheet_name="ensayos")
    emer = pd.read_excel(uploaded, sheet_name="emergencia")
except Exception as e:
    st.error(f"No pude leer el Excel: {e}")
    st.stop()

for c in ["fecha_siembra","pc_ini","pc_fin"]:
    ens[c] = pd.to_datetime(ens[c], errors="coerce")
emer["fecha"] = pd.to_datetime(emer["fecha"], errors="coerce")

need_ens = {"ensayo_id","loss_obs_pct","MAX_PLANTS_CAP","fecha_siembra","pc_ini","pc_fin"}
need_em  = {"ensayo_id","fecha","emer_rel"}
if not need_ens.issubset(ens.columns) or not need_em.issubset(emer.columns):
    st.error("Estructura inválida. Revisá columnas requeridas en 'ensayos' y 'emergencia'.")
    st.stop()

# Calcular x (estados)
w_states = {"S1": w_s1, "S2": w_s2, "S3": w_s3, "S4": w_s4}
with st.spinner("Calculando densidad efectiva x (ponderada por estados) por ensayo…"):
    xs = ens.apply(
        compute_x_states_for_row,
        axis=1,
        emer_df=emer,
        use_ciec=use_ciec,
        mode_canopy=mode_canopy,
        t_lag=int(t_lag), t_close=int(t_close),
        cov_max=float(cov_max), lai_max=float(lai_max), k_beer=float(k_beer),
        Ca=float(Ca), Cs=float(Cs), LAIhc=float(LAIhc),
        w_states=w_states
    )

ens = ens.copy()
ens["x_pl_m2_states"] = xs.astype(float)
y_obs = ens["loss_obs_pct"].to_numpy(float)
x_val = ens["x_pl_m2_states"].to_numpy(float)

# Ejecutar calibración
if run:
    fit = calibrate_hyperbolic(x_val, y_obs, alpha0=float(alpha0), Lmax0=float(Lmax0))
    a_b, L_b = fit["alpha"], fit["Lmax"]
    y_hat = fit["yhat"]

    st.success(f"Calibración OK · α = {a_b:.4f} · Lmax = {L_b:.2f} · "
               f"RMSE = {fit['rmse']:.2f} · MAE = {fit['mae']:.2f} · R² = {fit['r2']:.3f}")

    # ---------- Tabla ----------
    out = ens[["ensayo_id","loss_obs_pct","x_pl_m2_states","MAX_PLANTS_CAP"]].copy()
    out["predicho"] = y_hat
    out["alpha"] = float(a_b)
    out["Lmax"]  = float(L_b)
    out["RMSE"]  = float(fit["rmse"])
    out["MAE"]   = float(fit["mae"])
    out["R2"]    = float(fit["r2"])

    st.subheader("Tabla — Observado vs Predicho")
    st.dataframe(out, use_container_width=True)

    # ---------- Gráfico Observado vs Predicho ----------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=out["loss_obs_pct"], y=out["predicho"],
        mode="markers+text", text=out["ensayo_id"], textposition="top center",
        name="Ensayos"
    ))
    mx = max(float(out["loss_obs_pct"].max()), float(out["predicho"].max()), 1.0)
    fig.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines", name="1:1", line=dict(dash="dash")))
    fig.update_layout(
        title="Observado vs Predicho — x (estados) + hiperbólica",
        xaxis_title="Observado (%)", yaxis_title="Predicho (%)",
        margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Curva pérdida vs densidad ----------
    x_grid = np.linspace(0.0, max(1.0, float(np.nanmax(x_val))*1.25), 400)
    y_curve = loss_hyperbolic(x_grid, a_b, L_b)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_grid, y=y_curve, mode="lines", name="Curva hiperbólica ajustada"))
    fig2.add_trace(go.Scatter(x=x_val, y=y_obs, mode="markers", name="Observado (ensayos)"))
    fig2.update_layout(
        title=f"Función de pérdida — α={a_b:.4f}, Lmax={L_b:.1f}",
        xaxis_title="x (pl·m²) — densidad efectiva ponderada por estados",
        yaxis_title="Pérdida de rinde (%)",
        margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- Descargas ----------
    st.subheader("Descargas")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.download_button(
            "📥 CSV resultados",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="calibracion_final_estados_hiperbolica.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            out.to_excel(writer, sheet_name="resultados", index=False)
            pd.DataFrame({
                "alpha":[a_b], "Lmax":[L_b],
                "RMSE":[fit["rmse"]], "MAE":[fit["mae"]], "R2":[fit["r2"]],
                "use_ciec":[use_ciec],
                "w_S1":[w_s1], "w_S2":[w_s2], "w_S3":[w_s3], "w_S4":[w_s4],
                "canopy_mode":[mode_canopy], "t_lag":[t_lag], "t_close":[t_close],
                "cov_max":[cov_max], "lai_max":[lai_max], "k_beer":[k_beer],
                "Ca":[Ca], "Cs":[Cs], "LAIhc":[LAIhc]
            }).to_excel(writer, sheet_name="resumen", index=False)
        st.download_button(
            "📊 Excel resultados",
            data=bio.getvalue(),
            file_name="calibracion_final_estados_hiperbolica.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with c3:
        # Export PNG de Observado vs Predicho (requiere kaleido instalado)
        try:
            png1 = fig.to_image(format="png", scale=2)
            st.download_button("🖼️ PNG Obs vs Pred", data=png1, file_name="obs_vs_pred_estados_hiperbolica.png",
                               mime="image/png", use_container_width=True)
        except Exception:
            st.info("Para exportar PNG instalá `kaleido`:  `pip install -U kaleido`")

    with c4:
        # Export JSON con parámetros y configuración
        params = {
            "loss_kind": "hyperbolic",
            "alpha": float(a_b), "Lmax": float(L_b),
            "use_ciec": bool(use_ciec),
            "state_weights": {"S1": float(w_s1), "S2": float(w_s2), "S3": float(w_s3), "S4": float(w_s4)},
            "canopy": {
                "mode": mode_canopy, "t_lag": int(t_lag), "t_close": int(t_close),
                "cov_max": float(cov_max), "lai_max": float(lai_max),
                "k_beer": float(k_beer), "Ca": float(Ca), "Cs": float(Cs), "LAIhc": float(LAIhc)
            }
        }
        st.download_button(
            "🧾 JSON parámetros",
            data=json.dumps(params, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="predweem_config.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("Cargá el Excel y presioná **🧠 Ejecutar calibración**.")
