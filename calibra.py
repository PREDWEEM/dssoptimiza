# -*- coding: utf-8 -*-
# calibra_auc_pc_logistic_app.py — Calibración PREDWEEM con AUC en PC (Logística por defecto)

import io, json, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

# ---------------- UI base ----------------
st.set_page_config(page_title="Calibración PREDWEEM (Logística + AUC-PC)", layout="wide")
st.title("Calibración — AUC en Período Crítico con curva Logística")

uploaded = st.file_uploader("Excel con hojas: ensayos, tratamientos, emergencia", type=["xlsx"])
st.caption("Se calculará x como **AUC(EMERREL diaria)** dentro del PC, opcionalmente ponderado y con competencia del cultivo.")

with st.sidebar:
    st.header("AUC en PC — Opciones")
    pc_weight_kind = st.selectbox("Pesos en PC", ["Uniforme", "Triangular (centro)", "Gaussiana (σ relativo)"], index=0)
    sigma_rel = st.slider("σ relativo (solo Gaussiana)", 0.05, 0.50, 0.20, 0.01, disabled=(pc_weight_kind!="Gaussiana"))

    st.divider()
    st.subheader("Competencia del cultivo (Ciec)")
    use_ciec = st.checkbox("Multiplicar por (1 − Ciec) antes del AUC", value=True)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag   = st.number_input("Días a emergencia (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 180, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 10.0, 3.5, 0.1)
    k_beer  = st.number_input("k Beer–Lambert", 0.10, 1.50, 0.60, 0.05)
    Ca      = st.number_input("Ca (pl/m²)", 50, 700, 250, 10)
    Cs      = st.number_input("Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc   = st.number_input("LAIhc", 0.5, 10.0, 3.5, 0.1)

    st.divider()
    st.subheader("Tratamientos")
    apply_trat = st.checkbox("Aplicar tratamientos como reducción residual simple", value=True)

    st.divider()
    st.subheader("Curva logística — Bounds")
    # Rango sugerido para x50 en pl·m2 (ajustalo a tu rango típico)
    x50_min = st.number_input("x50 mínimo", 0.0, 200.0, 1.0, 0.5)
    x50_max = st.number_input("x50 máximo", x50_min+0.1, 500.0, 50.0, 0.5)
    beta_min = st.number_input("β mínimo", 0.01, 5.00, 0.05, 0.01)
    beta_max = st.number_input("β máximo", beta_min+0.01, 10.00, 3.00, 0.01)
    Lmin = st.number_input("Lmax mínimo (%)", 5.0, 500.0, 40.0, 1.0)
    Lmax = st.number_input("Lmax máximo (%)", Lmin+1.0, 1000.0, 150.0, 1.0)

    metric_kind = st.selectbox("Métrica a minimizar", ["RMSE", "MAE"], index=0)

# ---------------- Utils ----------------
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
    days = (idx.date - sow_date.date()).astype("timedelta64[D]").astype(int).astype(float)
    def logistic_between(days, start, end, y_max):
        end = start+1 if end<=start else end
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))
    if mode_canopy == "Cobertura dinámica (%)":
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max/100.0))
        fc = np.clip(fc,0.0,1.0)
        LAI = -np.log(np.clip(1.0-fc,1e-9,1.0))/max(1e-6,k_beer)
        LAI = np.clip(LAI,0.0,lai_max)
    else:
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        LAI = np.clip(LAI,0.0,lai_max)
        fc = 1.0 - np.exp(-k_beer*LAI)
        fc = np.clip(fc,0.0,1.0)
    return fc, LAI

def ciec_series(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer, Ca, Cs, LAIhc):
    _, LAI = compute_canopy(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer)
    Ca_s = Ca if Ca>0 else 1e-6; Cs_s = Cs if Cs>0 else 1e-6
    Ciec = (LAI / max(1e-6, LAIhc)) * (Ca_s / Cs_s)
    return np.clip(Ciec, 0.0, 1.0)

def weights_pc(idx, pc_ini, pc_fin, kind="Uniforme", sigma_rel=0.20):
    t = idx.values.astype("datetime64[ns]").astype("int64")/86400e9
    t0 = pd.to_datetime(pc_ini).to_datetime64().astype("int64")/86400e9
    t1 = pd.to_datetime(pc_fin).to_datetime64().astype("int64")/86400e9
    mask = (t>=t0) & (t<=t1)
    w = np.zeros_like(t, float)
    if not mask.any(): return w
    ti = t[mask]; mid = 0.5*(t0+t1); span = max(1e-6, (t1-t0))
    if kind=="Uniforme":
        w[mask]=1.0
    elif kind.startswith("Triangular"):
        w[mask]=1.0 - np.abs((ti-mid)/(span/2.0)); w[mask]=np.clip(w[mask],0.0,1.0)
    else:
        sigma = sigma_rel*span
        w[mask]=np.exp(-0.5*((ti-mid)/max(1e-6,sigma))**2)
    # normalizar promedio≈1 dentro del PC
    m = w[mask].mean()
    if m>1e-9: w[mask]=w[mask]/m
    return w

def auc_with_weights(idx, values, weights):
    t = idx.values.astype("datetime64[ns]").astype("int64")/86400e9
    y = np.asarray(values, float) * np.asarray(weights, float)
    mask = weights>0
    if mask.sum()<2: return 0.0
    return float(np.trapz(y[mask], t[mask]))

def apply_treatments(series: pd.Series, trat_sub: pd.DataFrame) -> pd.Series:
    s = series.copy()
    if trat_sub is None or trat_sub.empty: return s
    for _, tr in trat_sub.iterrows():
        if pd.isna(tr.get("tipo")) or pd.isna(tr.get("fecha_aplicacion")) or pd.isna(tr.get("eficacia_pct")): continue
        f_app = pd.to_datetime(tr["fecha_aplicacion"])
        resd = int(tr["residual_dias"]) if not pd.isna(tr.get("residual_dias")) else 0
        eff = float(tr["eficacia_pct"])/100.0
        m = (s.index>=f_app) & (s.index<=f_app+pd.Timedelta(days=resd))
        s.loc[m] = s.loc[m]*(1.0-eff)
    return s

def loss_logistic(x, x50, beta, Lmax):
    x = np.asarray(x, float)
    return Lmax / (1.0 + np.exp(-beta*(x - x50)))

# ---------------- Carga y cálculo de x ----------------
if uploaded is None:
    st.info("Subí el Excel para continuar.")
    st.stop()

try:
    ens = pd.read_excel(uploaded, sheet_name="ensayos")
    trat = pd.read_excel(uploaded, sheet_name="tratamientos")
    emer = pd.read_excel(uploaded, sheet_name="emergencia")
except Exception as e:
    st.error(f"No pude leer el Excel: {e}")
    st.stop()

for c in ["fecha_siembra","pc_ini","pc_fin"]:
    ens[c] = pd.to_datetime(ens[c], errors="coerce")
emer["fecha"] = pd.to_datetime(emer["fecha"], errors="coerce")

if not {"ensayo_id","loss_obs_pct","MAX_PLANTS_CAP"}.issubset(ens.columns):
    st.error("La hoja 'ensayos' debe incluir: ensayo_id, loss_obs_pct, MAX_PLANTS_CAP, fecha_siembra, pc_ini, pc_fin.")
    st.stop()

def compute_x_for_row(row):
    ens_id = row["ensayo_id"]
    f_sow = pd.to_datetime(row["fecha_siembra"])
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    max_cap = float(row["MAX_PLANTS_CAP"])
    sub = emer[emer["ensayo_id"]==ens_id][["fecha","emer_rel"]].dropna().copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_fin): return 0.0
    s = interp_daily(sub, f_sow, pc_fin)
    # (1-Ciec)
    if use_ciec:
        Ciec = ciec_series(s.index, f_sow, mode_canopy, int(t_lag), int(t_close), float(cov_max),
                           float(lai_max), float(k_beer), float(Ca), float(Cs), float(LAIhc))
        s = s*(1.0 - Ciec)
    # Tratamientos
    if apply_trat:
        tsub = trat[trat["ensayo_id"]==ens_id]
        s = apply_treatments(s, tsub)
    # Factor (siembra→pc_fin)
    w_all = np.ones(len(s), float)
    auc_all = auc_with_weights(s.index, s.values, w_all)
    if auc_all<=0.0: return 0.0
    factor = max_cap/auc_all
    # Pesos PC
    w_pc = weights_pc(s.index, pc_ini, pc_fin, pc_weight_kind, float(sigma_rel))
    x_pc = factor*auc_with_weights(s.index, s.values, w_pc)
    return float(x_pc)

with st.spinner("Calculando x (AUC-PC) por ensayo…"):
    ens = ens.copy()
    ens["x_pc"] = ens.apply(compute_x_for_row, axis=1)

obs = ens["loss_obs_pct"].to_numpy(float)
xvec = ens["x_pc"].to_numpy(float)

# ---------------- Calibración logística ----------------
def objective(params):
    x50, beta, Lmx = params
    pred = loss_logistic(xvec, x50, beta, Lmx)
    if metric_kind=="RMSE":
        return float(np.sqrt(np.mean((obs - pred)**2)))
    else:
        return float(np.mean(np.abs(obs - pred)))

colA, colB = st.columns([1,1])
with colA:
    x50_init = np.median(xvec) if np.isfinite(np.median(xvec)) else max(x50_min, 1.0)
    beta_init = max(beta_min + 0.01, 0.5)
    L_init = min(max(Lmin+1.0, 100.0), Lmax)
    if st.button("🧠 Ejecutar calibración (Logística)", use_container_width=True):
        bounds = [(float(x50_min), float(x50_max)),
                  (float(beta_min), float(beta_max)),
                  (float(Lmin), float(Lmax))]
        res = minimize(objective, x0=[x50_init, beta_init, L_init], bounds=bounds, method="L-BFGS-B")
        st.session_state.calib = {"ok": True, "res": res}
with colB:
    if st.button("🔁 Recalcular x (si cambiaste toggles)", use_container_width=True):
        st.experimental_rerun()

st.markdown("---")

if "calib" in st.session_state and st.session_state.calib.get("ok"):
    res = st.session_state.calib["res"]
    x50_b, beta_b, Lmax_b = [float(v) for v in res.x]
    pred = loss_logistic(xvec, x50_b, beta_b, Lmax_b)
    rmse = float(np.sqrt(np.mean((obs - pred)**2)))
    mae  = float(np.mean(np.abs(obs - pred)))
    ss_res = float(np.sum((obs - pred)**2))
    ss_tot = float(np.sum((obs - np.mean(obs))**2)) if len(obs)>1 else float("nan")
    R2 = (1 - ss_res/ss_tot) if (ss_tot and ss_tot!=0) else float("nan")

    st.success(f"x50 = {x50_b:.3f} · β = {beta_b:.3f} · Lmax = {Lmax_b:.2f} · {metric_kind} = {res.fun:.2f}")
    st.markdown(f"**RMSE:** {rmse:.2f} · **MAE:** {mae:.2f} · **R²:** {R2:.3f}")

    # Tabla de salidas
    df_out = ens.copy()
    df_out["predicho"] = pred
    df_out.rename(columns={"x_pc": "x_pc_pl_m2"}, inplace=True)
    st.dataframe(df_out[["ensayo_id","loss_obs_pct","predicho","x_pc_pl_m2","MAX_PLANTS_CAP"]], use_container_width=True)

    # Gráfico obs vs pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_out["loss_obs_pct"], y=df_out["predicho"],
                             mode="markers+text", text=df_out["ensayo_id"], textposition="top center",
                             name="Ensayos"))
    mx = max(float(df_out["loss_obs_pct"].max()), float(df_out["predicho"].max()), 1.0)
    fig.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines", name="1:1", line=dict(dash="dash")))
    fig.update_layout(title="Observado vs Predicho — Logística",
                      xaxis_title="Observado (%)", yaxis_title="Predicho (%)",
                      margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Descargas: CSV y Excel
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar CSV", data=csv_bytes, file_name="calibracion_logistica_resultados.csv", mime="text/csv", use_container_width=True)
    with c2:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="resultados")
            pd.DataFrame({"x50":[x50_b],"beta":[beta_b],"Lmax":[Lmax_b],
                          "RMSE":[rmse], "MAE":[mae], "R2":[R2],
                          "pc_weights":[pc_weight_kind], "use_ciec":[use_ciec]}).to_excel(writer, index=False, sheet_name="resumen")
        st.download_button("📊 Descargar Excel", data=bio.getvalue(),
                           file_name="calibracion_logistica_resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3:
        # Export JSON de parámetros
        params = {
            "loss_kind": "logistic",
            "x50": x50_b, "beta": beta_b, "Lmax": Lmax_b,
            "use_ciec": use_ciec,
            "pc_weights": pc_weight_kind,
            "canopy": {"mode": mode_canopy, "t_lag": int(t_lag), "t_close": int(t_close),
                       "cov_max": float(cov_max), "lai_max": float(lai_max), "k_beer": float(k_beer),
                       "Ca": int(Ca), "Cs": int(Cs), "LAIhc": float(LAIhc)}
        }
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        st.download_button("🧾 Descargar JSON (parámetros)", data=params_json.encode("utf-8"),
                           file_name="predweem_loss_params.json", mime="application/json",
                           use_container_width=True)

else:
    st.info("Ajustá bounds y presioná **🧠 Ejecutar calibración (Logística)**.")


