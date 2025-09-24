# Create the updated Streamlit script (v2.1) with dynamic canopy suppression and LAIhc as a calibratable parameter.
# File will be saved to /mnt/data/calibra_v2_1.py

script = r'''# -*- coding: utf-8 -*-
# PREDWEEM · Calibración v2.1
# - Sin Ciec_const
# - Ca/Cs (densidades trigo) incluidas por ensayo
# - Supresión del cultivo dinámica: Ciec(t) = (LAI(t)/LAIhc) * (Ca/Cs), s(t) = 1 - Ciec(t)
# - A2_eff = A2_ctrl * g_eq, con g_eq = promedio ponderado de s(t) usando los mismos pesos de A2_ctrl
# - Parámetros a calibrar: k_loss, w_S1..w_S4, LAIhc
# - Objetivo: minimizar RMSE entre pérdida observada (%) y predicha (%)

import io, json, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------- Utilidades -----------------------
def to_date(x):
    if pd.isna(x): return None
    if isinstance(x, (dt.date, dt.datetime)): return x.date() if isinstance(x, dt.datetime) else x
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def logistic_between(days, start, end, y_max):
    if end <= start: end = start + 1
    t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
    return y_max/(1.0+np.exp(-r*(days-t_mid)))

def canopy_LAI_series(dates, siembra, t_lag=7, t_close=45, lai_max=3.5, k_beer=0.6, mode="LAI"):
    """Genera serie diaria de LAI o partir de cobertura usando Beer-Lambert."""
    days = np.array([(d - siembra).days for d in dates], dtype=float)
    if mode == "LAI":
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        return np.clip(LAI, 0.0, lai_max)
    else:
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, min(0.95, 0.99)))
        fc = np.clip(fc, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc, 1e-9, 1.0))/max(1e-6, k_beer)
        return np.clip(LAI, 0.0, lai_max)

# ----------------------- Plantilla Excel -----------------------
def build_excel_template_v2() -> bytes:
    ensayos = pd.DataFrame([{
        "ensayo_id": "E001",
        "sitio": "BahiaBlanca",
        "campaña": "2025",
        "cultivo": "Trigo",
        "fecha_siembra": "2025-09-20",
        "pc_ini": "2025-09-25",
        "pc_fin": "2025-11-15",
        "Ca": 250,
        "Cs": 250,
        "MAX_PLANTS_CAP": 250,
        "rend_testigo_kg_ha": 4000,
        "rend_observado_kg_ha": 3600
    }])
    tratamientos = pd.DataFrame([
        {"ensayo_id": "E001", "tipo": "preemergente", "fecha_aplicacion": "2025-09-20",
         "eficacia_pct": 90, "residual_dias": 10, "actua_s1": 1, "actua_s2": 1, "actua_s3": 0, "actua_s4": 0},
        {"ensayo_id": "E001", "tipo": "graminicida", "fecha_aplicacion": "2025-10-05",
         "eficacia_pct": 85, "residual_dias": 10, "actua_s1": 1, "actua_s2": 1, "actua_s3": 1, "actua_s4": 0},
    ])
    fechas = pd.date_range("2025-09-01", "2025-11-30", freq="D")
    mid = fechas[int(len(fechas)*0.35)]
    emer = []
    for f in fechas:
        w = 1 - abs((f - mid).days) / max(1, len(fechas)*0.35)
        emer.append(max(0.0, w))
    emer_rel = (np.array(emer) / np.sum(emer)).round(6)
    emergencia = pd.DataFrame({"ensayo_id": "E001", "fecha": fechas.date, "emer_rel": emer_rel})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ensayos.to_excel(writer, index=False, sheet_name="ensayos")
        tratamientos.to_excel(writer, index=False, sheet_name="tratamientos")
        emergencia.to_excel(writer, index=False, sheet_name="emergencia")
    buf.seek(0)
    return buf.getvalue()

# ----------------------- Núcleo de predicción -----------------------
def assign_state_by_age(age_days, s1=(1,6), s2=(7,27), s3=(28,59)):
    if age_days is None or age_days < s1[0]: return None
    if s1[0] <= age_days <= s1[1]: return 0
    if s2[0] <= age_days <= s2[1]: return 1
    if s3[0] <= age_days <= s3[1]: return 2
    return 3

def combine_efficacies(effs):
    if not effs: return 0.0
    prod = 1.0
    for e in effs:
        e = max(0.0, min(1.0, float(e)))
        prod *= (1 - e)
    return 1 - prod

def build_daily_control_mask(dates, states, ttos_rows):
    n = len(dates)
    eff = np.zeros((n,4), dtype=float)
    windows = []
    for r in ttos_rows:
        tipo = str(r.get("tipo","")).strip().lower()
        fapp = to_date(r.get("fecha_aplicacion"))
        ef = float(r.get("eficacia_pct", 0))/100.0
        res = int(r.get("residual_dias",0))
        mask_states = [int(r.get("actua_s1",0))>0,
                       int(r.get("actua_s2",0))>0,
                       int(r.get("actua_s3",0))>0,
                       int(r.get("actua_s4",0))>0]
        if fapp is None: 
            continue
        if tipo == "presiembra":
            start = fapp - dt.timedelta(days=max(0,res)); end = fapp
        elif tipo in ("preemergente","post_selectivo","graminicida"):
            start = fapp; end = fapp + dt.timedelta(days=max(0,res))
        else:
            continue
        windows.append((tipo, start, end, ef, mask_states))
    for i, d in enumerate(dates):
        effs_state = [[] for _ in range(4)]
        for tipo, start, end, ef, mask_states in windows:
            if start <= d <= end:
                for sidx in range(4):
                    if mask_states[sidx]: effs_state[sidx].append(ef)
        for sidx in range(4): eff[i, sidx] = combine_efficacies(effs_state[sidx])
    return eff

def predict_loss_for_ensayo(row_e, df_emerg, df_ttos,
                            s1, s2, s3,
                            k_loss, w_states, escala_A2=100.0,
                            # nuevos parámetros de canopia
                            t_lag=7, t_close=45, lai_max=3.5, k_beer=0.6, LAIhc=3.5, canopy_mode="LAI"):
    siembra = to_date(row_e["fecha_siembra"])
    pc_ini  = to_date(row_e["pc_ini"])
    pc_fin  = to_date(row_e["pc_fin"])
    max_cap = float(row_e.get("MAX_PLANTS_CAP", 250))

    Ca = float(row_e.get("Ca", np.nan))
    Cs = float(row_e.get("Cs", np.nan))
    if any(pd.isna(v) for v in [Ca, Cs]) or Cs <= 0:
        return {"ok": False, "msg": "Faltan Ca/Cs o Cs<=0"}

    if siembra is None or pc_ini is None or pc_fin is None:
        return {"ok": False, "msg": "Fechas inválidas (siembra/PC)"}

    df_emerg = df_emerg.copy()
    df_emerg["fecha"] = df_emerg["fecha"].apply(to_date)
    df_emerg = df_emerg.dropna(subset=["fecha","emer_rel"]).sort_values("fecha")
    if df_emerg.empty:
        return {"ok": False, "msg": "Emergencia vacía"}

    dates = list(df_emerg["fecha"].values)
    age   = [(d - siembra).days for d in dates]
    states= [assign_state_by_age(a, s1, s2, s3) for a in age]

    emer_rel = df_emerg["emer_rel"].values.astype(float)
    pl_day   = emer_rel * max_cap

    ttos_rows = df_ttos.to_dict(orient="records") if df_ttos is not None else []
    eff_by_state = build_daily_control_mask(dates, states, ttos_rows)
    pl_ctrl = pl_day.copy()
    for i in range(len(dates)):
        sidx = states[i]
        if sidx is None: 
            continue
        e = eff_by_state[i, sidx]
        pl_ctrl[i] *= (1.0 - e)

    pc_mask = np.array([1.0 if (pc_ini <= d <= pc_fin) else 0.0 for d in dates])

    wS = np.maximum(0, np.array(w_states, dtype=float))
    if wS.sum() == 0: wS = np.array([1,1,1,1], dtype=float)

    w_day = np.zeros(len(dates), dtype=float)
    for i in range(len(dates)):
        sidx = states[i]
        w_day[i] = 0.0 if sidx is None else wS[sidx]

    # A2 controlado
    A2_ctrl = float(np.sum(pl_ctrl * w_day * pc_mask))

    # -------- Supresión dinámica: s(t) = 1 - Ciec(t) --------
    LAI_t = canopy_LAI_series(dates, siembra, t_lag=t_lag, t_close=t_close, lai_max=lai_max, k_beer=k_beer, mode=canopy_mode)
    Ciec_t = np.clip((LAI_t / max(1e-6, float(LAIhc))) * (Ca / float(Cs)), 0.0, 1.0)
    s_t = 1.0 - Ciec_t  # 0..1

    # g_eq ponderado por los mismos pesos de A2_ctrl
    weights = pl_ctrl * w_day * pc_mask
    den = float(np.sum(weights))
    g_eq = float(np.sum(s_t * weights)) / den if den > 0 else 1.0

    # A2 efectivo y pérdida
    A2_eff = A2_ctrl * g_eq
    k = max(1e-6, float(k_loss))
    Loss_pred = 100.0 * (1.0 - math.exp(-k * (A2_eff / float(escala_A2))))

    return {
        "ok": True,
        "A2_ctrl": A2_ctrl,
        "A2_eff": A2_eff,
        "g_eq": g_eq,
        "Loss_pred": Loss_pred,
        "LAI_mean": float(np.mean(LAI_t)),
    }

# ----------------------- Calibración -----------------------
def loss_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2))) if len(y_true) else np.nan
    mae  = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
    return rmse, mae

def random_search(objective_fn, n_iter, bounds, seed=123):
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(int(n_iter)):
        params = {k: rng.uniform(lo, hi) for k,(lo,hi) in bounds.items()}
        score = objective_fn(params)
        if score < best_score:
            best, best_score = params, score
    return best, best_score

# ----------------------- UI -----------------------
st.set_page_config(page_title="PREDWEEM · Calibración v2.1", layout="wide")
st.title("PREDWEEM — Calibración v2.1 (supresión dinámica con LAIhc)")

with st.expander("1) Descargar plantilla Excel", expanded=True):
    tpl_bytes = build_excel_template_v2()
    st.download_button("⬇️ Descargar plantilla v2.xlsx", data=tpl_bytes,
                       file_name="plantilla_calibracion_v2.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("2) Subir Excel con datos", expanded=True):
    up = st.file_uploader("Cargar Excel (v2)", type=["xlsx"])
    if up is not None:
        try:
            xls = pd.ExcelFile(up)
            df_e = pd.read_excel(xls, "ensayos")
            df_t = pd.read_excel(xls, "tratamientos")
            df_m = pd.read_excel(xls, "emergencia")
            st.success(f"Cargadas hojas: ensayos({len(df_e)}), tratamientos({len(df_t)}), emergencia({len(df_m)})")
        except Exception as ex:
            st.error(f"Error al leer Excel: {ex}")
            df_e = df_t = df_m = None
    else:
        df_e = df_t = df_m = None

with st.expander("3) Configuración del modelo", expanded=True):
    colA, colB, colC, colD = st.columns(4)
    s1_ini = colA.number_input("S1 ini (d)", value=1, min_value=0, max_value=30)
    s1_fin = colB.number_input("S1 fin (d)", value=6, min_value=0, max_value=60)
    s2_ini = colC.number_input("S2 ini (d)", value=7, min_value=0, max_value=90)
    s2_fin = colD.number_input("S2 fin (d)", value=27, min_value=0, max_value=120)
    s3_ini = st.number_input("S3 ini (d)", value=28, min_value=0, max_value=180)
    s3_fin = st.number_input("S3 fin (d)", value=59, min_value=0, max_value=240)

    st.markdown("**Ponderación por estado (se calibran):**")
    c1,c2,c3,c4 = st.columns(4)
    w1_lo = c1.number_input("w_S1 min", value=0.5, step=0.1)
    w1_hi = c1.number_input("w_S1 max", value=2.0, step=0.1)
    w2_lo = c2.number_input("w_S2 min", value=0.5, step=0.1)
    w2_hi = c2.number_input("w_S2 max", value=2.0, step=0.1)
    w3_lo = c3.number_input("w_S3 min", value=0.5, step=0.1)
    w3_hi = c3.number_input("w_S3 max", value=2.0, step=0.1)
    w4_lo = c4.number_input("w_S4 min", value=0.1, step=0.1)
    w4_hi = c4.number_input("w_S4 max", value=1.5, step=0.1)

    st.markdown("**Curva de pérdida:**  Loss% = 100 × (1 − exp(−k_loss × A2_eff / escala_A2))")
    col1, col2 = st.columns(2)
    k_lo = col1.number_input("k_loss min", value=0.01, step=0.01, format="%.3f")
    k_hi = col1.number_input("k_loss max", value=0.50, step=0.01, format="%.3f")
    escala_A2 = col2.number_input("escala_A2 (normaliza A2)", value=100.0, step=10.0)

with st.expander("4) Parámetros de canopia", expanded=True):
    st.caption("Estos definen la dinámica LAI(t). LAIhc **se calibra** en el paso siguiente.")
    colx, coly, colz, colk = st.columns(4)
    t_lag = colx.number_input("t_lag (días a emergencia)", value=7, step=1)
    t_close = coly.number_input("t_close (días a cierre)", value=45, step=1)
    lai_max = colz.number_input("LAI máximo", value=3.5, step=0.1)
    k_beer = colk.number_input("k (Beer–Lambert)", value=0.6, step=0.05)
    canopy_mode = st.selectbox("Modo canopia", ["LAI","Cobertura→LAI (Beer)"], index=0)

with st.expander("5) Calibración", expanded=True):
    st.markdown("Se calibra `k_loss`, `w_S1..w_S4` y **`LAIhc`**.")
    n_iter = st.number_input("Iteraciones (búsqueda aleatoria)", value=2000, min_value=50, step=100)
    seed = st.number_input("Seed", value=123, step=1)

    # RANGO DE LAIhc A CALIBRAR
    colh1, colh2 = st.columns(2)
    LAIhc_lo = colh1.number_input("LAIhc min", value=2.0, step=0.1)
    LAIhc_hi = colh2.number_input("LAIhc max", value=6.0, step=0.1)

    run = st.button("🚀 Ejecutar calibración")

    if run and (df_e is None or df_m is None):
        st.warning("Cargá el Excel antes de calibrar.")

    if run and df_e is not None and df_m is not None:
        df_e = df_e.copy()
        df_t = df_t.copy() if df_t is not None else pd.DataFrame(columns=["ensayo_id"])
        df_m = df_m.copy()

        if "loss_obs_pct" not in df_e.columns:
            if {"rend_testigo_kg_ha","rend_observado_kg_ha"}.issubset(df_e.columns):
                df_e["loss_obs_pct"] = 100.0 * (1.0 - (df_e["rend_observado_kg_ha"] / df_e["rend_testigo_kg_ha"]))
            else:
                st.error("Falta 'loss_obs_pct' o (rend_testigo_kg_ha, rend_observado_kg_ha) en ensayos.")
                st.stop()

        bounds = {
            "k_loss": (k_lo, k_hi),
            "w_S1": (w1_lo, w1_hi),
            "w_S2": (w2_lo, w2_hi),
            "w_S3": (w3_lo, w3_hi),
            "w_S4": (w4_lo, w4_hi),
            "LAIhc": (LAIhc_lo, LAIhc_hi),
        }

        S1 = (int(s1_ini), int(s1_fin))
        S2 = (int(s2_ini), int(s2_fin))
        S3 = (int(s3_ini), int(s3_fin))

        def objective(params):
            k = params["k_loss"]
            LAIhc_p = params["LAIhc"]
            wS = [params["w_S1"], params["w_S2"], params["w_S3"], params["w_S4"]]
            preds, obs = [], []
            for _, row in df_e.iterrows():
                ens_id = str(row["ensayo_id"])
                sub_m = df_m[df_m["ensayo_id"]==ens_id]
                sub_t = df_t[df_t["ensayo_id"]==ens_id]
                res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, k, wS,
                                              escala_A2=escala_A2,
                                              t_lag=int(t_lag), t_close=int(t_close),
                                              lai_max=float(lai_max), k_beer=float(k_beer),
                                              LAIhc=float(LAIhc_p),
                                              canopy_mode="LAI" if canopy_mode=="LAI" else "COVER")
                if res["ok"]:
                    preds.append(res["Loss_pred"])
                    obs.append(float(row["loss_obs_pct"]))
            if not preds:
                return np.inf
            rmse,_ = loss_metrics(obs, preds)
            return rmse

        with st.spinner("Calibrando..."):
            best, best_rmse = random_search(objective, n_iter=n_iter, bounds=bounds, seed=int(seed))

        st.success("¡Listo!")
        st.write("**Mejores parámetros:**")
        st.json(best)
        st.write(f"**RMSE (global):** {best_rmse:.3f}")

        # Reconstruir resultados por ensayo
        k = best["k_loss"]
        LAIhc_p = best["LAIhc"]
        wS = [best["w_S1"], best["w_S2"], best["w_S3"], best["w_S4"]]
        rows = []
        for _, row in df_e.iterrows():
            ens_id = str(row["ensayo_id"])
            sub_m = df_m[df_m["ensayo_id"]==ens_id]
            sub_t = df_t[df_t["ensayo_id"]==ens_id]
            res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, k, wS,
                                          escala_A2=escala_A2,
                                          t_lag=int(t_lag), t_close=int(t_close),
                                          lai_max=float(lai_max), k_beer=float(k_beer),
                                          LAIhc=float(LAIhc_p),
                                          canopy_mode="LAI" if canopy_mode=="LAI" else "COVER")
            if not res["ok"]:
                rows.append({"ensayo_id": ens_id, "ok": False, "msg": res.get("msg","")})
                continue
            rows.append({
                "ensayo_id": ens_id,
                "ok": True,
                "A2_ctrl": res["A2_ctrl"],
                "g_eq": res["g_eq"],
                "A2_eff": res["A2_eff"],
                "Loss_obs_pct": float(row["loss_obs_pct"]),
                "Loss_pred_pct": res["Loss_pred"],
                "LAI_mean": res["LAI_mean"]
            })
        df_res = pd.DataFrame(rows)
        df_val = df_res[df_res["ok"]==True].copy()
        if not df_val.empty:
            rmse, mae = loss_metrics(df_val["Loss_obs_pct"], df_val["Loss_pred_pct"])
            st.write(f"**RMSE validación:** {rmse:.3f}  |  **MAE:** {mae:.3f}")
            st.dataframe(df_val[["ensayo_id","A2_ctrl","g_eq","A2_eff","Loss_obs_pct","Loss_pred_pct","LAI_mean"]].round(3),
                         use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_val["Loss_obs_pct"], y=df_val["Loss_pred_pct"],
                mode="markers+text", text=df_val["ensayo_id"], textposition="top center",
                name="Ensayos"
            ))
            maxi = float(max(df_val["Loss_obs_pct"].max(), df_val["Loss_pred_pct"].max(), 1.0))
            fig.add_trace(go.Scatter(x=[0,maxi], y=[0,maxi], mode="lines", name="1:1", line=dict(dash="dash")))
            fig.update_layout(title="Pérdida de rinde: Observado vs Predicho",
                              xaxis_title="Observado (%)", yaxis_title="Predicho (%)", height=500)
            st.plotly_chart(fig, use_container_width=True)

            csv = df_val.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Descargar resultados.csv", data=csv,
                               file_name="calibracion_resultados_v2_1.csv", mime="text/csv")
        else:
            st.warning("No hay ensayos válidos para reportar.")

with st.expander("Notas", expanded=False):
    st.markdown("""
- **Supresión dinámica:** se calcula **día a día** como `Ciec(t) = (LAI(t)/LAIhc) * (Ca/Cs)`, y la supresión usada en A2 es su promedio ponderado.
- **LAIhc** es ahora **parámetro calibrado** (escala de alta competitividad del cultivo).
- `t_lag`, `t_close`, `lai_max`, `k_beer` definen la forma de `LAI(t)` (ajustables en UI, fijos en calibración).
- Se mantienen `k_loss`, `w_S1..w_S4`, `escala_A2` como antes.
""")
'''

path = "/mnt/data/calibra_v2_1.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(script)

path

