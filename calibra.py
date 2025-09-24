# -*- coding: utf-8 -*-
# PREDWEEM · Calibración v2.4.1 (patched export)
# - Acumulación desde siembra (lote limpio pre-siembra)
# - Opción para calibrar α y Lmax (curva hipérbola rectangular)
# - Descargas en memoria (no escribe a disco en la app)

import io, json, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============== Utilidades de descarga en memoria (sin disco) ==============
def offer_download_text(filename: str, text: str, label: str | None = None):
    st.download_button(
        label or f"⬇️ Descargar {filename}",
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain"
    )

def offer_download_bytes(filename: str, data: bytes, label: str | None = None, mime: str | None = None):
    st.download_button(
        label or f"⬇️ Descargar {filename}",
        data=data,
        file_name=filename,
        mime=mime or "application/octet-stream"
    )

# ----------------------- Utilidades varias -----------------------
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
    days = np.array([(d - siembra).days for d in dates], dtype=float)
    if mode == "LAI":
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        return np.clip(LAI, 0.0, lai_max)
    else:
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, min(0.95, 0.99)))
        fc = np.clip(fc, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc, 1e-9, 1.0))/max(1e-6, k_beer)
        return np.clip(LAI, 0.0, lai_max)

# ----------------------- Plantilla Excel base -----------------------
def build_excel_template_v2() -> bytes:
    ensayos = pd.DataFrame([{
        "ensayo_id": "E001",
        "sitio": "BahiaBlanca",
        "campaña": "2025",
        "cultivo": "Trigo",
        "fecha_siembra": "2025-06-15",
        "pc_ini": "2025-06-22",
        "pc_fin": "2025-08-31",
        "Ca": 250,
        "Cs": 250,
        "MAX_PLANTS_CAP": 250,
        "rend_testigo_kg_ha": 4000,
        "rend_observado_kg_ha": 3600
    }])
    tratamientos = pd.DataFrame([
        {"ensayo_id": "E001", "tipo": "preemergente", "fecha_aplicacion": "2025-06-15",
         "eficacia_pct": 90, "residual_dias": 10, "actua_s1": 1, "actua_s2": 1, "actua_s3": 0, "actua_s4": 0},
        {"ensayo_id": "E001", "tipo": "graminicida", "fecha_aplicacion": "2025-07-01",
         "eficacia_pct": 85, "residual_dias": 10, "actua_s1": 1, "actua_s2": 1, "actua_s3": 1, "actua_s4": 0},
    ])
    fechas = pd.date_range("2025-05-01", "2025-09-15", freq="D")
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

# ----------------------- Generador sintético v2 -----------------------
def build_synthetic_excel_v2(n_trials=8, start_all=dt.date(2025,5,1), end_all=dt.date(2025,9,15),
                             Cs_choices=(220,250,280,300), cap_choices=(250,300),
                             loss_range=(6,28), seed=20250924) -> tuple[bytes, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    trial_ids = [f"E{i:03d}" for i in range(1, n_trials+1)]
    all_dates = pd.date_range(start_all, end_all, freq="D").date

    ensayos_rows = []; trat_rows = []; emer_blocks = []
    for i, eid in enumerate(trial_ids):
        siembra = start_all + timedelta(days=int(15 + 4*i))
        pc_ini  = siembra + timedelta(days=int(7 + rng.integers(0, 3)))
        pc_fin  = siembra + timedelta(days=int(75 + rng.integers(0, 6)))

        Cs = int(rng.choice(Cs_choices))
        Ca = int(max(120, float(rng.normal(Cs, 25))))

        MAX_PLANTS_CAP = int(rng.choice(cap_choices))
        rend_testigo = int(float(rng.normal(4200, 200)))
        loss_target = float(rng.uniform(*loss_range))
        rend_obs = round(rend_testigo * (1 - loss_target/100.0), 1)

        ensayos_rows.append({
            "ensayo_id": eid, "sitio": f"S{i+1}", "campaña": "2025", "cultivo": "Trigo",
            "fecha_siembra": siembra.isoformat(), "pc_ini": pc_ini.isoformat(), "pc_fin": pc_fin.isoformat(),
            "Ca": Ca, "Cs": Cs, "MAX_PLANTS_CAP": MAX_PLANTS_CAP,
            "rend_testigo_kg_ha": rend_testigo, "rend_observado_kg_ha": rend_obs,
            "loss_obs_pct": round(loss_target, 2)
        })

        trat_rows.append({
            "ensayo_id": eid, "tipo": "preemergente",
            "fecha_aplicacion": siembra.isoformat(),
            "eficacia_pct": int(float(rng.uniform(78, 92))),
            "residual_dias": int(float(rng.uniform(7, 12))),
            "actua_s1": 1, "actua_s2": 1, "actua_s3": 0, "actua_s4": 0
        })
        trat_rows.append({
            "ensayo_id": eid, "tipo": "graminicida",
            "fecha_aplicacion": (siembra + timedelta(days=int(float(rng.uniform(10, 25))))).isoformat(),
            "eficacia_pct": int(float(rng.uniform(72, 90))),
            "residual_dias": int(float(rng.uniform(7, 14))),
            "actua_s1": 1, "actua_s2": 1, "actua_s3": 1, "actua_s4": 0
        })

        peak = siembra + timedelta(days=int(float(rng.uniform(7, 18))))
        sigma = float(rng.uniform(8, 14))
        vals = []
        for d in all_dates:
            x = (d - peak).days
            vals.append(math.exp(-0.5 * (x/sigma)**2))
        vals = np.array(vals, dtype=float)
        emer_rel = (vals / vals.sum()).astype(float)
        emer_blocks.append(pd.DataFrame({"ensayo_id": eid, "fecha": list(all_dates), "emer_rel": emer_rel}))

    df_ens = pd.DataFrame(ensayos_rows)
    df_trt = pd.DataFrame(trat_rows)
    df_emg = pd.concat(emer_blocks, ignore_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_ens.to_excel(writer, index=False, sheet_name="ensayos")
        df_trt.to_excel(writer, index=False, sheet_name="tratamientos")
        df_emg.to_excel(writer, index=False, sheet_name="emergencia")
    buf.seek(0)
    return buf.getvalue(), df_ens

# ----------------------- Curva de pérdida (hipérbola rectangular) -----------------------
def loss_pct_rect_hyperbola(x, alpha=0.375, Lmax=76.639):
    x = np.asarray(x, dtype=float)
    return (alpha * x) / (1.0 + (alpha * x / Lmax))

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
                            w_states,
                            t_lag=7, t_close=45, lai_max=3.5, k_beer=0.6, LAIhc=3.5, canopy_mode="LAI",
                            alpha=0.375, Lmax=76.639,
                            cut_at_pc_fin=True):
    siembra = to_date(row_e["fecha_siembra"])
    pc_ini  = to_date(row_e["pc_ini"])
    pc_fin  = to_date(row_e["pc_fin"])
    max_cap = float(row_e.get("MAX_PLANTS_CAP", 250))

    Ca = float(row_e.get("Ca", np.nan))
    Cs = float(row_e.get("Cs", np.nan))
    if any(pd.isna(v) for v in [Ca, Cs]) or Cs <= 0:
        return {"ok": False, "msg": "Faltan Ca/Cs o Cs<=0"}

    if siembra is None:
        return {"ok": False, "msg": "Falta fecha de siembra"}

    df_emerg = df_emerg.copy()
    df_emerg["fecha"] = df_emerg["fecha"].apply(to_date)
    df_emerg = df_emerg.dropna(subset=["fecha","emer_rel"]).sort_values("fecha")
    if df_emerg.empty:
        return {"ok": False, "msg": "Emergencia vacía"}

    dates = list(df_emerg["fecha"].values)
    age   = [(d - siembra).days for d in dates]
    states= [assign_state_by_age(a, s1, s2, s3) for a in age]

    emer_rel = df_emerg["emer_rel"].values.astype(float)

    # ======= Solo cuenta desde siembra (lote limpio antes) =======
    mask_since = np.array([d >= siembra for d in dates], dtype=float)  # piso en t=0
    if cut_at_pc_fin and (pc_fin is not None):
        mask_until = np.array([d <= pc_fin for d in dates], dtype=float)
    else:
        mask_until = np.ones_like(mask_since)
    eval_mask  = mask_since * mask_until

    pl_day = (emer_rel * max_cap) * mask_since

    ttos_rows = df_ttos.to_dict(orient="records") if df_ttos is not None else []
    eff_by_state = build_daily_control_mask(dates, states, ttos_rows)
    pl_ctrl = pl_day.copy()
    for i in range(len(dates)):
        sidx = states[i]
        if sidx is None:
            continue
        e = eff_by_state[i, sidx]
        pl_ctrl[i] *= (1.0 - e)

    pc_mask = eval_mask

    wS = np.maximum(0, np.array(w_states, dtype=float))
    if wS.sum() == 0: wS = np.array([1,1,1,1], dtype=float)

    w_day = np.zeros(len(dates), dtype=float)
    for i in range(len(dates)):
        sidx = states[i]
        w_day[i] = 0.0 if sidx is None else wS[sidx]

    A2_ctrl = float(np.sum(pl_ctrl * w_day * pc_mask))

    LAI_t = canopy_LAI_series(dates, siembra, t_lag=t_lag, t_close=t_close, lai_max=lai_max, k_beer=k_beer, mode=canopy_mode)
    Ciec_t = np.clip((LAI_t / max(1e-6, float(LAIhc))) * (Ca / float(Cs)), 0.0, 1.0)
    s_t = 1.0 - Ciec_t

    weights = pl_ctrl * w_day * pc_mask
    den = float(np.sum(weights))
    g_eq = float(np.sum(s_t * weights)) / den if den > 0 else 1.0

    A2_eff = A2_ctrl * g_eq
    Loss_pred = float(loss_pct_rect_hyperbola(A2_eff, alpha=alpha, Lmax=Lmax))

    return {"ok": True, "A2_ctrl": A2_ctrl, "A2_eff": A2_eff, "g_eq": g_eq,
            "Loss_pred": Loss_pred, "LAI_mean": float(np.mean(LAI_t))}

# ----------------------- Calibración helpers -----------------------
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
st.set_page_config(page_title="PREDWEEM · Calibración v2.4.1", layout="wide")
st.title("PREDWEEM — Calibración v2.4.1 (acumulación desde siembra; lote limpio pre-siembra)")

# 1) Plantilla y 1b) Generador sintético
with st.expander("1) Descargar plantilla Excel", expanded=False):
    tpl_bytes = build_excel_template_v2()
    offer_download_bytes("plantilla_calibracion_v2.xlsx", tpl_bytes, "⬇️ Descargar plantilla v2.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Hojas: ensayos (Ca, Cs), tratamientos, emergencia.")

with st.expander("1b) Generar datos sintéticos (multi-ensayo)", expanded=True):
    colg1, colg2, colg3 = st.columns(3)
    n_trials = colg1.number_input("Cantidad de ensayos", 2, 50, 8, 1)
    seed = colg2.number_input("Seed", 0, 99999999, 20250924, 1)
    cap_choice = colg3.multiselect("Tope MAX_PLANTS_CAP (pl·m²)", [250, 300], default=[250, 300])
    colh1, colh2 = st.columns(2)
    loss_min = colh1.number_input("Pérdida objetivo mínima (%)", 0.0, 80.0, 6.0, 0.5)
    loss_max = colh2.number_input("Pérdida objetivo máxima (%)", 0.0, 80.0, 28.0, 0.5)
    build_btn = st.button("🧪 Generar Excel sintético")

    if build_btn:
        if loss_min >= loss_max:
            st.error("El mínimo de pérdida debe ser menor que el máximo.")
        elif not cap_choice:
            st.error("Seleccioná al menos un valor de MAX_PLANTS_CAP.")
        else:
            xls_bytes, df_preview = build_synthetic_excel_v2(
                n_trials=int(n_trials),
                cap_choices=tuple(sorted(set(cap_choice))),
                loss_range=(float(loss_min), float(loss_max)),
                seed=int(seed)
            )
            st.success(f"Generado Excel sintético con {int(n_trials)} ensayos.")
            offer_download_bytes("calibracion_multiensayo_v2.xlsx", xls_bytes,
                                 "⬇️ Descargar calibracion_multiensayo_v2.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.dataframe(df_preview.head(10), use_container_width=True)

# 2) Subir Excel
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

# 3) Configuración del modelo
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

# 4) Parámetros de canopia
with st.expander("4) Parámetros de canopia", expanded=True):
    st.caption("Estos definen la dinámica LAI(t). LAIhc **se calibra** en el paso siguiente.")
    colx, coly, colz, colk = st.columns(4)
    t_lag = colx.number_input("t_lag (días a emergencia)", value=7, step=1)
    t_close = coly.number_input("t_close (días a cierre)", value=45, step=1)
    lai_max = colz.number_input("LAI máximo", value=3.5, step=0.1)
    k_beer = colk.number_input("k (Beer–Lambert)", value=0.6, step=0.05)
    canopy_mode = st.selectbox("Modo canopia", ["LAI","Cobertura→LAI (Beer)"], index=0)

# 5) Calibración
with st.expander("5) Calibración", expanded=True):
    st.markdown("Se calibran **w_S1..w_S4** y **LAIhc**. Opcionalmente, **α** y **Lmax** de la curva de pérdida.")
    n_iter = st.number_input("Iteraciones (búsqueda aleatoria)", value=3000, min_value=50, step=100)
    seed_cal = st.number_input("Seed (calibración)", value=123, step=1)

    colh1, colh2 = st.columns(2)
    LAIhc_lo = colh1.number_input("LAIhc min", value=2.0, step=0.1)
    LAIhc_hi = colh2.number_input("LAIhc max", value=6.0, step=0.1)

    st.divider()
    st.markdown("**Curva de pérdida** (por defecto α=0.375, Lmax=76.639)")
    calibrate_curve = st.checkbox("Calibrar α y Lmax", value=False)
    colp1, colp2 = st.columns(2)
    alpha_lo = colp1.number_input("α min", value=0.10, step=0.01, format="%.3f")
    alpha_hi = colp1.number_input("α max", value=0.80, step=0.01, format="%.3f")
    Lmax_lo  = colp2.number_input("Lmax min", value=40.0, step=1.0)
    Lmax_hi  = colp2.number_input("Lmax max", value=120.0, step=1.0)

    cut_at_pc_fin = st.checkbox("Cortar acumulación en pc_fin (techo)", value=True,
                                help="Siempre comienza en siembra. Si lo desactivás, suma hasta el final del dataset.")

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
            "w_S1": (w1_lo, w1_hi),
            "w_S2": (w2_lo, w2_hi),
            "w_S3": (w3_lo, w3_hi),
            "w_S4": (w4_lo, w4_hi),
            "LAIhc": (LAIhc_lo, LAIhc_hi),
        }
        if calibrate_curve:
            bounds.update({
                "alpha": (alpha_lo, alpha_hi),
                "Lmax":  (Lmax_lo,  Lmax_hi),
            })

        S1 = (int(s1_ini), int(s1_fin))
        S2 = (int(s2_ini), int(s2_fin))
        S3 = (int(s3_ini), int(s3_fin))

        def objective(params):
            LAIhc_p = params["LAIhc"]
            wS = [params["w_S1"], params["w_S2"], params["w_S3"], params["w_S4"]]
            alpha_p = float(params.get("alpha", 0.375))
            Lmax_p  = float(params.get("Lmax",  76.639))
            preds, obs = [], []
            for _, row in df_e.iterrows():
                ens_id = str(row["ensayo_id"])
                sub_m = df_m[df_m["ensayo_id"]==ens_id]
                sub_t = df_t[df_t["ensayo_id"]==ens_id]
                res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, wS,
                                              t_lag=int(t_lag), t_close=int(t_close),
                                              lai_max=float(lai_max), k_beer=float(k_beer),
                                              LAIhc=float(LAIhc_p),
                                              canopy_mode="LAI" if canopy_mode=="LAI" else "COVER",
                                              alpha=alpha_p, Lmax=Lmax_p,
                                              cut_at_pc_fin=bool(cut_at_pc_fin))
                if res["ok"]:
                    preds.append(res["Loss_pred"])
                    obs.append(float(row["loss_obs_pct"]))
            if not preds:
                return np.inf
            rmse,_ = loss_metrics(obs, preds)
            return rmse

        with st.spinner("Calibrando..."):
            best, best_rmse = random_search(objective, n_iter=n_iter, bounds=bounds, seed=int(seed_cal))

        st.success("¡Listo!")
        st.write("**Mejores parámetros:**")
        st.json(best)
        st.write(f"**RMSE (global):** {best_rmse:.3f}")

        # Reconstrucción por ensayo con los mejores
        LAIhc_p = best["LAIhc"]
        wS = [best["w_S1"], best["w_S2"], best["w_S3"], best["w_S4"]]
        alpha_p = float(best.get("alpha", 0.375))
        Lmax_p  = float(best.get("Lmax",  76.639))

        rows = []
        for _, row in df_e.iterrows():
            ens_id = str(row["ensayo_id"])
            sub_m = df_m[df_m["ensayo_id"]==ens_id]
            sub_t = df_t[df_t["ensayo_id"]==ens_id]
            res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, wS,
                                          t_lag=int(t_lag), t_close=int(t_close),
                                          lai_max=float(lai_max), k_beer=float(k_beer),
                                          LAIhc=float(LAIhc_p),
                                          canopy_mode="LAI" if canopy_mode=="LAI" else "COVER",
                                          alpha=alpha_p, Lmax=Lmax_p,
                                          cut_at_pc_fin=bool(cut_at_pc_fin))
            if not res["ok"]:
                rows.append({"ensayo_id": ens_id, "ok": False, "msg": res.get("msg","")})
                continue
            rows.append({
                "ensayo_id": ens_id, "ok": True,
                "A2_ctrl": res["A2_ctrl"], "g_eq": res["g_eq"], "A2_eff": res["A2_eff"],
                "Loss_obs_pct": float(row["loss_obs_pct"]), "Loss_pred_pct": res["Loss_pred"],
                "LAI_mean": res["LAI_mean"], "alpha": alpha_p, "Lmax": Lmax_p
            })
        df_res = pd.DataFrame(rows)
        df_val = df_res[df_res["ok"]==True].copy()
        if not df_val.empty:
            rmse, mae = loss_metrics(df_val["Loss_obs_pct"], df_val["Loss_pred_pct"])
            st.write(f"**RMSE validación:** {rmse:.3f}  |  **MAE:** {mae:.3f}")
            st.dataframe(df_val[["ensayo_id","A2_ctrl","g_eq","A2_eff","Loss_obs_pct","Loss_pred_pct","LAI_mean","alpha","Lmax"]].round(3),
                         use_container_width=True)

            # Gráfico 1: Observado vs Predicho
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

            # Gráfico 2: Curva de pérdida vs A2_eff
            max_cap_guess = float(df_e.get("MAX_PLANTS_CAP", pd.Series([250])).max())
            x_curve = np.linspace(0.0, max_cap_guess, 400)
            y_curve = loss_pct_rect_hyperbola(x_curve, alpha=alpha_p, Lmax=Lmax_p)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name=f"Pérdida % (α={alpha_p:.3f}, Lmax={Lmax_p:.1f})"))
            fig2.add_trace(go.Scatter(x=df_val["A2_eff"], y=df_val["Loss_pred_pct"], mode="markers+text",
                                      text=df_val["ensayo_id"], textposition="top center", name="Ensayos"))
            fig2.update_layout(title="Pérdida (%) vs A2_eff (acumulación desde siembra)",
                               xaxis_title="A2_eff (pl·m²)", yaxis_title="Pérdida (%)", height=500)
            st.plotly_chart(fig2, use_container_width=True)

            csv = df_val.to_csv(index=False).encode("utf-8")
            offer_download_bytes("calibracion_resultados_v2_4_1.csv", csv, "💾 Descargar resultados.csv", "text/csv")
        else:
            st.warning("No hay ensayos válidos para reportar.")

# 6) Exportar este script (SOLO DESCARGA; no escribe a disco)
with st.expander("Notas", expanded=False):
    st.markdown("""
- **Política t=0**: Toda la acumulación inicia en **fecha de siembra**; emergencias previas no suman (lote limpio).
- **Techo temporal**: Opción para **cortar en `pc_fin`** (activada por defecto). Si la desactivás, suma hasta el final del dataset.
- **Curva de pérdida**: `Loss%(x) = (α·x) / (1 + (α·x / Lmax))` con toggle para calibrar `α` y `Lmax`.
- **Supresión dinámica**: `Ciec(t) = (LAI(t)/LAIhc) · (Ca/Cs)`; `g_eq` ponderado.
- **Generador sintético** y descargas en memoria incluidos.
""")

# Botón de descarga del script actual (sin escribir a /mnt/data)
import inspect, sys
try:
    src = inspect.getsource(sys.modules["__main__"])
except Exception:
    src = inspect.getsource(sys.modules[__name__])

st.download_button(
    "⬇️ Descargar calibra_v2_4_1_clean_from_sow.py",
    data=src.encode("utf-8"),
    file_name="calibra_v2_4_1_clean_from_sow.py",
    mime="text/x-python",
)
