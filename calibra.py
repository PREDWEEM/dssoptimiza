# -*- coding: utf-8 -*-
# ============================================================
# PREDWEEM · Módulo de Calibración (Streamlit)
# - Plantilla Excel (openpyxl) con 3 hojas: ensayos, tratamientos, emergencia
# - Objetivo: minimizar RMSE de % pérdida de rinde (Obs vs Pred)
# - Optimizador: Búsqueda aleatoria reproducible (sin SciPy)
# - Estados (S1..S4) por edad desde siembra (configurables en UI)
# - Tipos de herbicida: presiembra, preemergente, post_selectivo, graminicida
# - Eficacias combinadas día/estado con residualidad
# ============================================================

import io, json, math, datetime as dt
from datetime import timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# Utilidades de fechas
# ------------------------------------------------------------
def to_date(x):
    if pd.isna(x): return None
    if isinstance(x, (dt.date, dt.datetime)): return x.date() if isinstance(x, dt.datetime) else x
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def daterange(d0, d1):
    d0, d1 = to_date(d0), to_date(d1)
    if d0 is None or d1 is None or d1 < d0:
        return []
    cur = d0
    out = []
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out

# ------------------------------------------------------------
# Plantilla Excel
# ------------------------------------------------------------
def build_excel_template():
    """
    Devuelve bytes de un .xlsx con 3 hojas:
    - ensayos: 1 fila por ensayo
    - tratamientos: N filas por ensayo (pueden ser 0..n)
    - emergencia: serie diaria con emer_rel (0..1) por ensayo
    """
    ensayos = pd.DataFrame([{
        "ensayo_id": "E001",
        "sitio": "BahiaBlanca",
        "campaña": "2025",
        "cultivo": "Trigo",
        "fecha_siembra": "2025-09-20",
        "pc_ini": "2025-09-25",
        "pc_fin": "2025-11-15",
        "Ciec_const": 0.20,  # supresión por canopia promedio (0..1)
        "MAX_PLANTS_CAP": 250,  # tope plantas·m2
        "rend_testigo_kg_ha": 4000,  # rinde sin malezas
        "rend_observado_kg_ha": 3600  # rinde con manejo observado
    }])

    tratamientos = pd.DataFrame([
        # tipo in ["presiembra", "preemergente", "post_selectivo", "graminicida"]
        {"ensayo_id": "E001", "tipo": "preemergente", "fecha_aplicacion": "2025-09-20",
         "eficacia_pct": 90, "residual_dias": 10,
         "actua_s1": 1, "actua_s2": 1, "actua_s3": 0, "actua_s4": 0},
        {"ensayo_id": "E001", "tipo": "graminicida", "fecha_aplicacion": "2025-10-05",
         "eficacia_pct": 85, "residual_dias": 10,
         "actua_s1": 1, "actua_s2": 1, "actua_s3": 1, "actua_s4": 0},
    ])

    # Emergencia relativa que suma ~1 en la campaña (ejemplo triangular)
    fechas = pd.date_range("2025-09-01", "2025-11-30", freq="D")
    mid = fechas[int(len(fechas)*0.35)]
    emer = []
    for f in fechas:
        # forma triangular simple centrada cerca de siembra
        w = 1 - abs((f - mid).days) / max(1, len(fechas)*0.35)
        emer.append(max(0.0, w))
    emer = np.array(emer)
    emer_rel = (emer / emer.sum()).round(6)
    emergencia = pd.DataFrame({
        "ensayo_id": "E001",
        "fecha": fechas.date,
        "emer_rel": emer_rel
    })

    buf = io.BytesIO()
    # Usar openpyxl para evitar dependencia a xlsxwriter
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ensayos.to_excel(writer, index=False, sheet_name="ensayos")
        tratamientos.to_excel(writer, index=False, sheet_name="tratamientos")
        emergencia.to_excel(writer, index=False, sheet_name="emergencia")
    buf.seek(0)
    return buf.getvalue()

# ------------------------------------------------------------
# Núcleo del modelo (predicción de pérdida)
# ------------------------------------------------------------
def assign_state_by_age(age_days, s1=(1,6), s2=(7,27), s3=(28,59)):
    """
    Devuelve 0..3 para S1..S4 según edad en días desde siembra.
    Bordes inclusivos.
    """
    if age_days is None or age_days < s1[0]:
        return None  # antes de nacer
    if s1[0] <= age_days <= s1[1]: return 0  # S1
    if s2[0] <= age_days <= s2[1]: return 1  # S2
    if s3[0] <= age_days <= s3[1]: return 2  # S3
    return 3  # S4

def combine_efficacies(effs):
    """
    Combinación multiplicativa independiente: 1 - Π(1-ef_i)
    effs en [0..1].
    """
    if not effs: return 0.0
    prod = 1.0
    for e in effs:
        prod *= (1 - max(0.0, min(1.0, e)))
    return 1 - prod

def build_daily_control_mask(dates, states, ttos_rows):
    """
    Para cada día, devuelve eficacia combinada por estado [S1..S4] en [0..1].
    - dates: lista de fechas
    - states: lista de índices de estado por día (0..3 o None)
    - ttos_rows: lista de dicts con: tipo, fecha_aplicacion, eficacia_pct, residual_dias, actua_s*
    Reglas simples:
      presiembra: aplica SOLO antes de siembra (el usuario debe cargar esa fecha)
      preemergente: desde día de aplicación hasta app+residual
      post_selectivo y graminicida: desde app hasta app+residual
    """
    n = len(dates)
    eff = np.zeros((n, 4), dtype=float)

    # Preprocesar ventanas
    windows = []
    for r in ttos_rows:
        tipo = str(r.get("tipo","")).strip().lower()
        fapp = to_date(r.get("fecha_aplicacion"))
        ef = float(r.get("eficacia_pct", 0))/100.0
        res = int(r.get("residual_dias", 0))
        mask_states = [int(r.get("actua_s1",0))>0,
                       int(r.get("actua_s2",0))>0,
                       int(r.get("actua_s3",0))>0,
                       int(r.get("actua_s4",0))>0]
        if fapp is None: 
            continue
        if tipo == "presiembra":
            # ventana arbitraria: fapp-res <= fapp (solo antes)
            start = fapp - timedelta(days=max(0,res))
            end = fapp
        elif tipo in ("preemergente","post_selectivo","graminicida"):
            start = fapp
            end = fapp + timedelta(days=max(0,res))
        else:
            # desconocido: ignorar
            continue
        windows.append((tipo, start, end, ef, mask_states))

    # Para cada día, acumular eficacias por estado
    for i, d in enumerate(dates):
        effs_state = [ [] for _ in range(4) ]
        for tipo, start, end, ef, mask_states in windows:
            if start <= d <= end:
                for sidx in range(4):
                    if mask_states[sidx]:
                        effs_state[sidx].append(ef)
        # combinar
        for sidx in range(4):
            eff[i, sidx] = combine_efficacies(effs_state[sidx])
    return eff  # (n_días, 4)

def predict_loss_for_ensayo(row_e, df_emerg, df_ttos,
                            s1, s2, s3,
                            k_loss, w_states, escala_A2=100.0):
    """
    row_e: fila de ensayos (siembra, PC, Ciec_const, MAX_PLANTS_CAP, rendidos)
    df_emerg: sub-dataframe emergencia para ensayo_id (fecha, emer_rel)
    df_ttos: sub-dataframe tratamientos para ensayo_id

    Retorna: dict con series y Loss_pred (%), A2_ctrl, etc.
    """
    siembra = to_date(row_e["fecha_siembra"])
    pc_ini = to_date(row_e["pc_ini"])
    pc_fin = to_date(row_e["pc_fin"])
    Ciec = float(row_e.get("Ciec_const", 0.0))
    max_cap = float(row_e.get("MAX_PLANTS_CAP", 250))

    if siembra is None or pc_ini is None or pc_fin is None:
        return {"ok": False, "msg": "Fechas inválidas (siembra/PC)"}

    # Alinear fechas de emergencia al rango [min, max]
    df_emerg = df_emerg.copy()
    df_emerg["fecha"] = df_emerg["fecha"].apply(to_date)
    df_emerg = df_emerg.dropna(subset=["fecha","emer_rel"]).sort_values("fecha")
    if df_emerg.empty:
        return {"ok": False, "msg": "Emergencia vacía"}

    dates = list(df_emerg["fecha"].values)

    # Edad desde siembra
    age = [ (d - siembra).days for d in dates ]
    # Estado por día
    states = [ assign_state_by_age(a, s1, s2, s3) for a in age ]

    # Plantas día crudas y con supresión
    emer_rel = df_emerg["emer_rel"].values.astype(float)
    pl_day = emer_rel * max_cap
    pl_sup = pl_day * (1.0 - Ciec)

    # Control diario por estado
    ttos_rows = df_ttos.to_dict(orient="records") if df_ttos is not None else []
    eff_by_state = build_daily_control_mask(dates, states, ttos_rows)  # (n,4)

    # Aplicar control según estado del individuo cada día
    pl_ctrl = pl_sup.copy()
    for i in range(len(dates)):
        sidx = states[i]
        if sidx is None: 
            continue
        e = eff_by_state[i, sidx]  # eficacia efectiva aplicable a ese estado
        pl_ctrl[i] = pl_ctrl[i] * (1.0 - e)

    # Filtro periodo crítico (PC)
    pc_mask = np.array([1.0 if (pc_ini <= d <= pc_fin) else 0.0 for d in dates])

    # Ponderación por estado
    wS = np.array(w_states, dtype=float)  # longitud 4
    wS = np.maximum(0, wS)
    if wS.sum() == 0:
        wS = np.array([1,1,1,1], dtype=float)

    w_day = np.zeros(len(dates), dtype=float)
    for i in range(len(dates)):
        sidx = states[i]
        if sidx is None: 
            w_day[i] = 0.0
        else:
            w_day[i] = wS[sidx]

    # A2_ctrl: suma en PC ponderada por estado
    A2_ctrl = float(np.sum(pl_ctrl * w_day * pc_mask))

    # Pérdida predicha (%), con forma exponencial saturante
    k = max(1e-6, float(k_loss))
    Loss_pred = 100.0 * (1.0 - math.exp(-k * (A2_ctrl / float(escala_A2))))

    return {
        "ok": True,
        "dates": dates,
        "pl_day": pl_day,
        "pl_sup": pl_sup,
        "pl_ctrl": pl_ctrl,
        "pc_mask": pc_mask,
        "states": states,
        "A2_ctrl": A2_ctrl,
        "Loss_pred": Loss_pred
    }

# ------------------------------------------------------------
# Calibración
# ------------------------------------------------------------
def loss_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2))) if len(y_true) else np.nan
    mae  = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
    return rmse, mae

def random_search(objective_fn, n_iter, bounds, seed=123):
    """
    bounds: dict nombre -> (low, high)
    Devuelve (best_params, best_score)
    """
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(int(n_iter)):
        params = {}
        for k,(lo,hi) in bounds.items():
            params[k] = rng.uniform(lo, hi)
        score = objective_fn(params)
        if score < best_score:
            best, best_score = params, score
    return best, best_score

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="PREDWEEM · Calibración", layout="wide")
st.title("PREDWEEM — Módulo de Calibración (% pérdida de rinde)")

with st.expander("1) Descargar plantilla Excel", expanded=True):
    st.write("Completá las 3 hojas: **ensayos**, **tratamientos**, **emergencia**.")
    tpl = build_excel_template()
    st.download_button("⬇️ Descargar plantilla.xlsx", data=tpl, file_name="plantilla_calibracion.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Usa fechas en formato ISO (YYYY-MM-DD). `emer_rel` debería sumar ≈ 1 por ensayo (el script no lo exige, pero es buena práctica).")

with st.expander("2) Subir Excel con datos", expanded=True):
    up = st.file_uploader("Cargar Excel completado", type=["xlsx"])
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

    st.markdown("**Ponderación por estado (se calibran, pero podés acotar rangos):**")
    c1,c2,c3,c4 = st.columns(4)
    w1_lo = c1.number_input("w_S1 min", value=0.5, step=0.1)
    w1_hi = c1.number_input("w_S1 max", value=2.0, step=0.1)
    w2_lo = c2.number_input("w_S2 min", value=0.5, step=0.1)
    w2_hi = c2.number_input("w_S2 max", value=2.0, step=0.1)
    w3_lo = c3.number_input("w_S3 min", value=0.5, step=0.1)
    w3_hi = c3.number_input("w_S3 max", value=2.0, step=0.1)
    w4_lo = c4.number_input("w_S4 min", value=0.1, step=0.1)
    w4_hi = c4.number_input("w_S4 max", value=1.5, step=0.1)

    st.markdown("**Curva de pérdida:**  Loss% = 100 × (1 − exp(−k_loss × A2_ctrl / escala_A2))")
    col1, col2 = st.columns(2)
    k_lo = col1.number_input("k_loss min", value=0.01, step=0.01, format="%.3f")
    k_hi = col1.number_input("k_loss max", value=0.50, step=0.01, format="%.3f")
    escala_A2 = col2.number_input("escala_A2 (normaliza A2)", value=100.0, step=10.0)

with st.expander("4) Calibración", expanded=True):
    n_iter = st.number_input("Iteraciones de búsqueda aleatoria", value=2000, min_value=50, step=100)
    seed = st.number_input("Seed", value=123, step=1)
    run = st.button("🚀 Ejecutar calibración")

    if run and (df_e is None or df_m is None):
        st.warning("Cargá el Excel antes de calibrar.")

    if run and df_e is not None and df_m is not None:
        # Pre-procesar dataframes
        df_e = df_e.copy()
        df_t = df_t.copy() if df_t is not None else pd.DataFrame(columns=["ensayo_id"])
        df_m = df_m.copy()

        # Calcular pérdida observada (%) si no está
        if "loss_obs_pct" not in df_e.columns:
            # si hay rend_testigo y rend_observado
            if {"rend_testigo_kg_ha","rend_observado_kg_ha"}.issubset(df_e.columns):
                df_e["loss_obs_pct"] = 100.0 * (1.0 - (df_e["rend_observado_kg_ha"] / df_e["rend_testigo_kg_ha"]))
            else:
                st.error("Falta 'loss_obs_pct' o (rend_testigo_kg_ha, rend_observado_kg_ha) en ensayos.")
                st.stop()

        # Bounds para random search
        bounds = {
            "k_loss": (k_lo, k_hi),
            "w_S1": (w1_lo, w1_hi),
            "w_S2": (w2_lo, w2_hi),
            "w_S3": (w3_lo, w3_hi),
            "w_S4": (w4_lo, w4_hi),
        }

        S1 = (int(s1_ini), int(s1_fin))
        S2 = (int(s2_ini), int(s2_fin))
        S3 = (int(s3_ini), int(s3_fin))

        # Objetivo: RMSE global
        def objective(params):
            k = params["k_loss"]
            wS = [params["w_S1"], params["w_S2"], params["w_S3"], params["w_S4"]]
            preds, obs = [], []
            for _, row in df_e.iterrows():
                ens_id = str(row["ensayo_id"])
                sub_m = df_m[df_m["ensayo_id"]==ens_id]
                sub_t = df_t[df_t["ensayo_id"]==ens_id] if df_t is not None else pd.DataFrame(columns=["ensayo_id"])
                res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, k, wS, escala_A2=escala_A2)
                if not res["ok"]: 
                    continue
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

        # Reconstruir resultados por ensayo con el mejor set
        k = best["k_loss"]
        wS = [best["w_S1"], best["w_S2"], best["w_S3"], best["w_S4"]]
        rows = []
        for _, row in df_e.iterrows():
            ens_id = str(row["ensayo_id"])
            sub_m = df_m[df_m["ensayo_id"]==ens_id]
            sub_t = df_t[df_t["ensayo_id"]==ens_id] if df_t is not None else pd.DataFrame(columns=["ensayo_id"])
            res = predict_loss_for_ensayo(row, sub_m, sub_t, S1, S2, S3, k, wS, escala_A2=escala_A2)
            if not res["ok"]:
                rows.append({"ensayo_id": ens_id, "ok": False, "msg": res.get("msg","")})
                continue
            rows.append({
                "ensayo_id": ens_id,
                "ok": True,
                "A2_ctrl": res["A2_ctrl"],
                "Loss_pred_pct": res["Loss_pred"],
                "Loss_obs_pct": float(row["loss_obs_pct"])
            })
        df_res = pd.DataFrame(rows)
        df_val = df_res[df_res["ok"]==True].copy()
        if not df_val.empty:
            rmse, mae = loss_metrics(df_val["Loss_obs_pct"], df_val["Loss_pred_pct"])
            st.write(f"**RMSE validación:** {rmse:.3f}  |  **MAE:** {mae:.3f}")

            # Tabla
            st.dataframe(df_val[["ensayo_id","A2_ctrl","Loss_obs_pct","Loss_pred_pct"]].round(3), use_container_width=True)

            # Scatter Obs vs Pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_val["Loss_obs_pct"], y=df_val["Loss_pred_pct"],
                mode="markers+text",
                text=df_val["ensayo_id"], textposition="top center",
                name="Ensayos"
            ))
            maxi = float(max(df_val["Loss_obs_pct"].max(), df_val["Loss_pred_pct"].max(), 1.0))
            fig.add_trace(go.Scatter(x=[0,maxi], y=[0,maxi], mode="lines", name="1:1", line=dict(dash="dash")))
            fig.update_layout(
                title="Pérdida de rinde: Observado vs Predicho",
                xaxis_title="Observado (%)",
                yaxis_title="Predicho (%)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Descargar CSV resultados
            csv = df_val.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Descargar resultados.csv", data=csv, file_name="calibracion_resultados.csv", mime="text/csv")
        else:
            st.warning("No hay ensayos válidos para reportar.")

with st.expander("Notas y supuestos del modelo", expanded=False):
    st.markdown("""
- `emer_rel` se interpreta como **proporción diaria** de nacimiento (≈ distrib. que suma 1).  
- `MAX_PLANTS_CAP` reescala `emer_rel` a **plantas·m²/día** en ausencia de supresión y control.  
- La **supresión** se aplica como factor constante `1 − Ciec_const`.  
- El **control químico** combina eficacias independientes por día/estado:  
  `ef_total = 1 − Π(1 − ef_i)` (con ventanas por `fecha_aplicacion` y `residual_dias`).  
- Estados por edad (configurables): por defecto S1=1–6d, S2=7–27d, S3=28–59d, S4=60+d.  
- `A2_ctrl` es la suma, dentro del **periodo crítico**, de `pl_ctrl × w_estado`.  
- Curva de pérdida: `Loss% = 100 × (1 − exp(−k_loss × A2_ctrl / escala_A2))`.  
- Parámetros calibrados: `k_loss`, `w_S1..w_S4` (acotados por UI).  
- **Extensiones**: Ciec dinámico diario, cohortes explícitas, grillas y recocido, validación cruzada, etc.
""")


