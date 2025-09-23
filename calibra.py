# ====================== Sección 1 — Metadatos y constantes ======================
APP_TITLE = "PREDWEEM · Calibración con datos independientes"
APP_VERSION = "v1.0"

# Convenciones de columnas esperadas
EMERGENCIA_COLS = {
    "fecha": "fecha",          # Fechas (día) de la serie
    "emerrel": "EMERREL"       # EMERREL diaria (0–1); si viene en %, marcalo en la UI
}

OBS_COLS = {
    "ensayo_id": "ensayo_id",                 # Identificador del ensayo/lote
    "fecha_siembra": "fecha_siembra",         # Fecha de siembra
    "tipo_herbicida": "tipo_herbicida",       # presiembra | preemergente | postemergente
    "momento_aplicacion": "momento_aplicacion",  # fecha (o días relativos) de la aplicación
    "dosis": "dosis",                         # numérico (opcional)
    "eficiencia_pct": "eficiencia_pct",       # 0–100 (si no se conoce, dejá vacío)
    "perdida_rinde_pct": "perdida_rinde_pct", # objetivo de calibración
    "plantas_avena": "plantas_avena"          # número final de plantas de avena fatua (pl·m²)
}

# Límites y defaults útiles en el resto de la app
SOW_MONTH_MIN = 5   # mayo
SOW_MONTH_MAX = 8   # agosto
TOPE_A2_OPTIONS = [250, 125, 62]

# ================== Sección 2 — Imports y configuración Streamlit =================
import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption(f"Versión {APP_VERSION}")

# ----------------------- Helpers de parseo y limpieza -----------------------
def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample):
        dec_guess = ","
    return sep_guess, dec_guess

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%","",regex=False)
    if decimal == ",":
        t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else:
        t = t.str.replace(",","",regex=False)
    return pd.to_numeric(t, errors="coerce")

def parse_csv_or_excel(file, kind: str, as_percent=False, dayfirst=True):
    """
    kind: 'emergencia' | 'observaciones'
    Retorna (DataFrame, info_dict)
    """
    name = getattr(file, "name", f"{kind}.bin").lower()
    raw = file.read()
    # ¿Excel?
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(raw))
        # hoja por defecto
        sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
        info = {"format": "excel", "sheet": sheet}
        return df, info
    # Caso CSV
    head = raw[:8000].decode("utf-8", errors="ignore")
    sep_guess, dec_guess = sniff_sep_dec(head)
    df = pd.read_csv(io.BytesIO(raw), sep=sep_guess, decimal=dec_guess, engine="python")
    info = {"format": "csv", "sep": sep_guess, "dec": dec_guess}
    return df, info

def ensure_columns(df: pd.DataFrame, required: dict, label: str):
    missing = [v for v in required.values() if v not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Faltan columnas: {missing}. Presentes: {list(df.columns)}")
# ==================== Sección 3 — Carga de datos + plantillas ====================
with st.sidebar:
    st.header("Entrada de datos")
    st.markdown("Podés cargar **Excel (.xlsx)** o **CSV**.")

    st.subheader("Serie de EMERGENCIA (EMERREL diaria)")
    up_emerg = st.file_uploader("Emergencia (EMERREL)", type=["xlsx","xls","csv"], key="up_emerg")
    emerg_as_percent = st.checkbox("EMERREL viene en %", value=True, help="Si está en 0–100%, activar. Si ya está en 0–1, desactivar.")
    emerg_dayfirst = st.checkbox("Fechas dd/mm/yyyy", value=True, key="emerg_dayfirst")

    st.subheader("Observaciones independientes (ensayos)")
    up_obs = st.file_uploader("Observaciones (ensayos)", type=["xlsx","xls","csv"], key="up_obs")
    obs_dayfirst = st.checkbox("Fechas dd/mm/yyyy (observaciones)", value=True, key="obs_dayfirst")

    st.divider()
    st.caption("¿Necesitás plantillas? Podés descargarlas abajo.")

    def build_emerg_template() -> bytes:
        df = pd.DataFrame({
            EMERGENCIA_COLS["fecha"]: pd.date_range("2021-06-01", periods=10, freq="D"),
            EMERGENCIA_COLS["emerrel"]: [0.0, 0.0, 0.01, 0.02, 0.05, 0.07, 0.04, 0.02, 0.01, 0.0],
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="emergencia", index=False)
        return buf.getvalue()

    def build_obs_template() -> bytes:
        df = pd.DataFrame({
            OBS_COLS["ensayo_id"]: ["E1","E2"],
            OBS_COLS["fecha_siembra"]: [pd.Timestamp("2021-06-10"), pd.Timestamp("2021-06-20")],
            OBS_COLS["tipo_herbicida"]: ["presiembra","preemergente"],
            OBS_COLS["momento_aplicacion"]: [pd.Timestamp("2021-05-30"), pd.Timestamp("2021-06-21")],
            OBS_COLS["dosis"]: [1.2, 0.8],
            OBS_COLS["eficiencia_pct"]: [70, 65],
            OBS_COLS["perdida_rinde_pct"]: [8.5, 5.2],
            OBS_COLS["plantas_avena"]: [45, 30],
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="observaciones", index=False)
        return buf.getvalue()

    st.download_button(
        "⬇️ Descargar plantilla EMERGENCIA (xlsx)",
        data=build_emerg_template(),
        file_name="plantilla_emergencia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.download_button(
        "⬇️ Descargar plantilla OBSERVACIONES (xlsx)",
        data=build_obs_template(),
        file_name="plantilla_observaciones.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ------------------------- Parseo de los archivos -------------------------
df_emergencia = None
df_observaciones = None

# Emergencia
if up_emerg is None:
    st.info("⏳ Cargá la serie de **EMERREL** (Excel o CSV) para continuar.")
else:
    try:
        dfE_raw, infoE = parse_csv_or_excel(up_emerg, kind="emergencia", as_percent=emerg_as_percent, dayfirst=emerg_dayfirst)
        ensure_columns(dfE_raw, EMERGENCIA_COLS, "EMERGENCIA")
        # Normalización
        dfE = dfE_raw.rename(columns={
            EMERGENCIA_COLS["fecha"]: "fecha",
            EMERGENCIA_COLS["emerrel"]: "EMERREL"
        }).copy()

        dfE["fecha"] = pd.to_datetime(dfE["fecha"], dayfirst=emerg_dayfirst, errors="coerce")
        # limpiar EMERREL
        # Intento detectar decimal de la columna original (solo si no vino desde Excel)
        if infoE.get("format") == "csv":
            head_vals = " ".join(dfE_raw[EMERGENCIA_COLS["emerrel"]].astype(str).head(100).tolist())
            dec_guess = "," if (head_vals.count(",")>head_vals.count(".") and re.search(r",\d", head_vals)) else "."
        else:
            dec_guess = "."
        dfE["EMERREL"] = clean_numeric_series(dfE["EMERREL"], decimal=dec_guess)
        if emerg_as_percent:
            dfE["EMERREL"] = dfE["EMERREL"] / 100.0

        dfE = dfE.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
        # Quitar duplicados (sumar)
        if dfE["fecha"].duplicated().any():
            dfE = dfE.groupby("fecha").agg({"EMERREL":"sum"}).reset_index()

        df_emergencia = dfE
        st.success(f"✅ EMERGENCIA cargada ({infoE.get('format')}). Filas: {len(dfE)}")
        st.dataframe(dfE.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error al leer EMERGENCIA: {e}")

# Observaciones
if up_obs is None:
    st.info("⏳ Cargá **Observaciones** (Excel o CSV) para continuar.")
else:
    try:
        dfO_raw, infoO = parse_csv_or_excel(up_obs, kind="observaciones", dayfirst=obs_dayfirst)
        ensure_columns(dfO_raw, OBS_COLS, "OBSERVACIONES")
        dfO = dfO_raw.rename(columns={
            OBS_COLS["ensayo_id"]: "ensayo_id",
            OBS_COLS["fecha_siembra"]: "fecha_siembra",
            OBS_COLS["tipo_herbicida"]: "tipo_herbicida",
            OBS_COLS["momento_aplicacion"]: "momento_aplicacion",
            OBS_COLS["dosis"]: "dosis",
            OBS_COLS["eficiencia_pct"]: "eficiencia_pct",
            OBS_COLS["perdida_rinde_pct"]: "perdida_rinde_pct",
            OBS_COLS["plantas_avena"]: "plantas_avena",
        }).copy()

        dfO["fecha_siembra"] = pd.to_datetime(dfO["fecha_siembra"], dayfirst=obs_dayfirst, errors="coerce")
        dfO["momento_aplicacion"] = pd.to_datetime(dfO["momento_aplicacion"], dayfirst=obs_dayfirst, errors="coerce")

        # Normalizar numéricos
        for col in ["dosis", "eficiencia_pct", "perdida_rinde_pct", "plantas_avena"]:
            if col in dfO.columns:
                # heurística simple de decimal
                if infoO.get("format") == "csv":
                    head_vals = " ".join(dfO_raw[col].astype(str).head(100).tolist()) if col in dfO_raw.columns else ""
                    dec_guess = "," if (head_vals.count(",")>head_vals.count(".") and re.search(r",\d", head_vals)) else "."
                else:
                    dec_guess = "."
                dfO[col] = clean_numeric_series(dfO[col], decimal=dec_guess)

        # Tipos válidos
        dfO["tipo_herbicida"] = (
            dfO["tipo_herbicida"].astype(str).str.strip().str.lower()
            .replace({
                "pre-siembra":"presiembra",
                "pre siembra":"presiembra",
                "preem":"preemergente",
                "pre-emergente":"preemergente",
                "pre emergente":"preemergente",
                "post":"postemergente",
                "post-emergente":"postemergente",
                "post emergente":"postemergente",
            })
        )

        df_observaciones = dfO
        st.success(f"✅ OBSERVACIONES cargadas ({infoO.get('format')}). Filas: {len(dfO)}")
        st.dataframe(dfO.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error al leer OBSERVACIONES: {e}")

# Chequeo final
if (df_emergencia is not None) and (df_observaciones is not None):
    st.success("📦 Datos listos. Podés continuar con el pipeline (cohortes, AUC, calibración).")
else:
    st.stop()

# =================== Sección 4 — Siembra/Canopia + preprocesado EMERREL ===================

with st.sidebar:
    st.header("Parámetros de canopia y tope A2")
    MAX_PLANTS_CAP = float(st.selectbox(
        "Tope A2 (pl·m²)",
        options=TOPE_A2_OPTIONS,
        index=0,
        help="Tope de densidad efectiva para el reescalado por área."
    ))
    st.caption(f"A2 tope seleccionado: **{int(MAX_PLANTS_CAP)} pl·m²**")

    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Usar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl·m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl·m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario competitivo)", 0.5, 10.0, 3.5, 0.1)

st.caption(
    "Las cohortes S1–S4 se calculan por **cada ensayo** en función de su **fecha de siembra** "
    "y la misma serie de EMERREL cargada."
)

# ------------------------ EMERREL a frecuencia diaria continua ------------------------
# Aseguramos rango diario continuo para aplicar rolling/shift de cohortes
dfE_full = df_emergencia.copy()
dfE_full = dfE_full.sort_values("fecha").reset_index(drop=True)

full_idx = pd.date_range(dfE_full["fecha"].min(), dfE_full["fecha"].max(), freq="D")
dfE_full = (
    dfE_full.set_index("fecha")
    .reindex(full_idx)
    .rename_axis("fecha")
    .reset_index()
)
dfE_full["EMERREL"] = dfE_full["EMERREL"].fillna(0.0)

# Helper: canopia
def compute_canopy(fechas: pd.Series, sow_date: dt.date, mode_canopy: str,
                   t_lag: int, t_close: int, cov_max: float, lai_max: float, k_beer: float):
    days_since_sow = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas], dtype=float)

    def logistic_between(days, start, end, y_max):
        if end <= start:
            end = start + 1
        t_mid = 0.5*(start+end)
        r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days - t_mid)))

    if mode_canopy == "Cobertura dinámica (%)":
        fc_dyn = np.where(days_since_sow < t_lag, 0.0,
                          logistic_between(days_since_sow, t_lag, t_close, cov_max/100.0))
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc_dyn, 1e-9, 1.0)) / max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days_since_sow < t_lag, 0.0,
                       logistic_between(days_since_sow, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc_dyn = 1 - np.exp(-k_beer * LAI)
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
    return fc_dyn, LAI

# Helper: AUC (trapecios) desde siembra
def _to_days(ts: pd.Series) -> np.ndarray:
    f = pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]")
    t_ns = f.astype("int64")
    return ((t_ns - t_ns[0]) / 1e9 / 86400.0).astype(float)

def auc_time(fecha: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, dtype=float)
    if mask is not None:
        f = f[mask]
        y_arr = y_arr[mask]
    if len(f) < 2:
        return 0.0
    tdays = _to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, tdays))

# ============ Sección 5 — Cohortes, Ciec y equivalencias por ENSAYO ============

ts_all = pd.to_datetime(dfE_full["fecha"])
emerrel_all = dfE_full["EMERREL"].astype(float).clip(lower=0.0).to_numpy()

# lógica de cohortes (definida respecto de la siembra)
def build_cohorts_for_sowing(sow_date: dt.date, ts: pd.Series, emer_series: np.ndarray):
    mask_since = (ts.dt.date >= sow_date)

    births = np.where(mask_since.to_numpy(), emer_series, 0.0)
    s = pd.Series(births, index=ts)

    # Definición de cohortes (como en tu app original)
    # S1: suma móvil 6d, desplazada +1d; S2: 21d +7d; S3: 32d +28d; S4: cumsum +60d
    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).reindex(ts).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).reindex(ts).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).reindex(ts).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).reindex(ts).to_numpy(float)

    return mask_since.to_numpy(), S1, S2, S3, S4

def build_env_for_ensayo(sow_date: dt.date) -> dict | None:
    # Canopia y Ciec
    FC, LAI = compute_canopy(ts_all, sow_date, mode_canopy,
                             int(t_lag), int(t_close),
                             float(cov_max), float(lai_max), float(k_beer))
    if use_ciec:
        Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
        Cs_safe = float(Cs) if float(Cs) > 0 else 1e-6
        Ciec = np.clip((LAI / max(1e-6, float(LAIhc))) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec = np.zeros_like(LAI, dtype=float)
    one_minus_Ciec = np.clip(1.0 - Ciec, 0.0, 1.0)

    # Cohortes
    mask_since, S1, S2, S3, S4 = build_cohorts_for_sowing(sow_date, ts_all, emerrel_all)

    # AUC cruda de EMERREL desde siembra (para factor de área → pl·m²)
    auc_cruda = auc_time(ts_all, emerrel_all, mask=mask_since)
    if auc_cruda <= 0:
        return None
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda

    # Aportes diarios por estado (antes de control)
    FC_S = {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0}
    S1_pl = np.where(mask_since, S1 * one_minus_Ciec * FC_S["S1"] * factor_area_to_plants, 0.0)
    S2_pl = np.where(mask_since, S2 * one_minus_Ciec * FC_S["S2"] * factor_area_to_plants, 0.0)
    S3_pl = np.where(mask_since, S3 * one_minus_Ciec * FC_S["S3"] * factor_area_to_plants, 0.0)
    S4_pl = np.where(mask_since, S4 * one_minus_Ciec * FC_S["S4"] * factor_area_to_plants, 0.0)

    # Serie base en pl·m²·día (para aplicar caps por A2 máximo si hace falta más adelante)
    base_pl_daily = np.where(mask_since, emerrel_all * factor_area_to_plants, 0.0)

    return {
        "ts": ts_all,
        "mask_since": mask_since,
        "FC": FC,
        "LAI": LAI,
        "Ciec": Ciec,
        "one_minus_Ciec": one_minus_Ciec,
        "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
        "base_pl_daily": base_pl_daily,
        "auc_cruda": float(auc_cruda),
        "factor_area_to_plants": float(factor_area_to_plants),
        "sow": sow_date,
    }

# --------- Construimos el entorno para CADA ensayo de las observaciones ----------
ensayos = []
env_by_id = {}

for i, row in df_observaciones.iterrows():
    ensayo_id = str(row["ensayo_id"])
    sow = pd.to_datetime(row["fecha_siembra"]).date()
    env = build_env_for_ensayo(sow)
    if env is None:
        st.warning(f"Ensayo {ensayo_id}: AUC(EMERREL) desde siembra = 0 → no se puede escalar a plantas.")
        continue
    env_by_id[ensayo_id] = env
    ensayos.append({
        "ensayo_id": ensayo_id,
        "siembra": str(sow),
        "factor_area_to_plants": env["factor_area_to_plants"],
        "AUC_cruda_desde_siembra": env["auc_cruda"]
    })

st.subheader("Resumen por ensayo (base para calibración)")
if len(ensayos) == 0:
    st.error("No hay ensayos válidos (ver fechas/EMERREL).")
    st.stop()
else:
    st.dataframe(pd.DataFrame(ensayos), use_container_width=True)

st.caption(
    "👉 Con estos entornos por ensayo, en la siguiente sección podrás **modelar los tratamientos** "
    "(presiembra/preemergente/post) y computar la **pérdida de rendimiento modelada** para calibrar "
    "contra el objetivo observado (`perdida_rinde_pct`)."
)

# =================== Sección 6 — Tratamientos y pérdida modelada ===================

st.header("Tratamientos & pérdidas — modelado base")

st.caption(
    "Reglas pedidas: **Presiembra residual** y **Preemergente residual** actúan SOLO sobre **S1 y S2**. "
    "Ventanas válidas: Presiembra residual ≤ siembra−14; Preemergente residual ∈ [siembra, siembra+10]. "
    "El Postemergente selectivo/residual actúa sobre **S1–S4** (sin restricción adicional aquí)."
)

# --------- Lectura de tratamientos desde df_observaciones ----------
# Se espera 1 o más filas por ensayo con columnas:
#  - ensayo_id (str)
#  - fecha_siembra (date)
#  - tipo (str: "presiembra_residual" | "preemergente_residual" | "postemergente")
#  - fecha_aplicacion (date)
#  - residual_dias (int, opcional; default 45)
#  - dosis (float, opcional; se usa en Emax)
#  - perdida_rinde_pct_obs (float, % observado)  [para la calibración]
#  - plantas_final_obs (float, opcional; diagnóstico)

REQ_COLS = ["ensayo_id", "fecha_siembra", "tipo", "fecha_aplicacion", "perdida_rinde_pct_obs"]
missing = [c for c in REQ_COLS if c not in df_observaciones.columns]
if missing:
    st.error(f"Faltan columnas en observaciones: {missing}")
    st.stop()

# Normalizamos tipos/fechas
df_tto = df_observaciones.copy()
df_tto["ensayo_id"] = df_tto["ensayo_id"].astype(str)
df_tto["tipo"] = df_tto["tipo"].astype(str).str.lower().str.strip()
df_tto["fecha_siembra"] = pd.to_datetime(df_tto["fecha_siembra"]).dt.date
df_tto["fecha_aplicacion"] = pd.to_datetime(df_tto["fecha_aplicacion"]).dt.date
if "residual_dias" not in df_tto.columns:
    df_tto["residual_dias"] = np.nan
if "dosis" not in df_tto.columns:
    df_tto["dosis"] = np.nan
if "plantas_final_obs" not in df_tto.columns:
    df_tto["plantas_final_obs"] = np.nan

# --------- Parámetros de Emax (a calibrar en Sección 7; acá sliders para preview) ----------
with st.expander("Parámetros provisionales de eficacia (pre-optimización)", expanded=False):
    Emax_psr = st.slider("Emax Presiembra residual (%)", 0, 100, 70, 1)
    ED50_psr = st.number_input("ED50 Presiembra (dosis)", 0.1, 500.0, 50.0, 0.1)

    Emax_pre = st.slider("Emax Preemergente residual (%)", 0, 100, 75, 1)
    ED50_pre = st.number_input("ED50 Preemergente (dosis)", 0.1, 500.0, 60.0, 0.1)

    Emax_post = st.slider("Emax Postemergente (%)", 0, 100, 65, 1)
    ED50_post = st.number_input("ED50 Post (dosis)", 0.1, 500.0, 40.0, 0.1)

# --------- Funciones de apoyo ----------
POST_RESIDUAL_DEFAULT = 45
PREEM_R_MAX_AFTER_SOW_DAYS = 10  # Preemergente: hasta siembra + 10
PRESIEMBRA_MIN_GAP = 14         # Presiembra: como máximo hasta siembra-14 (no dentro de (-14,0])

def emax_eff(dose, Emax, ED50, h=1.0):
    """E(d) = Emax * d^h / (ED50^h + d^h); si dose NaN→usa 1 (escala relativa)."""
    d = 1.0 if (dose is None or np.isnan(dose)) else float(dose)
    if d <= 0:
        return 0.0
    den = (ED50 ** h + d ** h)
    if den <= 0:
        return 0.0
    return float(np.clip(Emax * (d ** h) / den, 0.0, 100.0))

def weights_window(dates_array: np.ndarray, start_date: dt.date, days: int) -> np.ndarray:
    if (start_date is None) or (days is None) or (int(days) <= 0):
        return np.zeros_like(dates_array, float)
    d0 = start_date
    d1 = start_date + dt.timedelta(days=int(days))
    return ((dates_array >= d0) & (dates_array < d1)).astype(float)

def perdida_rinde_pct_model(x_plants: float) -> float:
    # Misma curva que tu app base
    return float(0.375 * x_plants / (1.0 + (0.375 * x_plants / 76.639)))

# --------- Aplicación de tratamientos sobre un entorno (por ensayo) ----------
def aplicar_tratamientos_y_perdida(env: dict, tratamientos_rows: pd.DataFrame,
                                   pars: dict) -> dict:
    """
    env: diccionario de build_env_for_ensayo()
    tratamientos_rows: sub-dataframe con todas las filas de ese ensayo
    pars: {'psr':(Emax,ED50),'pre':(Emax,ED50),'post':(Emax,ED50)}
    """
    ts = env["ts"]
    fechas_d = ts.dt.date.values
    mask_since = env["mask_since"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]
    factor_area = env["factor_area_to_plants"]

    # Multiplicadores por estado
    c1 = np.ones_like(fechas_d, float)
    c2 = np.ones_like(fechas_d, float)
    c3 = np.ones_like(fechas_d, float)
    c4 = np.ones_like(fechas_d, float)

    sow = env["sow"]
    n_warn = 0
    for _, r in tratamientos_rows.iterrows():
        tpo = str(r["tipo"]).lower().strip()
        app = r["fecha_aplicacion"]
        R = int(r["residual_dias"]) if not pd.isna(r["residual_dias"]) else POST_RESIDUAL_DEFAULT
        dose = None if pd.isna(r["dosis"]) else float(r["dosis"])

        # Ventanas y sensibilidades por tipo
        if tpo == "presiembra_residual":
            # Solo válido si app <= sow - 14
            if app > (sow - dt.timedelta(days=PRESIEMBRA_MIN_GAP)):
                n_warn += 1
                continue
            eff = emax_eff(dose, *pars["psr"])  # %
            w = weights_window(fechas_d, app, R)
            red = np.clip(1.0 - (eff/100.0)*w, 0.0, 1.0)
            np.multiply(c1, red, out=c1)  # S1 y S2 solamente
            np.multiply(c2, red, out=c2)

        elif tpo == "preemergente_residual":
            # Solo dentro de [sow, sow+10]
            if (app < sow) or (app > sow + dt.timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)):
                n_warn += 1
                continue
            eff = emax_eff(dose, *pars["pre"])  # %
            w = weights_window(fechas_d, app, R)
            red = np.clip(1.0 - (eff/100.0)*w, 0.0, 1.0)
            np.multiply(c1, red, out=c1)  # S1 y S2 solamente
            np.multiply(c2, red, out=c2)

        elif tpo == "postemergente":
            eff = emax_eff(dose, *pars["post"])  # %
            w = weights_window(fechas_d, app, R)
            red = np.clip(1.0 - (eff/100.0)*w, 0.0, 1.0)
            # Post: actúa sobre S1–S4
            np.multiply(c1, red, out=c1)
            np.multiply(c2, red, out=c2)
            np.multiply(c3, red, out=c3)
            np.multiply(c4, red, out=c4)

        else:
            n_warn += 1
            continue

    if n_warn > 0:
        st.info(f"({n_warn}) aplicaciones fuera de ventana o tipo no reconocido fueron ignoradas en algún ensayo.")

    # Serie controlada
    tot_ctrl_daily = (S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4)
    # x3: integral (suma diaria) desde siembra
    x3 = float(np.nansum(tot_ctrl_daily[mask_since]))
    loss_pct = perdida_rinde_pct_model(x3)

    # También devolvemos x2 (sin control) como diagnóstico
    x2 = float(np.nansum((S1_pl + S2_pl + S3_pl + S4_pl)[mask_since]))
    return {
        "x2": x2,
        "x3": x3,
        "loss_pct_model": loss_pct,
    }

# --------- Previsualización rápida por ensayo con parámetros provisionales ----------
pars_preview = {
    "psr": (Emax_psr, ED50_psr),
    "pre": (Emax_pre, ED50_pre),
    "post": (Emax_post, ED50_post),
}
rows_preview = []
for ensayo_id, env in env_by_id.items():
    sub = df_tto[df_tto["ensayo_id"] == ensayo_id]
    if sub.empty:
        continue
    r = aplicar_tratamientos_y_perdida(env, sub, pars_preview)
    obs = float(sub["perdida_rinde_pct_obs"].mean())
    rows_preview.append({
        "ensayo_id": ensayo_id,
        "siembra": str(env["sow"]),
        "loss_obs_%": obs,
        "loss_model_%": r["loss_pct_model"],
        "x2": r["x2"],
        "x3": r["x3"],
        "n_tratamientos": int(len(sub)),
    })

if rows_preview:
    st.subheader("Previsualización (con parámetros provisionales)")
    df_prev = pd.DataFrame(rows_preview)
    df_prev["error_abs_%"] = (df_prev["loss_model_%"] - df_prev["loss_obs_%"]).abs()
    st.dataframe(df_prev, use_container_width=True)
else:
    st.info("No hay ensayos con tratamientos para previsualizar.")


# =================== Sección 7 — Calibración por MSE de pérdida (%) ===================

st.header("Calibración de eficacia (Emax/ED50) por MSE")

with st.expander("Opciones de calibración", expanded=True):
    max_iters = st.number_input("Evaluaciones (búsqueda aleatoria)", 100, 100000, 4000, 100)
    seed = st.number_input("Semilla aleatoria", 0, 2_000_000_000, 123, 1)

    st.markdown("**Límites de parámetros (uniforme):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        Emax_lo, Emax_hi = st.slider("Emax (%) — rango común", 0, 100, (40, 95), 1)
    with c2:
        ED50_lo, ED50_hi = st.slider("ED50 (dosis) — rango común", 1, 300, (10, 120), 1)
    with c3:
        use_shared_post = st.checkbox("Usar mismos Emax/ED50 para POST que PRE", value=False)

    st.caption("El objetivo minimiza el **MSE** entre pérdida modelada (%) y pérdida observada (%), promediado en ensayos.")

valid_ensayos = [eid for eid in env_by_id.keys() if not df_tto[df_tto["ensayo_id"]==eid].empty]
if not valid_ensayos:
    st.warning("No hay ensayos válidos con tratamientos para calibrar.")
else:
    rng = np.random.default_rng(int(seed))

    def sample_params():
        def pick():
            return (float(rng.uniform(Emax_lo, Emax_hi)),
                    float(rng.uniform(ED50_lo, ED50_hi)))
        psr = pick()
        pre = pick()
        post = pre if use_shared_post else pick()
        return {"psr": psr, "pre": pre, "post": post}

    def evaluate_params(pars):
        errs = []
        for eid in valid_ensayos:
            env = env_by_id[eid]
            sub = df_tto[df_tto["ensayo_id"] == eid]
            obs = float(sub["perdida_rinde_pct_obs"].mean())
            r = aplicar_tratamientos_y_perdida(env, sub, pars)
            errs.append((r["loss_pct_model"] - obs) ** 2)
        return float(np.mean(errs))

    # Búsqueda aleatoria simple
    best = None
    best_pars = None
    prog = st.progress(0.0)
    for i in range(1, int(max_iters)+1):
        pars = sample_params()
        mse = evaluate_params(pars)
        if (best is None) or (mse < best):
            best = mse
            best_pars = pars
        if (i % max(1, int(max_iters)//100) == 0) or (i == int(max_iters)):
            prog.progress(min(1.0, i/float(max_iters)))
    prog.empty()

    st.subheader("Resultado de calibración")
    st.markdown(
        f"**MSE óptimo:** `{best:.4f}`  \n"
        f"**Emax/ED50 Presiembra:** `{best_pars['psr'][0]:.2f}` / `{best_pars['psr'][1]:.2f}`  \n"
        f"**Emax/ED50 Preemergente:** `{best_pars['pre'][0]:.2f}` / `{best_pars['pre'][1]:.2f}`  \n"
        f"**Emax/ED50 Postemergente:** `{best_pars['post'][0]:.2f}` / `{best_pars['post'][1]:.2f}`"
    )

    # Tabla por ensayo con parámetros calibrados
    rows_fit = []
    for eid in valid_ensayos:
        env = env_by_id[eid]
        sub = df_tto[df_tto["ensayo_id"] == eid]
        r = aplicar_tratamientos_y_perdida(env, sub, best_pars)
        obs = float(sub["perdida_rinde_pct_obs"].mean())
        rows_fit.append({
            "ensayo_id": eid,
            "siembra": str(env["sow"]),
            "loss_obs_%": obs,
            "loss_model_%_cal": r["loss_pct_model"],
            "x2": r["x2"],
            "x3": r["x3"],
            "n_tratamientos": int(len(sub)),
            "MSE_indiv": (r["loss_pct_model"] - obs) ** 2
        })
    df_fit = pd.DataFrame(rows_fit).sort_values("ensayo_id").reset_index(drop=True)

    st.subheader("Ajuste por ensayo (parámetros calibrados)")
    st.dataframe(df_fit, use_container_width=True)

    # Descargas
    st.download_button(
        "Descargar parámetros calibrados (JSON)",
        data=json.dumps({
            "MSE": best,
            "params": {
                "presiembra_residual": {"Emax": best_pars["psr"][0], "ED50": best_pars["psr"][1]},
                "preemergente_residual": {"Emax": best_pars["pre"][0], "ED50": best_pars["pre"][1]},
                "postemergente": {"Emax": best_pars["post"][0], "ED50": best_pars["post"][1]},
            }
        }, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="parametros_calibrados.json",
        mime="application/json"
    )
    st.download_button(
        "Descargar ajuste por ensayo (CSV)",
        data=df_fit.to_csv(index=False).encode("utf-8"),
        file_name="ajuste_por_ensayo.csv",
        mime="text/csv"
    )







