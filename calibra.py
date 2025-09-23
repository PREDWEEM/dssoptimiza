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










