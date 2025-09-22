# -*- coding: utf-8 -*-
import io, re, json, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Calibración PREDWEEM (ensayos independientes)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Carga un CSV de ensayos o pegá una URL raw. Si no tenés datos a mano, podés usar un ejemplo sintético.")

# ------------------ Helpers de IO ------------------
def sniff_sep_dec(sample_text: str):
    sample = sample_text[:4000]
    sep = max({s: sample.count(s) for s in [",",";","\t"]}, key=lambda k: {",":0,";":1,"\t":2}.get)  # preferencia suave
    # Si hay comas seguidas de dígitos, probablemente decimal=","
    dec = "," if (sample.count(",")>sample.count(".") and re.search(r",\d", sample)) else "."
    return sep, dec

def read_csv_flexible(file_or_bytes, sep_opt="auto", dec_opt="auto"):
    if isinstance(file_or_bytes, (bytes, bytearray)):
        head = file_or_bytes[:4000].decode("utf-8", errors="ignore")
        sep_g, dec_g = sniff_sep_dec(head)
        sep = sep_g if sep_opt=="auto" else sep_opt
        dec = dec_g if dec_opt=="auto" else dec_opt
        df = pd.read_csv(io.BytesIO(file_or_bytes), sep=sep, decimal=dec, engine="python")
        return df, {"sep": sep, "dec": dec}
    else:
        # file_or_bytes es un path o buffer ya abierto
        df = pd.read_csv(file_or_bytes)
        return df, {"sep": ",", "dec": "."}

@st.cache_data(show_spinner=False)
def fetch_url(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

# ------------------ Entrada de datos ------------------
with st.sidebar:
    st.header("Datos de ensayos")
    up = st.file_uploader("Subir ensayos.csv", type=["csv"])
    url = st.text_input("…o URL raw de GitHub/HTTP", placeholder="https://raw.githubusercontent.com/usuario/repo/main/ensayos.csv")
    use_demo = st.checkbox("Usar datos sintéticos de ejemplo", value=False)

df = None
meta = None
if up is not None:
    raw = up.read()
    df, meta = read_csv_flexible(raw)
elif url.strip():
    try:
        raw = fetch_url(url.strip())
        df, meta = read_csv_flexible(raw)
    except Exception as e:
        st.error(f"No pude leer la URL: {e}")
elif use_demo:
    # Dataset de ejemplo (3 ensayos)
    df = pd.DataFrame({
        "ensayo": ["E1","E2","E3"],
        "fecha_siembra": ["2025-06-01","2025-06-15","2025-07-01"],
        "tratamientos": ["pre70+post60","pre70","post60"],
        "dens_obs": [30, 55, 40],
        "rinde_obs": [3200, 2800, 3500],
        "rinde_clean": [4000, 3900, 4000],
    })
    meta = {"sep": ",", "dec": "."}
else:
    st.info("Subí un CSV, pegá una URL raw o marcá 'Usar datos sintéticos de ejemplo'.")
    st.stop()

st.success(f"Datos cargados. sep='{meta['sep']}', dec='{meta['dec']}'")
st.dataframe(df, use_container_width=True)

# ------------------ Limpieza mínima ------------------
need_cols = {"ensayo","fecha_siembra","tratamientos","dens_obs","rinde_obs","rinde_clean"}
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}")
    st.stop()

# Tipos
df["fecha_siembra"] = pd.to_datetime(df["fecha_siembra"], errors="coerce")
for c in ["dens_obs","rinde_obs","rinde_clean"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["fecha_siembra","dens_obs","rinde_obs","rinde_clean"]).reset_index(drop=True)

# Pérdida observada %
df["loss_obs"] = 100.0 * (df["rinde_clean"] - df["rinde_obs"]) / df["rinde_clean"]
st.caption("Cálculo de pérdida observada (%) = 100*(rinde_clean - rinde_obs)/rinde_clean")

# ------------------ Parámetros y simulador ------------------
st.markdown("### Parámetros a calibrar")
with st.expander("Rangos de parámetros"):
    fc2_min, fc2_max = st.slider("FC S2 (mín–máx)", 0.0, 1.0, (0.2, 0.5), 0.05)
    fc3_min, fc3_max = st.slider("FC S3 (mín–máx)", 0.0, 1.0, (0.5, 0.8), 0.05)
    fc4_min, fc4_max = st.slider("FC S4 (mín–máx)", 0.5, 1.2, (0.9, 1.1), 0.05)
    cap_opts = st.multiselect("MAX_PLANTS_CAP (opciones)", [125, 250, 500, 850], default=[250, 500])
    Imin, Imax = st.slider("Imax (pérdida máxima %)", 30, 100, (60, 90), 1)
    A50min, A50max = st.slider("A50 (eq·pl·m² al 50% pérdida)", 20, 500, (80, 200), 5)

st.caption("La pérdida estimada usa una curva hiperbólica: Loss(%) = (Imax * x) / (A50 + x)")

# ⚠️ Reemplazá esta función por tu simulador real (llamado a PREDWEEM)
def run_predweem(fecha_siembra, tratamientos, FC, MAX_CAP):
    # Aquí deberías:
    # 1) Generar EMERREL del ensayo (o leerla), 2) aplicar (1−Ciec), 3) estados S1–S4,
    # 4) aplicar tratamientos (eficiencia/ventanas), 5) cap A2, 6) integrar en PCI → x (eq·pl·m²)
    # Para demo, devolvemos una x ficticia “coherente” con el string de tratamientos:
    base = 150.0
    if "pre70" in str(tratamientos): base *= 0.30
    if "post60" in str(tratamientos): base *= 0.40
    # Ponderar por competitividad (muy grosero, sólo demo):
    peso = 0.2*FC["S2"] + 0.4*FC["S3"] + 0.4*FC["S4"]
    x = min(MAX_CAP, base * (0.6 + 0.8*peso))
    return max(0.0, float(x))

# Curva de pérdida
def loss_curve(x, Imax, A50):
    x = float(max(0.0, x))
    return (Imax * x) / (A50 + x)

# Objetivo
def objective(params_vec):
    # params = [FC_S2, FC_S3, FC_S4, MAX_CAP, Imax, A50]
    FC = {"S1": 0.0, "S2": params_vec[0], "S3": params_vec[1], "S4": params_vec[2]}
    MAX_CAP = int(round(params_vec[3]))
    Imax = params_vec[4]; A50 = params_vec[5]
    pred = []
    for _, r in df.iterrows():
        x = run_predweem(r["fecha_siembra"], r["tratamientos"], FC, MAX_CAP)
        pred.append(loss_curve(x, Imax, A50))
    pred = np.array(pred, dtype=float)
    obs = df["loss_obs"].to_numpy(dtype=float)
    return float(np.mean((obs - pred)**2))

# ------------------ Optimizador: SciPy si está, si no Random Search ------------------
use_scipy = False
try:
    from scipy.optimize import minimize
    use_scipy = True
except Exception:
    use_scipy = False

st.subheader("Calibración")
if use_scipy:
    st.caption("Usando SciPy: Nelder-Mead (o similar) sobre una inicialización razonable.")
    # Inicial
    x0 = np.array([
        np.mean([fc2_min, fc2_max]),
        np.mean([fc3_min, fc3_max]),
        np.mean([fc4_min, fc4_max]),
        cap_opts[0] if len(cap_opts) else 250,
        np.mean([Imin, Imax]),
        np.mean([A50min, A50max]),
    ], dtype=float)

    bounds = [
        (fc2_min, fc2_max),
        (fc3_min, fc3_max),
        (fc4_min, fc4_max),
        (min(cap_opts) if len(cap_opts) else 125, max(cap_opts) if len(cap_opts) else 850),
        (Imin, Imax),
        (A50min, A50max),
    ]

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 2000})
    params = res.x
    score = objective(params)**0.5
else:
    st.caption("SciPy no disponible → usando Búsqueda Aleatoria (fallback).")
    RNG = np.random.default_rng(123)
    best, best_score = None, np.inf
    caps = cap_opts if len(cap_opts) else [250, 500]
    for _ in range(4000):  # ajustar si querés
        params = np.array([
            RNG.uniform(fc2_min, fc2_max),
            RNG.uniform(fc3_min, fc3_max),
            RNG.uniform(fc4_min, fc4_max),
            RNG.choice(caps),
            RNG.uniform(Imin, Imax),
            RNG.uniform(A50min, A50max),
        ], dtype=float)
        sc = objective(params)
        if sc < best_score:
            best, best_score = params.copy(), sc
    params = best
    score = best_score**0.5

# ------------------ Resultados ------------------
FC = {"S1": 0.0, "S2": params[0], "S3": params[1], "S4": params[2]}
MAX_CAP = int(round(params[3])); Imax = params[4]; A50 = params[5]
st.markdown("### Parámetros calibrados")
st.json({
    "FC": FC,
    "MAX_PLANTS_CAP": MAX_CAP,
    "Imax": round(float(Imax), 3),
    "A50": round(float(A50), 3),
    "RMSE_loss_pct": round(float(score), 3),
})

# Observado vs Predicho
pred = []
for _, r in df.iterrows():
    x = run_predweem(r["fecha_siembra"], r["tratamientos"], FC, MAX_CAP)
    pred.append(loss_curve(x, Imax, A50))
df_eval = df[["ensayo","fecha_siembra","tratamientos","loss_obs"]].copy()
df_eval["loss_pred"] = np.array(pred, dtype=float)
st.subheader("Observado vs Predicho (pérdida %)")
st.dataframe(df_eval, use_container_width=True)

# Pequeño gráfico
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_eval["loss_obs"], y=df_eval["loss_pred"],
                         mode="markers+text", text=df_eval["ensayo"], textposition="top center",
                         name="Ensayos"))
mx = max(1.0, float(df_eval[["loss_obs","loss_pred"]].to_numpy().max()))
fig.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines", name="1:1", line=dict(dash="dash")))
fig.update_layout(title="Pérdida observada vs predicha (%)", xaxis_title="Observada (%)", yaxis_title="Predicha (%)",
                  margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)
