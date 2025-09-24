# -*- coding: utf-8 -*-
"""
calibra_predweem.py — Calibración modular para PREDWEEM

Objetivo mínimo (runnable sin dependencias extra):
- Si hay x_obs (pl·m²) y pérdida_% observada → calibrar curva de pérdida (i, A).
- Si NO hay x_obs → estimar x con un modelo simple a partir de EMERREL/EMERAC y cronograma, con
  factor de conversión "alpha" (pl·m² por unidad de EMERREL integrada) y tope A2; calibrar (alpha, i, A).

Notas:
- No requiere SciPy. Usa búsqueda aleatoria + grid local.
- La simulación de control es simplificada (todas las edades afectadas igual). Para calibración fina
  con S1–S4 y Ciec, integrar con tu app principal.
"""

import pandas as pd
import numpy as np
import json, math, os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# -----------------------------
# Utilidades
# -----------------------------

def read_excel_data(path: str):
    xl = pd.read_excel(path, sheet_name=None)
    meta = xl.get("metadata", pd.DataFrame())
    emerrel = xl.get("series_EMERREL", pd.DataFrame())
    emerac = xl.get("series_EMERAC", pd.DataFrame())
    inter = xl.get("intervenciones", pd.DataFrame())
    obs = xl.get("observaciones", pd.DataFrame())
    xobs = xl.get("x_obs", pd.DataFrame())
    canopy = xl.get("canopy", pd.DataFrame())
    return meta, emerrel, emerac, inter, obs, xobs, canopy


def ensure_loss_pct(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    if "perdida_pct" not in obs.columns or obs["perdida_pct"].isna().all():
        if {"rend_trat_kg_ha","rend_testigo_kg_ha"}.issubset(obs.columns):
            obs["perdida_pct"] = (obs["rend_testigo_kg_ha"] - obs["rend_trat_kg_ha"]) / obs["rend_testigo_kg_ha"] * 100.0
    return obs


def loss_rect_hyperbola(x, i, A):
    x = np.asarray(x, float)
    return (i * x) / (1.0 + (i * x / max(1e-9, A)))


@dataclass
class CalibConfig:
    a2_cap: float = 250.0
    pc_from_col: str = "pc_inicio"
    pc_to_col: str = "pc_fin"
    use_emerac: bool = False  # si no hay EMERREL
    emerrel_in_percent: bool = True
    seed: int = 123


# -----------------------------
# Estimador x (simple) cuando no hay x_obs
# -----------------------------

def estimate_x_simple(emerrel_df: pd.DataFrame,
                      meta_row: pd.Series,
                      inter_df: pd.DataFrame,
                      alpha: float,
                      a2_cap: float) -> float:
    """
    Estimación básica de x desde EMERREL integrando en el Periodo Crítico y aplicando
    una atenuación por intervenciones (todas las edades afectadas por igual).
    """
    df = emerrel_df.copy()
    df = df[df["id_experimento"] == meta_row["id_experimento"]]
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    if df.empty:
        return np.nan

    # Traer PC
    pc_ini = pd.to_datetime(meta_row["pc_inicio"]) if "pc_inicio" in meta_row else None
    pc_fin = pd.to_datetime(meta_row["pc_fin"]) if "pc_fin" in meta_row else None
    if pc_ini is None or pc_fin is None or pd.isna(pc_ini) or pd.isna(pc_fin):
        return np.nan

    # EMERREL (si está en %) → 0..1
    if df["EMERREL"].max() > 1.01:
        df["EMERREL"] = df["EMERREL"] / 100.0
    df = df[(df["fecha"] >= pc_ini) & (df["fecha"] <= pc_fin)]
    if df.empty:
        return 0.0

    base_daily_plants = df["EMERREL"].to_numpy(float) * float(alpha)

    # Aplicar control simple por ventanas
    inter_loc = inter_df[inter_df["id_experimento"] == meta_row["id_experimento"]].copy()
    fechas = pd.to_datetime(df["fecha"]).dt.date.values
    mult = np.ones_like(base_daily_plants, float)
    for _, r in inter_loc.iterrows():
        kind = str(r.get("tipo",""))
        d0 = pd.to_datetime(r.get("fecha")).date() if not pd.isna(r.get("fecha")) else None
        R = int(r.get("residual_dias", 0) or 0)
        eff = float(r.get("eficacia_pct", 0) or 0)/100.0
        if not d0 or R<=0 or eff<=0: 
            continue
        d1 = d0 + pd.Timedelta(days=R)
        mask = (fechas >= d0) & (fechas < d1)
        mult[mask] *= (1.0 - eff)

    daily_ctrl = base_daily_plants * mult
    # Tope A2 por acumulación
    cum = 0.0
    acc = []
    for v in daily_ctrl:
        allowed = max(0.0, a2_cap - cum)
        vv = min(max(0.0, v), allowed)
        acc.append(vv)
        cum += vv
    x = float(np.sum(acc))
    return x


# -----------------------------
# Calibración
# -----------------------------

def random_search_objective(y_true, x_vals, fit_alpha=False, a2_cap=250.0, n=5000, seed=123):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(n)):
        if fit_alpha:
            alpha = 10**rng.uniform(-1.0, 2.2)  # ~0.1..160
        else:
            alpha = None
        i = rng.uniform(0.05, 0.8)
        A = rng.uniform(30.0, 95.0)
        if fit_alpha:
            y_pred = loss_rect_hyperbola(np.asarray(x_vals)*alpha, i, A)
        else:
            y_pred = loss_rect_hyperbola(np.asarray(x_vals), i, A)
        err = np.nanmean((y_true - y_pred)**2)
        if (best is None) or (err < best["mse"]):
            best = {"i":float(i), "A":float(A), "alpha":(float(alpha) if fit_alpha else None), "mse":float(err)}
    return best


def local_refine(y_true, x_vals, sol, fit_alpha=False):
    # pequeño grid alrededor
    i0, A0, a0 = sol["i"], sol["A"], sol.get("alpha")
    grid_i = np.linspace(max(0.01, i0*0.6), i0*1.4, 11)
    grid_A = np.linspace(max(5.0, A0*0.7), A0*1.3, 11)
    grid_alpha = [a0] if (not fit_alpha or a0 is None) else np.linspace(max(0.01, a0*0.5), a0*1.5, 9)
    best = None
    for i in grid_i:
        for A in grid_A:
            for alpha in grid_alpha:
                if fit_alpha:
                    y_pred = loss_rect_hyperbola(np.asarray(x_vals)*alpha, i, A)
                else:
                    y_pred = loss_rect_hyperbola(np.asarray(x_vals), i, A)
                err = np.nanmean((y_true - y_pred)**2)
                if (best is None) or (err < best["mse"]):
                    best = {"i":float(i), "A":float(A), "alpha":(float(alpha) if fit_alpha else None), "mse":float(err)}
    return best


def calibrate_from_excel(path: str, cfg: CalibConfig) -> Dict:
    meta, emerrel, emerac, inter, obs, xobs, canopy = read_excel_data(path)
    obs = ensure_loss_pct(obs)
    # Preparar dataset (por ahora 1 experimento o varios agregados)
    ids = sorted(set(meta["id_experimento"])) if "id_experimento" in meta.columns else []
    if not ids:
        raise ValueError("metadata.id_experimento está vacío")

    y_list = []
    x_list = []
    x_for_alpha_list = []  # si fit_alpha

    for _, m in meta.iterrows():
        ide = m["id_experimento"]
        obs_row = obs[obs["id_experimento"]==ide]
        if obs_row.empty:
            continue
        y_true = float(obs_row.iloc[0]["perdida_pct"]) if "perdida_pct" in obs_row.columns else np.nan
        if np.isnan(y_true):
            continue

        # x observado?
        x_row = xobs[xobs["id_experimento"]==ide] if not xobs.empty else pd.DataFrame()
        if (not x_row.empty) and ("x_pl_m2" in x_row.columns) and (not pd.isna(x_row.iloc[0]["x_pl_m2"])):
            x_val = float(x_row.iloc[0]["x_pl_m2"])
            x_list.append(x_val)
            y_list.append(y_true)
        else:
            # estimar x desde EMERREL simple
            emer_df = emerrel if not cfg.use_emerac else emerac.rename(columns={"EMERAC":"EMERREL"})
            if emer_df.empty:
                continue
            x_hat = estimate_x_simple(emer_df, m, inter, alpha=1.0, a2_cap=cfg.a2_cap)
            if np.isnan(x_hat):
                continue
            # guardo x_hat "base" para calibrar alpha también
            x_for_alpha_list.append((x_hat, y_true))

    # Estrategias de calibración
    if x_list:
        # Caso 1: tengo x observado → calibrar i, A
        y = np.array(y_list, float)
        x = np.array(x_list, float)
        sol = random_search_objective(y, x, fit_alpha=False, a2_cap=cfg.a2_cap, n=5000, seed=cfg.seed)
        sol = local_refine(y, x, sol, fit_alpha=False)
        fitted = {"i":sol["i"], "A":sol["A"], "alpha":None, "mse":sol["mse"], "mode":"fit_iA_with_xobs"}
    elif x_for_alpha_list:
        # Caso 2: no hay x observado → calibrar alpha, i, A
        xy = np.array(x_for_alpha_list, float)
        x0 = xy[:,0]
        y = xy[:,1]
        sol = random_search_objective(y, x0, fit_alpha=True, a2_cap=cfg.a2_cap, n=12000, seed=cfg.seed)
        sol = local_refine(y, x0, sol, fit_alpha=True)
        fitted = {"i":sol["i"], "A":sol["A"], "alpha":sol["alpha"], "mse":sol["mse"], "mode":"fit_alpha_iA_from_EMERREL"}
    else:
        raise ValueError("No hay datos suficientes: cargá x_obs o series EMERREL/EMERAC + observaciones")

    return fitted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, required=True, help="Ruta al Excel de calibración")
    parser.add_argument("--a2", type=float, default=250.0, help="Tope A2 (pl·m²)")
    parser.add_argument("--use-emerac", action="store_true", help="Usar la hoja series_EMERAC en lugar de EMERREL")
    args = parser.parse_args()

    cfg = CalibConfig(a2_cap=args.a2, use_emerac=args.use_emerac)
    out = calibrate_from_excel(args.excel, cfg)

    # Guardar JSON de parámetros
    out_json = os.path.splitext(args.excel)[0] + "_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Guardado:", out_json)

    # También un CSV con resumen
    pd.DataFrame([out]).to_csv(os.path.splitext(args.excel)[0] + "_params.csv", index=False)
    print("i=%.4f  A=%.2f  alpha=%s  mse=%.4f  mode=%s" % (out["i"], out["A"], str(out.get("alpha")), out["mse"], out["mode"]))



