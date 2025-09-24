# -*- coding: utf-8 -*-
"""
calibra_predweem.py — Calibración completa PREDWEEM (multi‑ensayo + CV)

Características
- Lee el Excel de plantilla (sheets: metadata, series_EMERREL/EMERAC, intervenciones, observaciones, x_obs, canopy opcional).
- Arma dataset por id_experimento y permite dos rutas de calibración:
  (A) Con x_obs → ajusta i y A de la curva hiperbólica de pérdidas.
  (B) Sin x_obs → estima x a partir de EMERREL/EMERAC + cronograma simple (todas las edades afectadas igual),
      y ajusta alpha (factor EMERREL→plantas), i y A simultáneamente.
- Cross‑validation (k‑fold estratificada por id_experimento) para evaluar generalización.
- Salidas: JSON/CSV con parámetros, métricas globales y por fold; CSV con predicciones por ensayo.
- Sin dependencias pesadas: usa numpy/pandas + búsqueda aleatoria y refinamiento local.

Limitaciones (para mantenerlo portable)
- La estimación de x en la ruta (B) es simplificada: no separa S1–S4 ni aplica Ciec explícito.
  Para calibración fina, integrar con tu app principal (cohortes + Ciec) y usar la misma estructura de orquestación
  de este script (random search + refine + CV).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json, math, os, sys, argparse, itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# --------------------------------------------------
# Utilidades I/O
# --------------------------------------------------

def read_excel_data(path: str):
    xl = pd.read_excel(path, sheet_name=None)
    def get(name):
        return xl.get(name, pd.DataFrame())
    return (
        get("metadata"),
        get("series_EMERREL"),
        get("series_EMERAC"),
        get("intervenciones"),
        get("observaciones"),
        get("x_obs"),
        get("canopy"),
    )


def ensure_loss_pct(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    if "perdida_pct" not in obs.columns or obs["perdida_pct"].isna().all():
        if {"rend_trat_kg_ha","rend_testigo_kg_ha"}.issubset(obs.columns):
            obs["perdida_pct"] = (obs["rend_testigo_kg_ha"] - obs["rend_trat_kg_ha"]) / obs["rend_testigo_kg_ha"] * 100.0
    return obs

# --------------------------------------------------
# Modelo de pérdida — Recta hiperbólica
# --------------------------------------------------

def loss_rect_hyperbola(x, i, A):
    x = np.asarray(x, float)
    return (i * x) / (1.0 + (i * x / max(1e-9, A)))

# --------------------------------------------------
# Configuración
# --------------------------------------------------

@dataclass
class CalibConfig:
    a2_cap: float = 250.0
    use_emerac: bool = False
    emerrel_in_percent: Optional[bool] = None  # None = autodetectar por ensayo
    k_folds: int = 5
    seed: int = 123
    n_random: int = 12000
    refine_grid: int = 11  # puntos por dimensión en el refino local
    fit_alpha_when_no_x: bool = True

# --------------------------------------------------
# Estimación de x (ruta B) — Simple
# --------------------------------------------------

def estimate_x_simple(emer_df: pd.DataFrame, meta_row: pd.Series, inter_df: pd.DataFrame, alpha: float, a2_cap: float) -> float:
    df = emer_df.copy()
    df = df[df["id_experimento"] == meta_row["id_experimento"]]
    if df.empty:
        return np.nan
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    # Periodo Crítico desde metadata
    pc_ini = pd.to_datetime(meta_row.get("pc_inicio")) if "pc_inicio" in meta_row else None
    pc_fin = pd.to_datetime(meta_row.get("pc_fin")) if "pc_fin" in meta_row else None
    if pc_ini is None or pc_fin is None or pd.isna(pc_ini) or pd.isna(pc_fin):
        return np.nan

    # EMERREL a 0..1 si viene en % (autodetección si no fue forzada)
    emer = df.get("EMERREL")
    if emer is None or emer.isna().all():
        return np.nan
    if emer.max() > 1.01:
        emer = emer / 100.0
    df = df.assign(EMERREL=emer)

    df = df[(df["fecha"] >= pc_ini) & (df["fecha"] <= pc_fin)]
    if df.empty:
        return 0.0

    fechas = pd.to_datetime(df["fecha"]).dt.date.values
    base_daily_plants = df["EMERREL"].to_numpy(float) * float(alpha)

    # Intervenciones (todas las edades por igual)
    inter_loc = inter_df[inter_df["id_experimento"] == meta_row["id_experimento"]].copy()
    mult = np.ones_like(base_daily_plants, float)
    for _, r in inter_loc.iterrows():
        d0 = pd.to_datetime(r.get("fecha")).date() if not pd.isna(r.get("fecha")) else None
        R = int(r.get("residual_dias", 0) or 0)
        eff = float(r.get("eficacia_pct", 0) or 0) / 100.0
        if not d0 or R <= 0 or eff <= 0:
            continue
        d1 = d0 + pd.Timedelta(days=R)
        mask = (fechas >= d0) & (fechas < d1)
        mult[mask] *= (1.0 - eff)

    daily_ctrl = base_daily_plants * mult

    # Tope A2 (acumulado)
    cum = 0.0
    acc = []
    for v in daily_ctrl:
        allowed = max(0.0, a2_cap - cum)
        vv = min(max(0.0, v), allowed)
        acc.append(vv)
        cum += vv
    x = float(np.sum(acc))
    return x

# --------------------------------------------------
# Orquestación de calibración
# --------------------------------------------------

def build_dataset(path_excel: str, cfg: CalibConfig):
    meta, emerrel, emerac, inter, obs, xobs, canopy = read_excel_data(path_excel)
    if meta.empty:
        raise ValueError("Hoja 'metadata' vacía o faltante.")
    obs = ensure_loss_pct(obs)

    ids = list(dict.fromkeys(meta["id_experimento"].astype(str)))
    rows = []
    for _, m in meta.iterrows():
        ide = str(m["id_experimento"])
        obs_row = obs[obs["id_experimento"] == ide]
        if obs_row.empty:
            continue
        y_true = float(obs_row.iloc[0]["perdida_pct"]) if "perdida_pct" in obs_row.columns else np.nan
        if np.isnan(y_true):
            continue

        # Preferir x_obs si existe
        x_row = xobs[xobs["id_experimento"] == ide] if not xobs.empty else pd.DataFrame()
        if not x_row.empty and ("x_pl_m2" in x_row.columns) and (not pd.isna(x_row.iloc[0]["x_pl_m2"])):
            rows.append({"id": ide, "y": y_true, "x": float(x_row.iloc[0]["x_pl_m2"]), "x_from_emr": False})
        else:
            # Ruta B: EMERREL/EMERAC
            if cfg.use_emerac and not emerac.empty and "EMERAC" in emerac.columns:
                emer_df = emerac.rename(columns={"EMERAC": "EMERREL"}).copy()
            else:
                emer_df = emerrel.copy()
            if emer_df.empty or "EMERREL" not in emer_df.columns:
                continue
            # Autodetectar % si es necesario
            subset = emer_df[emer_df["id_experimento"] == ide]
            if subset.empty:
                continue
            em = subset["EMERREL"]
            if cfg.emerrel_in_percent is True:
                emer_df.loc[emer_df["id_experimento"] == ide, "EMERREL"] = em / 100.0
            elif cfg.emerrel_in_percent is None and em.max() > 1.01:
                emer_df.loc[emer_df["id_experimento"] == ide, "EMERREL"] = em / 100.0

            x_hat = estimate_x_simple(emer_df, m, inter, alpha=1.0, a2_cap=cfg.a2_cap)
            if np.isnan(x_hat):
                continue
            rows.append({"id": ide, "y": y_true, "x": float(x_hat), "x_from_emr": True})

    if not rows:
        raise ValueError("No hay filas válidas para calibración. Verificá observaciones/series.")
    df = pd.DataFrame(rows)
    return df

# --------------------------------------------------
# Búsqueda aleatoria + refinamiento local
# --------------------------------------------------

def random_search(y_true, x_vals, fit_alpha=False, n=10000, seed=123):
    rng = np.random.default_rng(seed)
    best = None
    x_vals = np.asarray(x_vals, float)
    y_true = np.asarray(y_true, float)
    for _ in range(int(n)):
        i = rng.uniform(0.05, 0.8)
        A = rng.uniform(30.0, 95.0)
        if fit_alpha:
            alpha = 10 ** rng.uniform(-1.0, 2.2)  # ~0.1..160
            y_pred = loss_rect_hyperbola(x_vals * alpha, i, A)
        else:
            alpha = None
            y_pred = loss_rect_hyperbola(x_vals, i, A)
        mse = float(np.nanmean((y_true - y_pred) ** 2))
        if (best is None) or (mse < best["mse"]):
            best = {"i": float(i), "A": float(A), "alpha": (float(alpha) if alpha is not None else None), "mse": mse}
    return best


def local_refine(y_true, x_vals, sol, fit_alpha=False, grid=11):
    x_vals = np.asarray(x_vals, float)
    y_true = np.asarray(y_true, float)
    i0, A0, a0 = sol["i"], sol["A"], sol.get("alpha")
    grid_i = np.linspace(max(0.01, i0 * 0.6), i0 * 1.4, grid)
    grid_A = np.linspace(max(5.0, A0 * 0.7), A0 * 1.3, grid)
    grid_alpha = [a0] if (not fit_alpha or a0 is None) else np.linspace(max(0.01, a0 * 0.5), a0 * 1.5, max(7, grid-2))
    best = None
    for i in grid_i:
        for A in grid_A:
            for alpha in grid_alpha:
                y_pred = loss_rect_hyperbola(x_vals * alpha, i, A) if fit_alpha else loss_rect_hyperbola(x_vals, i, A)
                mse = float(np.nanmean((y_true - y_pred) ** 2))
                if (best is None) or (mse < best["mse"]):
                    best = {"i": float(i), "A": float(A), "alpha": (float(alpha) if fit_alpha else None), "mse": mse}
    return best

# --------------------------------------------------
# Cross‑validation
# --------------------------------------------------

def kfold_indices(ids: List[str], k: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(list(set(ids))))
    rng.shuffle(uniq)
    folds = [list() for _ in range(k)]
    for i, idv in enumerate(uniq):
        folds[i % k].append(idv)
    return folds


def fit_and_eval(df: pd.DataFrame, cfg: CalibConfig):
    # definir si ajustamos alpha
    fit_alpha = bool(cfg.fit_alpha_when_no_x and df["x_from_emr"].any())

    folds = kfold_indices(df["id"].tolist(), max(2, cfg.k_folds), seed=cfg.seed)
    all_preds = []
    fold_summ = []

    for fold_idx, valid_ids in enumerate(folds, 1):
        is_valid = df["id"].isin(valid_ids)
        df_train = df.loc[~is_valid].reset_index(drop=True)
        df_valid = df.loc[is_valid].reset_index(drop=True)
        if df_train.empty or df_valid.empty:
            continue
        y_tr, x_tr = df_train["y"].to_numpy(float), df_train["x"].to_numpy(float)
        y_va, x_va = df_valid["y"].to_numpy(float), df_valid["x"].to_numpy(float)

        sol = random_search(y_tr, x_tr, fit_alpha=fit_alpha, n=cfg.n_random, seed=cfg.seed + fold_idx)
        sol = local_refine(y_tr, x_tr, sol, fit_alpha=fit_alpha, grid=cfg.refine_grid)

        # predicciones
        if fit_alpha and sol.get("alpha") is not None:
            yhat_tr = loss_rect_hyperbola(x_tr * sol["alpha"], sol["i"], sol["A"])
            yhat_va = loss_rect_hyperbola(x_va * sol["alpha"], sol["i"], sol["A"])
        else:
            yhat_tr = loss_rect_hyperbola(x_tr, sol["i"], sol["A"])
            yhat_va = loss_rect_hyperbola(x_va, sol["i"], sol["A"])

        def metrics(y, yhat):
            y, yhat = np.asarray(y, float), np.asarray(yhat, float)
            mse = float(np.nanmean((y - yhat) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.nanmean(np.abs(y - yhat)))
            # R^2 simple
            ss_res = float(np.nansum((y - yhat) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
            return rmse, mae, r2

        rmse_tr, mae_tr, r2_tr = metrics(y_tr, yhat_tr)
        rmse_va, mae_va, r2_va = metrics(y_va, yhat_va)

        fold_summ.append({
            "fold": fold_idx,
            "n_train": int(len(y_tr)),
            "n_valid": int(len(y_va)),
            "i": sol["i"],
            "A": sol["A"],
            "alpha": sol.get("alpha"),
            "rmse_train": rmse_tr,
            "mae_train": mae_tr,
            "r2_train": r2_tr,
            "rmse_valid": rmse_va,
            "mae_valid": mae_va,
            "r2_valid": r2_va,
        })

        # guardar predicciones por ensayo (valid)
        for _id, yv, xv, yhat in zip(df_valid["id"], y_va, x_va, yhat_va):
            all_preds.append({"fold": fold_idx, "id": _id, "y_true": yv, "x": xv, "y_pred": float(yhat)})

    # Re‑entrenar con todo para parámetros finales reportados
    fit_alpha_full = bool(cfg.fit_alpha_when_no_x and df["x_from_emr"].any())
    sol_full = random_search(df["y"], df["x"], fit_alpha=fit_alpha_full, n=cfg.n_random, seed=cfg.seed + 999)
    sol_full = local_refine(df["y"], df["x"], sol_full, fit_alpha=fit_alpha_full, grid=cfg.refine_grid)

    # Métricas globales CV
    cv_df = pd.DataFrame(fold_summ)
    preds_df = pd.DataFrame(all_preds)
    summary = {
        "i": sol_full["i"],
        "A": sol_full["A"],
        "alpha": sol_full.get("alpha"),
        "cv_rmse_valid_mean": float(cv_df["rmse_valid"].mean()) if not cv_df.empty else float("nan"),
        "cv_mae_valid_mean": float(cv_df["mae_valid"].mean()) if not cv_df.empty else float("nan"),
        "cv_r2_valid_mean": float(cv_df["r2_valid"].mean()) if not cv_df.empty else float("nan"),
        "mode": "fit_iA_with_xobs" if not df["x_from_emr"].any() else "fit_alpha_iA_from_EMERREL",
        "n_items": int(len(df)),
        "k_folds": int(len(cv_df)),
    }
    return summary, cv_df, preds_df

# --------------------------------------------------
# Main CLI
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True, help="Ruta al Excel de calibración")
    p.add_argument("--a2", type=float, default=250.0, help="Tope A2 (pl·m²) para la ruta B")
    p.add_argument("--use-emerac", action="store_true", help="Usar series_EMERAC en lugar de EMERREL")
    p.add_argument("--kfolds", type=int, default=5, help="Número de folds para CV")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-random", type=int, default=12000, help="Evaluaciones en la búsqueda aleatoria")
    p.add_argument("--refine-grid", type=int, default=11, help="Puntos por dimensión en refino local")
    p.add_argument("--no-fit-alpha", action="store_true", help="No ajustar alpha aunque no haya x_obs")
    p.add_argument("--emerrel-in-percent", choices=["auto","true","false"], default="auto",
                   help="Si EMERREL está en %, forzar conversión. 'auto' intenta detectar por ensayo.")
    args = p.parse_args()

    cfg = CalibConfig(
        a2_cap=args.a2,
        use_emerac=args.use_emerac,
        k_folds=max(2, args.kfolds),
        seed=args.seed,
        n_random=max(1000, args.n_random),
        refine_grid=max(7, args.refine_grid),
        fit_alpha_when_no_x=(not args.no_fit_alpha),
        emerrel_in_percent=(None if args.emerrel_in_percent=="auto" else (args.emerrel_in_percent=="true")),
    )

    df = build_dataset(args.excel, cfg)
    summary, cv_df, preds_df = fit_and_eval(df, cfg)

    base = os.path.splitext(args.excel)[0]
    out_json = base + "_params.json"
    out_csv  = base + "_params.csv"
    cv_csv   = base + "_cv.csv"
    preds_csv= base + "_preds.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    cv_df.to_csv(cv_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)

    print("Guardado:", out_json)
    print("Guardado:", out_csv)
    print("Guardado:", cv_csv)
    print("Guardado:", preds_csv)
    print("Resumen:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()



