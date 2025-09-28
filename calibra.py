# -*- coding: utf-8 -*-
"""
calibra_estados_normalizada.py

Calibración PREDWEEM usando densidad efectiva x (S1–S4) SIN SOLAPES,
NORMALIZADA por duración de estado, con función de pérdida HIPERBÓLICA.

- Lee hojas:  ensayos(ensayo_id, loss_obs_pct, MAX_PLANTS_CAP, fecha_siembra, pc_ini, pc_fin[, Ca, Cs])
              emergencia(ensayo_id, fecha, emer_rel)  # emer_rel en 0..1 o % (auto-conversión)
- x(t) = sum( S_k(t) * w_k / L_k ), con L = (7, 21, 32, T4), T4 = 60 fijo o "dinámico en PC"
- Ajuste: Loss(x) = (alpha * x) / (1 + (alpha * x / Lmax))
- Exporta: CSV + Excel (resultados + resumen), JSON de parámetros, y PNG Obs vs Pred.

Uso:
  python calibra_estados_normalizada.py --excel "calibra borde.xlsx" --outdir "./out"
Opciones clave:
  --t4 dynamic | fixed     (default: dynamic)
  --use-ciec               (aplica (1-Ciec) antes de estados)
  --w-s1 0.0 --w-s2 0.3 --w-s3 0.6 --w-s4 1.0
  --alpha0 0.05 --lmax0 80
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----------------- utilidades -----------------

def daily_series(sub_emerg: pd.DataFrame, start, end) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    s = sub_emerg.set_index("fecha")["emer_rel"].reindex(idx).interpolate("linear").fillna(0.0)
    s.index.name = "fecha"
    return s

def auc(idx: pd.DatetimeIndex, values: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    t = idx.values.astype("datetime64[ns]").astype("int64")/86400e9
    return float(np.trapz(np.asarray(values, float), t))

def compute_canopy(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                   mode_canopy: str, t_lag: int, t_close: int,
                   cov_max: float, lai_max: float, k_beer: float):
    days = (idx.date - sow_date.date()).astype("timedelta64[D]").astype(int).astype(float)

    def logistic_between(days_, start, end, y_max):
        end = start + 1 if end <= start else end
        t_mid = 0.5*(start+end); r = 4.0 / max(1.0, (end-start))
        return y_max / (1.0 + np.exp(-r*(days_ - t_mid)))

    if mode_canopy == "coverage":
        fc = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, cov_max))
        fc = np.clip(fc, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0-fc,1e-9,1.0))/max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days < t_lag, 0.0, logistic_between(days, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc = 1.0 - np.exp(-k_beer * LAI)
        fc = np.clip(fc, 0.0, 1.0)
    return fc, LAI

def ciec_series(idx: pd.DatetimeIndex, sow_date: pd.Timestamp,
                mode_canopy="coverage", t_lag=7, t_close=45,
                cov_max=0.85, lai_max=3.5, k_beer=0.6,
                Ca=250, Cs=250, LAIhc=3.5):
    _, LAI = compute_canopy(idx, sow_date, mode_canopy, t_lag, t_close, cov_max, lai_max, k_beer)
    Ca_s = Ca if Ca > 0 else 1e-6
    Cs_s = Cs if Cs > 0 else 1e-6
    Ciec = (LAI / max(1e-6, LAIhc)) * (Ca_s / Cs_s)
    return np.clip(Ciec, 0.0, 1.0)

def non_overlapping_states(emer_daily: pd.Series) -> pd.DataFrame:
    """Estados por edad (no solapados): S1 0–6, S2 7–27, S3 28–59, S4 ≥60 (pl·m²/día)."""
    v = emer_daily.to_numpy(float)
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])

    def sum_age_window(i, a, b):
        lo, hi = i - b, i - a
        if hi < 0:
            return 0.0
        lo_c, hi_c = max(0, lo), min(n-1, hi)
        if lo_c > hi_c:
            return 0.0
        return c[hi_c+1] - c[lo_c]

    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    for i in range(n):
        S1[i] = sum_age_window(i, 0, 6)
        S2[i] = sum_age_window(i, 7, 27)
        S3[i] = sum_age_window(i, 28, 59)
        j = i - 60
        S4[i] = c[j+1] if j >= 0 else 0.0
    return pd.DataFrame({"S1": S1, "S2": S2, "S3": S3, "S4": S4}, index=emer_daily.index)

def effective_T4_days(sow_date: pd.Timestamp, pc_ini: pd.Timestamp, pc_fin: pd.Timestamp) -> int:
    """Días efectivos de S4 dentro del PC (≥60 días desde siembra). Mínimo 1."""
    start_S4 = max(pc_ini, sow_date + pd.Timedelta(days=60))
    if start_S4 > pc_fin:
        return 1
    return max(1, (pc_fin - start_S4).days + 1)

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
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
    r2 = (1 - ss_res/ss_tot) if (ss_tot and ss_tot != 0) else float("nan")
    return {"alpha": a, "Lmax": L, "RMSE": rmse, "MAE": mae, "R2": r2, "yhat": yhat}

def plot_obs_vs_pred(y_obs, y_pred, outpath):
    plt.figure()
    plt.scatter(y_obs, y_pred)
    mx = max(1.0, float(np.max(y_obs)), float(np.max(y_pred)))
    plt.plot([0, mx], [0, mx], "k--")
    plt.xlabel("Observado (%)")
    plt.ylabel("Predicho (%)")
    plt.title("Observado vs Predicho")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ----------------- core -----------------

def compute_x_for_row(row, emer_df: pd.DataFrame, w_states: dict, t4_mode: str,
                      use_ciec: bool, canopy_cfg: dict):
    ens_id = row["ensayo_id"]
    f_sow  = pd.to_datetime(row["fecha_siembra"])
    pc_ini = pd.to_datetime(row["pc_ini"])
    pc_fin = pd.to_datetime(row["pc_fin"])
    cap    = float(row["MAX_PLANTS_CAP"])
    Ca     = int(row.get("Ca", 250) if not pd.isna(row.get("Ca", np.nan)) else 250)
    Cs     = int(row.get("Cs", 250) if not pd.isna(row.get("Cs", np.nan)) else 250)

    sub = emer_df[emer_df["ensayo_id"]==ens_id][["fecha","emer_rel"]].dropna().copy()
    if sub.empty or pd.isna(f_sow) or pd.isna(pc_fin):
        return 0.0

    # % -> 0..1 si hace falta
    if sub["emer_rel"].max() > 1.5:
        sub["emer_rel"] = sub["emer_rel"]/100.0

    s = daily_series(sub, f_sow, pc_fin)   # EMERREL diaria (0..1)
    auc_total = auc(s.index, s.values)
    if auc_total <= 0:
        return 0.0

    factor = cap / auc_total               # a pl·m²/día

    if use_ciec:
        Ciec = ciec_series(
            s.index, f_sow,
            mode_canopy=canopy_cfg["mode"],
            t_lag=int(canopy_cfg["t_lag"]),
            t_close=int(canopy_cfg["t_close"]),
            cov_max=float(canopy_cfg["cov_max"]),
            lai_max=float(canopy_cfg["lai_max"]),
            k_beer=float(canopy_cfg["k_beer"]),
            Ca=Ca, Cs=Cs, LAIhc=float(canopy_cfg["LAIhc"])
        )
        daily_eff = s.values * (1.0 - Ciec) * factor
    else:
        daily_eff = s.values * factor

    # Estados (pl·m²/día)
    states = non_overlapping_states(pd.Series(daily_eff, index=s.index))

    # Normalización por duración
    L1, L2, L3 = 7, 21, 32
    if t4_mode == "fixed":
        T4 = 60
    else:
        T4 = effective_T4_days(f_sow, pc_ini, pc_fin)

    D = (
        states["S1"] * w_states["S1"] / L1 +
        states["S2"] * w_states["S2"] / L2 +
        states["S3"] * w_states["S3"] / L3 +
        states["S4"] * w_states["S4"] / max(1, T4)
    ).to_numpy()

    mask_pc = (s.index >= pc_ini) & (s.index <= pc_fin)
    x_val = auc(s.index[mask_pc], D[mask_pc])  # ≈ pl·m²
    return float(x_val)

def main():
    ap = argparse.ArgumentParser(description="Calibración con x (S1–S4) normalizada por duración + hiperbólica.")
    ap.add_argument("--excel", required=True, help="Ruta al Excel (ensayos, emergencia)")
    ap.add_argument("--outdir", default="./out", help="Carpeta de salida")
    ap.add_argument("--t4", choices=["dynamic","fixed"], default="dynamic", help="T4=duración S4 (default: dynamic)")
    ap.add_argument("--use-ciec", action="store_true", help="Aplicar (1−Ciec) previo a estados")
    ap.add_argument("--w-s1", type=float, default=0.0)
    ap.add_argument("--w-s2", type=float, default=0.3)
    ap.add_argument("--w-s3", type=float, default=0.6)
    ap.add_argument("--w-s4", type=float, default=1.0)
    ap.add_argument("--alpha0", type=float, default=0.05)
    ap.add_argument("--lmax0",  type=float, default=80.0)
    # Canopy
    ap.add_argument("--canopy-mode", choices=["coverage","lai"], default="coverage")
    ap.add_argument("--t-lag",   type=int,   default=7)
    ap.add_argument("--t-close", type=int,   default=45)
    ap.add_argument("--cov-max", type=float, default=0.85)
    ap.add_argument("--lai-max", type=float, default=3.5)
    ap.add_argument("--k-beer",  type=float, default=0.6)
    ap.add_argument("--LAIhc",   type=float, default=3.5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Leer Excel
    ens = pd.read_excel(args.excel, sheet_name="ensayos")
    emer = pd.read_excel(args.excel, sheet_name="emergencia")
    for c in ["fecha_siembra","pc_ini","pc_fin"]:
        ens[c] = pd.to_datetime(ens[c], errors="coerce")
    emer["fecha"] = pd.to_datetime(emer["fecha"], errors="coerce")

    need_ens = {"ensayo_id","loss_obs_pct","MAX_PLANTS_CAP","fecha_siembra","pc_ini","pc_fin"}
    need_em  = {"ensayo_id","fecha","emer_rel"}
    if not need_ens.issubset(ens.columns) or not need_em.issubset(emer.columns):
        raise ValueError("Columnas requeridas ausentes. ensayos: ensayo_id, loss_obs_pct, MAX_PLANTS_CAP, fecha_siembra, pc_ini, pc_fin; "
                         "emergencia: ensayo_id, fecha, emer_rel")

    # Config básica
    w_states = {"S1": args.w_s1, "S2": args.w_s2, "S3": args.w_s3, "S4": args.w_s4}
    canopy_cfg = dict(
        mode=args.canopy_mode, t_lag=args.t_lag, t_close=args.t_close,
        cov_max=args.cov_max, lai_max=args.lai_max, k_beer=args.k_beer, LAIhc=args.LAIhc
    )

    # Calcular x por ensayo
    xs = []
    for _, row in ens.iterrows():
        xs.append(compute_x_for_row(
            row, emer_df=emer, w_states=w_states, t4_mode=args.t4,
            use_ciec=args.use_ciec, canopy_cfg=canopy_cfg
        ))
    ens = ens.copy()
    ens["x_pl_m2_states"] = np.array(xs, dtype=float)

    # Calibrar hiperbólica
    y_obs = ens["loss_obs_pct"].to_numpy(float)
    x_val = ens["x_pl_m2_states"].to_numpy(float)

    fit = calibrate_hyperbolic(x_val, y_obs, alpha0=args.alpha0, Lmax0=args.lmax0)
    a_b, L_b = fit["alpha"], fit["Lmax"]
    y_hat    = fit["yhat"]

    # Tabla final
    out = ens[["ensayo_id","loss_obs_pct","x_pl_m2_states","MAX_PLANTS_CAP"]].copy()
    out["predicho"] = y_hat
    out["alpha"]    = float(a_b)
    out["Lmax"]     = float(L_b)
    out["RMSE"]     = float(fit["RMSE"])
    out["MAE"]      = float(fit["MAE"])
    out["R2"]       = float(fit["R2"])
    out["T4_mode"]  = args.t4
    out["use_ciec"] = bool(args.use_ciec)

    # Exports
    csv_path  = os.path.join(args.outdir, "calibracion_estados_normalizada_hiperbolica.csv")
    xlsx_path = os.path.join(args.outdir, "calibracion_estados_normalizada_hiperbolica.xlsx")
    json_path = os.path.join(args.outdir, "predweem_config_estados_normalizada.json")
    png_path  = os.path.join(args.outdir, "obs_vs_pred_estados_normalizada_hiperbolica.png")

    out.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as wr:
        out.to_excel(wr, sheet_name="resultados", index=False)
        pd.DataFrame({
            "alpha":[a_b], "Lmax":[L_b],
            "RMSE":[fit["RMSE"]], "MAE":[fit["MAE"]], "R2":[fit["R2"]],
            "T4_mode":[args.t4], "use_ciec":[args.use_ciec],
            "w_S1":[args.w_s1], "w_S2":[args.w_s2], "w_S3":[args.w_s3], "w_S4":[args.w_s4],
            "canopy_mode":[args.canopy_mode], "t_lag":[args.t_lag], "t_close":[args.t_close],
            "cov_max":[args.cov_max], "lai_max":[args.lai_max], "k_beer":[args.k_beer], "LAIhc":[args.LAIhc]
        }).to_excel(wr, sheet_name="resumen", index=False)

    # JSON de parámetros para la app
    params = {
        "loss_kind": "hyperbolic",
        "alpha": float(a_b), "Lmax": float(L_b),
        "use_ciec": bool(args.use_ciec),
        "state_weights": {k: float(v) for k, v in w_states.items()},
        "state_durations": {"S1":7, "S2":21, "S3":32, "S4": (60 if args.t4=="fixed" else "effective_in_PC")},
        "t4_mode": args.t4,
        "canopy": canopy_cfg
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # Gráfico Obs vs Pred
    plot_obs_vs_pred(y_obs, y_hat, png_path)

    # Log básico
    print(f"alpha={a_b:.6f}  Lmax={L_b:.3f}  RMSE={fit['RMSE']:.2f}  MAE={fit['MAE']:.2f}  R2={fit['R2']:.3f}")
    print("Guardado:")
    print(" ", csv_path)
    print(" ", xlsx_path)
    print(" ", json_path)
    print(" ", png_path)

if __name__ == "__main__":
    main()

