# -*- coding: utf-8 -*-
# Calibración PREDWEEM con datos independientes
# Objetivo: minimizar RMSE de % pérdida de rinde ajustando (alpha, Lmax)
# Uso:
#   python calibra_predweem.py --calib_csv calibracion.csv --iters 5000 --seed 123
#
# Notas:
# - No requiere Streamlit ni SciPy. Optimización por búsqueda aleatoria + refinamiento local.
# - La serie EMERREL puede venir como % o 0–1; y como diaria o acumulada (EMERAC).
# - Cada ensayo puede definir sus propios parámetros de canopia/Ciec y manejo.

import argparse, io, json, math, random, sys
from datetime import timedelta, date
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

# ------------------ Utilidades I/O ------------------
def read_url_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r:
        return r.read()

def sniff_sep_dec(sample_text: str):
    sample = sample_text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and any(ch.isdigit() for ch in sample.split(",")[-1][:1]):
        dec_guess = ","
    return sep_guess, dec_guess

def parse_csv_bytes(raw: bytes, sep_opt="auto", dec_opt="auto"):
    # probar utf-8, si falla latin-1
    try:
        head = raw[:8000].decode("utf-8", errors="strict"); enc = "utf-8"
    except UnicodeDecodeError:
        head = raw[:8000].decode("latin-1", errors="ignore"); enc = "latin-1"
    sep_guess, dec_guess = sniff_sep_dec(head)
    sep = sep_guess if sep_opt == "auto" else sep_opt
    dec = dec_guess if dec_opt == "auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python", encoding=enc)
    return df

def to_days(ts: pd.Series) -> np.ndarray:
    f = pd.to_datetime(ts).to_numpy(dtype="datetime64[ns]")
    t_ns = f.astype("int64")
    return ((t_ns - t_ns[0]) / 1e9 / 86400.0).astype(float)

def auc_time(fechas: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fechas); y_arr = np.asarray(y, float)
    if mask is not None:
        f = f[mask]; y_arr = y_arr[mask]
    if len(f) < 2:
        return 0.0
    tdays = to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, tdays))

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%","",regex=False)
    if decimal == ",":
        t = t.str.replace(".","",regex=False).str.replace(",",".",regex=False)
    else:
        t = t.str.replace(",","",regex=False)
    return pd.to_numeric(t, errors="coerce")

# ------------------ Modelo núcleo (PREDWEEM reducido) ------------------
def compute_canopy(fechas: pd.Series, sow_date: date,
                   t_lag=7, t_close=45, cov_max=0.85, lai_max=3.5, k_beer=0.6,
                   mode_canopy="FC"):
    days_since_sow = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas], dtype=float)

    def logistic_between(days, start, end, y_max):
        if end <= start: end = start + 1
        t_mid = 0.5*(start+end); r = 4.0/max(1.0,(end-start))
        return y_max/(1.0+np.exp(-r*(days-t_mid)))

    if mode_canopy == "FC":
        fc_dyn = np.where(days_since_sow < t_lag, 0.0,
                          logistic_between(days_since_sow, t_lag, t_close, cov_max))
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0-fc_dyn,1e-9,1.0))/max(1e-6,k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(days_since_sow < t_lag, 0.0,
                       logistic_between(days_since_sow, t_lag, t_close, lai_max))
        LAI = np.clip(LAI, 0.0, lai_max)
        fc_dyn = 1 - np.exp(-k_beer*LAI)
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
    return fc_dyn, LAI

def cap_cumulative(series, cap, active_mask):
    y = np.asarray(series, dtype=float)
    out = np.zeros_like(y); cum = 0.0
    for i in range(len(y)):
        if bool(active_mask[i]):
            allowed = max(0.0, cap - cum)
            val = min(max(0.0, y[i]), allowed)
            out[i] = val; cum += val
        else:
            out[i] = 0.0
    return out

def predict_x_and_loss(emer_df: pd.DataFrame,
                       sow_date: date,
                       as_percent=True,
                       is_cumulative=False,
                       dayfirst=True,
                       MAX_PLANTS_CAP=250.0,
                       t_lag=7, t_close=45, cov_max=85.0, lai_max=3.5, k_beer=0.6,
                       use_ciec=True, Ca=250.0, Cs=250.0, LAIhc=3.5,
                       # Manejo (fechas opcionales + eficiencias y días)
                       pre_glifo_date=None, pre_glifo_eff=90,
                       pre_selNR_date=None, pre_selNR_eff=60, NR_DAYS_DEFAULT=10,
                       preR_date=None, preR_days=None, preR_eff=70,
                       preemR_date=None, preemR_days=None, preemR_eff=70,
                       post_gram_date=None, post_gram_eff=65, POST_GRAM_FORWARD_DAYS=11,
                       post_selR_date=None, post_selR_days=None, post_selR_eff=70,
                       # Pérdida (parámetros a calibrar)
                       alpha=0.375, Lmax=76.639):
    # --- Parse serie EMERREL ---
    df = emer_df.copy()
    if df.shape[1] < 2:
        raise ValueError("La serie EMERREL debe tener al menos 2 columnas (fecha, valor).")
    c_fecha, c_valor = df.columns[:2]
    fechas = pd.to_datetime(df[c_fecha], dayfirst=bool(dayfirst), errors="coerce")
    sample_str = df[c_valor].astype(str).head(200).str.cat(sep=" ")
    dec_for_col = "," if (sample_str.count(",")>sample_str.count(".") and "," in sample_str) else "."
    vals = clean_numeric_series(df[c_valor], decimal=dec_for_col)
    df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
    if df.empty:
        raise ValueError("Serie EMERREL vacía tras limpieza.")
    emerrel = df["valor"].astype(float)
    if as_percent: emerrel = emerrel / 100.0
    if is_cumulative: emerrel = emerrel.diff().fillna(0.0).clip(lower=0.0)
    emerrel = emerrel.clip(lower=0.0)
    ts = pd.to_datetime(df["fecha"])
    mask_since = (ts.dt.date >= sow_date)

    # --- Canopia y Ciec ---
    FC, LAI = compute_canopy(ts, sow_date,
                             int(t_lag), int(t_close),
                             float(cov_max)/100.0, float(lai_max), float(k_beer),
                             mode_canopy="FC")
    if use_ciec:
        Ca_safe = Ca if Ca > 0 else 1e-6
        Cs_safe = Cs if Cs > 0 else 1e-6
        Ciec = np.clip((LAI / max(1e-6, LAIhc)) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec = np.zeros_like(LAI, dtype=float)
    one_minus_Ciec = np.clip(1.0 - Ciec, 0.0, 1.0)

    # --- Cohortes (S1..S4) sobre nacimientos desde siembra ---
    births = np.where(mask_since.to_numpy(), emerrel.to_numpy(float), 0.0)
    s = pd.Series(births, index=ts)
    S1 = s.rolling(6, min_periods=0).sum().shift(1).fillna(0.0).reindex(ts).to_numpy(float)
    S2 = s.rolling(21, min_periods=0).sum().shift(7).fillna(0.0).reindex(ts).to_numpy(float)
    S3 = s.rolling(32, min_periods=0).sum().shift(28).fillna(0.0).reindex(ts).to_numpy(float)
    S4 = s.cumsum().shift(60).fillna(0.0).reindex(ts).to_numpy(float)
    FC_S = {"S1":0.0, "S2":0.3, "S3":0.6, "S4":1.0}

    # --- Equivalencia por área / AUC (desde siembra) ---
    auc_cruda = auc_time(ts, emerrel.to_numpy(float), mask=mask_since)
    if not np.isfinite(auc_cruda) or auc_cruda <= 0:
        return np.nan, np.nan
    factor_area = MAX_PLANTS_CAP / auc_cruda

    S1_pl = np.where(mask_since, S1*one_minus_Ciec*FC_S["S1"]*factor_area, 0.0)
    S2_pl = np.where(mask_since, S2*one_minus_Ciec*FC_S["S2"]*factor_area, 0.0)
    S3_pl = np.where(mask_since, S3*one_minus_Ciec*FC_S["S3"]*factor_area, 0.0)
    S4_pl = np.where(mask_since, S4*one_minus_Ciec*FC_S["S4"]*factor_area, 0.0)
    base_pl_daily = np.where(mask_since, emerrel.to_numpy(float)*factor_area, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since.to_numpy())
    sup_cap = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_pl_daily_cap)

    # --- Controles (ventanas discretas con eficiencias) ---
    fechas_d = ts.dt.date.values
    c1 = np.ones_like(fechas_d, float)
    c2 = np.ones_like(fechas_d, float)
    c3 = np.ones_like(fechas_d, float)
    c4 = np.ones_like(fechas_d, float)

    def apply(weights, eff, states):
        if eff is None: eff = 0
        eff = float(eff)
        if eff <= 0 or not states: return
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(weights,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)

    def win_one_day(d):
        if d is None or pd.isna(d): return np.zeros_like(fechas_d, float)
        d0 = pd.to_datetime(d).date()
        return ((fechas_d >= d0) & (fechas_d < (d0 + timedelta(days=1)))).astype(float)

    def win_residual(d, days):
        w = np.zeros_like(fechas_d, float)
        if d is None or pd.isna(d) or days is None or days <= 0: return w
        d0 = pd.to_datetime(d).date(); d1 = d0 + timedelta(days=int(days))
        return ((fechas_d >= d0) & (fechas_d < d1)).astype(float)

    # pre glifo (1d) S1..S4
    if pre_glifo_date:
        apply(win_one_day(pre_glifo_date), pre_glifo_eff, ["S1","S2","S3","S4"])
    # pre selectivo NR (10d) S1..S4
    if pre_selNR_date:
        apply(win_residual(pre_selNR_date, NR_DAYS_DEFAULT), pre_selNR_eff, ["S1","S2","S3","S4"])
    # presiembra residual S1..S2
    if preR_date and preR_days:
        apply(win_residual(preR_date, preR_days), preR_eff, ["S1","S2"])
    # preemergente residual S1..S2
    if preemR_date and preemR_days:
        apply(win_residual(preemR_date, preemR_days), preemR_eff, ["S1","S2"])
    # post graminicida S1..S3
    if post_gram_date:
        apply(win_residual(post_gram_date, POST_GRAM_FORWARD_DAYS), post_gram_eff, ["S1","S2","S3"])
    # post selectivo residual S1..S4
    if post_selR_date and post_selR_days:
        apply(win_residual(post_selR_date, post_selR_days), post_selR_eff, ["S1","S2","S3","S4"])

    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)
    x = float(np.nansum(plantas_ctrl_cap[mask_since.to_numpy()]))

    # pérdida (%)
    loss_pred = float(alpha * x / (1.0 + (alpha * x / max(1e-9, Lmax))))
    return x, loss_pred

# ------------------ Calibración ------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return np.inf
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def objective_alpha_Lmax(params, trials):
    alpha, Lmax = params
    # límites suaves
    if not (0.0 < alpha <= 3.0 and 5.0 <= Lmax <= 100.0):
        return np.inf
    preds = []
    obs = []
    for tr in trials:
        x, lp = predict_x_and_loss(**tr, alpha=float(alpha), Lmax=float(Lmax))
        preds.append(lp)
        obs.append(tr["observed_loss_pct"])
    return rmse(obs, preds)

def random_search(trials, iters=5000, seed=123, alpha_range=(0.05, 1.5), Lmax_range=(20.0, 90.0)):
    random.seed(seed); best = {"alpha": None, "Lmax": None, "rmse": np.inf}
    for i in range(1, int(iters)+1):
        a = random.uniform(*alpha_range)
        L = random.uniform(*Lmax_range)
        val = objective_alpha_Lmax((a, L), trials)
        if val < best["rmse"]:
            best.update({"alpha": a, "Lmax": L, "rmse": val})
    return best

def local_refine(trials, alpha, Lmax, steps=50, scale_alpha=0.2, scale_L=0.2):
    best = {"alpha": alpha, "Lmax": Lmax, "rmse": objective_alpha_Lmax((alpha, Lmax), trials)}
    for _ in range(int(steps)):
        a_try = max(1e-4, np.random.normal(best["alpha"], scale_alpha*max(1e-3,best["alpha"])))
        L_try = max(5.0,  np.random.normal(best["Lmax"],  scale_L*max(1e-3,best["Lmax"])))
        val = objective_alpha_Lmax((a_try, L_try), trials)
        if val < best["rmse"]:
            best = {"alpha": a_try, "Lmax": L_try, "rmse": val}
    return best

# ------------------ Main ------------------
def build_trial_row(row):
    # cargar EMERREL
    raw = read_url_bytes(str(row["emerrel_url"]))
    df_em = parse_csv_bytes(raw)
    # flags
    def get_bool(name, default=False):
        v = row.get(name, default)
        if isinstance(v, str):
            return v.strip().lower() in ("1","true","t","yes","y","si","sí")
        return bool(v)
    # util parse fecha
    def maybe_date(v):
        return None if pd.isna(v) or v=="" else pd.to_datetime(v).date()

    trial = {
        "emer_df": df_em,
        "sow_date": pd.to_datetime(row["sow_date"]).date(),
        "as_percent": get_bool("as_percent", True),
        "is_cumulative": get_bool("is_cumulative", False),
        "dayfirst": get_bool("dayfirst", True),
        "MAX_PLANTS_CAP": float(row.get("MAX_PLANTS_CAP", 250.0)),
        "t_lag": int(row.get("t_lag", 7)),
        "t_close": int(row.get("t_close", 45)),
        "cov_max": float(row.get("cov_max", 85.0)),
        "lai_max": float(row.get("lai_max", 3.5)),
        "k_beer": float(row.get("k_beer", 0.6)),
        "use_ciec": get_bool("use_ciec", True),
        "Ca": float(row.get("Ca", 250.0)),
        "Cs": float(row.get("Cs", 250.0)),
        "LAIhc": float(row.get("LAIhc", 3.5)),
        # manejo
        "pre_glifo_date": maybe_date(row.get("pre_glifo_date")),
        "pre_glifo_eff": float(row.get("pre_glifo_eff", 90)),
        "pre_selNR_date": maybe_date(row.get("pre_selNR_date")),
        "pre_selNR_eff": float(row.get("pre_selNR_eff", 60)),
        "preR_date": maybe_date(row.get("preR_date")),
        "preR_days": None if pd.isna(row.get("preR_days")) else int(row.get("preR_days")),
        "preR_eff": float(row.get("preR_eff", 70)),
        "preemR_date": maybe_date(row.get("preemR_date")),
        "preemR_days": None if pd.isna(row.get("preemR_days")) else int(row.get("preemR_days")),
        "preemR_eff": float(row.get("preemR_eff", 70)),
        "post_gram_date": maybe_date(row.get("post_gram_date")),
        "post_gram_eff": float(row.get("post_gram_eff", 65)),
        "post_selR_date": maybe_date(row.get("post_selR_date")),
        "post_selR_days": None if pd.isna(row.get("post_selR_days")) else int(row.get("post_selR_days")),
        "post_selR_eff": float(row.get("post_selR_eff", 70)),
        # observado
        "observed_loss_pct": float(row["observed_loss_pct"]),
    }
    return trial

def main():
    ap = argparse.ArgumentParser(description="Calibración PREDWEEM (objetivo: % pérdida de rinde)")
    ap.add_argument("--calib_csv", required=True, help="CSV con ensayos (ver plantilla en la explicación)")
    ap.add_argument("--iters", type=int, default=5000, help="Iteraciones de búsqueda aleatoria (default: 5000)")
    ap.add_argument("--refine", type=int, default=200, help="Pasos de refinamiento local (default: 200)")
    ap.add_argument("--seed", type=int, default=123, help="Semilla aleatoria")
    ap.add_argument("--alpha_min", type=float, default=0.05)
    ap.add_argument("--alpha_max", type=float, default=1.5)
    ap.add_argument("--Lmin", type=float, default=20.0)
    ap.add_argument("--Lmax", type=float, default=90.0)
    args = ap.parse_args()

    calib = pd.read_csv(args.calib_csv)
    trials = []
    print(f"Leyendo {len(calib)} ensayos…")
    for i, row in calib.iterrows():
        try:
            tr = build_trial_row(row)
            trials.append(tr)
        except Exception as e:
            print(f"[WARN] Ensayo idx={i} omitido por error: {e}", file=sys.stderr)

    if not len(trials):
        print("No hay ensayos válidos para calibrar.", file=sys.stderr)
        sys.exit(1)

    print("Optimizando (α, Lmax)…")
    best = random_search(
        trials,
        iters=args.iters,
        seed=args.seed,
        alpha_range=(args.alpha_min, args.alpha_max),
        Lmax_range=(args.Lmin, args.Lmax),
    )
    best = local_refine(trials, best["alpha"], best["Lmax"], steps=args.refine)

    # Predicciones por ensayo
    rows = []
    for idx, tr in enumerate(trials):
        x, lp = predict_x_and_loss(**tr, alpha=best["alpha"], Lmax=best["Lmax"])
        rows.append({
            "trial_idx": idx,
            "sow_date": tr["sow_date"],
            "observed_loss_pct": tr["observed_loss_pct"],
            "x_effective_pl_m2": x,
            "pred_loss_pct": lp,
            "abs_error": abs(lp - tr["observed_loss_pct"]),
        })
    res_df = pd.DataFrame(rows)
    overall_rmse = rmse(res_df["observed_loss_pct"], res_df["pred_loss_pct"])

    print("\n===== Resultados de calibración =====")
    print(f"alpha = {best['alpha']:.6f}")
    print(f"Lmax  = {best['Lmax']:.6f} (%)")
    print(f"RMSE  = {overall_rmse:.4f} puntos porcentuales")
    out_csv = "calibration_results.csv"
    res_df.to_csv(out_csv, index=False)
    with open("calibrated_params.json","w",encoding="utf-8") as f:
        json.dump({"alpha": best["alpha"], "Lmax": best["Lmax"], "rmse": overall_rmse}, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado:\n- {out_csv}\n- calibrated_params.json")

if __name__ == "__main__":
    main()







