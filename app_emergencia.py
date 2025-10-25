        # ===========================================================
        # 🔍 Descomposición de pérdida: dentro y fuera del PCC
        # ===========================================================
        st.subheader("📉 Descomposición de pérdida (PCC vs fuera del PCC)")

        AUC_in  = float(envb["AUC_in"]) if "AUC_in" in envb else 0
        AUC_out = float(envb["AUC_out"]) if "AUC_out" in envb else 0
        AUC_tot = AUC_in + AUC_out if (AUC_in + AUC_out) > 0 else 1.0
        loss_total = best["loss_pct"]
        prop_in, prop_out = AUC_in/AUC_tot, AUC_out/AUC_tot
        loss_in, loss_out = loss_total*prop_in, loss_total*prop_out

        df_loss = pd.DataFrame({
            "Componente": ["Dentro PCC","Fuera PCC","Total"],
            "AUC ponderado": [AUC_in,AUC_out,AUC_tot],
            "Proporción (%)": [prop_in*100,prop_out*100,100],
            "Pérdida (%)": [loss_in,loss_out,loss_total]
        })
        st.dataframe(df_loss.style.format({
            "AUC ponderado":"{:.2f}","Proporción (%)":"{:.1f}","Pérdida (%)":"{:.2f}"
        }), use_container_width=True)

        fig_loss_pcc = go.Figure()
        fig_loss_pcc.add_trace(go.Bar(
            x=["Dentro PCC","Fuera PCC"], y=[loss_in,loss_out],
            name="Pérdida (%)", marker_color=["gold","lightblue"]
        ))
        fig_loss_pcc.update_layout(title="Contribución relativa del PCC a la pérdida total",
                                   yaxis_title="Pérdida (%)", xaxis_title="Componente")
        st.plotly_chart(fig_loss_pcc, use_container_width=True)

        st.markdown(
            f"💡 **Interpretación:** del total de pérdida estimada (**{loss_total:.2f}%**), "
            f"aprox. **{loss_in:.2f}% ({prop_in*100:.1f}%)** ocurrió **dentro del PCC**, "
            f"y **{loss_out:.2f}% ({prop_out*100:.1f}%)** fuera de él."
        )



# =====================================================
#                OPTIMIZACIÓN
# =====================================================
st.markdown("---")
st.header("🧠 Optimización")

with st.sidebar:
    st.header("Optimización (variables habilitadas)")
    sow_search_from = st.date_input("Buscar siembra desde", value=sow_min, min_value=sow_min, max_value=sow_max, key="sow_from")
    sow_search_to   = st.date_input("Buscar siembra hasta",  value=sow_max, min_value=sow_min, max_value=sow_max, key="sow_to")
    sow_step_days   = st.number_input("Paso de siembra (días)", 1, 30, 7, 1)

    use_preR_opt      = st.checkbox("Incluir presiembra + residual (≤ siembra−14; S1–S2)", value=True)
    use_preemR_opt    = st.checkbox("Incluir preemergente + residual (siembra..siembra+10; S1–S2)", value=True)
    use_post_selR_opt = st.checkbox("Incluir post + residual (≥ siembra + 20; S1–S4)", value=True)
    use_post_gram_opt = st.checkbox(f"Incluir graminicida post (+{POST_GRAM_FORWARD_DAYS-1}d; S1–S3)", value=True)

    ef_preR_opt      = st.slider("Eficiencia presiembraR (%)", 0, 100, 70, 1)   if use_preR_opt else 0
    ef_preemR_opt    = st.slider("Eficiencia preemergenteR (%)", 0, 100, 70, 1) if use_preemR_opt else 0
    ef_post_selR_opt = st.slider("Eficiencia post residual (%)", 0, 100, 70, 1) if use_post_selR_opt else 0
    ef_post_gram_opt = st.slider("Eficiencia graminicida post (%)", 0, 100, 65, 1) if use_post_gram_opt else 0

    preR_min_back   = st.number_input("PresiembraR: buscar hasta X días antes de siembra", 14, 120, 45, 1)
    preR_step_days  = st.number_input("Paso fechas PRESIEMBRA (días)", 1, 30, 7, 1)
    preem_step_days = st.number_input("Paso fechas PREEMERGENTE (días)", 1, 10, 3, 1)

    post_days_fw    = st.number_input("Post: días después de siembra (máximo)", 20, 180, 60, 1)
    post_step_days  = st.number_input("Paso fechas POST (días)", 1, 30, 7, 1)

    res_min, res_max = st.slider("Residualidad (min–max) [días]", min_value=15, max_value=120, value=(30, 60), step=5)
    res_step = st.number_input("Paso de residualidad (días)", min_value=1, max_value=30, value=5, step=1)

    optimizer  = st.selectbox("Optimizador", ["Grid (combinatorio)", "Búsqueda aleatoria", "Recocido simulado"], index=0)
    max_evals  = st.number_input("Máx. evaluaciones", 100, 100000, 4000, 100)
    top_k_show = st.number_input("Top-k a mostrar", 1, 20, 5, 1)

    if optimizer == "Recocido simulado":
        sa_iters   = st.number_input("Iteraciones (SA)", 100, 50000, 5000, 100)
        sa_T0      = st.number_input("Temperatura inicial", 0.01, 50.0, 5.0, 0.1)
        sa_cooling = st.number_input("Factor de enfriamiento (γ)", 0.80, 0.9999, 0.995, 0.0001)

    st.subheader("Periodo Crítico de Competencia (PCC)")
    PCC_ON = st.checkbox("Activar sensibilidad PCC", value=True)
    _pcc_ini_default = sow_date + timedelta(days=20)
    _pcc_fin_default = sow_date + timedelta(days=45)
    PCC_INI = st.date_input("Inicio PCC", value=_pcc_ini_default, min_value=sow_min, max_value=sow_max + timedelta(days=120))
    PCC_FIN = st.date_input("Fin PCC",    value=_pcc_fin_default, min_value=PCC_INI, max_value=sow_max + timedelta(days=160))
    SENS_FACTOR = st.number_input("Factor de sensibilidad PCC (×)", 0.10, 5.00, 1.25, 0.05)

    # Exponer para otros bloques (back-compat)
    globals()["PCC_INI"] = PCC_INI
    globals()["PCC_FIN"] = PCC_FIN
    globals()["SENS_FACTOR"] = SENS_FACTOR
    globals()["PCC_ON"] = PCC_ON

    st.subheader("Ejecución")
    c1, c2 = st.columns(2)
    with c1:
        start_clicked = st.button("▶️ Iniciar", use_container_width=True, disabled=st.session_state.opt_running)
    with c2:
        stop_clicked  = st.button("⏹️ Detener", use_container_width=True, disabled=not st.session_state.opt_running)
    if start_clicked:
        st.session_state.opt_stop = False
        st.session_state.opt_running = True
    if stop_clicked:
        st.session_state.opt_stop = True

# Validaciones
if sow_search_from > sow_search_to: st.error("Rango de siembra inválido (desde > hasta)."); st.stop()
if res_min >= res_max: st.error("Residualidad: el mínimo debe ser menor que el máximo."); st.stop()
if res_step <= 0: st.error("El paso de residualidad debe ser > 0."); st.stop()
if PCC_ON and PCC_FIN < PCC_INI: st.error("PCC: la fecha de fin debe ser ≥ inicio."); st.stop()

# Datos base optimización
ts_all        = pd.to_datetime(df_plot["fecha"])
fechas_d_all  = ts_all.dt.date.values
emerrel_all   = df_plot["EMERREL"].astype(float).clip(lower=0.0).to_numpy()

mode_canopy_opt            = mode_canopy
t_lag_opt, t_close_opt     = int(t_lag), int(t_close)
cov_max_opt, lai_max_opt   = float(cov_max), float(lai_max)
k_beer_opt                 = float(k_beer)
use_ciec_opt, Ca_opt, Cs_opt, LAIhc_opt = use_ciec, float(Ca), float(Cs), float(LAIhc)

def compute_ciec_for(sd):
    FCx, LAIx = compute_canopy(ts_all, sd, mode_canopy_opt, t_lag_opt, t_close_opt, cov_max_opt, lai_max_opt, k_beer_opt)
    if use_ciec_opt:
        Ca_safe = Ca_opt if Ca_opt > 0 else 1e-6
        Cs_safe = Cs_opt if Cs_opt > 0 else 1e-6
        Ciec_loc = np.clip((LAIx / max(1e-6, LAIhc_opt)) * (Ca_safe / Cs_safe), 0.0, 1.0)
    else:
        Ciec_loc = np.zeros_like(LAIx, float)
    return np.clip(1.0 - Ciec_loc, 0.0, 1.0)

def _apply_pcc_sensitivity(one_minus_arr: np.ndarray) -> np.ndarray:
    dates_dt = ts_all.dt.date.values
    sens = np.ones_like(one_minus_arr, dtype=float)
    if PCC_ON:
        mask = (dates_dt >= PCC_INI) & (dates_dt <= PCC_FIN)
        sens = np.where(mask, float(SENS_FACTOR), 1.0)
    # No acotamos por arriba para permitir >1 si la sensibilidad amplifica la presión
    return np.clip(one_minus_arr * sens, 0.0, None)

def recompute_for_sow(sow_d: dt.date, T12: int, T23: int, T34: int):
    mask_since = (ts_all.dt.date >= sow_d)
    one_minus  = compute_ciec_for(sow_d)
    one_minus_sens = _apply_pcc_sensitivity(one_minus)

    births = np.where(mask_since.to_numpy(), emerrel_all, 0.0)

    # ------------------ ESTADOS FENOLÓGICOS SECUENCIALES (S1→S4) ------------------
    S1 = births.copy()
    S2 = np.zeros_like(births)
    S3 = np.zeros_like(births)
    S4 = np.zeros_like(births)

    for i in range(len(births)):
        if i - int(T12) >= 0:
            moved = births[i - int(T12)]
            S1[i - int(T12)] -= moved
            S2[i] += moved
        if i - (int(T12) + int(T23)) >= 0:
            moved = births[i - (int(T12) + int(T23))]
            S2[i - (int(T12) + int(T23))] -= moved
            S3[i] += moved
        if i - (int(T12) + int(T23) + int(T34)) >= 0:
            moved = births[i - (int(T12) + int(T23) + int(T34))]
            S3[i - (int(T12) + int(T23) + int(T34))] -= moved
            S4[i] += moved

    S1 = np.clip(S1, 0.0, None)
    S2 = np.clip(S2, 0.0, None)
    S3 = np.clip(S3, 0.0, None)
    S4 = np.clip(S4, 0.0, None)

    total_states = S1 + S2 + S3 + S4
    emeac = np.cumsum(births)
    scale = np.divide(np.clip(emeac, 1e-9, None), np.clip(total_states, 1e-9, None))
    scale = np.minimum(scale, 1.0)
    S1 *= scale; S2 *= scale; S3 *= scale; S4 *= scale

    # Escalado por AUC
    auc_cruda_loc = auc_time(ts_all, emerrel_all, mask=mask_since)
    if auc_cruda_loc <= 0:
        return None

    factor_area = MAX_PLANTS_CAP / auc_cruda_loc

    # Aporte por estado ponderado por (1−Ciec) sens. y coeficientes S1..S4
    S1_pl = np.where(mask_since, S1 * one_minus_sens * 0.0 * factor_area, 0.0)
    S2_pl = np.where(mask_since, S2 * one_minus_sens * 0.3 * factor_area, 0.0)
    S3_pl = np.where(mask_since, S3 * one_minus_sens * 0.6 * factor_area, 0.0)
    S4_pl = np.where(mask_since, S4 * one_minus_sens * 1.0 * factor_area, 0.0)

    base_pl_daily     = np.where(mask_since, emerrel_all * factor_area, 0.0)
    base_pl_daily_cap = cap_cumulative(base_pl_daily, MAX_PLANTS_CAP, mask_since.to_numpy())
    sup_cap           = np.minimum(S1_pl + S2_pl + S3_pl + S4_pl, base_pl_daily_cap)

    return {
        "mask_since": mask_since.to_numpy(),
        "factor_area": factor_area,
        "auc_cruda": auc_cruda_loc,
        "S_pl": (S1_pl, S2_pl, S3_pl, S4_pl),
        "sup_cap": sup_cap,
        "ts": ts_all,
        "fechas_d": fechas_d_all,
        "one_minus_sens": one_minus_sens
    }

# Acciones y combinatoria
def act_presiembraR(date_val, R, eff): return {"kind":"preR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_preemR(date_val, R, eff):     return {"kind":"preemR",  "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2"]}
def act_post_selR(date_val, R, eff):  return {"kind":"postR",   "date":pd.to_datetime(date_val).date(), "days":int(R), "eff":eff, "states":["S1","S2","S3","S4"]}
def act_post_gram(date_val, eff):     return {"kind":"post_gram","date":pd.to_datetime(date_val).date(), "days":POST_GRAM_FORWARD_DAYS, "eff":eff, "states":["S1","S2","S3"]}

def evaluate(sd: dt.date, schedule: list):
    sow = pd.to_datetime(sd)
    sow_plus_20 = sow + pd.Timedelta(days=20)
    # Reglas duras de fechas
    for a in schedule:
        d = pd.to_datetime(a["date"])
        if a["kind"] == "postR" and d < sow_plus_20: return None
        if a["kind"] == "preR" and d > (sow - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)): return None
        if a["kind"] == "preemR" and (d < sow or d > (sow + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS))): return None

    env = recompute_for_sow(sd, int(T12), int(T23), int(T34))
    if env is None: return None
    mask_since = env["mask_since"]; factor_area = env["factor_area"]
    S1_pl, S2_pl, S3_pl, S4_pl = env["S_pl"]; sup_cap = env["sup_cap"]
    ts_local, fechas_d_local = env["ts"], env["fechas_d"]

    # Controles por estado (1.0 = sin control)
    c1 = np.ones_like(fechas_d_local, float)
    c2 = np.ones_like(fechas_d_local, float)
    c3 = np.ones_like(fechas_d_local, float)
    c4 = np.ones_like(fechas_d_local, float)

    def _remaining_in_window_eval(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1_pl * c1 * w)
        if "S2" in states: rem += np.sum(S2_pl * c2 * w)
        if "S3" in states: rem += np.sum(S3_pl * c3 * w)
        if "S4" in states: rem += np.sum(S4_pl * c4 * w)
        return float(rem)

    def _apply_eval(w, eff, states):
        if eff <= 0: return False
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)
        return True

    eff_accum_pre = 0.0     # preR
    eff_accum_pre2 = 0.0    # preR + preemR
    eff_accum_all = 0.0     # + postR

    def _eff_from_to(prev_eff, this_eff):
        return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    order = {"preR":0,"preemR":1,"postR":2,"post_gram":3,"NR_pre":4}
    schedule_sorted = sorted(schedule, key=lambda a: order.get(a["kind"], 9))

    for a in schedule_sorted:
        d0, d1 = a["date"], a["date"] + pd.Timedelta(days=int(a["days"]))
        w = ((fechas_d_local >= d0) & (fechas_d_local < d1)).astype(float)

        if a["kind"] == "preR":
            if _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN and a["eff"] > 0:
                _apply_eval(w, a["eff"], ["S1","S2"])
                eff_accum_pre = _eff_from_to(0.0, a["eff"]/100.0)

        elif a["kind"] == "preemR":
            if eff_accum_pre < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2"])
                    eff_accum_pre2 = _eff_from_to(eff_accum_pre, a["eff"]/100.0)
                else:
                    eff_accum_pre2 = eff_accum_pre
            else:
                eff_accum_pre2 = eff_accum_pre

        elif a["kind"] == "postR":
            if eff_accum_pre2 < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2","S3","S4"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2","S3","S4"])
                    eff_accum_all = _eff_from_to(eff_accum_pre2, a["eff"]/100.0)
                else:
                    eff_accum_all = eff_accum_pre2
            else:
                eff_accum_all = eff_accum_pre2

        elif a["kind"] == "post_gram":
            if eff_accum_all < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2","S3"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2","S3"])

        else:
            if a.get("states"):
                if _remaining_in_window_eval(w, a["states"]) > EPS_REMAIN and a["eff"] > 0:
                    _apply_eval(w, a["eff"], a["states"])

    tot_ctrl = S1_pl*c1 + S2_pl*c2 + S3_pl*c3 + S4_pl*c4
    plantas_ctrl_cap = np.minimum(tot_ctrl, sup_cap)

    X2loc = float(np.nansum(sup_cap[mask_since])); X3loc = float(np.nansum(plantas_ctrl_cap[mask_since]))
    loss3 = _loss(X3loc)

    auc_cruda_loc = env["auc_cruda"]
    sup_equiv  = np.divide(sup_cap,          factor_area, out=np.zeros_like(sup_cap),          where=(factor_area>0))
    ctrl_equiv = np.divide(plantas_ctrl_cap, factor_area, out=np.zeros_like(plantas_ctrl_cap), where=(factor_area>0))
    auc_sup      = auc_time(ts_local, sup_equiv,  mask=mask_since)
    auc_sup_ctrl = auc_time(ts_local, ctrl_equiv, mask=mask_since)
    A2_sup  = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup/auc_cruda_loc))
    A2_ctrl = min(MAX_PLANTS_CAP, MAX_PLANTS_CAP*(auc_sup_ctrl/auc_cruda_loc))

    return {"sow": sd, "loss_pct": float(loss3), "x2": X2loc, "x3": X3loc, "A2_sup": A2_sup, "A2_ctrl": A2_ctrl, "schedule": schedule}

# Candidatos discretos
def daterange(start_date, end_date, step_days):
    out=[]; cur=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    while cur<=end: out.append(cur); cur=cur+pd.Timedelta(days=int(step_days))
    return out

sow_candidates = daterange(sow_search_from, sow_search_to, sow_step_days)

def pre_sow_dates(sd):
    start = pd.to_datetime(sd) - pd.Timedelta(days=int(preR_min_back))
    end   = pd.to_datetime(sd) - pd.Timedelta(days=PRESIEMBRA_R_MIN_DAYS_BEFORE_SOW)
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preR_step_days))
    return out

def preem_dates(sd):
    start = pd.to_datetime(sd); end = pd.to_datetime(sd) + pd.Timedelta(days=PREEM_R_MAX_AFTER_SOW_DAYS)
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(preem_step_days))
    return out

def post_dates(sd):
    start = pd.to_datetime(sd) + pd.Timedelta(days=20)
    end   = pd.to_datetime(sd) + pd.Timedelta(days=int(post_days_fw))
    if end < start: return []
    cur, out = start, []
    while cur <= end: out.append(cur); cur = cur + pd.Timedelta(days=int(post_step_days))
    return out

res_days = list(range(int(res_min), int(res_max) + 1, int(res_step)))
if int(res_max) not in res_days: res_days.append(int(res_max))

status_ph = st.empty()
prog_ph = st.empty()
results = []

def build_all_scenarios():
    scenarios = []
    for sd in sow_candidates:
        grp = []
        if use_preR_opt:
            grp.append([act_presiembraR(d, R, ef_preR_opt) for d in pre_sow_dates(sd) for R in res_days])
        if use_preemR_opt:
            grp.append([act_preemR(d, R, ef_preemR_opt) for d in preem_dates(sd) for R in res_days])
        if use_post_selR_opt:
            grp.append([act_post_selR(d, R, ef_post_selR_opt) for d in post_dates(sd) for R in res_days])
        if use_post_gram_opt:
            grp.append([act_post_gram(d, ef_post_gram_opt) for d in post_dates(sd)])
        combos = [[]]
        for r in range(1, len(grp)+1):
            for subset in itertools.combinations(range(len(grp)), r):
                for p in itertools.product(*[grp[i] for i in subset]):
                    combos.append(list(p))
        scenarios.extend([(pd.to_datetime(sd).date(), sch) for sch in combos])
    return scenarios

def sample_random_scenario():
    sd = random.choice(sow_candidates)
    schedule = []
    if use_preR_opt and random.random()<0.7:
        cand = [d for d in pre_sow_dates(sd)]
        if cand: schedule.append(act_presiembraR(random.choice(cand), random.choice(res_days), ef_preR_opt))
    if use_preemR_opt and random.random()<0.7:
        cand = [d for d in preem_dates(sd)]
        if cand: schedule.append(act_preemR(random.choice(cand), random.choice(res_days), ef_preemR_opt))
    if use_post_selR_opt and random.random()<0.7:
        cand = [d for d in post_dates(sd)]
        if cand: schedule.append(act_post_selR(random.choice(cand), random.choice(res_days), ef_post_selR_opt))
    if use_post_gram_opt and random.random()<0.7:
        cand = [d for d in post_dates(sd)]
        if cand: schedule.append(act_post_gram(random.choice(cand), ef_post_gram_opt))
    return (pd.to_datetime(sd).date(), schedule)

if factor_area_to_plants is None or not np.isfinite(auc_cruda):
    st.info("Necesitás AUC(EMERREL cruda) > 0 para optimizar.")
else:
    if st.session_state.opt_running:
        status_ph.info("Optimizando…")
        if optimizer == "Grid (combinatorio)":
            scenarios = build_all_scenarios()
            total = len(scenarios)
            st.caption(f"Se evaluarán {total:,} configuraciones")
            if total > max_evals:
                random.seed(123)
                scenarios = random.sample(scenarios, k=int(max_evals))
                st.caption(f"Se muestrean {len(scenarios):,} configs (límite)")
            prog = prog_ph.progress(0.0); n = len(scenarios); step = max(1, n//100)
            for i,(sd,sch) in enumerate(scenarios,1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Detenida. Progreso: {i-1:,}/{n:,}")
                    break
                r = evaluate(sd, sch)
                if r is not None: results.append(r)
                if i % step == 0 or i == n: prog.progress(min(1.0, i/n))
            prog_ph.empty()
        elif optimizer == "Búsqueda aleatoria":
            N = int(max_evals); prog = prog_ph.progress(0.0)
            for i in range(1, N+1):
                if st.session_state.opt_stop:
                    status_ph.warning(f"Detenida. Progreso: {i-1:,}/{N:,}")
                    break
                sd, sch = sample_random_scenario()
                r = evaluate(sd, sch)
                if r is not None: results.append(r)
                if i % max(1, N//100) == 0 or i == N: prog.progress(min(1.0, i/N))
            prog_ph.empty()
        else:
            cur = sample_random_scenario()
            cur_eval = evaluate(*cur)
            tries=0
            while cur_eval is None and tries<200:
                cur = sample_random_scenario(); cur_eval = evaluate(*cur); tries+=1
            if cur_eval is None:
                status_ph.error("No fue posible encontrar un estado inicial válido.")
            else:
                best_eval = cur_eval; cur_loss = cur_eval["loss_pct"]; T = float(sa_T0)
                prog = prog_ph.progress(0.0)
                for it in range(1, int(sa_iters)+1):
                    if st.session_state.opt_stop:
                        status_ph.warning(f"Detenida en iteración {it-1:,}/{int(sa_iters):,}.")
                        break
                    cand = sample_random_scenario()
                    cand_eval = evaluate(*cand)
                    if cand_eval is not None:
                        d = cand_eval["loss_pct"] - cur_loss
                        if d <= 0 or random.random() < _math.exp(-d / max(1e-9, T)):
                            cur, cur_eval, cur_loss = cand, cand_eval, cand_eval["loss_pct"]
                            results.append(cur_eval)
                            if cur_loss < best_eval["loss_pct"]:
                                best_eval = cur_eval
                    T *= float(sa_cooling)
                    if it % max(1, int(sa_iters)//100) == 0 or it == int(sa_iters):
                        prog.progress(min(1.0, it/float(sa_iters)))
                results.append(best_eval)
                prog_ph.empty()
        st.session_state.opt_running = False
        st.session_state.opt_stop = False
        status_ph.success("Optimización finalizada.")
    else:
        status_ph.info("Listo para optimizar. Ajustá parámetros y presioná **Iniciar**.")

# ------------------ RECOMPUTE & PLOT BEST ------------------
def recompute_apply_best(best):
    sow_best = pd.to_datetime(best["sow"]).date()
    env = recompute_for_sow(sow_best, int(T12), int(T23), int(T34))
    if env is None: return None

    ts_b, fechas_d_b = env["ts"], env["fechas_d"]
    mask_since_b = env["mask_since"]
    S1p, S2p, S3p, S4p = env["S_pl"]
    sup_cap_b = env["sup_cap"]

    c1 = np.ones_like(fechas_d_b, float)
    c2 = np.ones_like(fechas_d_b, float)
    c3 = np.ones_like(fechas_d_b, float)
    c4 = np.ones_like(fechas_d_b, float)

    def _remaining_in_window_eval(w, states):
        rem = 0.0
        if "S1" in states: rem += np.sum(S1p * c1 * w)
        if "S2" in states: rem += np.sum(S2p * c2 * w)
        if "S3" in states: rem += np.sum(S3p * c3 * w)
        if "S4" in states: rem += np.sum(S4p * c4 * w)
        return float(rem)

    def _apply_eval(w, eff, states):
        if eff <= 0: return False
        reduc = np.clip(1.0 - (eff/100.0)*np.clip(w,0.0,1.0), 0.0, 1.0)
        if "S1" in states: np.multiply(c1, reduc, out=c1)
        if "S2" in states: np.multiply(c2, reduc, out=c2)
        if "S3" in states: np.multiply(c3, reduc, out=c3)
        if "S4" in states: np.multiply(c4, reduc, out=c4)
        return True

    eff_accum_pre = 0.0
    eff_accum_pre2 = 0.0
    eff_accum_all = 0.0
    def _eff_from_to(prev_eff, this_eff): return 1.0 - (1.0 - prev_eff) * (1.0 - this_eff)

    order = {"preR":0,"preemR":1,"postR":2,"post_gram":3,"NR_pre":4}
    for a in sorted(best["schedule"], key=lambda a: order.get(a["kind"], 9)):
        ini = pd.to_datetime(a["date"]).date()
        fin = (pd.to_datetime(a["date"]) + pd.Timedelta(days=int(a["days"]))).date()
        w = ((fechas_d_b >= ini) & (fechas_d_b < fin)).astype(float)

        if a["kind"] == "preR":
            if _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN and a["eff"] > 0:
                _apply_eval(w, a["eff"], ["S1","S2"])
                eff_accum_pre = _eff_from_to(0.0, a["eff"]/100.0)
        elif a["kind"] == "preemR":
            if eff_accum_pre < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2"])
                    eff_accum_pre2 = _eff_from_to(eff_accum_pre, a["eff"]/100.0)
                else:
                    eff_accum_pre2 = eff_accum_pre
            else:
                eff_accum_pre2 = eff_accum_pre
        elif a["kind"] == "postR":
            if eff_accum_pre2 < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2","S3","S4"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2","S3","S4"])
                    eff_accum_all = _eff_from_to(eff_accum_pre2, a["eff"]/100.0)
                else:
                    eff_accum_all = eff_accum_pre2
            else:
                eff_accum_all = eff_accum_pre2
        elif a["kind"] == "post_gram":
            if eff_accum_all < EPS_EXCLUDE and a["eff"] > 0:
                if _remaining_in_window_eval(w, ["S1","S2","S3"]) > EPS_REMAIN:
                    _apply_eval(w, a["eff"], ["S1","S2","S3"])
        else:
            if a.get("states"):
                if _remaining_in_window_eval(w, a["states"]) > EPS_REMAIN and a["eff"] > 0:
                    _apply_eval(w, a["eff"], a["states"])

    total_ctrl_daily = (S1p*c1 + S2p*c2 + S3p*c3 + S4p*c4)
    eps = 1e-12
    scale = np.where(total_ctrl_daily > eps, np.minimum(1.0, sup_cap_b / total_ctrl_daily), 0.0)

    S1_ctrl_cap_b = S1p * c1 * scale
    S2_ctrl_cap_b = S2p * c2 * scale
    S3_ctrl_cap_b = S3p * c3 * scale
    S4_ctrl_cap_b = S4p * c4 * scale

    df_daily_b = pd.DataFrame({
        "fecha": ts_b,
        "pl_sin_ctrl_cap": np.where(mask_since_b, sup_cap_b, 0.0),
        "pl_con_ctrl_cap": np.where(mask_since_b, S1_ctrl_cap_b+S2_ctrl_cap_b+S3_ctrl_cap_b+S4_ctrl_cap_b, 0.0),
    })
    df_week_b = df_daily_b.set_index("fecha").resample("W-MON").sum().reset_index()

    return {"ts_b": ts_b, "mask_since_b": mask_since_b, "S_ctrl": (S1_ctrl_cap_b, S2_ctrl_cap_b, S3_ctrl_cap_b, S4_ctrl_cap_b),
            "week": df_week_b, "sup_cap_b": sup_cap_b, "daily_ctrl_cap": S1_ctrl_cap_b+S2_ctrl_cap_b+S3_ctrl_cap_b+S4_ctrl_cap_b}

# ------------------ REPORTE MEJORES ------------------
if results:
    results_sorted = sorted(results, key=lambda r: (r["loss_pct"], r["x3"]))
    best = results_sorted[0]

    st.subheader("🏆 Mejor escenario")
    st.markdown(
        f"**Siembra:** **{best['sow']}**  \n"
        f"**Pérdida estimada:** **{best['loss_pct']:.2f}%**  \n"
        f"**x₂:** {best['x2']:.1f} · **x₃:** {best['x3']:.1f} pl·m²  \n"
        f"**A2_sup:** {best['A2_sup']:.1f} · **A2_ctrl:** {best['A2_ctrl']:.1f} pl·m²"
    )

    def schedule_df(sch):
        rows=[]
        for a in sch:
            ini = pd.to_datetime(a["date"])
            fin = ini + pd.Timedelta(days=int(a["days"]))
            rows.append({
                "Intervención": a["kind"],
                "Inicio": str(ini.date()),
                "Fin": str(fin.date()),
                "Duración (d)": int(a["days"]),
                "Eficiencia (%)": int(a["eff"]),
                "Estados": ",".join(a["states"])
            })
        return pd.DataFrame(rows)

    df_best = schedule_df(best["schedule"])
    if len(df_best):
        st.dataframe(df_best, use_container_width=True)
        st.download_button("Descargar mejor cronograma (CSV)",
                           df_best.to_csv(index=False).encode("utf-8"),
                           "mejor_cronograma.csv", "text/csv")

    envb = recompute_apply_best(best)
    if envb is None:
        st.info("No se pudieron recomputar series para el mejor escenario.")
    else:
        ts_b = envb["ts_b"]; df_week_b = envb["week"]; S1c, S2c, S3c, S4c = envb["S_ctrl"]
        sup_cap_b = envb["sup_cap_b"]; daily_ctrl_cap_b = envb["daily_ctrl_cap"]

        st.subheader("📊 Gráfico 1 — Mejor escenario")
        fig_best1 = go.Figure()
        fig_best1.add_trace(go.Scatter(x=ts, y=df_plot["EMERREL"], mode="lines", name="EMERREL (cruda)"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_sin_ctrl_cap"], name="Aporte semanal (sin control, cap)", yaxis="y2", mode="lines+markers"))
        fig_best1.add_trace(go.Scatter(x=df_week_b["fecha"], y=df_week_b["pl_con_ctrl_cap"], name="Aporte semanal (con control, cap)", yaxis="y2", mode="lines+markers", line=dict(dash="dot")))

        # Curva Ciec recomputada para la siembra óptima
        one_minus_best = compute_ciec_for(best["sow"])
        Ciec_best = 1.0 - one_minus_best
        fig_best1.add_trace(go.Scatter(x=ts_b, y=Ciec_best, mode="lines", name="Ciec (mejor)", yaxis="y3"))

        fig_best1.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            title="EMERREL y plantas·m²·semana · Mejor escenario",
            xaxis_title="Tiempo", yaxis_title="EMERREL",
            yaxis2=dict(overlaying="y", side="right", title="pl·m²·sem⁻¹", range=[0,100]),
            yaxis3=dict(overlaying="y", side="right", title="Ciec", position=0.97, range=[0,1])
        )

        for a in best["schedule"]:
            x0 = pd.to_datetime(a["date"]); x1 = x0 + pd.Timedelta(days=int(a["days"]))
            fig_best1.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="rgba(30,144,255,0.18)", opacity=0.18)
            fig_best1.add_annotation(x=x0 + (x1-x0)/2, y=0.86, xref="x", yref="paper",
                                     text=a["kind"], showarrow=False, bgcolor="rgba(30,144,255,0.85)")
        st.plotly_chart(fig_best1, use_container_width=True)

        # Pérdida vs x
        X2_b = float(np.nansum(sup_cap_b[envb["mask_since_b"]]))
        X3_b = float(np.nansum(daily_ctrl_cap_b[envb["mask_since_b"]]))
        x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400); y_curve = _loss(x_curve)
        fig2_best = go.Figure()
        fig2_best.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x"))
        fig2_best.add_trace(go.Scatter(x=[X2_b], y=[_loss(X2_b)], mode="markers+text", name="x₂ (sin ctrl)", text=[f"x₂={X2_b:.1f}"], textposition="top center"))
        fig2_best.add_trace(go.Scatter(x=[X3_b], y=[_loss(X3_b)], mode="markers+text", name="x₃ (con ctrl)", text=[f"x₃={X3_b:.1f}"], textposition="top right"))
        fig2_best.update_layout(title="Figura 2 — Pérdida de rendimiento (%) vs x", xaxis_title="x (pl·m²)", yaxis_title="Pérdida (%)")
        st.plotly_chart(fig2_best, use_container_width=True)

        # Dinámica S1–S4 (con control + cap)
        df_states_week_b = (
            pd.DataFrame({"fecha": ts_b, "S1": S1c, "S2": S2c, "S3": S3c, "S4": S4c})
            .set_index("fecha").resample("W-MON").sum().reset_index()
        )
        st.subheader("Figura 4 — Dinámica temporal de S1–S4 (con control + cap)")
        fig_states = go.Figure()
        for col in ["S1","S2","S3","S4"]:
            fig_states.add_trace(go.Scatter(x=df_states_week_b["fecha"], y=df_states_week_b[col], mode="lines", name=col, stackgroup="one"))
        fig_states.update_layout(title="Aportes semanales por estado (con control + cap)", xaxis_title="Tiempo", yaxis_title="pl·m²·sem⁻¹")
        st.plotly_chart(fig_states, use_container_width=True)

        # --------- Descomposición de pérdida PCC vs Fuera PCC ---------
        st.subheader("📉 Descomposición de pérdida: PCC vs fuera PCC (mejor escenario)")
        dates_best = np.array([pd.Timestamp(d).date() for d in ts_b])
        mask_pcc = (dates_best >= PCC_INI) & (dates_best <= PCC_FIN) if PCC_ON else np.zeros_like(dates_best, bool)
        mask_grow = envb["mask_since_b"]

        x3_daily = np.where(mask_grow, daily_ctrl_cap_b, 0.0)
        x2_daily = np.where(mask_grow, sup_cap_b, 0.0)

        X3_pcc   = float(np.nansum(x3_daily[mask_pcc]))
        X3_off   = float(np.nansum(x3_daily[~mask_pcc]))
        X2_pcc   = float(np.nansum(x2_daily[mask_pcc]))
        X2_off   = float(np.nansum(x2_daily[~mask_pcc]))
        L3_pcc   = float(_loss(X3_pcc))
        L3_off   = float(_loss(X3_off))

        df_pcc = pd.DataFrame({
            "Segmento":["PCC","Fuera PCC","Total"],
            "x₃ (pl·m²)":[X3_pcc, X3_off, X3_pcc+X3_off],
            "x₂ (pl·m²)":[X2_pcc, X2_off, X2_pcc+X2_off],
            "Pérdida estimada (%)":[L3_pcc, L3_off, float(_loss(X3_pcc+X3_off))]
        })
        st.dataframe(df_pcc, use_container_width=True)

        fig_pcc = go.Figure()
        fig_pcc.add_trace(go.Bar(x=["PCC","Fuera PCC"], y=[X3_pcc, X3_off], name="x₃ (pl·m²)"))
        fig_pcc.update_layout(title="Contribución a x₃ por segmento (mejor escenario)", xaxis_title="Segmento", yaxis_title="x₃ (pl·m²)")
        st.plotly_chart(fig_pcc, use_container_width=True)
else:
    st.info("Aún no hay resultados de optimización para mostrar.")











