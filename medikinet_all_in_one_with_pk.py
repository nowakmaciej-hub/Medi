import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Hard cap on number of doses suggested by optimizer
MAX_DOSES = 3

# ===== App config =====
st.set_page_config(page_title="Medikinet CR – All‑in‑One", layout="wide")
st.title("Medikinet CR – All‑in‑One")
st.caption("Improved PK model based on clinical methylphenidate data. For educational purposes only — not medical advice.")

# ===== Core model =====
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start, dose_mg, t0_hours, fed):
    # split 50/50 (Medikinet CR formulation)
    # Support IR only dosing
    if isinstance(dose_mg, dict):
        mg = dose_mg["mg"]
        ir_only = dose_mg.get("type", "IR+ER") == "IR only"
    else:
        mg = dose_mg
        ir_only = False
    if ir_only:
        ir_mg = mg
        er_mg = 0.0
    else:
        ir_mg = 0.5 * mg
        er_mg = 0.5 * mg
    
    # Updated tmax based on Medikinet CR literature
    # Fed state delays absorption more significantly
    ir_tmax, er_tmax = (2.0, 5.0) if fed else (1.25, 3.5)
    
    # Adjusted widths for more realistic profiles
    ir_sigma, er_sigma = 0.6, 1.5
    
    # Adjusted amplitudes - food increases bioavailability slightly
    if fed:
        ir_amp = ir_mg * 0.85
        er_amp = er_mg * 0.65
    else:
        ir_amp = ir_mg * 0.75
        er_amp = er_mg * 0.55
    
    ir = gaussian_peak(hours_from_start, t0_hours + ir_tmax, ir_sigma, ir_amp)
    er = gaussian_peak(hours_from_start, t0_hours + er_tmax, er_sigma, er_amp)
    return ir, er, ir + er


def _bateman(t, t0, ka, ke, amp):
    import numpy as _np
    dt = t - t0
    y = _np.zeros_like(t)
    mask = dt > 0
    if not _np.any(mask):
        return y
    dtm = dt[mask]
    
    # Handle case where ka ≈ ke (use L'Hôpital's rule approximation)
    if abs(ka - ke) < 0.01:
        # When ka ≈ ke, use the limiting form
        y_val = amp * ka * dtm * _np.exp(-ke * dtm)
    else:
        denom = ka - ke
        # Standard Bateman equation
        y_val = amp * (ka / denom) * (_np.exp(-ke * dtm) - _np.exp(-ka * dtm))
    
    y[mask] = y_val
    return y

# Updated PK parameters based on methylphenidate literature
PK_CFG_DEFAULT = {
    # Methylphenidate typically has t½ of 2-2.5h
    "ke": 0.3466,            # 1/h (t½ ≈ 2.0 h) - more accurate for methylphenidate
    
    # Fed state parameters (slower absorption, higher lag)
    "ka_ir_fed": 1.5,        # 1/h - IR absorption slowed by food
    "ka_er_fed": 0.4,        # 1/h - ER absorption rate
    "tlag_ir_fed": 0.5,      # h - slight delay for IR with food
    "tlag_er_fed": 2.5,      # h - significant lag for ER coating dissolution
    
    # Fasted state parameters (faster absorption)
    "ka_ir_fast": 2.5,       # 1/h - rapid IR absorption when fasted
    "ka_er_fast": 0.6,       # 1/h - ER still controlled release
    "tlag_ir_fast": 0.0,     # h - minimal lag for IR when fasted
    "tlag_er_fast": 1.0,     # h - ER coating dissolution lag
    
    # Bioavailability scaling - food slightly increases bioavailability
    "ir_scale_fed": 0.85,    # Higher with food
    "er_scale_fed": 0.65,
    "ir_scale_fast": 0.75,   # Lower when fasted
    "er_scale_fast": 0.55,
}

def pk_dose_profile(hours_from_start, dose_entry, t0_hours, fed, cfg=None):
    """Two-input PK model: IR Bateman + lagged ER Bateman with shared ke.
    Updated to better reflect Medikinet CR pharmacokinetics."""
    if cfg is None:
        cfg = PK_CFG_DEFAULT
    
    # Accept dict doses (with 'mg' & optional 'type') or plain number
    if isinstance(dose_entry, dict):
        mg = dose_entry.get("mg", dose_entry.get("dose", 0))
        is_ir_only = (dose_entry.get("type", "IR+ER") == "IR only")
    else:
        mg = float(dose_entry)
        is_ir_only = False

    # Medikinet CR is 50:50 IR:ER
    if is_ir_only:
        ir_mg, er_mg = mg, 0.0
    else:
        ir_mg, er_mg = 0.5*mg, 0.5*mg

    ke = float(cfg.get("ke", 0.3466))
    
    if fed:
        ka_ir = float(cfg.get("ka_ir_fed", 1.5))
        ka_er = float(cfg.get("ka_er_fed", 0.4))
        tlag_ir = float(cfg.get("tlag_ir_fed", 0.5))
        tlag_er = float(cfg.get("tlag_er_fed", 2.5))
        ir_scale = cfg.get("ir_scale_fed", 0.85)
        er_scale = cfg.get("er_scale_fed", 0.65)
    else:
        ka_ir = float(cfg.get("ka_ir_fast", 2.5))
        ka_er = float(cfg.get("ka_er_fast", 0.6))
        tlag_ir = float(cfg.get("tlag_ir_fast", 0.0))
        tlag_er = float(cfg.get("tlag_er_fast", 1.0))
        ir_scale = cfg.get("ir_scale_fast", 0.75)
        er_scale = cfg.get("er_scale_fast", 0.55)

    ir_amp = ir_scale * ir_mg
    er_amp = er_scale * er_mg

    ir = _bateman(hours_from_start, t0_hours + tlag_ir, ka_ir, ke, ir_amp)
    er = _bateman(hours_from_start, t0_hours + tlag_er, ka_er, ke, er_amp)
    return ir, er, ir + er

def parse_time_to_hours(t_str, start_hour):
    hh, mm = map(int, t_str.split(":"))
    # Relative hours from plot start; allow negative values (earlier than start)
    rel = (hh + mm/60) - start_hour
    return rel

def simulate_total(t_axis, doses, start_hour):
    total = np.zeros_like(t_axis)
    parts = []
    for d in doses:
        t0 = parse_time_to_hours(d["time_str"], start_hour)
        if st.session_state.get("model_engine", "pk") == "pk":
            ir, er, tot = pk_dose_profile(t_axis, d, t0, d["fed"])
        else:
            ir, er, tot = dose_profile(t_axis, d, t0, d["fed"])
        total += tot
        label_type = d.get("type", "IR+ER")
        if label_type == "IR only":
            parts.append((f"IR {d['mg']}mg @ {d['time_str']} (IR only)" + (" (fed)" if d["fed"] else " (fasted)"), ir))
        else:
            parts.append((f"IR {d['mg']/2:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), ir))
            parts.append((f"ER {d['mg']/2:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), er))
    return total, parts


def compute_t_end(total_curve, t_axis, start_hour):
    """Return latest t (hours-from-start) to plot.
    Stop at 23:00 or when curve ~0 (1% of peak), whichever is earlier.
    """
    import numpy as _np
    if total_curve is None or _np.allclose(total_curve, 0):
        return 0.0
    peak = float(_np.max(total_curve))
    if peak <= 1e-12:
        return 0.0
    eps = 0.01 * peak
    idx = _np.where(total_curve > eps)[0]
    if idx.size == 0:
        return 0.0
    # small margin to show the tail
    last_t = float(t_axis[int(idx[-1])]) + 0.25
    # cap by 23:00
    return min(last_t, 23.0 - start_hour)

def compute_t_min(doses, start_hour):
    """Return earliest time (possibly negative) to include on the t-axis
    so that we see peaks from doses taken before start_hour.
    """
    t_min = 0.0
    for d in doses or []:
        # relative time of taking the dose
        hh, mm = map(int, d["time_str"].split(":"))
        t0 = (hh + mm/60) - start_hour  # can be negative
        # Updated tmax for improved model
        ir_tmax, er_tmax = (2.0, 5.0) if d.get("fed", False) else (1.25, 3.5)
        ir_sigma, er_sigma = 0.6, 1.5
        # earliest relevant times ~ (center - 3*sigma)
        cand_ir = t0 + ir_tmax - 3*ir_sigma
        cand_er = t0 + er_tmax - 3*er_sigma
        t_min = min(t_min, cand_ir, cand_er)
    # clamp to a sensible floor to avoid huge canvases
    return max(t_min, -12.0)

def safe_trapz(y, x):
    y = np.asarray(y); x = np.asarray(x)
    if y.size < 2 or x.size < 2: return 0.0
    return float(np.trapz(y, x))

# ===== Shared controls =====
with st.sidebar:
    mode = st.selectbox("Mode", ["Simulator", "Optimizer"], index=0)
    start_hour = st.number_input("Plot start hour", 0, 23, 8)
    duration_h = st.slider("Duration (hours)", 6, 24, 12)
    chart_height = st.slider("Chart height (px)", 120, 600, 240, 10,
                             help="Adjust plot height to fit your screen.")
    use_container = st.checkbox("Use container width (full)", False,
                             help="If on, chart spans the full content width and keeps aspect ratio. Turn OFF to control exact width.")
    chart_width = st.slider("Chart width (px)", 400, 1200, 700, 10,
                             help="Only used when full-width is OFF.")
    compact = st.checkbox("Compact chart mode", True,
                          help="Tighter margins and smaller legend labels.")

    st.markdown("---")
    _model_choice = st.radio("Model engine", ["PK (Bateman, improved)", "Gaussian (simplified)"], index=0,
                             help="The PK model uses more realistic pharmacokinetic parameters based on methylphenidate literature.")
    if _model_choice.startswith("Gaussian"):
        st.session_state["model_engine"] = "gaussian"
    else:
        st.session_state["model_engine"] = "pk"
    
    # Show PK parameters info
    if st.session_state.get("model_engine") == "pk":
        with st.expander("PK Model Parameters"):
            st.markdown("""
            **Updated parameters based on clinical data:**
            - **t½**: 2.0 hours (ke = 0.347 h⁻¹)
            - **IR peak (fasted)**: ~1.25 hours
            - **IR peak (fed)**: ~2.0 hours  
            - **ER peak (fasted)**: ~3.5 hours
            - **ER peak (fed)**: ~5.0 hours
            - **Food effect**: Delays absorption, slight ↑ bioavailability
            """)

t = np.linspace(0, duration_h, int(duration_h * 60))  # minute grid

# ===== Simulator =====
def simulator_ui():
    st.subheader("Simulator")
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []
    with st.expander("Add dose"):
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: h = st.number_input("Hour", 0, 23, 8, key="sim_h")
        with c2: m = st.number_input("Min", 0, 59, 0, step=5, key="sim_m")
        with c3: mg = st.selectbox("Dose", [10,20,30,40], index=1, key="sim_mg")
        with c4: fed = st.selectbox("With food?", ["Fasted","Fed"], index=0, key="sim_fed")
        with c5: ir_type = st.selectbox("Type", ["IR+ER","IR only"], index=0, key="sim_type")
        if st.button("➕ Add dose", key="sim_add"):
            st.session_state.sim_doses.append({
                "time_str": f"{int(h):02d}:{int(m):02d}",
                "mg": int(mg),
                "fed": fed=="Fed",
                "type": ir_type
            })
    # Show and manage current doses
    if st.session_state.sim_doses:
        st.write("Current doses:")
        for i, d in enumerate(list(st.session_state.sim_doses)):
            c1, c2, c3, c4, c5 = st.columns([2,2,2,1,1])
            c1.write(f"Time: **{d['time_str']}**")
            c2.write(f"Dose: **{d['mg']} mg**")
            c3.write("With food: **" + ("Yes" if d['fed'] else "No") + "**")
            c4.write(f"Type: **{d.get('type', 'IR+ER')}**")
            if c5.button("🗑 Remove", key=f"sim_rm_{i}"):
                st.session_state.sim_doses.pop(i)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        # dynamic time axis to include pre-start doses
        t_min = compute_t_min(st.session_state.sim_doses, start_hour)
        t = np.linspace(t_min, duration_h, int((duration_h - t_min)*60))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
        t_end = compute_t_end(total, t, start_hour)
        t = np.linspace(t_min, t_end, max(2, int((t_end - t_min)*60)))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
    else:
        # no doses; default axis
        t = np.linspace(0, duration_h, int(duration_h*60))
        total, parts = np.zeros_like(t), []
        st.info("No doses yet.")
    
    # Display concentration info
    if st.session_state.sim_doses and len(total) > 0:
        peak_conc = np.max(total)
        peak_time_idx = np.argmax(total)
        peak_time = start_hour + t[peak_time_idx]
        st.info(f"**Peak concentration**: {peak_conc:.2f} units at {peak_time:.1f}:00")
    
    fig = plt.figure(figsize=((chart_width/100.0) if not use_container else 8,
                                        chart_height/100.0), dpi=100)
    plt.plot(start_hour+t, total, label="Total concentration", linewidth=2)
    if st.checkbox("Show IR/ER components", False, key="sim_show"):
        for lbl, y in parts:
            plt.plot(start_hour+t, y, "--", label=lbl, alpha=0.7)
    plt.xlabel("Hour of day"); plt.ylabel("Conc. (arb. units)"); 
    plt.title("Simulator – " + ("Improved PK Model" if st.session_state.get("model_engine","pk")=="pk" else "Gaussian Model"))
    plt.grid(True, linestyle="--", alpha=0.3); plt.legend(fontsize=(8 if compact else None))
    if not st.session_state.get("sim_show", True) and st.session_state.sim_doses:
        for d in st.session_state.sim_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            x = hh + mm/60.0
            plt.axvline(x, alpha=0.2, linestyle=":", color='gray')
            if (start_hour + t[0]) <= x <= (start_hour + t[-1]):
                import numpy as _np
                y = _np.interp(x - start_hour, t, total)
                plt.scatter([x], [y], marker="o", s=50, zorder=5)
    plt.tight_layout(pad=(0.1 if compact else 0.5));
    st.pyplot(fig, use_container_width=use_container)

# ===== Optimizer =====
def _objective_score_for_display(doses, start_hour, duration_h, target_start, target_end, lam_out, lam_rough, lam_peak):
    t_axis = np.linspace(0, duration_h, int(duration_h*60))
    total, _ = simulate_total(t_axis, doses, start_hour)
    return objective(total, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)

def objective(total_curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour):
    hours = start_hour + t_axis
    if target_end >= target_start:
        inside = (hours >= target_start) & (hours <= target_end)
    else:
        inside = (hours >= target_start) | (hours <= target_end)
    area_in  = safe_trapz(total_curve[inside],  t_axis[inside])
    area_out = safe_trapz(total_curve[~inside], t_axis[~inside])
    dcdt = np.gradient(total_curve, t_axis)
    rough = float(np.trapz(dcdt**2, t_axis))
    peak = float(np.max(total_curve))
    return area_in - lam_out*area_out - lam_rough*rough - lam_peak*peak

def times_in_window_grid(start_hour, duration_h, step_min, target_start, target_end, buffer_h):
    step_h = step_min/60.0
    grid = (np.arange(start_hour, start_hour+duration_h+1e-9, step_h) % 24)
    def in_expanded(h):
        s, e = target_start%24, target_end%24
        if e >= s: return (h >= s-buffer_h) and (h <= e+buffer_h)
        return (h >= s-buffer_h) or (h <= e+buffer_h)
    cands = [h for h in grid if in_expanded(h)]
    return [f"{int(h)%24:02d}:{int(round((h%1)*60))%60:02d}" for h in cands]

def fill_to_limit(current, t_axis, start_hour, mg_limit, fed, step_min,
                  lam_out, lam_rough, lam_peak,
                  target_start, target_end, buffer_h, min_gap_min):
    """Greedily add doses AND/OR upgrade 10→20 mg to reach the mg_limit,
    while respecting MAX_DOSES and min-gap. Uses best absolute objective criteria.
    """
    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)

    # Start state
    current = [dict(d) for d in current]  # copy
    total_curve, _ = simulate_total(t_axis, current, start_hour)
    used_mg = sum(d['mg'] for d in current)

    # Build candidate time grid
    cand_times = times_in_window_grid(start_hour, int(t_axis[-1]), step_min, target_start, target_end, buffer_h)

    def violates_gap(tstr, picks):
        def parse(tstr): 
            hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr) - parse(d['time_str']) + 12) % 24 - 12) < (min_gap_min/60.0):
                return True
        return False

    # Phase A: ADD new doses until we hit MAX_DOSES or no time slots fit
    while used_mg + 10 <= mg_limit and len(current) < MAX_DOSES:
        best_s = None
        best_add = None
        best_curve = None
        for tstr in cand_times:
            if violates_gap(tstr, current):
                continue
            for dose in (20, 10):  # prefer 20 when filling
                if used_mg + dose > mg_limit: 
                    continue
                trial = current + [{"time_str": tstr, "mg": dose, "fed": fed}]
                trial_curve, _ = simulate_total(t_axis, trial, start_hour)
                s = score(trial_curve)
                if (best_s is None) or (s > best_s + 1e-12):
                    best_s, best_add, best_curve = s, {"time_str": tstr, "mg": dose, "fed": fed}, trial_curve
        if best_add is None:
            break
        current.append(best_add)
        used_mg += best_add["mg"]
        total_curve = best_curve

    # Phase B: if we still have headroom, UPGRADE existing 10mg -> 20mg by best gain
    while used_mg + 10 <= mg_limit:
        indices_10 = [i for i,d in enumerate(current) if d["mg"] == 10]
        if not indices_10:
            break
        best_s = None
        best_idx = None
        best_curve = None
        for i in indices_10:
            trial = [dict(d) for d in current]
            trial[i]["mg"] = 20
            trial_curve, _ = simulate_total(t_axis, trial, start_hour)
            s = score(trial_curve)
            if (best_s is None) or (s > best_s + 1e-12):
                best_s, best_idx, best_curve = s, i, trial_curve
        if best_idx is None:
            break
        current[best_idx]["mg"] = 20
        used_mg += 10
        total_curve = best_curve

    return current, total_curve

def enforce_morning_first_20(doses, mg_limit):
    """Ensure the earliest dose (by clock time) is 20mg, adjusting mg values without changing times."""
    if not doses:
        return doses
    # find earliest by clock time
    def to_minutes(d):
        hh, mm = map(int, d["time_str"].split(":"))
        return hh*60 + mm
    idx_sorted = sorted(range(len(doses)), key=lambda i: to_minutes(doses[i]))
    first_idx = idx_sorted[0]
    if doses[first_idx]["mg"] == 20:
        return doses
    # look for any other 20mg
    idx20 = next((i for i in range(len(doses)) if doses[i]["mg"] == 20), None)
    total_mg = sum(d["mg"] for d in doses)
    if idx20 is not None:
        # swap mg
        doses[first_idx]["mg"], doses[idx20]["mg"] = doses[idx20]["mg"], doses[first_idx]["mg"]
        return doses
    # all 10mg
    if total_mg + 10 <= mg_limit and len(doses) <= MAX_DOSES:
        doses[first_idx]["mg"] = 20
        return doses
    # try to remove the last (by time) 10mg and upgrade earliest
    if len(doses) >= 2:
        last_idx = idx_sorted[-1]
        if doses[last_idx]["mg"] == 10:
            # remove last, upgrade first
            doses.pop(last_idx)
            doses[first_idx]["mg"] = 20
            return doses
    return doses

def greedy_optimize(start_hour, duration_h, mg_limit, fed, step_min,
                    lam_out, lam_rough, lam_peak,
                    target_start, target_end, buffer_h, min_gap_min,
                    force_use_all=False):
    t_axis = np.linspace(0, duration_h, int(duration_h * 60))
    current = []
    current_total, _ = simulate_total(t_axis, current, start_hour)
    used_mg = 0
    cand_times = times_in_window_grid(start_hour, duration_h, step_min, target_start, target_end, buffer_h)

    def violates_gap(tstr, picks):
        def parse(tstr): 
            hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr) - parse(d['time_str']) + 12) % 24 - 12) < (min_gap_min/60.0):
                return True
        return False

    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)

    while True:
        if len(current) >= MAX_DOSES:
 
