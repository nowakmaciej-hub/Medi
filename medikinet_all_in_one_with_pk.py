import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.special import gamma as gamma_func

# Hard cap on number of doses suggested by optimizer
MAX_DOSES = 3

# ===== App config =====
st.set_page_config(page_title="Medikinet CR – Advanced PK Models", layout="wide")
st.title("Medikinet CR – Advanced Pharmacokinetic Models")
st.caption("Multiple PK models with increasing complexity. For educational purposes only — not medical advice.")

# ===== Core model =====
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start, dose_mg, t0_hours, fed):
    # split 50/50 (Medikinet CR formulation)
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
    """One-compartment model with first-order absorption and elimination."""
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
    """Two-input PK model: IR Bateman + lagged ER Bateman with shared ke."""
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

def weibull_release(t, scale, shape):
    """Weibull function for ER drug release kinetics."""
    t = np.maximum(t, 0)
    return 1 - np.exp(-(t/scale)**shape)

def two_compartment_model(t, dose_mg, t0, fed, is_ir_only=False, add_variability=False):
    """Two-compartment PK model with Weibull ER release.
    More realistic for methylphenidate distribution and elimination."""
    
    # Convert dose to relative units
    if isinstance(dose_mg, dict):
        mg = dose_mg["mg"]
        is_ir_only = dose_mg.get("type", "IR+ER") == "IR only"
    else:
        mg = float(dose_mg)
    
    # Population variability (if enabled)
    if add_variability:
        # Add 20% CV to key parameters
        var_factor = np.random.lognormal(0, 0.2, 4)
    else:
        var_factor = np.ones(4)
    
    # Two-compartment parameters (based on methylphenidate literature)
    # Central compartment
    k10 = 0.693 * var_factor[0]  # Elimination from central (1/h)
    k12 = 0.5 * var_factor[1]    # Central to peripheral (1/h)
    k21 = 0.3 * var_factor[2]    # Peripheral to central (1/h)
    ka = 2.0 * var_factor[3]      # Absorption rate constant (1/h)
    
    # Volume and bioavailability adjustments
    if fed:
        F = 0.85  # Bioavailability with food
        tlag_ir = 0.5
        tlag_er = 2.0
        weibull_scale = 4.0  # Hours for 63% ER release
        weibull_shape = 2.2  # Shape parameter
    else:
        F = 0.75  # Bioavailability fasted
        tlag_ir = 0.0
        tlag_er = 1.0
        weibull_scale = 3.0
        weibull_shape = 2.0
    
    dt = t - t0
    conc = np.zeros_like(t)
    
    # Only calculate for t > t0
    mask = dt > 0
    if not np.any(mask):
        return np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    
    t_calc = dt[mask]
    
    if is_ir_only:
        # IR only - simple first-order input
        ir_input = mg * F * ka * np.exp(-ka * np.maximum(t_calc - tlag_ir, 0))
        ir_input[t_calc < tlag_ir] = 0
        er_conc = np.zeros_like(t_calc)
    else:
        # 50:50 IR:ER split
        ir_dose = 0.5 * mg * F
        er_dose = 0.5 * mg * F
        
        # IR component - immediate release after lag
        ir_input = ir_dose * ka * np.exp(-ka * np.maximum(t_calc - tlag_ir, 0))
        ir_input[t_calc < tlag_ir] = 0
        
        # ER component - Weibull release function
        t_er = np.maximum(t_calc - tlag_er, 0)
        release_fraction = weibull_release(t_er, weibull_scale, weibull_shape)
        
        # Calculate release rate (derivative of Weibull)
        release_rate = np.zeros_like(t_er)
        mask_er = t_er > 0
        if np.any(mask_er):
            release_rate[mask_er] = (er_dose * weibull_shape / weibull_scale * 
                                    (t_er[mask_er]/weibull_scale)**(weibull_shape-1) * 
                                    np.exp(-(t_er[mask_er]/weibull_scale)**weibull_shape))
        
        # ER absorption (released drug then absorbed)
        er_absorbed = np.zeros_like(t_calc)
        for i in range(1, len(t_calc)):
            if t_calc[i] > tlag_er:
                # Convolve release with absorption
                er_absorbed[i] = release_rate[i] * np.exp(-ka * 0.5)  # Simplified convolution
        
        er_conc = er_absorbed * 0.8  # Scaling factor for ER
    
    # Two-compartment disposition (simplified for both IR and ER)
    # Using hybrid constants for approximate solution
    alpha = 0.5 * (k12 + k21 + k10 + np.sqrt((k12 + k21 + k10)**2 - 4*k21*k10))
    beta = 0.5 * (k12 + k21 + k10 - np.sqrt((k12 + k21 + k10)**2 - 4*k21*k10))
    
    A = (ka * (k21 - alpha)) / ((ka - alpha) * (beta - alpha))
    B = (ka * (k21 - beta)) / ((ka - beta) * (alpha - beta))
    
    # IR concentration profile
    ir_conc = np.zeros_like(t_calc)
    for i, tc in enumerate(t_calc):
        if tc > tlag_ir:
            tc_adj = tc - tlag_ir
            if is_ir_only:
                dose_factor = mg * F
            else:
                dose_factor = 0.5 * mg * F
            ir_conc[i] = dose_factor * (A * np.exp(-alpha * tc_adj) + 
                                        B * np.exp(-beta * tc_adj) - 
                                        (A + B) * np.exp(-ka * tc_adj))
    
    # Combine concentrations
    total_conc = ir_conc + er_conc
    
    # Map back to full time array
    ir_full = np.zeros_like(t)
    er_full = np.zeros_like(t)
    total_full = np.zeros_like(t)
    
    ir_full[mask] = ir_conc
    er_full[mask] = er_conc if not is_ir_only else 0
    total_full[mask] = total_conc
    
    return ir_full, er_full, total_full

def parse_time_to_hours(t_str, start_hour):
    hh, mm = map(int, t_str.split(":"))
    rel = (hh + mm/60) - start_hour
    return rel

def simulate_total(t_axis, doses, start_hour):
    total = np.zeros_like(t_axis)
    parts = []
    
    # Get model engine and settings
    model = st.session_state.get("model_engine", "pk")
    add_var = st.session_state.get("add_variability", False)
    
    for d in doses:
        t0 = parse_time_to_hours(d["time_str"], start_hour)
        
        if model == "2comp":
            ir, er, tot = two_compartment_model(t_axis, d, t0, d["fed"], 
                                               is_ir_only=(d.get("type", "IR+ER") == "IR only"),
                                               add_variability=add_var)
        elif model == "pk":
            ir, er, tot = pk_dose_profile(t_axis, d, t0, d["fed"])
        else:  # gaussian
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
    """Return latest t (hours-from-start) to plot."""
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
    last_t = float(t_axis[int(idx[-1])]) + 0.25
    return min(last_t, 23.0 - start_hour)

def compute_t_min(doses, start_hour):
    """Return earliest time to include on the t-axis."""
    t_min = 0.0
    for d in doses or []:
        hh, mm = map(int, d["time_str"].split(":"))
        t0 = (hh + mm/60) - start_hour
        ir_tmax, er_tmax = (2.0, 5.0) if d.get("fed", False) else (1.25, 3.5)
        ir_sigma, er_sigma = 0.6, 1.5
        cand_ir = t0 + ir_tmax - 3*ir_sigma
        cand_er = t0 + er_tmax - 3*er_sigma
        t_min = min(t_min, cand_ir, cand_er)
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
    chart_height = st.slider("Chart height (px)", 120, 600, 300, 10)
    use_container = st.checkbox("Use container width (full)", True)
    chart_width = st.slider("Chart width (px)", 400, 1200, 700, 10,
                             help="Only used when full-width is OFF.")
    compact = st.checkbox("Compact chart mode", True)

    st.markdown("---")
    st.markdown("### Model Selection")
    _model_choice = st.radio(
        "Pharmacokinetic Model", 
        ["2-Compartment (most realistic)", "1-Compartment PK (Bateman)", "Gaussian (simplified)"], 
        index=0,
        help="Choose model complexity. 2-compartment is most realistic."
    )
    
    if _model_choice.startswith("2-Comp"):
        st.session_state["model_engine"] = "2comp"
        # Population variability option
        st.session_state["add_variability"] = st.checkbox(
            "Add population variability", 
            False,
            help="Simulates inter-individual differences (20% CV)"
        )
    elif _model_choice.startswith("1-Comp"):
        st.session_state["model_engine"] = "pk"
        st.session_state["add_variability"] = False
    else:
        st.session_state["model_engine"] = "gaussian"
        st.session_state["add_variability"] = False
    
    # Show model info
    with st.expander("Model Details"):
        if st.session_state["model_engine"] == "2comp":
            st.markdown("""
            **2-Compartment Model Features:**
            - Central & peripheral compartments
            - Weibull function for ER release
            - More realistic distribution phase
            - Better terminal elimination
            - Optional population variability
            
            **Parameters:**
            - k₁₀: 0.693 h⁻¹ (elimination)
            - k₁₂: 0.5 h⁻¹ (distribution)
            - k₂₁: 0.3 h⁻¹ (redistribution)
            - Weibull shape: 2.0-2.2
            """)
        elif st.session_state["model_engine"] == "pk":
            st.markdown("""
            **1-Compartment Model:**
            - Bateman function
            - First-order absorption/elimination
            - t½: 2.0 hours
            - Simple lag times
            """)
        else:
            st.markdown("""
            **Gaussian Model:**
            - Simple peak approximation
            - Fast computation
            - Less physiologically accurate
            """)

t = np.linspace(0, duration_h, int(duration_h * 60))

# ===== Simulator =====
def simulator_ui():
    st.subheader("Simulator")
    
    # Quick presets
    with st.expander("Quick Presets"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Standard Day (20-10-10)"):
                st.session_state.sim_doses = [
                    {"time_str": "08:00", "mg": 20, "fed": True, "type": "IR+ER"},
                    {"time_str": "12:00", "mg": 10, "fed": True, "type": "IR+ER"},
                    {"time_str": "16:00", "mg": 10, "fed": False, "type": "IR+ER"}
                ]
        with col2:
            if st.button("Extended Day (20-20)"):
                st.session_state.sim_doses = [
                    {"time_str": "07:00", "mg": 20, "fed": True, "type": "IR+ER"},
                    {"time_str": "13:00", "mg": 20, "fed": True, "type": "IR+ER"}
                ]
        with col3:
            if st.button("Low Dose (10-10)"):
                st.session_state.sim_doses = [
                    {"time_str": "08:00", "mg": 10, "fed": True, "type": "IR+ER"},
                    {"time_str": "14:00", "mg": 10, "fed": False, "type": "IR+ER"}
                ]
    
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []
    
    with st.expander("Add dose"):
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: h = st.number_input("Hour", 0, 23, 8, key="sim_h")
        with c2: m = st.number_input("Min", 0, 59, 0, step=5, key="sim_m")
        with c3: mg = st.selectbox("Dose", [5,10,15,20,30,40], index=3, key="sim_mg")
        with c4: fed = st.selectbox("With food?", ["Fasted","Fed"], index=1, key="sim_fed")
        with c5: ir_type = st.selectbox("Type", ["IR+ER","IR only"], index=0, key="sim_type")
        if st.button("➕ Add dose", key="sim_add"):
            st.session_state.sim_doses.append({
                "time_str": f"{int(h):02d}:{int(m):02d}",
                "mg": int(mg),
                "fed": fed=="Fed",
                "type": ir_type
            })
    
    # Show current doses
    if st.session_state.sim_doses:
        st.write("**Current doses:**")
        for i, d in enumerate(list(st.session_state.sim_doses)):
            c1, c2, c3, c4, c5 = st.columns([2,2,2,1,1])
            c1.write(f"Time: **{d['time_str']}**")
            c2.write(f"Dose: **{d['mg']} mg**")
            c3.write("Food: **" + ("Yes" if d['fed'] else "No") + "**")
            c4.write(f"**{d.get('type', 'IR+ER')}**")
            if c5.button("🗑", key=f"sim_rm_{i}"):
                st.session_state.sim_doses.pop(i)
                st.rerun()
        
        # Calculate metrics
        total_daily = sum(d['mg'] for d in st.session_state.sim_doses)
        st.info(f"**Total daily dose:** {total_daily} mg")
        
        # Dynamic time axis
        t_min = compute_t_min(st.session_state.sim_doses, start_hour)
        t = np.linspace(t_min, duration_h, int((duration_h - t_min)*60))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
        t_end = compute_t_end(total, t, start_hour)
        t = np.linspace(t_min, t_end, max(2, int((t_end - t_min)*60)))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
        
        # Calculate pharmacokinetic metrics
        if len(total) > 0 and np.max(total) > 0:
            peak_conc = np.max(total)
            peak_time_idx = np.argmax(total)
            peak_time = start_hour + t[peak_time_idx]
            
            # Find therapeutic window (20-80% of peak)
            threshold_low = 0.2 * peak_conc
            threshold_high = 0.8 * peak_conc
            above_low = np.where(total > threshold_low)[0]
            above_high = np.where(total > threshold_high)[0]
            
            if len(above_low) > 0:
                duration_20 = (t[above_low[-1]] - t[above_low[0]])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Concentration", f"{peak_conc:.1f} units")
                with col2:
                    st.metric("Time to Peak", f"{peak_time:.1f}:00")
                with col3:
                    st.metric("Duration >20% peak", f"{duration_20:.1f} hours")
    else:
        t = np.linspace(0, duration_h, int(duration_h*60))
        total, parts = np.zeros_like(t), []
        st.info("No doses yet. Add doses above to see the concentration curve.")
    
    # Plot
    fig = plt.figure(figsize=((chart_width/100.0) if not use_container else 10,
                              chart_height/100.0), dpi=100)
    
    # Main concentration curve
    plt.plot(start_hour+t, total, label="Total concentration", linewidth=2.5, color='darkblue')
    
    # Show components
    if st.checkbox("Show IR/ER components", False, key="sim_show"):
        for lbl, y in parts:
            plt.plot(start_hour+t, y, "--", label=lbl, alpha=0.6, linewidth=1.5)
    
    # Add therapeutic range shading (example)
    if st.checkbox("Show therapeutic range", True, key="sim_therapeutic"):
        if len(total) > 0 and np.max(total) > 0:
            peak = np.max(total)
            plt.axhspan(0.2*peak, 0.8*peak, alpha=0.1, color='green', label='Therapeutic range (20-80% peak)')
    
    
