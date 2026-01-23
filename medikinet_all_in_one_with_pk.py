import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Removed unused gamma_func import

# Hard cap on number of doses suggested by optimizer
MAX_DOSES = 3

# Version number
VERSION = "v1.0.0"

# ===== App config =====
st.set_page_config(page_title="Medikinet CR – Advanced PK Models", layout="wide")
st.title("Medikinet CR – Advanced Pharmacokinetic Models")
st.caption("Multiple PK models with increasing complexity. For educational purposes only — not medical advice.")
st.text(f"Version: {VERSION}")

# ===== Core model =====
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start, dose_mg, t0_hours, fed):
    """Calculate concentration profile for a single dose."""
    # split 50/50 (Medikinet CR formulation)
    if isinstance(dose_mg, dict):
        mg = dose_mg.get("mg", 0)
        ir_only = dose_mg.get("type", "IR+ER") == "IR only"
    else:
        mg = dose_mg
        ir_only = False

    # Validate dose
    if mg <= 0:
        return np.zeros_like(hours_from_start), np.zeros_like(hours_from_start), np.zeros_like(hours_from_start)
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
    dt = t - t0
    y = np.zeros_like(t)
    mask = dt > 0
    if not np.any(mask):
        return y
    dtm = dt[mask]
    
    # Handle case where ka ≈ ke (use L'Hôpital's rule approximation)
    if abs(ka - ke) < 0.01:
        # When ka ≈ ke, use the limiting form
        y_val = amp * ka * dtm * np.exp(-ke * dtm)
    else:
        denom = ka - ke
        # Standard Bateman equation
        y_val = amp * (ka / denom) * (np.exp(-ke * dtm) - np.exp(-ka * dtm))
    
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

    # Validate dose
    if mg <= 0:
        return np.zeros_like(hours_from_start), np.zeros_like(hours_from_start), np.zeros_like(hours_from_start)

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
        mg = dose_mg.get("mg", 0)
        is_ir_only = dose_mg.get("type", "IR+ER") == "IR only"
    else:
        mg = float(dose_mg)

    # Validate dose
    if mg <= 0:
        return np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    
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
    """Parse time string to hours relative to start_hour."""
    try:
        hh, mm = map(int, t_str.split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"Invalid time: {t_str}")
        rel = (hh + mm/60) - start_hour
        return rel
    except (ValueError, AttributeError) as e:
        st.error(f"Error parsing time '{t_str}': {e}")
        return 0.0

def simulate_total(t_axis, doses, start_hour):
    """Simulate total concentration from multiple doses."""
    if len(t_axis) == 0:
        return np.array([]), []

    total = np.zeros_like(t_axis)
    parts = []

    # Get model engine and settings
    model = st.session_state.get("model_engine", "pk")
    add_var = st.session_state.get("add_variability", False)

    if not doses:
        return total, parts

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
    if total_curve is None or np.allclose(total_curve, 0):
        return 0.0
    peak = float(np.max(total_curve))
    if peak <= 1e-12:
        return 0.0
    eps = 0.01 * peak
    idx = np.where(total_curve > eps)[0]
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
    """Safely compute trapezoidal integration."""
    y = np.asarray(y)
    x = np.asarray(x)
    if y.size < 2 or x.size < 2 or y.size != x.size:
        return 0.0
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
    
    # Dose markers
    if st.session_state.sim_doses:
        for d in st.session_state.sim_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            x = hh + mm/60.0
            plt.axvline(x, alpha=0.2, linestyle=":", color='gray')
            if (start_hour + t[0]) <= x <= (start_hour + t[-1]):
                y = np.interp(x - start_hour, t, total)
                plt.scatter([x], [y], marker="v", s=80, zorder=5, color='red', label='_nolegend_')
    
    plt.xlabel("Hour of day", fontsize=11)
    plt.ylabel("Concentration (arbitrary units)", fontsize=11)
    model_name = {"2comp": "2-Compartment Model", "pk": "1-Compartment PK", "gaussian": "Gaussian"}.get(st.session_state.get("model_engine"), "Unknown")
    plt.title(f"Methylphenidate Concentration – {model_name}", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=(8 if compact else 10), loc='best')
    
    # X-axis formatting
    ax = plt.gca()
    if len(t) > 0:
        x_start = int(start_hour + t[0])
        x_end = int(start_hour + t[-1]) + 1
        if x_start < x_end:
            ax.set_xticks(range(x_start, x_end))
            ax.set_xticklabels([f"{h%24:02d}:00" for h in range(x_start, x_end)], rotation=45)
    
    plt.tight_layout(pad=(0.5 if compact else 1.0))
    st.pyplot(fig, use_container_width=use_container)
    
    # Export option
    if st.session_state.sim_doses:
        if st.button("Export dose schedule"):
            schedule = {"doses": st.session_state.sim_doses, "model": st.session_state.get("model_engine")}
            st.json(schedule)

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
    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)
    
    current = [dict(d) for d in current]
    total_curve, _ = simulate_total(t_axis, current, start_hour)
    used_mg = sum(d['mg'] for d in current)
    
    cand_times = times_in_window_grid(start_hour, int(t_axis[-1]), step_min, target_start, target_end, buffer_h)
    
    def violates_gap(tstr, picks):
        def parse(tstr): 
            hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr) - parse(d['time_str']) + 12) % 24 - 12) < (min_gap_min/60.0):
                return True
        return False
    
    while used_mg + 10 <= mg_limit and len(current) < MAX_DOSES:
        best_s = None
        best_add = None
        best_curve = None
        for tstr in cand_times:
            if violates_gap(tstr, current):
                continue
            for dose in (20, 10):
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
    if not doses:
        return doses
    def to_minutes(d):
        hh, mm = map(int, d["time_str"].split(":"))
        return hh*60 + mm
    idx_sorted = sorted(range(len(doses)), key=lambda i: to_minutes(doses[i]))
    first_idx = idx_sorted[0]
    if doses[first_idx]["mg"] == 20:
        return doses
    idx20 = next((i for i in range(len(doses)) if doses[i]["mg"] == 20), None)
    total_mg = sum(d["mg"] for d in doses)
    if idx20 is not None:
        doses[first_idx]["mg"], doses[idx20]["mg"] = doses[idx20]["mg"], doses[first_idx]["mg"]
        return doses
    if total_mg + 10 <= mg_limit and len(doses) <= MAX_DOSES:
        doses[first_idx]["mg"] = 20
        return doses
    if len(doses) >= 2:
        last_idx = idx_sorted[-1]
        if doses[last_idx]["mg"] == 10:
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
            break
        base = score(current_total)
        best_gain, best = 0.0, None
        best_abs_s, best_abs_add, best_abs_curve = None, None, None
        for tstr in cand_times:
            if len(current) >= MAX_DOSES:
                break
            for dose in (10,20):
                if used_mg + dose > mg_limit: continue
                if len(current) + 1 > MAX_DOSES: continue
                if violates_gap(tstr, current): continue
                trial = current + [{"time_str": tstr, "mg": dose, "fed": fed}]
                trial_total, _ = simulate_total(t_axis, trial, start_hour)
                s = score(trial_total)
                gain = s - base
                if gain > best_gain + 1e-12:
                    best_gain, best = gain, {"time_str": tstr, "mg": dose, "fed": fed, "_total": trial_total}
                if (best_abs_s is None) or (s > best_abs_s + 1e-12):
                    best_abs_s, best_abs_add, best_abs_curve = s, {"time_str": tstr, "mg": dose, "fed": fed, "_total": trial_total}, trial_total
        if best is None:
            if force_use_all and (sum(d['mg'] for d in current) + 10 <= mg_limit) and (len(current) < MAX_DOSES) and best_abs_add is not None:
                best = best_abs_add
            else:
                break
        current.append({k:v for k,v in best.items() if k!='_total'})
        current_total = best["_total"]
        used_mg += best["mg"]
        if used_mg + 10 > mg_limit: break
    return current, current_total, t_axis

def trim_to_mg_limit(doses, t_axis, start_hour, mg_limit,
                        lam_out, lam_rough, lam_peak, target_start, target_end):
    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)
    doses = doses[:]
    if not doses: 
        return doses
    base_curve, _ = simulate_total(t_axis, doses, start_hour)
    base_score = score(base_curve)
    while sum(d['mg'] for d in doses) > mg_limit and doses:
        best_idx = None
        best_drop = float('inf')
        for i in range(len(doses)):
            trial = doses[:i] + doses[i+1:]
            trial_curve, _ = simulate_total(t_axis, trial, start_hour)
            s = score(trial_curve)
            drop = base_score - s
            if drop < best_drop - 1e-12:
                best_drop = drop
                best_idx = i
                best_trial_curve = trial_curve
                best_trial_score = s
        if best_idx is None:
            break
        doses.pop(best_idx)
        base_curve = best_trial_curve
        base_score = best_trial_score
    return doses

def refine_split_twenty(doses, t_axis, start_hour, step_min,
                        lam_out, lam_rough, lam_peak, target_start, target_end, min_gap_min):
    if not doses or len(doses) >= MAX_DOSES: return doses, simulate_total(t_axis, doses, start_hour)[0]
    def score(curve): 
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)
    grid = times_in_window_grid(start_hour, int(t_axis[-1]), step_min, target_start, target_end, 0.0)
    def violates_gap(tstr, picks):
        def parse(tstr): hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr)-parse(d['time_str'])+12)%24-12) < (min_gap_min/60.0):
                return True
        return False
    best = doses[:]
    best_curve, _ = simulate_total(t_axis, best, start_hour); best_score = score(best_curve)
    improved = True
    while improved:
        improved = False
        for i, d in enumerate(list(best)):
            if d["mg"] != 20: continue
            base = best[:i] + best[i+1:]
            if len(base) + 2 > MAX_DOSES:
                continue
            for t1 in grid:
                if violates_gap(t1, base): continue
                for t2 in grid:
                    if violates_gap(t2, base + [{"time_str": t1,"mg":10,"fed":d["fed"]}]): continue
                    trial = base + [{"time_str": t1, "mg": 10, "fed": d["fed"]}, {"time_str": t2, "mg": 10, "fed": d["fed"]}]
                    trial_curve, _ = simulate_total(t_axis, trial, start_hour)
                    s = score(trial_curve)
                    if s > best_score + 1e-12:
                        best, best_curve, best_score = trial, trial_curve, s
                        improved = True
                        break
                if improved: break
    return best, best_curve

def optimizer_ui():
    st.subheader("Dose Schedule Optimizer")
    
    # Optimization targets
    st.markdown("### Target Coverage Window")
    c1,c2,c3 = st.columns(3)
    with c1:
        target_start = st.number_input("Start hour", 0, 23, 8, help="When you need coverage to begin")
        lambda_out  = st.slider("Penalty outside window", 0.0, 10.0, 1.0, 0.1)
    with c2:
        target_end   = st.number_input("End hour", 0, 23, 20, help="When coverage can end")
        lambda_rough = st.slider("Smoothness penalty", 0.0, 1.0, 0.15, 0.01,
                                  help="Reduces fluctuations")
    with c3:
        daily_limit  = st.number_input("Daily mg limit", 10, 120, 40, 10)
        lambda_peak  = st.slider("Peak penalty", 0.0, 5.0, 1.5, 0.1,
                                help="Limits maximum concentration")
    
    st.markdown("### Constraints")
    c4,c5,c6 = st.columns(3)
    with c4:
        step_min = st.selectbox("Time granularity", [15,30,60], index=1)
    with c5:
        cand_buffer_h = st.slider("Search buffer ±h", 0.0, 4.0, 2.0, 0.5,
                                 help="How far outside target to search")
    with c6:
        min_gap_min = st.slider("Min dose gap (min)", 60, 360, 180, 15)
    
    fed = st.checkbox("Assume doses with food", True)
    use_all_limit = st.checkbox("Use full daily allowance", False)
    morning_20 = st.checkbox("Prioritize morning dose (20mg)", True)
    
    st.caption("**Max doses/day:** 3 (regulatory limit)")
    
    if st.button("🔍 Optimize Schedule", type="primary"):
        with st.spinner("Optimizing..."):
            try:
                # Run optimization
                opt_doses, opt_curve, t_axis = greedy_optimize(
                    start_hour, duration_h, int(daily_limit), fed, int(step_min),
                    lambda_out, lambda_rough, lambda_peak,
                    target_start, target_end, cand_buffer_h, int(min_gap_min),
                    force_use_all=use_all_limit
                )

                opt_doses, opt_curve = refine_split_twenty(
                    opt_doses, t_axis, start_hour, int(step_min),
                    lambda_out, lambda_rough, lambda_peak,
                    target_start, target_end, int(min_gap_min)
                )

                if sum(d['mg'] for d in opt_doses) > int(daily_limit):
                    opt_doses = trim_to_mg_limit(
                        opt_doses, t_axis, start_hour, int(daily_limit),
                        lambda_out, lambda_rough, lambda_peak, target_start, target_end
                    )
                    opt_curve, _ = simulate_total(t_axis, opt_doses, start_hour)

                if use_all_limit and sum(d['mg'] for d in opt_doses) + 10 <= int(daily_limit):
                    opt_doses, opt_curve = fill_to_limit(
                        opt_doses, t_axis, start_hour, int(daily_limit), fed, int(step_min),
                        lambda_out, lambda_rough, lambda_peak,
                        target_start, target_end, cand_buffer_h, int(min_gap_min)
                    )

                if morning_20:
                    opt_doses = enforce_morning_first_20(opt_doses, int(daily_limit))
                    opt_curve, _ = simulate_total(t_axis, opt_doses, start_hour)
            
                import datetime as _dt
                st.session_state["_opt_last"] = {
                    "doses": opt_doses,
                    "t_axis": t_axis,
                    "ts": _dt.datetime.now().strftime("%H:%M:%S"),
                    "score": float(_objective_score_for_display(opt_doses, start_hour, duration_h, target_start, target_end, lambda_out, lambda_rough, lambda_peak))
                }
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.info("Try adjusting the constraints or target window.")
    
    # Display results
    last = st.session_state.get("_opt_last", None)
    if last is not None:
        opt_doses = last["doses"]
        t_axis = last["t_axis"]
    else:
        opt_doses, t_axis = [], np.linspace(0, duration_h, int(duration_h*60))
    
    if opt_doses:
        total_mg = sum(d["mg"] for d in opt_doses)
        
        # Display schedule as cards
        st.markdown("### 📋 Optimized Schedule")
        cols = st.columns(len(opt_doses))
        for i, (col, d) in enumerate(zip(cols, opt_doses)):
            with col:
                st.metric(
                    f"Dose {i+1}",
                    f"{d['mg']} mg",
                    f"@ {d['time_str']}"
                )
        
        st.success(f"**Total:** {total_mg}/{int(daily_limit)} mg | **Score:** {last.get('score', 0):.2f}")
    else:
        st.warning("No schedule found. Try adjusting penalties or constraints.")
    
    # Plot optimized schedule
    if opt_doses or last:
        t_min = compute_t_min(opt_doses, start_hour) if opt_doses else 0.0
        t_plot = np.linspace(t_min, duration_h, int((duration_h - t_min)*60))
        total_all, components = simulate_total(t_plot, opt_doses, start_hour) if opt_doses else (np.zeros_like(t_plot), [])
        
        fig = plt.figure(figsize=((chart_width/100.0) if not use_container else 10,
                                  chart_height/100.0), dpi=100)
        
        # Target window
        if target_end >= target_start:
            plt.axvspan(target_start, target_end, alpha=0.08, color='green', label="Target window")
        else:
            plt.axvspan(target_start, 23, alpha=0.08, color='green')
            plt.axvspan(0, target_end, alpha=0.08, color='green')
        
        # Concentration curve
        plt.plot(start_hour+t_plot, total_all, label="Optimized concentration", linewidth=2.5, color='darkblue')
        
        # Components
        if st.checkbox("Show components", False, key="opt_show"):
            for lbl, y in components:
                plt.plot(start_hour+t_plot, y, "--", label=lbl, alpha=0.6)
        
        # Dose markers
        for d in opt_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            t0 = (hh + mm/60) - start_hour
            plt.axvline(start_hour + t0, linestyle=":", alpha=0.5, color='red')
            idx = int(np.argmin(np.abs(t_plot - t0)))
            if 0 <= idx < len(t_plot):
                plt.scatter(start_hour + t_plot[idx], total_all[idx], s=100, color='red', zorder=5)
        
        plt.xlabel("Hour of day", fontsize=11)
        plt.ylabel("Concentration (arbitrary units)", fontsize=11)
        model_name = {"2comp": "2-Compartment", "pk": "1-Compartment", "gaussian": "Gaussian"}.get(st.session_state.get("model_engine"), "")
        plt.title(f"Optimized Schedule – {model_name} Model", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(fontsize=9, loc='best')
        
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=use_container)

# ===== Main Router =====
if mode == "Simulator":
    simulator_ui()
else:
    optimizer_ui()