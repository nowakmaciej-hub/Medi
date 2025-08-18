import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
    rel = (hh + mm/60) - start_hour
    return rel

def simulate_total(t_axis, doses, start_hour):
    total = np.zeros_like(t_axis)
    parts = []
    
    # Get model engine
    model = st.session_state.get("model_engine", "pk")
    
    for d in doses:
        t0 = parse_time_to_hours(d["time_str"], start_hour)
        
        if model == "pk":
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
        return duration_h
    peak = float(np.max(total_curve))
    if peak <= 1e-12:
        return duration_h
    eps = 0.01 * peak
    idx = np.where(total_curve > eps)[0]
    if idx.size == 0:
        return duration_h
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
    st.markdown("### Settings")
    mode = st.selectbox("Mode", ["Simulator", "Optimizer"], index=0)
    start_hour = st.number_input("Plot start hour", 0, 23, 8)
    duration_h = st.slider("Duration (hours)", 6, 24, 16)
    chart_height = st.slider("Chart height (px)", 200, 600, 400, 10)
    use_container = st.checkbox("Use full width", True)
    if not use_container:
        chart_width = st.slider("Chart width (px)", 400, 1200, 800, 10)
    else:
        chart_width = 800
    compact = st.checkbox("Compact chart mode", False)

    st.markdown("---")
    st.markdown("### Model Selection")
    _model_choice = st.radio(
        "Pharmacokinetic Model", 
        ["1-Compartment PK (Improved)", "Gaussian (Simplified)"], 
        index=0,
        help="Choose model complexity."
    )
    
    if _model_choice.startswith("1-Comp"):
        st.session_state["model_engine"] = "pk"
    else:
        st.session_state["model_engine"] = "gaussian"
    
    # Show model info
    with st.expander("Model Details"):
        if st.session_state["model_engine"] == "pk":
            st.markdown("""
            **1-Compartment Model Features:**
            - Bateman function (first-order absorption/elimination)
            - t½ = 2.0 hours (more accurate for methylphenidate)
            - Realistic food effects (delays & bioavailability)
            - Separate IR/ER lag times
            
            **Parameters:**
            - ke: 0.347 h⁻¹ (elimination)
            - ka_IR: 2.5 h⁻¹ (fasted), 1.5 h⁻¹ (fed)
            - ka_ER: 0.6 h⁻¹ (fasted), 0.4 h⁻¹ (fed)
            """)
        else:
            st.markdown("""
            **Gaussian Model:**
            - Simple peak approximation
            - Fast computation
            - Less physiologically accurate
            - Good for rough estimates
            """)

# ===== Simulator =====
def simulator_ui():
    st.subheader("📊 Concentration Simulator")
    
    # Initialize doses
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []
    
    # Quick presets
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Standard (20-10-10)", use_container_width=True):
            st.session_state.sim_doses = [
                {"time_str": "08:00", "mg": 20, "fed": True, "type": "IR+ER"},
                {"time_str": "12:00", "mg": 10, "fed": True, "type": "IR+ER"},
                {"time_str": "16:00", "mg": 10, "fed": False, "type": "IR+ER"}
            ]
            st.rerun()
    with col2:
        if st.button("Extended (20-20)", use_container_width=True):
            st.session_state.sim_doses = [
                {"time_str": "07:00", "mg": 20, "fed": True, "type": "IR+ER"},
                {"time_str": "13:00", "mg": 20, "fed": True, "type": "IR+ER"}
            ]
            st.rerun()
    with col3:
        if st.button("Clear All", use_container_width=True):
            st.session_state.sim_doses = []
            st.rerun()
    
    # Add dose form
    with st.expander("➕ Add New Dose", expanded=len(st.session_state.sim_doses) == 0):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: 
            h = st.number_input("Hour", 0, 23, 8, key="sim_h")
        with c2: 
            m = st.number_input("Min", 0, 59, 0, step=15, key="sim_m")
        with c3: 
            mg = st.selectbox("Dose (mg)", [5, 10, 15, 20, 30, 40], index=3, key="sim_mg")
        with c4: 
            fed = st.selectbox("With food?", ["Fasted", "Fed"], index=1, key="sim_fed")
        with c5: 
            ir_type = st.selectbox("Type", ["IR+ER", "IR only"], index=0, key="sim_type")
        
        if st.button("Add Dose", type="primary", use_container_width=True):
            st.session_state.sim_doses.append({
                "time_str": f"{int(h):02d}:{int(m):02d}",
                "mg": int(mg),
                "fed": fed == "Fed",
                "type": ir_type
            })
            st.rerun()
    
    # Display current doses
    if st.session_state.sim_doses:
        st.markdown("### Current Schedule")
        for i, d in enumerate(st.session_state.sim_doses):
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            with col1:
                st.write(f"**{d['time_str']}**")
            with col2:
                st.write(f"**{d['mg']} mg**")
            with col3:
                st.write(f"**{'Fed' if d['fed'] else 'Fasted'}**")
            with col4:
                st.write(f"**{d.get('type', 'IR+ER')}**")
            with col5:
                if st.button("🗑️", key=f"rm_{i}"):
                    st.session_state.sim_doses.pop(i)
                    st.rerun()
        
        # Calculate metrics
        total_daily = sum(d['mg'] for d in st.session_state.sim_doses)
        
        # Calculate concentration curve
        t_min = compute_t_min(st.session_state.sim_doses, start_hour)
        t_max = duration_h
        t = np.linspace(t_min, t_max, int((t_max - t_min) * 60))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
        
        # Display metrics
        if len(total) > 0 and np.max(total) > 0:
            peak_conc = np.max(total)
            peak_time_idx = np.argmax(total)
            peak_time = start_hour + t[peak_time_idx]
            
            # Find duration above thresholds
            threshold_20 = 0.2 * peak_conc
            above_20 = np.where(total > threshold_20)[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Daily", f"{total_daily} mg")
            with col2:
                st.metric("Peak Level", f"{peak_conc:.1f}")
            with col3:
                st.metric("Peak Time", f"{peak_time:.1f}h")
            with col4:
                if len(above_20) > 0:
                    duration = t[above_20[-1]] - t[above_20[0]]
                    st.metric("Duration >20%", f"{duration:.1f}h")
                else:
                    st.metric("Duration >20%", "0h")
    else:
        t = np.linspace(0, duration_h, int(duration_h * 60))
        total, parts = np.zeros_like(t), []
        st.info("👆 Add doses above to see concentration curves")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(chart_width/100, chart_height/100), dpi=100)
    
    # Plot concentration
    ax.plot(start_hour + t, total, label="Total concentration", linewidth=2.5, color='darkblue')
    
    # Show components if requested
    show_components = st.checkbox("Show IR/ER components", False)
    if show_components and parts:
        for label, curve in parts:
            ax.plot(start_hour + t, curve, '--', label=label, alpha=0.6, linewidth=1.5)
    
    # Show therapeutic range
    show_range = st.checkbox("Show therapeutic range", True)
    if show_range and len(total) > 0 and np.max(total) > 0:
        peak = np.max(total)
        ax.axhspan(0.2 * peak, 0.8 * peak, alpha=0.1, color='green', label='Therapeutic range')
    
    # Add dose markers
    for d in st.session_state.sim_doses:
        hh, mm = map(int, d["time_str"].split(":"))
        dose_time = hh + mm/60
        ax.axvline(dose_time, alpha=0.3, linestyle=':', color='gray')
        
        # Add dose point on curve
        if start_hour + t[0] <= dose_time <= start_hour + t[-1]:
            y_val = np.interp(dose_time - start_hour, t, total)
            ax.scatter([dose_time], [y_val], s=80, color='red', zorder=5, marker='v')
    
    # Formatting
    ax.set_xlabel("Time of Day", fontsize=11)
    ax.set_ylabel("Concentration (arbitrary units)", fontsize=11)
    model_name = "1-Compartment PK Model" if st.session_state.get("model_engine") == "pk" else "Gaussian Model"
    ax.set_title(f"Methylphenidate Concentration - {model_name}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    
    # Set x-axis to show hours
    x_ticks = np.arange(max(0, int(start_hour + t[0])), min(24, int(start_hour + t[-1]) + 1), 2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(h):02d}:00" for h in x_ticks])
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=use_container)

# ===== Optimizer =====
def objective(total_curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour):
    hours = start_hour + t_axis
    if target_end >= target_start:
        inside = (hours >= target_start) & (hours <= target_end)
    else:
        inside = (hours >= target_start) | (hours <= target_end)
    area_in = safe_trapz(total_curve[inside], t_axis[inside]) if np.any(inside) else 0
    area_out = safe_trapz(total_curve[~inside], t_axis[~inside]) if np.any(~inside) else 0
    dcdt = np.gradient(total_curve, t_axis) if len(total_curve) > 1 else np.zeros_like(total_curve)
    rough = float(np.trapz(dcdt**2, t_axis)) if len(dcdt) > 1 else 0
    peak = float(np.max(total_curve)) if len(total_curve) > 0 else 0
    return area_in - lam_out*area_out - lam_rough*rough - lam_peak*peak

def optimizer_ui():
    st.subheader("🎯 Dose Schedule Optimizer")
    
    # Target settings
    st.markdown("### Target Coverage")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_start = st.number_input("Start hour", 0, 23, 8)
        lambda_out = st.slider("Outside penalty", 0.0, 5.0, 1.0, 0.1)
    with col2:
        target_end = st.number_input("End hour", 0, 23, 20)
        lambda_rough = st.slider("Smoothness", 0.0, 1.0, 0.2, 0.05)
    with col3:
        daily_limit = st.number_input("Daily limit (mg)", 10, 80, 40, 10)
        lambda_peak = st.slider("Peak penalty", 0.0, 3.0, 1.0, 0.1)
    
    # Constraints
    st.markdown("### Constraints")
    col4, col5, col6 = st.columns(3)
    with col4:
        step_min = st.selectbox("Time resolution", [15, 30, 60], index=1)
    with col5:
        min_gap_min = st.slider("Min gap (min)", 60, 360, 180, 30)
    with col6:
        fed = st.checkbox("Take with food", True)
    
    # Run optimization
    if st.button("🔍 Optimize Schedule", type="primary", use_container_width=True):
        with st.spinner("Finding optimal schedule..."):
            # Simple greedy optimization
            best_doses = []
            best_score = -float('inf')
            
            # Try different combinations
            for n_doses in [1, 2, 3]:
                if n_doses * 10 > daily_limit:
                    continue
                    
                # Generate time candidates
                time_candidates = []
                for h in range(max(0, target_start - 2), min(24, target_end + 2)):
                    for m in range(0, 60, step_min):
                        time_candidates.append(f"{h:02d}:{m:02d}")
                
                # Try combinations
                if n_doses == 1:
                    for t1 in time_candidates:
                        for mg1 in [10, 20, 30, 40]:
                            if mg1 > daily_limit:
                                continue
                            doses = [{"time_str": t1, "mg": mg1, "fed": fed, "type": "IR+ER"}]
                            t_axis = np.linspace(0, duration_h, int(duration_h * 60))
                            total, _ = simulate_total(t_axis, doses, start_hour)
                            score = objective(total, t_axis, target_start, target_end, 
                                           lambda_out, lambda_rough, lambda_peak, start_hour)
                            if score > best_score:
                                best_score = score
                                best_doses = doses
                
                elif n_doses == 2:
                    # Sample fewer combinations for speed
                    sample_times = time_candidates[::2]  # Every other time
                    for i, t1 in enumerate(sample_times[:10]):
                        for t2 in sample_times[i+4:i+14]:  # Ensure gap
                            for mg1 in [10, 20]:
                                for mg2 in [10, 20]:
                                    if mg1 + mg2 > daily_limit:
                                        continue
                                    doses = [
                                        {"time_str": t1, "mg": mg1, "fed": fed, "type": "IR+ER"},
                                        {"time_str": t2, "mg": mg2, "fed": fed, "type": "IR+ER"}
                                    ]
                                    t_axis = np.linspace(0, duration_h, int(duration_h * 60))
                                    total, _ = simulate_total(t_axis, doses, start_hour)
                                    score = objective(total, t_axis, target_start, target_end,
                                                   lambda_out, lambda_rough, lambda_peak, start_hour)
                                    if score > best_score:
                                   
