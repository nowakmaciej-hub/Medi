"""
Advanced Pharmacokinetic Modeling & Optimization Tool
======================================================
Supports: Methylphenidate (Medikinet CR) & Lisdexamfetamine (Vyvanse)
With advanced PK models, optimizer, and beautiful UI

For educational purposes only - NOT medical advice
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as mpatches

# ===== Configuration =====
VERSION = "v2.0.0-alpha"
MAX_DOSES = 4  # Increased from 3 for more flexibility

# ===== Custom Styling =====
def inject_custom_css():
    """Inject beautiful custom CSS for modern UI"""
    st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }

    /* Card styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stMetric label {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    .stMetric .metric-value {
        color: white !important;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 10px;
        font-weight: 600;
    }

    /* Success boxes */
    .element-container .stSuccess {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
    }

    /* Warning boxes */
    .element-container .stWarning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== Data Classes =====
@dataclass
class DrugParameters:
    """Pharmacokinetic parameters for a drug"""
    name: str
    # Absorption
    ka_ir: float  # 1/h - IR absorption rate
    ka_er: float  # 1/h - ER absorption rate (if applicable)
    tlag_ir: float  # h - IR lag time
    tlag_er: float  # h - ER lag time
    # Distribution & Elimination
    ke: float  # 1/h - elimination rate constant
    vd: float  # L/kg - volume of distribution
    # Bioavailability
    f_ir: float  # IR bioavailability
    f_er: float  # ER bioavailability
    # Formulation
    ir_fraction: float  # Fraction of IR (0.5 for 50:50 formulation)
    # Conversion (for prodrugs)
    is_prodrug: bool = False
    conversion_rate: Optional[float] = None  # 1/h - prodrug to active conversion

@dataclass
class PersonParameters:
    """Individual person parameters affecting drug response"""
    body_weight: float = 70.0  # kg
    sleep_quality: float = 1.0  # 0.5-1.5 multiplier
    metabolism_rate: float = 1.0  # 0.5-1.5 multiplier
    effectiveness_coefficient: float = 1.0  # 0.5-1.5 response sensitivity
    tolerance_level: float = 0.0  # 0-0.3 - reduces effectiveness over time

# ===== Drug Definitions =====
def get_drug_params(drug_name: str, fed: bool = True) -> DrugParameters:
    """Get pharmacokinetic parameters for specified drug"""

    if drug_name == "Methylphenidate CR":
        # Medikinet CR - 50:50 IR:ER formulation
        # Based on literature: t½ ≈ 2-3h
        if fed:
            return DrugParameters(
                name="Methylphenidate CR",
                ka_ir=1.5, ka_er=0.4,
                tlag_ir=0.5, tlag_er=2.5,
                ke=0.3466,  # t½ = 2.0h
                vd=3.0,  # L/kg (estimated)
                f_ir=0.85, f_er=0.70,
                ir_fraction=0.5,
                is_prodrug=False
            )
        else:
            return DrugParameters(
                name="Methylphenidate CR",
                ka_ir=2.5, ka_er=0.6,
                tlag_ir=0.0, tlag_er=1.0,
                ke=0.3466,
                vd=3.0,
                f_ir=0.75, f_er=0.60,
                ir_fraction=0.5,
                is_prodrug=False
            )

    elif drug_name == "Lisdexamfetamine":
        # Lisdexamfetamine (Vyvanse) - prodrug converted to d-amphetamine
        # Slower onset, smoother profile
        # Conversion t½ ≈ 1h, d-amphetamine t½ ≈ 10h
        if fed:
            return DrugParameters(
                name="Lisdexamfetamine",
                ka_ir=0.8, ka_er=0.0,  # No ER formulation
                tlag_ir=0.5, tlag_er=0.0,
                ke=0.0693,  # d-amphetamine t½ = 10h
                vd=4.5,  # L/kg
                f_ir=0.95, f_er=0.0,
                ir_fraction=1.0,  # All IR (but prodrug conversion slows it)
                is_prodrug=True,
                conversion_rate=0.8  # Conversion t½ ≈ 0.87h
            )
        else:
            return DrugParameters(
                name="Lisdexamfetamine",
                ka_ir=1.2, ka_er=0.0,
                tlag_ir=0.2, tlag_er=0.0,
                ke=0.0693,
                vd=4.5,
                f_ir=0.96, f_er=0.0,
                ir_fraction=1.0,
                is_prodrug=True,
                conversion_rate=1.0
            )

    elif drug_name == "Dexamfetamine IR":
        # Immediate release d-amphetamine (for comparison)
        if fed:
            return DrugParameters(
                name="Dexamfetamine IR",
                ka_ir=1.5, ka_er=0.0,
                tlag_ir=0.3, tlag_er=0.0,
                ke=0.0693,  # t½ = 10h
                vd=4.0,
                f_ir=0.90, f_er=0.0,
                ir_fraction=1.0,
                is_prodrug=False
            )
        else:
            return DrugParameters(
                name="Dexamfetamine IR",
                ka_ir=2.0, ka_er=0.0,
                tlag_ir=0.1, tlag_er=0.0,
                ke=0.0693,
                vd=4.0,
                f_ir=0.88, f_er=0.0,
                ir_fraction=1.0,
                is_prodrug=False
            )

    else:
        raise ValueError(f"Unknown drug: {drug_name}")

# ===== Core PK Models =====
def bateman_equation(t: np.ndarray, t0: float, ka: float, ke: float,
                     dose: float, f: float, vd: float) -> np.ndarray:
    """
    One-compartment model with first-order absorption and elimination.
    Returns concentration in ng/mL (or equivalent units)

    C(t) = (F * Dose / Vd) * (ka / (ka - ke)) * (e^(-ke*t) - e^(-ka*t))
    """
    dt = t - t0
    y = np.zeros_like(t)
    mask = dt > 0

    if not np.any(mask):
        return y

    dtm = dt[mask]

    # Handle ka ≈ ke case (L'Hôpital's rule)
    if abs(ka - ke) < 0.01:
        y[mask] = (f * dose / vd) * ka * dtm * np.exp(-ke * dtm)
    else:
        y[mask] = (f * dose / vd) * (ka / (ka - ke)) * (
            np.exp(-ke * dtm) - np.exp(-ka * dtm)
        )

    return y

def prodrug_model(t: np.ndarray, t0: float, ka: float, k_conv: float, ke: float,
                  dose: float, f: float, vd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-step model: Prodrug -> Active Drug -> Elimination

    Returns: (prodrug_concentration, active_drug_concentration)
    """
    dt = t - t0
    prodrug = np.zeros_like(t)
    active = np.zeros_like(t)
    mask = dt > 0

    if not np.any(mask):
        return prodrug, active

    dtm = dt[mask]

    # Prodrug concentration (absorption and conversion)
    if abs(ka - k_conv) < 0.01:
        prodrug[mask] = (f * dose / vd) * ka * dtm * np.exp(-k_conv * dtm)
    else:
        prodrug[mask] = (f * dose / vd) * (ka / (ka - k_conv)) * (
            np.exp(-k_conv * dtm) - np.exp(-ka * dtm)
        )

    # Active drug concentration (from prodrug conversion)
    # This is a two-compartment cascade: absorption -> prodrug -> active -> elimination
    alpha = ka
    beta = k_conv
    gamma = ke

    # Simplified analytical solution for linear cascade
    C1 = ka * k_conv / ((ka - k_conv) * (ka - ke))
    C2 = ka * k_conv / ((k_conv - ka) * (k_conv - ke))
    C3 = ka * k_conv / ((ke - ka) * (ke - k_conv))

    active[mask] = (f * dose / vd) * (
        C1 * np.exp(-ke * dtm) +
        C2 * np.exp(-k_conv * dtm) +
        C3 * np.exp(-ka * dtm)
    )

    return prodrug, active

def simulate_dose_profile(t: np.ndarray, dose_mg: float, t0: float,
                         drug_params: DrugParameters,
                         person_params: PersonParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate concentration profile for a single dose with person-specific parameters.

    Returns: (ir_component, er_component, total_concentration)
    """
    # Apply metabolism rate to elimination
    ke_adjusted = drug_params.ke * person_params.metabolism_rate

    # Apply sleep quality to bioavailability (better sleep = better absorption)
    f_ir = drug_params.f_ir * person_params.sleep_quality
    f_er = drug_params.f_er * person_params.sleep_quality

    # Calculate dose split
    ir_dose = dose_mg * drug_params.ir_fraction
    er_dose = dose_mg * (1 - drug_params.ir_fraction)

    # Volume of distribution adjusted for body weight
    vd_total = drug_params.vd * person_params.body_weight

    ir_component = np.zeros_like(t)
    er_component = np.zeros_like(t)

    # IR component
    if ir_dose > 0:
        if drug_params.is_prodrug:
            _, ir_component = prodrug_model(
                t, t0 + drug_params.tlag_ir,
                drug_params.ka_ir, drug_params.conversion_rate,
                ke_adjusted, ir_dose, f_ir, vd_total
            )
        else:
            ir_component = bateman_equation(
                t, t0 + drug_params.tlag_ir,
                drug_params.ka_ir, ke_adjusted,
                ir_dose, f_ir, vd_total
            )

    # ER component
    if er_dose > 0:
        er_component = bateman_equation(
            t, t0 + drug_params.tlag_er,
            drug_params.ka_er, ke_adjusted,
            er_dose, f_er, vd_total
        )

    total = ir_component + er_component

    # Apply effectiveness coefficient to final concentration
    total = total * person_params.effectiveness_coefficient
    ir_component = ir_component * person_params.effectiveness_coefficient
    er_component = er_component * person_params.effectiveness_coefficient

    # Apply tolerance (reduces effect over time during the day)
    if person_params.tolerance_level > 0:
        decay = np.exp(-person_params.tolerance_level * t / 24.0)
        total = total * decay
        ir_component = ir_component * decay
        er_component = er_component * decay

    return ir_component, er_component, total

# ===== Simulation Functions =====
def simulate_total_concentration(t_axis: np.ndarray, doses: List[Dict],
                                start_hour: float, drug_name: str,
                                person_params: PersonParameters) -> Tuple[np.ndarray, List]:
    """Simulate total concentration from multiple doses"""
    if len(t_axis) == 0:
        return np.array([]), []

    total = np.zeros_like(t_axis)
    components = []

    if not doses:
        return total, components

    for d in doses:
        # Parse time
        hh, mm = map(int, d["time_str"].split(":"))
        t0 = (hh + mm/60) - start_hour

        # Get drug parameters
        drug_params = get_drug_params(drug_name, d.get("fed", True))

        # Simulate this dose
        ir, er, dose_total = simulate_dose_profile(
            t_axis, d["mg"], t0, drug_params, person_params
        )

        total += dose_total

        # Add to components list
        if drug_params.is_prodrug:
            label = f"{drug_name} {d['mg']}mg @ {d['time_str']}"
            if d.get("fed"):
                label += " (fed)"
            components.append((label, dose_total))
        else:
            if drug_params.ir_fraction == 1.0:
                # All IR
                label = f"{drug_name} {d['mg']}mg @ {d['time_str']}"
                if d.get("fed"):
                    label += " (fed)"
                components.append((label, ir))
            else:
                # IR + ER
                components.append((f"IR {d['mg']*drug_params.ir_fraction:.0f}mg @ {d['time_str']}", ir))
                components.append((f"ER {d['mg']*(1-drug_params.ir_fraction):.0f}mg @ {d['time_str']}", er))

    return total, components

# ===== Utility Functions =====
def parse_time_to_hours(t_str: str, start_hour: float) -> float:
    """Parse time string to hours relative to start_hour"""
    try:
        hh, mm = map(int, t_str.split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"Invalid time: {t_str}")
        return (hh + mm/60) - start_hour
    except (ValueError, AttributeError) as e:
        st.error(f"Error parsing time '{t_str}': {e}")
        return 0.0

def compute_metrics(total_curve: np.ndarray, t_axis: np.ndarray, start_hour: float) -> Dict:
    """Compute pharmacokinetic metrics"""
    if len(total_curve) == 0 or np.max(total_curve) < 1e-9:
        return {}

    peak_conc = np.max(total_curve)
    peak_idx = np.argmax(total_curve)
    peak_time = start_hour + t_axis[peak_idx]

    # AUC (area under curve)
    auc = float(np.trapz(total_curve, t_axis))

    # Time above thresholds
    threshold_20 = 0.2 * peak_conc
    threshold_50 = 0.5 * peak_conc

    above_20 = np.where(total_curve > threshold_20)[0]
    above_50 = np.where(total_curve > threshold_50)[0]

    duration_20 = (t_axis[above_20[-1]] - t_axis[above_20[0]]) if len(above_20) > 0 else 0.0
    duration_50 = (t_axis[above_50[-1]] - t_axis[above_50[0]]) if len(above_50) > 0 else 0.0

    return {
        "peak_conc": peak_conc,
        "peak_time": peak_time,
        "auc": auc,
        "duration_20": duration_20,
        "duration_50": duration_50
    }

# ===== Improved Optimizer =====
def objective_function(params: np.ndarray, t_axis: np.ndarray, start_hour: float,
                      drug_name: str, person_params: PersonParameters,
                      target_start: float, target_end: float,
                      lambda_out: float, lambda_smooth: float, lambda_peak: float,
                      fed: bool, max_daily: int) -> float:
    """
    Objective function for optimization.
    params: flattened array [time1, dose1, time2, dose2, ...]

    Maximizes coverage in target window while minimizing:
    - Coverage outside window
    - Fluctuations (smoothness)
    - Peak concentration (to avoid side effects)
    """
    n_doses = len(params) // 2

    if n_doses == 0:
        return 1e9  # Penalty for no doses

    # Parse doses
    doses = []
    total_mg = 0
    for i in range(n_doses):
        time_hours = params[2*i] % 24
        dose_mg = params[2*i + 1]

        if dose_mg < 5:  # Minimum dose
            continue

        doses.append({
            "time_str": f"{int(time_hours):02d}:{int((time_hours % 1) * 60):02d}",
            "mg": dose_mg,
            "fed": fed
        })
        total_mg += dose_mg

    # Penalty for exceeding daily limit
    if total_mg > max_daily:
        return 1e9

    if len(doses) == 0:
        return 1e9

    # Simulate
    total_curve, _ = simulate_total_concentration(
        t_axis, doses, start_hour, drug_name, person_params
    )

    # Compute objective components
    hours_of_day = start_hour + t_axis

    # Target window mask
    if target_end >= target_start:
        in_window = (hours_of_day >= target_start) & (hours_of_day <= target_end)
    else:
        in_window = (hours_of_day >= target_start) | (hours_of_day <= target_end)

    # Coverage in and out of window
    coverage_in = np.trapz(total_curve[in_window], t_axis[in_window]) if np.any(in_window) else 0
    coverage_out = np.trapz(total_curve[~in_window], t_axis[~in_window]) if np.any(~in_window) else 0

    # Smoothness (penalize rapid changes)
    gradient = np.gradient(total_curve, t_axis)
    roughness = np.trapz(gradient**2, t_axis)

    # Peak concentration
    peak = np.max(total_curve)

    # Objective: maximize coverage in window, minimize out-of-window, smoothness, and peak
    score = coverage_in - lambda_out * coverage_out - lambda_smooth * roughness - lambda_peak * peak

    # We minimize, so negate
    return -score

def optimize_schedule_advanced(start_hour: float, duration_h: float, max_daily: int,
                               drug_name: str, person_params: PersonParameters,
                               target_start: float, target_end: float,
                               lambda_out: float, lambda_smooth: float, lambda_peak: float,
                               fed: bool, n_doses: int = 3) -> List[Dict]:
    """
    Advanced optimizer using differential evolution.
    Much better than greedy approach!
    """
    t_axis = np.linspace(0, duration_h, int(duration_h * 60))

    # Bounds: [time1, dose1, time2, dose2, ...]
    # Times: within start_hour to start_hour+duration_h
    # Doses: 5-40 mg
    bounds = []
    for i in range(n_doses):
        bounds.append((start_hour, start_hour + duration_h))  # Time
        bounds.append((5, min(40, max_daily)))  # Dose

    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(t_axis, start_hour, drug_name, person_params, target_start, target_end,
              lambda_out, lambda_smooth, lambda_peak, fed, max_daily),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        workers=1,
        updating='deferred',
        polish=True
    )

    # Parse results
    params = result.x
    n_result_doses = len(params) // 2

    doses = []
    for i in range(n_result_doses):
        time_hours = params[2*i] % 24
        dose_mg = round(params[2*i + 1] / 5) * 5  # Round to nearest 5mg

        if dose_mg >= 5:
            doses.append({
                "time_str": f"{int(time_hours):02d}:{int((time_hours % 1) * 60):02d}",
                "mg": int(dose_mg),
                "fed": fed
            })

    # Sort by time
    doses.sort(key=lambda d: parse_time_to_hours(d["time_str"], 0))

    # Limit total to max_daily
    total = sum(d["mg"] for d in doses)
    while total > max_daily and doses:
        # Remove smallest dose
        min_dose_idx = min(range(len(doses)), key=lambda i: doses[i]["mg"])
        removed = doses.pop(min_dose_idx)
        total -= removed["mg"]

    return doses

# ===== App Configuration =====
st.set_page_config(
    page_title="Advanced PK Optimizer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

# ===== Header =====
st.markdown(f"""
<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.9); border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='margin:0; color: #667eea;'>💊 Advanced Pharmacokinetic Optimizer</h1>
    <p style='margin: 10px 0 0 0; color: #764ba2; font-size: 1.2em;'>
        Methylphenidate & Lisdexamfetamine Modeling Tool
    </p>
    <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9em;'>
        Version {VERSION} | For educational purposes only - NOT medical advice
    </p>
</div>
""", unsafe_allow_html=True)

# ===== Sidebar Controls =====
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Mode selection
    mode = st.selectbox(
        "Mode",
        ["🎮 Simulator", "🚀 Advanced Optimizer"],
        index=0
    )

    st.markdown("---")

    # Drug selection
    st.markdown("### 💊 Drug Selection")
    drug_name = st.selectbox(
        "Active Substance",
        ["Methylphenidate CR", "Lisdexamfetamine", "Dexamfetamine IR"],
        index=0,
        help="Lisdexamfetamine is a prodrug that converts to dexamfetamine"
    )

    # Drug info
    with st.expander("ℹ️ Drug Information"):
        if drug_name == "Methylphenidate CR":
            st.markdown("""
            **Methylphenidate CR (Medikinet)**
            - 50:50 IR:ER formulation
            - Half-life: ~2 hours
            - Duration: 6-8 hours
            - Peak: 1.5-3 hours
            """)
        elif drug_name == "Lisdexamfetamine":
            st.markdown("""
            **Lisdexamfetamine (Vyvanse)**
            - Prodrug → dexamfetamine
            - Half-life: ~10 hours (active)
            - Duration: 10-14 hours
            - Smoother profile, less abuse potential
            - Conversion time: ~1 hour
            """)
        else:
            st.markdown("""
            **Dexamfetamine IR**
            - Immediate release
            - Half-life: ~10 hours
            - Duration: 4-6 hours peak effect
            - Faster onset than lisdexamfetamine
            """)

    st.markdown("---")

    # Person parameters
    st.markdown("### 👤 Personal Factors")

    body_weight = st.slider(
        "Body Weight (kg)",
        40, 120, 70, 5,
        help="Affects volume of distribution"
    )

    sleep_quality = st.slider(
        "Sleep Quality Factor",
        0.5, 1.5, 1.0, 0.1,
        help="Poor sleep (0.5) reduces bioavailability, good sleep (1.5) improves it"
    )

    metabolism_rate = st.slider(
        "Metabolism Rate",
        0.5, 1.5, 1.0, 0.1,
        help="Slow (0.5) = longer duration, Fast (1.5) = shorter duration"
    )

    effectiveness = st.slider(
        "Effectiveness Coefficient",
        0.5, 1.5, 1.0, 0.1,
        help="Individual sensitivity to medication effects"
    )

    tolerance = st.slider(
        "Tolerance Level",
        0.0, 0.3, 0.0, 0.05,
        help="Reduction in effect over the course of the day"
    )

    # Create person params
    person_params = PersonParameters(
        body_weight=body_weight,
        sleep_quality=sleep_quality,
        metabolism_rate=metabolism_rate,
        effectiveness_coefficient=effectiveness,
        tolerance_level=tolerance
    )

    st.markdown("---")

    # Time settings
    st.markdown("### ⏰ Time Settings")
    start_hour = st.number_input("Day Start Hour", 0, 23, 6)
    duration_h = st.slider("Simulation Duration (hours)", 8, 24, 18)

    st.markdown("---")

    # Chart settings
    st.markdown("### 📊 Chart Settings")
    chart_height = st.slider("Chart Height (px)", 200, 800, 400, 50)
    show_components = st.checkbox("Show IR/ER Components", False)
    show_therapeutic = st.checkbox("Show Therapeutic Range", True)

# ===== Main App =====
if "🎮 Simulator" in mode:
    st.markdown("## 🎮 Dose Simulator")

    # Quick presets
    with st.expander("⚡ Quick Presets"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Standard Day"):
                if drug_name == "Methylphenidate CR":
                    st.session_state.sim_doses = [
                        {"time_str": "07:00", "mg": 20, "fed": True},
                        {"time_str": "12:00", "mg": 10, "fed": True},
                        {"time_str": "16:00", "mg": 10, "fed": False}
                    ]
                else:
                    st.session_state.sim_doses = [
                        {"time_str": "07:00", "mg": 40, "fed": True}
                    ]

        with col2:
            if st.button("Extended Day"):
                if drug_name == "Methylphenidate CR":
                    st.session_state.sim_doses = [
                        {"time_str": "07:00", "mg": 20, "fed": True},
                        {"time_str": "13:00", "mg": 20, "fed": True}
                    ]
                else:
                    st.session_state.sim_doses = [
                        {"time_str": "07:00", "mg": 60, "fed": True}
                    ]

        with col3:
            if st.button("Low Dose"):
                if drug_name == "Methylphenidate CR":
                    st.session_state.sim_doses = [
                        {"time_str": "08:00", "mg": 10, "fed": True}
                    ]
                else:
                    st.session_state.sim_doses = [
                        {"time_str": "08:00", "mg": 20, "fed": True}
                    ]

        with col4:
            if st.button("Clear All"):
                st.session_state.sim_doses = []

    # Initialize doses
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []

    # Add dose interface
    with st.expander("➕ Add New Dose"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            h = st.number_input("Hour", 0, 23, 8, key="sim_h")
        with c2:
            m = st.number_input("Minute", 0, 59, 0, 5, key="sim_m")
        with c3:
            if drug_name == "Methylphenidate CR":
                mg = st.selectbox("Dose (mg)", [5, 10, 15, 20, 30, 40], index=3, key="sim_mg")
            else:
                mg = st.selectbox("Dose (mg)", [20, 30, 40, 50, 60, 70], index=2, key="sim_mg")
        with c4:
            fed = st.selectbox("Food Status", ["Fasted", "Fed"], index=1, key="sim_fed")

        if st.button("➕ Add Dose", type="primary"):
            st.session_state.sim_doses.append({
                "time_str": f"{int(h):02d}:{int(m):02d}",
                "mg": int(mg),
                "fed": fed == "Fed"
            })
            st.rerun()

    # Show current doses
    if st.session_state.sim_doses:
        st.markdown("### 📋 Current Schedule")

        for i, d in enumerate(st.session_state.sim_doses):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.metric(f"Dose {i+1}", f"⏰ {d['time_str']}")
            with col2:
                st.metric("Amount", f"💊 {d['mg']} mg")
            with col3:
                st.metric("Food", "🍽️ Fed" if d['fed'] else "🚫 Fasted")
            with col4:
                if st.button("🗑️", key=f"sim_del_{i}"):
                    st.session_state.sim_doses.pop(i)
                    st.rerun()

        # Calculate and display
        t_axis = np.linspace(0, duration_h, int(duration_h * 60))
        total_curve, components = simulate_total_concentration(
            t_axis, st.session_state.sim_doses, start_hour, drug_name, person_params
        )

        # Metrics
        metrics = compute_metrics(total_curve, t_axis, start_hour)

        if metrics:
            st.markdown("### 📊 Pharmacokinetic Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Peak Concentration",
                    f"{metrics['peak_conc']:.2f} ng/mL"
                )
            with col2:
                peak_h = int(metrics['peak_time'])
                peak_m = int((metrics['peak_time'] % 1) * 60)
                st.metric(
                    "Time to Peak",
                    f"{peak_h:02d}:{peak_m:02d}"
                )
            with col3:
                st.metric(
                    "Duration >20% Peak",
                    f"{metrics['duration_20']:.1f} hours"
                )
            with col4:
                st.metric(
                    "Total Daily Dose",
                    f"{sum(d['mg'] for d in st.session_state.sim_doses)} mg"
                )

        # Plot
        fig, ax = plt.subplots(figsize=(12, chart_height/100), dpi=100)

        # Main curve
        hours_of_day = start_hour + t_axis
        ax.plot(hours_of_day, total_curve, linewidth=3, color='#667eea', label='Total Concentration')

        # Components
        if show_components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
            for (label, curve), color in zip(components, colors):
                ax.plot(hours_of_day, curve, '--', linewidth=2, alpha=0.7, label=label, color=color)

        # Therapeutic range
        if show_therapeutic and len(total_curve) > 0 and np.max(total_curve) > 0:
            peak = np.max(total_curve)
            ax.axhspan(0.2*peak, 0.8*peak, alpha=0.15, color='green', label='Therapeutic Range (20-80%)')

        # Dose markers
        for d in st.session_state.sim_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            dose_time = hh + mm/60
            ax.axvline(dose_time, color='red', linestyle=':', alpha=0.5, linewidth=2)
            # Add dose annotation
            ax.annotate(f"{d['mg']}mg", xy=(dose_time, ax.get_ylim()[1]*0.95),
                       ha='center', fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Plasma Concentration (ng/mL)', fontsize=12, fontweight='bold')
        ax.set_title(f'{drug_name} - Concentration Profile', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        # Format x-axis
        ax.set_xlim(start_hour, start_hour + duration_h)
        ax.set_xticks(range(int(start_hour), int(start_hour + duration_h) + 1, 2))
        ax.set_xticklabels([f"{h%24:02d}:00" for h in range(int(start_hour), int(start_hour + duration_h) + 1, 2)])

        plt.tight_layout()
        st.pyplot(fig)

        # Export
        if st.button("📥 Export Schedule as JSON"):
            export_data = {
                "drug": drug_name,
                "doses": st.session_state.sim_doses,
                "person_params": {
                    "body_weight": body_weight,
                    "sleep_quality": sleep_quality,
                    "metabolism_rate": metabolism_rate,
                    "effectiveness": effectiveness,
                    "tolerance": tolerance
                },
                "metrics": metrics
            }
            st.json(export_data)

    else:
        st.info("👆 Add doses above to see the concentration profile")

else:  # Optimizer mode
    st.markdown("## 🚀 Advanced Schedule Optimizer")

    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin:0; color: #2d3748;'>
            <strong>This optimizer uses differential evolution</strong> - a powerful global optimization algorithm
            that explores the solution space intelligently to find the best dose schedule for your needs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Optimization parameters
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Target Window")
        target_start = st.number_input("Start Hour", 0, 23, 8, help="When coverage should begin")
        target_end = st.number_input("End Hour", 0, 23, 20, help="When coverage should end")

        st.markdown("### 💊 Dose Constraints")
        if drug_name == "Methylphenidate CR":
            max_daily = st.number_input("Max Daily Dose (mg)", 10, 80, 40, 10)
        else:
            max_daily = st.number_input("Max Daily Dose (mg)", 20, 100, 60, 10)

        n_doses = st.slider("Number of Doses", 1, MAX_DOSES, 2)
        fed_opt = st.checkbox("Assume doses with food", True)

    with col2:
        st.markdown("### ⚖️ Optimization Weights")

        lambda_out = st.slider(
            "Out-of-Window Penalty",
            0.0, 10.0, 2.0, 0.5,
            help="Penalize concentration outside target window"
        )

        lambda_smooth = st.slider(
            "Smoothness Weight",
            0.0, 5.0, 0.5, 0.1,
            help="Prefer smoother concentration profiles"
        )

        lambda_peak = st.slider(
            "Peak Penalty",
            0.0, 5.0, 1.0, 0.5,
            help="Limit maximum concentration (reduce side effects)"
        )

    # Run optimization
    if st.button("🚀 Optimize Schedule", type="primary"):
        with st.spinner("🧬 Running differential evolution optimizer... This may take 30-60 seconds..."):
            try:
                optimized_doses = optimize_schedule_advanced(
                    start_hour, duration_h, max_daily,
                    drug_name, person_params,
                    target_start, target_end,
                    lambda_out, lambda_smooth, lambda_peak,
                    fed_opt, n_doses
                )

                # Store in session state
                st.session_state.opt_doses = optimized_doses
                st.success(f"✅ Optimization complete! Found {len(optimized_doses)} dose(s)")

            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.info("Try adjusting the parameters or constraints")

    # Display optimized results
    if "opt_doses" in st.session_state and st.session_state.opt_doses:
        opt_doses = st.session_state.opt_doses

        st.markdown("### 📋 Optimized Schedule")

        # Display as cards
        cols = st.columns(len(opt_doses))
        for i, (col, d) in enumerate(zip(cols, opt_doses)):
            with col:
                st.metric(
                    f"Dose {i+1}",
                    f"{d['mg']} mg",
                    f"@ {d['time_str']}"
                )

        total_mg = sum(d['mg'] for d in opt_doses)
        st.success(f"**Total Daily Dose:** {total_mg} mg / {max_daily} mg")

        # Simulate and plot
        t_axis = np.linspace(0, duration_h, int(duration_h * 60))
        total_curve, components = simulate_total_concentration(
            t_axis, opt_doses, start_hour, drug_name, person_params
        )

        # Metrics
        metrics = compute_metrics(total_curve, t_axis, start_hour)

        if metrics:
            st.markdown("### 📊 Optimized Profile Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Peak Conc.", f"{metrics['peak_conc']:.2f} ng/mL")
            with col2:
                peak_h = int(metrics['peak_time'])
                peak_m = int((metrics['peak_time'] % 1) * 60)
                st.metric("Peak Time", f"{peak_h:02d}:{peak_m:02d}")
            with col3:
                st.metric("Duration >50%", f"{metrics['duration_50']:.1f} h")
            with col4:
                st.metric("AUC", f"{metrics['auc']:.1f}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, chart_height/100), dpi=100)

        hours_of_day = start_hour + t_axis

        # Target window shading
        if target_end >= target_start:
            ax.axvspan(target_start, target_end, alpha=0.1, color='green', label='Target Window')
        else:
            ax.axvspan(target_start, 24, alpha=0.1, color='green')
            ax.axvspan(0, target_end, alpha=0.1, color='green')

        # Main curve
        ax.plot(hours_of_day, total_curve, linewidth=3.5, color='#667eea',
               label='Optimized Concentration', zorder=5)

        # Components
        if show_components:
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(components)))
            for (label, curve), color in zip(components, colors):
                ax.plot(hours_of_day, curve, '--', linewidth=2, alpha=0.7, label=label, color=color)

        # Therapeutic range
        if show_therapeutic and len(total_curve) > 0 and np.max(total_curve) > 0:
            peak = np.max(total_curve)
            ax.axhspan(0.2*peak, 0.8*peak, alpha=0.15, color='blue', label='Therapeutic Range')

        # Dose markers
        for d in opt_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            dose_time = hh + mm/60
            ax.axvline(dose_time, color='red', linestyle='--', alpha=0.6, linewidth=2.5)
            # Annotation
            ax.annotate(f"💊 {d['mg']}mg", xy=(dose_time, ax.get_ylim()[1]*0.9),
                       ha='center', fontsize=11, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red'))

        ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
        ax.set_ylabel('Plasma Concentration (ng/mL)', fontsize=13, fontweight='bold')
        ax.set_title(f'Optimized {drug_name} Schedule', fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.legend(loc='best', fontsize=10, framealpha=0.95, shadow=True)

        # Format x-axis
        ax.set_xlim(start_hour, start_hour + duration_h)
        ax.set_xticks(range(int(start_hour), int(start_hour + duration_h) + 1, 2))
        ax.set_xticklabels([f"{h%24:02d}:00" for h in range(int(start_hour), int(start_hour + duration_h) + 1, 2)])

        plt.tight_layout()
        st.pyplot(fig)

        # Export
        if st.button("📥 Export Optimized Schedule"):
            export_data = {
                "drug": drug_name,
                "optimized_doses": opt_doses,
                "person_params": {
                    "body_weight": body_weight,
                    "sleep_quality": sleep_quality,
                    "metabolism_rate": metabolism_rate,
                    "effectiveness": effectiveness,
                    "tolerance": tolerance
                },
                "optimization_params": {
                    "target_start": target_start,
                    "target_end": target_end,
                    "max_daily": max_daily,
                    "lambda_out": lambda_out,
                    "lambda_smooth": lambda_smooth,
                    "lambda_peak": lambda_peak
                },
                "metrics": metrics
            }
            st.json(export_data)

    else:
        st.info("👆 Configure parameters above and click 'Optimize Schedule' to find the best dosing strategy")

# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
    <p><strong>⚠️ Disclaimer:</strong> This tool is for educational and research purposes only.</p>
    <p>Do NOT use this for medical decisions. Always consult with a qualified healthcare provider.</p>
    <p style='margin-top: 10px; font-size: 0.8em;'>
        Built with ❤️ using Streamlit | Version {VERSION}
    </p>
</div>
""".format(VERSION=VERSION), unsafe_allow_html=True)
