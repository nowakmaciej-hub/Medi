# Advanced Pharmacokinetic Modeling & Optimization Tool 💊

A beautiful, feature-rich Streamlit application for pharmacokinetic modeling of ADHD medications with advanced optimization algorithms.

## 🚀 Features

### Supported Medications
- **Methylphenidate CR (Medikinet)**: 50:50 IR:ER formulation
- **Lisdexamfetamine (Vyvanse)**: Prodrug with conversion modeling
- **Dexamfetamine IR**: Immediate release d-amphetamine

### Core Features
- **🎮 Interactive Simulator**: Visualize concentration profiles for custom dose schedules
- **🚀 Advanced Optimizer**: Differential evolution algorithm for optimal dose scheduling
- **👤 Personal Factors**: Body weight, sleep quality, metabolism rate, effectiveness, tolerance
- **📊 Beautiful UI**: Modern gradient design with custom styling
- **🧬 Accurate PK Models**: Bateman equations, prodrug conversion models, realistic parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
streamlit run medikinet_advanced_pk_optimizer.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Optimization Algorithm

The advanced optimizer uses **differential evolution** - a powerful global optimization technique that:
- Explores the entire solution space intelligently
- Avoids local optima
- Finds optimal dose timing and amounts
- Considers multiple constraints simultaneously

### Optimization Objectives
- ✅ Maximize coverage in target window
- ❌ Minimize out-of-window coverage
- 📈 Smooth concentration profile
- ⚠️ Limit peak concentration (reduce side effects)

## 🧬 Pharmacokinetic Models

### Bateman Equation (1-Compartment)
- First-order absorption and elimination
- Realistic lag times for IR and ER components
- Accounts for food effects on bioavailability

### Prodrug Conversion Model (Lisdexamfetamine)
- Two-stage kinetics: absorption → conversion → elimination
- Models enzymatic conversion to active metabolite
- Slower onset, smoother profile

### Personal Factors
- **Body Weight**: Affects volume of distribution (clinically established)
- **Sleep Quality**: Hypothetical modifier for subjective response (0.5-1.5x) - relationship not clinically established
- **Metabolism Rate**: Adjusts elimination rate (0.5-1.5x) - proxy for CYP enzyme activity
- **Effectiveness**: Individual response sensitivity (0.5-1.5x) - accounts for receptor sensitivity
- **Tolerance**: Progressive reduction in effect over the day - simplified acute tolerance model

## 📊 Metrics Calculated

- **Peak Concentration (Cmax)**: Maximum plasma concentration
- **Time to Peak (Tmax)**: When maximum concentration occurs
- **AUC**: Area under the curve (total drug exposure)
- **Duration >20% peak**: Time above 20% of peak concentration
- **Duration >50% peak**: Time above 50% of peak concentration

## 🎨 UI Features

- **Modern gradient design** with purple/blue color scheme
- **Responsive layout** with beautiful cards and metrics
- **Interactive controls** with sliders and toggles
- **Real-time visualization** with customizable charts
- **Export functionality** for schedules and parameters

## ⚠️ Important Disclaimer

**For educational and research purposes only - NOT medical advice**

This tool simulates pharmacokinetic profiles based on mathematical models and literature values. It should NEVER be used for medical decision-making. Always consult with a qualified healthcare provider for medication management.

## 🔬 Model Limitations

This tool uses simplified pharmacokinetic models. Users should be aware of the following limitations:

- **One-compartment model**: Does not capture tissue distribution dynamics
- **Population averages**: PK parameters are derived from literature averages, not individual measurements
- **No genetic factors**: Does not account for CYP2D6 polymorphisms or other pharmacogenomic variations
- **No drug interactions**: Does not model concurrent medications that may affect PK
- **Simplified tolerance**: Acute tolerance model is a basic exponential decay approximation
- **No food composition effects**: Only models fed/fasted states, not specific meal effects
- **Concentration values are relative**: Displayed values are model-derived and not absolute clinical measurements
- **Therapeutic range is illustrative**: The 20-80% range shown is not a clinically validated threshold

## 📝 Version History

- **v2.0.0-alpha**: Advanced optimizer with lisdexamfetamine support, beautiful UI, personal factors
- **v1.0.0**: Initial release with basic PK models

## License

Educational use only

---

Built with ❤️ using Python, Streamlit, NumPy, SciPy, and Matplotlib
