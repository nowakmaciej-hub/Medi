# Medikinet CR - Pharmacokinetic Modeling Tool

A Streamlit-based interactive pharmacokinetic (PK) modeling application for Medikinet CR (methylphenidate controlled release).

## Features

- **Multiple PK Models**: Choose between Gaussian, 1-Compartment (Bateman), or 2-Compartment models
- **Simulator Mode**: Visualize concentration curves for custom dose schedules
- **Optimizer Mode**: Automatically optimize dose schedules based on target coverage windows
- **Realistic Parameters**: Based on methylphenidate pharmacokinetic literature

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run medikinet_all_in_one_with_pk.py
```

The app will open in your browser at `http://localhost:8501`

## Models

### 2-Compartment Model (Most Realistic)
- Central and peripheral compartments
- Weibull function for extended-release kinetics
- Optional population variability (20% CV)

### 1-Compartment PK (Bateman)
- First-order absorption and elimination
- Simple lag times for IR and ER components

### Gaussian Model (Simplified)
- Fast computation
- Good for quick approximations

## Important Notes

**For educational purposes only - NOT medical advice**

This tool simulates pharmacokinetic profiles based on simplified models and should not be used for medical decision-making.

## License

Educational use only
