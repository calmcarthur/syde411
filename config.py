"""Configuration constants and solver settings for the microgrid optimizer."""

# Solver Settings
DEFAULT_SOLVER = "cbc"  # Open-source MILP solver
FALLBACK_SOLVER = ["glpk", "highs"]  # Solvers to try if CBC isn't found

# Physical and Economic Constants
# These values match the hardcoded parameters in 'edited_optimizer.py'
FIXED_CONSTANTS = {
    # Battery Storage Parameters
    "Max Battery Capacity": 500.0,       # kWh (B_max)
    "Min Battery Capacity": 100.0,       # kWh (B_min)
    "Initial State of Charge": 250.0,    # kWh (SoC_initial)
    "Max Charge and Discharge Rate": 125.0, # kW (R_max)
    "Charging Efficiency": 0.95,         # (eta_ch)
    "Discharging Efficiency": 0.95,      # (eta_dis)
    "Degradation Cost": 0.025,           # $/kWh (alpha)
    "Min Charge/Discharge": 20.0,        # kW (P_min)

    # Generator Parameters
    "Generator Capacity": 118.0,         # kW (C_gen)
    "Generator Rate": 0.32,              # $/kWh fuel cost (G_rate)
    "Generator Startup Cost": 300.0,     # $ per start (C_start)
    "Generator Ramp Limit": 8.0,         # kW/hr (Ramp_limit)
    "Generator Min Up Time": 6.0,        # hours (MinUp_hours)

    # Grid Parameters
    "Grid Capacity": 60.0,               # kW import limit (Grid_cap)
}