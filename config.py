"""Project-wide configuration constants for the microgrid optimizer."""

# Battery and system constants
FIXED_CONSTANTS = {
    "Max Battery Capacity": 500,  # kWh
    "Min Battery Capacity": 100,  # kWh
    "Initial State of Charge": 250,  # kWh
    "Max Charge and Discharge Rate": 125,  # kW
    "Charging Efficiency": 0.95,  # decimal
    "Discharging Efficiency": 0.95,  # decimal
    "Degradation Cost": 0.025,  # $/kWh
    "Generator Capacity": 100,  # kW
    "Generator Rate": 0.32,  # $/kWh
}

# Default solver preference order
DEFAULT_SOLVER = "cbc"
FALLBACK_SOLVER = "cbc"  # update if adding more options later
