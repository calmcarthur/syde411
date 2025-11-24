"""Default 24-hour load, solar, and price profiles for the optimizer."""

DEFAULT_PROFILES = {
    # Load (kW) - Unchanged from original for system stability
    "L(t) (kW)": {
        1: 60, 2: 60, 3: 60, 4: 60, 5: 70, 6: 80, 7: 100, 8: 150, 9: 200, 10: 220,
        11: 240, 12: 250, 13: 252, 14: 255, 15: 257, 16: 255, 17: 252, 18: 250,
        19: 240, 20: 220, 21: 200, 22: 150, 23: 100, 24: 80,
    },
    # Available Solar (kW) - Unchanged from original
    "A(t) (kW)": {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 10, 6: 0, 7: 10, 8: 30, 9: 45, 10: 80,
        11: 100, 12: 120, 13: 125, 14: 130, 15: 100, 16: 85, 17: 60, 18: 45,
        19: 30, 20: 0, 21: 0, 22: 0, 23: 10, 24: 0,
    },
    # Grid Price ($/kWh) - More Sporadic and Challenging
    "P(t) ($/kWh)": {
        # Early Morning: Volatile (Charge/No-Charge ambiguity)
        1: 0.15, 2: 0.08, 3: 0.20, 4: 0.1, 5: 0.1, 6: 0.15,
        # Mid-Day: High prices persist, challenging solar use decisions
        7: 0.22, 8: 0.18, 9: 0.28, 10: 0.25,
        # Peak Load/Solar Overlap: Extreme volatility, forcing generator/battery decisions
        11: 0.35, 12: 0.15, 13: 0.40, 14: 0.25, 15: 0.30, 16: 0.18,
        # Evening Peak: Extreme spike when solar is gone
        17: 0.45, 18: 0.38, 19: 0.15, 20: 0.40,
        # Night: Low prices, then a late spike (challenging reusability constraint)
        21: 0.1, 22: 0.08, 23: 0.22, 24: 0.1,
    },
}

def get_default_profiles():
    """Return a shallow copy of the default profile dictionary for safe reuse."""
    return {k: v.copy() for k, v in DEFAULT_PROFILES.items()}
