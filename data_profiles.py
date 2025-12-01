"""Hourly data profiles for load, solar, and grid prices."""

def get_default_profiles() -> dict[str, dict[int, float]]:
    """
    Returns dictionary of hourly data profiles (1-24).
    Data extracted from original edited_optimizer.py.
    """
    
    # Load Profile (L) in kW
    L_data = {
        1: 60,  2: 60,  3: 60,  4: 60,
        5: 70,  6: 80,  7: 100, 8: 150,
        9: 200, 10: 220, 11: 240, 12: 250,
        13: 252, 14: 255, 15: 257, 16: 255,
        17: 252, 18: 250, 19: 240, 20: 220,
        21: 200, 22: 150, 23: 100, 24: 80,
    }

    # Solar Availability (A) in kW
    A_data = {
        1: 0,   2: 0,   3: 0,   4: 0,
        5: 0,   6: 0,   7: 10,  8: 30,
        9: 45, 10: 80, 11: 100, 12: 120,
        13: 125, 14: 130, 15: 100, 16: 85,
        17: 60, 18: 45, 19: 30, 20: 0,
        21: 0,  22: 0,  23: 0,  24: 0,
    }

    # Grid Price (P) in $/kWh
    P_data = {
        1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10,
        5: 0.10, 6: 0.10, 7: 0.18, 8: 0.18,
        9: 0.18, 10: 0.18, 11: 0.25, 12: 0.25,
        13: 0.25, 14: 0.25, 15: 0.25, 16: 0.25,
        17: 0.35, 18: 0.35, 19: 0.35, 20: 0.35,
        21: 0.18, 22: 0.18, 23: 0.10, 24: 0.10,
    }

    return {
        "L(t) (kW)": L_data,
        "A(t) (kW)": A_data,
        "P(t) ($/kWh)": P_data
    }