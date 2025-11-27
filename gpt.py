from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary, Reals,
    Objective, Constraint, ConstraintList, minimize, SolverFactory, TerminationCondition, value
)


def build_model():
    m = ConcreteModel()

    # ========================
    # Sets
    # ========================
    m.T = RangeSet(1, 24)

    # ========================
    # Data (your table)
    # ========================
    L_data = {
        1: 60,  2: 60,  3: 60,  4: 60,
        5: 70,  6: 80,  7: 100, 8: 150,
        9: 200, 10: 220, 11: 240, 12: 250,
        13: 252, 14: 255, 15: 257, 16: 255,
        17: 252, 18: 250, 19: 240, 20: 220,
        21: 200, 22: 150, 23: 100, 24: 80,
    }

    A_data = {
        1: 0,   2: 0,   3: 0,   4: 0,
        5: 0,   6: 0,   7: 10,  8: 30,
        9: 45, 10: 80, 11: 100, 12: 120,
        13: 125, 14: 130, 15: 100, 16: 85,
        17: 60, 18: 45, 19: 30, 20: 0,
        21: 0,  22: 0,  23: 0,  24: 0,
    }

    P_data = {
        1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10,
        5: 0.10, 6: 0.10, 7: 0.18, 8: 0.18,
        9: 0.18, 10: 0.18, 11: 0.25, 12: 0.25,
        13: 0.25, 14: 0.25, 15: 0.25, 16: 0.25,
        17: 0.35, 18: 0.35, 19: 0.35, 20: 0.35,
        21: 0.18, 22: 0.18, 23: 0.10, 24: 0.10,
    }

    m.L = Param(m.T, initialize=L_data)  # load (kW)
    m.A = Param(m.T, initialize=A_data)  # solar available (kW)
    m.P = Param(m.T, initialize=P_data)  # grid price ($/kWh)


# Grid_cap      = 40
# C_start       = 300
# Ramp_limit    = 5
# MinUp_hours   = 6
# C_gen         = 120
    # Scalars
    m.B_min = Param(initialize=100.0)   # kWh
    m.B_max = Param(initialize=500.0)   # kWh
    m.SoC_initial = Param(initialize=250.0)   # kWh
    m.R_max = Param(initialize=125.0)   # kW charge/discharge limit
    m.eta_ch = Param(initialize=0.95)
    m.eta_dis = Param(initialize=0.95)
    m.alpha = Param(initialize=0.025)   # $/kWh battery throughput
    m.C_gen = Param(initialize=118.0, mutable=True)   # kW max generator
    m.G_rate = Param(initialize=0.32)    # $/kWh generator cost
    m.C_start = Param(initialize=300.0, mutable=True)    # $ per startup
    m.P_min = Param(initialize=20.0)    # kW min charge/discharge when on
    # kW max grid import per hour
    m.Grid_cap = Param(initialize=60.0, mutable=True)
    # kW/hour generator ramp limit
    m.Ramp_limit = Param(initialize=8, mutable=True)
    # minimum generator up-time
    m.MinUp_hours = Param(initialize=6, mutable=True)

    # Saving this for reference
    # # Scalars
    # m.B_min = Param(initialize=100.0)   # kWh
    # m.B_max = Param(initialize=500.0)   # kWh
    # m.SoC_initial = Param(initialize=250.0)   # kWh
    # m.R_max = Param(initialize=125.0)   # kW charge/discharge limit
    # m.eta_ch = Param(initialize=0.95)
    # m.eta_dis = Param(initialize=0.95)
    # m.alpha = Param(initialize=0.025)   # $/kWh battery throughput
    # m.C_gen = Param(initialize=100.0)   # kW max generator
    # m.G_rate = Param(initialize=0.32)    # $/kWh generator cost
    # m.C_start = Param(initialize=40.0)    # $ per startup
    # m.P_min = Param(initialize=20.0)    # kW min charge/discharge when on
    # m.Grid_cap = Param(initialize=80.0)    # kW max grid import per hour
    # m.Ramp_limit = Param(initialize=20.0)    # kW/hour generator ramp limit
    # m.MinUp_hours = Param(initialize=4)       # minimum generator up-time

    # ========================
    # Variables
    # ========================
    # Continuous
    m.g = Var(m.T, domain=NonNegativeReals)  # grid
    m.s = Var(m.T, domain=NonNegativeReals)  # solar used
    m.b_ch = Var(m.T, domain=NonNegativeReals)  # battery charge
    m.b_dis = Var(m.T, domain=NonNegativeReals)  # battery discharge
    m.SoC = Var(m.T, domain=Reals)             # state of charge
    m.gen = Var(m.T, domain=NonNegativeReals)  # generator output (kW)

    # Binary
    m.u_ch = Var(m.T, domain=Binary)   # charging on/off
    m.u_dis = Var(m.T, domain=Binary)   # discharging on/off
    m.gen_on = Var(m.T, domain=Binary)   # generator on/off
    m.y_start = Var(m.T, domain=Binary)   # generator start events

    # ========================
    # Objective
    # ========================
    def total_cost_rule(m):
        return sum(
            m.P[t] * m.g[t]                          # grid cost
            + m.alpha * (m.b_ch[t] + m.b_dis[t])     # battery degradation
            + m.G_rate * m.gen[t]                    # generator energy cost
            + m.C_start * m.y_start[t]               # startup cost
            for t in m.T
        )

    m.Obj = Objective(rule=total_cost_rule, sense=minimize)

    # ========================
    # Constraints
    # ========================

    # Energy balance: solar + battery + grid + generator = load + charging
    def energy_balance_rule(m, t):
        return m.s[t] + m.b_dis[t] + m.g[t] + m.gen[t] == m.L[t] + m.b_ch[t]

    m.EnergyBalance = Constraint(m.T, rule=energy_balance_rule)

    # Solar usage
    def solar_limit_rule(m, t):
        return m.s[t] <= m.A[t]

    m.SolarLimit = Constraint(m.T, rule=solar_limit_rule)

    # Grid cap
    def grid_cap_rule(m, t):
        return m.g[t] <= m.Grid_cap

    m.GridLimit = Constraint(m.T, rule=grid_cap_rule)

    # SoC dynamics
    def soc_dynamics_rule(m, t):
        if t == 1:
            return m.SoC[t] == m.SoC_initial + m.eta_ch * m.b_ch[t] - (1.0 / m.eta_dis) * m.b_dis[t]
        else:
            return m.SoC[t] == m.SoC[t-1] + m.eta_ch * m.b_ch[t] - (1.0 / m.eta_dis) * m.b_dis[t]

    m.SoCDynamics = Constraint(m.T, rule=soc_dynamics_rule)

    # SoC bounds
    def soc_bounds_rule(m, t):
        return (m.B_min, m.SoC[t], m.B_max)

    m.SoCBounds = Constraint(m.T, rule=soc_bounds_rule)

    # Charge / discharge gating by binaries
    def charge_max_rule(m, t):
        return m.b_ch[t] <= m.R_max * m.u_ch[t]

    def discharge_max_rule(m, t):
        return m.b_dis[t] <= m.R_max * m.u_dis[t]

    m.ChargeMax = Constraint(m.T, rule=charge_max_rule)
    m.DischargeMax = Constraint(m.T, rule=discharge_max_rule)

    # Min charge/discharge when on
    def min_charge_rule(m, t):
        # if u_ch=0, this is b_ch >= 0; if u_ch=1, b_ch >= P_min
        return m.b_ch[t] >= m.P_min * m.u_ch[t]

    def min_discharge_rule(m, t):
        return m.b_dis[t] >= m.P_min * m.u_dis[t]

    m.MinCharge = Constraint(m.T, rule=min_charge_rule)
    m.MinDischarge = Constraint(m.T, rule=min_discharge_rule)

    # No simultaneous charge/discharge
    def exclusivity_rule(m, t):
        return m.u_ch[t] + m.u_dis[t] <= 1

    m.Exclusivity = Constraint(m.T, rule=exclusivity_rule)

    # Generator output only if on
    def gen_output_cap_rule(m, t):
        return m.gen[t] <= m.C_gen * m.gen_on[t]

    m.GenOutputCap = Constraint(m.T, rule=gen_output_cap_rule)

    # Generator ramp constraints
    def ramp_up_rule(m, t):
        if t == 1:
            return Constraint.Skip
        return m.gen[t] - m.gen[t-1] <= m.Ramp_limit

    def ramp_down_rule(m, t):
        if t == 1:
            return Constraint.Skip
        return m.gen[t-1] - m.gen[t] <= m.Ramp_limit

    m.RampUp = Constraint(m.T, rule=ramp_up_rule)
    m.RampDown = Constraint(m.T, rule=ramp_down_rule)

    # Startup definition: start when 0 -> 1
    def startup_def_rule(m, t):
        if t == 1:
            return m.y_start[t] >= m.gen_on[t]
        else:
            return m.y_start[t] >= m.gen_on[t] - m.gen_on[t-1]

    m.StartupDef = Constraint(m.T, rule=startup_def_rule)

    # Minimum up-time enforcement
    # sum_{k=t}^{t+MinUp-1} gen_on[k] >= MinUp * y_start[t]
    m.MinUp = ConstraintList()
    for t in m.T:
        end_t = min(t + int(m.MinUp_hours.value) - 1, m.T.last())
        window_len = end_t - t + 1
        m.MinUp.add(
            sum(m.gen_on[k] for k in range(t, end_t + 1)
                ) >= window_len * m.y_start[t]
        )

    # Reusability of battery at end
    def reusability_rule(m):
        return m.SoC[m.T.last()] >= m.SoC_initial

    m.Reusability = Constraint(rule=reusability_rule)

    return m


if __name__ == "__main__":
    model = build_model()
    solver = SolverFactory("cbc")
    result = solver.solve(model, tee=True)
    print(result)
    # Optional: quickly print total cost and some key decisions
    if result.solver.termination_condition == TerminationCondition.optimal:
        print("Total cost:", value(model.Obj))
    else:
        print("Model infeasible.")
        exit()
    print("Generator on hours:", [
          t for t in model.T if model.gen_on[t]() > 0.5])
