"""Pyomo-based MILP optimizer for the hospital microgrid scheduling problem."""

from __future__ import annotations

import pyomo.environ as pyo


class MicrogridOptimizer:
    """Constructs and solves the mixed-integer microgrid scheduling problem."""

    def __init__(self, constants: dict, discrete_data: dict):
        self.constants = constants
        self.discrete_data = discrete_data
        self.time_horizon = list(range(1, 25))  # hours 1..24 inclusive
        self.model = None
        self._build_model()

    # ------------------------------------------------------------------
    # Model construction helpers
    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        self.model = pyo.ConcreteModel()
        self._build_parameters()
        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_parameters(self) -> None:
        model = self.model
        model.T = pyo.Set(initialize=self.time_horizon)

        model.B_max = pyo.Param(initialize=self.constants["Max Battery Capacity"])
        model.B_min = pyo.Param(initialize=self.constants["Min Battery Capacity"])
        model.SoC_initial = pyo.Param(
            initialize=self.constants["Initial State of Charge"],
        )
        model.R_max = pyo.Param(initialize=self.constants["Max Charge and Discharge Rate"])
        model.eta_ch = pyo.Param(initialize=self.constants["Charging Efficiency"])
        model.eta_dis = pyo.Param(initialize=self.constants["Discharging Efficiency"])
        model.alpha = pyo.Param(initialize=self.constants["Degradation Cost"])
        model.C_gen = pyo.Param(initialize=self.constants["Generator Capacity"])
        model.G_rate = pyo.Param(initialize=self.constants["Generator Rate"])

        model.L = pyo.Param(model.T, initialize=self.discrete_data["L(t) (kW)"])
        model.A = pyo.Param(model.T, initialize=self.discrete_data["A(t) (kW)"])
        model.P = pyo.Param(model.T, initialize=self.discrete_data["P(t) ($/kWh)"])

    def _build_variables(self) -> None:
        model = self.model
        model.g = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.b_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.b_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.u_ch = pyo.Var(model.T, domain=pyo.Binary)
        model.u_dis = pyo.Var(model.T, domain=pyo.Binary)
        model.gen_on = pyo.Var(model.T, domain=pyo.Binary)
        model.SoC = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    def _build_objective(self) -> None:
        model = self.model

        def total_cost_rule(m):
            grid_cost = sum(m.P[t] * m.g[t] for t in m.T)
            degradation_cost = sum(m.alpha * (m.b_ch[t] + m.b_dis[t]) for t in m.T)
            generator_cost = sum(m.gen_on[t] * m.C_gen * m.G_rate for t in m.T)
            return grid_cost + degradation_cost + generator_cost

        model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    def _build_constraints(self) -> None:
        model = self.model

        def energy_balance_rule(m, t):
            inputs = m.s[t] + m.b_dis[t] + m.g[t] + m.gen_on[t] * m.C_gen
            outputs = m.L[t] + m.b_ch[t]
            return inputs == outputs

        model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

        def solar_usage_rule(m, t):
            return (0, m.s[t], m.A[t])

        model.solar_usage = pyo.Constraint(model.T, rule=solar_usage_rule)

        def soc_update_rule(m, t):
            prev_soc = m.SoC_initial if t == 1 else m.SoC[t - 1]
            return m.SoC[t] == prev_soc + m.eta_ch * m.b_ch[t] - (1 / m.eta_dis) * m.b_dis[t]

        model.soc_update = pyo.Constraint(model.T, rule=soc_update_rule)

        def soc_limits_rule(m, t):
            return (m.B_min, m.SoC[t], m.B_max)

        model.soc_limits = pyo.Constraint(model.T, rule=soc_limits_rule)

        def discharge_rate_rule(m, t):
            return m.b_dis[t] <= m.R_max * m.u_dis[t]

        model.discharge_limit = pyo.Constraint(model.T, rule=discharge_rate_rule)

        def charge_rate_rule(m, t):
            return m.b_ch[t] <= m.R_max * m.u_ch[t]

        model.charge_limit = pyo.Constraint(model.T, rule=charge_rate_rule)

        def binary_limit_rule(m, t):
            return m.u_ch[t] + m.u_dis[t] <= 1

        model.mode_limit = pyo.Constraint(model.T, rule=binary_limit_rule)

        def reusability_rule(m):
            return m.SoC[24] >= m.SoC_initial

        model.reusability = pyo.Constraint(rule=reusability_rule)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve_model(self, solver: str = "cbc", fallback: str | None = None):
        solver_to_use = self._resolve_solver(solver, fallback)
        opt = pyo.SolverFactory(solver_to_use)
        return opt.solve(self.model, tee=True)

    def summarize_schedule(self) -> None:
        model = self.model
        if pyo.value(model.total_cost) is None:
            print("Model has not been solved yet.")
            return

        print("\n" + "=" * 50)
        print(f"🌟 Optimal Total Operational Cost: ${pyo.value(model.total_cost):.2f}")
        print("=" * 50 + "\n")
        header = "{:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5}"
        print("Hourly Scheduling Results:")
        print("-" * 75)
        print(header.format("Hr", "Load", "Grid", "Solar", "BatCh", "BatDis", "SoC", "GenOn"))
        print("-" * 75)
        row = "{:<5} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f}"
        for t in self.time_horizon:
            print(
                row.format(
                    t,
                    pyo.value(model.L[t]),
                    pyo.value(model.g[t]),
                    pyo.value(model.s[t]),
                    pyo.value(model.b_ch[t]),
                    pyo.value(model.b_dis[t]),
                    pyo.value(model.SoC[t]),
                    pyo.value(model.gen_on[t]),
                )
            )
        print("-" * 75)

    @staticmethod
    def _resolve_solver(primary: str, fallback: str | None):
        if pyo.SolverFactory(primary).available():
            return primary
        if fallback and pyo.SolverFactory(fallback).available():
            print(f"Warning: {primary} not found. Falling back to {fallback}...")
            return fallback
        raise RuntimeError(
            f"No available MILP solver found (tried '{primary}'" + (f" and '{fallback}'" if fallback else "") + ")."
        )
