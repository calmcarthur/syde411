"""Pyomo-based MILP optimizer for the hospital microgrid scheduling problem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyomo.environ as pyo

try:  # optional HiGHS python interface for Appsi solvers
    import pyomo.contrib.appsi.solvers.highs  # noqa: F401
except ImportError:
    pass


class MicrogridOptimizer:
    """Constructs and solves the mixed-integer microgrid scheduling problem."""

    def __init__(self, constants: dict, discrete_data: dict):
        self.constants = constants
        self.discrete_data = discrete_data
        self.time_horizon = list(range(1, 25))  # hours 1..24 inclusive
        self.model = None
        self._last_solver: str | None = None
        self._last_results: Any = None
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
    def solve_model(
        self,
        solver: str | Sequence[str] = "cbc",
        fallback: str | Sequence[str] | None = None,
        *,
        tee: bool = False,
    ):
        solver_to_use = self._resolve_solver(solver, fallback)
        opt = pyo.SolverFactory(solver_to_use)
        results = opt.solve(self.model, tee=tee)
        self._last_solver = solver_to_use
        self._last_results = results
        return results

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

        validation = self.validate_solution()
        print("\nValidation checks (tolerance 1e-5):")
        print(f" - Energy balance max deviation: {validation['energy_balance_max_violation']:.2e} kW")
        print(
            " - SoC within bounds "
            f"[{pyo.value(model.B_min):.2f}, {pyo.value(model.B_max):.2f}]: {validation['within_soc_bounds']}"
        )
        print(
            " - Final SoC >= initial: "
            f"{validation['reusability_met']} (Δ={validation['reusability_gap']:.2f} kWh)"
        )

    def validate_solution(self, tolerance: float = 1e-5) -> dict[str, float | bool]:
        """Check basic feasibility conditions for the solved schedule."""

        if self.model is None:
            raise RuntimeError("Optimization model has not been built.")

        model = self.model
        if pyo.value(model.total_cost) is None:
            raise RuntimeError("Solve the model before calling validate_solution().")

        energy_balance_deviation = []
        soc_values = []
        for t in self.time_horizon:
            generated = (
                pyo.value(model.s[t])
                + pyo.value(model.b_dis[t])
                + pyo.value(model.g[t])
                + pyo.value(model.gen_on[t]) * pyo.value(model.C_gen)
            )
            consumed = pyo.value(model.L[t]) + pyo.value(model.b_ch[t])
            energy_balance_deviation.append(abs(generated - consumed))
            soc_values.append(pyo.value(model.SoC[t]))

        soc_min = min(soc_values)
        soc_max = max(soc_values)
        soc_initial = pyo.value(model.SoC_initial)
        soc_final = pyo.value(model.SoC[self.time_horizon[-1]])

        return {
            "energy_balance_max_violation": max(energy_balance_deviation),
            "within_soc_bounds": (soc_min >= pyo.value(model.B_min) - tolerance)
            and (soc_max <= pyo.value(model.B_max) + tolerance),
            "soc_min": soc_min,
            "soc_max": soc_max,
            "reusability_met": soc_final >= soc_initial - tolerance,
            "reusability_gap": soc_final - soc_initial,
        }

    def extract_key_metrics(self) -> dict[str, float]:
        """Return aggregated metrics that help compare scenarios."""

        model = self.model
        if pyo.value(model.total_cost) is None:
            raise RuntimeError("Solve the model before requesting metrics.")

        grid_energy = sum(pyo.value(model.g[t]) for t in self.time_horizon)
        solar_used = sum(pyo.value(model.s[t]) for t in self.time_horizon)
        battery_charge = sum(pyo.value(model.b_ch[t]) for t in self.time_horizon)
        battery_discharge = sum(pyo.value(model.b_dis[t]) for t in self.time_horizon)
        generator_hours = sum(pyo.value(model.gen_on[t]) for t in self.time_horizon)

        return {
            "total_cost": pyo.value(model.total_cost),
            "grid_energy": grid_energy,
            "solar_used": solar_used,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "generator_hours": generator_hours,
        }

    def get_schedule(self) -> dict[str, list[float]]:
        """Return hourly values for key decision variables."""

        model = self.model
        if pyo.value(model.total_cost) is None:
            raise RuntimeError("Solve the model before requesting the schedule.")

        schedule = {
            "hour": [],
            "load": [],
            "grid": [],
            "solar": [],
            "battery_charge": [],
            "battery_discharge": [],
            "soc": [],
            "generator_on": [],
        }

        for t in self.time_horizon:
            schedule["hour"].append(t)
            schedule["load"].append(pyo.value(model.L[t]))
            schedule["grid"].append(pyo.value(model.g[t]))
            schedule["solar"].append(pyo.value(model.s[t]))
            schedule["battery_charge"].append(pyo.value(model.b_ch[t]))
            schedule["battery_discharge"].append(pyo.value(model.b_dis[t]))
            schedule["soc"].append(pyo.value(model.SoC[t]))
            schedule["generator_on"].append(pyo.value(model.gen_on[t]))

        return schedule

    @property
    def last_solver(self) -> str | None:
        return self._last_solver

    @property
    def last_results(self):
        return self._last_results

    @staticmethod
    def _resolve_solver(
        primary: str | Sequence[str],
        fallback: str | Sequence[str] | None,
    ) -> str:
        candidates: list[str] = []

        def _add_candidates(value: str | Sequence[str] | None) -> None:
            if value is None:
                return
            if isinstance(value, str):
                candidates.append(value)
                return
            candidates.extend(value)

        _add_candidates(primary)
        _add_candidates(fallback)

        seen: set[str] = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            factory = pyo.SolverFactory(name)
            if factory is None:
                continue

            try:
                available = factory.available(False)
            except Exception:  # pragma: no cover - defensive path
                available = True

            if available or name.startswith("appsi_"):
                if name != primary and isinstance(primary, str):
                    print(f"Warning: {primary} not found. Falling back to {name}...")
                return name

        tried = ", ".join(repr(name) for name in seen) or "<none>"
        raise RuntimeError(f"No available MILP solver found (tried {tried}).")
