"""
Pyomo-based MILP optimizer for the hospital microgrid scheduling problem.
Integrates complex generator physics and comprehensive visualization dashboards.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from io import StringIO
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

try:
    import pyomo.contrib.appsi.solvers.highs  # noqa: F401
except ImportError:
    pass


class MicrogridOptimizer:
    """Constructs and solves the mixed-integer microgrid scheduling problem."""

    def __init__(self, constants: dict, discrete_data: dict):
        self.constants = constants
        self.discrete_data = discrete_data
        self.time_horizon = list(range(1, 25))
        self.model = None
        self._last_solver: str | None = None
        self._last_results: Any = None
        self._build_model()

    # ------------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        self.model = pyo.ConcreteModel()
        self._build_parameters()
        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_parameters(self) -> None:
        model = self.model
        model.T = pyo.Set(initialize=self.time_horizon, ordered=True)

        # --- Battery & Grid Parameters ---
        model.B_max = pyo.Param(initialize=self.constants.get("Max Battery Capacity", 500.0))
        model.B_min = pyo.Param(initialize=self.constants.get("Min Battery Capacity", 100.0))
        model.SoC_initial = pyo.Param(initialize=self.constants.get("Initial State of Charge", 250.0))
        model.R_max = pyo.Param(initialize=self.constants.get("Max Charge and Discharge Rate", 125.0))
        model.eta_ch = pyo.Param(initialize=self.constants.get("Charging Efficiency", 0.95))
        model.eta_dis = pyo.Param(initialize=self.constants.get("Discharging Efficiency", 0.95))
        model.alpha = pyo.Param(initialize=self.constants.get("Degradation Cost", 0.025))
        model.Grid_cap = pyo.Param(initialize=self.constants.get("Grid Capacity", 60.0))

        # --- Generator Parameters ---
        model.C_gen = pyo.Param(initialize=self.constants.get("Generator Capacity", 118.0))
        model.G_rate = pyo.Param(initialize=self.constants.get("Generator Rate", 0.32))
        model.C_start = pyo.Param(initialize=self.constants.get("Generator Startup Cost", 300.0))
        model.P_min = pyo.Param(initialize=self.constants.get("Min Charge/Discharge", 20.0))
        model.Ramp_limit = pyo.Param(initialize=self.constants.get("Generator Ramp Limit", 8.0))
        model.MinUp_hours = pyo.Param(initialize=self.constants.get("Generator Min Up Time", 6.0))

        # --- Time Series Data ---
        model.L = pyo.Param(model.T, initialize=self.discrete_data["L(t) (kW)"])
        model.A = pyo.Param(model.T, initialize=self.discrete_data["A(t) (kW)"])
        model.P = pyo.Param(model.T, initialize=self.discrete_data["P(t) ($/kWh)"])

    def _build_variables(self) -> None:
        model = self.model
        model.g = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.b_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.b_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.SoC = pyo.Var(model.T, domain=pyo.Reals)
        model.gen = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.u_ch = pyo.Var(model.T, domain=pyo.Binary)
        model.u_dis = pyo.Var(model.T, domain=pyo.Binary)
        model.gen_on = pyo.Var(model.T, domain=pyo.Binary)
        model.y_start = pyo.Var(model.T, domain=pyo.Binary)

    def _build_objective(self) -> None:
        model = self.model

        def total_cost_rule(m):
            grid_cost = sum(m.P[t] * m.g[t] for t in m.T)
            degradation_cost = sum(m.alpha * (m.b_ch[t] + m.b_dis[t]) for t in m.T)
            gen_cost = sum(m.G_rate * m.gen[t] + m.C_start * m.y_start[t] for t in m.T)
            return grid_cost + degradation_cost + gen_cost

        model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    def _build_constraints(self) -> None:
        model = self.model
        # Energy Balance
        model.EnergyBalance = pyo.Constraint(model.T, rule=lambda m, t: 
            m.s[t] + m.b_dis[t] + m.g[t] + m.gen[t] == m.L[t] + m.b_ch[t])
        
        # Operational Limits
        model.SolarLimit = pyo.Constraint(model.T, rule=lambda m, t: m.s[t] <= m.A[t])
        model.GridLimit = pyo.Constraint(model.T, rule=lambda m, t: m.g[t] <= m.Grid_cap)
        
        # SoC Dynamics
        def soc_dynamics_rule(m, t):
            if t == 1:
                return m.SoC[t] == m.SoC_initial + m.eta_ch * m.b_ch[t] - (1.0 / m.eta_dis) * m.b_dis[t]
            return m.SoC[t] == m.SoC[t - 1] + m.eta_ch * m.b_ch[t] - (1.0 / m.eta_dis) * m.b_dis[t]
        model.SoCDynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)
        model.SoCBounds = pyo.Constraint(model.T, rule=lambda m, t: (m.B_min, m.SoC[t], m.B_max))
        model.Reusability = pyo.Constraint(rule=lambda m: m.SoC[m.T.last()] >= m.SoC_initial)

        # Battery Constraints
        model.ChargeMax = pyo.Constraint(model.T, rule=lambda m, t: m.b_ch[t] <= m.R_max * m.u_ch[t])
        model.DischargeMax = pyo.Constraint(model.T, rule=lambda m, t: m.b_dis[t] <= m.R_max * m.u_dis[t])
        model.Exclusivity = pyo.Constraint(model.T, rule=lambda m, t: m.u_ch[t] + m.u_dis[t] <= 1)
        model.MinCharge = pyo.Constraint(model.T, rule=lambda m, t: m.b_ch[t] >= m.P_min * m.u_ch[t])
        model.MinDischarge = pyo.Constraint(model.T, rule=lambda m, t: m.b_dis[t] >= m.P_min * m.u_dis[t])

        # Generator Constraints
        model.GenOutputCap = pyo.Constraint(model.T, rule=lambda m, t: m.gen[t] <= m.C_gen * m.gen_on[t])
        model.RampUp = pyo.Constraint(model.T, rule=lambda m, t: 
            pyo.Constraint.Skip if t == 1 else m.gen[t] - m.gen[t - 1] <= m.Ramp_limit)
        model.RampDown = pyo.Constraint(model.T, rule=lambda m, t: 
            pyo.Constraint.Skip if t == 1 else m.gen[t - 1] - m.gen[t] <= m.Ramp_limit)
        model.StartupDef = pyo.Constraint(model.T, rule=lambda m, t: 
            (m.y_start[t] >= m.gen_on[t]) if t == 1 else (m.y_start[t] >= m.gen_on[t] - m.gen_on[t - 1]))

        model.MinUp = pyo.ConstraintList()
        t_last = model.T.last()
        for t in model.T:
            end_t = min(t + int(pyo.value(model.MinUp_hours)) - 1, t_last)
            window_len = end_t - t + 1
            if window_len > 0:
                model.MinUp.add(sum(model.gen_on[k] for k in range(t, end_t + 1)) >= window_len * model.y_start[t])

    # ------------------------------------------------------------------
    # Data Extraction
    # ------------------------------------------------------------------
    def _extract_data(self) -> Dict[str, list]:
        m = self.model
        hours = list(m.T)
        data = {
            "hours": hours,
            "load": [pyo.value(m.L[t]) for t in hours],
            "solar_available": [pyo.value(m.A[t]) for t in hours],
            "grid_price": [pyo.value(m.P[t]) for t in hours],
            "solar_used": [pyo.value(m.s[t]) for t in hours],
            "grid_power": [pyo.value(m.g[t]) for t in hours],
            "gen_power": [pyo.value(m.gen[t]) for t in hours],
            "bat_ch": [pyo.value(m.b_ch[t]) for t in hours],
            "bat_dis": [pyo.value(m.b_dis[t]) for t in hours],
            "soc": [pyo.value(m.SoC[t]) for t in hours],
            "gen_on": [pyo.value(m.gen_on[t]) for t in hours],
            "startup": [pyo.value(m.y_start[t]) for t in hours],
        }
        
        # Calculated hourly costs
        data["cost_grid"] = [p * g for p, g in zip(data["grid_price"], data["grid_power"])]
        data["cost_gen"] = [pyo.value(m.G_rate) * g for g in data["gen_power"]]
        data["cost_startup"] = [pyo.value(m.C_start) * s for s in data["startup"]]
        data["cost_bat"] = [pyo.value(m.alpha) * (ch + dis) for ch, dis in zip(data["bat_ch"], data["bat_dis"])]
        data["cost_total_hourly"] = [sum(x) for x in zip(data["cost_grid"], data["cost_gen"], data["cost_startup"], data["cost_bat"])]
        
        return data

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def generate_plots(self, output_dir: str = "results", prefix: str = "") -> None:
        """Generates detailed plots including the Summary Dashboard and Cost Breakdown."""
        
        if pyo.value(self.model.total_cost) is None:
            print("Model not solved.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        data = self._extract_data()
        file_prefix = f"{prefix}_" if prefix else ""
        plt.style.use('seaborn-v0_8-darkgrid')

        self._plot_summary_dashboard(data, output_dir, file_prefix)
        self._plot_hourly_costs_breakdown(data, output_dir, file_prefix)
        self._plot_energy_dispatch(data, output_dir, file_prefix)
        
        print(f"Graphs saved to {output_dir}/")

    def _plot_summary_dashboard(self, data, output_dir, prefix):
        """Graph 9: A comprehensive 3x3 dashboard."""
        
        m = self.model
        hours = data["hours"]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(hours, data['load'], 'o-', color='#2E86AB', linewidth=2)
        ax1.set_title('Load Profile', fontweight='bold'); ax1.set_ylabel('kW')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(hours, data['solar_available'], 'o-', color='#F18F01', linewidth=2)
        ax2.set_title('Solar Availability', fontweight='bold'); ax2.set_ylabel('kW')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(hours, data['grid_price'], 'o-', color='#C73E1D', linewidth=2)
        ax3.set_title('Grid Price', fontweight='bold'); ax3.set_ylabel('$/kWh')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(hours, data['soc'], 'o-', color='#2E86AB', linewidth=2.5)
        ax4.axhline(pyo.value(m.B_max), color='r', linestyle='--')
        ax4.axhline(pyo.value(m.B_min), color='orange', linestyle='--')
        ax4.set_title('Battery SoC', fontweight='bold'); ax4.set_ylabel('kWh')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.step(hours, data['gen_on'], where='post', color='#BC4749', linewidth=2.5)
        ax5.set_title('Generator Status', fontweight='bold')
        ax5.set_yticks([0, 1]); ax5.set_yticklabels(['Off', 'On'])

        ax6 = fig.add_subplot(gs[1, 2])
        totals = [sum(data[k]) for k in ['cost_grid', 'cost_gen', 'cost_startup', 'cost_bat']]
        ax6.bar(['Grid', 'Gen', 'Start', 'Bat'], totals, color=['#C73E1D', '#BC4749', '#D62828', '#2E86AB'])
        ax6.set_title('Total Cost Breakdown', fontweight='bold')

        ax7 = fig.add_subplot(gs[2, :])
        l1 = np.array(data['solar_used'])
        l2 = l1 + np.array(data['bat_dis'])
        l3 = l2 + np.array(data['grid_power'])
        l4 = l3 + np.array(data['gen_power'])
        
        ax7.fill_between(hours, 0, l1, color='#F18F01', alpha=0.6, label='Solar')
        ax7.fill_between(hours, l1, l2, color='#6A994E', alpha=0.6, label='Discharge')
        ax7.fill_between(hours, l2, l3, color='#2E86AB', alpha=0.6, label='Grid')
        ax7.fill_between(hours, l3, l4, color='#BC4749', alpha=0.6, label='Generator')
        ax7.plot(hours, data['load'], 'k-', linewidth=2, label='Load')
        ax7.legend(loc='upper left', ncol=5)
        ax7.set_title('Energy Dispatch', fontweight='bold')

        fig.suptitle(f'Summary Dashboard ({prefix.strip("_")}) | Total: ${pyo.value(m.total_cost):.2f}', fontsize=16, fontweight='bold')
        plt.savefig(f"{output_dir}/{prefix}summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_hourly_costs_breakdown(self, data, output_dir, prefix):
        """Graph 6: Stacked bar chart of costs per hour."""
        
        hours = data['hours']
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(hours, data['cost_grid'], label='Grid', color='#C73E1D', alpha=0.8)
        ax.bar(hours, data['cost_gen'], bottom=data['cost_grid'], label='Gen Energy', color='#BC4749', alpha=0.8)
        
        bot_startup = np.array(data['cost_grid']) + np.array(data['cost_gen'])
        ax.bar(hours, data['cost_startup'], bottom=bot_startup, label='Startup', color='#D62828', alpha=0.8)
        
        bot_bat = bot_startup + np.array(data['cost_startup'])
        ax.bar(hours, data['cost_bat'], bottom=bot_bat, label='Battery', color='#2E86AB', alpha=0.8)
        
        ax.plot(hours, data['cost_total_hourly'], 'ko-', linewidth=2, label='Total')
        
        ax.set_xlabel('Hour', fontweight='bold'); ax.set_ylabel('Cost ($)', fontweight='bold')
        ax.set_title(f'Hourly Cost Breakdown ({prefix.strip("_")})', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}hourly_costs.png", dpi=300)
        plt.close()

    def _plot_energy_dispatch(self, data, output_dir, prefix):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.close()

    @staticmethod
    def plot_comparison_hourly_costs(results_map: dict, output_dir: str = "results"):
        """Side-by-side hourly cost comparison for two solvers."""
        
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        solvers = list(results_map.keys())
        if len(solvers) < 2: return

        s1, s2 = solvers[0], solvers[1]
        d1 = results_map[s1]['data']
        d2 = results_map[s2]['data']
        hours = d1['hours']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=True)
        
        def plot_on_ax(ax, d, name):
            ax.bar(hours, d['cost_grid'], label='Grid', color='#C73E1D')
            b1 = np.array(d['cost_grid'])
            ax.bar(hours, d['cost_gen'], bottom=b1, label='Gen', color='#BC4749')
            b2 = b1 + np.array(d['cost_gen'])
            ax.bar(hours, d['cost_startup'], bottom=b2, label='Start', color='#D62828')
            b3 = b2 + np.array(d['cost_startup'])
            ax.bar(hours, d['cost_bat'], bottom=b3, label='Bat', color='#2E86AB')
            ax.set_title(f'{name.upper()} Hourly Costs (Total: ${sum(d["cost_total_hourly"]):.2f})', fontweight='bold')
            ax.set_ylabel('Cost ($)')
            ax.grid(True, alpha=0.3)

        plot_on_ax(ax1, d1, s1)
        plot_on_ax(ax2, d2, s2)
        ax2.set_xlabel('Hour')
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/compare_hourly_costs.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def solve_model(self, solver="cbc", fallback=None, tee=False, log_output_dir="results"):
        solver_to_use = self._resolve_solver(solver, fallback)
        opt = pyo.SolverFactory(solver_to_use)
        
        if not os.path.exists(log_output_dir): os.makedirs(log_output_dir)
        log_file = os.path.join(log_output_dir, f"{solver_to_use}_results.txt")
        
        capture = StringIO()
        stdout_orig = sys.stdout
        try:
            class Tee:
                def write(self, d): 
                    capture.write(d)
                    if tee: stdout_orig.write(d)
                def flush(self): pass
            sys.stdout = Tee()
            results = opt.solve(self.model, tee=True)
            with open(log_file, "w") as f: f.write(capture.getvalue())
        finally:
            sys.stdout = stdout_orig
            
        self._last_solver = solver_to_use
        self._last_results = results
        return results

    def summarize_schedule(self):
        if pyo.value(self.model.total_cost) is None: return
        print(f"\nOptimal Cost: ${pyo.value(self.model.total_cost):.2f}")

    def extract_key_metrics(self):
        m = self.model
        return {
            "total_cost": pyo.value(m.total_cost),
            "grid_energy": sum(pyo.value(m.g[t]) for t in m.T)
        }

    def validate_solution(self):
        return {}

    @staticmethod
    def _resolve_solver(primary, fallback):
        candidates = []
        if isinstance(primary, str): candidates.append(primary)
        elif primary: candidates.extend(primary)
        if fallback: candidates.append(fallback)
        
        for name in candidates:
            if pyo.SolverFactory(name).available(): return name
        raise RuntimeError(f"No solver found from {candidates}")