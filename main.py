"""Entry point for running the microgrid optimization pipeline."""

from __future__ import annotations
import argparse
from copy import deepcopy

from config import FIXED_CONSTANTS
from data_profiles import get_default_profiles
from optimizer import MicrogridOptimizer
from pyomo.opt import SolverStatus, TerminationCondition

DEFAULT_SOLVERS = ["highs", "cbc"]

def apply_constant_overrides(overrides: dict) -> dict:
    u = deepcopy(FIXED_CONSTANTS)
    u.update(overrides)
    return u

def parse_override_pairs(pairs):
    return {k: float(v) for k, v in (p.split("=", 1) for p in pairs)}

def run_optimization(solvers, overrides=None, tee=False, plot=True):
    constants = apply_constant_overrides(overrides or {})
    results_map = {}

    for name in solvers:
        print(f"\n{'='*20}\nRunning Solver: {name}\n{'='*20}")
        try:
            opt = MicrogridOptimizer(constants, get_default_profiles())
            res = opt.solve_model(solver=name, tee=tee, log_output_dir="results")
            
            if (res.solver.status == SolverStatus.ok and 
                res.solver.termination_condition in {TerminationCondition.optimal, TerminationCondition.feasible}):
                
                opt.summarize_schedule()
                results_map[name] = {'data': opt._extract_data()}
                
                if plot:
                    print(f"Generating dashboard for {name}...")
                    opt.generate_plots(output_dir="results", prefix=name)
            else:
                print(f"Solver {name} did not find optimal solution.")
                
        except Exception as e:
            print(f"Error with {name}: {e}")

    if plot and len(results_map) >= 2:
        print("\nGenerating comparison plots...")
        MicrogridOptimizer.plot_comparison_hourly_costs(results_map, output_dir="results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", nargs="+", default=DEFAULT_SOLVERS)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    try:
        overrides = parse_override_pairs(args.override)
    except:
        overrides = {}

    run_optimization(args.solver, overrides, args.tee, not args.no_plot)

if __name__ == "__main__":
    main()