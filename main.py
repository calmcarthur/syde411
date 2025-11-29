"""Entry point for running the microgrid optimization pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Iterable

from pyomo.opt import SolverStatus, TerminationCondition

from config import DEFAULT_SOLVER, FALLBACK_SOLVER, FIXED_CONSTANTS
from data_profiles import get_default_profiles
from optimizer import MicrogridOptimizer


PARAMETER_PRESETS = {
    "baseline": {},
    "high_initial_soc": {"Initial State of Charge": 400},
    "tight_capacity": {"Max Battery Capacity": 450, "Min Battery Capacity": 150},
    "lower_minimum": {"Min Battery Capacity": 80},
}

SOLVER_PRESETS = {
    "cbc": ["cbc"],
    "glpk_then_cbc": ["glpk", "cbc"],
    "highs_then_cbc": ["highs", "cbc"],
}


def apply_constant_overrides(overrides: dict[str, float]) -> dict:
    updated = deepcopy(FIXED_CONSTANTS)
    updated.update(overrides)
    return updated


def parse_override_pairs(pairs: Iterable[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be in 'Key=Value' format: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        try:
            overrides[key] = float(value)
        except ValueError as exc:  # keep message readable
            raise ValueError(f"Could not parse numeric value for '{key}': {value}") from exc
    return overrides


def run_optimization(
    *,
    solver: list[str] | str | None = None,
    fallback: list[str] | str | None = None,
    overrides: dict[str, float] | None = None,
    tee: bool = False,
    compare: bool = False,
) -> None:
    constants = apply_constant_overrides(overrides or {})
    fallback_solver = fallback if fallback else FALLBACK_SOLVER

    if compare:
        solver_list: list[str] = _normalize_solver_list(solver)
        if len(solver_list) < 2:
            print("Provide at least two solvers with --compare to see differences.")
        for name in solver_list:
            print(f"\n=== Solver: {name} ===")
            try:
                optimizer = MicrogridOptimizer(constants, get_default_profiles())
                results = optimizer.solve_model(solver=name, fallback=fallback_solver, tee=tee)
            except RuntimeError as exc:
                print(f"  Error: {exc}")
                continue

            if not _is_optimal(results):
                print("  Solver did not reach an optimal solution.")
                print(f"  Status: {results.solver.status}")
                print(f"  Termination: {results.solver.termination_condition}")
                continue

            metrics = optimizer.extract_key_metrics()
            validation = optimizer.validate_solution()
            print(f"  Total cost: ${metrics['total_cost']:.2f}")
            print(
                "  Grid energy: "
                f"{metrics['grid_energy']:.2f} kWh | Solar used: {metrics['solar_used']:.2f} kWh"
            )
            print(
                "  Battery charge/discharge: "
                f"{metrics['battery_charge']:.2f} / {metrics['battery_discharge']:.2f} kWh"
            )
            print(f"  Generator on-hours: {metrics['generator_hours']:.0f}")
            print(
                "  Validation → energy balance max deviation: "
                f"{validation['energy_balance_max_violation']:.2e} kW"
            )
            print(
                "  Validation → SoC range: "
                f"{validation['soc_min']:.2f} – {validation['soc_max']:.2f} kWh"
            )
            print(
                "  Validation → Final vs initial SoC gap: "
                f"{validation['reusability_gap']:.2f} kWh"
            )
        return

    profiles = get_default_profiles()
    optimizer = MicrogridOptimizer(constants, profiles)

    primary_solver = solver if solver else DEFAULT_SOLVER

    try:
        results = optimizer.solve_model(solver=primary_solver, fallback=fallback_solver, tee=tee)
    except RuntimeError as exc:
        print(f"Error during solving: {exc}")
        return

    if _is_optimal(results):
        optimizer.summarize_schedule()
    else:
        print("Solver did not find an optimal solution.")
        print(f"Status: {results.solver.status}")
        print(f"Termination: {results.solver.termination_condition}")


def run_experiments(
    solver_labels: list[str] | None,
    parameter_labels: list[str] | None,
    *,
    tee: bool = False,
) -> None:
    solver_sets = _select_presets(solver_labels, SOLVER_PRESETS)
    parameter_sets = _select_presets(parameter_labels, PARAMETER_PRESETS)

    for param_name, overrides in parameter_sets:
        constants = apply_constant_overrides(overrides)
        print(f"\n=== Parameter scenario: {param_name} ===")
        for solver_name, solver_order in solver_sets:
            optimizer = MicrogridOptimizer(constants, get_default_profiles())
            print(f"\n[{solver_name}] Trying solvers in order: {', '.join(solver_order)}")
            try:
                results = optimizer.solve_model(solver=solver_order, fallback=None, tee=tee)
            except RuntimeError as exc:
                print(f"  Skipped: {exc}")
                continue

            if not _is_optimal(results):
                print("  Solver terminated without an optimal solution.")
                print(f"  Status: {results.solver.status}")
                print(f"  Termination: {results.solver.termination_condition}")
                continue

            metrics = optimizer.extract_key_metrics()
            validation = optimizer.validate_solution()

            print(f"  Solver used: {optimizer.last_solver}")
            print(f"  Total cost: ${metrics['total_cost']:.2f}")
            print(
                "  Grid energy: "
                f"{metrics['grid_energy']:.2f} kWh | Solar used: {metrics['solar_used']:.2f} kWh"
            )
            print(
                "  Battery charge/discharge: "
                f"{metrics['battery_charge']:.2f} / {metrics['battery_discharge']:.2f} kWh"
            )
            print(f"  Generator on-hours: {metrics['generator_hours']:.0f}")
            print(
                "  Validation → energy balance max deviation: "
                f"{validation['energy_balance_max_violation']:.2e} kW"
            )
            print(
                "  Validation → SoC range: "
                f"{validation['soc_min']:.2f} – {validation['soc_max']:.2f} kWh"
            )
            print(
                "  Validation → Final vs initial SoC gap: "
                f"{validation['reusability_gap']:.2f} kWh"
            )


def _select_presets(
    labels: list[str] | None,
    presets: dict[str, list[str] | dict[str, float]],
):
    if labels is None:
        return list(presets.items())

    selected = []
    for label in labels:
        if label not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset '{label}'. Available options: {available}")
        selected.append((label, presets[label]))
    return selected


def _normalize_solver_list(selection: list[str] | str | None) -> list[str]:
    if selection is None:
        return [DEFAULT_SOLVER, FALLBACK_SOLVER] if DEFAULT_SOLVER != FALLBACK_SOLVER else [DEFAULT_SOLVER]
    if isinstance(selection, str):
        return [selection]
    return list(selection)


def _is_optimal(results) -> bool:
    status_ok = results.solver.status == SolverStatus.ok
    termination_ok = results.solver.termination_condition in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    }
    return status_ok and termination_ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hospital microgrid optimization scenarios.")
    parser.add_argument(
        "--solver",
        nargs="+",
        help="Ordered solvers to try for a single run (default is config.DEFAULT_SOLVER).",
    )
    parser.add_argument(
        "--fallback",
        nargs="+",
        help="Additional solvers to try if the primary list fails (default is config.FALLBACK_SOLVER).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a constant as 'Name=Value'. Repeat for multiple overrides.",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run the predefined grid of solver and parameter experiments.",
    )
    parser.add_argument(
        "--solver-set",
        nargs="+",
        help="Names of solver presets to include when running experiments.",
    )
    parser.add_argument(
        "--parameter-set",
        nargs="+",
        help="Names of parameter presets to include when running experiments.",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Stream detailed solver logs (Pyomo tee=True).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Solve sequentially with each solver in --solver and display a side-by-side summary.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        overrides = parse_override_pairs(args.override or [])
    except ValueError as exc:
        parser.error(str(exc))

    if args.experiments:
        run_experiments(args.solver_set, args.parameter_set, tee=args.tee)
    else:
        run_optimization(
            solver=args.solver,
            fallback=args.fallback,
            overrides=overrides,
            tee=args.tee,
            compare=args.compare,
        )


if __name__ == "__main__":
    main()