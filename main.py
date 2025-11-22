"""Entry point for running the microgrid optimization pipeline."""

from pyomo.opt import SolverStatus, TerminationCondition

from config import DEFAULT_SOLVER, FALLBACK_SOLVER, FIXED_CONSTANTS
from data_profiles import get_default_profiles
from optimizer import MicrogridOptimizer


def run_optimization() -> None:
    profiles = get_default_profiles()
    optimizer = MicrogridOptimizer(FIXED_CONSTANTS, profiles)

    try:
        results = optimizer.solve_model(solver=DEFAULT_SOLVER, fallback=FALLBACK_SOLVER)
    except RuntimeError as exc:
        print(f"Error during solving: {exc}")
        return

    status_ok = results.solver.status == SolverStatus.ok
    termination_ok = results.solver.termination_condition in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    }

    if status_ok and termination_ok:
        optimizer.summarize_schedule()
    else:
        print("Solver did not find an optimal solution.")
        print(f"Status: {results.solver.status}")
        print(f"Termination: {results.solver.termination_condition}")


if __name__ == "__main__":
    run_optimization()