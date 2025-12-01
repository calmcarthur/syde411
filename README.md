# Hospital Microgrid Optimization Framework

This project implements a Mixed-Integer Linear Programming (MILP)
optimizer for scheduling energy resources in a hospital microgrid. It
optimizes the dispatch of a grid connection, solar PV, battery energy
storage, and a diesel generator to minimize operational costs while
satisfying demand and physical constraints.

## Project Structure

``` text
.
├── main.py              # Entry point: Handles CLI arguments and orchestration
├── optimizer.py         # Core Logic: Pyomo model definition, physics constraints, and plotting
├── config.py            # Configuration: Solver defaults and physical constants
├── data_profiles.py     # Data: Hourly load, solar, and price profiles
├── figures/             # Output: Generated plots and solver logs (created automatically)
└── README.md            # Documentation
```

## Installation

### 1. Python Dependencies

Install the required Python packages using pip:

``` bash
pip install pyomo matplotlib numpy highspy
```

Note: `highspy` installs the HiGHS solver libraries directly into your
Python environment.

### 2. Solvers

The project supports multiple MILP solvers. You need at least one
installed:

-   HiGHS (Recommended): Installed via `pip install highspy` (easiest).
-   CBC:
    -   Linux: `sudo apt-get install coinor-cbc`
    -   macOS: `brew install cbc`
    -   Conda: `conda install -c conda-forge coincbc`
-   GLPK:
    -   Linux: `sudo apt-get install glpk-utils`
    -   macOS: `brew install glpk`

------------------------------------------------------------------------

## 💻 Usage

The primary entry point is `main.py`.

### Basic Run

Run the optimization using the default solvers (`highs`, then `cbc`) and
generate the dashboard:

``` bash
python main.py
```

### Command Line Arguments

  ------------------------------------------------------------------------------------------
  Argument                Description             Example
  ----------------------- ----------------------- ------------------------------------------
  `--solver`              Specify one or more     `--solver glpk` or `--solver highs cbc`
                          solvers to use.         

  `--override`            Overwrite physical      `--override "Max Battery Capacity=1000"`
                          constants defined in    
                          `config.py`.            

  `--tee`                 Stream detailed solver  `--tee`
                          logs to the console.    

  `--no-plot`             Disable the generation  `--no-plot`
                          of result graphs.       
  ------------------------------------------------------------------------------------------

### Examples

**1. Compare Solvers Side-by-Side** Run the model with both HiGHS and
CBC to check for consistency in results:

``` bash
python main.py --solver highs cbc
```

**2. Sensitivity Analysis** Test a scenario where fuel is more expensive
and the battery is larger:

``` bash
python main.py --override "Generator Rate=0.50" --override "Max Battery Capacity=800"
```

**3. Debugging** Run with a specific solver and watch the solver's live
output stream:

``` bash
python main.py --solver cbc --tee
```

------------------------------------------------------------------------

## Configuration & Logic

### Physical Model (`optimizer.py`)

The model includes the following constraints:

-   Energy Balance: Supply must equal demand at every hour.
-   Generator Physics: Includes minimum up-time, ramp-up/down limits,
    and startup costs.
-   Battery Dynamics: Tracks State of Charge (SoC) with round-trip
    efficiency and charge/discharge rate limits.
-   Reusability: Forces the battery to end the day with at least as much
    energy as it started with.

### Constants (`config.py`)

Key parameters include:

-   `Generator Capacity`: Max output in kW.
-   `Generator Min Up Time`: Minimum hours the generator must stay on
    once started.
-   `Max Battery Capacity`: Battery size in kWh.
-   `Grid Capacity`: Max import limit in kW.

------------------------------------------------------------------------

## Results & Output

The `optimizer.py` generates detailed visualizations and results files 
in the `results/` folder:

-   `$model_name$_summary_dashboard.png`: Master view
-   `$model_name$_price_vs_usage.png`: Correlations with price
-   `compare_hourly_costs.png`: Solver comparison (if multiple solvers)
-   `$model_name$_results.txt`: Raw solver logs
