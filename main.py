## main.py - Microgrid Energy Optimization MILP

import pyomo.environ as pyo

class MicrogridOptimizer:
    """
    Implements a Mixed-Integer Linear Program (MILP) for microgrid 
    energy scheduling optimization over a 24-hour period.
    The objective is to minimize the total operational cost.
    """
    
    def __init__(self, constants, discrete_data):
        """
        Initializes the optimizer with model constants and time-series data.
        
        Args:
            constants (dict): Dictionary of fixed system parameters.
            discrete_data (dict): Dictionary containing L(t), A(t), P(t) for each hour.
        """
        self.constants = constants
        self.discrete_data = discrete_data
        self.model = pyo.ConcreteModel()
        self.T = list(range(1, 25)) # Time steps (hours 1 to 24)
        
        self._define_parameters()
        self._define_variables()
        self._define_objective()
        self._define_constraints()
        
    def _define_parameters(self):
        """
        Defines all necessary parameters from the constants and discrete data.
        """
        model = self.model
        
        # Sets
        model.T = pyo.Set(initialize=self.T) 
        
        # Fixed System Constants
        model.B_max = pyo.Param(initialize=self.constants['Max Battery Capacity'], doc='Max Battery Capacity (kWh)')
        model.B_min = pyo.Param(initialize=self.constants['Min Battery Capacity'], doc='Min Battery Capacity (kWh)')
        model.SoC_initial = pyo.Param(initialize=self.constants['Initial State of Charge'], doc='Initial State of Charge (kWh)')
        model.R_max = pyo.Param(initialize=self.constants['Max Charge and Discharge Rate'], doc='Max Charge/Discharge Rate (kW)')
        model.eta_ch = pyo.Param(initialize=self.constants['Charging Efficiency'], doc='Charging Efficiency')
        model.eta_dis = pyo.Param(initialize=self.constants['Discharging Efficiency'], doc='Discharging Efficiency')
        model.alpha = pyo.Param(initialize=self.constants['Degradation Cost'], doc='Degradation Cost (\$/kWh)')
        model.C_gen = pyo.Param(initialize=self.constants['Generator Capacity'], doc='Generator Capacity (kW)')
        model.G_rate = pyo.Param(initialize=self.constants['Generator Rate'], doc='Generator Cost Rate (\$/kWh)')
        
        # Discrete 24hr Time-Series Constants
        model.L = pyo.Param(model.T, initialize=self.discrete_data['L(t) (kW)'], doc='Load (kW)')
        model.A = pyo.Param(model.T, initialize=self.discrete_data['A(t) (kW)'], doc='Available Solar (kW)')
        model.P = pyo.Param(model.T, initialize=self.discrete_data['P(t) (\$/kWh)'], doc='Grid Price (\$/kWh)')
        
    def _define_variables(self):
        """
        Defines the decision and state variables.
        """
        model = self.model
        
        # Decision Variables (Continuous)
        model.g = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Power drawn from the grid (kW)')
        model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Power used from solar (kW)')
        model.b_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Power charged to the battery (kW)')
        model.b_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Power discharged from the battery (kW)')
        
        # Decision Variables (Binary)
        model.u_ch = pyo.Var(model.T, domain=pyo.Binary, doc='If the battery is charging')
        model.u_dis = pyo.Var(model.T, domain=pyo.Binary, doc='If the battery is discharging')
        model.gen_on = pyo.Var(model.T, domain=pyo.Binary, doc='If the generator is on or off')
        
        # State Variables
        # SoC(0) is fixed by SoC_initial, so we define SoC for hours 1 to 24
        model.SoC = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='State of Charge (kWh) at end of hour t')

    def _define_objective(self):
        """
        Defines the objective function: Minimize total cost over 24 hours.
        
        The objective is:
        Min Sum_t=1^24 [ P(t)*g(t) + alpha*(b_ch(t) + b_dis(t)) + gen_on(t)*(C_gen*G_rate) ]
        """
        model = self.model
        
        def total_cost_rule(model):
            # Cost of power drawn from the grid (P(t)*g(t))
            grid_cost = sum(model.P[t] * model.g[t] for t in model.T)
            
            # Cost of battery degradation (alpha*(b_ch(t) + b_dis(t)))
            # Note: The proposal uses alpha*() but the variable is alpha* on the table, 
            # we will use model.alpha based on the table's "Degradation Cost" parameter.
            degradation_cost = sum(model.alpha * (model.b_ch[t] + model.b_dis[t]) for t in model.T)
            
            # Cost of running the generator (gen_on(t)*(C_gen*G_rate))
            # Note: Generator must run at max capacity C_gen (kW)
            generator_cost = sum(model.gen_on[t] * model.C_gen * model.G_rate for t in model.T)
            
            return grid_cost + degradation_cost + generator_cost
            
        model.C = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize, doc='Minimize Total Operational Cost')

    def _define_constraints(self):
        """
        Defines all constraints for the MILP model.
        """
        model = self.model
        
        # 1. Energy Balance Constraint
        # s(t) + b_dis(t) + G(t) + (gen_on(t)*C_gen) = L(t) + b_ch(t)
        # where G(t) is g(t), power from the grid
        def energy_balance_rule(model, t):
            # Input sources: Solar (s), Battery Discharge (b_dis), Grid (g), Generator (gen_on*C_gen)
            input_sources = model.s[t] + model.b_dis[t] + model.g[t] + (model.gen_on[t] * model.C_gen)
            # Output loads: Load (L), Battery Charge (b_ch)
            output_loads = model.L[t] + model.b_ch[t]
            return input_sources == output_loads
        model.EnergyBalance = pyo.Constraint(model.T, rule=energy_balance_rule, doc='Energy balance for each hour')

        # 2. Solar Usage Constraint
        # 0 <= s(t) <= A(t)
        def solar_usage_rule(model, t):
            return (0, model.s[t], model.A[t])
        model.SolarUsage = pyo.Constraint(model.T, rule=solar_usage_rule, doc='Solar power used cannot exceed available solar power')

        # 3. State of Charge Update Constraint
        # SoC(t) = SoC(t-1) + (eta_char * b_ch(t)) - (1/eta_dis * b_dis(t))
        # Note: We handle the initial state SoC(0) separately.
        def soc_update_rule(model, t):
            if t == 1:
                # Use SoC_initial for SoC(t-1) when t=1
                SoC_prev = model.SoC_initial
            else:
                # For t>1, use the previous hour's SoC
                SoC_prev = model.SoC[t-1]
            
            # State of Charge at end of hour t
            soc_at_t = SoC_prev + (model.eta_ch * model.b_ch[t]) - (1/model.eta_dis * model.b_dis[t])
            return model.SoC[t] == soc_at_t
        model.SoCUpdate = pyo.Constraint(model.T, rule=soc_update_rule, doc='Update battery state of charge')
        
        # Note: The SoC(0) = SoC_initial constraint is implicitly handled in SoCUpdate rule

        # 4. Battery Charge Limits Constraint
        # B_min <= SoC(t) <= B_max
        def soc_limits_rule(model, t):
            return (model.B_min, model.SoC[t], model.B_max)
        model.SoCLimits = pyo.Constraint(model.T, rule=soc_limits_rule, doc='Battery State of Charge limits')
        
        # 5. Battery Draw Rate Constraint
        # 0 <= b_dis(t) <= R_max * u_dis(t) (Using Big M for the binary variable)
        # Note: We must link b_dis(t) to u_dis(t). If u_dis=0, b_dis=0. If u_dis=1, b_dis <= R_max.
        def dis_rate_max_rule(model, t):
            # This enforces b_dis(t) <= R_max * u_dis(t)
            return model.b_dis[t] <= model.R_max * model.u_dis[t]
        model.DisRateMax = pyo.Constraint(model.T, rule=dis_rate_max_rule, doc='Discharge rate max limit and link to binary')
        
        # 6. Battery Charge Rate Constraint
        # 0 <= b_ch(t) <= R_max * u_ch(t) (Using Big M for the binary variable)
        # Note: We must link b_ch(t) to u_ch(t). If u_ch=0, b_ch=0. If u_ch=1, b_ch <= R_max.
        def ch_rate_max_rule(model, t):
            # This enforces b_ch(t) <= R_max * u_ch[t]
            return model.b_ch[t] <= model.R_max * model.u_ch[t]
        model.ChRateMax = pyo.Constraint(model.T, rule=ch_rate_max_rule, doc='Charge rate max limit and link to binary')
        
        # 7. Battery Binary Constraint
        # u_ch(t) + u_dis(t) <= 1
        # The battery cannot charge and discharge simultaneously.
        def binary_limit_rule(model, t):
            return model.u_ch[t] + model.u_dis[t] <= 1
        model.BinaryLimit = pyo.Constraint(model.T, rule=binary_limit_rule, doc='Cannot charge and discharge at the same time')

        # 8. Non-negativity Constraint (Implicitly handled by variable domain)
        # g(t) >= 0 (Already set via domain=pyo.NonNegativeReals for model.g)
        
        # 9. Reusability Constraint
        # SoC(24) >= SoC_initial
        def reusability_rule(model):
            return model.SoC[24] >= model.SoC_initial
        model.Reusability = pyo.Constraint(rule=reusability_rule, doc='Ensure the battery is at least at initial charge at the end of the day')

    def solve(self, solver='gurobi'):
        """
        Solves the constructed MILP model.
        
        Args:
            solver (str): The name of the solver to use (e.g., 'gurobi', 'cbc').
        
        Returns:
            pyo.SolverResults: The results object from the solver.
        """
        print(f"Attempting to solve the MILP model using the {solver} solver...")
        # Check if the solver is available/registered with Pyomo
        if not pyo.SolverFactory(solver).available():
            # Fallback to a common, open-source solver if the specified one isn't found
            print(f"Warning: {solver} not found. Trying 'cbc'...")
            solver = 'cbc'
            if not pyo.SolverFactory(solver).available():
                 # Final fallback to a generic error if 'cbc' isn't found
                 raise RuntimeError("No available MILP solver found (tried 'gurobi' and 'cbc'). Please install a solver like CBC.")

        opt = pyo.SolverFactory(solver)
        # Set a time limit for the solver for complex models
        # opt.options['timelimit'] = 60 
        
        results = opt.solve(self.model, tee=True)
        return results

    def print_results(self):
        """
        Prints key results from the solved model.
        """
        if pyo.value(self.model.C) is not None:
            print("\n" + "="*50)
            print(f"🌟 Optimal Total Operational Cost: \${pyo.value(self.model.C):.2f}")
            print("="*50 + "\n")
            
            # Print decision variables
            print("Hourly Scheduling Results:")
            print("-" * 75)
            header = "{:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5} | {:<5}"
            print(header.format("Hr", "Load", "Grid", "Solar", "BatCh", "BatDis", "SoC", "GenOn"))
            print("-" * 75)
            
            for t in range(1, 25):
                row = "{:<5} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f} | {:<5.0f}"
                print(row.format(
                    t,
                    pyo.value(self.model.L[t]),
                    pyo.value(self.model.g[t]),
                    pyo.value(self.model.s[t]),
                    pyo.value(self.model.b_ch[t]),
                    pyo.value(self.model.b_dis[t]),
                    pyo.value(self.model.SoC[t]),
                    pyo.value(self.model.gen_on[t])
                ))
            print("-" * 75)
        else:
            print("Model was not solved successfully or no feasible solution found.")


if __name__ == '__main__':
    # --- Data Definition based on the proposal ---

    # Fixed Constants
    FIXED_CONSTANTS = {
        'Max Battery Capacity': 500, # kWh
        'Min Battery Capacity': 100, # kWh
        'Initial State of Charge': 250, # kWh
        'Max Charge and Discharge Rate': 125, # kW
        'Charging Efficiency': 0.95, # decimal
        'Discharging Efficiency': 0.95, # decimal
        'Degradation Cost': 0.025, # $/kWh
        'Generator Capacity': 100, # kW
        'Generator Rate': 0.32, # $/kWh
    }
    
    # Discrete 24hr Time-Series Data
    DISCRETE_DATA = {
        'L(t) (kW)': {
            1: 60, 2: 60, 3: 60, 4: 60, 5: 70, 6: 80, 7: 100, 8: 150, 9: 200, 10: 220, 
            11: 240, 12: 250, 13: 252, 14: 255, 15: 257, 16: 255, 17: 252, 18: 250, 
            19: 240, 20: 220, 21: 200, 22: 150, 23: 100, 24: 80
        },
        'A(t) (kW)': {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 10, 6: 0, 7: 10, 8: 30, 9: 45, 10: 80, 
            11: 100, 12: 120, 13: 125, 14: 130, 15: 100, 16: 85, 17: 60, 18: 45, 
            19: 30, 20: 0, 21: 0, 22: 0, 23: 10, 24: 0
        },
        'P(t) (\$/kWh)': {
            1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.18, 8: 0.18, 9: 0.18, 10: 0.18, 
            11: 0.25, 12: 0.25, 13: 0.25, 14: 0.25, 15: 0.25, 16: 0.25, 17: 0.35, 18: 0.35, 
            19: 0.35, 20: 0.35, 21: 0.18, 22: 0.18, 23: 0.1, 24: 0.1
        },
    }
    
    # --- Model Execution ---
    
    optimizer = MicrogridOptimizer(FIXED_CONSTANTS, DISCRETE_DATA)
    
    # Attempt to solve the model. 'gurobi' is a commercial, high-performance solver, 
    # but 'cbc' is a common open-source alternative.
    # Ensure you have an MILP solver installed and accessible (e.g., CBC or Gurobi).
    try:
        results = optimizer.solve(solver='cbc') 
        
        # Check the solver status
        if (results.solver.status == pyo.SolverStatus.ok) and \
           (results.solver.termination_condition == pyo.TerminationCondition.optimal or \
            results.solver.termination_condition == pyo.TerminationCondition.feasible):
            
            optimizer.print_results()
            
        else:
            print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)

    except RuntimeError as e:
        print(f"Error during solving: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")