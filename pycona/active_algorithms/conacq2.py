from .algorithm_core import AlgorithmCAInteractive
from .algorithm_core import ActiveCAEnv
import cpmpy as cp
from pysat.solvers import Glucose3
import numpy as np 
import random
import time  # Import time

class ConAcq2(AlgorithmCAInteractive):
    """
    Implementation of the ConAcq.2 algorithm adapted for PyConA.
    """

    def __init__(self, ca_env=None):
        super().__init__(ca_env)

    def learn(self, instance, oracle=None, verbose=0, X=None, metrics=None):
        """
        Main learning loop.
        """
        # 1. Initialize Environment
        if self.env is None:
            self.env = ActiveCAEnv()
        self.env.init_state(instance, oracle, verbose, metrics)
        
        # Ensure bias is constructed
        if len(self.env.instance.bias) == 0:
            if X is None: X = instance.X  # Default to all variables
            self.env.instance.construct_bias(X)

        # 2. Setup Internal State for ConAcq2
        initial_bias = list(self.env.instance.bias)
        c_to_id = {c: i+1 for i, c in enumerate(initial_bias)}
        
        # Initialize SAT Solver (Version Space)
        sat_solver = Glucose3()
        
        bias_pointer = 0

        print(f"--- Starting ConAcq2 on {len(initial_bias)} constraints ---")

        # --- TIMEOUT SETUP ---
        start_time = time.time()
        timeout_seconds = 3600  # 1 Hour
        
        while True:
            # --- TIMEOUT CHECK ---
            if time.time() - start_time > timeout_seconds:
                print(f"!!! Timeout reached ({timeout_seconds}s). Stopping ConAcq2.")
                break

            # --- PHASE 1: SELECTION ---
            current_bias = self.env.instance.bias
            if not current_bias:
                print("Convergence: Bias is empty.")
                break
                
            # Simple round-robin over the *current* bias
            target_c = current_bias[bias_pointer % len(current_bias)]
            c_id = c_to_id[target_c]
            
            # Check if this target is already implied by SAT
            if not sat_solver.solve(assumptions=[-c_id]):
                # Implied: Add to Learned, Remove from Bias
                if verbose >= 2: print(f"  [Inferred] {target_c}")
                self.env.add_to_cl(target_c)
                self.env.remove_from_bias([target_c])
                sat_solver.add_clause([c_id])
                continue

            # --- PHASE 2: GENERATION (Near Miss) ---
            query_vars = self._generate_near_miss(target_c, current_bias)
            
            if query_vars is None:
                # Implicitly learned (UNSAT means target_c is required)
                if verbose >= 2: print(f"  [Implicit] {target_c}")
                self.env.add_to_cl(target_c)
                self.env.remove_from_bias([target_c])
                sat_solver.add_clause([c_id])
                continue

            # EXTRACT VALUES HERE
            query_values = [v.value() for v in query_vars]

            # --- PHASE 3: ORACLE ---
            # self.env.ask_membership_query expects variables for logging/visualization
            is_valid = self.env.ask_membership_query(query_vars)
            
            if verbose >= 2:
                print(f"Query: {query_values} -> Valid? {is_valid}")

            # --- PHASE 4: UPDATE ---
            if is_valid:
                # Positive: Remove all constraints violated by this valid query
                violated = []
                for c in current_bias:
                    # PASS query_values (integers), NOT query_vars
                    if not self._check_constraint(c, query_values):
                        violated.append(c)
                
                if violated:
                    if verbose >= 2: print(f"  (+) Valid. Pruned {len(violated)}")
                    # Add clauses to SAT: NOT(violated)
                    for c in violated:
                        sat_solver.add_clause([-c_to_id[c]])
                    self.env.remove_from_bias(violated)
            else:
                # Negative
                # Find Kappa (violated constraints in current bias)
                
                kappa = []
                kappa_ids = []
                
                for c in current_bias:
                    # Check using integer values
                    if not self._check_constraint(c, query_values):
                        kappa.append(c)
                        kappa_ids.append(c_to_id[c])
                
                if kappa_ids:
                    # --- NEW: PRINTING THE EXPLANATION ---
                    if verbose >= 2:
                        print(f"  (-) Invalid. Explanation size: {len(kappa)}")
                        # print(f"      Query: {query_values}")
                        # print("      -----------------------------")
                    # -------------------------------------

                    sat_solver.add_clause(kappa_ids)
                    
                    # Optimization: If only 1 constraint is violated, learn it immediately
                    if len(kappa) <= 4:
                        for target in kappa:
                            if verbose >= 1:
                                print(f"  [Direct] Single violation found! Learning: {target}")
                            self.env.add_to_cl(target)
                            self.env.remove_from_bias([target])                    
                  
            
            # Move pointer
            bias_pointer += 1

        sat_solver.delete()
        return self.env.instance.cl

    def _check_constraint(self, c, assignment):
        """Helper to check cpmpy constraint against a list of values."""
        # assignment must be a list of INTEGERS
        vars = self.env.instance.X
        
        # Backup old values (in case other parts of the code rely on state)
        old_vals = [v.value() for v in vars]
        
        try:
            # Apply new values
            for var, val in zip(vars, assignment):
                var._value = val
            
            # Evaluate
            result = c.value()
        except Exception as e:
            # Fallback for safety
            print(f"Error checking constraint {c}: {e}")
            result = True # Conservative assumption
        finally:
            # Restore
            for var, val in zip(vars, old_vals):
                var._value = val
            
        return bool(result)

    def _generate_near_miss(self, target_c, current_bias):
        """
        Efficient Near-Miss Generator.
        Strategy: Prioritize satisfying constraints that share variables with the target.
        This reduces solver overhead and produces high-quality queries quickly.
        """
        from cpmpy.transformations.get_variables import get_variables
        import random

       
        model = cp.Model()
        model += list(self.env.instance.cl)
        model += ~target_c

      
        target_vars = set(get_variables(target_c))
        
      
        related_bias = []
        other_bias = []
        
        for c in current_bias:
            if c is target_c: 
                continue
                
            # Check intersection of variables
            c_vars = get_variables(c)
            if not set(c_vars).isdisjoint(target_vars):
                related_bias.append(c)
            else:
                other_bias.append(c)
        
        soft_constraints = []
        
        # A. Add ALL related constraints (These are critical)
        soft_constraints.extend([c * 100 for c in related_bias])
        
        
        if len(other_bias) > 100:
            sample_other = random.sample(other_bias, 100)
        else:
            sample_other = other_bias
            
        soft_constraints.extend([c * 1 for c in sample_other])

        # 4. Maximize Satisfaction
        if soft_constraints:
            model.maximize(cp.sum(soft_constraints))

       
        if model.solve(solver="ortools", time_limit=10.0, num_search_workers=4):
            return self.env.instance.X

       
        if verbose >= 2: print(f"  [!] Optimization timed out on target {target_c}. Fallback.")
        
        model_sat = cp.Model()
        model_sat += list(self.env.instance.cl)
        model_sat += ~target_c
        
       
        flat_vars = self.env.instance.X
        heuristic_obj = sum([v for v in flat_vars]) 

        if random.choice([True, False]):
            model_sat.minimize(heuristic_obj)
        else:
            model_sat.maximize(heuristic_obj)

        if model_sat.solve(solver="ortools", time_limit=2.0):
            return self.env.instance.X

        return None
