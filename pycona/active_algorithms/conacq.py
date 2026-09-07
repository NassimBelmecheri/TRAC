from .algorithm_core import AlgorithmCAInteractive
from .algorithm_core import ActiveCAEnv
import cpmpy as cp
from pysat.solvers import Glucose3
import numpy as np
import random



class ConAcq1(AlgorithmCAInteractive):
    """
    Passive Learning Algorithm (ConAcq.1) with Convergence Rate Tracking.
    """

    def __init__(self, ca_env=None):
        super().__init__(ca_env)

    def learn(self, instance, oracle=None, verbose=1):
        # 1. Initialize Environment
        if self.env is None:
            self.env = ActiveCAEnv()
        self.env.init_state(instance, oracle, verbose=verbose)
        
        # 2. Acquire Data
        if not hasattr(oracle, 'dataset'):
            raise ValueError("For ConAcq1, Oracle must have .dataset attribute (Positives, Negatives).")
        
        positives, negatives = oracle.dataset
        print(f"--- Starting ConAcq1 ---")
        print(f"    Data: {len(positives)} Pos, {len(negatives)} Neg")

        # 3. Setup Bias
        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(instance.X)

        # Mapping and Sets
        flat_bias = self._flatten_constraints(self.env.instance.bias)
        c_to_id = {c: i+1 for i, c in enumerate(flat_bias)}
        id_to_c = {i+1: c for i, c in enumerate(flat_bias)}
        
        # S_ids tracks the Most Specific hypothesis (starts full)
        S_ids = set(c_to_id.values())
        
        # --- METRICS SETUP ---
        # Get target constraints strings for fast comparison
        target_strs = set()
        if hasattr(oracle, 'constraints'):
            target_strs = set(str(c) for c in oracle.constraints)
        
        # List to store convergence history: [(n_examples, size_S, precision), ...]
        history = []
        # ---------------------

        sat_solver = Glucose3()

        # --- PHASE 1: POSITIVES (Calculate S and Track Convergence) ---
        print("    [Phase 1] Processing Positive Examples...")
        
        for i, ex in enumerate(positives):
            to_remove = []
            
            # Pruning Logic
            for c_id in list(S_ids):
                c = id_to_c[c_id]
                # Check if example violates constraint c
                if not self._check_constraint(c, ex):
                    to_remove.append(c_id)
            
            for c_id in to_remove:
                S_ids.remove(c_id)
                sat_solver.add_clause([-c_id]) 

            # --- CONVERGENCE RATE CALCULATION ---
            # Calculate stats every 10 examples (or every 1 for small datasets) to save time
            if i % 10 == 0 or i == len(positives) - 1:
                current_S_size = len(S_ids)
                
                # How many constraints in current S are actually in Target?
                # Note: In ConAcq1, if data is correct, S always contains ALL target constraints.
                # So intersection should be equal to |CT|.
                # Precision = |CT| / |S|. When Precision=1.0, we have converged.
                
                # Check overlap (expensive, so we rely on string matching)
                current_S_strs = set(str(id_to_c[cid]) for cid in S_ids)
                
                # True Positives (Constraints in S that are actually Target)
                tp = len(current_S_strs.intersection(target_strs))
                
                # Precision of S (How much of S is "True"?)
                precision_S = tp / current_S_size if current_S_size > 0 else 0
                
                # Recall of S (Did we accidentally delete true rules?)
                # Should always be 1.0 in passive learning with correct data
                recall_S = tp / len(target_strs) if len(target_strs) > 0 else 0

                history.append({
                    "n_examples": i + 1,
                    "size_S": current_S_size,
                    "true_constraints_in_S": tp,
                    "precision": round(precision_S, 4),
                    "recall": round(recall_S, 4)
                })
                
                if verbose >= 2:
                    print(f"       Ex {i+1}: |S|={current_S_size}, Precision={precision_S:.2f}")
            # ------------------------------------

        if verbose:
            print(f"    [S] Final Size: {len(S_ids)}")

        # Save history to metrics for later plotting
        if self.env.metrics:
            self.env.metrics.conacq1_history = history

        # --- PHASE 2: NEGATIVES (Build SAT Formula) ---
        print("    [Phase 2] Processing Negative Examples...")
        for i, ex in enumerate(negatives):
            kappa_ids = []
            for c_id in S_ids:
                c = id_to_c[c_id]
                if not self._check_constraint(c, ex):
                    kappa_ids.append(c_id)
            
            if kappa_ids:
                sat_solver.add_clause(kappa_ids)

        # --- PHASE 3: SOLVE (Calculate G) ---
        print("    [Solver] Computing G...")
        if sat_solver.solve():
            sat_model = sat_solver.get_model()
            learned_G = []
            for lit in sat_model:
                if lit > 0 and lit in id_to_c:
                    learned_G.append(id_to_c[lit])
            
            # Final Metrics
            self.env.instance.cl = learned_G
            print(f"    [Result] S size: {len(S_ids)} | G (Learned) size: {len(learned_G)}")
            
            sat_solver.delete()
            return learned_G
        else:
            print("    [Error] UNSAT: No set of constraints can explain this data.")
            sat_solver.delete()
            return []

    # --- HELPERS ---
    def _flatten_constraints(self, clist):
        flat = []
        if isinstance(clist, (list, tuple, np.ndarray)):
            for item in clist:
                flat.extend(self._flatten_constraints(item))
        else:
            flat.append(clist)
        return flat

    def _check_constraint(self, c, assignment):
        vars_flat = self._flatten_constraints(self.env.instance.variables)
        old_vals = [v.value() for v in vars_flat]
        
        # Apply assignment
        for var, val in zip(vars_flat, assignment):
            var._value = val
            
        try:
            result = c.value()
        except:
            result = True # Assume satisfied if eval fails
            
        # Restore
        for var, val in zip(vars_flat, old_vals):
            var._value = val
        return bool(result)