import time
import cpmpy as cp
from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from .. import Metrics
from ..utils import *
from ..find_constraint.findc_obj import findc_obj_splithalf
import random
class BruteCA(AlgorithmCAInteractive):
    """
    Implements the 'Brute' Constraint Acquisition algorithm.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        super().__init__(ca_env)

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None, metrics: Metrics = None):
        if X is None:
            X = instance.X
            
        # 1. Initialize Environment
        self.env.init_state(instance, oracle, verbose, metrics)

        # 2. Ensure Bias exists
        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(X)

        # 2b. Process smaller scopes first (e.g. unary before binary).
        # A negative answer to a partial query on a scope S may actually be
        # caused by a constraint on a *sub-scope* of S (typically a unary
        # constraint on one endpoint). Because BruteCA has no FindScope step,
        # it would misattribute such a violation to a constraint on the full
        # scope S and learn a spurious constraint. Learning the smaller scopes
        # first puts those sub-scope constraints into the learned network L, so
        # the L-respecting violation generation below no longer triggers them.
        self.env.instance.bias.sort(key=lambda c: len(get_scope(c)))

        # Main Loop: Continue while there are constraints in the bias
        while len(self.env.instance.bias) > 0:
            
            # --- 1. PICK ---
            # Pick the first available constraint
            c = self.env.instance.bias[0]

            # --- 2. GENERATE VIOLATION ---
            # Generate e that satisfies L (learned) AND violates c
            gen_start = time.time()
            
            query_assignment = self._generate_violation(c, X)
            
            gen_end = time.time()
            self.env.metrics.increase_generation_time(gen_end - gen_start)

            # --- 3. ASK ORACLE ---
            if len(query_assignment) > 0:
                self.env.metrics.increase_generated_queries()
                
                # --- SANITIZATION ---
                # Fix None values to 0 to prevent Environment crash
                for var in self.env.instance.X:
                    if var.value() is None:
                        var._value = 0
                # --------------------

                # Ask the oracle
                is_valid = self.env.ask_membership_query(query_assignment)
                
                if is_valid:
                    # Case B: Oracle says YES (Refuted).
                    if self.env.verbose > 2: print(f"  -> Oracle: YES (Refuted {c})")
                    self.env.remove_from_bias([c])
                
                else:
                    # Case C: Oracle says NO (Confirmed).
                    if self.env.verbose > 2: print("  -> Oracle: NO (Confirmed)")
                    
                    scope_vars = get_scope(c)
                    
                    # Run FindC to find the EXACT constraint on this scope
                    learned_c = self.env.run_findc(scope_vars)

                    if learned_c is not None:
                        # 1. Add to learned network
                        self.env.add_to_cl(learned_c)
                        
                        if self.env.verbose >= 1:
                            print(f"\nLearned {learned_c}")

                        # --- NEW: REMOVE ALL CONSTRAINTS WITH SAME SCOPE ---
                        # Instead of just removing learned_c, we remove everything 
                        # in the bias that affects these specific variables.
                        # This prevents redundancy (learning <, then checking <=, etc.)
                        
                        target_scope_set = set(scope_vars)
                        constraints_to_remove = []
                        
                        for bias_c in self.env.instance.bias:
                            # Compare scopes (using set to ignore order of vars)
                            if set(get_scope(bias_c)) == target_scope_set:
                                constraints_to_remove.append(bias_c)
                        
                        if constraints_to_remove:
                            self.env.remove_from_bias(constraints_to_remove)
                        # ---------------------------------------------------
                        
                    else:
                        # Fallback: Just remove the current one if FindC failed
                        self.env.remove_from_bias([c])
            else:
                # UNSAT (Implicit)
                if self.env.verbose > 2: print(f"  [Implied] Cannot violate {c}")
                self.env.remove_from_bias([c])

        self.env.metrics.finalize_statistics()


    def _generate_violation(self, target_c, X, max_random_tries=2000, time_limit=0.5):
        """
        Generate a query e that VIOLATES target_c while SATISFYING the learned
        network L on the scope (paper Algorithm 1: "generate e that satisfies L
        and violates c").

        Respecting L is essential for soundness: without it, a random assignment
        that violates target_c may also violate an already-learned sub-scope
        constraint (e.g. a unary constraint on one of the scope variables), and
        the oracle's negative answer would be misattributed to target_c's scope.

        Returns the scope args (with violating values set) or an empty list when
        no violation exists (target_c is implied by L -> remove it from the bias).
        """
        if X is None:
            X = self.env.instance.X

        # Variables involved in the constraint we want to violate
        scope_vars = get_scope(target_c)

        # Bounds (assuming uniform domains; grab from first var)
        lb, ub = scope_vars[0].get_bounds()

        # Constraints already learned on this scope: the counter-example must satisfy them.
        sub_cl = get_con_subset(self.env.instance.cl, scope_vars)

        # --- STRATEGY 1: RANDOM SAMPLING ON SCOPE (bounded, L-respecting) ---
        start = time.time()
        for _ in range(max_random_tries):
            for v in scope_vars:
                v._value = int(random.randint(lb, ub))
            # Want: violates target_c AND satisfies every learned constraint on the scope
            if not target_c.value() and all(check_value(c) for c in sub_cl):
                return target_c._args
            if time.time() - start > time_limit:
                break

        # --- STRATEGY 2: CP FALLBACK (satisfy L on the scope, violate target_c) ---
        try:
            m = cp.Model(list(sub_cl))
            m += ~target_c
            if m.solve():
                return target_c._args
        except Exception:
            pass

        # --- No violation possible: target_c is implied by L, signal "no query" ---
        return []

            
            
    def generate_findc_query(self, L, delta,time_limit=0.2):
        """
        Generate a findc query.

        Constraints from B are taken into account as soft constraints.
        The objective function used is the one presented in the same paper, doing a dichotomy search on delta.
        Changes directly the values of the variables.

        :param L: Learned network in the given scope.
        :param delta: Candidate constraints in the given scope.
        :return: Boolean value representing a success or failure on the generation.
        """
        tmp = cp.Model(L)

        sat = sum([c for c in delta])  # Get the amount of satisfied constraints from B

        # At least 1 violated and at least 1 satisfied:
        # We want this to assure that each answer of the user will reduce the set of candidates
        # If all are violated, we already know that the example will be a non-solution due to previous answers!
        tmp += sat < len(delta)
        tmp += sat > 0

        # Try first without objective
        s = cp.SolverLookup.get("ortools", tmp)
        flag = s.solve()

        if not flag:
            # UNSAT, stop here
            return False

        Y = get_scope(delta[0])
        Y = list(dict.fromkeys(Y))  # Remove duplicates

        # Next solve will change the values of the variables in the lY2
        # So we need to return them to the original ones to continue if we don't find a solution next
        values = [x.value() for x in Y]

        # So a solution was found, try to find a better one now
        s.solution_hint(Y, values)

        objective = findc_obj_splithalf(sat, delta, ca_env=self.env)
        # Run with the objective
        s.maximize(objective)  # We want to try and do it like a dichotomic search

        flag2 = s.solve(time_limit)

        if not flag2:
            restore_scope_values(Y, values)
            return flag

        else:
            return flag2
    def findc_bruteca(self,scope):
        # Initialize delta
        delta = get_con_subset(self.env.instance.bias, scope)
        delta = [c for c in delta if check_value(c) is False]

        if len(delta) == 1:
            c = delta[0]
            return c

        if len(delta) == 0:
            return None
        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.env.instance.cl, scope)

        # Save current variable values
        scope_values = [x.value() for x in scope]

        while True:

            # Generate a counter example to reduce the candidates
            flag = self.generate_findc_query(sub_cl, delta)

            if flag is False:
                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                   return None
                restore_scope_values(scope, scope_values)

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            self.env.metrics.increase_findc_queries()

            if self.env.ask_membership_query(scope):
                # delta <- delta \setminus K_{delta}(e)
                [self.env.remove_from_bias(c) for c in delta if check_value(c) is False]
                delta = [c for c in delta if check_value(c) is not False]
            else:  # user says UNSAT
                # delta <- K_{delta}(e)
                [self.env.remove_from_bias(c) for c in delta if check_value(c) is not False]
                delta = [c for c in delta if check_value(c) is False]
