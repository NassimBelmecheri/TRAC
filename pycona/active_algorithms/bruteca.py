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
                    
                    # Run FindC to find the EXACT constraint on this scope.
                    # We use BruteCA's own solver-free FindC (findc_bruteca),
                    # inspired by the QuAcq2 `find_con` loop, instead of the
                    # environment's default solver-based FindC.
                    learned_c = self.findc_bruteca(scope_vars)

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


    def _generate_violation(self, target_c, X, max_random_tries=2000):
        """
        Generate a query e that VIOLATES target_c, using ONLY random sampling on
        the scope - no constraint solver anywhere (the original FASTCA generator).

        We draw random values on the scope variables until the assignment
        violates target_c. Violating a single candidate constraint on its own
        scope is easy (a violation practically always exists and is hit within a
        handful of tries), so this is fast and never invokes a CP solver. The
        ``max_random_tries`` bound is only a safety net; if it is exhausted the
        candidate is dropped from the bias.

        Note: this intentionally does NOT constrain the sample to satisfy the
        learned network L on the scope. That keeps the loop cheap (one constraint
        evaluation per try). If strict L-respecting generation is ever needed
        (e.g. to avoid over-acquisition under a perfect oracle), it should be
        added as an opt-in, not via a solver in this hot loop.
        """
        if X is None:
            X = self.env.instance.X

        # Variables involved in the constraint we want to violate
        scope_vars = get_scope(target_c)

        # Bounds (assuming uniform domains; grab from first var)
        lb, ub = scope_vars[0].get_bounds()

        # --- RANDOM SAMPLING ON SCOPE (bounded) - NO SOLVER ---
        for _ in range(max_random_tries):
            for v in scope_vars:
                v._value = int(random.randint(lb, ub))
            if not target_c.value():
                return target_c._args

        # --- No violation found within the budget: drop the candidate ---
        return []

            
            
    def generate_findc_query(self, L, delta, max_tries=2000):
        """
        Generate a FindC discriminating example by RANDOM sampling - NO solver.

        Draw random values on the scope until the assignment (i) satisfies the
        learned network L on the scope and (ii) satisfies at least one but not
        all candidates in ``delta`` (so the oracle's answer strictly reduces
        ``delta``). Values are set in place on the scope variables, mirroring the
        contract of the original solver-based generator.

        :param L: Learned network in the given scope.
        :param delta: Candidate constraints in the given scope.
        :return: True if a discriminating example was set, else False.
        """
        Y = get_scope(delta[0])
        Y = list(dict.fromkeys(Y))  # unique scope variables
        bounds = [v.get_bounds() for v in Y]
        saved = [v.value() for v in Y]
        n = len(delta)

        for _ in range(max_tries):
            for v, (lb, ub) in zip(Y, bounds):
                v._value = int(random.randint(int(lb), int(ub)))
            # (i) respect the learned network on the scope
            if not all(check_value(c) is not False for c in L):
                continue
            # (ii) satisfy at least one but not all candidates in delta
            n_sat = sum(1 for c in delta if check_value(c) is not False)
            if 0 < n_sat < n:
                return True

        # No discriminating example found - restore values and signal failure
        restore_scope_values(Y, saved)
        return False
    def findc_bruteca(self, scope):
        """Solver-free FindC for a confirmed scope (inspired by QuAcq2 ``find_con``).

        ``delta`` is the set of bias constraints on ``scope`` that are violated by
        the current (invalid) example - the candidates for the real constraint.
        We repeatedly generate a discriminating example that respects the learned
        network and satisfies some-but-not-all of ``delta`` (via random sampling,
        no solver), ask the oracle, and shrink ``delta``: on a YES answer the
        violated candidates are removed, on a NO answer only the violated ones are
        kept. When no discriminator can be found the survivors are equivalent
        w.r.t. C_L, so any one is returned. Returns the learned constraint, or
        ``None`` to signal a graceful collapse (candidate not in the bias).
        """
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
