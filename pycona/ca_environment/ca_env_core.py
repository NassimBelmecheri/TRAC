from abc import ABC
from cpmpy.expressions.core import Expression


class CAEnv(ABC):
    """
    Abstract class interface for CA environments.
    """

    def __init__(self):
        """
        Initialize the CA environment.
        """
        self._instance = None
        self.metrics = None
        self.verbose = 0
        self.converged = False

    def init_state(self, **kwargs):
        """ Initialize the state of the CA environment. """
        self._converged = False

    @property
    def instance(self):
        """ Getter method for _instance """
        return self._instance

    @instance.setter
    def instance(self, instance):
        """ Setter method for _instance """
        self._instance = instance

    @property
    def metrics(self):
        """ Getter method for _metrics """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """ Setter method for _metrics """
        self._metrics = metrics

    @property
    def verbose(self):
        """ Get the verbosity of the environment """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """ Set the verbosity of the environment """
        self._verbose = verbose

    @property
    def converged(self):
        """ Get the convergence value """
        return self._converged

    @converged.setter
    def converged(self, converged):
        """ Set the convergence value """
        self._converged = converged

    def remove_from_bias(self, C):
        """
        Remove given constraints from the bias (candidates)

        :param C: list of constraints to be removed from B
        """
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), "remove_from_bias accepts as input a list of constraints or a constraint"

        if self.verbose >= 3:
            print(f"removing the following constraints from bias: {C}")

        # Fast path: filter by object identity instead of rebuilding a set of the
        # whole bias (which would call cpmpy's expensive Expression.__hash__ on
        # every candidate on every removal - O(n) hashing per call, O(n^2) over a
        # full acquisition). All callers remove constraints taken *from* the bias,
        # so identity is sufficient and correct.
        remove_ids = {id(c) for c in C}
        self.instance.bias = [c for c in self.instance.bias if id(c) not in remove_ids]

    def add_to_cl(self, C):
        """
        Add the given constraints to the list of learned constraints

        :param C: Constraints to add to CL
        """
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), "add_to_cl accepts as input a list of constraints or a constraint"

        if self.verbose >= 3:
            print(f"adding the following constraints to C_L: {C}")

        # Add constraint(s) c to the learned network and remove them from the bias
        self.instance.cl.extend(C)
        remove_ids = {id(c) for c in C}
        self.instance.bias = [c for c in self.instance.bias if id(c) not in remove_ids]

        self.metrics.cl += len(C)
        if self.verbose == 1:
            for c in C:
                print("L", end="")
