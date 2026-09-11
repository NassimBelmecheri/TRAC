import ctypes
import os
import sys
from typing import List, Tuple, Optional, Callable, Dict, Any

# Locate shared library
_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_NAME = "c_engine.dll" if sys.platform == "win32" else "c_engine.so"
_LIB_PATH = os.path.join(_DIR, _LIB_NAME)

if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(f"Could not find compiled library at {_LIB_PATH}. Please compile c_engine.")

_lib = ctypes.CDLL(_LIB_PATH)

# Constants & Enums
UNASSIGNED = -99999999

# Algorithms
ALGO_QUACQ = 0
ALGO_PQUACQ = 1
ALGO_MQUACQ = 2
ALGO_MQUACQ2 = 3
ALGO_GROWACQ = 4
ALGO_BRUTECA = 5
ALGO_CONACQ1 = 6
ALGO_CONACQ2 = 7

ALGO_MAP = {
    "quacq": ALGO_QUACQ,
    "pquacq": ALGO_PQUACQ,
    "mquacq": ALGO_MQUACQ,
    "mquacq2": ALGO_MQUACQ2,
    "growacq": ALGO_GROWACQ,
    "bruteca": ALGO_BRUTECA,
    "conacq1": ALGO_CONACQ1,
    "conacq2": ALGO_CONACQ2,
}

# Relations
OP_NE = 0
OP_EQ = 1
OP_GT = 2
OP_LT = 3
OP_GE = 4
OP_LE = 5
OP_DIFF_OFFSET_EQ = 6
OP_DIFF_OFFSET_NE = 7
OP_ABS_DIFF_EQ = 8
OP_ABS_DIFF_NE = 9
OP_ABS_DIFF_LT = 10
OP_ABS_DIFF_LE = 11
OP_ABS_DIFF_GT = 12
OP_ABS_DIFF_GE = 13
OP_SUM_OFFSET_EQ = 14
OP_SUM_OFFSET_NE = 15
OP_GOLOMB_EQ = 16
OP_GOLOMB_NE = 17
OP_GOLOMB_LT = 18
OP_GOLOMB_GT = 19
OP_TERNARY_DIFF_EQ = 20
OP_TERNARY_SUM_EQ = 21
OP_TABLE_ALLOWED = 22
OP_TABLE_FORBIDDEN = 23
OP_UNARY_EQ = 30
OP_UNARY_NE = 31
OP_UNARY_GT = 32
OP_UNARY_LT = 33
OP_UNARY_GE = 34
OP_UNARY_LE = 35

# Solve modes
SOLVE_ANY = 0
SOLVE_VIOLATE_BIAS = 1
SOLVE_MAX_BIAS = 2
SOLVE_FINDC_SPLITHALF = 3

# Structure for metrics
class CAcquisitionMetrics(ctypes.Structure):
    _fields_ = [
        ("membership_queries", ctypes.c_int),
        ("findscope_queries", ctypes.c_int),
        ("findc_queries", ctypes.c_int),
        ("learned_constraints", ctypes.c_int),
        ("total_time_sec", ctypes.c_double),
        ("solve_time_sec", ctypes.c_double),
        ("converged", ctypes.c_int),
    ]

# Callback signature: int callback(const int* assignment, int n_vars, const int* scope, int scope_size, void* user_data)
ORACLE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p
)

# C-API bindings
_lib.quacq_create.argtypes = [ctypes.c_int]
_lib.quacq_create.restype = ctypes.c_void_p

_lib.quacq_free.argtypes = [ctypes.c_void_p]
_lib.quacq_free.restype = None

_lib.quacq_set_domain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_set_domain.restype = None

_lib.quacq_add_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.quacq_add_constraint.restype = ctypes.c_int

_lib.quacq_add_target_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.quacq_add_target_constraint.restype = ctypes.c_int

_lib.quacq_add_table_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.quacq_add_table_constraint.restype = ctypes.c_int

_lib.quacq_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_solve.restype = ctypes.c_int

_lib.quacq_count_bias_rejects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_count_bias_rejects.restype = ctypes.c_int

_lib.quacq_remove_bias_rejects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_remove_bias_rejects.restype = ctypes.c_int

_lib.quacq_run.argtypes = [ctypes.c_void_p, ORACLE_CALLBACK, ctypes.c_void_p, ctypes.c_int]
_lib.quacq_run.restype = ctypes.c_int

_lib.quacq_run_algorithm.argtypes = [ctypes.c_void_p, ctypes.c_int, ORACLE_CALLBACK, ctypes.c_void_p, ctypes.c_int]
_lib.quacq_run_algorithm.restype = ctypes.c_int

_lib.quacq_set_findscope_version.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.quacq_set_findscope_version.restype = None

_lib.quacq_set_findc_version.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.quacq_set_findc_version.restype = None

_lib.quacq_generate_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_generate_query.restype = ctypes.c_int

_lib.quacq_generate_tqgen_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
_lib.quacq_generate_tqgen_query.restype = ctypes.c_int

_lib.quacq_find_scope.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.quacq_find_scope.restype = ctypes.c_int

_lib.quacq_find_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.quacq_find_c.restype = ctypes.c_int

_lib.quacq_get_metrics.argtypes = [ctypes.c_void_p, ctypes.POINTER(CAcquisitionMetrics)]
_lib.quacq_get_metrics.restype = None

_lib.quacq_get_cl_count.argtypes = [ctypes.c_void_p]
_lib.quacq_get_cl_count.restype = ctypes.c_int

_lib.quacq_get_bias_count.argtypes = [ctypes.c_void_p]
_lib.quacq_get_bias_count.restype = ctypes.c_int

_lib.quacq_promote_bias_to_cl.argtypes = [ctypes.c_void_p]
_lib.quacq_promote_bias_to_cl.restype = None

_lib.quacq_get_cl_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.quacq_get_cl_constraint.restype = ctypes.c_int

_lib.quacq_get_bias_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.quacq_get_bias_constraint.restype = ctypes.c_int

_lib.quacq_check_constraint.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.quacq_check_constraint.restype = ctypes.c_int


class CQuAcqEngine:
    """
    Python wrapper around the high-speed C++ Constraint Acquisition engine.
    """

    def __init__(self, num_vars: int):
        self.num_vars = num_vars
        self._handle = _lib.quacq_create(num_vars)
        self._c_callback_ref = None  # Prevent GC of ctypes callback

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.quacq_free(self._handle)
            self._handle = None

    def set_domain(self, var: int, values: List[int]):
        c_vals = (ctypes.c_int * len(values))(*values)
        _lib.quacq_set_domain(self._handle, var, len(values), c_vals)

    def add_cl_constraint(self, op: int, scope: List[int], param: int = 0) -> int:
        c_scope = (ctypes.c_int * len(scope))(*scope)
        return _lib.quacq_add_constraint(self._handle, 0, op, len(scope), c_scope, param)

    def add_bias_constraint(self, op: int, scope: List[int], param: int = 0) -> int:
        c_scope = (ctypes.c_int * len(scope))(*scope)
        return _lib.quacq_add_constraint(self._handle, 1, op, len(scope), c_scope, param)

    def add_target_constraint(self, op: int, scope: List[int], param: int = 0) -> int:
        c_scope = (ctypes.c_int * len(scope))(*scope)
        return _lib.quacq_add_target_constraint(self._handle, op, len(scope), c_scope, param)

    def add_table_constraint(self, is_bias: bool, scope: List[int], tuples: List[List[int]], is_allowed: bool = True) -> int:
        c_scope = (ctypes.c_int * len(scope))(*scope)
        flat = []
        for t in tuples:
            flat.extend(t)
        c_flat = (ctypes.c_int * len(flat))(*flat)
        return _lib.quacq_add_table_constraint(
            self._handle, 1 if is_bias else 0, len(scope), c_scope, len(tuples), c_flat, 1 if is_allowed else 0
        )

    def solve(self, mode: int = SOLVE_ANY, scope: Optional[List[int]] = None, timeout_sec: float = 10.0) -> Optional[List[int]]:
        out_asgn = (ctypes.c_int * self.num_vars)()
        if scope is not None and len(scope) > 0:
            c_scope = (ctypes.c_int * len(scope))(*scope)
            scope_size = len(scope)
        else:
            c_scope = None
            scope_size = 0

        res = _lib.quacq_solve(self._handle, mode, scope_size, c_scope, timeout_sec, out_asgn)
        if res == 1:
            return list(out_asgn)
        return None

    def solve_ex(self, mode: int = SOLVE_ANY, scope: Optional[List[int]] = None,
                 timeout_sec: float = 10.0):
        """Like :meth:`solve` but returns ``(status, assignment)`` where status is
        ``"SAT"`` / ``"UNSAT"`` / ``"TIMEOUT"`` / ``"ERROR"``. This lets callers
        tell a *proof* of infeasibility (UNSAT) apart from the solver simply
        running out of time (TIMEOUT) - the plain :meth:`solve` collapses both to
        ``None``."""
        out_asgn = (ctypes.c_int * self.num_vars)()
        if scope is not None and len(scope) > 0:
            c_scope = (ctypes.c_int * len(scope))(*scope)
            scope_size = len(scope)
        else:
            c_scope = None
            scope_size = 0

        res = _lib.quacq_solve(self._handle, mode, scope_size, c_scope, timeout_sec, out_asgn)
        status = {1: "SAT", 0: "UNSAT", 2: "TIMEOUT"}.get(res, "ERROR")
        return status, (list(out_asgn) if res == 1 else None)
        c_asgn = (ctypes.c_int * len(assignment))(*assignment)
        return _lib.quacq_count_bias_rejects(self._handle, c_asgn)

    def remove_bias_rejects(self, assignment: List[int]) -> int:
        c_asgn = (ctypes.c_int * len(assignment))(*assignment)
        return _lib.quacq_remove_bias_rejects(self._handle, c_asgn)

    def generate_query(self, scope: Optional[List[int]] = None, timeout_sec: float = 5.0) -> Optional[List[int]]:
        out_asgn = (ctypes.c_int * self.num_vars)()
        if scope is not None and len(scope) > 0:
            c_scope = (ctypes.c_int * len(scope))(*scope)
            scope_size = len(scope)
        else:
            c_scope = None
            scope_size = 0

        res = _lib.quacq_generate_query(self._handle, scope_size, c_scope, timeout_sec, out_asgn)
        if res == 1:
            return list(out_asgn)
        return None

    def generate_tqgen_query(self, scope: Optional[List[int]] = None, tau: float = 0.2, alpha: float = 0.8) -> Optional[List[int]]:
        out_asgn = (ctypes.c_int * self.num_vars)()
        if scope is not None and len(scope) > 0:
            c_scope = (ctypes.c_int * len(scope))(*scope)
            scope_size = len(scope)
        else:
            c_scope = None
            scope_size = 0

        res = _lib.quacq_generate_tqgen_query(self._handle, scope_size, c_scope, tau, alpha, out_asgn)
        if res == 1:
            return list(out_asgn)
        return None

    def find_scope(self, query: List[int], Y: List[int]) -> List[int]:
        c_q = (ctypes.c_int * len(query))(*query)
        c_y = (ctypes.c_int * len(Y))(*Y)
        out_size = ctypes.c_int(0)
        out_scope = (ctypes.c_int * len(Y))()
        _lib.quacq_find_scope(self._handle, c_q, len(Y), c_y, ctypes.byref(out_size), out_scope)
        return [out_scope[i] for i in range(out_size.value)]

    def find_c(self, scope: List[int], query: List[int]) -> Optional[Tuple[int, int]]:
        c_sc = (ctypes.c_int * len(scope))(*scope)
        c_q = (ctypes.c_int * len(query))(*query)
        out_type = ctypes.c_int(-1)
        out_param = ctypes.c_int(0)
        res = _lib.quacq_find_c(self._handle, len(scope), c_sc, c_q, ctypes.byref(out_type), ctypes.byref(out_param))
        if res == 0:
            return (out_type.value, out_param.value)
        return None

    def set_findscope_version(self, version: int):
        _lib.quacq_set_findscope_version(self._handle, version)

    def set_findc_version(self, version: int):
        _lib.quacq_set_findc_version(self._handle, version)

    def run(self, algorithm: str = "quacq",
            oracle_func: Optional[Callable[[List[int], List[int]], bool]] = None,
            max_queries: int = 50000) -> int:
        """
        Run constraint acquisition.
        :param algorithm: one of 'quacq', 'pquacq', 'mquacq', 'mquacq2', 'growacq', 'bruteca'
        :param oracle_func: python callback (assignment, scope) -> bool
        :param max_queries: maximum queries
        """
        algo_code = ALGO_MAP.get(algorithm.lower(), ALGO_QUACQ)

        if oracle_func is not None:
            def _wrapped_cb(c_asgn, n_vars, c_sc, sc_size, user_data):
                asgn = [c_asgn[i] for i in range(n_vars)]
                sc = [c_sc[i] for i in range(sc_size)]
                ans = oracle_func(asgn, sc)
                return 1 if ans else 0

            c_cb = ORACLE_CALLBACK(_wrapped_cb)
            self._c_callback_ref = c_cb
        else:
            c_cb = ORACLE_CALLBACK()  # NULL function pointer

        return _lib.quacq_run_algorithm(self._handle, algo_code, c_cb, None, max_queries)

    def get_metrics(self) -> Dict[str, Any]:
        m = CAcquisitionMetrics()
        _lib.quacq_get_metrics(self._handle, ctypes.byref(m))
        return {
            "membership_queries": m.membership_queries,
            "findscope_queries": m.findscope_queries,
            "findc_queries": m.findc_queries,
            "total_queries": m.membership_queries + m.findscope_queries + m.findc_queries,
            "learned_constraints": m.learned_constraints,
            "total_time_sec": m.total_time_sec,
            "solve_time_sec": m.solve_time_sec,
            "converged": bool(m.converged),
        }

    def get_cl_count(self) -> int:
        return _lib.quacq_get_cl_count(self._handle)

    def get_bias_count(self) -> int:
        return _lib.quacq_get_bias_count(self._handle)

    def promote_bias_to_cl(self):
        """
        Promote all remaining constraints in the bias to CL (logically implied at convergence).
        """
        _lib.quacq_promote_bias_to_cl(self._handle)

    def get_bias_constraints(self) -> List[Dict[str, Any]]:
        count = self.get_bias_count()
        cons = []
        for i in range(count):
            out_type = ctypes.c_int(0)
            out_scope_size = ctypes.c_int(0)
            out_scope = (ctypes.c_int * self.num_vars)()
            out_param = ctypes.c_int(0)
            res = _lib.quacq_get_bias_constraint(
                self._handle, i, ctypes.byref(out_type), ctypes.byref(out_scope_size), out_scope, ctypes.byref(out_param)
            )
            if res == 0:
                sc = [out_scope[j] for j in range(out_scope_size.value)]
                cons.append({
                    "id": i,
                    "type": out_type.value,
                    "scope": sc,
                    "param": out_param.value
                })
        return cons

    def get_cl_constraints(self) -> List[Dict[str, Any]]:
        count = self.get_cl_count()
        cons = []
        for i in range(count):
            out_type = ctypes.c_int(0)
            out_scope_size = ctypes.c_int(0)
            out_scope = (ctypes.c_int * self.num_vars)()
            out_param = ctypes.c_int(0)
            res = _lib.quacq_get_cl_constraint(
                self._handle, i, ctypes.byref(out_type), ctypes.byref(out_scope_size), out_scope, ctypes.byref(out_param)
            )
            if res == 0:
                sc = [out_scope[j] for j in range(out_scope_size.value)]
                cons.append({
                    "id": i,
                    "type": out_type.value,
                    "scope": sc,
                    "param": out_param.value
                })
        return cons
