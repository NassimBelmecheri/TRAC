#ifndef C_ENGINE_COMMON_H
#define C_ENGINE_COMMON_H

#include <vector>
#include <string>
#include <cmath>
#include <climits>
#include <algorithm>
#include <cstdint>
#include <chrono>

namespace quacq {

const int UNASSIGNED = -99999999;

enum ResultStatus {
    STATUS_SAT = 1,
    STATUS_UNSAT = 0,
    STATUS_TIMEOUT = 2,
    STATUS_ERROR = -1
};

enum EvalStatus {
    EVAL_SAT = 1,
    EVAL_VIOL = 0,
    EVAL_UNDEF = -1
};

enum ConstraintOp {
    // Binary comparison
    OP_NE = 0,             // x != y
    OP_EQ = 1,             // x == y
    OP_GT = 2,             // x > y
    OP_LT = 3,             // x < y
    OP_GE = 4,             // x >= y
    OP_LE = 5,             // x <= y

    // Binary difference / offset
    OP_DIFF_OFFSET_EQ = 6, // x - y == c
    OP_DIFF_OFFSET_NE = 7, // x - y != c
    OP_ABS_DIFF_EQ = 8,    // |x - y| == c
    OP_ABS_DIFF_NE = 9,    // |x - y| != c
    OP_ABS_DIFF_LT = 10,   // |x - y| < c
    OP_ABS_DIFF_LE = 11,   // |x - y| <= c
    OP_ABS_DIFF_GT = 12,   // |x - y| > c
    OP_ABS_DIFF_GE = 13,   // |x - y| >= c

    // Binary sum / offset
    OP_SUM_OFFSET_EQ = 14, // x + y == c
    OP_SUM_OFFSET_NE = 15, // x + y != c

    // Quaternary (Golomb rulers)
    OP_GOLOMB_EQ = 16,     // |x1 - x2| == |x3 - x4|
    OP_GOLOMB_NE = 17,     // |x1 - x2| != |x3 - x4|
    OP_GOLOMB_LT = 18,     // |x1 - x2| < |x3 - x4|
    OP_GOLOMB_GT = 19,     // |x1 - x2| > |x3 - x4|

    // Ternary linear
    OP_TERNARY_DIFF_EQ = 20, // x1 - x2 == x3
    OP_TERNARY_SUM_EQ = 21,  // x1 + x2 == x3

    // Extensional / Table
    OP_TABLE_ALLOWED = 22,   // Tuple must be in allowed list
    OP_TABLE_FORBIDDEN = 23, // Tuple must NOT be in forbidden list

    // Unary comparison (param = constant c)
    OP_UNARY_EQ = 30,      // x == c
    OP_UNARY_NE = 31,      // x != c
    OP_UNARY_GT = 32,      // x > c
    OP_UNARY_LT = 33,      // x < c
    OP_UNARY_GE = 34,      // x >= c
    OP_UNARY_LE = 35       // x <= c
};

enum VarHeuristic {
    VAR_LEXICO = 0,
    VAR_DOM = 1,
    VAR_DOM_WDEG = 2,
    VAR_BDEG = 3
};

enum ValHeuristic {
    VAL_LEXICO = 0,
    VAL_RANDOM = 1,
    VAL_MAX_BIAS_VIOL = 2
};

enum SolveMode {
    SOLVE_ANY = 0,              // Any solution satisfying CL
    SOLVE_VIOLATE_BIAS = 1,     // Satisfies CL and violates at least 1 constraint in Bias
    SOLVE_MAX_BIAS = 2,         // Satisfies CL and maximizes violated constraints in Bias
    SOLVE_FINDC_SPLITHALF = 3   // Satisfies CL and splits delta candidates as close to 50/50 as possible
};

struct AcquisitionMetrics {
    int membership_queries;
    int findscope_queries;
    int findc_queries;
    int learned_constraints;
    double total_time_sec;
    double solve_time_sec;
    int converged;
};

// Oracle callback signature: returns 1 if YES/SAT, 0 if NO/UNSAT
typedef int (*OracleCallback)(const int* assignment, int n_vars, const int* scope, int scope_size, void* user_data);

} // namespace quacq

#endif // C_ENGINE_COMMON_H
