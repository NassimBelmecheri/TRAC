#ifndef C_ENGINE_CONSTRAINT_H
#define C_ENGINE_CONSTRAINT_H

#include "common.h"
#include "domain.h"
#include <vector>
#include <string>
#include <cmath>

namespace quacq {

class Constraint {
public:
    int id;
    int type;
    std::vector<int> scope;
    int param;
    int weight;
    std::vector<std::vector<int>> table; // for table constraints

    Constraint() : id(-1), type(OP_NE), param(0), weight(1) {}

    Constraint(int id_, int type_, const std::vector<int>& scope_, int param_ = 0)
        : id(id_), type(type_), scope(scope_), param(param_), weight(1) {}

    // Check if an assignment satisfies the constraint
    EvalStatus check(const std::vector<int>& assignment) const {
        for (int v : scope) {
            if (v < 0 || v >= static_cast<int>(assignment.size())) return EVAL_UNDEF;
            if (assignment[v] == UNASSIGNED) return EVAL_UNDEF;
        }

        bool satisfied = false;
        switch (type) {
            case OP_NE:
                satisfied = (assignment[scope[0]] != assignment[scope[1]]);
                break;
            case OP_EQ:
                satisfied = (assignment[scope[0]] == assignment[scope[1]]);
                break;
            case OP_GT:
                satisfied = (assignment[scope[0]] > assignment[scope[1]]);
                break;
            case OP_LT:
                satisfied = (assignment[scope[0]] < assignment[scope[1]]);
                break;
            case OP_GE:
                satisfied = (assignment[scope[0]] >= assignment[scope[1]]);
                break;
            case OP_LE:
                satisfied = (assignment[scope[0]] <= assignment[scope[1]]);
                break;

            case OP_DIFF_OFFSET_EQ:
                satisfied = ((assignment[scope[0]] - assignment[scope[1]]) == param);
                break;
            case OP_DIFF_OFFSET_NE:
                satisfied = ((assignment[scope[0]] - assignment[scope[1]]) != param);
                break;
            case OP_ABS_DIFF_EQ:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) == param);
                break;
            case OP_ABS_DIFF_NE:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) != param);
                break;
            case OP_ABS_DIFF_LT:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) < param);
                break;
            case OP_ABS_DIFF_LE:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) <= param);
                break;
            case OP_ABS_DIFF_GT:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) > param);
                break;
            case OP_ABS_DIFF_GE:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) >= param);
                break;

            case OP_SUM_OFFSET_EQ:
                satisfied = ((assignment[scope[0]] + assignment[scope[1]]) == param);
                break;
            case OP_SUM_OFFSET_NE:
                satisfied = ((assignment[scope[0]] + assignment[scope[1]]) != param);
                break;

            case OP_GOLOMB_EQ:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) ==
                             std::abs(assignment[scope[2]] - assignment[scope[3]]));
                break;
            case OP_GOLOMB_NE:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) !=
                             std::abs(assignment[scope[2]] - assignment[scope[3]]));
                break;
            case OP_GOLOMB_LT:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) <
                             std::abs(assignment[scope[2]] - assignment[scope[3]]));
                break;
            case OP_GOLOMB_GT:
                satisfied = (std::abs(assignment[scope[0]] - assignment[scope[1]]) >
                             std::abs(assignment[scope[2]] - assignment[scope[3]]));
                break;

            case OP_TERNARY_DIFF_EQ:
                satisfied = ((assignment[scope[0]] - assignment[scope[1]]) == assignment[scope[2]]);
                break;
            case OP_TERNARY_SUM_EQ:
                satisfied = ((assignment[scope[0]] + assignment[scope[1]]) == assignment[scope[2]]);
                break;

            case OP_UNARY_EQ:
                satisfied = (assignment[scope[0]] == param);
                break;
            case OP_UNARY_NE:
                satisfied = (assignment[scope[0]] != param);
                break;
            case OP_UNARY_GT:
                satisfied = (assignment[scope[0]] > param);
                break;
            case OP_UNARY_LT:
                satisfied = (assignment[scope[0]] < param);
                break;
            case OP_UNARY_GE:
                satisfied = (assignment[scope[0]] >= param);
                break;
            case OP_UNARY_LE:
                satisfied = (assignment[scope[0]] <= param);
                break;

            case OP_TABLE_ALLOWED: {
                bool found = false;
                for (const auto& t : table) {
                    bool match = true;
                    for (size_t i = 0; i < scope.size(); ++i) {
                        if (t[i] != assignment[scope[i]]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) { found = true; break; }
                }
                satisfied = found;
                break;
            }
            case OP_TABLE_FORBIDDEN: {
                bool found = false;
                for (const auto& t : table) {
                    bool match = true;
                    for (size_t i = 0; i < scope.size(); ++i) {
                        if (t[i] != assignment[scope[i]]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) { found = true; break; }
                }
                satisfied = !found;
                break;
            }
            default:
                satisfied = true;
                break;
        }

        return satisfied ? EVAL_SAT : EVAL_VIOL;
    }

    inline bool is_violated(const std::vector<int>& assignment) const {
        return check(assignment) == EVAL_VIOL;
    }

    inline bool is_satisfied(const std::vector<int>& assignment) const {
        return check(assignment) == EVAL_SAT;
    }

    // Direct binary evaluation between two values v1, v2
    inline bool eval_binary(int v1, int v2) const {
        switch (type) {
            case OP_NE: return v1 != v2;
            case OP_EQ: return v1 == v2;
            case OP_GT: return v1 > v2;
            case OP_LT: return v1 < v2;
            case OP_GE: return v1 >= v2;
            case OP_LE: return v1 <= v2;
            case OP_DIFF_OFFSET_EQ: return (v1 - v2) == param;
            case OP_DIFF_OFFSET_NE: return (v1 - v2) != param;
            case OP_ABS_DIFF_EQ: return std::abs(v1 - v2) == param;
            case OP_ABS_DIFF_NE: return std::abs(v1 - v2) != param;
            case OP_ABS_DIFF_LT: return std::abs(v1 - v2) < param;
            case OP_ABS_DIFF_LE: return std::abs(v1 - v2) <= param;
            case OP_ABS_DIFF_GT: return std::abs(v1 - v2) > param;
            case OP_ABS_DIFF_GE: return std::abs(v1 - v2) >= param;
            case OP_SUM_OFFSET_EQ: return (v1 + v2) == param;
            case OP_SUM_OFFSET_NE: return (v1 + v2) != param;
            default: return true;
        }
    }

    // Direct unary evaluation
    inline bool eval_unary(int v) const {
        switch (type) {
            case OP_UNARY_EQ: return v == param;
            case OP_UNARY_NE: return v != param;
            case OP_UNARY_GT: return v > param;
            case OP_UNARY_LT: return v < param;
            case OP_UNARY_GE: return v >= param;
            case OP_UNARY_LE: return v <= param;
            default: return true;
        }
    }
};

} // namespace quacq

#endif // C_ENGINE_CONSTRAINT_H
