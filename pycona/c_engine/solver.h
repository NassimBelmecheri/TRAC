#ifndef C_ENGINE_SOLVER_H
#define C_ENGINE_SOLVER_H

#include "common.h"
#include "domain.h"
#include "constraint.h"
#include "constraint_net.h"
#include <vector>
#include <queue>
#include <chrono>
#include <random>

namespace quacq {

class FastSolver {
public:
    int num_vars;
    std::vector<Domain> initial_domains;
    std::vector<Domain> domains;
    std::vector<int> assignment;
    std::vector<int> best_assignment;
    int best_score;

    ConstraintNet cl;
    ConstraintNet bias;

    VarHeuristic var_heuristic;
    ValHeuristic val_heuristic;
    SolveMode mode;

    std::vector<int> target_scope;
    std::vector<bool> in_target_scope;

    std::chrono::steady_clock::time_point deadline;
    bool timeout_flag;
    double time_limit_sec;

    // Metrics
    long long nodes_visited;
    long long backtracks;

    FastSolver(int n_vars = 0);

    void set_num_vars(int n_vars);
    void set_domain(int var, const std::vector<int>& raw_vals);
    void set_cl(const ConstraintNet& c_net);
    void set_bias(const ConstraintNet& b_net);

    void reset_domains();

    // Query generation & solving
    ResultStatus solve(SolveMode m = SOLVE_ANY,
                       const std::vector<int>& scope = {},
                       double timeout_sec = 10.0);

    const std::vector<int>& get_assignment() const {
        return assignment;
    }

private:
    bool mac(int level);
    int select_variable();
    std::vector<int> order_values(int var);
    bool propagate(int level, int assigned_var);
    bool revise(int con_idx, int var, int level);
    double calc_wdeg(int var) const;
    int eval_val_bias_violations(int var, int val) const;
    bool is_timed_out();
};

} // namespace quacq

#endif // C_ENGINE_SOLVER_H
