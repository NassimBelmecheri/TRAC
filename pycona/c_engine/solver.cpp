#include "solver.h"
#include <iostream>
#include <algorithm>
#include <climits>
#include <random>

namespace quacq {

FastSolver::FastSolver(int n_vars)
    : num_vars(n_vars),
      best_score(-1),
      var_heuristic(VAR_DOM_WDEG),
      val_heuristic(VAL_MAX_BIAS_VIOL),
      mode(SOLVE_ANY),
      timeout_flag(false),
      time_limit_sec(10.0),
      nodes_visited(0),
      backtracks(0) {
    if (n_vars > 0) {
        set_num_vars(n_vars);
    }
}

void FastSolver::set_num_vars(int n_vars) {
    num_vars = n_vars;
    initial_domains.assign(n_vars, Domain());
    domains.assign(n_vars, Domain());
    assignment.assign(n_vars, UNASSIGNED);
    best_assignment.assign(n_vars, UNASSIGNED);
    in_target_scope.assign(n_vars, false);
    cl.set_num_vars(n_vars);
    bias.set_num_vars(n_vars);
}

void FastSolver::set_domain(int var, const std::vector<int>& raw_vals) {
    if (var >= 0 && var < num_vars) {
        initial_domains[var].init(raw_vals);
        domains[var] = initial_domains[var];
    }
}

void FastSolver::set_cl(const ConstraintNet& c_net) {
    cl = c_net;
    if (cl.num_vars < num_vars) {
        cl.num_vars = num_vars;
    }
    cl.rebuild_index();
}

void FastSolver::set_bias(const ConstraintNet& b_net) {
    bias = b_net;
    if (bias.num_vars < num_vars) {
        bias.num_vars = num_vars;
    }
    bias.rebuild_index();
}

void FastSolver::reset_domains() {
    domains = initial_domains;
    std::fill(assignment.begin(), assignment.end(), UNASSIGNED);
}

bool FastSolver::is_timed_out() {
    if (timeout_flag) return true;
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
        timeout_flag = true;
        return true;
    }
    return false;
}

ResultStatus FastSolver::solve(SolveMode m, const std::vector<int>& scope, double timeout_sec) {
    mode = m;
    time_limit_sec = timeout_sec;
    timeout_flag = false;
    nodes_visited = 0;
    backtracks = 0;
    best_score = -1;

    reset_domains();

    if (scope.empty()) {
        target_scope.resize(num_vars);
        for (int i = 0; i < num_vars; ++i) target_scope[i] = i;
        in_target_scope.assign(num_vars, true);
    } else {
        target_scope = scope;
        in_target_scope.assign(num_vars, false);
        for (int v : target_scope) {
            if (v >= 0 && v < num_vars) in_target_scope[v] = true;
        }
    }

    deadline = std::chrono::steady_clock::now() +
               std::chrono::milliseconds(static_cast<long long>(timeout_sec * 1000.0));

    // Initial AC3 on active variables at root level 0
    if (!propagate(0, -1)) {
        return STATUS_UNSAT;
    }

    bool success = mac(1);

    if (timeout_flag) {
        if (mode == SOLVE_MAX_BIAS && best_score >= 0) {
            assignment = best_assignment;
            return STATUS_SAT;
        }
        return STATUS_TIMEOUT;
    }

    if (mode == SOLVE_MAX_BIAS) {
        if (best_score >= 0) {
            assignment = best_assignment;
            return STATUS_SAT;
        }
        return STATUS_UNSAT;
    }

    return success ? STATUS_SAT : STATUS_UNSAT;
}

int FastSolver::select_variable() {
    int best_var = -1;
    double best_ratio = 1e18;
    int min_domain_size = INT_MAX;

    for (int v : target_scope) {
        if (assignment[v] != UNASSIGNED) continue;
        int d_size = domains[v].size();
        if (d_size == 0) return -1; // Wipeout

        if (var_heuristic == VAR_DOM_WDEG) {
            double w = calc_wdeg(v);
            double ratio = static_cast<double>(d_size) / (w + 1e-4);
            if (ratio < best_ratio) {
                best_ratio = ratio;
                best_var = v;
            }
        } else if (var_heuristic == VAR_DOM) {
            if (d_size < min_domain_size) {
                min_domain_size = d_size;
                best_var = v;
            }
        } else if (var_heuristic == VAR_LEXICO) {
            return v;
        } else { // VAR_BDEG
            int bdeg = (v < static_cast<int>(bias.var_to_cons.size())) ? static_cast<int>(bias.var_to_cons[v].size()) : 0;
            if (-bdeg < best_ratio) {
                best_ratio = -bdeg;
                best_var = v;
            }
        }
    }

    return best_var;
}

double FastSolver::calc_wdeg(int var) const {
    double w = 1.0;
    if (var >= static_cast<int>(cl.var_to_cons.size())) return w;

    for (int c_idx : cl.var_to_cons[var]) {
        const auto& c = cl.get(c_idx);
        bool has_unassigned_other = false;
        for (int other : c.scope) {
            if (other != var && in_target_scope[other] && assignment[other] == UNASSIGNED) {
                has_unassigned_other = true;
                break;
            }
        }
        if (has_unassigned_other) {
            w += c.weight;
        }
    }
    return w;
}

int FastSolver::eval_val_bias_violations(int var, int val) const {
    int count = 0;
    if (var >= static_cast<int>(bias.var_to_cons.size())) return 0;

    for (int c_idx : bias.var_to_cons[var]) {
        const auto& c = bias.get(c_idx);
        bool ready = true;
        for (int other : c.scope) {
            if (other != var && (other >= num_vars || assignment[other] == UNASSIGNED)) {
                ready = false;
                break;
            }
        }
        if (ready) {
            // Test temporary assignment
            std::vector<int> temp_scope_vals(c.scope.size());
            bool match = true;
            for (size_t i = 0; i < c.scope.size(); ++i) {
                int sv = c.scope[i];
                int v = (sv == var) ? val : assignment[sv];
                if (v == UNASSIGNED) { match = false; break; }
                temp_scope_vals[i] = v;
            }
            if (match) {
                // Check if violated
                std::vector<int> dummy(num_vars, UNASSIGNED);
                for (size_t i = 0; i < c.scope.size(); ++i) {
                    dummy[c.scope[i]] = temp_scope_vals[i];
                }
                if (c.is_violated(dummy)) {
                    ++count;
                }
            }
        }
    }
    return count;
}

std::vector<int> FastSolver::order_values(int var) {
    const auto& dom = domains[var];
    std::vector<int> active_vals;
    active_vals.reserve(dom.size());

    for (int i = 0; i < dom.total_size(); ++i) {
        if (dom.is_valid_idx(i)) {
            active_vals.push_back(dom.get_val(i));
        }
    }

    if (val_heuristic == VAL_MAX_BIAS_VIOL && bias.size() > 0) {
        std::vector<std::pair<int, int>> scored;
        scored.reserve(active_vals.size());
        for (int val : active_vals) {
            int score = eval_val_bias_violations(var, val);
            scored.push_back({score, val});
        }
        std::sort(scored.begin(), scored.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first > b.first;
        });
        for (size_t i = 0; i < active_vals.size(); ++i) {
            active_vals[i] = scored[i].second;
        }
    } else if (val_heuristic == VAL_RANDOM) {
        static std::mt19937 rng(1337);
        std::shuffle(active_vals.begin(), active_vals.end(), rng);
    }

    return active_vals;
}

bool FastSolver::revise(int con_idx, int var, int level) {
    auto& c = cl.get(con_idx);
    auto& dom = domains[var];
    bool pruned_any = false;
    (void)pruned_any;

    if (c.scope.size() == 1) { // Unary
        for (int i = 0; i < dom.total_size(); ++i) {
            if (dom.is_valid_idx(i)) {
                int val = dom.get_val(i);
                if (!c.eval_unary(val)) {
                    dom.prune_idx(i, level);
                    pruned_any = true;
                }
            }
        }
    } else if (c.scope.size() == 2) { // Binary
        int other = (c.scope[0] == var) ? c.scope[1] : c.scope[0];
        bool is_var_first = (c.scope[0] == var);

        if (assignment[other] != UNASSIGNED) {
            int other_val = assignment[other];
            for (int i = 0; i < dom.total_size(); ++i) {
                if (dom.is_valid_idx(i)) {
                    int val = dom.get_val(i);
                    bool sat = is_var_first ? c.eval_binary(val, other_val)
                                            : c.eval_binary(other_val, val);
                    if (!sat) {
                        dom.prune_idx(i, level);
                        pruned_any = true;
                    }
                }
            }
        } else {
            // other is unassigned, check support in domains[other]
            const auto& other_dom = domains[other];
            for (int i = 0; i < dom.total_size(); ++i) {
                if (dom.is_valid_idx(i)) {
                    int val = dom.get_val(i);
                    bool has_support = false;
                    for (int j = 0; j < other_dom.total_size(); ++j) {
                        if (other_dom.is_valid_idx(j)) {
                            int oval = other_dom.get_val(j);
                            bool sat = is_var_first ? c.eval_binary(val, oval)
                                                    : c.eval_binary(oval, val);
                            if (sat) {
                                has_support = true;
                                break;
                            }
                        }
                    }
                    if (!has_support) {
                        dom.prune_idx(i, level);
                        pruned_any = true;
                    }
                }
            }
        }
    } else {
        // Higher arity: if all other variables in scope are assigned, prune non-supporting values
        bool all_others_assigned = true;
        for (int sv : c.scope) {
            if (sv != var && assignment[sv] == UNASSIGNED) {
                all_others_assigned = false;
                break;
            }
        }
        if (all_others_assigned) {
            std::vector<int> test_asgn = assignment;
            for (int i = 0; i < dom.total_size(); ++i) {
                if (dom.is_valid_idx(i)) {
                    test_asgn[var] = dom.get_val(i);
                    if (c.is_violated(test_asgn)) {
                        dom.prune_idx(i, level);
                        pruned_any = true;
                    }
                }
            }
        }
    }

    if (dom.empty()) {
        c.weight++; // Increase weight for dom/wdeg
        return false; // Wipeout!
    }

    return true;
}

bool FastSolver::propagate(int level, int assigned_var) {
    std::queue<std::pair<int, int>> q; // (constraint_idx, var_to_revise)

    if (assigned_var >= 0 && assigned_var < static_cast<int>(cl.var_to_cons.size())) {
        for (int c_idx : cl.var_to_cons[assigned_var]) {
            const auto& c = cl.get(c_idx);
            for (int sv : c.scope) {
                if (sv != assigned_var && in_target_scope[sv] && assignment[sv] == UNASSIGNED) {
                    q.push({c_idx, sv});
                }
            }
        }
    } else if (assigned_var == -1) {
        for (int c_idx = 0; c_idx < cl.size(); ++c_idx) {
            const auto& c = cl.get(c_idx);
            for (int sv : c.scope) {
                if (in_target_scope[sv] && assignment[sv] == UNASSIGNED) {
                    q.push({c_idx, sv});
                }
            }
        }
    }

    while (!q.empty()) {
        auto arc = q.front();
        q.pop();
        int c_idx = arc.first;
        int var = arc.second;

        int prev_size = domains[var].size();
        if (!revise(c_idx, var, level)) {
            return false; // Domain wipeout
        }

        if (domains[var].size() < prev_size) {
            // Domain changed, push affected neighbor arcs
            if (var < static_cast<int>(cl.var_to_cons.size())) {
                for (int next_c_idx : cl.var_to_cons[var]) {
                    if (next_c_idx == c_idx) continue;
                    const auto& nc = cl.get(next_c_idx);
                    for (int sv : nc.scope) {
                        if (sv != var && in_target_scope[sv] && assignment[sv] == UNASSIGNED) {
                            q.push({next_c_idx, sv});
                        }
                    }
                }
            }
        }
    }

    return true;
}

bool FastSolver::mac(int level) {
    if (is_timed_out()) return false;
    ++nodes_visited;

    int var = select_variable();
    if (var == -1) {
        // All variables in target_scope are assigned!
        if (mode == SOLVE_ANY) {
            return true;
        } else if (mode == SOLVE_VIOLATE_BIAS) {
            int rejects = bias.count_rejects(assignment);
            return rejects >= 1;
        } else if (mode == SOLVE_MAX_BIAS) {
            int rejects = bias.count_rejects(assignment);
            if (rejects > best_score) {
                best_score = rejects;
                best_assignment = assignment;
                if (best_score == bias.size()) {
                    return true; // Reached maximum possible
                }
            }
            return false; // Continue search for better
        } else if (mode == SOLVE_FINDC_SPLITHALF) {
            int rejects = bias.count_rejects(assignment);
            if (rejects > 0 && rejects < bias.size()) {
                return true;
            }
            return false;
        }
        return true;
    }

    std::vector<int> vals = order_values(var);

    for (int val : vals) {
        if (is_timed_out()) break;

        assignment[var] = val;

        // Prune all other values from domains[var] at this level
        auto& dom = domains[var];
        for (int i = 0; i < dom.total_size(); ++i) {
            if (dom.is_valid_idx(i) && dom.get_val(i) != val) {
                dom.prune_idx(i, level);
            }
        }

        bool ok = propagate(level, var);
        if (ok) {
            if (mac(level + 1)) {
                return true;
            }
        }

        // Backtrack
        ++backtracks;
        for (int v : target_scope) {
            domains[v].restore(level);
        }
        assignment[var] = UNASSIGNED;
    }

    return false;
}

} // namespace quacq
