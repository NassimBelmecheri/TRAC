#include "quacq.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <chrono>

namespace quacq {

QuAcqEngine::QuAcqEngine(int n_vars)
    : num_vars(n_vars),
      solver(n_vars),
      cl(n_vars),
      bias(n_vars),
      target_ct(n_vars),
      has_target_ct(false),
      oracle_cb(nullptr),
      oracle_user_data(nullptr),
      query_timeout_sec(5.0),
      findscope_version(FINDSCOPE_V2),
      findc_version(FINDC_V1) {
    metrics = {0, 0, 0, 0, 0.0, 0.0, 0};
    if (n_vars > 0) {
        set_num_vars(n_vars);
    }
}

void QuAcqEngine::set_num_vars(int n_vars) {
    num_vars = n_vars;
    solver.set_num_vars(n_vars);
    cl.set_num_vars(n_vars);
    bias.set_num_vars(n_vars);
    target_ct.set_num_vars(n_vars);
}

void QuAcqEngine::set_domain(int var, const std::vector<int>& raw_vals) {
    solver.set_domain(var, raw_vals);
}

void QuAcqEngine::add_cl_constraint(const Constraint& c) {
    cl.add(c);
}

void QuAcqEngine::add_bias_constraint(const Constraint& c) {
    bias.add(c);
}

void QuAcqEngine::set_target_network(const ConstraintNet& ct) {
    target_ct = ct;
    has_target_ct = true;
}

void QuAcqEngine::set_oracle(OracleCallback cb, void* user_data) {
    oracle_cb = cb;
    oracle_user_data = user_data;
}

bool QuAcqEngine::ask_oracle(const std::vector<int>& assignment, const std::vector<int>& scope) {
    metrics.membership_queries++;

    if (oracle_cb != nullptr) {
        return oracle_cb(assignment.data(), static_cast<int>(assignment.size()),
                         scope.data(), static_cast<int>(scope.size()),
                         oracle_user_data) != 0;
    }

    if (has_target_ct) {
        std::vector<int> projected(num_vars, UNASSIGNED);
        for (int v : scope) {
            if (v >= 0 && v < num_vars) {
                projected[v] = assignment[v];
            }
        }
        for (const auto& c : target_ct.constraints) {
            bool all_in = true;
            for (int sv : c.scope) {
                if (projected[sv] == UNASSIGNED) {
                    all_in = false;
                    break;
                }
            }
            if (all_in && c.is_violated(projected)) {
                return false;
            }
        }
        return true;
    }

    return true;
}

std::vector<int> QuAcqEngine::generate_query(const std::vector<int>& scope) {
    solver.set_cl(cl);
    solver.set_bias(bias);

    auto t0 = std::chrono::steady_clock::now();
    ResultStatus status = solver.solve(SOLVE_VIOLATE_BIAS, scope, query_timeout_sec);
    auto t1 = std::chrono::steady_clock::now();

    metrics.solve_time_sec += std::chrono::duration<double>(t1 - t0).count();

    if (status == STATUS_SAT) {
        return solver.get_assignment();
    }
    return {};
}

std::vector<int> QuAcqEngine::generate_tqgen_query(const std::vector<int>& scope, double tau, double alpha) {
    std::vector<int> Y = scope.empty() ? std::vector<int>(num_vars) : scope;
    if (scope.empty()) {
        for (int i = 0; i < num_vars; ++i) Y[i] = i;
    }

    double current_tau = tau;
    while (!Y.empty()) {
        solver.set_cl(cl);
        solver.set_bias(bias);
        ResultStatus status = solver.solve(SOLVE_VIOLATE_BIAS, Y, current_tau);
        if (status == STATUS_SAT) {
            return solver.get_assignment();
        }
        int new_size = std::max(1, static_cast<int>(std::floor(Y.size() * alpha)));
        if (new_size >= static_cast<int>(Y.size())) break;
        Y.resize(new_size);
    }
    return {};
}

int QuAcqEngine::run_algorithm(AlgorithmType algo, int max_queries) {
    switch (algo) {
        case ALGO_QUACQ: return run_quacq(max_queries);
        case ALGO_PQUACQ: return run_pquacq(max_queries);
        case ALGO_MQUACQ: return run_mquacq(max_queries);
        case ALGO_MQUACQ2: return run_mquacq2(max_queries);
        case ALGO_GROWACQ: return run_growacq(max_queries, ALGO_QUACQ);
        case ALGO_BRUTECA: return run_bruteca(max_queries);
        default: return run_quacq(max_queries);
    }
}

int QuAcqEngine::run_quacq(int max_queries) {
    auto start_all = std::chrono::steady_clock::now();
    std::vector<int> all_vars(num_vars);
    for (int i = 0; i < num_vars; ++i) all_vars[i] = i;

    while (metrics.membership_queries < max_queries) {
        if (bias.size() == 0) {
            metrics.converged = 1;
            break;
        }

        std::vector<int> query = generate_query(all_vars);
        if (query.empty()) {
            metrics.converged = 1;
            break;
        }

        bool answer = ask_oracle(query, all_vars);
        if (answer) {
            bias.remove_rejects(query);
        } else {
            std::vector<int> scope = find_scope(query, all_vars);
            if (!scope.empty()) {
                if (findc_version == FINDC_V2) {
                    std::vector<Constraint> cons = find_c_v2(scope, query);
                    for (const auto& c : cons) {
                        cl.add(c);
                        std::vector<Constraint> remaining_bias;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                remaining_bias.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(remaining_bias);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                    }
                    remove_scope_from_bias(scope);
                } else {
                    Constraint c = find_c(scope, query);
                    if (c.id >= 0 || c.type >= 0) {
                        cl.add(c);
                        std::vector<Constraint> remaining_bias;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                remaining_bias.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(remaining_bias);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                    }
                }
            } else {
                bias.remove_rejects(query);
            }
        }
    }

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

int QuAcqEngine::run_pquacq(int max_queries, int alpha_cutoff) {
    auto start_all = std::chrono::steady_clock::now();
    std::vector<int> all_vars(num_vars);
    for (int i = 0; i < num_vars; ++i) all_vars[i] = i;

    while (metrics.membership_queries < max_queries) {
        if (bias.size() == 0) {
            metrics.converged = 1;
            break;
        }

        std::vector<int> query = generate_query(all_vars);
        if (query.empty()) {
            metrics.converged = 1;
            break;
        }

        bool answer = ask_oracle(query, all_vars);
        if (answer) {
            bias.remove_rejects(query);
        } else {
            std::vector<int> scope = find_scope(query, all_vars);
            if (!scope.empty()) {
                if (findc_version == FINDC_V2) {
                    std::vector<Constraint> cons = find_c_v2(scope, query);
                    for (const auto& c : cons) {
                        cl.add(c);
                        std::vector<Constraint> remaining_bias;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                remaining_bias.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(remaining_bias);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                        predict_and_ask(c.type, alpha_cutoff);
                    }
                    remove_scope_from_bias(scope);
                } else {
                    Constraint c = find_c(scope, query);
                    if (c.id >= 0 || c.type >= 0) {
                        cl.add(c);
                        std::vector<Constraint> remaining_bias;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                remaining_bias.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(remaining_bias);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                        predict_and_ask(c.type, alpha_cutoff);
                    }
                }
            } else {
                bias.remove_rejects(query);
            }
        }
    }

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

double QuAcqEngine::compute_adamic_adar(const std::vector<std::vector<bool>>& adj,
                                        const std::vector<int>& degrees,
                                        int u, int v) const {
    double score = 0.0;
    for (int w = 0; w < num_vars; ++w) {
        if (w != u && w != v && adj[u][w] && adj[v][w]) {
            int deg = degrees[w];
            if (deg > 1) {
                score += 1.0 / std::log(deg);
            }
        }
    }
    return score;
}

int QuAcqEngine::predict_and_ask(int relation_type, int alpha_cutoff) {
    std::vector<std::vector<bool>> adj(num_vars, std::vector<bool>(num_vars, false));
    std::vector<int> degrees(num_vars, 0);

    for (const auto& c : cl.constraints) {
        if (c.type == relation_type && c.scope.size() == 2) {
            int u = c.scope[0];
            int v = c.scope[1];
            if (!adj[u][v]) {
                adj[u][v] = true;
                adj[v][u] = true;
                degrees[u]++;
                degrees[v]++;
            }
        }
    }

    int neg_count = 0;
    while (neg_count < alpha_cutoff) {
        int best_idx = -1;
        double best_score = -1.0;

        for (size_t i = 0; i < bias.constraints.size(); ++i) {
            const auto& bc = bias.constraints[i];
            if (bc.type == relation_type && bc.scope.size() == 2) {
                int u = bc.scope[0];
                int v = bc.scope[1];
                if (degrees[u] > 0 && degrees[v] > 0) {
                    double s = compute_adamic_adar(adj, degrees, u, v);
                    if (s > best_score) {
                        best_score = s;
                        best_idx = static_cast<int>(i);
                    }
                }
            }
        }

        if (best_idx == -1 || best_score <= 0.0) break;

        Constraint cand = bias.constraints[best_idx];
        bias.constraints.erase(bias.constraints.begin() + best_idx);
        bias.rebuild_index();

        bool accepted = true;
        if (has_target_ct) {
            bool found = false;
            for (const auto& tc : target_ct.constraints) {
                if (tc.type == cand.type && tc.scope == cand.scope && tc.param == cand.param) {
                    found = true;
                    break;
                }
            }
            accepted = found;
        }

        if (accepted) {
            cl.add(cand);
            metrics.learned_constraints++;
            int u = cand.scope[0];
            int v = cand.scope[1];
            adj[u][v] = true;
            adj[v][u] = true;
            degrees[u]++;
            degrees[v]++;
        } else {
            neg_count++;
        }
    }
    return neg_count;
}

int QuAcqEngine::run_mquacq(int max_queries) {
    auto start_all = std::chrono::steady_clock::now();
    std::vector<int> all_vars(num_vars);
    for (int i = 0; i < num_vars; ++i) all_vars[i] = i;

    while (metrics.membership_queries < max_queries) {
        if (bias.size() == 0) {
            metrics.converged = 1;
            break;
        }

        std::vector<int> query = generate_query(all_vars);
        if (query.empty()) {
            metrics.converged = 1;
            break;
        }

        // Find ALL constraints this query violates: locate a violated scope,
        // learn it, then keep scanning the remaining variables for more. This
        // learns the same network as QuAcq with fewer generated queries.
        std::vector<int> Y = all_vars;
        while (Y.size() > 1 && metrics.membership_queries < max_queries) {
            std::vector<int> proj(num_vars, UNASSIGNED);
            for (int v : Y) proj[v] = query[v];
            if (bias.count_rejects(proj) == 0) break;

            bool ans = ask_oracle(query, Y);
            if (ans) { bias.remove_rejects(proj); break; }

            std::vector<int> scope = find_scope(query, Y);
            if (scope.empty()) { bias.remove_rejects(proj); break; }

            if (findc_version == FINDC_V2) {
                std::vector<Constraint> cons = find_c_v2(scope, query);
                for (const auto& c : cons) {
                    cl.add(c);
                    std::vector<Constraint> rem;
                    for (const auto& bc : bias.constraints) {
                        if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) rem.push_back(bc);
                    }
                    bias.constraints = std::move(rem);
                    bias.rebuild_index();
                    metrics.learned_constraints++;
                }
                remove_scope_from_bias(scope);
            } else {
                Constraint c = find_c(scope, query);
                if (c.id >= 0 || c.type >= 0) {
                    cl.add(c);
                    std::vector<Constraint> rem;
                    for (const auto& bc : bias.constraints) {
                        if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) rem.push_back(bc);
                    }
                    bias.constraints = std::move(rem);
                    bias.rebuild_index();
                    metrics.learned_constraints++;
                }
            }

            std::unordered_set<int> sc(scope.begin(), scope.end());
            std::vector<int> nextY;
            for (int v : Y) if (sc.find(v) == sc.end()) nextY.push_back(v);
            if (nextY.size() == Y.size()) break;
            Y = std::move(nextY);
        }
    }

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

void QuAcqEngine::mquacq_find_all_cons(std::vector<int> Y,
                                      std::set<std::vector<int>>& scopes,
                                      int max_queries) {
    if (metrics.membership_queries >= max_queries) return;

    int rej = bias.count_rejects(Y);
    if (rej == 0) return;

    if (!scopes.empty()) {
        auto it = scopes.begin();
        std::vector<int> s = *it;
        scopes.erase(it);

        for (int x : s) {
            std::vector<int> Y2;
            for (int y_val : Y) {
                if (y_val != x) Y2.push_back(y_val);
            }
            mquacq_find_all_cons(Y2, scopes, max_queries);
        }
    } else {
        bool ans = ask_oracle(Y, Y);
        if (ans) {
            bias.remove_rejects(Y);
        } else {
            std::vector<int> scope = find_scope(Y, Y);
            if (!scope.empty()) {
                if (findc_version == FINDC_V2) {
                    std::vector<Constraint> cons = find_c_v2(scope, Y);
                    for (const auto& c : cons) {
                        cl.add(c);
                        std::vector<Constraint> rem;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                rem.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(rem);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                    }
                    scopes.insert(scope);
                    mquacq_find_all_cons(Y, scopes, max_queries);
                } else {
                    Constraint c = find_c(scope, Y);
                    if (c.id >= 0 || c.type >= 0) {
                        cl.add(c);
                        std::vector<Constraint> rem;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                rem.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(rem);
                        bias.rebuild_index();
                        metrics.learned_constraints++;

                        scopes.insert(scope);
                        mquacq_find_all_cons(Y, scopes, max_queries);
                    }
                }
            } else {
                bias.remove_rejects(Y);
            }
        }
    }
}

int QuAcqEngine::run_mquacq2(int max_queries) {
    auto start_all = std::chrono::steady_clock::now();
    std::vector<int> all_vars(num_vars);
    for (int i = 0; i < num_vars; ++i) all_vars[i] = i;

    while (metrics.membership_queries < max_queries) {
        if (bias.size() == 0) {
            metrics.converged = 1;
            break;
        }

        std::vector<int> query = generate_query(all_vars);
        if (query.empty()) {
            metrics.converged = 1;
            break;
        }

        std::vector<int> current_Y = all_vars;
        while (current_Y.size() > 1 && metrics.membership_queries < max_queries) {
            std::vector<int> proj(num_vars, UNASSIGNED);
            for (int v : current_Y) proj[v] = query[v];
            // Stop once the query no longer violates any bias constraint on
            // current_Y. NB: count_rejects expects the projected *assignment*, not
            // the list of variable indices - passing current_Y here made the loop
            // never execute (membership_queries never advanced -> infinite loop).
            if (bias.count_rejects(proj) == 0) break;

            bool ans = ask_oracle(query, current_Y);
            if (ans) {
                bias.remove_rejects(proj);
                break;
            } else {
                std::vector<int> scope = find_scope(query, current_Y);
                if (scope.empty()) {
                    bias.remove_rejects(proj);
                    break;
                }

                if (findc_version == FINDC_V2) {
                    std::vector<Constraint> cons = find_c_v2(scope, query);
                    for (const auto& c : cons) {
                        cl.add(c);
                        std::vector<Constraint> rem;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                rem.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(rem);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                    }
                    remove_scope_from_bias(scope);
                } else {
                    Constraint c = find_c(scope, query);
                    if (c.id >= 0 || c.type >= 0) {
                        cl.add(c);
                        std::vector<Constraint> rem;
                        for (const auto& bc : bias.constraints) {
                            if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                rem.push_back(bc);
                            }
                        }
                        bias.constraints = std::move(rem);
                        bias.rebuild_index();
                        metrics.learned_constraints++;
                    }
                }

                std::unordered_set<int> sc_set(scope.begin(), scope.end());
                std::vector<int> next_Y;
                for (int v : current_Y) {
                    if (sc_set.find(v) == sc_set.end()) {
                        next_Y.push_back(v);
                    }
                }
                if (next_Y.size() == current_Y.size() || next_Y.size() <= 1) {
                    break;
                }
                current_Y = std::move(next_Y);
            }
        }
    }

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

int QuAcqEngine::run_growacq(int max_queries, AlgorithmType inner_algo) {
    auto start_all = std::chrono::steady_clock::now();
    double prev_timeout = query_timeout_sec;
    query_timeout_sec = 0.2; // Standard TQGen cutoff for incremental GrowAcq

    ConstraintNet full_bias = bias;
    bias.clear();
    bias.set_num_vars(num_vars);

    std::vector<int> current_vars;
    current_vars.push_back(0);

    for (int v = 1; v < num_vars && metrics.membership_queries < max_queries; ++v) {
        current_vars.push_back(v);
        std::unordered_set<int> var_set(current_vars.begin(), current_vars.end());

        for (const auto& c : full_bias.constraints) {
            bool in_scope = true;
            bool has_v = false;
            for (int sv : c.scope) {
                if (var_set.find(sv) == var_set.end()) { in_scope = false; break; }
                if (sv == v) has_v = true;
            }
            if (in_scope && has_v) {
                bias.add(c);
            }
        }

        while (bias.size() > 0 && metrics.membership_queries < max_queries) {
            std::vector<int> query = generate_query(current_vars);
            if (query.empty()) {
                // The 0.2s TQGen cutoff can fail to find a violating query for a
                // *nearly*-entailed constraint even though one exists (near
                // convergence C_L is highly constraining, e.g. golomb's ruler).
                // Before giving up on this increment, confirm genuine convergence
                // with a longer solve - otherwise growacq stops early and its
                // network is too weak (a solution of it can violate the target).
                double save = query_timeout_sec;
                query_timeout_sec = std::max(5.0, prev_timeout);
                query = generate_query(current_vars);
                query_timeout_sec = save;
                if (query.empty()) break;
            }

            bool answer = ask_oracle(query, current_vars);
            if (answer) {
                std::vector<int> proj(num_vars, UNASSIGNED);
                for (int x : current_vars) proj[x] = query[x];
                bias.remove_rejects(proj);
            } else {
                std::vector<int> scope = find_scope(query, current_vars);
                if (!scope.empty()) {
                    if (findc_version == FINDC_V2) {
                        std::vector<Constraint> cons = find_c_v2(scope, query);
                        for (const auto& c : cons) {
                            cl.add(c);
                            std::vector<Constraint> rem;
                            for (const auto& bc : bias.constraints) {
                                if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                    rem.push_back(bc);
                                }
                            }
                            bias.constraints = std::move(rem);
                            bias.rebuild_index();
                            metrics.learned_constraints++;
                        }
                        remove_scope_from_bias(scope);
                    } else {
                        Constraint c = find_c(scope, query);
                        if (c.id >= 0 || c.type >= 0) {
                            cl.add(c);
                            std::vector<Constraint> rem;
                            for (const auto& bc : bias.constraints) {
                                if (!(bc.type == c.type && bc.scope == c.scope && bc.param == c.param)) {
                                    rem.push_back(bc);
                                }
                            }
                            bias.constraints = std::move(rem);
                            bias.rebuild_index();
                            metrics.learned_constraints++;
                        }
                    }
                } else {
                    std::vector<int> proj(num_vars, UNASSIGNED);
                    for (int x : current_vars) proj[x] = query[x];
                    bias.remove_rejects(proj);
                }
            }
        }
    }

    // Converged: all variables were introduced and their bias emptied within
    // the query budget.
    if (metrics.membership_queries < max_queries) {
        metrics.converged = 1;
    }

    query_timeout_sec = prev_timeout;
    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

int QuAcqEngine::run_bruteca(int max_queries) {
    auto start_all = std::chrono::steady_clock::now();

    std::sort(bias.constraints.begin(), bias.constraints.end(), [](const Constraint& a, const Constraint& b) {
        return a.scope.size() < b.scope.size();
    });
    bias.rebuild_index();

    std::vector<bool> active(bias.constraints.size(), true);

    for (size_t idx = 0; idx < bias.constraints.size() && metrics.membership_queries < max_queries; ++idx) {
        if (!active[idx]) continue;
        Constraint target_c = bias.constraints[idx];
        active[idx] = false;

        std::vector<int> sc = target_c.scope;
        std::vector<int> query(num_vars, UNASSIGNED);
        bool found_violation = false;
        ConstraintNet sub_cl = cl.get_subnetwork(sc);

        for (int tries = 0; tries < 100; ++tries) {
            for (int v : sc) {
                const auto& d = solver.initial_domains[v];
                if (d.size() > 0) {
                    query[v] = d.get_val(rand() % d.total_size());
                }
            }
            if (target_c.is_violated(query) && sub_cl.is_valid_solution(query)) {
                found_violation = true;
                break;
            }
        }

        if (!found_violation) {
            ConstraintNet temp_bias(num_vars);
            temp_bias.add(target_c);
            FastSolver s(num_vars);
            s.initial_domains = solver.initial_domains;
            s.set_cl(sub_cl);
            s.set_bias(temp_bias);
            ResultStatus res = s.solve(SOLVE_VIOLATE_BIAS, sc, 0.05);
            if (res == STATUS_SAT) {
                query = s.get_assignment();
                found_violation = true;
            }
        }

        if (!found_violation) {
            continue;
        }

        bool ans = ask_oracle(query, sc);

        if (ans) {
            for (size_t j = idx + 1; j < bias.constraints.size(); ++j) {
                if (active[j] && bias.constraints[j].is_violated(query)) {
                    active[j] = false;
                }
            }
        } else {
            if (findc_version == FINDC_V2) {
                std::vector<Constraint> cons = find_c_v2(sc, query);
                for (const auto& c : cons) {
                    cl.add(c);
                    metrics.learned_constraints++;
                }
            } else {
                Constraint c = find_c(sc, query);
                if (c.id >= 0 || c.type >= 0) {
                    cl.add(c);
                    metrics.learned_constraints++;
                }
            }

            std::unordered_set<int> sc_set(sc.begin(), sc.end());
            for (size_t j = idx + 1; j < bias.constraints.size(); ++j) {
                if (active[j] && bias.constraints[j].scope.size() == sc.size()) {
                    bool match = true;
                    for (int sv : bias.constraints[j].scope) {
                        if (sc_set.find(sv) == sc_set.end()) { match = false; break; }
                    }
                    if (match) active[j] = false;
                }
            }
        }
    }

    // Converged: every bias candidate was processed within the query budget.
    if (metrics.membership_queries < max_queries) {
        metrics.converged = 1;
    }

    std::vector<Constraint> kept;
    for (size_t i = 0; i < bias.constraints.size(); ++i) {
        if (active[i]) kept.push_back(bias.constraints[i]);
    }
    bias.constraints = std::move(kept);
    bias.rebuild_index();

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

int QuAcqEngine::run_conacq1(const std::vector<std::vector<int>>& positives,
                             const std::vector<std::vector<int>>&) {
    auto start_all = std::chrono::steady_clock::now();

    for (const auto& pos : positives) {
        bias.remove_rejects(pos);
    }

    for (const auto& c : bias.constraints) {
        cl.add(c);
        metrics.learned_constraints++;
    }
    bias.clear();

    auto end_all = std::chrono::steady_clock::now();
    metrics.total_time_sec = std::chrono::duration<double>(end_all - start_all).count();
    return metrics.learned_constraints;
}

std::vector<int> QuAcqEngine::find_scope(const std::vector<int>& query, const std::vector<int>& Y) {
    if (findscope_version == FINDSCOPE_V1) {
        return find_scope_v1(query, Y);
    } else {
        return find_scope_v2(query, Y);
    }
}

std::vector<int> QuAcqEngine::find_scope_v1(const std::vector<int>& query, const std::vector<int>& Y) {
    return find_scope_v1_rec(query, {}, Y, false);
}

std::vector<int> QuAcqEngine::find_scope_v1_rec(const std::vector<int>& query,
                                               std::vector<int> R,
                                               std::vector<int> Y,
                                               bool do_ask) {
    if (do_ask) {
        metrics.findscope_queries++;
        bool ans = ask_oracle(query, R);
        if (ans) {
            std::vector<int> proj_R(num_vars, UNASSIGNED);
            for (int v : R) proj_R[v] = query[v];
            bias.remove_rejects(proj_R);
        } else {
            return {};
        }
    }

    if (Y.size() == 1) {
        return Y;
    }

    size_t mid = Y.size() / 2;
    std::vector<int> Y1(Y.begin(), Y.begin() + mid);
    std::vector<int> Y2(Y.begin() + mid, Y.end());

    std::vector<int> R_union_Y1 = R;
    for (int v : Y1) {
        if (std::find(R_union_Y1.begin(), R_union_Y1.end(), v) == R_union_Y1.end()) {
            R_union_Y1.push_back(v);
        }
    }

    std::vector<int> S1 = find_scope_v1_rec(query, R_union_Y1, Y2, true);

    std::vector<int> R_union_S1 = R;
    for (int v : S1) {
        if (std::find(R_union_S1.begin(), R_union_S1.end(), v) == R_union_S1.end()) {
            R_union_S1.push_back(v);
        }
    }

    std::vector<int> S2 = find_scope_v1_rec(query, R_union_S1, Y1, true);

    std::vector<int> result = S1;
    for (int v : S2) {
        if (std::find(result.begin(), result.end(), v) == result.end()) {
            result.push_back(v);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> QuAcqEngine::find_scope_v2(const std::vector<int>& query, const std::vector<int>& Y) {
    std::vector<int> current_kappa_b;
    std::vector<int> proj(num_vars, UNASSIGNED);
    for (int v : Y) proj[v] = query[v];
    current_kappa_b = bias.get_rejects(proj);

    return find_scope_v2_rec(query, {}, Y, current_kappa_b);
}

std::vector<int> QuAcqEngine::find_scope_v2_rec(const std::vector<int>& query,
                                               std::vector<int> R,
                                               std::vector<int> Y,
                                               std::vector<int>& current_kappa_b) {
    if (!R.empty()) {
        std::vector<int> proj_R(num_vars, UNASSIGNED);
        for (int v : R) proj_R[v] = query[v];
        int r_rejects = bias.count_rejects(proj_R);

        if (r_rejects > 0) {
            metrics.findscope_queries++;
            bool ans = ask_oracle(query, R);
            if (ans) {
                bias.remove_rejects(proj_R);
                std::vector<int> proj_RY(num_vars, UNASSIGNED);
                for (int v : R) proj_RY[v] = query[v];
                for (int v : Y) proj_RY[v] = query[v];
                current_kappa_b = bias.get_rejects(proj_RY);
            } else {
                return {};
            }
        }
    }

    if (Y.size() == 1) {
        return Y;
    }

    size_t mid = Y.size() / 2;
    std::vector<int> Y1(Y.begin(), Y.begin() + mid);
    std::vector<int> Y2(Y.begin() + mid, Y.end());

    std::vector<int> RY1 = R;
    for (int v : Y1) {
        if (std::find(RY1.begin(), RY1.end(), v) == RY1.end()) {
            RY1.push_back(v);
        }
    }

    std::vector<int> proj_RY1(num_vars, UNASSIGNED);
    for (int v : RY1) proj_RY1[v] = query[v];
    int k_RY1 = bias.count_rejects(proj_RY1);

    std::vector<int> S1;
    if (k_RY1 < static_cast<int>(current_kappa_b.size())) {
        S1 = find_scope_v2_rec(query, RY1, Y2, current_kappa_b);
    }

    std::vector<int> RS1 = R;
    for (int v : S1) {
        if (std::find(RS1.begin(), RS1.end(), v) == RS1.end()) {
            RS1.push_back(v);
        }
    }

    std::vector<int> proj_RS1(num_vars, UNASSIGNED);
    for (int v : RS1) proj_RS1[v] = query[v];
    int k_RS1 = bias.count_rejects(proj_RS1);

    std::vector<int> S2;
    if (k_RS1 < static_cast<int>(current_kappa_b.size())) {
        S2 = find_scope_v2_rec(query, RS1, Y1, current_kappa_b);
    }

    std::vector<int> result = S1;
    for (int v : S2) {
        if (std::find(result.begin(), result.end(), v) == result.end()) {
            result.push_back(v);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

Constraint QuAcqEngine::find_c(const std::vector<int>& scope, const std::vector<int>& query) {
    return find_c_v1(scope, query);
}

Constraint QuAcqEngine::find_c_v1(const std::vector<int>& scope, const std::vector<int>& query) {
    metrics.findc_queries++;

    ConstraintNet delta(num_vars);
    std::unordered_set<int> scope_set(scope.begin(), scope.end());

    for (const auto& c : bias.constraints) {
        if (c.scope.size() == scope.size()) {
            bool match = true;
            for (int sv : c.scope) {
                if (scope_set.find(sv) == scope_set.end()) {
                    match = false;
                    break;
                }
            }
            if (match && c.is_violated(query)) {
                delta.add(c);
            }
        }
    }

    if (delta.size() == 1) return delta.get(0);
    if (delta.size() == 0) return Constraint();

    ConstraintNet sub_cl = cl.get_subnetwork(scope);

    while (delta.size() > 1) {
        FastSolver findc_solver(num_vars);
        findc_solver.initial_domains = solver.initial_domains;
        findc_solver.set_cl(sub_cl);
        findc_solver.set_bias(delta);

        // Look for an example that satisfies C_L on the scope AND splits delta
        // (violates some but not all candidates). If none exists, the surviving
        // candidates are equivalent w.r.t. C_L and we stop, returning one of them.
        //
        // We must NOT fall back to SOLVE_ANY on failure: a non-splitting example
        // does not shrink delta, so the loop would spin forever. This loop is also
        // not bounded by max_queries (ask_oracle is called inside it), which is
        // why the old SOLVE_ANY fallback caused hangs on non-normalised networks
        // (e.g. n-queens) where SPLITHALF legitimately becomes UNSAT.
        ResultStatus res = findc_solver.solve(SOLVE_FINDC_SPLITHALF, scope, query_timeout_sec);
        if (res != STATUS_SAT) break;

        std::vector<int> example = findc_solver.get_assignment();
        int rej = delta.count_rejects(example);
        if (rej <= 0 || rej >= static_cast<int>(delta.size())) break;  // not discriminating

        bool ans = ask_oracle(example, scope);

        if (ans) {
            delta.remove_rejects(example);
        } else {
            std::vector<Constraint> kept;
            for (const auto& c : delta.constraints) {
                if (c.is_violated(example)) {
                    kept.push_back(c);
                }
            }
            delta.constraints = std::move(kept);
            delta.rebuild_index();
        }
    }

    if (delta.size() > 0) return delta.get(0);
    return Constraint();
}

void QuAcqEngine::remove_scope_from_bias(const std::vector<int>& scope) {
    std::unordered_set<int> sset(scope.begin(), scope.end());
    std::vector<Constraint> kept;
    kept.reserve(bias.constraints.size());
    for (const auto& bc : bias.constraints) {
        // Remove candidates whose DISTINCT variable set equals the scope's set
        // (the same match find_c_v2 uses to build delta). Set equality - not
        // array length (which skips repeated-variable quaternaries) and not
        // subset (which would drop unconfirmed proper sub-scope constraints and
        // make the network too weak).
        std::unordered_set<int> cvars(bc.scope.begin(), bc.scope.end());
        bool same = (cvars.size() == sset.size());
        if (same) {
            for (int v : cvars) {
                if (sset.find(v) == sset.end()) { same = false; break; }
            }
        }
        if (!same) kept.push_back(bc);
    }
    bias.constraints = std::move(kept);
    bias.rebuild_index();
}

std::vector<Constraint> QuAcqEngine::find_c_v2(const std::vector<int>& scope_in, const std::vector<int>& query) {
    metrics.findc_queries++;

    // Work with the DISTINCT variables of the scope. A constraint's scope array
    // may repeat a variable - e.g. golomb's |x0-x1| != |x1-x2| has scope array
    // [0,1,1,2] (4 entries, 3 distinct vars) - and FindScope hands us the
    // distinct-variable SET, so candidates must be matched by variable set
    // (subset), never by array length. Matching on array length made FindC skip
    // exactly these repeated-variable quaternary constraints, so the query-gen
    // algorithms could never learn them and livelocked on golomb.
    std::vector<int> scope;
    {
        std::unordered_set<int> seen;
        for (int v : scope_in) if (seen.insert(v).second) scope.push_back(v);
    }
    std::unordered_set<int> scope_set(scope.begin(), scope.end());
    ConstraintNet delta(num_vars);

    for (const auto& c : bias.constraints) {
        // Delta = every candidate whose DISTINCT variable set is a SUBSET of the
        // scope. This catches repeated-variable quaternaries (scope array
        // [0,1,1,2] -> variable set {0,1,2}) that array-length matching skipped,
        // and also lets find_c confirm real sub-scope constraints that the query
        // happens to expose. Confirmed constraints are removed from the bias by
        // the caller (exact match); only the *exact-scope* refuted candidates are
        // swept by remove_scope_from_bias (set equality), so unconfirmed
        // sub-scope constraints survive to be handled on their own scope.
        bool match = true;
        for (int sv : c.scope) {
            if (scope_set.find(sv) == scope_set.end()) { match = false; break; }
        }
        if (match) delta.add(c);
    }

    if (delta.empty()) return {};

    ConstraintNet sub_cl = cl.get_subnetwork(scope);
    std::vector<Constraint> confirmed;

    // Exhaustive, query-based FindC for scopes small enough to enumerate. A bias
    // candidate is confirmed iff *every* oracle-accepted assignment on the scope
    // satisfies it (i.e. it is entailed by the target restricted to the scope).
    // This is sound AND complete on the scope: it recovers the full conjunction
    // and never keeps a spurious constraint - which the split-based heuristic
    // cannot guarantee on non-normalised scopes (n-queens diagonals, golomb
    // ordering). Larger scopes fall back to the iterative FindAllC loop below.
    long long prod = 1;
    const long long ENUM_LIMIT = 4096;
    for (int v : scope) {
        prod *= std::max(1, solver.initial_domains[v].total_size());
        if (prod > ENUM_LIMIT) break;
    }

    if (prod <= ENUM_LIMIT) {
        std::vector<std::vector<int>> accepted;
        std::vector<int> test_asgn(num_vars, UNASSIGNED);
        std::vector<int> pos(scope.size(), 0);
        while (true) {
            for (size_t k = 0; k < scope.size(); ++k) {
                test_asgn[scope[k]] = solver.initial_domains[scope[k]].get_val(pos[k]);
            }
            if (sub_cl.is_valid_solution(test_asgn) && ask_oracle(test_asgn, scope)) {
                accepted.push_back(test_asgn);
            }
            int k = static_cast<int>(scope.size()) - 1;
            while (k >= 0) {
                pos[k]++;
                if (pos[k] < solver.initial_domains[scope[k]].total_size()) break;
                pos[k] = 0;
                k--;
            }
            if (k < 0) break;
        }
        for (const auto& c : delta.constraints) {
            bool all_ok = !accepted.empty();
            for (const auto& acc : accepted) {
                if (c.is_violated(acc)) { all_ok = false; break; }
            }
            if (all_ok) confirmed.push_back(c);
        }
    } else {
        // Random-sampling confirmation for large scopes (the sampled analogue of
        // the exhaustive branch above, for scopes whose domain product is too big
        // to enumerate - e.g. golomb's quaternary |xi-xj| vs |xk-xl| over domain
        // 1..35). We draw assignments on the scope that respect the learned
        // context (`sub_cl`, e.g. the ordering), ask the oracle, and:
        //   * a candidate is REAL iff no oracle-accepted sample violates it
        //     (accepted assignments are genuine solutions, so a real constraint
        //     is never violated by one), and it is violated by at least one
        //     rejected sample (so it is actually active on this scope).
        // This is sound and, with enough samples, complete - and crucially it
        // never confirms a spurious order-relation (G<, G>, G==) on a distance
        // pairing, because among the accepted golomb solutions both distance
        // orderings occur, so any such spurious candidate is refuted. The old
        // split-half FindAllC could not reliably separate the ~60 candidates that
        // share one 4-variable scope and confirmed spurious constraints.
        const int MAX_ORACLE = 1200;              // oracle-query budget for this scope
        const long long MAX_TRIES = (long long)MAX_ORACLE * 200;
        std::vector<std::vector<int>> accepted;
        std::vector<char> active(delta.constraints.size(), 0);
        std::vector<int> test(num_vars, UNASSIGNED);
        int oracle_used = 0;

        for (long long t = 0; t < MAX_TRIES && oracle_used < MAX_ORACLE; ++t) {
            for (int v : scope) {
                const auto& d = solver.initial_domains[v];
                if (d.total_size() > 0) test[v] = d.get_val(rand() % d.total_size());
            }
            if (!sub_cl.is_valid_solution(test)) continue;   // respect the context
            ++oracle_used;
            if (ask_oracle(test, scope)) {
                accepted.push_back(test);
            } else {
                for (size_t i = 0; i < delta.constraints.size(); ++i) {
                    if (!active[i] && delta.constraints[i].is_violated(test)) active[i] = 1;
                }
            }
        }

        for (size_t i = 0; i < delta.constraints.size(); ++i) {
            if (!active[i]) continue;                    // never violated by a solution => not evidently real
            const auto& c = delta.constraints[i];
            bool real = true;
            for (const auto& a : accepted) {
                if (c.is_violated(a)) { real = false; break; }   // a genuine solution violates c => spurious
            }
            if (real) confirmed.push_back(c);
        }
    }

    return confirmed;
}

} // namespace quacq
