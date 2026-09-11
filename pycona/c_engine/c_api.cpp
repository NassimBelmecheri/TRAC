#include "c_api.h"
#include "quacq.h"
#include <vector>
#include <cstring>

using namespace quacq;

extern "C" {

EXPORT_API void* quacq_create(int num_vars) {
    return new QuAcqEngine(num_vars);
}

EXPORT_API void quacq_free(void* handle) {
    if (handle) {
        delete static_cast<QuAcqEngine*>(handle);
    }
}

EXPORT_API void quacq_set_domain(void* handle, int var, int count, const int* values) {
    if (!handle || !values || count <= 0) return;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> vals(values, values + count);
    engine->set_domain(var, vals);
}

EXPORT_API int quacq_add_constraint(void* handle, int is_bias, int type, int scope_size, const int* scope, int param) {
    if (!handle || !scope || scope_size <= 0) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc(scope, scope + scope_size);
    Constraint c(is_bias ? engine->bias.size() : engine->cl.size(), type, sc, param);
    if (is_bias) {
        return engine->bias.add(c);
    } else {
        return engine->cl.add(c);
    }
}

EXPORT_API int quacq_add_target_constraint(void* handle, int type, int scope_size, const int* scope, int param) {
    if (!handle || !scope || scope_size <= 0) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc(scope, scope + scope_size);
    Constraint c(engine->target_ct.size(), type, sc, param);
    engine->target_ct.add(c);
    engine->has_target_ct = true;
    return engine->target_ct.size() - 1;
}

EXPORT_API int quacq_add_table_constraint(void* handle, int is_bias, int scope_size, const int* scope,
                                          int num_tuples, const int* flat_tuples, int is_allowed) {
    if (!handle || !scope || scope_size <= 0 || !flat_tuples || num_tuples <= 0) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc(scope, scope + scope_size);
    Constraint c(is_bias ? engine->bias.size() : engine->cl.size(),
                 is_allowed ? OP_TABLE_ALLOWED : OP_TABLE_FORBIDDEN, sc);
    c.table.resize(num_tuples);
    for (int i = 0; i < num_tuples; ++i) {
        c.table[i].assign(flat_tuples + i * scope_size, flat_tuples + (i + 1) * scope_size);
    }
    if (is_bias) {
        return engine->bias.add(c);
    } else {
        return engine->cl.add(c);
    }
}

EXPORT_API int quacq_solve(void* handle, int mode, int scope_size, const int* scope,
                           double timeout_sec, int* out_assignment) {
    if (!handle || !out_assignment) return STATUS_ERROR;
    auto* engine = static_cast<QuAcqEngine*>(handle);

    std::vector<int> sc;
    if (scope && scope_size > 0) {
        sc.assign(scope, scope + scope_size);
    }

    engine->solver.set_cl(engine->cl);
    engine->solver.set_bias(engine->bias);

    ResultStatus status = engine->solver.solve(static_cast<SolveMode>(mode), sc, timeout_sec);
    const auto& asgn = engine->solver.get_assignment();
    std::memcpy(out_assignment, asgn.data(), asgn.size() * sizeof(int));
    return static_cast<int>(status);
}

EXPORT_API int quacq_count_bias_rejects(void* handle, const int* assignment) {
    if (!handle || !assignment) return 0;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> asgn(assignment, assignment + engine->num_vars);
    return engine->bias.count_rejects(asgn);
}

EXPORT_API int quacq_remove_bias_rejects(void* handle, const int* assignment) {
    if (!handle || !assignment) return 0;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> asgn(assignment, assignment + engine->num_vars);
    int prev_size = engine->bias.size();
    engine->bias.remove_rejects(asgn);
    return prev_size - engine->bias.size();
}

EXPORT_API int quacq_run(void* handle, OracleCallback oracle_cb, void* user_data, int max_queries) {
    if (!handle) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    if (oracle_cb) {
        engine->set_oracle(oracle_cb, user_data);
    }
    return engine->run_quacq(max_queries);
}

EXPORT_API int quacq_run_algorithm(void* handle, int algo_type, OracleCallback oracle_cb, void* user_data, int max_queries) {
    if (!handle) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    if (oracle_cb) {
        engine->set_oracle(oracle_cb, user_data);
    }
    return engine->run_algorithm(static_cast<AlgorithmType>(algo_type), max_queries);
}

EXPORT_API void quacq_set_findscope_version(void* handle, int version) {
    if (!handle) return;
    static_cast<QuAcqEngine*>(handle)->findscope_version = static_cast<FindScopeVersion>(version);
}

EXPORT_API void quacq_set_findc_version(void* handle, int version) {
    if (!handle) return;
    static_cast<QuAcqEngine*>(handle)->findc_version = static_cast<FindCVersion>(version);
}

EXPORT_API int quacq_generate_query(void* handle, int scope_size, const int* scope, double timeout_sec, int* out_assignment) {
    if (!handle || !out_assignment) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc;
    if (scope && scope_size > 0) sc.assign(scope, scope + scope_size);
    engine->query_timeout_sec = timeout_sec;
    std::vector<int> q = engine->generate_query(sc);
    if (q.empty()) return 0;
    std::memcpy(out_assignment, q.data(), q.size() * sizeof(int));
    return 1;
}

EXPORT_API int quacq_generate_tqgen_query(void* handle, int scope_size, const int* scope, double tau, double alpha, int* out_assignment) {
    if (!handle || !out_assignment) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc;
    if (scope && scope_size > 0) sc.assign(scope, scope + scope_size);
    std::vector<int> q = engine->generate_tqgen_query(sc, tau, alpha);
    if (q.empty()) return 0;
    std::memcpy(out_assignment, q.data(), q.size() * sizeof(int));
    return 1;
}

EXPORT_API int quacq_find_scope(void* handle, const int* query, int Y_size, const int* Y, int* out_scope_size, int* out_scope) {
    if (!handle || !query || !Y || Y_size <= 0 || !out_scope_size || !out_scope) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> q(query, query + engine->num_vars);
    std::vector<int> y_vec(Y, Y + Y_size);
    std::vector<int> scope = engine->find_scope(q, y_vec);
    *out_scope_size = static_cast<int>(scope.size());
    for (size_t i = 0; i < scope.size(); ++i) {
        out_scope[i] = scope[i];
    }
    return *out_scope_size;
}

EXPORT_API int quacq_find_c(void* handle, int scope_size, const int* scope, const int* query, int* out_type, int* out_param) {
    if (!handle || !scope || scope_size <= 0 || !query || !out_type || !out_param) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<int> sc(scope, scope + scope_size);
    std::vector<int> q(query, query + engine->num_vars);
    Constraint c = engine->find_c(sc, q);
    *out_type = c.type;
    *out_param = c.param;
    return c.id >= 0 ? 0 : -1;
}

EXPORT_API int quacq_run_conacq1(void* handle, int num_pos, const int* flat_pos, int num_neg, const int* flat_neg) {
    if (!handle) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    std::vector<std::vector<int>> positives(num_pos, std::vector<int>(engine->num_vars));
    for (int i = 0; i < num_pos; ++i) {
        positives[i].assign(flat_pos + i * engine->num_vars, flat_pos + (i + 1) * engine->num_vars);
    }
    std::vector<std::vector<int>> negatives(num_neg, std::vector<int>(engine->num_vars));
    for (int i = 0; i < num_neg; ++i) {
        negatives[i].assign(flat_neg + i * engine->num_vars, flat_neg + (i + 1) * engine->num_vars);
    }
    return engine->run_conacq1(positives, negatives);
}

EXPORT_API void quacq_get_metrics(void* handle, AcquisitionMetrics* out_metrics) {
    if (!handle || !out_metrics) return;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    *out_metrics = engine->metrics;
}

EXPORT_API int quacq_get_cl_count(void* handle) {
    if (!handle) return 0;
    return static_cast<QuAcqEngine*>(handle)->cl.size();
}

EXPORT_API int quacq_get_bias_count(void* handle) {
    if (!handle) return 0;
    return static_cast<QuAcqEngine*>(handle)->bias.size();
}

EXPORT_API void quacq_promote_bias_to_cl(void* handle) {
    if (!handle) return;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    for (const auto& c : engine->bias.constraints) {
        engine->cl.add(c);
        engine->metrics.learned_constraints++;
    }
    engine->bias.clear();
}

EXPORT_API int quacq_get_cl_constraint(void* handle, int idx, int* out_type, int* out_scope_size,
                                       int* out_scope, int* out_param) {
    if (!handle) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    if (idx < 0 || idx >= engine->cl.size()) return -1;
    const auto& c = engine->cl.get(idx);
    if (out_type) *out_type = c.type;
    if (out_scope_size) *out_scope_size = static_cast<int>(c.scope.size());
    if (out_scope) {
        for (size_t i = 0; i < c.scope.size(); ++i) {
            out_scope[i] = c.scope[i];
        }
    }
    if (out_param) *out_param = c.param;
    return 0;
}

EXPORT_API int quacq_get_bias_constraint(void* handle, int idx, int* out_type, int* out_scope_size,
                                         int* out_scope, int* out_param) {
    if (!handle) return -1;
    auto* engine = static_cast<QuAcqEngine*>(handle);
    if (idx < 0 || idx >= engine->bias.size()) return -1;
    const auto& c = engine->bias.get(idx);
    if (out_type) *out_type = c.type;
    if (out_scope_size) *out_scope_size = static_cast<int>(c.scope.size());
    if (out_scope) {
        for (size_t i = 0; i < c.scope.size(); ++i) {
            out_scope[i] = c.scope[i];
        }
    }
    if (out_param) *out_param = c.param;
    return 0;
}

EXPORT_API int quacq_check_constraint(int type, int scope_size, const int* scope, int param,
                                      const int* assignment, int n_vars) {
    if (!scope || !assignment || scope_size <= 0) return EVAL_UNDEF;
    std::vector<int> sc(scope, scope + scope_size);
    Constraint c(0, type, sc, param);
    std::vector<int> asgn(assignment, assignment + n_vars);
    return static_cast<int>(c.check(asgn));
}

} // extern "C"
