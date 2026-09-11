#ifndef C_ENGINE_QUACQ_H
#define C_ENGINE_QUACQ_H

#include "common.h"
#include "solver.h"
#include "constraint_net.h"
#include <vector>
#include <set>
#include <unordered_set>

namespace quacq {

enum AlgorithmType {
    ALGO_QUACQ = 0,
    ALGO_PQUACQ = 1,
    ALGO_MQUACQ = 2,
    ALGO_MQUACQ2 = 3,
    ALGO_GROWACQ = 4,
    ALGO_BRUTECA = 5,
    ALGO_CONACQ1 = 6,
    ALGO_CONACQ2 = 7
};

enum FindScopeVersion {
    FINDSCOPE_V1 = 1, // IJCAI 2013
    FINDSCOPE_V2 = 2  // AIJ 2023 (kappa branch pruning)
};

enum FindCVersion {
    FINDC_V1 = 1, // Classic single constraint
    FINDC_V2 = 2  // Multi-constraint / conjunction
};

class QuAcqEngine {
public:
    int num_vars;
    FastSolver solver;
    ConstraintNet cl;
    ConstraintNet bias;
    ConstraintNet target_ct; // Optional: ground truth target network if running pure C++ benchmarks
    bool has_target_ct;

    OracleCallback oracle_cb;
    void* oracle_user_data;

    AcquisitionMetrics metrics;
    double query_timeout_sec;

    FindScopeVersion findscope_version;
    FindCVersion findc_version;

    explicit QuAcqEngine(int n_vars = 0);

    void set_num_vars(int n_vars);
    void set_domain(int var, const std::vector<int>& raw_vals);
    void add_cl_constraint(const Constraint& c);
    void add_bias_constraint(const Constraint& c);
    void set_target_network(const ConstraintNet& ct);

    void set_oracle(OracleCallback cb, void* user_data);

    // Master runner
    int run_algorithm(AlgorithmType algo, int max_queries = 50000);

    // Algorithm implementations
    int run_quacq(int max_queries = 50000);
    int run_pquacq(int max_queries = 50000, int alpha_cutoff = 4);
    int run_mquacq(int max_queries = 50000);
    int run_mquacq2(int max_queries = 50000);
    int run_growacq(int max_queries = 50000, AlgorithmType inner_algo = ALGO_QUACQ);
    int run_bruteca(int max_queries = 50000);
    int run_conacq1(const std::vector<std::vector<int>>& positives,
                    const std::vector<std::vector<int>>& negatives);

    // Query Generators
    std::vector<int> generate_query(const std::vector<int>& scope = {});
    std::vector<int> generate_tqgen_query(const std::vector<int>& scope, double tau, double alpha);

    // FindScope implementations
    std::vector<int> find_scope(const std::vector<int>& query, const std::vector<int>& Y);
    std::vector<int> find_scope_v1(const std::vector<int>& query, const std::vector<int>& Y);
    std::vector<int> find_scope_v2(const std::vector<int>& query, const std::vector<int>& Y);

    // FindC implementations
    Constraint find_c(const std::vector<int>& scope, const std::vector<int>& query);
    Constraint find_c_v1(const std::vector<int>& scope, const std::vector<int>& query);
    std::vector<Constraint> find_c_v2(const std::vector<int>& scope, const std::vector<int>& query);

    // Remove every bias candidate on the given scope set (used by the
    // query-generation algorithms after find_c_v2 has tested them all).
    void remove_scope_from_bias(const std::vector<int>& scope);

    // PQuAcq Recommendation / Link prediction
    int predict_and_ask(int relation_type, int alpha_cutoff = 4);

    bool ask_oracle(const std::vector<int>& assignment, const std::vector<int>& scope);

private:
    std::vector<int> find_scope_v1_rec(const std::vector<int>& query,
                                       std::vector<int> R,
                                       std::vector<int> Y,
                                       bool do_ask);

    std::vector<int> find_scope_v2_rec(const std::vector<int>& query,
                                       std::vector<int> R,
                                       std::vector<int> Y,
                                       std::vector<int>& current_kappa_b);

    void mquacq_find_all_cons(std::vector<int> Y,
                              std::set<std::vector<int>>& scopes,
                              int max_queries);

    double compute_adamic_adar(const std::vector<std::vector<bool>>& adj,
                               const std::vector<int>& degrees,
                               int u, int v) const;
};

} // namespace quacq

#endif // C_ENGINE_QUACQ_H
