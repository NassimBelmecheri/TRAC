#ifndef C_ENGINE_CONSTRAINT_NET_H
#define C_ENGINE_CONSTRAINT_NET_H

#include "common.h"
#include "constraint.h"
#include <vector>
#include <algorithm>
#include <unordered_set>

namespace quacq {

class ConstraintNet {
public:
    std::vector<Constraint> constraints;
    std::vector<std::vector<int>> var_to_cons;
    int num_vars;

    explicit ConstraintNet(int n_vars = 0) : num_vars(n_vars) {
        if (n_vars > 0) {
            var_to_cons.resize(n_vars);
        }
    }

    void set_num_vars(int n_vars) {
        num_vars = n_vars;
        var_to_cons.assign(n_vars, std::vector<int>());
        rebuild_index();
    }

    void rebuild_index() {
        if (num_vars <= 0) return;
        var_to_cons.assign(num_vars, std::vector<int>());
        for (size_t i = 0; i < constraints.size(); ++i) {
            for (int v : constraints[i].scope) {
                if (v >= 0 && v < num_vars) {
                    var_to_cons[v].push_back(static_cast<int>(i));
                }
            }
        }
    }

    int add(const Constraint& c) {
        int idx = static_cast<int>(constraints.size());
        constraints.push_back(c);
        for (int v : c.scope) {
            if (v >= num_vars) {
                num_vars = v + 1;
                var_to_cons.resize(num_vars);
            }
            var_to_cons[v].push_back(idx);
        }
        return idx;
    }

    inline int size() const {
        return static_cast<int>(constraints.size());
    }

    inline bool empty() const {
        return constraints.empty();
    }

    inline const Constraint& get(int i) const {
        return constraints[i];
    }

    inline Constraint& get(int i) {
        return constraints[i];
    }

    void clear() {
        constraints.clear();
        for (auto& vec : var_to_cons) {
            vec.clear();
        }
    }

    // Fast reject counting
    int count_rejects(const std::vector<int>& assignment) const {
        int count = 0;
        for (const auto& c : constraints) {
            if (c.is_violated(assignment)) {
                ++count;
            }
        }
        return count;
    }

    // Get indices of constraints violated by assignment
    std::vector<int> get_rejects(const std::vector<int>& assignment) const {
        std::vector<int> rejects;
        rejects.reserve(constraints.size() / 4);
        for (size_t i = 0; i < constraints.size(); ++i) {
            if (constraints[i].is_violated(assignment)) {
                rejects.push_back(static_cast<int>(i));
            }
        }
        return rejects;
    }

    // Remove all constraints violated by assignment (B <- B \ kappa_B(e))
    void remove_rejects(const std::vector<int>& assignment) {
        std::vector<Constraint> kept;
        kept.reserve(constraints.size());
        for (const auto& c : constraints) {
            if (!c.is_violated(assignment)) {
                kept.push_back(c);
            }
        }
        constraints = std::move(kept);
        rebuild_index();
    }

    // Remove all constraints accepted by assignment
    void remove_accepts(const std::vector<int>& assignment) {
        std::vector<Constraint> kept;
        kept.reserve(constraints.size());
        for (const auto& c : constraints) {
            if (!c.is_satisfied(assignment)) {
                kept.push_back(c);
            }
        }
        constraints = std::move(kept);
        rebuild_index();
    }

    // Check if assignment satisfies all constraints in network
    bool is_valid_solution(const std::vector<int>& assignment) const {
        for (const auto& c : constraints) {
            if (c.is_violated(assignment)) return false;
        }
        return true;
    }

    // Extract subnetwork with scopes entirely inside given variable subset
    ConstraintNet get_subnetwork(const std::vector<int>& scope_subset) const {
        std::vector<bool> in_subset(num_vars, false);
        for (int v : scope_subset) {
            if (v >= 0 && v < num_vars) in_subset[v] = true;
        }

        ConstraintNet sub(num_vars);
        for (const auto& c : constraints) {
            bool all_in = true;
            for (int v : c.scope) {
                if (v < 0 || v >= num_vars || !in_subset[v]) {
                    all_in = false;
                    break;
                }
            }
            if (all_in) {
                sub.add(c);
            }
        }
        return sub;
    }
};

} // namespace quacq

#endif // C_ENGINE_CONSTRAINT_NET_H
