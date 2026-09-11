#ifndef C_ENGINE_DOMAIN_H
#define C_ENGINE_DOMAIN_H

#include "common.h"
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

namespace quacq {

class Domain {
public:
    std::vector<int> values;
    std::vector<int> states; // -1 = active, >= 0 = level at which it was pruned
    int active_size;
    std::unordered_map<int, int> val_to_idx;

    Domain() : active_size(0) {}

    explicit Domain(const std::vector<int>& raw_vals) {
        init(raw_vals);
    }

    void init(const std::vector<int>& raw_vals) {
        values = raw_vals;
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        active_size = static_cast<int>(values.size());
        states.assign(active_size, -1);
        val_to_idx.clear();
        for (int i = 0; i < active_size; ++i) {
            val_to_idx[values[i]] = i;
        }
    }

    inline int size() const {
        return active_size;
    }

    inline int total_size() const {
        return static_cast<int>(values.size());
    }

    inline bool empty() const {
        return active_size <= 0;
    }

    inline bool is_valid_idx(int idx) const {
        return states[idx] == -1;
    }

    inline bool contains_val(int val) const {
        auto it = val_to_idx.find(val);
        if (it == val_to_idx.end()) return false;
        return states[it->second] == -1;
    }

    inline int get_idx_of(int val) const {
        auto it = val_to_idx.find(val);
        if (it != val_to_idx.end()) return it->second;
        return -1;
    }

    inline int get_val(int idx) const {
        return values[idx];
    }

    inline bool prune_idx(int idx, int level) {
        if (states[idx] == -1) {
            states[idx] = level;
            --active_size;
            return true;
        }
        return false;
    }

    inline bool prune_val(int val, int level) {
        int idx = get_idx_of(val);
        if (idx >= 0) {
            return prune_idx(idx, level);
        }
        return false;
    }

    void restore(int level) {
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i] >= level) {
                states[i] = -1;
                ++active_size;
            }
        }
    }

    int first_active_idx() const {
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i] == -1) return static_cast<int>(i);
        }
        return -1;
    }
};

} // namespace quacq

#endif // C_ENGINE_DOMAIN_H
