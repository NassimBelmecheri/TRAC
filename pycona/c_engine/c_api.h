#ifndef C_ENGINE_C_API_H
#define C_ENGINE_C_API_H

#include "common.h"

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

extern "C" {

EXPORT_API void* quacq_create(int num_vars);
EXPORT_API void quacq_free(void* handle);

EXPORT_API void quacq_set_domain(void* handle, int var, int count, const int* values);

EXPORT_API int quacq_add_constraint(void* handle, int is_bias, int type, int scope_size, const int* scope, int param);

EXPORT_API int quacq_add_target_constraint(void* handle, int type, int scope_size, const int* scope, int param);

EXPORT_API int quacq_add_table_constraint(void* handle, int is_bias, int scope_size, const int* scope,
                                          int num_tuples, const int* flat_tuples, int is_allowed);

EXPORT_API int quacq_solve(void* handle, int mode, int scope_size, const int* scope,
                           double timeout_sec, int* out_assignment);

EXPORT_API int quacq_count_bias_rejects(void* handle, const int* assignment);
EXPORT_API int quacq_remove_bias_rejects(void* handle, const int* assignment);

EXPORT_API int quacq_run(void* handle, quacq::OracleCallback oracle_cb, void* user_data, int max_queries);
EXPORT_API int quacq_run_algorithm(void* handle, int algo_type, quacq::OracleCallback oracle_cb, void* user_data, int max_queries);

EXPORT_API void quacq_set_findscope_version(void* handle, int version);
EXPORT_API void quacq_set_findc_version(void* handle, int version);

EXPORT_API int quacq_generate_query(void* handle, int scope_size, const int* scope, double timeout_sec, int* out_assignment);
EXPORT_API int quacq_generate_tqgen_query(void* handle, int scope_size, const int* scope, double tau, double alpha, int* out_assignment);

EXPORT_API int quacq_find_scope(void* handle, const int* query, int Y_size, const int* Y, int* out_scope_size, int* out_scope);
EXPORT_API int quacq_find_c(void* handle, int scope_size, const int* scope, const int* query, int* out_type, int* out_param);

EXPORT_API int quacq_run_conacq1(void* handle, int num_pos, const int* flat_pos, int num_neg, const int* flat_neg);

EXPORT_API void quacq_get_metrics(void* handle, quacq::AcquisitionMetrics* out_metrics);

EXPORT_API int quacq_get_cl_count(void* handle);
EXPORT_API int quacq_get_bias_count(void* handle);

EXPORT_API void quacq_promote_bias_to_cl(void* handle);

EXPORT_API int quacq_get_cl_constraint(void* handle, int idx, int* out_type, int* out_scope_size,
                                       int* out_scope, int* out_param);
EXPORT_API int quacq_get_bias_constraint(void* handle, int idx, int* out_type, int* out_scope_size,
                                         int* out_scope, int* out_param);

EXPORT_API int quacq_check_constraint(int type, int scope_size, const int* scope, int param,
                                      const int* assignment, int n_vars);

} // extern "C"

#endif // C_ENGINE_C_API_H
