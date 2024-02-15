#include "../integer_funque_filters.h"

int integer_compute_ssim_funque_neon(i_dwt2buffers *ref, i_dwt2buffers *dist, SsimScore_int *score,
                                     int max_val, float K1, float K2, int pending_div,
                                     int32_t *div_lookup);
int integer_compute_ms_ssim_funque_neon(i_dwt2buffers *ref, i_dwt2buffers *dist,
                                        MsSsimScore_int *score, int max_val, float K1, float K2,
                                        int pending_div, int32_t *div_lookup, int n_levels,
                                        int is_pyr);
int integer_mean_2x2_ms_ssim_funque_neon(int32_t *var_x_cum, int32_t *var_y_cum,
                                         int32_t *cov_xy_cum, int width, int height, int level);
