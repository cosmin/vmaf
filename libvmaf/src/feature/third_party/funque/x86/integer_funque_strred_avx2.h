/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "../integer_funque_filters.h"
#include "../funque_strred_options.h"
#include "../common/macros.h"
#include "../funque_global_options.h"

int integer_compute_srred_funque_avx2(const struct i_dwt2buffers *ref,
                                      const struct i_dwt2buffers *dist, size_t width, size_t height,
                                      float **spat_scales_ref, float **spat_scales_dist,
                                      struct strred_results *strred_scores, int block_size, int level,
                                      uint32_t *log_18, uint32_t *log_22, int32_t shift_val_arg,
                                      double sigma_nsq_t, uint8_t check_enable_spatial_csf);

int integer_compute_strred_funque_avx2(const struct i_dwt2buffers *ref,
                                       const struct i_dwt2buffers *dist,
                                       struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist,
                                       size_t width, size_t height, float **spat_scales_ref,
                                       float **spat_scales_dist, struct strred_results *strred_scores,
                                       int block_size, int level, uint32_t *log_18, uint32_t *log_22,
                                       int32_t shift_val_arg, double sigma_nsq_t,
                                       uint8_t check_enable_spatial_csf);

void integer_subract_subbands_avx2(const dwt2_dtype *ref_src, const dwt2_dtype *ref_prev_src,
                                   dwt2_dtype *ref_dst, const dwt2_dtype *dist_src,
                                   const dwt2_dtype *dist_prev_src, dwt2_dtype *dist_dst, int width,
                                   int height);

float integer_rred_entropies_and_scales_avx2(const dwt2_dtype *x_t, const dwt2_dtype *y_t,
                                             size_t width, size_t height, uint32_t *log_18,
                                             uint32_t *log_22, double sigma_nsq_arg,
                                             int32_t shift_val, uint8_t enable_temporal,
                                             float *spat_scales_x, float *spat_scales_y,
                                             uint8_t check_enable_spatial_csf);