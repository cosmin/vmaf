/*   SPDX-License-Identifier: BSD-3-Clause
*   Copyright (C) 2022 Intel Corporation.
*/
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
int integer_compute_ssim_funque_avx512(i_dwt2buffers *ref, i_dwt2buffers *dist, SsimScore_int *score,
                                       int max_val, float K1, float K2, int pending_div,
                                       int32_t *div_lookup);
int integer_compute_ms_ssim_funque_avx512(i_dwt2buffers *ref, i_dwt2buffers *dist,
                                          MsSsimScore_int *score, int max_val, float K1, float K2,
                                          int pending_div, int32_t *div_lookup, int n_levels,
                                          int is_pyr);
int integer_mean_2x2_ms_ssim_funque_avx512(int32_t *var_x_cum, int32_t *var_y_cum,
                                           int32_t *cov_xy_cum, int width, int height, int level);
int integer_ms_ssim_shift_cum_buffer_funque_avx512(int32_t *var_x_cum, int32_t *var_y_cum, int32_t *cov_xy_cum,
                                                   int width, int height, int level, uint8_t csf_pending_div[4],
                                                   uint8_t csf_pending_div_lp1[4]);