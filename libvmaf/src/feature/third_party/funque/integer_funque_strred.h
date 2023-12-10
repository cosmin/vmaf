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

#include "funque_global_options.h"

#define STRRED_REFLECT_PAD 1

#define VARIANCE_SHIFT_FACTOR 6
#define STRRED_Q_FORMAT 26
#define TWO_POWER_Q_FACTOR (1 << STRRED_Q_FORMAT)

#define LOGE_BASE2 1.442684682

int integer_compute_strred_funque_c(const struct i_dwt2buffers* ref,
                                    const struct i_dwt2buffers* dist,
                                    struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                                    size_t width, size_t height,
                                    struct strred_results* strred_scores, int block_size, int level,
                                    uint32_t* log_18, uint32_t* log_22, int32_t shift_val,
                                    double sigma_nsq_t, uint8_t enable_spatial_csf);

int integer_copy_prev_frame_strred_funque_c(const struct i_dwt2buffers* ref,
                                            const struct i_dwt2buffers* dist,
                                            struct i_dwt2buffers* prev_ref,
                                            struct i_dwt2buffers* prev_dist, size_t width,
                                            size_t height);

void strred_funque_log_generate(uint32_t *log_18);
void strred_funque_generate_log22(uint32_t *log_22);
uint32_t strred_get_best_u18_from_u64(uint64_t temp, int *x);
uint32_t strred_get_best_u22_from_u64(uint64_t temp, int *x);