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
#define STRRED_STABILITY 0
#define PENDING_SHIFT_FACTOR 12

#define PENDING_SHIFT_FACTOR_NEW 12
#define Q_FORMAT_MULTIPLIED_IN_LOG_TABLE_NEW 26
#define TWO_POW_Q_FACT_NEW (1 << Q_FORMAT_MULTIPLIED_IN_LOG_TABLE_NEW)


#define KEEP_SPAT_IN_INTEGER 0
#define DEBUG_STRRED 1
#define USE_FLOAT_CODE 1
#define LOGE_BASE2 1.442684682

typedef struct strred_results {
    double srred_vals[MAX_LEVELS];
    double trred_vals[MAX_LEVELS];
    double strred_vals[MAX_LEVELS];
    double spat_vals[MAX_LEVELS];
    double temp_vals[MAX_LEVELS];
    double spat_temp_vals[MAX_LEVELS];
} strred_results;

int integer_compute_strred_funque_c(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                          struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                          size_t width, size_t height, struct strred_results* strred_scores,
                          int block_size, int level, uint32_t *log_18, uint32_t *log_22, int32_t shift_val,
                          uint32_t sigma_nsq_t, uint8_t enable_spatial_csf);

int integer_copy_prev_frame_strred_funque_c(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                                  struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                                  size_t width, size_t height);