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
#include "funque_strred_options.h"

typedef struct strredbuffers {
    float* bands[4];
    int width;
    int height;
    ptrdiff_t stride;
} strredbuffers;

typedef struct strred_results {
    double srred_vals[MAX_LEVELS];
    double trred_vals[MAX_LEVELS];
    double strred_vals[MAX_LEVELS];
    double spat_vals[MAX_LEVELS];
    double temp_vals[MAX_LEVELS];
    double spat_temp_vals[MAX_LEVELS];
    double spat_vals_cumsum, temp_vals_cumsum, spat_temp_vals_cumsum;

} strred_results;

int compute_srred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist,
                         size_t width, size_t height, float** spat_scales_ref, float** spat_scales_dist,
                         struct strred_results* strred_scores, int block_size, int level);

int compute_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist,
                          struct strredbuffers* prev_ref, struct strredbuffers* prev_dist,
                          size_t width, size_t height, float** spat_scales_ref, float** spat_scales_dist, 
                          struct strred_results* strred_scores, int block_size, int level);