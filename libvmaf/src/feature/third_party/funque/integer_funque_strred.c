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

#include <math.h>
#include <mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "integer_funque_filters.h"
#include "funque_strred_options.h"
#include "common/macros.h"
#include "integer_funque_strred.h"

// just change the store offset to reduce multiple calculation when getting log2f value
void strred_funque_log_generate(uint32_t *log_18)
{
    uint64_t i;
    uint64_t start = (unsigned int) pow(2, 17);
    uint64_t end = (unsigned int) pow(2, 18);
    for(i = start; i < end; i++) {
        log_18[i] = (uint32_t) round(log2((double) i) * (1 << STRRED_Q_FORMAT));
    }
}

void strred_funque_generate_log22(uint32_t *log_22)
{
    uint64_t i;
    uint64_t start = (unsigned int) pow(2, 21);
    uint64_t end = (unsigned int) pow(2, 22);
    for(i = start; i < end; i++) {
        log_22[i] = (uint32_t) round(log2((double) i) * (1 << STRRED_Q_FORMAT));
    }
}

void strred_integer_reflect_pad(const dwt2_dtype *src, size_t width, size_t height, int reflect,
                                dwt2_dtype *dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for(size_t i = reflect; i != (out_height - reflect); i++) {
        for(int j = 0; j != reflect; j++) {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width],
               sizeof(dwt2_dtype) * width);

        for(int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] =
                dest[i * out_width + out_width - reflect - 2 - j];
    }

    for(int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width],
               &dest[reflect * out_width + (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width],
               &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width],
               sizeof(dwt2_dtype) * out_width);
    }
}

int integer_copy_prev_frame_strred_funque_c(const struct i_dwt2buffers *ref,
                                            const struct i_dwt2buffers *dist,
                                            struct i_dwt2buffers *prev_ref,
                                            struct i_dwt2buffers *prev_dist, size_t width,
                                            size_t height)
{
    int subband;
    int total_subbands = DEFAULT_STRRED_SUBBANDS;

    for(subband = 1; subband < total_subbands; subband++) {
        memcpy(prev_ref->bands[subband], ref->bands[subband], width * height * sizeof(dwt2_dtype));
        memcpy(prev_dist->bands[subband], dist->bands[subband],
               width * height * sizeof(dwt2_dtype));
    }

    prev_ref->width = ref->width;
    prev_ref->height = ref->height;
    prev_ref->stride = ref->stride;

    prev_dist->width = dist->width;
    prev_dist->height = dist->height;
    prev_dist->stride = dist->stride;

    return 0;
}

void integer_subract_subbands_c(const dwt2_dtype *ref_src, const dwt2_dtype *ref_prev_src,
                                dwt2_dtype *ref_dst, const dwt2_dtype *dist_src,
                                const dwt2_dtype *dist_prev_src, dwt2_dtype *dist_dst, size_t width,
                                size_t height)
{
    size_t i, j;

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

int integer_compute_srred_funque_c(const struct i_dwt2buffers *ref,
                                   const struct i_dwt2buffers *dist, size_t width, size_t height,
                                   float **spat_scales_ref, float **spat_scales_dist,
                                   struct strred_results *strred_scores, int block_size, int level,
                                   uint32_t *log_18, uint32_t *log_22, int32_t shift_val_arg,
                                   double sigma_nsq_t, uint8_t check_enable_spatial_csf)
{
    int ret;
    UNUSED(block_size);
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband;
    float spat_values[DEFAULT_STRRED_SUBBANDS], fspat_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    for(subband = 1; subband < total_subbands; subband++) {
        enable_temp = 0;
        spat_values[subband] = 0;

        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else {
            shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
        }
        spat_values[subband] = integer_rred_entropies_and_scales(
            ref->bands[subband], dist->bands[subband], width, height, log_18, log_22, sigma_nsq_t,
            shift_val, enable_temp, spat_scales_ref[subband], spat_scales_dist[subband],
            check_enable_spatial_csf);
        fspat_val[subband] = spat_values[subband] / (width * height);
    }

    strred_scores->spat_vals[level] = (fspat_val[1] + fspat_val[2] + fspat_val[3]) / 3;

    // Add equations to compute S-RRED using norm factors
    int norm_factor, num_level;
    for(num_level = 0; num_level <= level; num_level++) norm_factor = num_level + 1;

    strred_scores->spat_vals_cumsum += strred_scores->spat_vals[level];

    strred_scores->srred_vals[level] = strred_scores->spat_vals_cumsum / norm_factor;

    ret = 0;
    return ret;
}

int integer_compute_strred_funque_c(const struct i_dwt2buffers *ref,
                                    const struct i_dwt2buffers *dist,
                                    struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist,
                                    size_t width, size_t height, float **spat_scales_ref,
                                    float **spat_scales_dist, struct strred_results *strred_scores,
                                    int block_size, int level, uint32_t *log_18, uint32_t *log_22,
                                    int32_t shift_val_arg, double sigma_nsq_t,
                                    uint8_t check_enable_spatial_csf)
{
    int ret;
    UNUSED(block_size);
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband;
    float temp_values[DEFAULT_STRRED_SUBBANDS], ftemp_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    for(subband = 1; subband < total_subbands; subband++) {
        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else {
            shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
        }

        if(prev_ref != NULL && prev_dist != NULL) {
            enable_temp = 1;
            dwt2_dtype *ref_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            dwt2_dtype *dist_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            temp_values[subband] = 0;

            integer_subract_subbands_c(ref->bands[subband], prev_ref->bands[subband], ref_temporal,
                                       dist->bands[subband], prev_dist->bands[subband],
                                       dist_temporal, width, height);
            temp_values[subband] = integer_rred_entropies_and_scales(
                ref_temporal, dist_temporal, width, height, log_18, log_22, sigma_nsq_t, shift_val,
                enable_temp, spat_scales_ref[subband], spat_scales_dist[subband],
                check_enable_spatial_csf);
            ftemp_val[subband] = temp_values[subband] / (width * height);

            free(ref_temporal);
            free(dist_temporal);
        }
    }
    strred_scores->temp_vals[level] = (ftemp_val[1] + ftemp_val[2] + ftemp_val[3]) / 3;
    strred_scores->spat_temp_vals[level] =
        strred_scores->spat_vals[level] * strred_scores->temp_vals[level];

    // Add equations to compute ST-RRED using norm factors
    int norm_factor, num_level;
    for(num_level = 0; num_level <= level; num_level++) norm_factor = num_level + 1;

    strred_scores->temp_vals_cumsum += strred_scores->temp_vals[level];
    strred_scores->spat_temp_vals_cumsum += strred_scores->spat_temp_vals[level];

    strred_scores->trred_vals[level] = strred_scores->temp_vals_cumsum / norm_factor;
    strred_scores->strred_vals[level] = strred_scores->spat_temp_vals_cumsum / norm_factor;

    ret = 0;
    return ret;
}