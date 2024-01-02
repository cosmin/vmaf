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
#include <arm_neon.h>

#include "integer_funque_strred_neon.h"
#include "../integer_funque_strred.h"

void integer_subract_subbands_neon(const dwt2_dtype *ref_src, const dwt2_dtype *ref_prev_src,
                                    dwt2_dtype *ref_dst, const dwt2_dtype *dist_src,
                                    const dwt2_dtype *dist_prev_src, dwt2_dtype *dist_dst,
                                    int width, int height)
{
    int i, j;
    for(i = 0; i < height; i++)
    {
        for(j = 0; j <= width - 8; j += 8)
        {
            int16x8_t ref_src_vector = vld1q_s16(&ref_src[i * width + j]);
            int16x8_t ref_prev_src_vector = vld1q_s16(&ref_prev_src[i * width + j]);
            int16x8_t dist_src_vector = vld1q_s16(&dist_src[i * width + j]);
            int16x8_t dist_prev_src_vector = vld1q_s16(&dist_prev_src[i * width + j]);

            int16x8_t diff_ref = vsubq_s16(ref_src_vector, ref_prev_src_vector);
            int16x8_t diff_dist = vsubq_s16(dist_src_vector, dist_prev_src_vector);

            vst1q_s16(&ref_dst[i * width + j], diff_ref);
            vst1q_s16(&dist_dst[i * width + j], diff_dist);
        }
        for(; j < width; j++)
        {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

float integer_rred_entropies_and_scales_neon(const dwt2_dtype *x_t, const dwt2_dtype *y_t, size_t width,
                                        size_t height, uint32_t *log_18, uint32_t *log_22,
                                        double sigma_nsq_arg, int32_t shift_val,
                                        uint8_t enable_temporal, float *spat_scales_x,
                                        float *spat_scales_y, uint8_t check_enable_spatial_csf)
{
    int kh = STRRED_WINDOW_SIZE;
    int kw = STRRED_WINDOW_SIZE;

    /* amount of reflecting */
    int x_reflect = (int) ((STRRED_WINDOW_SIZE - 1) / 2);
    int y_reflect = (int) ((STRRED_WINDOW_SIZE - 1) / 2);
    size_t strred_width, strred_height;

#if STRRED_REFLECT_PAD
    strred_width = width;
    strred_height = height;
#else
    strred_width = width - (2 * x_reflect);
    strred_height = height - (2 * x_reflect);
#endif

    size_t r_width = strred_width + (2 * x_reflect);
    size_t r_height = strred_height + (2 * x_reflect);
    dwt2_dtype *x_pad_t;
    dwt2_dtype *y_pad_t;

#if STRRED_REFLECT_PAD
    x_pad_t = (dwt2_dtype *) malloc(sizeof(dwt2_dtype *) * (strred_width + (2 * x_reflect)) *
                                    (strred_height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype *) malloc(sizeof(dwt2_dtype *) * (strred_width + (2 * y_reflect)) *
                                    (strred_height + (2 * y_reflect)));

    strred_integer_reflect_pad(x_t, strred_width, strred_height, x_reflect, x_pad_t);
    strred_integer_reflect_pad(y_t, strred_width, strred_height, y_reflect, y_pad_t);

#else
    x_pad_t = x_t;
    y_pad_t = y_t;
#endif

    float agg_abs_accum = 0;
    int16_t knorm_fact =
        25891;  // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21;
    uint32_t entr_const = (uint32_t) (log2f(2 * PI_CONSTANT * EULERS_CONSTANT) * TWO_POWER_Q_FACTOR);

    {
        int width_p1 = r_width + 1;
        int height_p1 = r_height + 1;

        int32_t *interim_1_x = (int32_t *) calloc(width_p1, sizeof(int32_t));
        int32_t *interim_1_y = (int32_t *) calloc(width_p1, sizeof(int32_t));
        int64_t *interim_2_x = (int64_t *) calloc(width_p1, sizeof(int64_t));
        int64_t *interim_2_y = (int64_t *) calloc(width_p1, sizeof(int64_t));

        int i = 0;
        dwt2_dtype src_x_val, src_y_val;
        int32_t src_xx_val, src_yy_val;

        // The height loop is broken into 2 parts,
        // 1st loop, prev kh row is not available to subtract during vertical summation
        for(i = 1; i < kh + 1; i++) {
            int src_offset = (i - 1) * r_width;

            /**
             * In this loop the pixels are summated vertically and stored in interim buffer
             * The interim buffer is of size 1 row
             * inter_sum = prev_inter_sum + cur_metric_val
             *
             * where inter_sum will have vertical pixel sums,
             * prev_inter_sum will have prev rows inter_sum and
             * cur_metric_val can be srcx or srcy or srcxx or srcyy or srcxy
             * The previous kh row metric val is not subtracted since it is not available here
             */
            for(int j = 1; j < width_p1; j++) {
                int j_minus1 = j - 1;
                src_x_val = x_pad_t[src_offset + j_minus1];
                src_y_val = y_pad_t[src_offset + j_minus1];
                src_xx_val = (int32_t) src_x_val * src_x_val;
                src_yy_val = (int32_t) src_y_val * src_y_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val;
            }
        }
        /**
         * The strred score calculations would start from the kh,kw index of var & covar
         * Hence horizontal sum of first kh rows are not used, hence that computation is avoided
         */
        // score computation for 1st row of variance & covariance i.e. kh row of padded img

        if(check_enable_spatial_csf == 1)
            agg_abs_accum += strred_horz_integralsum_spatial_csf(
                kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_18, log_22,
                interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x,
                spat_scales_y, i - kh, shift_val);
        else
            agg_abs_accum += strred_horz_integralsum_wavelet(
                kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_18, log_22,
                interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x,
                spat_scales_y, i - kh, shift_val);

        // 2nd loop, core loop
        for(; i < height_p1; i++) {
            int src_offset = (i - 1) * r_width;
            int pre_kh_src_offset = (i - 1 - kh) * r_width;
            int src_x_prekh_val, src_y_prekh_val;
            int src_xx_prekh_val, src_yy_prekh_val;
            /**
             * This loop is similar to the loop across columns seen in 1st for loop
             * In this loop the pixels are summated vertically and stored in interim buffer
             * The interim buffer is of size 1 row
             * inter_sum = prev_inter_sum + cur_metric_val - prev_kh-row_metric_val
             */
            for(int j = 1; j < width_p1; j++) {
                int j_minus1 = j - 1;
                src_x_val = x_pad_t[src_offset + j_minus1];
                src_y_val = y_pad_t[src_offset + j_minus1];

                src_x_prekh_val = x_pad_t[pre_kh_src_offset + j_minus1];
                src_y_prekh_val = y_pad_t[pre_kh_src_offset + j_minus1];
                src_xx_val = (int32_t) src_x_val * src_x_val;
                src_yy_val = (int32_t) src_y_val * src_y_val;
                src_xx_prekh_val = (int32_t) src_x_prekh_val * src_x_prekh_val;
                src_yy_prekh_val = (int32_t) src_y_prekh_val * src_y_prekh_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val - src_x_prekh_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val - src_y_prekh_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val - src_xx_prekh_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val - src_yy_prekh_val;
            }

            // horizontal summation and score compuations
            if(check_enable_spatial_csf == 1)
                agg_abs_accum += strred_horz_integralsum_spatial_csf(
                    kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_18,
                    log_22, interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal,
                    spat_scales_x, spat_scales_y, (i - kh) * width_p1, shift_val);
            else
                agg_abs_accum += strred_horz_integralsum_wavelet(
                    kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_18,
                    log_22, interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal,
                    spat_scales_x, spat_scales_y, (i - kh) * width_p1, shift_val);
        }

        free(interim_1_x);
        free(interim_1_y);
        free(interim_2_x);
        free(interim_2_y);
    }

#if STRRED_REFLECT_PAD
    free(x_pad_t);
    free(y_pad_t);
#endif
    return agg_abs_accum;
}

int integer_compute_strred_funque_neon(const struct i_dwt2buffers *ref,
                                    const struct i_dwt2buffers *dist,
                                    struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist,
                                    size_t width, size_t height,
                                    struct strred_results *strred_scores, int block_size, int level,
                                    uint32_t *log_18, uint32_t *log_22, int32_t shift_val_arg,
                                    double sigma_nsq_t, uint8_t check_enable_spatial_csf)
{
    int ret;
    UNUSED(block_size);
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband;
    float spat_values[DEFAULT_STRRED_SUBBANDS], temp_values[DEFAULT_STRRED_SUBBANDS];
    float fspat_val[DEFAULT_STRRED_SUBBANDS], ftemp_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    /* amount of reflecting */
    int x_reflect = (int) ((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    float *scales_spat_x = (float *) calloc(r_width * r_height, sizeof(float));
    float *scales_spat_y = (float *) calloc(r_width * r_height, sizeof(float));

    for(subband = 1; subband < total_subbands; subband++) {
        enable_temp = 0;
        spat_values[subband] = 0;

        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else {
            shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
        }
        spat_values[subband] = integer_rred_entropies_and_scales_neon(
            ref->bands[subband], dist->bands[subband], width, height, log_18, log_22, sigma_nsq_t,
            shift_val, enable_temp, scales_spat_x, scales_spat_y, check_enable_spatial_csf);
        fspat_val[subband] = spat_values[subband] / (width * height);

        if(prev_ref != NULL && prev_dist != NULL) {
            enable_temp = 1;
            dwt2_dtype *ref_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            dwt2_dtype *dist_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            temp_values[subband] = 0;

            integer_subract_subbands_neon(ref->bands[subband], prev_ref->bands[subband], ref_temporal,
                                     dist->bands[subband], prev_dist->bands[subband], dist_temporal,
                                     width, height);
            temp_values[subband] = integer_rred_entropies_and_scales_neon(
                ref_temporal, dist_temporal, width, height, log_18, log_22, sigma_nsq_t, shift_val,
                enable_temp, scales_spat_x, scales_spat_y, check_enable_spatial_csf);
            ftemp_val[subband] = temp_values[subband] / (width * height);

            free(ref_temporal);
            free(dist_temporal);
        }
    }
    strred_scores->spat_vals[level] = (fspat_val[1] + fspat_val[2] + fspat_val[3]) / 3;
    strred_scores->temp_vals[level] = (ftemp_val[1] + ftemp_val[2] + ftemp_val[3]) / 3;
    strred_scores->spat_temp_vals[level] =
        strred_scores->spat_vals[level] * strred_scores->temp_vals[level];

    // Add equations to compute ST-RRED using norm factors
    int norm_factor, num_level;
    for(num_level = 0; num_level <= level; num_level++)
        norm_factor = num_level + 1;

    strred_scores->spat_vals_cumsum += strred_scores->spat_vals[level];
    strred_scores->temp_vals_cumsum += strred_scores->temp_vals[level];
    strred_scores->spat_temp_vals_cumsum += strred_scores->spat_temp_vals[level];

    strred_scores->srred_vals[level] = strred_scores->spat_vals_cumsum / norm_factor;
    strred_scores->trred_vals[level] = strred_scores->temp_vals_cumsum / norm_factor;
    strred_scores->strred_vals[level] = strred_scores->spat_temp_vals_cumsum / norm_factor;

    free(scales_spat_x);
    free(scales_spat_y);

    ret = 0;
    return ret;
}