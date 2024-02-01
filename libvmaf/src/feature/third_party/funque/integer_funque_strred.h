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

#define STRRED_REFLECT_PAD 1

#define VARIANCE_SHIFT_FACTOR 6
#define STRRED_Q_FORMAT 26
#define TWO_POWER_Q_FACTOR (1 << STRRED_Q_FORMAT)

#define LOGE_BASE2 1.442684682

int integer_compute_srred_funque_c(const struct i_dwt2buffers *ref,
                                   const struct i_dwt2buffers *dist, size_t width, size_t height,
                                   float **spat_scales_ref, float **spat_scales_dist,
                                   struct strred_results *strred_scores, int block_size, int level,
                                   uint32_t *log_18, uint32_t *log_22, int32_t shift_val,
                                   double sigma_nsq_t, uint8_t enable_spatial_csf);

int integer_compute_strred_funque_c(const struct i_dwt2buffers *ref,
                                    const struct i_dwt2buffers *dist,
                                    struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist,
                                    size_t width, size_t height, float **spat_scales_ref,
                                    float **spat_scales_dist, struct strred_results *strred_scores,
                                    int block_size, int level, uint32_t *log_18, uint32_t *log_22,
                                    int32_t shift_val, double sigma_nsq_t,
                                    uint8_t enable_spatial_csf);

int integer_copy_prev_frame_strred_funque_c(const struct i_dwt2buffers *ref,
                                            const struct i_dwt2buffers *dist,
                                            struct i_dwt2buffers *prev_ref,
                                            struct i_dwt2buffers *prev_dist, size_t width,
                                            size_t height);

void integer_subract_subbands_c(const dwt2_dtype *ref_src, const dwt2_dtype *ref_prev_src,
                                dwt2_dtype *ref_dst, const dwt2_dtype *dist_src,
                                const dwt2_dtype *dist_prev_src, dwt2_dtype *dist_dst, size_t width,
                                size_t height);

void strred_integer_reflect_pad(const dwt2_dtype *src, size_t width, size_t height, int reflect,
                                dwt2_dtype *dest);

void strred_funque_generate_log22(uint32_t *log_22);
void strred_funque_log_generate(uint32_t *log_18);

FORCE_INLINE inline uint32_t strred_get_best_u22_from_u64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if(k > 42) {
        k -= 42;
        temp = temp << k;
        *x = -k;

    } else if(k < 41) {
        k = 42 - k;
        temp = (temp + (1 << (k - 1))) >> k;
        *x = k;
    } else {
        *x = 0;
        if(temp >> 22) {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (uint32_t) temp;
}

static inline float strred_horz_integralsum_spatial_csf(
    int kw, int width_p1, int16_t knorm_fact, int16_t knorm_shift, uint32_t entr_const,
    double sigma_nsq_arg, uint32_t *log_18, uint32_t *log_22, int32_t *interim_1_x,
    int64_t *interim_2_x, int32_t *interim_1_y, int64_t *interim_2_y, uint8_t enable_temporal,
    float *spat_scales_x, float *spat_scales_y, int32_t spat_row_idx, int32_t pending_div_fac)
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y;
    int32_t mx, my;
    int32_t var_x, var_y;

    // 1st column vals are 0, hence intialising to 0
    int_1_x = 0;
    int_1_y = 0;
    int_2_x = 0;
    int_2_y = 0;

    /**
     * The horizontal accumulation similar to vertical accumulation
     * metric_sum = prev_col_metric_sum + interim_metric_vertical_sum
     * The previous kw col interim metric sum is not subtracted since it is not available here
     */

    int64_t mul_x, mul_y;
    int64_t add_x, add_y;
    int ex, ey, sx, sy;
    uint32_t e_look_x, e_look_y;
    uint32_t s_look_x, s_look_y;
    int64_t entropy_x, entropy_y, scale_x, scale_y;

    float fentropy_x, fentropy_y, fscale_x, fscale_y;
    float aggregate = 0;

    int32_t pending_div_minus_var_fac = pending_div_fac - VARIANCE_SHIFT_FACTOR;
    int64_t div_fac = (int64_t) (1 << pending_div_minus_var_fac) * 255 * 255 * 81;
    uint64_t sigma_nsq = div_fac * sigma_nsq_arg;
    uint64_t const_val = div_fac;
    int64_t sub_val =
        (int64_t) ((log2(255.0 * 255 * 81) + pending_div_minus_var_fac) * TWO_POWER_Q_FACTOR);

    for(int j = 1; j < kw + 1; j++) {
        int_1_x = interim_1_x[j] + int_1_x;
        int_1_y = interim_1_y[j] + int_1_y;
        int_2_x = interim_2_x[j] + int_2_x;
        int_2_y = interim_2_y[j] + int_2_y;
    }

    {
        mx = int_1_x;
        my = int_1_y;
        var_x = ((int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_y = ((int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

        mul_x = (uint64_t) (var_x + sigma_nsq);
        mul_y = (uint64_t) (var_y + sigma_nsq);
        e_look_x = strred_get_best_u22_from_u64((uint64_t) mul_x, &ex);
        e_look_y = strred_get_best_u22_from_u64((uint64_t) mul_y, &ey);
        entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
        entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

        add_x = (uint64_t) ((var_x + const_val));
        add_y = (uint64_t) ((var_y + const_val));
        s_look_x = strred_get_best_u22_from_u64((uint64_t) add_x, &sx);
        s_look_y = strred_get_best_u22_from_u64((uint64_t) add_y, &sy);
        scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
        scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);

        entropy_x = entropy_x - sub_val;
        entropy_y = entropy_y - sub_val;
        scale_x = scale_x - sub_val;
        scale_y = scale_y - sub_val;

        fentropy_x = (float) entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fentropy_y = (float) entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_x = (float) scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_y = (float) scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

        if(enable_temporal == 1) {
            aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx] -
                              fentropy_y * fscale_y * spat_scales_y[spat_row_idx]);
        } else {
            spat_scales_x[spat_row_idx] = fscale_x;
            spat_scales_y[spat_row_idx] = fscale_y;
            aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        }
    }

    /**
     * The score needs to be calculated for kw column as well,
     * whose interim result calc is different from rest of the columns,
     * hence calling strred_calc_entropy_scale for kw column separately
     */

    // Similar to prev loop, but previous kw col interim metric sum is subtracted
    for(int j = kw + 1; j < width_p1; j++) {
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];

        mx = int_1_x;
        my = int_1_y;
        var_x = ((int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_y = ((int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

        mul_x = (uint64_t) (var_x + sigma_nsq);
        mul_y = (uint64_t) (var_y + sigma_nsq);
        e_look_x = strred_get_best_u22_from_u64((uint64_t) mul_x, &ex);
        e_look_y = strred_get_best_u22_from_u64((uint64_t) mul_y, &ey);
        entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
        entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

        add_x = (uint64_t) ((var_x + const_val));
        add_y = (uint64_t) ((var_y + const_val));
        s_look_x = strred_get_best_u22_from_u64((uint64_t) add_x, &sx);
        s_look_y = strred_get_best_u22_from_u64((uint64_t) add_y, &sy);
        scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
        scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);

        entropy_x = entropy_x - sub_val;
        entropy_y = entropy_y - sub_val;
        scale_x = scale_x - sub_val;
        scale_y = scale_y - sub_val;

        fentropy_x = (float) entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fentropy_y = (float) entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_x = (float) scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_y = (float) scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

        if(enable_temporal == 1) {
            aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx + j - kw] -
                              fentropy_y * fscale_y * spat_scales_y[spat_row_idx + j - kw]);
        } else {
            spat_scales_x[spat_row_idx + j - kw] = fscale_x;
            spat_scales_y[spat_row_idx + j - kw] = fscale_y;
            aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        }
    }
    return aggregate;
}

static inline float strred_horz_integralsum_wavelet(
    int kw, int width_p1, int16_t knorm_fact, int16_t knorm_shift, uint32_t entr_const,
    double sigma_nsq_arg, uint32_t *log_18, uint32_t *log_22, int32_t *interim_1_x,
    int64_t *interim_2_x, int32_t *interim_1_y, int64_t *interim_2_y, uint8_t enable_temporal,
    float *spat_scales_x, float *spat_scales_y, int32_t spat_row_idx, int32_t pending_div_fac)
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y;
    int32_t mx, my;
    int32_t var_x, var_y;

    // 1st column vals are 0, hence intialising to 0
    int_1_x = 0;
    int_1_y = 0;
    int_2_x = 0;
    int_2_y = 0;

    /**
     * The horizontal accumulation similar to vertical accumulation
     * metric_sum = prev_col_metric_sum + interim_metric_vertical_sum
     * The previous kw col interim metric sum is not subtracted since it is not available here
     */

    int64_t mul_x, mul_y;
    int64_t add_x, add_y;
    int ex, ey, sx, sy;
    uint32_t e_look_x, e_look_y;
    uint32_t s_look_x, s_look_y;
    int64_t entropy_x, entropy_y, scale_x, scale_y;

    float fentropy_x, fentropy_y, fscale_x, fscale_y;
    float aggregate = 0;

    int32_t pending_div_minus_var_fac = pending_div_fac - VARIANCE_SHIFT_FACTOR;
    int64_t div_fac = (int64_t) (1 << pending_div_minus_var_fac) * 255 * 255 * 81;
    uint64_t sigma_nsq = div_fac * sigma_nsq_arg;
    uint64_t const_val = div_fac;
    int64_t sub_val =
        (int64_t) ((log2(255.0 * 255 * 81) + pending_div_minus_var_fac) * TWO_POWER_Q_FACTOR);

    for(int j = 1; j < kw + 1; j++) {
        int_1_x = interim_1_x[j] + int_1_x;
        int_1_y = interim_1_y[j] + int_1_y;
        int_2_x = interim_2_x[j] + int_2_x;
        int_2_y = interim_2_y[j] + int_2_y;
    }

    {
        mx = int_1_x;
        my = int_1_y;
        var_x = ((int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_y = ((int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

        mul_x = (uint64_t) (var_x + sigma_nsq);
        mul_y = (uint64_t) (var_y + sigma_nsq);
        e_look_x = strred_get_best_u22_from_u64((uint64_t) mul_x, &ex);
        e_look_y = strred_get_best_u22_from_u64((uint64_t) mul_y, &ey);
        entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
        entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

        add_x = (uint64_t) (var_x + const_val);
        add_y = (uint64_t) (var_y + const_val);
        s_look_x = strred_get_best_u22_from_u64((uint64_t) add_x, &sx);
        s_look_y = strred_get_best_u22_from_u64((uint64_t) add_y, &sy);
        scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
        scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);

        entropy_x = entropy_x - sub_val;
        entropy_y = entropy_y - sub_val;
        scale_x = scale_x - sub_val;
        scale_y = scale_y - sub_val;

        fentropy_x = (float) entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fentropy_y = (float) entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_x = (float) scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_y = (float) scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

        if(enable_temporal == 1) {
            aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx] -
                              fentropy_y * fscale_y * spat_scales_y[spat_row_idx]);
        } else {
            spat_scales_x[spat_row_idx] = fscale_x;
            spat_scales_y[spat_row_idx] = fscale_y;
            aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        }
    }

    /**
     * The score needs to be calculated for kw column as well,
     * whose interim result calc is different from rest of the columns,
     * hence calling strred_calc_entropy_scale for kw column separately
     */

    // Similar to prev loop, but previous kw col interim metric sum is subtracted
    for(int j = kw + 1; j < width_p1; j++) {
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];

        mx = int_1_x;
        my = int_1_y;
        var_x = ((int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_y = ((int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) +
                 (1 << (VARIANCE_SHIFT_FACTOR - 1))) >>
                VARIANCE_SHIFT_FACTOR;
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

        mul_x = (uint64_t) (var_x + sigma_nsq);
        mul_y = (uint64_t) (var_y + sigma_nsq);
        e_look_x = strred_get_best_u22_from_u64((uint64_t) mul_x, &ex);
        e_look_y = strred_get_best_u22_from_u64((uint64_t) mul_y, &ey);
        entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
        entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

        add_x = (uint64_t) (var_x + const_val);
        add_y = (uint64_t) (var_y + const_val);
        s_look_x = strred_get_best_u22_from_u64((uint64_t) add_x, &sx);
        s_look_y = strred_get_best_u22_from_u64((uint64_t) add_y, &sy);
        scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
        scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);

        entropy_x = entropy_x - sub_val;
        entropy_y = entropy_y - sub_val;
        scale_x = scale_x - sub_val;
        scale_y = scale_y - sub_val;

        fentropy_x = (float) entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fentropy_y = (float) entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_x = (float) scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
        fscale_y = (float) scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

        if(enable_temporal == 1) {
            aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx + j - kw] -
                              fentropy_y * fscale_y * spat_scales_y[spat_row_idx + j - kw]);
        } else {
            spat_scales_x[spat_row_idx + j - kw] = fscale_x;
            spat_scales_y[spat_row_idx + j - kw] = fscale_y;
            aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        }
    }
    return aggregate;
}

static inline float integer_rred_entropies_and_scales(const dwt2_dtype *x_t, const dwt2_dtype *y_t,
                                                      size_t width, size_t height, uint32_t *log_18,
                                                      uint32_t *log_22, double sigma_nsq_arg,
                                                      int32_t shift_val, uint8_t enable_temporal,
                                                      float *spat_scales_x, float *spat_scales_y,
                                                      uint8_t check_enable_spatial_csf)
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
    uint32_t entr_const =
        (uint32_t) (log2f(2 * PI_CONSTANT * EULERS_CONSTANT) * TWO_POWER_Q_FACTOR);

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
