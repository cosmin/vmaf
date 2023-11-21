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
void strred_funque_log_generate(uint32_t* log_18)
{
    uint64_t i;
    uint64_t start = (unsigned int)pow(2, 17);
    uint64_t end = (unsigned int)pow(2, 18);
	for (i = start; i < end; i++)
    {
		log_18[i] = (uint32_t)round(log2((double)i) * (1 << STRRED_Q_FORMAT));
    }
}

void strred_funque_generate_log22(uint32_t* log_22)
{
    uint64_t i;
    uint64_t start = (unsigned int)pow(2, 21);
    uint64_t end = (unsigned int)pow(2, 22);
	for (i = start; i < end; i++)
    {
		log_22[i] = (uint32_t)round(log2((double)i) * (1 << STRRED_Q_FORMAT));
    }
}

uint32_t strred_get_best_u18_from_u64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 46)
    {
        k -= 46;
        temp = temp << k;
        *x = -k;

    }
    else if (k < 45)
    {
        k = 46 - k;
        temp = temp >> k;
        *x = k;
    }
    else
    {
        *x = 0;
        if (temp >> 18)
        {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (uint32_t)temp;
}

uint32_t strred_get_best_u22_from_u64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 42)
    {
        k -= 42;
        temp = temp << k;
        *x = -k;

    }
    else if (k < 41)
    {
        k = 42 - k;
        temp = temp >> k;
        *x = k;
    }
    else
    {
        *x = 0;
        if (temp >> 22)
        {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (uint32_t)temp;
}

void strred_integer_reflect_pad(const dwt2_dtype* src, size_t width, size_t height, int reflect, dwt2_dtype* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(dwt2_dtype) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
    }
}

float strred_horz_integralsum_spatial_csf(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   uint32_t entr_const, double sigma_nsq_arg, uint32_t *log_18, uint32_t *log_22,
                                   int32_t *interim_1_x, int64_t *interim_2_x,
                                   int32_t *interim_1_y, int64_t *interim_2_y, uint8_t enable_temporal,
                                   float *spat_scales_x, float *spat_scales_y, int32_t spat_row_idx, int32_t pending_div_fac)
{
    static int32_t int_1_x, int_1_y;
    static int64_t int_2_x, int_2_y;
    int32_t mx, my;
    int32_t var_x, var_y;

    //1st column vals are 0, hence intialising to 0
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

           float temp_mul_fac = ((256 * 256 * 128) / (255.0*255*81));
           float fentropy_x, fentropy_y, fscale_x, fscale_y;
           float aggregate = 0;

           int64_t div_fac = (int64_t)(1 << pending_div_fac) * 255 * 255 * 81;
           uint64_t sigma_nsq = div_fac * sigma_nsq_arg;
           uint64_t const_val = div_fac;
           int64_t sub_val = (int64_t)((log2(255.0 * 255 * 81) + pending_div_fac) * TWO_POWER_Q_FACTOR);

    for (int j=1; j<kw+1; j++)
    {
        int_1_x = interim_1_x[j] + int_1_x;
        int_1_y = interim_1_y[j] + int_1_y;
        int_2_x = interim_2_x[j] + int_2_x;
        int_2_y = interim_2_y[j] + int_2_y;
    }

    {
        mx = int_1_x;
        my = int_1_y;
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift));
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift));
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

                   mul_x = (uint64_t)(var_x + sigma_nsq);
                   mul_y = (uint64_t)(var_y + sigma_nsq);
                   e_look_x = strred_get_best_u22_from_u64((uint64_t)mul_x, &ex);
                   e_look_y = strred_get_best_u22_from_u64((uint64_t)mul_y, &ey);
                   entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
                   entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

#if 1
                   add_x = (uint64_t)((var_x + const_val));
                   add_y = (uint64_t)((var_y + const_val));
                   s_look_x = strred_get_best_u22_from_u64((uint64_t)add_x, &sx);
                   s_look_y = strred_get_best_u22_from_u64((uint64_t)add_y, &sy);
                   scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
                   scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);
#else
//                   add_x = (uint64_t)(var_x + const_val);
//                   add_y = (uint64_t)(var_y + const_val);
//                   fscale_x = ((log2(add_x) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
//                   fscale_y = ((log2(add_y) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
#endif

                   entropy_x = entropy_x - sub_val;
                   entropy_y = entropy_y - sub_val;
                   scale_x = scale_x - sub_val;
                   scale_y = scale_y - sub_val;

                   fentropy_x = (float)entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fentropy_y = (float)entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fscale_x = (float)scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fscale_y = (float)scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

                    if(enable_temporal == 1)
                    {
                        aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx] - fentropy_y * fscale_y * spat_scales_y[spat_row_idx]);
                    }
                    else
                    {
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

    //Similar to prev loop, but previous kw col interim metric sum is subtracted
    for (int j=kw+1; j<width_p1; j++)
    {
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];

        mx = int_1_x;
        my = int_1_y;
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift));
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift));
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;


                   mul_x = (uint64_t)(var_x + sigma_nsq);
                   mul_y = (uint64_t)(var_y + sigma_nsq);
                   e_look_x = strred_get_best_u22_from_u64((uint64_t)mul_x, &ex);
                   e_look_y = strred_get_best_u22_from_u64((uint64_t)mul_y, &ey);
                   entropy_x = log_22[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
                   entropy_y = log_22[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;
#if 1
                   add_x = (uint64_t)((var_x + const_val));
                   add_y = (uint64_t)((var_y + const_val));
                   s_look_x = strred_get_best_u22_from_u64((uint64_t)add_x, &sx);
                   s_look_y = strred_get_best_u22_from_u64((uint64_t)add_y, &sy);
                   scale_x = log_22[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
                   scale_y = log_22[s_look_y] + (sy * TWO_POWER_Q_FACTOR);
#else
//                   add_x = (uint64_t)(var_x + const_val);
//                   add_y = (uint64_t)(var_y + const_val);
//                   fscale_x = ((log2(add_x) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
//                   fscale_y = ((log2(add_y) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
#endif
                   entropy_x = entropy_x - sub_val;
                   entropy_y = entropy_y - sub_val;
                   scale_x = scale_x - sub_val;
                   scale_y = scale_y - sub_val;

                   fentropy_x = (float)entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fentropy_y = (float)entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fscale_x = (float)scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fscale_y = (float)scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

                    if(enable_temporal == 1)
                    {
                        aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx + j - kw] - fentropy_y * fscale_y * spat_scales_y[spat_row_idx + j - kw]);
                    }
                    else
                    {
                        spat_scales_x[spat_row_idx + j - kw] = fscale_x;
                        spat_scales_y[spat_row_idx + j - kw] = fscale_y;
                        aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
                    }
    }
    return aggregate;
}


//This function does summation of horizontal intermediate_vertical_sums & then 
//numerator denominator score calculations are done
float strred_horz_integralsum_wavelet(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   uint32_t entr_const, double sigma_nsq_arg, uint32_t *log_18,
                                   int32_t *interim_1_x, int64_t *interim_2_x,
                                   int32_t *interim_1_y, int64_t *interim_2_y, uint8_t enable_temporal,
                                   float *spat_scales_x, float *spat_scales_y, int32_t spat_row_idx, int32_t pending_div_fac)
{

    static int32_t int_1_x, int_1_y;
    static int64_t int_2_x, int_2_y;
    int32_t mx, my;
    int32_t var_x, var_y;

    //1st column vals are 0, hence intialising to 0
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
           float temp_mul_fac = ((256 * 256 * 128) / (255.0*255*81));

           int64_t div_fac = (int64_t)(1 << pending_div_fac) * 255 * 255 * 81;
           uint64_t sigma_nsq = div_fac * sigma_nsq_arg;
           uint64_t const_val = div_fac;
           int64_t sub_val = (int64_t)((log2(255.0 * 255 * 81) + pending_div_fac) * TWO_POWER_Q_FACTOR);

    for (int j=1; j<kw+1; j++)
    {
        // int j_minus1 = j-1;
        int_1_x = interim_1_x[j] + int_1_x;
        int_1_y = interim_1_y[j] + int_1_y;
        int_2_x = interim_2_x[j] + int_2_x;
        int_2_y = interim_2_y[j] + int_2_y;
    }

    {
        mx = int_1_x;
        my = int_1_y;
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift));
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift));
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

                   mul_x = (uint64_t)(var_x + sigma_nsq);
                   mul_y = (uint64_t)(var_y + sigma_nsq);
                   e_look_x = strred_get_best_u18_from_u64((uint64_t)mul_x, &ex);
                   e_look_y = strred_get_best_u18_from_u64((uint64_t)mul_y, &ey);
                   entropy_x = log_18[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
                   entropy_y = log_18[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

#if 0
                   add_x = (uint64_t)(var_x + const_val);
                   add_y = (uint64_t)(var_y + const_val);
                   s_look_x = strred_get_best_u22_from_u64((uint64_t)add_x, &sx);
                   s_look_y = strred_get_best_u22_from_u64((uint64_t)add_y, &sy);
                   scale_x = log_18[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
                   scale_y = log_18[s_look_y] + (sy * TWO_POWER_Q_FACTOR);
#else
                    add_x = (uint64_t)(var_x + const_val);
                    add_y = (uint64_t)(var_y + const_val);
                    fscale_x = ((log2(add_x) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                    fscale_y = ((log2(add_y) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
#endif
                   entropy_x = entropy_x - sub_val;
                   entropy_y = entropy_y - sub_val;
                   //scale_x = scale_x - sub_val;
                   //scale_y = scale_y - sub_val;

                   fentropy_x = (float)entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fentropy_y = (float)entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   //fscale_x = (float)scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   //fscale_y = (float)scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

                    if(enable_temporal == 1)
                    {
                        aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx] - fentropy_y * fscale_y * spat_scales_y[spat_row_idx]);
                    }
                    else
                    {
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

    //Similar to prev loop, but previous kw col interim metric sum is subtracted
    for (int j=kw+1; j<width_p1; j++)
    {
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];

        mx = int_1_x;
        my = int_1_y;
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift));
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift));
        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;


                   mul_x = (uint64_t)(var_x + sigma_nsq);
                   mul_y = (uint64_t)(var_y + sigma_nsq);
                   e_look_x = strred_get_best_u18_from_u64((uint64_t)mul_x, &ex);
                   e_look_y = strred_get_best_u18_from_u64((uint64_t)mul_y, &ey);
                   entropy_x = log_18[e_look_x] + (ex * TWO_POWER_Q_FACTOR) + entr_const;
                   entropy_y = log_18[e_look_y] + (ey * TWO_POWER_Q_FACTOR) + entr_const;

#if 0
                   add_x = (uint64_t)(var_x + const_val);
                   add_y = (uint64_t)(var_y + const_val);
                   s_look_x = strred_get_best_u18_from_u64((uint64_t)add_x, &sx);
                   s_look_y = strred_get_best_u18_from_u64((uint64_t)add_y, &sy);
                   scale_x = log_18[s_look_x] + (sx * TWO_POWER_Q_FACTOR);
                   scale_y = log_18[s_look_y] + (sy * TWO_POWER_Q_FACTOR);
#else
                   add_x = (uint64_t)(var_x + const_val);
                   add_y = (uint64_t)(var_y + const_val);
                   fscale_x = ((log2(add_x) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fscale_y = ((log2(add_y) * TWO_POWER_Q_FACTOR) - sub_val) / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
#endif
                   entropy_x = entropy_x - sub_val;
                   entropy_y = entropy_y - sub_val;
                   scale_x = scale_x - sub_val;
                   scale_y = scale_y - sub_val;

                   fentropy_x = (float)entropy_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   fentropy_y = (float)entropy_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   //fscale_x = (float)scale_x / (TWO_POWER_Q_FACTOR * LOGE_BASE2);
                   //fscale_y = (float)scale_y / (TWO_POWER_Q_FACTOR * LOGE_BASE2);

                    if(enable_temporal == 1)
                    {
                        aggregate += fabs(fentropy_x * fscale_x * spat_scales_x[spat_row_idx + j - kw] - fentropy_y * fscale_y * spat_scales_y[spat_row_idx + j - kw]);
                    }
                    else
                    {
                        spat_scales_x[spat_row_idx + j - kw] = fscale_x;
                        spat_scales_y[spat_row_idx + j - kw] = fscale_y;
                        aggregate += fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
                    }
    }
    return aggregate;
}


float integer_rred_entropies_and_scales(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, uint32_t* log_18, uint32_t *log_22, double sigma_nsq_arg, int32_t shift_val, uint8_t enable_temporal, float *spat_scales_x, float *spat_scales_y, uint8_t check_enable_spatial_csf)
{
    int ret = 1;

    int kh = STRRED_WINDOW_SIZE;
    int kw = STRRED_WINDOW_SIZE;
    int k_norm = kw * kh;

    /* amount of reflecting */
    int x_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    int y_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    size_t strred_width, strred_height;

#if STRRED_REFLECT_PAD
    strred_width  = width;
    strred_height = height;
#else
    strred_width = width - (2 * x_reflect);
    strred_height = height - (2 * x_reflect);
#endif

    size_t r_width = strred_width + (2 * x_reflect);
    size_t r_height = strred_height + (2 * x_reflect);
    dwt2_dtype* x_pad_t;
    dwt2_dtype* y_pad_t;

#if STRRED_REFLECT_PAD
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (strred_width + (2 * x_reflect)) * (strred_height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (strred_width + (2 * y_reflect)) * (strred_height + (2 * y_reflect)));

    strred_integer_reflect_pad(x_t, strred_width, strred_height, x_reflect, x_pad_t);
    strred_integer_reflect_pad(y_t, strred_width, strred_height, y_reflect, y_pad_t);

#else
    x_pad_t = x_t;
    y_pad_t = y_t;
#endif

    float agg_abs_accum = 0;
    int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    uint32_t entr_const = (uint32_t)(log2f(2 * M_PI * EULERS_CONSTANT) * TWO_POWER_Q_FACTOR);

    {
        int width_p1 = r_width + 1;
        int height_p1 = r_height + 1;

        int32_t *interim_1_x = (int32_t*)calloc(width_p1, sizeof(int32_t));
        int32_t *interim_1_y = (int32_t*)calloc(width_p1, sizeof(int32_t));
        int64_t *interim_2_x = (int64_t*)calloc(width_p1, sizeof(int64_t));
        int64_t *interim_2_y = (int64_t*)calloc(width_p1, sizeof(int64_t));

        int i = 0;
        dwt2_dtype src_x_val, src_y_val;
        int32_t src_xx_val, src_yy_val;

        //The height loop is broken into 2 parts, 
        //1st loop, prev kh row is not available to subtract during vertical summation
        for (i=1; i<kh+1; i++)
        {
            int src_offset = (i-1) * r_width;

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
            for (int j=1; j<width_p1; j++)
            {
                int j_minus1 = j-1;
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
        //score computation for 1st row of variance & covariance i.e. kh row of padded img

        if(check_enable_spatial_csf == 1)
            agg_abs_accum += strred_horz_integralsum_spatial_csf(kw, width_p1, knorm_fact, knorm_shift, 
                             entr_const, sigma_nsq_arg, log_18, log_22,
                             interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x, spat_scales_y, i-kh, shift_val);
        else
            agg_abs_accum += strred_horz_integralsum_wavelet(kw, width_p1, knorm_fact, knorm_shift, 
                             entr_const, sigma_nsq_arg, log_18,
                             interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x, spat_scales_y, i-kh, shift_val);

        //2nd loop, core loop 
        for(; i<height_p1; i++)
        {
            int src_offset = (i-1) * r_width;
            int pre_kh_src_offset = (i-1-kh) * r_width;
            int src_x_prekh_val, src_y_prekh_val;
            int src_xx_prekh_val, src_yy_prekh_val;
            /**
             * This loop is similar to the loop across columns seen in 1st for loop
             * In this loop the pixels are summated vertically and stored in interim buffer
             * The interim buffer is of size 1 row
             * inter_sum = prev_inter_sum + cur_metric_val - prev_kh-row_metric_val
            */
            for (int j=1; j<width_p1; j++)
            {
                int j_minus1 = j-1;
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

            //horizontal summation and score compuations
            if(check_enable_spatial_csf == 1)
                agg_abs_accum += strred_horz_integralsum_spatial_csf(kw, width_p1, knorm_fact, knorm_shift,  
                                 entr_const, sigma_nsq_arg, log_18, log_22,
                                 interim_1_x, interim_2_x, 
                                 interim_1_y, interim_2_y, enable_temporal, spat_scales_x, spat_scales_y, (i-kh)*width_p1, shift_val);
            else
                agg_abs_accum += strred_horz_integralsum_wavelet(kw, width_p1, knorm_fact, knorm_shift,  
                                 entr_const, sigma_nsq_arg, log_18, 
                                 interim_1_x, interim_2_x, 
                                 interim_1_y, interim_2_y, enable_temporal, spat_scales_x, spat_scales_y, (i-kh)*width_p1, shift_val);
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

int integer_copy_prev_frame_strred_funque_c(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                                  struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                                  size_t width, size_t height)
{
    int subband;
    int total_subbands = DEFAULT_STRRED_SUBBANDS;

    for(subband = 1; subband < total_subbands; subband++) {
        memcpy(prev_ref->bands[subband], ref->bands[subband], width * height * sizeof(dwt2_dtype));
        memcpy(prev_dist->bands[subband], dist->bands[subband], width * height * sizeof(dwt2_dtype));
    }

    return 0;
}

void integer_subract_subbands(const dwt2_dtype* ref_src, const dwt2_dtype* ref_prev_src, dwt2_dtype* ref_dst,
                      const dwt2_dtype* dist_src, const dwt2_dtype* dist_prev_src, dwt2_dtype* dist_dst,
                      size_t width, size_t height)
{
    size_t i, j;

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

int integer_compute_strred_funque_c(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                          struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                          size_t width, size_t height, struct strred_results* strred_scores,
                          int block_size, int level, uint32_t *log_18, uint32_t *log_22, int32_t shift_val_arg,
                          double sigma_nsq_t, uint8_t check_enable_spatial_csf)
{
    int ret;

    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband, num_level;
    float spat_values[DEFAULT_STRRED_SUBBANDS], temp_values[DEFAULT_STRRED_SUBBANDS];
    float fspat_val[DEFAULT_STRRED_SUBBANDS], ftemp_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    /* amount of reflecting */
    int x_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    int y_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    float *scales_spat_x = (float*)calloc(r_width * r_height, sizeof(float));
    float *scales_spat_y = (float*)calloc(r_width * r_height, sizeof(float));

    for(subband = 1; subband < total_subbands; subband++) {
        size_t i, j;
        enable_temp = 0;
        int32_t Q_Factor = 0;
        spat_values[subband] = 0;

        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else
       {
           shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
       }

        spat_values[subband] = integer_rred_entropies_and_scales(ref->bands[subband], dist->bands[subband], width, height, log_18, log_22, sigma_nsq_t, shift_val, enable_temp, scales_spat_x, scales_spat_y, check_enable_spatial_csf);
        fspat_val[subband] = spat_values[subband] / (width * height);

        if(prev_ref != NULL && prev_dist != NULL) {
            enable_temp = 1;
            dwt2_dtype* ref_temporal = (dwt2_dtype*) calloc(width * height, sizeof(dwt2_dtype));
            dwt2_dtype* dist_temporal = (dwt2_dtype*) calloc(width * height, sizeof(dwt2_dtype));
            temp_values[subband] = 0;

            integer_subract_subbands(ref->bands[subband], prev_ref->bands[subband], ref_temporal, dist->bands[subband], prev_dist->bands[subband], dist_temporal, width, height);
            temp_values[subband] = integer_rred_entropies_and_scales(ref_temporal, dist_temporal, width, height, log_18, log_22, sigma_nsq_t, shift_val, enable_temp, scales_spat_x, scales_spat_y, check_enable_spatial_csf);
            ftemp_val[subband] = temp_values[subband] / (width * height);

            free(ref_temporal);
            free(dist_temporal);
        }
    }
    strred_scores->spat_vals[level] = (fspat_val[1] + fspat_val[2] + fspat_val[3]) / 3;
    strred_scores->temp_vals[level] = (ftemp_val[1] + ftemp_val[2] + ftemp_val[3]) / 3;
    strred_scores->spat_temp_vals[level] = strred_scores->spat_vals[level] * strred_scores->temp_vals[level];

    // Add equations to compute ST-RRED using norm factors
    int norm_factor;
    static double spat_vals_cumsum, temp_vals_cumsum, spat_temp_vals_cumsum;
    for(num_level = 0; num_level <= level; num_level++)
        norm_factor = num_level + 1;

    if(level == 0) {
        spat_vals_cumsum = strred_scores->spat_vals[level];
        temp_vals_cumsum = strred_scores->temp_vals[level];
        spat_temp_vals_cumsum = strred_scores->spat_temp_vals[level];
    } else {
        for(num_level = 1; num_level <= level; num_level++) {
            spat_vals_cumsum += strred_scores->spat_vals[num_level];
            temp_vals_cumsum += strred_scores->temp_vals[num_level];
            spat_temp_vals_cumsum += strred_scores->spat_temp_vals[num_level];
        }
    }

    strred_scores->srred_vals[level] = spat_vals_cumsum / norm_factor;
    strred_scores->trred_vals[level] = temp_vals_cumsum / norm_factor;
    strred_scores->strred_vals[level] = spat_temp_vals_cumsum / norm_factor;

    free(scales_spat_x);
    free(scales_spat_y);

    ret = 0;
    return ret;
}