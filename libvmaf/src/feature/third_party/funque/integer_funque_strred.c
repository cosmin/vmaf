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

#if USE_LOG_18
// just change the store offset to reduce multiple calculation when getting log value
void strred_funque_log_generate(uint32_t* log_18)
{
    uint64_t i;
    uint64_t start = (unsigned int)pow(2, 17);
    uint64_t end = (unsigned int)pow(2, 18);
	for (i = start; i < end; i++)
    {
		log_18[i] = (uint32_t)round(log((double)i) * (1 << Q_FORMAT_TO_MULTIPLY_LOG));
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

#else

uint32_t strred_get_best_u18_from_u32(uint32_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 14)
    {
        k -= 14;
        temp = temp << k;
        *x = -k;

    }
    else if (k < 13)
    {
        k = 14 - k;
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


void strred_log_generate(uint16_t *log_16)
{
    for (unsigned i = 32767; i < 65536; ++i) {
        log_16[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
}

uint16_t strred_get_best_u16_from_u64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 48)
    {
        k -= 48;
        temp = temp << k;
        *x = -k;

    }
    else if (k < 47)
    {
        k = 48 - k;
        temp = temp >> k;
        *x = k;
    }
    else
    {
        *x = 0;
        if (temp >> 16)
        {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (uint16_t)temp;
}

//int32_t strred_log2_32(const uint16_t log_16, uint32_t temp)
//{
//    int k = __builtin_clz(temp);
//    k = 16 - k;
//    temp = temp >> k;
//    return log_16[temp] + 2048 * k;
//}
//
//int32_t strred_log2_64(const uint16_t log_16, uint64_t x)
//{
//    //assert(temp >= 0x20000);
//    int k = __builtin_clzll(x);
//    k = 48 - k;
//    x = x >> k;
//    return log_16[x] + 2048 * k;
//}
#endif

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

//This function does summation of horizontal intermediate_vertical_sums & then 
//numerator denominator score calculations are done
#if STRRED_STABILITY
static inline void strred_horz_integralsum(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   int16_t exp, int32_t sigma_nsq, uint32_t *log_18,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num, int64_t *num_power,
                                   int64_t *score_den, int64_t *den_power, int64_t shift_val, int k_norm)
#else
static inline void strred_horz_integralsum(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   uint32_t entr_const, uint32_t sigma_nsq, uint32_t *log_18,
                                   int32_t *interim_1_x, int64_t *interim_2_x,
                                   int32_t *interim_1_y, int64_t *interim_2_y, float *spat_abs_accum, int16_t *power_factor)
#endif
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y;
    int32_t mx, my;
    int32_t var_x, var_y;

    uint32_t entropy_x, entropy_y, scale_x, scale_y;
    float spat_aggregate;

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
    int32_t add_x, add_y;
    uint64_t const_val;
    int16_t ex, ey, sx, sy;
    uint32_t look_x, look_y;
    float fentropy_x, fentropy_y, fscale_x, fscale_y;

    // FILE *fptr;
    // fptr = fopen("debug_int.txt", "w");

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
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) >> STRRED_COMPUTE_METRIC_R_SHIFT;
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) >> STRRED_COMPUTE_METRIC_R_SHIFT;

        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

#if USE_FLOAT_CODE
        const_val = (1 << 32);
        fentropy_x = log((var_x + sigma_nsq) * entr_const);
        fentropy_y = log((var_y + sigma_nsq) * entr_const);
        fscale_x = log(const_val + var_x);
        fscale_y = log(const_val + var_y);

        spat_aggregate = fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        *spat_abs_accum += spat_aggregate;

#else

        mul_x = (int64_t)(var_x + sigma_nsq) * entr_const;
        mul_y = (int64_t)(var_y + sigma_nsq) * entr_const;
        look_x = strred_get_best_u18_from_u64((uint64_t)mul_x, &ex);
        look_y = strred_get_best_u18_from_u64((uint64_t)mul_y, &ey);
        entropy_x = log_18[look_x]; // Div by Q26 to compare with float
        entropy_y = log_18[look_y];


        const_val = (1 << 32);
        add_x = (int64_t)var_x + const_val;
        add_y = (int64_t)var_y + const_val;
        look_x = strred_get_best_u18_from_u64((uint64_t)add_x, &sx);
        look_y = strred_get_best_u18_from_u64((uint64_t)add_y, &sy);
        scale_x = log_18[look_x];
        scale_y = log_18[look_y];

#if KEEP_SPAT_IN_INTEGER

        float fentropy_x = entropy_x / (1 << (Q_FORMAT_TO_MULTIPLY_LOG + ex)); // Divide here by the Q-Factor to match score with Float
        float fentropy_y = entropy_y / (1 << (Q_FORMAT_TO_MULTIPLY_LOG + ey));
        float fscale_x = scale_x  / (1 << (Q_FORMAT_TO_MULTIPLY_LOG + sx));
        float fscale_y = scale_y  / (1 << (Q_FORMAT_TO_MULTIPLY_LOG + sy));

        spat_aggregate = abs(entropy_x * scale_x - entropy_y * scale_y);
        *power_factor += ex * sx / ey * sy;
        *spat_abs_accum += spat_aggregate;
#else

        fentropy_x = entropy_x / (1 << ex); // Divide here by the Q-Factor to match score with Float
        fentropy_y = entropy_y / (1 << ey);
        fscale_x = scale_x  / (1 << sx);
        fscale_y = scale_y  / (1 << sy);

        fentropy_x = fentropy_x / (1 << Q_FORMAT_TO_MULTIPLY_LOG); // Divide here by the Q-Factor to match score with Float
        fentropy_y = fentropy_y / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        fscale_x = fscale_x  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        fscale_y = fscale_y  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);

        spat_aggregate = fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        *spat_abs_accum += spat_aggregate;

        //spat_aggregate = abs(entropy_x * scale_x - entropy_y * scale_y);
        //*power_factor += ex * sx / ey * sy;
        //*spat_abs_accum += spat_aggregate;

#endif
#endif


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
        var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) >> STRRED_COMPUTE_METRIC_R_SHIFT;
        var_y = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) >> STRRED_COMPUTE_METRIC_R_SHIFT;

        var_x = (var_x < 0) ? 0 : var_x;
        var_y = (var_y < 0) ? 0 : var_y;

#if USE_FLOAT_CODE

        const_val = (1 << 32);
        fentropy_x = log((var_x + sigma_nsq) * entr_const);
        fentropy_y = log((var_y + sigma_nsq) * entr_const);
        fscale_x = log(const_val + var_x);
        fscale_y = log(const_val + var_y);

        spat_aggregate = fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        *spat_abs_accum += spat_aggregate;

#else

#if !USE_LOG_18
        entropy_x = log(var_x + sigma_nsq) + entr_const;
        entropy_y = log(var_y + sigma_nsq) + entr_const;
        scale_x = log(1 + var_x);
        scale_y = log(1 + var_y);
#else
        mul_x = (uint64_t)(var_x + sigma_nsq) * entr_const;
        mul_y = (uint64_t)(var_y + sigma_nsq) * entr_const;
        look_x = strred_get_best_u18_from_u64((uint64_t)mul_x, &ex);
        look_y = strred_get_best_u18_from_u64((uint64_t)mul_y, &ey);
        entropy_x = log_18[look_x]; // Divide here by the Q-Factor to match score with Float
        entropy_y = log_18[look_y];

        const_val = (1 << 32);
        add_x = (uint64_t)var_x + const_val;
        add_y = (uint64_t)var_y + const_val;
        look_x = strred_get_best_u18_from_u64((uint64_t)add_x, &sx);
        look_y = strred_get_best_u18_from_u64((uint64_t)add_y, &sy);
        scale_x = log_18[look_x];
        scale_y = log_18[look_y];
#endif

#if KEEP_SPAT_IN_INTEGER
        float fentropy_x = entropy_x / (1 << Q_FORMAT_TO_MULTIPLY_LOG); // Divide here by the Q-Factor to match score with Float
        float fentropy_y = entropy_y / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        float fscale_x = scale_x  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        float fscale_y = scale_y  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);

        spat_aggregate = abs(entropy_x * scale_x - entropy_y * scale_y);
        *power_factor += ex * sx / ey * sy;
        *spat_abs_accum += spat_aggregate;
#else
        fentropy_x = entropy_x / (1 << ex); // Divide here by the Q-Factor to match score with Float
        fentropy_y = entropy_y / (1 << ey);
        fscale_x = scale_x  / (1 << sx);
        fscale_y = scale_y  / (1 << sy);

        fentropy_x = fentropy_x / (1 << Q_FORMAT_TO_MULTIPLY_LOG); // Divide here by the Q-Factor to match score with Float
        fentropy_y = fentropy_y / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        fscale_x = fscale_x  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);
        fscale_y = fscale_y  / (1 << Q_FORMAT_TO_MULTIPLY_LOG);

        spat_aggregate = fabs(fentropy_x * fscale_x - fentropy_y * fscale_y);
        *spat_abs_accum += spat_aggregate;

#endif
#endif

        // TODO: Add Support ffor log2 for entropy and scale
    }
}


float integer_rred_entropies_and_scales(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, uint32_t* log_18, uint32_t sigma_nsq_arg, int32_t shift_val, int32_t *Q_Fact)
{
    int ret = 1;

    int kh = STRRED_WINDOW_SIZE;
    int kw = STRRED_WINDOW_SIZE;
    int k_norm = kw * kh;

    int x_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2); // amount for reflecting
    int y_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2); // amount for reflecting
    size_t strred_width, strred_height;

#if STRRED_REFLECT_PAD
    strred_width  = width;
    strred_height = height;
#else
    strred_width = width - (2 * x_reflect);
    strred_height = height - (2 * x_reflect);
#endif

    size_t r_width = strred_width + (2 * x_reflect); // after reflect pad
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

    int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    float spat_agg_abs_accum = 0;
    int16_t power_fac = 0;
    uint32_t entr_const = 2 * M_PI * EULERS_CONSTANT * k_norm * knorm_fact * knorm_shift;

    uint32_t sigma_nsq_t = (uint32_t)4*knorm_fact*knorm_fact;
#if STRRED_STABILITY
	double sigma_nsq_base = sigma_nsq_arg / (255.0*255.0);	
#if USE_DYNAMIC_SIGMA_NSQ
	sigma_nsq_base = sigma_nsq_base * (2 << (strred_level + 1));
#endif
	sigma_nsq_t = (int64_t)((int64_t)(sigma_nsq_base*shift_val*shift_val*k_norm)) >> STRRED_COMPUTE_METRIC_R_SHIFT ;
#endif

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

#if STRRED_STABILITY
        strred_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift, 
                             entr_const, sigma_nsq_t, log_18,
                             interim_1_x, interim_1_y,
                             interim_2_x, interim_2_y, interim_x_y,
                             &score_num_t, &num_power, &score_den_t, &den_power, shift_val, k_norm);
#else
        strred_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift, 
                             entr_const, sigma_nsq_t, log_18,
                             interim_1_x, interim_2_x, interim_1_y, interim_2_y, &spat_agg_abs_accum, &power_fac);
//        Q_Fact += power_fac;

#endif

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
#if STRRED_STABILITY
            strred_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift,  
                                 entr_const, sigma_nsq_t, log_18, 
                                 interim_1_x, interim_1_y,
                                 interim_2_x, interim_2_y, interim_x_y,
                                 &score_num_t, &num_power, 
                                 &score_den_t, &den_power, shift_val, k_norm);
#else
            strred_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift,  
                                 entr_const, sigma_nsq_t, log_18, 
                                 interim_1_x, interim_2_x, 
                                 interim_1_y, interim_2_y, &spat_agg_abs_accum, &power_fac);
//            *Q_Fact += power_fac;
#endif
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

    //*Q_Fact = power_fac + 2 * Q_FORMAT_TO_MULTIPLY_LOG;
    return spat_agg_abs_accum;

//    int64_t values = spat_agg_abs_accum / (width * height);
//    return values;
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


int integer_compute_strred_funque_c(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                          struct i_dwt2buffers* prev_ref, struct i_dwt2buffers* prev_dist,
                          size_t width, size_t height, struct strred_results* strred_scores,
                          int block_size, int level, uint32_t *log_18, int32_t shift_val, uint32_t sigma_nsq_t)
{
    int ret;

    size_t subband, num_level;
    float spat_values[DEFAULT_STRRED_SUBBANDS];
    int64_t temp_values[DEFAULT_STRRED_SUBBANDS];
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    float fspat_val[DEFAULT_STRRED_SUBBANDS];

    for(subband = 1; subband < total_subbands; subband++) {
        size_t i, j;
        float val;
        int32_t Q_Factor = 0;
        spat_values[subband] = 0;

        spat_values[subband] = integer_rred_entropies_and_scales(ref->bands[subband], dist->bands[subband], width, height, log_18, sigma_nsq_t, shift_val, &Q_Factor);

        fspat_val[subband] = spat_values[subband] / (width * height);
    }

    strred_scores->spat_vals[level] = (fspat_val[1] + fspat_val[2] + fspat_val[3]) / 3;

    // Add equations to compute ST-RRED using norm factors
    int norm_factor;
    static double spat_vals_cumsum, temp_vals_cumsum, spat_temp_vals_cumsum;
    for(num_level = 0; num_level <= level; num_level++)
        norm_factor = num_level + 1;

    if(level == 0) {
        spat_vals_cumsum = strred_scores->spat_vals[level];
    } else {
        for(num_level = 1; num_level <= level; num_level++) {
            spat_vals_cumsum += strred_scores->spat_vals[num_level];
        }
    }

    strred_scores->srred_vals[level] = spat_vals_cumsum / norm_factor;

    ret = 0;
    return ret;
}