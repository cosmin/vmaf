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

#include <math.h>
#include <mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../funque_vif_options.h"
#include "../integer_funque_filters.h"
#include "../common/macros.h"
#include "../integer_funque_vif.h"
#include "integer_funque_vif_avx512.h"
#include <immintrin.h>

#if USE_DYNAMIC_SIGMA_NSQ
int integer_compute_vif_funque_avx512(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_18, int vif_level)
#else
int integer_compute_vif_funque_avx512(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_18)
#endif
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = kw * kh;

    int x_reflect = (int)((kh - stride) / 2); // amount for reflecting
    int y_reflect = (int)((kw - stride) / 2);
    size_t vif_width, vif_height;

#if VIF_REFLECT_PAD
    vif_width  = width;
    vif_height = height;
#else
    vif_width = width - (2 * y_reflect);
    vif_height = height - (2 * x_reflect);
#endif

    size_t r_width = vif_width + (2 * x_reflect); // after reflect pad
    size_t r_height = vif_height + (2 * x_reflect);    


    dwt2_dtype* x_pad_t, *y_pad_t;

#if VIF_REFLECT_PAD
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (vif_width + (2 * x_reflect)) * (vif_height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (vif_width + (2 * y_reflect)) * (vif_height + (2 * y_reflect)));
    integer_reflect_pad(x_t, vif_width, vif_height, x_reflect, x_pad_t);
    integer_reflect_pad(y_t, vif_width, vif_height, y_reflect, y_pad_t);
#else
    x_pad_t = x_t;
    y_pad_t = y_t;
#endif

    int64_t exp_t = 1; // using 1 because exp in Q32 format is still 0
    int32_t sigma_nsq_t = (int64_t)((int64_t)sigma_nsq_arg*shift_val*shift_val*k_norm) >> VIF_COMPUTE_METRIC_R_SHIFT ;
#if VIF_STABILITY
	double sigma_nsq_base = sigma_nsq_arg / (255.0*255.0);	
#if USE_DYNAMIC_SIGMA_NSQ
	sigma_nsq_base = sigma_nsq_base * (2 << (vif_level + 1));
#endif
	sigma_nsq_t = (int64_t)((int64_t)(sigma_nsq_base*shift_val*shift_val*k_norm)) >> VIF_COMPUTE_METRIC_R_SHIFT ;
#endif
    int64_t score_num_t = 0;
    int64_t num_power = 0;
    int64_t score_den_t = 0;
    int64_t den_power = 0;

    int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 

    {
        int width_p1 = r_width + 1;
        int height_p1 = r_height + 1;
        int64_t *interim_2_x = (int64_t*)malloc(width_p1 * sizeof(int64_t));
        int32_t *interim_1_x = (int32_t*)malloc(width_p1 * sizeof(int32_t));
        int64_t *interim_2_y = (int64_t*)malloc(width_p1 * sizeof(int64_t));
        int32_t *interim_1_y = (int32_t*)malloc(width_p1 * sizeof(int32_t));
        int64_t *interim_x_y = (int64_t*)malloc(width_p1 * sizeof(int64_t));

        memset(interim_2_x, 0, width_p1 * sizeof(int64_t));
        memset(interim_1_x, 0, width_p1 * sizeof(int32_t));
        memset(interim_2_y, 0, width_p1 * sizeof(int64_t));
        memset(interim_1_y, 0, width_p1 * sizeof(int32_t));
        memset(interim_x_y, 0, width_p1 * sizeof(int64_t));

        int i = 0;

        int width_p1_32 = (width_p1) - ((width_p1) % 32);
        int width_p1_16 = (width_p1) - ((width_p1) % 16);
        int width_p1_8 = (width_p1) - (width_p1 % 8);

        __m512i interim_1_x_512, interim_1_x16_512, interim_1_y_512, interim_1_y16_512, \
        interim_2_x0_512, interim_2_x8_512, interim_2_x16_512, interim_2_x24_512, \
        interim_2_y0_512, interim_2_y8_512, interim_2_y16_512, interim_2_y24_512, \
        interim_x_y0_512, interim_x_y8_512, interim_x_y16_512, interim_x_y24_512;

        //The height loop is broken into 2 parts, 
        //1st loop, prev kh row is not available to subtract during vertical summation
        int j = 1;
        for (; j < width_p1_32; j+=32)
        {
            int j_minus1 = j-1;

            interim_1_x_512 = interim_1_x16_512 = interim_1_y_512 = interim_1_y16_512 = \
            interim_2_x0_512 = interim_2_x8_512 = interim_2_x16_512 = interim_2_x24_512 = \
            interim_2_y0_512 = interim_2_y8_512 = interim_2_y16_512 = interim_2_y24_512 = \
            interim_x_y0_512 = interim_x_y8_512 = interim_x_y16_512 = interim_x_y24_512 = _mm512_setzero_si512();

            for (i=1; i<kh+1; i++)
            {
                int src_offset = (i-1) * r_width;

                __m512i src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1)));
                __m512i src_x16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1 + 16)));
                __m512i src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1)));
                __m512i src_y16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1 + 16)));
                // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
                interim_1_x_512 = _mm512_add_epi32(interim_1_x_512, src_x);
                interim_1_x16_512 = _mm512_add_epi32(interim_1_x16_512, src_x16);
                interim_1_y_512 = _mm512_add_epi32(interim_1_y_512, src_y);
                interim_1_y16_512 = _mm512_add_epi32(interim_1_y16_512, src_y16);

                __m512i src_x0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x));
                __m512i src_x8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x, 1));
                __m512i src_x16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x16));
                __m512i src_x24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x16, 1));
                
                __m512i src_y0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y));
                __m512i src_y8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y, 1));
                __m512i src_y16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y16));
                __m512i src_y24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y16, 1));

                __m512i mul_x_x0 = _mm512_mul_epi32(src_x0_64, src_x0_64);
                __m512i mul_x_x8 = _mm512_mul_epi32(src_x8_64, src_x8_64);
                __m512i mul_x_x16 = _mm512_mul_epi32(src_x16_64, src_x16_64);
                __m512i mul_x_x24 = _mm512_mul_epi32(src_x24_64, src_x24_64);

                __m512i mul_y_y0 = _mm512_mul_epi32(src_y0_64, src_y0_64); 
                __m512i mul_y_y8 = _mm512_mul_epi32(src_y8_64, src_y8_64);
                __m512i mul_y_y16 = _mm512_mul_epi32(src_y16_64, src_y16_64); 
                __m512i mul_y_y24 = _mm512_mul_epi32(src_y24_64, src_y24_64);

                __m512i mul_x_y0 = _mm512_mul_epi32(src_x0_64, src_y0_64);
                __m512i mul_x_y8 = _mm512_mul_epi32(src_x8_64, src_y8_64);
                __m512i mul_x_y16 = _mm512_mul_epi32(src_x16_64, src_y16_64);
                __m512i mul_x_y24 = _mm512_mul_epi32(src_x24_64, src_y24_64);

                interim_2_x0_512 = _mm512_add_epi64(interim_2_x0_512, mul_x_x0);
                interim_2_x8_512 = _mm512_add_epi64(interim_2_x8_512, mul_x_x8);
                interim_2_x16_512 = _mm512_add_epi64(interim_2_x16_512, mul_x_x16);
                interim_2_x24_512 = _mm512_add_epi64(interim_2_x24_512, mul_x_x24);

                interim_2_y0_512 = _mm512_add_epi64(interim_2_y0_512, mul_y_y0); 
                interim_2_y8_512 = _mm512_add_epi64(interim_2_y8_512, mul_y_y8);
                interim_2_y16_512 = _mm512_add_epi64(interim_2_y16_512, mul_y_y16); 
                interim_2_y24_512 = _mm512_add_epi64(interim_2_y24_512, mul_y_y24);

                interim_x_y0_512 = _mm512_add_epi64(interim_x_y0_512, mul_x_y0);
                interim_x_y8_512 = _mm512_add_epi64(interim_x_y8_512, mul_x_y8);
                interim_x_y16_512 = _mm512_add_epi64(interim_x_y16_512, mul_x_y16);
                interim_x_y24_512 = _mm512_add_epi64(interim_x_y24_512, mul_x_y24);
            }
                _mm512_storeu_si512((__m512i*)(interim_1_x + j), interim_1_x_512);
                _mm512_storeu_si512((__m512i*)(interim_1_x + j + 16), interim_1_x16_512);

                _mm512_storeu_si512((__m512i*)(interim_2_x + j), interim_2_x0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 8), interim_2_x8_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 16), interim_2_x16_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 24), interim_2_x24_512);

                _mm512_storeu_si512((__m512i*)(interim_1_y + j), interim_1_y_512);
                _mm512_storeu_si512((__m512i*)(interim_1_y + j + 16), interim_1_y16_512);

                _mm512_storeu_si512((__m512i*)(interim_2_y + j), interim_2_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 8), interim_2_y8_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 16), interim_2_y16_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 24), interim_2_y24_512);

                _mm512_storeu_si512((__m512i*)(interim_x_y + j), interim_x_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 8), interim_x_y8_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 16), interim_x_y16_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 24), interim_x_y24_512);
        }

        for (; j < width_p1_16; j+=16)
        {
            int j_minus1 = j-1;

            interim_1_x_512 = interim_1_y_512 = \
            interim_2_x0_512 = interim_2_x8_512 = interim_2_y0_512 = interim_2_y8_512 = \
            interim_x_y0_512 = interim_x_y8_512 = _mm512_setzero_si512();

            for (i=1; i<kh+1; i++)
            {
                int src_offset = (i-1) * r_width;
                // peut etre unpack si permute apres
                __m512i src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1)));
                __m512i src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1)));

                // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
                interim_1_x_512 = _mm512_add_epi32(interim_1_x_512, src_x);
                interim_1_y_512 = _mm512_add_epi32(interim_1_y_512, src_y);

                __m512i src_x0 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x));
                __m512i src_x8 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x, 1));
                __m512i src_y0 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y));
                __m512i src_y8 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y, 1));

                // 0 2 4 6 8 10 12 14
                __m512i mul_x_x0 = _mm512_mul_epi32(src_x0, src_x0);
                // 1 3 5 7 9 11 13 15
                __m512i mul_x_x8 = _mm512_mul_epi32(src_x8, src_x8);

                __m512i mul_y_y0 = _mm512_mul_epi32(src_y0, src_y0); 
                __m512i mul_y_y8 = _mm512_mul_epi32(src_y8, src_y8);

                __m512i mul_x_y0 = _mm512_mul_epi32(src_x0, src_y0);
                __m512i mul_x_y8 = _mm512_mul_epi32(src_x8, src_y8);

                interim_2_x0_512 = _mm512_add_epi64(interim_2_x0_512, mul_x_x0);
                interim_2_x8_512 = _mm512_add_epi64(interim_2_x8_512, mul_x_x8);

                interim_2_y0_512 = _mm512_add_epi64(interim_2_y0_512, mul_y_y0); 
                interim_2_y8_512 = _mm512_add_epi64(interim_2_y8_512, mul_y_y8);

                interim_x_y0_512 = _mm512_add_epi64(interim_x_y0_512, mul_x_y0);
                interim_x_y8_512 = _mm512_add_epi64(interim_x_y8_512, mul_x_y8);
            }
                _mm512_storeu_si512((__m512i*)(interim_1_x + j), interim_1_x_512);

                _mm512_storeu_si512((__m512i*)(interim_2_x + j), interim_2_x0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 8), interim_2_x8_512);

                _mm512_storeu_si512((__m512i*)(interim_1_y + j), interim_1_y_512);

                _mm512_storeu_si512((__m512i*)(interim_2_y + j), interim_2_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 8), interim_2_y8_512);

                _mm512_storeu_si512((__m512i*)(interim_x_y + j), interim_x_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 8), interim_x_y8_512);
        }

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
            for (j = width_p1_16 + 1; j<width_p1; j++)
            {
                int j_minus1 = j-1;
                dwt2_dtype src_x_val = x_pad_t[src_offset + j_minus1];
                dwt2_dtype src_y_val = y_pad_t[src_offset + j_minus1];

                int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
                int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val;
                interim_x_y[j] = interim_x_y[j] + src_xy_val;
            }
        }
        /**
         * The vif score calculations would start from the kh,kw index of var & covar
         * Hence horizontal sum of first kh rows are not used, hence that computation is avoided
         */
        //score computation for 1st row of variance & covariance i.e. kh row of padded img

#if VIF_STABILITY
        vif_horz_integralsum_avx512(kw, width_p1, knorm_fact, knorm_shift, 
                             exp_t, sigma_nsq_t, log_18,
                             interim_1_x, interim_1_y,
                             interim_2_x, interim_2_y, interim_x_y,
                             &score_num_t, &num_power, &score_den_t, &den_power, shift_val, k_norm);
#else
        vif_horz_integralsum_avx512(kw, width_p1, knorm_fact, knorm_shift, 
                             exp_t, sigma_nsq_t, log_18,
                             interim_1_x, interim_1_y,
                             interim_2_x, interim_2_y, interim_x_y,
                             &score_num_t, &num_power, &score_den_t, &den_power);
#endif
        
        for(; i<height_p1; i++)
        {
            int src_offset = (i-1) * r_width;
            int pre_kh_src_offset = (i-1-kh) * r_width;
            for (int j = 1; j < width_p1_32; j += 32)
            {
                int j_minus1 = j-1;
                interim_1_x_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + j));
                interim_1_x16_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + j + 16));

                interim_2_x0_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j));
                interim_2_x8_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 8));
                interim_2_x16_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 16));
                interim_2_x24_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 24));

                interim_1_y_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + j));
                interim_1_y16_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + j + 16));

                interim_2_y0_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j));
                interim_2_y8_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 8));
                interim_2_y16_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 16));
                interim_2_y24_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 24));

                interim_x_y0_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j));
                interim_x_y8_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 8));
                interim_x_y16_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 16));
                interim_x_y24_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 24));

                __m512i src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1)));
                __m512i src_x16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1 + 16)));
                __m512i src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1)));
                __m512i src_y16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1 + 16)));

                __m512i prev_src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + pre_kh_src_offset + j_minus1)));
                __m512i prev_src_x16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + pre_kh_src_offset + j_minus1 + 16)));
                __m512i prev_src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + pre_kh_src_offset + j_minus1)));
                __m512i prev_src_y16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + pre_kh_src_offset + j_minus1 + 16)));

                __m512i src_x0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x));
                __m512i src_x8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x, 1));
                __m512i src_x16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x16));
                __m512i src_x24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x16, 1));

                __m512i prev_src_x0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_x));
                __m512i prev_src_x8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_x, 1));
                __m512i prev_src_x16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_x16));
                __m512i prev_src_x24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_x16, 1));

                __m512i src_y0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y));
                __m512i src_y8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y, 1));
                __m512i src_y16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y16));
                __m512i src_y24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y16, 1));

                __m512i prev_src_y0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_y));
                __m512i prev_src_y8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_y, 1));
                __m512i prev_src_y16_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_y16));
                __m512i prev_src_y24_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_y16, 1));

                __m512i x_m_px = _mm512_sub_epi32(src_x, prev_src_x);
                __m512i x16_m_px16 = _mm512_sub_epi32(src_x16, prev_src_x16);
                __m512i y_m_py = _mm512_sub_epi32(src_y, prev_src_y);
                __m512i y16_m_py16 = _mm512_sub_epi32(src_y16, prev_src_y16);

                __m512i mul_x_x0 = _mm512_mul_epi32(src_x0_64, src_x0_64);
                __m512i mul_x_x8 = _mm512_mul_epi32(src_x8_64, src_x8_64);
                __m512i mul_x_x16 = _mm512_mul_epi32(src_x16_64, src_x16_64);
                __m512i mul_x_x24 = _mm512_mul_epi32(src_x24_64, src_x24_64);

                __m512i mul_px_px0 = _mm512_mul_epi32(prev_src_x0_64, prev_src_x0_64);
                __m512i mul_px_px8 = _mm512_mul_epi32(prev_src_x8_64, prev_src_x8_64);
                __m512i mul_px_px16 = _mm512_mul_epi32(prev_src_x16_64, prev_src_x16_64);
                __m512i mul_px_px24 = _mm512_mul_epi32(prev_src_x24_64, prev_src_x24_64);

                __m512i mul_y_y0 = _mm512_mul_epi32(src_y0_64, src_y0_64);
                __m512i mul_y_y8 = _mm512_mul_epi32(src_y8_64, src_y8_64);
                __m512i mul_y_y16 = _mm512_mul_epi32(src_y16_64, src_y16_64);
                __m512i mul_y_y24 = _mm512_mul_epi32(src_y24_64, src_y24_64);

                __m512i mul_py_py0 = _mm512_mul_epi32(prev_src_y0_64, prev_src_y0_64);
                __m512i mul_py_py8 = _mm512_mul_epi32(prev_src_y8_64, prev_src_y8_64);
                __m512i mul_py_py16 = _mm512_mul_epi32(prev_src_y16_64, prev_src_y16_64);
                __m512i mul_py_py24 = _mm512_mul_epi32(prev_src_y24_64, prev_src_y24_64);

                __m512i mul_x_y0 = _mm512_mul_epi32(src_x0_64, src_y0_64);
                __m512i mul_x_y8 = _mm512_mul_epi32(src_x8_64, src_y8_64);
                __m512i mul_x_y16 = _mm512_mul_epi32(src_x16_64, src_y16_64);
                __m512i mul_x_y24 = _mm512_mul_epi32(src_x24_64, src_y24_64);

                __m512i mul_px_py0 = _mm512_mul_epi32(prev_src_x0_64, prev_src_y0_64);
                __m512i mul_px_py8 = _mm512_mul_epi32(prev_src_x8_64, prev_src_y8_64);
                __m512i mul_px_py16 = _mm512_mul_epi32(prev_src_x16_64, prev_src_y16_64);
                __m512i mul_px_py24 = _mm512_mul_epi32(prev_src_x24_64, prev_src_y24_64);

                interim_1_x_512 = _mm512_add_epi32(interim_1_x_512, x_m_px);
                interim_1_x16_512 = _mm512_add_epi32(interim_1_x16_512, x16_m_px16);
                interim_1_y_512 = _mm512_add_epi32(interim_1_y_512, y_m_py);
                interim_1_y16_512 = _mm512_add_epi32(interim_1_y16_512, y16_m_py16);

                __m512i xx_m_pxpx0 = _mm512_sub_epi64(mul_x_x0, mul_px_px0);
                __m512i xx_m_pxpx8 = _mm512_sub_epi64(mul_x_x8, mul_px_px8);
                __m512i xx_m_pxpx16 = _mm512_sub_epi64(mul_x_x16, mul_px_px16);
                __m512i xx_m_pxpx24 = _mm512_sub_epi64(mul_x_x24, mul_px_px24);

                __m512i yy_m_pypy0 = _mm512_sub_epi64(mul_y_y0, mul_py_py0);
                __m512i yy_m_pypy8 = _mm512_sub_epi64(mul_y_y8, mul_py_py8);
                __m512i yy_m_pypy16 = _mm512_sub_epi64(mul_y_y16, mul_py_py16);
                __m512i yy_m_pypy24 = _mm512_sub_epi64(mul_y_y24, mul_py_py24);

                __m512i xy_m_pxpy0 = _mm512_sub_epi64(mul_x_y0, mul_px_py0);
                __m512i xy_m_pxpy8 = _mm512_sub_epi64(mul_x_y8, mul_px_py8);
                __m512i xy_m_pxpy16 = _mm512_sub_epi64(mul_x_y16, mul_px_py16);
                __m512i xy_m_pxpy24 = _mm512_sub_epi64(mul_x_y24, mul_px_py24);

                interim_2_x0_512 = _mm512_add_epi64(interim_2_x0_512, xx_m_pxpx0);
                interim_2_x8_512 = _mm512_add_epi64(interim_2_x8_512, xx_m_pxpx8);
                interim_2_x16_512 = _mm512_add_epi64(interim_2_x16_512, xx_m_pxpx16);
                interim_2_x24_512 = _mm512_add_epi64(interim_2_x24_512, xx_m_pxpx24);

                interim_2_y0_512 = _mm512_add_epi64(interim_2_y0_512, yy_m_pypy0); 
                interim_2_y8_512 = _mm512_add_epi64(interim_2_y8_512, yy_m_pypy8);
                interim_2_y16_512 = _mm512_add_epi64(interim_2_y16_512, yy_m_pypy16);
                interim_2_y24_512 = _mm512_add_epi64(interim_2_y24_512, yy_m_pypy24);

                interim_x_y0_512 = _mm512_add_epi64(interim_x_y0_512, xy_m_pxpy0);
                interim_x_y8_512 = _mm512_add_epi64(interim_x_y8_512, xy_m_pxpy8);
                interim_x_y16_512 = _mm512_add_epi64(interim_x_y16_512, xy_m_pxpy16);
                interim_x_y24_512 = _mm512_add_epi64(interim_x_y24_512, xy_m_pxpy24);

                _mm512_storeu_si512((__m512i*)(interim_1_x + j), interim_1_x_512);
                _mm512_storeu_si512((__m512i*)(interim_1_x + j + 16), interim_1_x16_512);

                _mm512_storeu_si512((__m512i*)(interim_2_x + j), interim_2_x0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 8), interim_2_x8_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 16), interim_2_x16_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 24), interim_2_x24_512);

                _mm512_storeu_si512((__m512i*)(interim_1_y + j), interim_1_y_512);
                _mm512_storeu_si512((__m512i*)(interim_1_y + j + 16), interim_1_y16_512);

                _mm512_storeu_si512((__m512i*)(interim_2_y + j), interim_2_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 8), interim_2_y8_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 16), interim_2_y16_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 24), interim_2_y24_512);

                _mm512_storeu_si512((__m512i*)(interim_x_y + j), interim_x_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 8), interim_x_y8_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 16), interim_x_y16_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 24), interim_x_y24_512);

            }

            for (int j = width_p1_32 + 1; j < width_p1_16; j += 16)
            {
                int j_minus1 = j-1;
                interim_1_x_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + j));

                interim_2_x0_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j));
                interim_2_x8_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 8));

                interim_1_y_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + j));

                interim_2_y0_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j));
                interim_2_y8_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 8));

                interim_x_y0_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j));
                interim_x_y8_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 8));

                __m512i src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + src_offset + j_minus1)));
                __m512i src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + src_offset + j_minus1)));

                __m512i prev_src_x = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(x_pad_t + pre_kh_src_offset + j_minus1)));
                __m512i prev_src_y = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(y_pad_t + pre_kh_src_offset + j_minus1)));

                __m512i src_x0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_x));
                __m512i src_x8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_x, 1));

                __m512i prev_src_x0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_x));
                __m512i prev_src_x8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_x, 1));

                __m512i src_y0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src_y));
                __m512i src_y8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src_y, 1));
                
                __m512i prev_src_y0_64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prev_src_y));
                __m512i prev_src_y8_64 = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prev_src_y, 1));
                
                __m512i x_m_px = _mm512_sub_epi32(src_x, prev_src_x);
                __m512i y_m_py = _mm512_sub_epi32(src_y, prev_src_y);

                __m512i mul_x_x0 = _mm512_mul_epi32(src_x0_64, src_x0_64);
                __m512i mul_x_x8 = _mm512_mul_epi32(src_x8_64, src_x8_64);

                __m512i mul_px_px0 = _mm512_mul_epi32(prev_src_x0_64, prev_src_x0_64);
                __m512i mul_px_px8 = _mm512_mul_epi32(prev_src_x8_64, prev_src_x8_64);

                __m512i mul_y_y0 = _mm512_mul_epi32(src_y0_64, src_y0_64);
                __m512i mul_y_y8 = _mm512_mul_epi32(src_y8_64, src_y8_64);

                __m512i mul_py_py0 = _mm512_mul_epi32(prev_src_y0_64, prev_src_y0_64);
                __m512i mul_py_py8 = _mm512_mul_epi32(prev_src_y8_64, prev_src_y8_64);

                __m512i mul_x_y0 = _mm512_mul_epi32(src_x0_64, src_y0_64);
                __m512i mul_x_y8 = _mm512_mul_epi32(src_x8_64, src_y8_64);

                __m512i mul_px_py0 = _mm512_mul_epi32(prev_src_x0_64, prev_src_y0_64);
                __m512i mul_px_py8 = _mm512_mul_epi32(prev_src_x8_64, prev_src_y8_64);

                interim_1_x_512 = _mm512_add_epi32(interim_1_x_512, x_m_px);
                interim_1_y_512 = _mm512_add_epi32(interim_1_y_512, y_m_py);

                __m512i xx_m_pxpx0 = _mm512_sub_epi64(mul_x_x0, mul_px_px0);
                __m512i xx_m_pxpx8 = _mm512_sub_epi64(mul_x_x8, mul_px_px8);

                __m512i yy_m_pypy0 = _mm512_sub_epi64(mul_y_y0, mul_py_py0);
                __m512i yy_m_pypy8 = _mm512_sub_epi64(mul_y_y8, mul_py_py8);

                __m512i xy_m_pxpy0 = _mm512_sub_epi64(mul_x_y0, mul_px_py0);
                __m512i xy_m_pxpy8 = _mm512_sub_epi64(mul_x_y8, mul_px_py8);

                interim_2_x0_512 = _mm512_add_epi64(interim_2_x0_512, xx_m_pxpx0);
                interim_2_x8_512 = _mm512_add_epi64(interim_2_x8_512, xx_m_pxpx8);

                interim_2_y0_512 = _mm512_add_epi64(interim_2_y0_512, yy_m_pypy0); 
                interim_2_y8_512 = _mm512_add_epi64(interim_2_y8_512, yy_m_pypy8);

                interim_x_y0_512 = _mm512_add_epi64(interim_x_y0_512, xy_m_pxpy0);
                interim_x_y8_512 = _mm512_add_epi64(interim_x_y8_512, xy_m_pxpy8);

                _mm512_storeu_si512((__m512i*)(interim_1_x + j), interim_1_x_512);

                _mm512_storeu_si512((__m512i*)(interim_2_x + j), interim_2_x0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_x + j + 8), interim_2_x8_512);

                _mm512_storeu_si512((__m512i*)(interim_1_y + j), interim_1_y_512);

                _mm512_storeu_si512((__m512i*)(interim_2_y + j), interim_2_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_2_y + j + 8), interim_2_y8_512);

                _mm512_storeu_si512((__m512i*)(interim_x_y + j), interim_x_y0_512);
                _mm512_storeu_si512((__m512i*)(interim_x_y + j + 8), interim_x_y8_512);
            }

            for (int j = width_p1_16 + 1 ; j < width_p1; j++)
            {
                int j_minus1 = j-1;
                dwt2_dtype src_x_val = x_pad_t[src_offset + j_minus1];
                dwt2_dtype src_y_val = y_pad_t[src_offset + j_minus1];

                dwt2_dtype src_x_prekh_val = x_pad_t[pre_kh_src_offset + j_minus1];
                dwt2_dtype src_y_prekh_val = y_pad_t[pre_kh_src_offset + j_minus1];
                int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
                int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

                int32_t src_xx_prekh_val = (int32_t) src_x_prekh_val * src_x_prekh_val;
                int32_t src_yy_prekh_val = (int32_t) src_y_prekh_val * src_y_prekh_val;
                int32_t src_xy_prekh_val = (int32_t) src_x_prekh_val * src_y_prekh_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val - src_x_prekh_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val - src_xx_prekh_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val - src_y_prekh_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val - src_yy_prekh_val;
                interim_x_y[j] = interim_x_y[j] + src_xy_val - src_xy_prekh_val;
            }

            //horizontal summation and score compuations
#if VIF_STABILITY
            vif_horz_integralsum_avx512(kw, width_p1, knorm_fact, knorm_shift,  
                                 exp_t, sigma_nsq_t, log_18, 
                                 interim_1_x, interim_1_y,
                                 interim_2_x, interim_2_y, interim_x_y,
                                 &score_num_t, &num_power, 
                                 &score_den_t, &den_power, shift_val, k_norm);
#else
            vif_horz_integralsum_avx512(kw, width_p1, knorm_fact, knorm_shift,  
                                 exp_t, sigma_nsq_t, log_18, 
                                 interim_1_x, interim_1_y,
                                 interim_2_x, interim_2_y, interim_x_y,
                                 &score_num_t, &num_power, 
                                 &score_den_t, &den_power);
#endif
        }

        free(interim_2_x);
        free(interim_1_x);
        free(interim_2_y);
        free(interim_1_y);
        free(interim_x_y);
    }

    double power_double_num = (double)num_power;
    double power_double_den = (double)den_power;

#if VIF_STABILITY
	*score_num = (((double)score_num_t/(double)(1<<26)) + power_double_num);
    *score_den = (((double)score_den_t/(double)(1<<26)) + power_double_den);
	*score += ((*score_den) == 0.0) ? 1.0 : ((*score_num) / (*score_den));
#else
    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;
    double add_exp = 1e-4*s_height*s_width;

    *score_num = (((double)score_num_t/(double)(1<<26)) + power_double_num) + add_exp;
    *score_den = (((double)score_den_t/(double)(1<<26)) + power_double_den) + add_exp;
    *score = *score_num / *score_den;
#endif

#if VIF_REFLECT_PAD
    free(x_pad_t);
    free(y_pad_t);
#endif

    ret = 0;

    return ret;
}