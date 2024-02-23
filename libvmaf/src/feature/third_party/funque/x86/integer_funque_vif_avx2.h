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
#include <assert.h>

#include <immintrin.h>
#include "../integer_funque_vif.h"

#if USE_DYNAMIC_SIGMA_NSQ
int integer_compute_vif_funque_avx2(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_lut, int vif_level);
#else
int integer_compute_vif_funque_avx2(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_lut);
#endif

//This function does summation of horizontal intermediate_vertical_sums & then 
//numerator denominator score calculations are done
#if VIF_STABILITY
static inline void vif_horz_integralsum_avx2(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   int16_t exp, int32_t sigma_nsq, uint32_t *log_lut,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num, int64_t *num_power,
                                   int64_t *score_den, int64_t *den_power, int64_t shift_val, int k_norm)
#else
static inline void vif_horz_integralsum_avx2(int kw, int width_p1,
                                   int16_t knorm_fact, int16_t knorm_shift,
                                   int16_t exp, int32_t sigma_nsq, uint32_t *log_lut,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num, int64_t *num_power,
                                   int64_t *score_den, int64_t *den_power)
#endif
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y, int_x_y;
    int width_p1_8 = (width_p1) - ((width_p1 - kw - 1) % 8);
    //1st column vals are 0, hence intialising to 0
    int_1_x = 0;
    int_1_y = 0;
    int_2_x = 0;
    int_2_y = 0;
    int_x_y = 0;
    /**
     * The horizontal accumulation similar to vertical accumulation
     * metric_sum = prev_col_metric_sum + interim_metric_vertical_sum
     * The previous kw col interim metric sum is not subtracted since it is not available here
     */

    __m256i interim_1_x_256 = _mm256_loadu_si256((__m256i*)(interim_1_x + 1));

    __m256i interim_1_y_256 = _mm256_loadu_si256((__m256i*)(interim_1_y + 1));

    __m256i interim_2_x0_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + 1));
    __m256i interim_2_x4_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + 5));

    __m256i interim_2_y0_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + 1));
    __m256i interim_2_y4_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + 5));

    __m256i interim_x_y0_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + 1));
    __m256i interim_x_y4_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + 5));

    __m128i int_1_x_r4 = _mm_add_epi32(_mm256_castsi256_si128(interim_1_x_256), _mm256_extracti128_si256(interim_1_x_256, 1));
    __m128i int_1_y_r4 = _mm_add_epi32(_mm256_castsi256_si128(interim_1_y_256), _mm256_extracti128_si256(interim_1_y_256, 1));
    __m256i sum_x04 = _mm256_add_epi64(interim_2_x0_256, interim_2_x4_256);
    __m256i sum_y04 = _mm256_add_epi64(interim_2_y0_256, interim_2_y4_256);
    __m256i sum_xy04 = _mm256_add_epi64(interim_x_y0_256, interim_x_y4_256);

    __m128i int_1_x_r2 = _mm_hadd_epi32(int_1_x_r4, int_1_x_r4);
    __m128i int_1_y_r2 = _mm_hadd_epi32(int_1_y_r4, int_1_y_r4);
    __m128i int_2_x_r2 = _mm_add_epi64(_mm256_castsi256_si128(sum_x04), _mm256_extracti128_si256(sum_x04, 1));
    __m128i int_2_y_r2 = _mm_add_epi64(_mm256_castsi256_si128(sum_y04), _mm256_extracti128_si256(sum_y04, 1));
    __m128i int_x_y_r2 = _mm_add_epi64(_mm256_castsi256_si128(sum_xy04), _mm256_extracti128_si256(sum_xy04, 1));

    __m128i int_1_x_r1 = _mm_hadd_epi32(int_1_x_r2, int_1_x_r2);
    __m128i int_1_y_r1 = _mm_hadd_epi32(int_1_y_r2, int_1_y_r2);
    __m128i int_2_x_r1 = _mm_add_epi64(int_2_x_r2, _mm_unpackhi_epi64(int_2_x_r2, _mm_setzero_si128()));
    __m128i int_2_y_r1 = _mm_add_epi64(int_2_y_r2, _mm_unpackhi_epi64(int_2_y_r2, _mm_setzero_si128()));
    __m128i int_x_y_r1 = _mm_add_epi64(int_x_y_r2, _mm_unpackhi_epi64(int_x_y_r2, _mm_setzero_si128()));

    int32_t int_1_x0 = _mm_extract_epi32(int_1_x_r1, 0);
    int32_t int_1_y0 = _mm_extract_epi32(int_1_y_r1, 0);
    int64_t int_2_x0 = _mm_extract_epi64(int_2_x_r1, 0);
    int64_t int_2_y0 = _mm_extract_epi64(int_2_y_r1, 0);
    int64_t int_x_y0 = _mm_extract_epi64(int_x_y_r1, 0);

    int_1_x = interim_1_x[kw] + int_1_x0;
    int_1_y = interim_1_y[kw] + int_1_y0;
    int_2_x = interim_2_x[kw] + int_2_x0;
    int_2_y = interim_2_y[kw] + int_2_y0;
    int_x_y = interim_x_y[kw] + int_x_y0;  

    /**
     * The score needs to be calculated for kw column as well,
     * whose interim result calc is different from rest of the columns,
     * hence calling vif_stats_calc for kw column separately
     */

#if VIF_STABILITY
    vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                    knorm_fact, knorm_shift,
                    exp, sigma_nsq, log_lut,
                    score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
    vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                    knorm_fact, knorm_shift,
                    exp, sigma_nsq, log_lut,
                    score_num, num_power, score_den, den_power);
#endif

    __m256i interim_1_x9_256, interim_2_x9_256, interim_2_x13_256, interim_1_y9_256, \
    interim_2_y9_256, interim_2_y13_256, interim_x_y9_256, interim_x_y13_256;
    //Similar to prev loop, but previous kw col interim metric sum is subtracted
    int j;
    for (j = kw+1; j<width_p1_8; j+=8)
    {
        interim_1_x9_256 = _mm256_loadu_si256((__m256i*)(interim_1_x + j));

        interim_2_x9_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + j));
        interim_2_x13_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + j + 4));

        interim_1_y9_256 = _mm256_loadu_si256((__m256i*)(interim_1_y + j));

        interim_2_y9_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + j));
        interim_2_y13_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + j + 4));

        interim_x_y9_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + j));
        interim_x_y13_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + j + 4));

        __m256i interim_1x9_m_x0 = _mm256_sub_epi32(interim_1_x9_256, interim_1_x_256);
        __m256i interim_1y9_m_y0 = _mm256_sub_epi32(interim_1_y9_256, interim_1_y_256);

        __m256i interim_2x9_m_x0 = _mm256_sub_epi64(interim_2_x9_256, interim_2_x0_256);
        __m256i interim_2x13_m_x4 = _mm256_sub_epi64(interim_2_x13_256, interim_2_x4_256);

        __m256i interim_2y9_m_y0 = _mm256_sub_epi64(interim_2_y9_256, interim_2_y0_256);
        __m256i interim_2y13_m_y4 = _mm256_sub_epi64(interim_2_y13_256, interim_2_y4_256);

        __m256i interim_xy9_m_xy0 = _mm256_sub_epi64(interim_x_y9_256, interim_x_y0_256);
        __m256i interim_xy13_m_xy4 = _mm256_sub_epi64(interim_x_y13_256, interim_x_y4_256);

        int_1_x0 = int_1_x + _mm256_extract_epi32(interim_1x9_m_x0, 0);
        int_1_y0 = int_1_y + _mm256_extract_epi32(interim_1y9_m_y0, 0);
        int_2_x0 = int_2_x + _mm256_extract_epi64(interim_2x9_m_x0, 0);
        int_2_y0 = int_2_y + _mm256_extract_epi64(interim_2y9_m_y0, 0);
        int_x_y0 = int_x_y + _mm256_extract_epi64(interim_xy9_m_xy0, 0);

        int int_1_x1 = int_1_x0 + _mm256_extract_epi32(interim_1x9_m_x0, 1);
        int int_1_y1 = int_1_y0 + _mm256_extract_epi32(interim_1y9_m_y0, 1);
        int64_t int_2_x1 = int_2_x0 + _mm256_extract_epi64(interim_2x9_m_x0, 1);
        int64_t int_2_y1 = int_2_y0 + _mm256_extract_epi64(interim_2y9_m_y0, 1);
        int64_t int_x_y1 = int_x_y0 + _mm256_extract_epi64(interim_xy9_m_xy0, 1);

        int int_1_x2 = int_1_x1 + _mm256_extract_epi32(interim_1x9_m_x0, 2);
        int int_1_y2 = int_1_y1 + _mm256_extract_epi32(interim_1y9_m_y0, 2);
        int64_t int_2_x2 = int_2_x1 + _mm256_extract_epi64(interim_2x9_m_x0, 2);
        int64_t int_2_y2 = int_2_y1 + _mm256_extract_epi64(interim_2y9_m_y0, 2);
        int64_t int_x_y2 = int_x_y1 + _mm256_extract_epi64(interim_xy9_m_xy0, 2);

        int int_1_x3 = int_1_x2 + _mm256_extract_epi32(interim_1x9_m_x0, 3);
        int int_1_y3 = int_1_y2 + _mm256_extract_epi32(interim_1y9_m_y0, 3);
        int64_t int_2_x3 = int_2_x2 + _mm256_extract_epi64(interim_2x9_m_x0, 3);
        int64_t int_2_y3 = int_2_y2 + _mm256_extract_epi64(interim_2y9_m_y0, 3);
        int64_t int_x_y3 = int_x_y2 + _mm256_extract_epi64(interim_xy9_m_xy0, 3);

        int int_1_x4 = int_1_x3 + _mm256_extract_epi32(interim_1x9_m_x0, 4);
        int int_1_y4 = int_1_y3 + _mm256_extract_epi32(interim_1y9_m_y0, 4);
        int64_t int_2_x4 = int_2_x3 + _mm256_extract_epi64(interim_2x13_m_x4, 0);
        int64_t int_2_y4 = int_2_y3 + _mm256_extract_epi64(interim_2y13_m_y4, 0);
        int64_t int_x_y4 = int_x_y3 + _mm256_extract_epi64(interim_xy13_m_xy4, 0);

        int int_1_x5 = int_1_x4 + _mm256_extract_epi32(interim_1x9_m_x0, 5);
        int int_1_y5 = int_1_y4 + _mm256_extract_epi32(interim_1y9_m_y0, 5);
        int64_t int_2_x5 = int_2_x4 + _mm256_extract_epi64(interim_2x13_m_x4, 1);
        int64_t int_2_y5 = int_2_y4 + _mm256_extract_epi64(interim_2y13_m_y4, 1);
        int64_t int_x_y5 = int_x_y4 + _mm256_extract_epi64(interim_xy13_m_xy4, 1);

        int int_1_x6 = int_1_x5 + _mm256_extract_epi32(interim_1x9_m_x0, 6);
        int int_1_y6 = int_1_y5 + _mm256_extract_epi32(interim_1y9_m_y0, 6);
        int64_t int_2_x6 = int_2_x5 + _mm256_extract_epi64(interim_2x13_m_x4, 2);
        int64_t int_2_y6 = int_2_y5 + _mm256_extract_epi64(interim_2y13_m_y4, 2);
        int64_t int_x_y6 = int_x_y5 + _mm256_extract_epi64(interim_xy13_m_xy4, 2);

        int int_1_x7 = int_1_x6 + _mm256_extract_epi32(interim_1x9_m_x0, 7);
        int int_1_y7 = int_1_y6 + _mm256_extract_epi32(interim_1y9_m_y0, 7);
        int64_t int_2_x7 = int_2_x6 + _mm256_extract_epi64(interim_2x13_m_x4, 3);
        int64_t int_2_y7 = int_2_y6 + _mm256_extract_epi64(interim_2y13_m_y4, 3);
        int64_t int_x_y7 = int_x_y6 + _mm256_extract_epi64(interim_xy13_m_xy4, 3);

#if VIF_STABILITY
        vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
        vif_stats_calc(int_1_x0, int_1_y0, int_2_x0, int_2_y0, int_x_y0,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x1, int_1_y1, int_2_x1, int_2_y1, int_x_y1,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x2, int_1_y2, int_2_x2, int_2_y2, int_x_y2,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x3, int_1_y3, int_2_x3, int_2_y3, int_x_y3,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x4, int_1_y4, int_2_x4, int_2_y4, int_x_y4,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x5, int_1_y5, int_2_x5, int_2_y5, int_x_y5,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x6, int_1_y6, int_2_x6, int_2_y6, int_x_y6,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
        vif_stats_calc(int_1_x7, int_1_y7, int_2_x7, int_2_y7, int_x_y7,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
#endif

    interim_1_x_256 = _mm256_loadu_si256((__m256i*)(interim_1_x + j - 1));
    interim_1_y_256 = _mm256_loadu_si256((__m256i*)(interim_1_y + j - 1));
    interim_2_x0_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + j - 1));
    interim_2_x4_256 = _mm256_loadu_si256((__m256i*)(interim_2_x + j + 3));
    interim_2_y0_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + j - 1));
    interim_2_y4_256 = _mm256_loadu_si256((__m256i*)(interim_2_y + j + 3));
    interim_x_y0_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + j - 1));
    interim_x_y4_256 = _mm256_loadu_si256((__m256i*)(interim_x_y + j + 3));

    int_1_x = int_1_x7;
    int_1_y = int_1_y7;
    int_2_x = int_2_x7;
    int_2_y = int_2_y7;
    int_x_y = int_x_y7;
    }

    for (; j<width_p1; j++)
    {
        // int j_minus1 = j-1;
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];

        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];

        int_x_y = interim_x_y[j] + int_x_y - interim_x_y[j - kw];

#if VIF_STABILITY
        vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
        vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_lut,
                        score_num, num_power, score_den, den_power);
#endif
    }
}
