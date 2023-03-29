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

#define Multiply64Bit_512(ab, cd, res){ \
    __m512i ac = _mm512_mul_epu32(ab, cd); \
    __m512i b = _mm512_srli_epi64(ab, 32); \
    __m512i bc = _mm512_mul_epu32(b, cd); \
    __m512i d = _mm512_srli_epi64(cd, 32); \
    __m512i ad = _mm512_mul_epu32(ab, d); \
    __m512i high = _mm512_add_epi64(bc, ad); \
    high = _mm512_slli_epi64(high, 32); \
    res = _mm512_add_epi64(high, ac); } 

#define shift15_64b_signExt_512(a, r)\
{ \
    r = _mm512_add_epi64( _mm512_srli_epi64(a, 6) , _mm512_and_si512(a, _mm512_set1_epi64(0xFC00000000000000)));\
}

#if VIF_STABILITY
static inline void vif_stats_calc_avx512(__m512i int_1_x_512, __m512i int_1_y_512, __m512i int_2_x0_512, __m512i int_2_x4_512,
                             __m512i int_2_y0_512, __m512i int_2_y4_512, __m512i int_x_y0_512, __m512i int_x_y4_512,
                             int16_t knorm_fact, int16_t knorm_shift,  
                             int16_t exp, int32_t sigma_nsq, uint32_t *log_18,
                             int64_t *score_num, int64_t *num_power,
                             int64_t *score_den, int64_t *den_power,int64_t shift_val, int k_norm)
#else
static inline void vif_stats_calc_avx512(__m512i int_1_x_512, __m512i int_1_y_512, __m512i int_2_x0_512, __m512i int_2_x4_512,
                             __m512i int_2_y0_512, __m512i int_2_y4_512, __m512i int_x_y0_512, __m512i int_x_y4_512,
                             int16_t knorm_fact, int16_t knorm_shift,  
                             int16_t exp, int32_t sigma_nsq, uint32_t *log_18,
                             int64_t *score_num, int64_t *num_power,
                             int64_t *score_den, int64_t *den_power)
#endif                          ²
{
    __m512i sigma_32b = _mm512_set1_epi32(sigma_nsq);
    __m512i sigma_512 = _mm512_set1_epi64(sigma_nsq);
    __m512i kf_512 = _mm512_set1_epi64(knorm_fact);
    __m512i exp_512 = _mm512_set1_epi64(exp);
    __m512i zero_512 = _mm512_setzero_si512();

    __m512i int_1_x4 = _mm512_srli_epi64(int_1_x_512, 32);
    __m512i int_1_y4 = _mm512_srli_epi64(int_1_y_512, 32);

    __m512i xx0_512, xx4_512, yy0_512, yy4_512, xy0_512, xy4_512;
    __m512i kxx0_512, kxx4_512, kyy0_512, kyy4_512, kxy0_512, kxy4_512;

    xx0_512 = _mm512_mul_epi32(int_1_x_512, int_1_x_512);
    xx4_512 = _mm512_mul_epi32(int_1_x4, int_1_x4);
    yy0_512 = _mm512_mul_epi32(int_1_y_512, int_1_y_512);
    yy4_512 = _mm512_mul_epi32(int_1_y4, int_1_y4);
    xy0_512 = _mm512_mul_epi32(int_1_x_512, int_1_y_512);
    xy4_512 = _mm512_mul_epi32(int_1_x4, int_1_y4);

    Multiply64Bit_512(xx0_512, kf_512, kxx0_512);
    Multiply64Bit_512(xx4_512, kf_512, kxx4_512);
    Multiply64Bit_512(yy0_512, kf_512, kyy0_512);
    Multiply64Bit_512(yy4_512, kf_512, kyy4_512);
    Multiply64Bit_512(xy0_512, kf_512, kxy0_512);
    Multiply64Bit_512(xy4_512, kf_512, kxy4_512);

    __mmask8 mask_neg_xx0 = _mm512_cmpgt_epi64_mask(zero_512, kxx0_512);
    __mmask8 mask_neg_xx4 = _mm512_cmpgt_epi64_mask(zero_512, kxx4_512);
    __mmask8 mask_neg_yy0 = _mm512_cmpgt_epi64_mask(zero_512, kyy0_512);
    __mmask8 mask_neg_yy4 = _mm512_cmpgt_epi64_mask(zero_512, kyy4_512);
    __mmask8 mask_neg_xy0 = _mm512_cmpgt_epi64_mask(zero_512, kxy0_512);
    __mmask8 mask_neg_xy4 = _mm512_cmpgt_epi64_mask(zero_512, kxy4_512);

    kxx0_512 = _mm512_srli_epi64(kxx0_512, knorm_shift);
    kxx4_512 = _mm512_srli_epi64(kxx4_512, knorm_shift);
    kyy0_512 = _mm512_srli_epi64(kyy0_512, knorm_shift);
    kyy4_512 = _mm512_srli_epi64(kyy4_512, knorm_shift);
    kxy0_512 = _mm512_srli_epi64(kxy0_512, knorm_shift);
    kxy4_512 = _mm512_srli_epi64(kxy4_512, knorm_shift);

    __m512i sign_extend_xx0 = _mm512_mask_blend_epi64(mask_neg_xx0, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));
    __m512i sign_extend_xx4 = _mm512_mask_blend_epi64(mask_neg_xx4, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));
    __m512i sign_extend_yy0 = _mm512_mask_blend_epi64(mask_neg_yy0, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));
    __m512i sign_extend_yy4 = _mm512_mask_blend_epi64(mask_neg_yy4, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));
    __m512i sign_extend_xy0 = _mm512_mask_blend_epi64(mask_neg_xy0, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));
    __m512i sign_extend_xy4 = _mm512_mask_blend_epi64(mask_neg_xy4, zero_512, _mm512_set1_epi64(0xFFFFF80000000000));

    kxx0_512 = _mm512_or_epi64(kxx0_512, sign_extend_xx0);
    kxx4_512 = _mm512_or_epi64(kxx4_512, sign_extend_xx4);
    kyy0_512 = _mm512_or_epi64(kyy0_512, sign_extend_yy0);
    kyy4_512 = _mm512_or_epi64(kyy4_512, sign_extend_yy4);
    kxy0_512 = _mm512_or_epi64(kxy0_512, sign_extend_xy0);
    kxy4_512 = _mm512_or_epi64(kxy4_512, sign_extend_xy4);

    kxx0_512 = _mm512_sub_epi64(int_2_x0_512, kxx0_512);
    kxx4_512 = _mm512_sub_epi64(int_2_x4_512, kxx4_512);
    kyy0_512 = _mm512_sub_epi64(int_2_y0_512, kyy0_512);
    kyy4_512 = _mm512_sub_epi64(int_2_y4_512, kyy4_512);
    kxy0_512 = _mm512_sub_epi64(int_x_y0_512, kxy0_512);
    kxy4_512 = _mm512_sub_epi64(int_x_y4_512, kxy4_512);

    __m512i var_x0_512, var_x4_512, var_y0_512, var_y4_512, cov_xy0_512, cov_xy4_512;

    shift15_64b_signExt_512(kxx0_512, var_x0_512);
    shift15_64b_signExt_512(kxx4_512, var_x4_512);
    shift15_64b_signExt_512(kyy0_512, var_y0_512);
    shift15_64b_signExt_512(kyy4_512, var_y4_512);
    shift15_64b_signExt_512(kxy0_512, cov_xy0_512);
    shift15_64b_signExt_512(kxy4_512, cov_xy4_512);

    __mmask8 mask_x0 = _mm512_cmpgt_epi64_mask(exp_512, var_x0_512);
    __mmask8 mask_x4 = _mm512_cmpgt_epi64_mask(exp_512, var_x4_512);
    __mmask8 mask_y0 = _mm512_cmpgt_epi64_mask(exp_512, var_y0_512);
    __mmask8 mask_y4 = _mm512_cmpgt_epi64_mask(exp_512, var_y4_512);

    __mmask8 mask_xy0 = _kor_mask8(mask_x0, mask_y0);
    __mmask8 mask_xy4 = _kor_mask8(mask_x4, mask_y4);

    var_x0_512 = _mm512_mask_blend_epi64(mask_x0, var_x0_512, zero_512);
    var_x4_512 = _mm512_mask_blend_epi64(mask_x4, var_x4_512, zero_512);
    var_y0_512 = _mm512_mask_blend_epi64(mask_y0, var_y0_512, zero_512);
    var_y4_512 = _mm512_mask_blend_epi64(mask_y4, var_y4_512, zero_512);
    
    cov_xy0_512 = _mm512_mask_blend_epi64(mask_xy0, cov_xy0_512, zero_512);
    cov_xy4_512 = _mm512_mask_blend_epi64(mask_xy4, cov_xy4_512, zero_512); 

    __m512i g_den0_512 = _mm512_add_epi64(var_x0_512, exp_512);
    __m512i g_den4_512 = _mm512_add_epi64(var_x4_512, exp_512);
    __m512i sv_sq_0 = _mm512_mul_epi32(cov_xy0_512, cov_xy0_512);
    __m512i sv_sq_4 = _mm512_mul_epi32(cov_xy4_512, cov_xy4_512);

    sv_sq_0[0] /= g_den0_512[0];
    sv_sq_0[1] /= g_den0_512[1];
    sv_sq_0[2] /= g_den0_512[2];
    sv_sq_0[3] /= g_den0_512[3];
    sv_sq_0[4] /= g_den0_512[4];
    sv_sq_0[5] /= g_den0_512[5];
    sv_sq_0[6] /= g_den0_512[6];
    sv_sq_0[7] /= g_den0_512[7];

    sv_sq_4[0] /= g_den4_512[0];
    sv_sq_4[1] /= g_den4_512[1];
    sv_sq_4[2] /= g_den4_512[2];
    sv_sq_4[3] /= g_den4_512[3];
    sv_sq_4[4] /= g_den4_512[4];
    sv_sq_4[5] /= g_den4_512[5];
    sv_sq_4[6] /= g_den4_512[6];
    sv_sq_4[7] /= g_den4_512[7];

    // 0 2 4 6 8 10 12 14 
    sv_sq_0 = _mm512_sub_epi64(var_y0_512, sv_sq_0);
    // 1 3 5 7 9 11 13 15
    sv_sq_4 = _mm512_sub_epi64(var_y4_512, sv_sq_4);

    // g_num < 0
    __mmask8 maskz_g_num0 = _mm512_cmpgt_epi64_mask(cov_xy0_512, _mm512_setzero_si512());
    __mmask8 maskz_g_num4 = _mm512_cmpgt_epi64_mask(cov_xy4_512, _mm512_setzero_si512());
    // g_den > 0
    __mmask8 maskz_g_den0 = _mm512_cmpgt_epi64_mask(_mm512_setzero_si512(), g_den0_512);
    __mmask8 maskz_g_den4 = _mm512_cmpgt_epi64_mask(_mm512_setzero_si512(), g_den4_512);

    // if((g_num < 0 && g_den > 0) || (g_den < 0 && g_num > 0))
    __mmask8 cond_0 = _kxnor_mask8(maskz_g_num0, maskz_g_den0);
    __mmask8 cond_4 = _kxnor_mask8(maskz_g_num4, maskz_g_den4);

    __m512i g_num_0 = _mm512_mask_blend_epi64(cond_0, cov_xy0_512, zero_512);
    __m512i g_num_4 = _mm512_mask_blend_epi64(cond_4, cov_xy4_512, zero_512);

    sv_sq_0 = _mm512_mask_blend_epi64(cond_0, sv_sq_0, var_x0_512);
    sv_sq_4 = _mm512_mask_blend_epi64(cond_4, sv_sq_4, var_x4_512);

    // if (sv_sq < exp)
    __mmask8 mask_sv0 = _mm512_cmpgt_epi64_mask(exp_512, sv_sq_0);
    __mmask8 mask_sv4 = _mm512_cmpgt_epi64_mask(exp_512, sv_sq_4);
    sv_sq_0 = _mm512_mask_blend_epi64(mask_sv0, sv_sq_0, exp_512);
    sv_sq_4 = _mm512_mask_blend_epi64(mask_sv4, sv_sq_4, exp_512);

    // ((int64_t)g_num * g_num)
    __m512i p1_0 = _mm512_mul_epi32(g_num_0, g_num_0);
    __m512i p1_4 = _mm512_mul_epi32(g_num_4, g_num_4);

    // ((int64_t)g_num * g_num)/g_den;
    p1_0[0] /= g_den0_512[0];
    p1_0[1] /= g_den0_512[1];
    p1_0[2] /= g_den0_512[2];
    p1_0[3] /= g_den0_512[3];
    p1_0[4] /= g_den0_512[4];
    p1_0[5] /= g_den0_512[5];
    p1_0[6] /= g_den0_512[6];
    p1_0[7] /= g_den0_512[7];

    p1_4[0] /= g_den4_512[0];
    p1_4[1] /= g_den4_512[1];
    p1_4[2] /= g_den4_512[2];
    p1_4[3] /= g_den4_512[3];
    p1_4[4] /= g_den4_512[4];
    p1_4[5] /= g_den4_512[5];
    p1_4[6] /= g_den4_512[6];
    p1_4[7] /= g_den4_512[7];

    // (p1 * p2)
    __m512i p1_mul_p2_0 = _mm512_mul_epi32(p1_0, var_x0_512);
    __m512i p1_mul_p2_4 = _mm512_mul_epi32(p1_4, var_x4_512);
    
    // ((int64_t) sv_sq + sigma_nsq)
    __m512i n2_0 = _mm512_add_epi64(sv_sq_0, sigma_512);
    __m512i n2_4 = _mm512_add_epi64(sv_sq_4, sigma_512);
    // g_den * ((int64_t) sv_sq + sigma_nsq)
    n2_0 = _mm512_mul_epi32(n2_0, g_den0_512);
    n2_4 = _mm512_mul_epi32(n2_4, g_den4_512);
    // n2 + (p1 * p2)
    __m512i n1_0 = _mm512_add_epi64(n2_0, p1_mul_p2_0);
    __m512i n1_4 = _mm512_add_epi64(n2_4, p1_mul_p2_4);

    __m512i log_in_num_1_0, log_in_num_1_4, log_in_num_2_0, log_in_num_2_4;
    __m512i x1_0, x1_4, x2_0, x2_4;

    x1_0 = _mm512_lzcnt_epi64(n1_0);
    x1_4 = _mm512_lzcnt_epi64(n1_4);
    x2_0 = _mm512_lzcnt_epi64(n2_0);
    x2_4 = _mm512_lzcnt_epi64(n2_4);

    x1_0 = _mm512_sub_epi64(_mm512_set1_epi64(46), x1_0);
    x1_4 = _mm512_sub_epi64(_mm512_set1_epi64(46), x1_4);
    x2_0 = _mm512_sub_epi64(_mm512_set1_epi64(46), x2_0);
    x2_4 = _mm512_sub_epi64(_mm512_set1_epi64(46), x2_4);

    x1_0 = _mm512_max_epi64(x1_0, _mm512_setzero_si512());
    x1_4 = _mm512_max_epi64(x1_4, _mm512_setzero_si512());
    x2_0 = _mm512_max_epi64(x2_0, _mm512_setzero_si512());
    x2_4 = _mm512_max_epi64(x2_4, _mm512_setzero_si512());

    log_in_num_1_0 = _mm512_srav_epi64(n1_0, x1_0);
    log_in_num_1_4 = _mm512_srav_epi64(n1_4, x1_4);
    log_in_num_2_0 = _mm512_srav_epi64(n2_0, x2_0);
    log_in_num_2_4 = _mm512_srav_epi64(n2_4, x2_4);

    __m512i log_18_1_0 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_num_1_0, log_18, 4));
    __m512i log_18_1_4 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_num_1_4, log_18, 4));
    __m512i log_18_2_0 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_num_2_0, log_18, 4));
    __m512i log_18_2_4 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_num_2_4, log_18, 4));

    __m512i log_18_0 = _mm512_sub_epi64(log_18_1_0, log_18_2_0);
    __m512i log_18_4 = _mm512_sub_epi64(log_18_1_4, log_18_2_4);

    __m512i x1 = _mm512_sub_epi64(x1_0, x2_0);
    __m512i x2 = _mm512_sub_epi64(x1_4, x2_4);

#if VIF_STABILITY
    __mmask8 cov_xy0_lt_z = _mm512_cmplt_epi64_mask(g_num_0, zero_512);
    __mmask8 cov_xy4_lt_z = _mm512_cmplt_epi64_mask(g_num_4, zero_512);

    log_18_0 = _mm512_mask_blend_epi32(cov_xy0_lt_z, log_18_0, zero_512);
    log_18_4 = _mm512_mask_blend_epi32(cov_xy4_lt_z, log_18_4, zero_512);
    x1 = _mm512_mask_blend_epi64(cov_xy0_lt_z, x1, zero_512);
    x2 = _mm512_mask_blend_epi64(cov_xy4_lt_z, x2, zero_512);
    
    __m512i shift_val_512 = _mm512_set1_epi32(shift_val);
    __m512i k_norm_512 = _mm512_set1_epi32(k_norm);
    __mmask8 mask_var_x0_lt_sigma = _mm512_cmplt_epi64_mask(var_x0_512, sigma_512);
    __mmask8 mask_var_x4_lt_sigma = _mm512_cmplt_epi64_mask(var_x4_512, sigma_512);
    
    // shift_val*shift_val
    __m512i shift_val_sq = _mm512_mullo_epi32(shift_val_512, shift_val_512);
    // (shift_val*shift_val*k_norm)
    __m512i shift_val_sq_mul_k_norm_512 = _mm512_mul_epi32(shift_val_sq, k_norm_512);
    // ((shift_val*shift_val*k_norm)>> VIF_COMPUTE_METRIC_R_SHIFT)
    shift_val_sq_mul_k_norm_512 = _mm512_srli_epi64(shift_val_sq_mul_k_norm_512, VIF_COMPUTE_METRIC_R_SHIFT);
    // ((int32_t)((var_y * sigma_max_inv)))
    __m512i var_y0_mul_4 = _mm512_slli_epi64(var_y0_512, 2);
    __m512i var_y4_mul_4 = _mm512_slli_epi64(var_y4_512, 2);
    // ((shift_val*shift_val*k_norm)>> VIF_COMPUTE_METRIC_R_SHIFT) - ((int32_t)((var_y * sigma_max_inv)))
    __m512i tmp_num_0 = _mm512_sub_epi64(shift_val_sq_mul_k_norm_512, var_y0_mul_4);
    __m512i tmp_num_4 = _mm512_sub_epi64(shift_val_sq_mul_k_norm_512, var_y4_mul_4);
    
    log_18_0 = _mm512_mask_blend_epi64(mask_var_x0_lt_sigma, log_18_0, tmp_num_0);
    log_18_4 = _mm512_mask_blend_epi64(mask_var_x4_lt_sigma, log_18_4, tmp_num_4);
    x1 = _mm512_mask_blend_epi64(mask_var_x0_lt_sigma, x1, zero_512);
    x2 = _mm512_mask_blend_epi64(mask_var_x4_lt_sigma, x2, zero_512);
#endif

    log_18_0 = _mm512_add_epi64(log_18_0, log_18_4);
    __m256i r4 = _mm256_add_epi64(_mm512_castsi512_si256(log_18_0), _mm512_extracti64x4_epi64(log_18_0, 1));
    __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(r4), _mm256_extracti64x2_epi64(r4, 1));
    int64_t temp_num = _mm_extract_epi64(r2, 0) + _mm_extract_epi64(r2, 1);

    __m512i temp_power_num_0 = _mm512_add_epi64(x1, x2);
    __m256i r4_x = _mm256_add_epi64(_mm512_castsi512_si256(temp_power_num_0), _mm512_extracti64x4_epi64(temp_power_num_0, 1));
    __m128i r2_x = _mm_add_epi64(_mm256_castsi256_si128(r4_x), _mm256_extracti64x2_epi64(r4_x, 1));
    int32_t temp_power_num = _mm_extract_epi64(r2_x, 0) + _mm_extract_epi64(r2_x, 1);
    *score_num += temp_num;
    *num_power += temp_power_num;

    __m512i d1_0 = _mm512_add_epi64(sigma_512, var_x0_512);
    __m512i d1_4 = _mm512_add_epi64(sigma_512, var_x4_512);
    __m512i d2 = sigma_512;

    __m512i log_in_den_1_0, log_in_den_1_4, log_in_den_2_0, log_in_den_2_4;
    __m512i y1_0, y1_4, y2_0;

    y1_0 = _mm512_lzcnt_epi64(d1_0);
    y1_4 = _mm512_lzcnt_epi64(d1_4);
    y2_0 = _mm512_lzcnt_epi64(sigma_512);

    y1_0 = _mm512_sub_epi64(_mm512_set1_epi64(46), y1_0);
    y1_4 = _mm512_sub_epi64(_mm512_set1_epi64(46), y1_4);
    y2_0 = _mm512_sub_epi64(_mm512_set1_epi64(46), y2_0);

    __m512i y1_0_pos = _mm512_max_epi64(y1_0, zero_512);
    __m512i y1_4_pos = _mm512_max_epi64(y1_4, zero_512);
    __m512i y2_0_pos = _mm512_max_epi64(y2_0, zero_512);

    __m512i y1_0_neg = _mm512_min_epi64(y1_0, zero_512);
    __m512i y1_4_neg = _mm512_min_epi64(y1_4, zero_512);
    __m512i y2_0_neg = _mm512_min_epi64(y2_0, zero_512);
    y1_0_neg = _mm512_abs_epi64(y1_0_neg);
    y1_4_neg = _mm512_abs_epi64(y1_4_neg);
    y2_0_neg = _mm512_abs_epi64(y2_0_neg);

    log_in_den_1_0 = _mm512_srav_epi64(d1_0, y1_0_pos);
    log_in_den_1_4 = _mm512_srav_epi64(d1_4, y1_4_pos);
    log_in_den_2_0 = _mm512_srav_epi64(sigma_512, y2_0_pos);

    log_in_den_1_0 = _mm512_sllv_epi64(log_in_den_1_0, y1_0_neg);
    log_in_den_1_4 = _mm512_sllv_epi64(log_in_den_1_4, y1_4_neg);
    log_in_den_2_0 = _mm512_sllv_epi64(log_in_den_2_0, y2_0_neg);

    log_18_1_0 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_den_1_0, log_18, 4));
    log_18_1_4 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_den_1_4, log_18, 4));
    log_18_2_0 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_den_2_0, log_18, 4));
    log_18_2_4 = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(log_in_den_2_0, log_18, 4));

    log_18_0 = _mm512_sub_epi64(log_18_1_0, log_18_2_0);
    log_18_4 = _mm512_sub_epi64(log_18_1_4, log_18_2_4);

    __m512i y1 = _mm512_sub_epi64(y1_0, y2_0);
    __m512i y2 = _mm512_sub_epi64(y1_4, y2_0);
#if VIF_STABILITY
    log_18_0 = _mm512_mask_blend_epi64(mask_var_x0_lt_sigma, log_18_0, shift_val_sq_mul_k_norm_512);
    log_18_4 = _mm512_mask_blend_epi64(mask_var_x4_lt_sigma, log_18_4, shift_val_sq_mul_k_norm_512);
    y1 = _mm512_mask_blend_epi64(mask_var_x0_lt_sigma, y1, zero_512);
    y2 = _mm512_mask_blend_epi64(mask_var_x4_lt_sigma, y2, zero_512);
#endif

    log_18_0 = _mm512_add_epi64(log_18_0, log_18_4);
    r4 = _mm256_add_epi64(_mm512_castsi512_si256(log_18_0), _mm512_extracti64x4_epi64(log_18_0, 1));
    r2 = _mm_add_epi64(_mm256_castsi256_si128(r4), _mm256_extracti64x2_epi64(r4, 1));
    int64_t temp_den = _mm_extract_epi64(r2, 0) + _mm_extract_epi64(r2, 1);

    __m512i temp_power_den_0 = _mm512_add_epi64(y1, y2);
    __m256i r4_y = _mm256_add_epi64(_mm512_castsi512_si256(temp_power_den_0), _mm512_extracti64x4_epi64(temp_power_den_0, 1));
    __m128i r2_y = _mm_add_epi64(_mm256_castsi256_si128(r4_y), _mm256_extracti64x2_epi64(r4_y, 1));
    int32_t temp_power_den = _mm_extract_epi64(r2_y, 0) + _mm_extract_epi64(r2_y, 1);

    *score_den += temp_den;
    *den_power += temp_power_den;
}

#if USE_DYNAMIC_SIGMA_NSQ
int integer_compute_vif_funque_avx512(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_18, int vif_level);
#else
int integer_compute_vif_funque_avx512(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                 double* score, double* score_num, double* score_den, 
                                 int k, int stride, double sigma_nsq_arg, 
                                 int64_t shift_val, uint32_t* log_18);
#endif

#if VIF_STABILITY
static inline void vif_horz_integralsum_avx512(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, 
                                   int16_t exp, int32_t sigma_nsq, uint32_t *log_18,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num, int64_t *num_power,
                                   int64_t *score_den, int64_t *den_power, int64_t shift_val, int k_norm)
#else
static inline void vif_horz_integralsum_avx512(int kw, int width_p1,
                                   int16_t knorm_fact, int16_t knorm_shift,
                                   int16_t exp, int32_t sigma_nsq, uint32_t *log_18,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num, int64_t *num_power,
                                   int64_t *score_den, int64_t *den_power)
#endif
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y, int_x_y;
    int width_p1_16 = (width_p1) - ((width_p1 - kw - 1) % 16);
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

    __m512i interim_1_x0_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + 1));
    __m512i interim_1_y0_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + 1));

    __m256i interim_1_x0_256 = _mm512_castsi512_si256(interim_1_x0_512);
    __m256i interim_1_y0_256 = _mm512_castsi512_si256(interim_1_y0_512);

    __m512i interim_2_x0_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + 1));
    __m512i interim_2_y0_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + 1));
    __m512i interim_x_y0_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + 1));
    __m512i interim_2_x8_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + 9));
    __m512i interim_2_y8_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + 9));
    __m512i interim_x_y8_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + 9));

    __m128i int_1_x_r4 = _mm_add_epi32(_mm256_castsi256_si128(interim_1_x0_256), _mm256_extracti128_si256(interim_1_x0_256, 1));
    __m128i int_1_y_r4 = _mm_add_epi32(_mm256_castsi256_si128(interim_1_y0_256), _mm256_extracti128_si256(interim_1_y0_256, 1));

    __m256i sum_x04 = _mm256_add_epi64(_mm512_castsi512_si256(interim_2_x0_512), _mm512_extracti64x4_epi64(interim_2_x0_512, 1));
    __m256i sum_y04 = _mm256_add_epi64(_mm512_castsi512_si256(interim_2_y0_512), _mm512_extracti64x4_epi64(interim_2_y0_512, 1));
    __m256i sum_xy04 = _mm256_add_epi64(_mm512_castsi512_si256(interim_x_y0_512), _mm512_extracti64x4_epi64(interim_x_y0_512, 1));

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
                    exp, sigma_nsq, log_18,
                    score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
    vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                    knorm_fact, knorm_shift,
                    exp, sigma_nsq, log_18,
                    score_num, num_power, score_den, den_power);
#endif

    __m512i interim_1_x9_512, interim_2_x9_512, interim_2_x17_512, interim_1_y9_512, \
    interim_2_y9_512, interim_2_y17_512, interim_x_y9_512, interim_x_y17_512;
    //Similar to prev loop, but previous kw col interim metric sum is subtracted
    int j;
    for (j = kw+1; j<width_p1_16; j+=16)
    {
        interim_1_x9_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + j));

        interim_2_x9_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j));
        interim_2_x17_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 8));

        interim_1_y9_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + j));

        interim_2_y9_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j));
        interim_2_y17_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 8));

        interim_x_y9_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j));
        interim_x_y17_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 8));

        __m512i interim_1x9_m_x0 = _mm512_sub_epi32(interim_1_x9_512, interim_1_x0_512);
        __m512i interim_1y9_m_y0 = _mm512_sub_epi32(interim_1_y9_512, interim_1_y0_512);

        __m512i interim_2x9_m_x0 = _mm512_sub_epi64(interim_2_x9_512, interim_2_x0_512);
        __m512i interim_2x17_m_x8 = _mm512_sub_epi64(interim_2_x17_512, interim_2_x8_512);

        __m512i interim_2y9_m_y0 = _mm512_sub_epi64(interim_2_y9_512, interim_2_y0_512);
        __m512i interim_2y17_m_y8 = _mm512_sub_epi64(interim_2_y17_512, interim_2_y8_512);

        __m512i interim_xy9_m_xy0 = _mm512_sub_epi64(interim_x_y9_512, interim_x_y0_512);
        __m512i interim_xy17_m_xy8 = _mm512_sub_epi64(interim_x_y17_512, interim_x_y8_512);

        __m256i interim_1x9_m_x0_lo = _mm512_castsi512_si256(interim_1x9_m_x0);
        __m256i interim_1x9_m_x0_hi = _mm512_extracti32x8_epi32(interim_1x9_m_x0, 1);
        
        __m256i interim_1y9_m_y0_lo = _mm512_castsi512_si256(interim_1y9_m_y0);
        __m256i interim_1y9_m_y0_hi = _mm512_extracti32x8_epi32(interim_1y9_m_y0, 1);

        __m256i interim_2x9_m_x0_lo = _mm512_castsi512_si256(interim_2x9_m_x0);
        __m256i interim_2x9_m_x0_hi = _mm512_extracti32x8_epi32(interim_2x9_m_x0, 1);
        __m256i interim_2x17_m_x8_lo = _mm512_castsi512_si256(interim_2x17_m_x8);
        __m256i interim_2x17_m_x8_hi = _mm512_extracti32x8_epi32(interim_2x17_m_x8, 1);

        __m256i interim_2y9_m_y0_lo = _mm512_castsi512_si256(interim_2y9_m_y0);
        __m256i interim_2y9_m_y0_hi = _mm512_extracti32x8_epi32(interim_2y9_m_y0, 1);
        __m256i interim_2y17_m_y8_lo = _mm512_castsi512_si256(interim_2y17_m_y8);
        __m256i interim_2y17_m_y8_hi = _mm512_extracti32x8_epi32(interim_2y17_m_y8, 1);

        __m256i interim_xy9_m_xy0_lo = _mm512_castsi512_si256(interim_xy9_m_xy0);
        __m256i interim_xy9_m_xy0_hi = _mm512_extracti32x8_epi32(interim_xy9_m_xy0, 1);
        __m256i interim_xy17_m_xy8_lo = _mm512_castsi512_si256(interim_xy17_m_xy8);
        __m256i interim_xy17_m_xy8_hi = _mm512_extracti32x8_epi32(interim_xy17_m_xy8, 1);

        int_1_x0 = int_1_x + _mm256_extract_epi32(interim_1x9_m_x0_lo, 0);
        int_1_y0 = int_1_y + _mm256_extract_epi32(interim_1y9_m_y0_lo, 0);
        int_2_x0 = int_2_x + _mm256_extract_epi64(interim_2x9_m_x0_lo, 0);
        int_2_y0 = int_2_y + _mm256_extract_epi64(interim_2y9_m_y0_lo, 0);
        int_x_y0 = int_x_y + _mm256_extract_epi64(interim_xy9_m_xy0_lo, 0);

        int int_1_x1 = int_1_x0 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 1);
        int int_1_y1 = int_1_y0 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 1);
        int64_t int_2_x1 = int_2_x0 + _mm256_extract_epi64(interim_2x9_m_x0_lo, 1);
        int64_t int_2_y1 = int_2_y0 + _mm256_extract_epi64(interim_2y9_m_y0_lo, 1);
        int64_t int_x_y1 = int_x_y0 + _mm256_extract_epi64(interim_xy9_m_xy0_lo, 1);

        int int_1_x2 = int_1_x1 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 2);
        int int_1_y2 = int_1_y1 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 2);
        int64_t int_2_x2 = int_2_x1 + _mm256_extract_epi64(interim_2x9_m_x0_lo, 2);
        int64_t int_2_y2 = int_2_y1 + _mm256_extract_epi64(interim_2y9_m_y0_lo, 2);
        int64_t int_x_y2 = int_x_y1 + _mm256_extract_epi64(interim_xy9_m_xy0_lo, 2);

        int int_1_x3 = int_1_x2 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 3);
        int int_1_y3 = int_1_y2 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 3);
        int64_t int_2_x3 = int_2_x2 + _mm256_extract_epi64(interim_2x9_m_x0_lo, 3);
        int64_t int_2_y3 = int_2_y2 + _mm256_extract_epi64(interim_2y9_m_y0_lo, 3);
        int64_t int_x_y3 = int_x_y2 + _mm256_extract_epi64(interim_xy9_m_xy0_lo, 3);

        int int_1_x4 = int_1_x3 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 4);
        int int_1_y4 = int_1_y3 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 4);
        int64_t int_2_x4 = int_2_x3 + _mm256_extract_epi64(interim_2x9_m_x0_hi, 0);
        int64_t int_2_y4 = int_2_y3 + _mm256_extract_epi64(interim_2y9_m_y0_hi, 0);
        int64_t int_x_y4 = int_x_y3 + _mm256_extract_epi64(interim_xy9_m_xy0_hi, 0);

        int int_1_x5 = int_1_x4 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 5);
        int int_1_y5 = int_1_y4 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 5);
        int64_t int_2_x5 = int_2_x4 + _mm256_extract_epi64(interim_2x9_m_x0_hi, 1);
        int64_t int_2_y5 = int_2_y4 + _mm256_extract_epi64(interim_2y9_m_y0_hi, 1);
        int64_t int_x_y5 = int_x_y4 + _mm256_extract_epi64(interim_xy9_m_xy0_hi, 1);

        int int_1_x6 = int_1_x5 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 6);
        int int_1_y6 = int_1_y5 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 6);
        int64_t int_2_x6 = int_2_x5 + _mm256_extract_epi64(interim_2x9_m_x0_hi, 2);
        int64_t int_2_y6 = int_2_y5 + _mm256_extract_epi64(interim_2y9_m_y0_hi, 2);
        int64_t int_x_y6 = int_x_y5 + _mm256_extract_epi64(interim_xy9_m_xy0_hi, 2);

        int int_1_x7 = int_1_x6 + _mm256_extract_epi32(interim_1x9_m_x0_lo, 7);
        int int_1_y7 = int_1_y6 + _mm256_extract_epi32(interim_1y9_m_y0_lo, 7);
        int64_t int_2_x7 = int_2_x6 + _mm256_extract_epi64(interim_2x9_m_x0_hi, 3);
        int64_t int_2_y7 = int_2_y6 + _mm256_extract_epi64(interim_2y9_m_y0_hi, 3);
        int64_t int_x_y7 = int_x_y6 + _mm256_extract_epi64(interim_xy9_m_xy0_hi, 3);

        int int_1_x8 = int_1_x7 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 0);
        int int_1_y8 = int_1_y7 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 0);
        int64_t int_2_x8 = int_2_x7 + _mm256_extract_epi64(interim_2x17_m_x8_lo, 0);
        int64_t int_2_y8 = int_2_y7 + _mm256_extract_epi64(interim_2y17_m_y8_lo, 0);
        int64_t int_x_y8 = int_x_y7 + _mm256_extract_epi64(interim_xy17_m_xy8_lo, 0);

        int int_1_x9 = int_1_x8 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 1);
        int int_1_y9 = int_1_y8 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 1);
        int64_t int_2_x9 = int_2_x8 + _mm256_extract_epi64(interim_2x17_m_x8_lo, 1);
        int64_t int_2_y9 = int_2_y8 + _mm256_extract_epi64(interim_2y17_m_y8_lo, 1);
        int64_t int_x_y9 = int_x_y8 + _mm256_extract_epi64(interim_xy17_m_xy8_lo, 1);

        int int_1_x10 = int_1_x9 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 2);
        int int_1_y10 = int_1_y9 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 2);
        int64_t int_2_x10 = int_2_x9 + _mm256_extract_epi64(interim_2x17_m_x8_lo, 2);
        int64_t int_2_y10 = int_2_y9 + _mm256_extract_epi64(interim_2y17_m_y8_lo, 2);
        int64_t int_x_y10 = int_x_y9 + _mm256_extract_epi64(interim_xy17_m_xy8_lo, 2);

        int int_1_x11 = int_1_x10 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 3);
        int int_1_y11 = int_1_y10 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 3);
        int64_t int_2_x11 = int_2_x10 + _mm256_extract_epi64(interim_2x17_m_x8_lo, 3);
        int64_t int_2_y11 = int_2_y10 + _mm256_extract_epi64(interim_2y17_m_y8_lo, 3);
        int64_t int_x_y11 = int_x_y10 + _mm256_extract_epi64(interim_xy17_m_xy8_lo, 3);

        int int_1_x12 = int_1_x11 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 4);
        int int_1_y12 = int_1_y11 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 4);
        int64_t int_2_x12 = int_2_x11 + _mm256_extract_epi64(interim_2x17_m_x8_hi, 0);
        int64_t int_2_y12 = int_2_y11 + _mm256_extract_epi64(interim_2y17_m_y8_hi, 0);
        int64_t int_x_y12 = int_x_y11 + _mm256_extract_epi64(interim_xy17_m_xy8_hi, 0);

        int int_1_x13 = int_1_x12 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 5);
        int int_1_y13 = int_1_y12 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 5);
        int64_t int_2_x13 = int_2_x12 + _mm256_extract_epi64(interim_2x17_m_x8_hi, 1);
        int64_t int_2_y13 = int_2_y12 + _mm256_extract_epi64(interim_2y17_m_y8_hi, 1);
        int64_t int_x_y13 = int_x_y12 + _mm256_extract_epi64(interim_xy17_m_xy8_hi, 1);

        int int_1_x14 = int_1_x13 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 6);
        int int_1_y14 = int_1_y13 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 6);
        int64_t int_2_x14 = int_2_x13 + _mm256_extract_epi64(interim_2x17_m_x8_hi, 2);
        int64_t int_2_y14 = int_2_y13 + _mm256_extract_epi64(interim_2y17_m_y8_hi, 2);
        int64_t int_x_y14 = int_x_y13 + _mm256_extract_epi64(interim_xy17_m_xy8_hi, 2);

        int int_1_x15 = int_1_x14 + _mm256_extract_epi32(interim_1x9_m_x0_hi, 7);
        int int_1_y15 = int_1_y14 + _mm256_extract_epi32(interim_1y9_m_y0_hi, 7);
        int64_t int_2_x15 = int_2_x14 + _mm256_extract_epi64(interim_2x17_m_x8_hi, 3);
        int64_t int_2_y15 = int_2_y14 + _mm256_extract_epi64(interim_2y17_m_y8_hi, 3);
        int64_t int_x_y15 = int_x_y14 + _mm256_extract_epi64(interim_xy17_m_xy8_hi, 3);

         __m512i int_1_x_512 = _mm512_set_epi32(int_1_x15, int_1_x14, int_1_x13, int_1_x12, int_1_x11, int_1_x10, int_1_x9, int_1_x8,
        int_1_x7, int_1_x6, int_1_x5, int_1_x4, int_1_x3, int_1_x2, int_1_x1, int_1_x0);
        __m512i int_1_y_512 = _mm512_set_epi32(int_1_y15, int_1_y14, int_1_y13, int_1_y12, int_1_y11, int_1_y10, int_1_y9, int_1_y8,
        int_1_y7, int_1_y6, int_1_y5, int_1_y4, int_1_y3, int_1_y2, int_1_y1, int_1_y0);
        __m512i int_2_x0_512 = _mm512_set_epi64(int_2_x14, int_2_x12, int_2_x10, int_2_x8, int_2_x6, int_2_x4, int_2_x2, int_2_x0);
        __m512i int_2_x1_512 = _mm512_set_epi64(int_2_x15, int_2_x13, int_2_x11, int_2_x9, int_2_x7, int_2_x5, int_2_x3, int_2_x1);
        __m512i int_2_y0_512 = _mm512_set_epi64(int_2_y14, int_2_y12, int_2_y10, int_2_y8, int_2_y6, int_2_y4, int_2_y2, int_2_y0);
        __m512i int_2_y1_512 = _mm512_set_epi64(int_2_y15, int_2_y13, int_2_y11, int_2_y9, int_2_y7, int_2_y5, int_2_y3, int_2_y1);
        __m512i int_x_y0_512 = _mm512_set_epi64(int_x_y14, int_x_y12, int_x_y10, int_x_y8, int_x_y6, int_x_y4, int_x_y2, int_x_y0);
        __m512i int_x_y1_512 = _mm512_set_epi64(int_x_y15, int_x_y13, int_x_y11, int_x_y9, int_x_y7, int_x_y5, int_x_y3, int_x_y1);

#if VIF_STABILITY
        vif_stats_calc_avx512( int_1_x_512, int_1_y_512, int_2_x0_512, int_2_x1_512, 
                        int_2_y0_512, int_2_y1_512, int_x_y0_512, int_x_y1_512,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_18,
                        score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
        vif_stats_calc_avx512( int_1_x_512, int_1_y_512, int_2_x0_512, int_2_x1_512, 
                        int_2_y0_512, int_2_y1_512, int_x_y0_512, int_x_y1_512,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_18,
                        score_num, num_power, score_den, den_power);

#endif
        interim_1_x0_512 = _mm512_loadu_si512((__m512i*)(interim_1_x + j + 7));
        interim_1_y0_512 = _mm512_loadu_si512((__m512i*)(interim_1_y + j + 7));
        interim_2_x0_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 7));
        interim_2_x8_512 = _mm512_loadu_si512((__m512i*)(interim_2_x + j + 15));
        interim_2_y0_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 7));
        interim_2_y8_512 = _mm512_loadu_si512((__m512i*)(interim_2_y + j + 15));
        interim_x_y0_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 7));
        interim_x_y8_512 = _mm512_loadu_si512((__m512i*)(interim_x_y + j + 15));

        int_1_x = int_1_x15;
        int_1_y = int_1_y15;
        int_2_x = int_2_x15;
        int_2_y = int_2_y15;
        int_x_y = int_x_y15;
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
                        exp, sigma_nsq, log_18,
                        score_num, num_power, score_den, den_power, shift_val, k_norm);
#else
        vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                        knorm_fact, knorm_shift,
                        exp, sigma_nsq, log_18,
                        score_num, num_power, score_den, den_power);
#endif
    }
}