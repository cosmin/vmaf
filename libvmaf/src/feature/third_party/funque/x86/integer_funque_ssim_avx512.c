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
#include <stdlib.h>
#include <assert.h>
#include "../integer_funque_filters.h"
#include "../integer_funque_ssim.h"
#include "../funque_ssim_options.h"
#include "integer_funque_ssim_avx512.h"
#include <immintrin.h>

#define cvt_1_16x16_to_2_32x8_512(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a_16x16)); \
    r_32x8_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a_16x16, 1)); \
}
#define cvt_1_16x16_to_2_32x8_256(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_16x16)); \
    r_32x8_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a_16x16, 1)); \
}

#define cvt_1_32x8_to_2_64x4_512(a_32x8, r_64x4_lo, r_64x4_hi) \
{ \
    r_64x4_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a_32x8)); \
    r_64x4_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(a_32x8, 1)); \
}
#define cvt_1_32x8_to_2_64x4_256(a_32x8, r_64x4_lo, r_64x4_hi) \
{ \
    r_64x4_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a_32x8)); \
    r_64x4_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(a_32x8, 1)); \
}

#define cvt_1_32x4_to_2_64x2(a_32x8, r_64x4_lo, r_64x4_hi) \
{ \
    r_64x4_lo = _mm_cvtepi32_epi64(a_32x8); \
    r_64x4_hi = _mm_cvtepi32_epi64(_mm_shuffle_epi32(a_32x8, 0x0E)); \
}
#define cvt_1_16x8_to_2_32x4(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm_cvtepi16_epi32(a_16x16); \
    r_32x8_hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(a_16x16, 0x0E)); \
}

#define Multiply64Bit_512(ab, cd, res) \
{ \
    __m512i ac = _mm512_mul_epu32(ab, cd); \
    __m512i b = _mm512_srli_epi64(ab, 32); \
    __m512i bc = _mm512_mul_epu32(b, cd); \
    __m512i d = _mm512_srli_epi64(cd, 32); \
    __m512i ad = _mm512_mul_epu32(ab, d); \
    __m512i high = _mm512_add_epi64(bc, ad); \
    high = _mm512_slli_epi64(high, 32); \
    res = _mm512_add_epi64(high, ac); \
}

#define Multiply64Bit_256(ab, cd, res) \
{ \
    __m256i ac = _mm256_mul_epu32(ab, cd); \
    __m256i b = _mm256_srli_epi64(ab, 32); \
    __m256i bc = _mm256_mul_epu32(b, cd); \
    __m256i d = _mm256_srli_epi64(cd, 32); \
    __m256i ad = _mm256_mul_epu32(ab, d); \
    __m256i high = _mm256_add_epi64(bc, ad); \
    high = _mm256_slli_epi64(high, 32); \
    res = _mm256_add_epi64(high, ac); \
} 

#define Multiply64Bit_128(ab, cd, res) \
{ \
    __m128i ac = _mm_mul_epu32(ab, cd); \
    __m128i b = _mm_srli_epi64(ab, 32); \
    __m128i bc = _mm_mul_epu32(b, cd); \
    __m128i d = _mm_srli_epi64(cd, 32); \
    __m128i ad = _mm_mul_epu32(ab, d); \
    __m128i high = _mm_add_epi64(bc, ad); \
    high = _mm_slli_epi64(high, 32); \
    res = _mm_add_epi64(high, ac); \
} 

static inline int16_t get_best_i16_from_u64(uint64_t temp, int *power)
{
    assert(temp >= 0x20000);
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t) temp;
}

int integer_compute_ssim_funque_avx512(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;

    int width = ref->width;
    int height = ref->height;

    /**
     * C1 is constant is added to ref^2, dist^2, 
     *  - hence we have to multiply by pending_div^2
     * As per floating point,C1 is added to 2*(mx/win_dim)*(my/win_dim) & (mx/win_dim)*(mx/win_dim)+(my/win_dim)*(my/win_dim)
     * win_dim = 1 << n_levels, where n_levels = 1
     * Since win_dim division is avoided for mx & my, C1 is left shifted by 1
     */
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div*pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    /**
     * shifts are handled similar to C1
     * not shifted left because the other terms to which this is added undergoes equivalent right shift 
     */
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div*pending_div) >> (SSIM_INTER_VAR_SHIFTS+SSIM_INTER_CS_SHIFT-2)));

    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map;
    ssim_accum_dtype map_num;
    ssim_accum_dtype map_den;
    int16_t i16_map_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

#if ENABLE_MINK3POOL
    ssim_accum_dtype rowcube_1minus_map = 0;
    ssim_accum_dtype test = 0;
    double accumcube_1minus_map = 0;
    const ssim_inter_dtype const_1 = 32768;  //div_Q_factor>>SSIM_SHIFT_DIV
    
    __m512i const_1_512 = _mm512_set1_epi64(32768);
    __m512i accum_rowcube_512 = _mm512_setzero_si512();

    __m256i const_1_256 = _mm256_set1_epi64x(32768);
    __m256i accum_rowcube_256 = _mm256_setzero_si256();

    __m128i const_1_128 = _mm_set1_epi64x(32768);
    __m128i accum_rowcube_128 = _mm_setzero_si128();
#else
    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_map_sq = 0;
    ssim_accum_dtype map_sq_insum = 0;

    __m512i accum_map_512 = _mm512_setzero_si512();
    __m512i accum_map_sq_512 = _mm512_setzero_si512();

    __m256i accum_map_256 = _mm256_setzero_si256();
    __m256i accum_map_sq_256 = _mm256_setzero_si256();

    __m128i accum_map_128 = _mm_setzero_si128();
    __m128i accum_map_sq_128 = _mm_setzero_si128();

#endif
    __m512i C1_512 = _mm512_set1_epi32(C1);
    __m512i C2_512 = _mm512_set1_epi32(C2);

    __m256i C1_256 = _mm256_set1_epi32(C1);
    __m256i C2_256 = _mm256_set1_epi32(C2);

    __m128i C1_128 = _mm_set1_epi32(C1);
    __m128i C2_128 = _mm_set1_epi32(C2);

    int64_t *numVal = (int64_t *)malloc(width * sizeof(int64_t));
    int64_t *denVal = (int64_t *)malloc(width * sizeof(int64_t));

	int width_rem_size32 = width - (width % 32);
    int width_rem_size16 = width - (width % 16);
    int width_rem_size8 = width - (width % 8);
    int index = 0, j, k = 0;

    for (int i = 0; i < height; i++)
    {
        j = 0;
        for (; j < width_rem_size32; j+=32)
        {
            index = i * width + j;

            __m512i ref_b0 = _mm512_loadu_si512((__m512i*)(ref->bands[0] + index));
            __m512i dis_b0 = _mm512_loadu_si512((__m512i*)(dist->bands[0] + index));

            __m512i ref_b0_lo, ref_b0_hi, dis_b0_lo, dis_b0_hi;

            cvt_1_16x16_to_2_32x8_512(ref_b0, ref_b0_lo, ref_b0_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b0, dis_b0_lo, dis_b0_hi);

            __m512i var_x_b0_lo = _mm512_mullo_epi32(ref_b0_lo, ref_b0_lo);
            __m512i var_x_b0_hi = _mm512_mullo_epi32(ref_b0_hi, ref_b0_hi);
            __m512i var_y_b0_lo = _mm512_mullo_epi32(dis_b0_lo, dis_b0_lo);
            __m512i var_y_b0_hi = _mm512_mullo_epi32(dis_b0_hi, dis_b0_hi);
            __m512i cov_xy_b0_lo = _mm512_mullo_epi32(ref_b0_lo, dis_b0_lo);
            __m512i cov_xy_b0_hi = _mm512_mullo_epi32(ref_b0_hi, dis_b0_hi);

            __m512i ref_b1 = _mm512_loadu_si512((__m512i*)(ref->bands[1] + index));
            __m512i dis_b1 = _mm512_loadu_si512((__m512i*)(dist->bands[1] + index));
            __m512i ref_b2 = _mm512_loadu_si512((__m512i*)(ref->bands[2] + index));
            __m512i dis_b2 = _mm512_loadu_si512((__m512i*)(dist->bands[2] + index));
            __m512i ref_b3 = _mm512_loadu_si512((__m512i*)(ref->bands[3] + index));
            __m512i dis_b3 = _mm512_loadu_si512((__m512i*)(dist->bands[3] + index));

            __m512i ref_b1_lo, ref_b1_hi, dis_b1_lo, dis_b1_hi, \
            ref_b2_lo, ref_b2_hi, dis_b2_lo, dis_b2_hi, \
            ref_b3_lo, ref_b3_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x16_to_2_32x8_512(ref_b1, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b1, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b2, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b2, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b3, ref_b3_lo, ref_b3_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b3, dis_b3_lo, dis_b3_hi);

            __m512i var_x_b1_lo = _mm512_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m512i var_x_b1_hi = _mm512_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m512i var_y_b1_lo = _mm512_mullo_epi32(dis_b1_lo, dis_b1_lo);
            __m512i var_y_b1_hi = _mm512_mullo_epi32(dis_b1_hi, dis_b1_hi);

            __m512i cov_xy_b1_lo = _mm512_mullo_epi32(ref_b1_lo, dis_b1_lo);
            __m512i cov_xy_b1_hi = _mm512_mullo_epi32(ref_b1_hi, dis_b1_hi);

            __m512i var_x_b2_lo = _mm512_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m512i var_x_b2_hi = _mm512_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m512i var_y_b2_lo = _mm512_mullo_epi32(dis_b2_lo, dis_b2_lo);
            __m512i var_y_b2_hi = _mm512_mullo_epi32(dis_b2_hi, dis_b2_hi);
            __m512i cov_xy_b2_lo = _mm512_mullo_epi32(ref_b2_lo, dis_b2_lo);
            __m512i cov_xy_b2_hi = _mm512_mullo_epi32(ref_b2_hi, dis_b2_hi);

            __m512i var_x_b3_lo = _mm512_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m512i var_x_b3_hi = _mm512_mullo_epi32(ref_b3_hi, ref_b3_hi);
            __m512i var_y_b3_lo = _mm512_mullo_epi32(dis_b3_lo, dis_b3_lo);
            __m512i var_y_b3_hi = _mm512_mullo_epi32(dis_b3_hi, dis_b3_hi);
            __m512i cov_xy_b3_lo = _mm512_mullo_epi32(ref_b3_lo, dis_b3_lo);
            __m512i cov_xy_b3_hi = _mm512_mullo_epi32(ref_b3_hi, dis_b3_hi);

            __m512i var_x_lo = _mm512_add_epi32(var_x_b1_lo, var_x_b2_lo);
            __m512i var_x_hi = _mm512_add_epi32(var_x_b1_hi, var_x_b2_hi);
            __m512i var_y_lo = _mm512_add_epi32(var_y_b1_lo, var_y_b2_lo);
            __m512i var_y_hi = _mm512_add_epi32(var_y_b1_hi, var_y_b2_hi);
            __m512i cov_xy_lo = _mm512_add_epi32(cov_xy_b1_lo, cov_xy_b2_lo);
            __m512i cov_xy_hi = _mm512_add_epi32(cov_xy_b1_hi, cov_xy_b2_hi);
            var_x_lo = _mm512_add_epi32(var_x_lo, var_x_b3_lo);
            var_x_hi = _mm512_add_epi32(var_x_hi, var_x_b3_hi);
            var_y_lo = _mm512_add_epi32(var_y_lo, var_y_b3_lo);
            var_y_hi = _mm512_add_epi32(var_y_hi, var_y_b3_hi);
            cov_xy_lo = _mm512_add_epi32(cov_xy_lo, cov_xy_b3_lo);
            cov_xy_hi = _mm512_add_epi32(cov_xy_hi, cov_xy_b3_hi);
        
            __m512i l_den_lo = _mm512_add_epi32(var_x_b0_lo, var_y_b0_lo);
            __m512i l_den_hi = _mm512_add_epi32(var_x_b0_hi, var_y_b0_hi);

            var_x_lo = _mm512_srai_epi32(var_x_lo, SSIM_INTER_VAR_SHIFTS);
            var_x_hi = _mm512_srai_epi32(var_x_hi, SSIM_INTER_VAR_SHIFTS);
            var_y_lo = _mm512_srai_epi32(var_y_lo, SSIM_INTER_VAR_SHIFTS);
            var_y_hi = _mm512_srai_epi32(var_y_hi, SSIM_INTER_VAR_SHIFTS);
            cov_xy_lo = _mm512_srai_epi32(cov_xy_lo, SSIM_INTER_VAR_SHIFTS);
            cov_xy_hi = _mm512_srai_epi32(cov_xy_hi, SSIM_INTER_VAR_SHIFTS);

            l_den_lo = _mm512_srai_epi32(l_den_lo, SSIM_INTER_L_SHIFT);
            l_den_hi = _mm512_srai_epi32(l_den_hi, SSIM_INTER_L_SHIFT);

            __m512i l_num_lo = _mm512_add_epi32(cov_xy_b0_lo, C1_512);
            __m512i l_num_hi = _mm512_add_epi32(cov_xy_b0_hi, C1_512);

            __m512i cs_den_lo = _mm512_add_epi32(var_x_lo, var_y_lo);
            __m512i cs_den_hi = _mm512_add_epi32(var_x_hi, var_y_hi);
            __m512i cs_num_lo = _mm512_add_epi32(cov_xy_lo, C2_512);
            __m512i cs_num_hi = _mm512_add_epi32(cov_xy_hi, C2_512);

            cs_den_lo = _mm512_srai_epi32(cs_den_lo, SSIM_INTER_CS_SHIFT);
            cs_den_hi = _mm512_srai_epi32(cs_den_hi, SSIM_INTER_CS_SHIFT);
            
            l_den_lo = _mm512_add_epi32(l_den_lo, C1_512);
            l_den_hi = _mm512_add_epi32(l_den_hi, C1_512);

            cs_den_lo = _mm512_add_epi32(cs_den_lo, C2_512);
            cs_den_hi = _mm512_add_epi32(cs_den_hi, C2_512);

            __m512i map_num_lo0, map_num_lo1, map_num_hi0, map_num_hi1;
            __m512i map_den_lo0, map_den_lo1, map_den_hi0, map_den_hi1;

            map_num_lo0 = _mm512_mul_epi32(l_num_lo, cs_num_lo);
            map_num_lo1 = _mm512_mul_epi32(_mm512_srai_epi64(l_num_lo, 32), _mm512_srai_epi64(cs_num_lo, 32));

            map_num_hi0 = _mm512_mul_epi32(l_num_hi, cs_num_hi);
            map_num_hi1 = _mm512_mul_epi32(_mm512_srai_epi64(l_num_hi, 32), _mm512_srai_epi64(cs_num_hi, 32));

            map_den_lo0 = _mm512_mul_epi32(l_den_lo, cs_den_lo);
            map_den_lo1 = _mm512_mul_epi32(_mm512_srai_epi64(l_den_lo, 32), _mm512_srai_epi64(cs_den_lo, 32));

            map_den_hi0 = _mm512_mul_epi32(l_den_hi, cs_den_hi);
            map_den_hi1 = _mm512_mul_epi32(_mm512_srai_epi64(l_den_hi, 32), _mm512_srai_epi64(cs_den_hi, 32));

            __m512i zcnt_lo0 = _mm512_lzcnt_epi64(map_den_lo0);
            __m512i zcnt_lo1 = _mm512_lzcnt_epi64(map_den_lo1);
            __m512i zcnt_hi0 = _mm512_lzcnt_epi64(map_den_hi0);
            __m512i zcnt_hi1 = _mm512_lzcnt_epi64(map_den_hi1);

            zcnt_lo0 = _mm512_sub_epi64(_mm512_set1_epi64(49), zcnt_lo0);
            zcnt_lo1 = _mm512_sub_epi64(_mm512_set1_epi64(49), zcnt_lo1);
            zcnt_hi0 = _mm512_sub_epi64(_mm512_set1_epi64(49), zcnt_hi0);
            zcnt_hi1 = _mm512_sub_epi64(_mm512_set1_epi64(49), zcnt_hi1);

            map_den_lo0 = _mm512_srav_epi64(map_den_lo0, zcnt_lo0);
            map_num_lo0 = _mm512_srav_epi64(map_num_lo0, zcnt_lo0);
            map_den_lo1 = _mm512_srav_epi64(map_den_lo1, zcnt_lo1);
            map_num_lo1 = _mm512_srav_epi64(map_num_lo1, zcnt_lo1);
            map_den_hi0 = _mm512_srav_epi64(map_den_hi0, zcnt_hi0);
            map_num_hi0 = _mm512_srav_epi64(map_num_hi0, zcnt_hi0);
            map_den_hi1 = _mm512_srav_epi64(map_den_hi1, zcnt_hi1);
            map_num_hi1 = _mm512_srav_epi64(map_num_hi1, zcnt_hi1);

            map_den_lo0 = _mm512_add_epi64(map_den_lo0, _mm512_set1_epi64(32768));
            map_den_lo1 = _mm512_add_epi64(map_den_lo1, _mm512_set1_epi64(32768));
            map_den_hi0 = _mm512_add_epi64(map_den_hi0, _mm512_set1_epi64(32768));
            map_den_hi1 = _mm512_add_epi64(map_den_hi1, _mm512_set1_epi64(32768));

            __m256i div_lookup_lo0 = _mm512_i64gather_epi32(map_den_lo0, div_lookup, 4);
            __m256i div_lookup_lo1 = _mm512_i64gather_epi32(map_den_lo1, div_lookup, 4);
            __m256i div_lookup_hi0 = _mm512_i64gather_epi32(map_den_hi0, div_lookup, 4);
            __m256i div_lookup_hi1 = _mm512_i64gather_epi32(map_den_hi1, div_lookup, 4);
            __m512i map_lo0, map_lo1, map_hi0, map_hi1, map_sq_lo0, map_sq_lo1, map_sq_hi0, map_sq_hi1;

            Multiply64Bit_512(map_num_lo0, _mm512_cvtepi32_epi64(div_lookup_lo0), map_lo0);
            Multiply64Bit_512(map_num_lo1, _mm512_cvtepi32_epi64(div_lookup_lo1), map_lo1);
            Multiply64Bit_512(map_num_hi0, _mm512_cvtepi32_epi64(div_lookup_hi0), map_hi0);
            Multiply64Bit_512(map_num_hi1, _mm512_cvtepi32_epi64(div_lookup_hi1), map_hi1);

            map_lo0 = _mm512_srai_epi64(map_lo0, SSIM_SHIFT_DIV);
            map_lo1 = _mm512_srai_epi64(map_lo1, SSIM_SHIFT_DIV);
            map_hi0 = _mm512_srai_epi64(map_hi0, SSIM_SHIFT_DIV);
            map_hi1 = _mm512_srai_epi64(map_hi1, SSIM_SHIFT_DIV);

#if ENABLE_MINK3POOL
            __m512i const1_minus_map_lo0 = _mm512_sub_epi64(const_1_512, map_lo0);
            __m512i const1_minus_map_lo1 = _mm512_sub_epi64(const_1_512, map_lo1);
            __m512i const1_minus_map_hi0 = _mm512_sub_epi64(const_1_512, map_hi0);
            __m512i const1_minus_map_hi1 = _mm512_sub_epi64(const_1_512, map_hi1);

            __m512i const1_minus_map_sq_lo0 = _mm512_mul_epi32(const1_minus_map_lo0, const1_minus_map_lo0);
            __m512i const1_minus_map_sq_lo1 = _mm512_mul_epi32(const1_minus_map_lo1, const1_minus_map_lo1);
            __m512i const1_minus_map_sq_hi0 = _mm512_mul_epi32(const1_minus_map_hi0, const1_minus_map_hi0);
            __m512i const1_minus_map_sq_hi1 = _mm512_mul_epi32(const1_minus_map_hi1, const1_minus_map_hi1);

            __m512i rowcube_1minus_map_lo0, rowcube_1minus_map_lo1, rowcube_1minus_map_hi0, rowcube_1minus_map_hi1;

            Multiply64Bit_512(const1_minus_map_sq_lo0, const1_minus_map_lo0, rowcube_1minus_map_lo0);
            Multiply64Bit_512(const1_minus_map_sq_lo1, const1_minus_map_lo1, rowcube_1minus_map_lo1);
            Multiply64Bit_512(const1_minus_map_sq_hi0, const1_minus_map_hi0, rowcube_1minus_map_hi0);
            Multiply64Bit_512(const1_minus_map_sq_hi1, const1_minus_map_hi1, rowcube_1minus_map_hi1);

            rowcube_1minus_map_lo0 = _mm512_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_lo1);
            rowcube_1minus_map_hi0 = _mm512_add_epi64(rowcube_1minus_map_hi0, rowcube_1minus_map_hi1);
            rowcube_1minus_map_lo0 = _mm512_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_hi0);
            accum_rowcube_512 = _mm512_add_epi64(accum_rowcube_512, rowcube_1minus_map_lo0);
#else
            Multiply64Bit_512(map_lo0, map_lo0, map_sq_lo0);
            Multiply64Bit_512(map_lo1, map_lo1, map_sq_lo1);
            Multiply64Bit_512(map_hi0, map_hi0, map_sq_hi0);
            Multiply64Bit_512(map_hi1, map_hi1, map_sq_hi1);

            map_lo0 = _mm512_add_epi64(map_lo0, map_lo1);
            map_hi0 = _mm512_add_epi64(map_hi0, map_hi1);
            map_lo0 = _mm512_add_epi64(map_lo0, map_hi0);
            accum_map_512 = _mm512_add_epi64(accum_map_512, map_lo0);

            map_sq_lo0 = _mm512_add_epi64(map_sq_lo0, map_sq_lo1);
            map_sq_hi0 = _mm512_add_epi64(map_sq_hi0, map_sq_hi1);
            map_sq_lo0 = _mm512_add_epi64(map_sq_lo0, map_sq_hi0);
            accum_map_sq_512 = _mm512_add_epi64(accum_map_sq_512, map_sq_lo0);
#endif
        }

        for (; j < width_rem_size16; j+=16)
        {
            index = i * width + j;

            __m256i ref_b0 = _mm256_loadu_si256((__m256i*)(ref->bands[0] + index));
            __m256i dis_b0 = _mm256_loadu_si256((__m256i*)(dist->bands[0] + index));
            
            __m256i ref_b0_lo, ref_b0_hi, dis_b0_lo, dis_b0_hi;
            
            cvt_1_16x16_to_2_32x8_256(ref_b0, ref_b0_lo, ref_b0_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b0, dis_b0_lo, dis_b0_hi);

            __m256i var_x_b0_lo = _mm256_mullo_epi32(ref_b0_lo, ref_b0_lo);
            __m256i var_x_b0_hi = _mm256_mullo_epi32(ref_b0_hi, ref_b0_hi);
            __m256i var_y_b0_lo = _mm256_mullo_epi32(dis_b0_lo, dis_b0_lo);
            __m256i var_y_b0_hi = _mm256_mullo_epi32(dis_b0_hi, dis_b0_hi);
            __m256i cov_xy_b0_lo = _mm256_mullo_epi32(ref_b0_lo, dis_b0_lo);
            __m256i cov_xy_b0_hi = _mm256_mullo_epi32(ref_b0_hi, dis_b0_hi);

            __m256i ref_b1 = _mm256_loadu_si256((__m256i*)(ref->bands[1] + index));
            __m256i dis_b1 = _mm256_loadu_si256((__m256i*)(dist->bands[1] + index));
            __m256i ref_b2 = _mm256_loadu_si256((__m256i*)(ref->bands[2] + index));
            __m256i dis_b2 = _mm256_loadu_si256((__m256i*)(dist->bands[2] + index));
            __m256i ref_b3 = _mm256_loadu_si256((__m256i*)(ref->bands[3] + index));
            __m256i dis_b3 = _mm256_loadu_si256((__m256i*)(dist->bands[3] + index));

            __m256i ref_b1_lo, ref_b1_hi, dis_b1_lo, dis_b1_hi, \
            ref_b2_lo, ref_b2_hi, dis_b2_lo, dis_b2_hi, \
            ref_b3_lo, ref_b3_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x16_to_2_32x8_256(ref_b1, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b1, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8_256(ref_b2, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b2, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8_256(ref_b3, ref_b3_lo, ref_b3_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b3, dis_b3_lo, dis_b3_hi);

            __m256i var_x_b1_lo = _mm256_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m256i var_x_b1_hi = _mm256_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m256i var_y_b1_lo = _mm256_mullo_epi32(dis_b1_lo, dis_b1_lo);
            __m256i var_y_b1_hi = _mm256_mullo_epi32(dis_b1_hi, dis_b1_hi);

            __m256i cov_xy_b1_lo = _mm256_mullo_epi32(ref_b1_lo, dis_b1_lo);
            __m256i cov_xy_b1_hi = _mm256_mullo_epi32(ref_b1_hi, dis_b1_hi);

            __m256i var_x_b2_lo = _mm256_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m256i var_x_b2_hi = _mm256_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m256i var_y_b2_lo = _mm256_mullo_epi32(dis_b2_lo, dis_b2_lo);
            __m256i var_y_b2_hi = _mm256_mullo_epi32(dis_b2_hi, dis_b2_hi);
            __m256i cov_xy_b2_lo = _mm256_mullo_epi32(ref_b2_lo, dis_b2_lo);
            __m256i cov_xy_b2_hi = _mm256_mullo_epi32(ref_b2_hi, dis_b2_hi);

            __m256i var_x_b3_lo = _mm256_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m256i var_x_b3_hi = _mm256_mullo_epi32(ref_b3_hi, ref_b3_hi);
            __m256i var_y_b3_lo = _mm256_mullo_epi32(dis_b3_lo, dis_b3_lo);
            __m256i var_y_b3_hi = _mm256_mullo_epi32(dis_b3_hi, dis_b3_hi);
            __m256i cov_xy_b3_lo = _mm256_mullo_epi32(ref_b3_lo, dis_b3_lo);
            __m256i cov_xy_b3_hi = _mm256_mullo_epi32(ref_b3_hi, dis_b3_hi);

            __m256i var_x_lo = _mm256_add_epi32(var_x_b1_lo, var_x_b2_lo);
            __m256i var_x_hi = _mm256_add_epi32(var_x_b1_hi, var_x_b2_hi);
            __m256i var_y_lo = _mm256_add_epi32(var_y_b1_lo, var_y_b2_lo);
            __m256i var_y_hi = _mm256_add_epi32(var_y_b1_hi, var_y_b2_hi);
            __m256i cov_xy_lo = _mm256_add_epi32(cov_xy_b1_lo, cov_xy_b2_lo);
            __m256i cov_xy_hi = _mm256_add_epi32(cov_xy_b1_hi, cov_xy_b2_hi);
            var_x_lo = _mm256_add_epi32(var_x_lo, var_x_b3_lo);
            var_x_hi = _mm256_add_epi32(var_x_hi, var_x_b3_hi);
            var_y_lo = _mm256_add_epi32(var_y_lo, var_y_b3_lo);
            var_y_hi = _mm256_add_epi32(var_y_hi, var_y_b3_hi);
            cov_xy_lo = _mm256_add_epi32(cov_xy_lo, cov_xy_b3_lo);
            cov_xy_hi = _mm256_add_epi32(cov_xy_hi, cov_xy_b3_hi);
        
            __m256i l_den_lo = _mm256_add_epi32(var_x_b0_lo, var_y_b0_lo);
            __m256i l_den_hi = _mm256_add_epi32(var_x_b0_hi, var_y_b0_hi);

            var_x_lo = _mm256_srai_epi32(var_x_lo, SSIM_INTER_VAR_SHIFTS);
            var_x_hi = _mm256_srai_epi32(var_x_hi, SSIM_INTER_VAR_SHIFTS);
            var_y_lo = _mm256_srai_epi32(var_y_lo, SSIM_INTER_VAR_SHIFTS);
            var_y_hi = _mm256_srai_epi32(var_y_hi, SSIM_INTER_VAR_SHIFTS);
            cov_xy_lo = _mm256_srai_epi32(cov_xy_lo, SSIM_INTER_VAR_SHIFTS);
            cov_xy_hi = _mm256_srai_epi32(cov_xy_hi, SSIM_INTER_VAR_SHIFTS);

            l_den_lo = _mm256_srai_epi32(l_den_lo, SSIM_INTER_L_SHIFT);
            l_den_hi = _mm256_srai_epi32(l_den_hi, SSIM_INTER_L_SHIFT);

            __m256i l_num_lo = _mm256_add_epi32(cov_xy_b0_lo, C1_256);
            __m256i l_num_hi = _mm256_add_epi32(cov_xy_b0_hi, C1_256);

            __m256i cs_den_lo = _mm256_add_epi32(var_x_lo, var_y_lo);
            __m256i cs_den_hi = _mm256_add_epi32(var_x_hi, var_y_hi);
            __m256i cs_num_lo = _mm256_add_epi32(cov_xy_lo, C2_256);
            __m256i cs_num_hi = _mm256_add_epi32(cov_xy_hi, C2_256);

            cs_den_lo = _mm256_srai_epi32(cs_den_lo, SSIM_INTER_CS_SHIFT);
            cs_den_hi = _mm256_srai_epi32(cs_den_hi, SSIM_INTER_CS_SHIFT);
            
            l_den_lo = _mm256_add_epi32(l_den_lo, C1_256);
            l_den_hi = _mm256_add_epi32(l_den_hi, C1_256);

            cs_den_lo = _mm256_add_epi32(cs_den_lo, C2_256);
            cs_den_hi = _mm256_add_epi32(cs_den_hi, C2_256);

            __m256i map_num_lo0, map_num_lo1, map_num_hi0, map_num_hi1;
            __m256i map_den_lo0, map_den_lo1, map_den_hi0, map_den_hi1;

            map_num_lo0 = _mm256_mul_epi32(l_num_lo, cs_num_lo);
            map_num_lo1 = _mm256_mul_epi32(_mm256_srai_epi64(l_num_lo, 32), _mm256_srai_epi64(cs_num_lo, 32));

            map_num_hi0 = _mm256_mul_epi32(l_num_hi, cs_num_hi);
            map_num_hi1 = _mm256_mul_epi32(_mm256_srai_epi64(l_num_hi, 32), _mm256_srai_epi64(cs_num_hi, 32));

            map_den_lo0 = _mm256_mul_epi32(l_den_lo, cs_den_lo);
            map_den_lo1 = _mm256_mul_epi32(_mm256_srai_epi64(l_den_lo, 32), _mm256_srai_epi64(cs_den_lo, 32));

            map_den_hi0 = _mm256_mul_epi32(l_den_hi, cs_den_hi);
            map_den_hi1 = _mm256_mul_epi32(_mm256_srai_epi64(l_den_hi, 32), _mm256_srai_epi64(cs_den_hi, 32));

            __m256i zcnt_lo0 = _mm256_lzcnt_epi64(map_den_lo0);
            __m256i zcnt_lo1 = _mm256_lzcnt_epi64(map_den_lo1);
            __m256i zcnt_hi0 = _mm256_lzcnt_epi64(map_den_hi0);
            __m256i zcnt_hi1 = _mm256_lzcnt_epi64(map_den_hi1);

            zcnt_lo0 = _mm256_sub_epi64(_mm256_set1_epi64x(49), zcnt_lo0);
            zcnt_lo1 = _mm256_sub_epi64(_mm256_set1_epi64x(49), zcnt_lo1);
            zcnt_hi0 = _mm256_sub_epi64(_mm256_set1_epi64x(49), zcnt_hi0);
            zcnt_hi1 = _mm256_sub_epi64(_mm256_set1_epi64x(49), zcnt_hi1);

            map_den_lo0 = _mm256_srav_epi64(map_den_lo0, zcnt_lo0);
            map_num_lo0 = _mm256_srav_epi64(map_num_lo0, zcnt_lo0);
            map_den_lo1 = _mm256_srav_epi64(map_den_lo1, zcnt_lo1);
            map_num_lo1 = _mm256_srav_epi64(map_num_lo1, zcnt_lo1);
            map_den_hi0 = _mm256_srav_epi64(map_den_hi0, zcnt_hi0);
            map_num_hi0 = _mm256_srav_epi64(map_num_hi0, zcnt_hi0);
            map_den_hi1 = _mm256_srav_epi64(map_den_hi1, zcnt_hi1);
            map_num_hi1 = _mm256_srav_epi64(map_num_hi1, zcnt_hi1);

            map_den_lo0 = _mm256_add_epi64(map_den_lo0, _mm256_set1_epi64x(32768));
            map_den_lo1 = _mm256_add_epi64(map_den_lo1, _mm256_set1_epi64x(32768));
            map_den_hi0 = _mm256_add_epi64(map_den_hi0, _mm256_set1_epi64x(32768));
            map_den_hi1 = _mm256_add_epi64(map_den_hi1, _mm256_set1_epi64x(32768));

            __m128i div_lookup_lo0 = _mm256_i64gather_epi32(div_lookup, map_den_lo0, 4);
            __m128i div_lookup_lo1 = _mm256_i64gather_epi32(div_lookup, map_den_lo1, 4);
            __m128i div_lookup_hi0 = _mm256_i64gather_epi32(div_lookup, map_den_hi0, 4);
            __m128i div_lookup_hi1 = _mm256_i64gather_epi32(div_lookup, map_den_hi1, 4);
            __m256i map_lo0, map_lo1, map_hi0, map_hi1, map_sq_lo0, map_sq_lo1, map_sq_hi0, map_sq_hi1;

            Multiply64Bit_256(map_num_lo0, _mm256_cvtepi32_epi64(div_lookup_lo0), map_lo0);
            Multiply64Bit_256(map_num_lo1, _mm256_cvtepi32_epi64(div_lookup_lo1), map_lo1);
            Multiply64Bit_256(map_num_hi0, _mm256_cvtepi32_epi64(div_lookup_hi0), map_hi0);
            Multiply64Bit_256(map_num_hi1, _mm256_cvtepi32_epi64(div_lookup_hi1), map_hi1);

            map_lo0 = _mm256_srai_epi64(map_lo0, SSIM_SHIFT_DIV);
            map_lo1 = _mm256_srai_epi64(map_lo1, SSIM_SHIFT_DIV);
            map_hi0 = _mm256_srai_epi64(map_hi0, SSIM_SHIFT_DIV);
            map_hi1 = _mm256_srai_epi64(map_hi1, SSIM_SHIFT_DIV);

#if ENABLE_MINK3POOL
            __m256i const1_minus_map_lo0 = _mm256_sub_epi64(const_1_256, map_lo0);
            __m256i const1_minus_map_lo1 = _mm256_sub_epi64(const_1_256, map_lo1);
            __m256i const1_minus_map_hi0 = _mm256_sub_epi64(const_1_256, map_hi0);
            __m256i const1_minus_map_hi1 = _mm256_sub_epi64(const_1_256, map_hi1);

            __m256i const1_minus_map_sq_lo0 = _mm256_mul_epi32(const1_minus_map_lo0, const1_minus_map_lo0);
            __m256i const1_minus_map_sq_lo1 = _mm256_mul_epi32(const1_minus_map_lo1, const1_minus_map_lo1);
            __m256i const1_minus_map_sq_hi0 = _mm256_mul_epi32(const1_minus_map_hi0, const1_minus_map_hi0);
            __m256i const1_minus_map_sq_hi1 = _mm256_mul_epi32(const1_minus_map_hi1, const1_minus_map_hi1);

            __m256i rowcube_1minus_map_lo0, rowcube_1minus_map_lo1, rowcube_1minus_map_hi0, rowcube_1minus_map_hi1;

            Multiply64Bit_256(const1_minus_map_sq_lo0, const1_minus_map_lo0, rowcube_1minus_map_lo0);
            Multiply64Bit_256(const1_minus_map_sq_lo1, const1_minus_map_lo1, rowcube_1minus_map_lo1);
            Multiply64Bit_256(const1_minus_map_sq_hi0, const1_minus_map_hi0, rowcube_1minus_map_hi0);
            Multiply64Bit_256(const1_minus_map_sq_hi1, const1_minus_map_hi1, rowcube_1minus_map_hi1);
            
            rowcube_1minus_map_lo0 = _mm256_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_lo1);
            rowcube_1minus_map_hi0 = _mm256_add_epi64(rowcube_1minus_map_hi0, rowcube_1minus_map_hi1);
            rowcube_1minus_map_lo0 = _mm256_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_hi0);
            accum_rowcube_256 = _mm256_add_epi64(accum_rowcube_256, rowcube_1minus_map_lo0);

#else
            Multiply64Bit_256(map_lo0, map_lo0, map_sq_lo0);
            Multiply64Bit_256(map_lo1, map_lo1, map_sq_lo1);
            Multiply64Bit_256(map_hi0, map_hi0, map_sq_hi0);
            Multiply64Bit_256(map_hi1, map_hi1, map_sq_hi1);

            map_lo0 = _mm256_add_epi64(map_lo0, map_lo1);
            map_hi0 = _mm256_add_epi64(map_hi0, map_hi1);
            map_lo0 = _mm256_add_epi64(map_lo0, map_hi0);
            accum_map_256 = _mm256_add_epi64(accum_map_256, map_lo0);

            map_sq_lo0 = _mm256_add_epi64(map_sq_lo0, map_sq_lo1);
            map_sq_hi0 = _mm256_add_epi64(map_sq_hi0, map_sq_hi1);
            map_sq_lo0 = _mm256_add_epi64(map_sq_lo0, map_sq_hi0);
            accum_map_sq_256 = _mm256_add_epi64(accum_map_sq_256, map_sq_lo0);
#endif
        }

        for (; j < width_rem_size8; j+=8)
        {
            index = i * width + j;

            __m128i ref_b0 = _mm_loadu_si128((__m128i*)(ref->bands[0] + index));
            __m128i dis_b0 = _mm_loadu_si128((__m128i*)(dist->bands[0] + index));

            __m128i ref_b0_lo, ref_b0_hi, dis_b0_lo, dis_b0_hi;

            cvt_1_16x8_to_2_32x4(ref_b0, ref_b0_lo, ref_b0_hi);
            cvt_1_16x8_to_2_32x4(dis_b0, dis_b0_lo, dis_b0_hi);

            __m128i var_x_b0_lo = _mm_mullo_epi32(ref_b0_lo, ref_b0_lo);
            __m128i var_x_b0_hi = _mm_mullo_epi32(ref_b0_hi, ref_b0_hi);
            __m128i var_y_b0_lo = _mm_mullo_epi32(dis_b0_lo, dis_b0_lo);
            __m128i var_y_b0_hi = _mm_mullo_epi32(dis_b0_hi, dis_b0_hi);
            __m128i cov_xy_b0_lo = _mm_mullo_epi32(ref_b0_lo, dis_b0_lo);
            __m128i cov_xy_b0_hi = _mm_mullo_epi32(ref_b0_hi, dis_b0_hi);

            __m128i ref_b1 = _mm_loadu_si128((__m128i*)(ref->bands[1] + index));
            __m128i dis_b1 = _mm_loadu_si128((__m128i*)(dist->bands[1] + index));
            __m128i ref_b2 = _mm_loadu_si128((__m128i*)(ref->bands[2] + index));
            __m128i dis_b2 = _mm_loadu_si128((__m128i*)(dist->bands[2] + index));
            __m128i ref_b3 = _mm_loadu_si128((__m128i*)(ref->bands[3] + index));
            __m128i dis_b3 = _mm_loadu_si128((__m128i*)(dist->bands[3] + index));

            __m128i ref_b1_lo, ref_b1_hi, dis_b1_lo, dis_b1_hi, \
            ref_b2_lo, ref_b2_hi, dis_b2_lo, dis_b2_hi, \
            ref_b3_lo, ref_b3_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x8_to_2_32x4(ref_b1, ref_b1_lo, ref_b1_hi);
            cvt_1_16x8_to_2_32x4(dis_b1, dis_b1_lo, dis_b1_hi);
            cvt_1_16x8_to_2_32x4(ref_b2, ref_b2_lo, ref_b2_hi);
            cvt_1_16x8_to_2_32x4(dis_b2, dis_b2_lo, dis_b2_hi);
            cvt_1_16x8_to_2_32x4(ref_b3, ref_b3_lo, ref_b3_hi);
            cvt_1_16x8_to_2_32x4(dis_b3, dis_b3_lo, dis_b3_hi);

            __m128i var_x_b1_lo = _mm_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m128i var_x_b1_hi = _mm_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m128i var_y_b1_lo = _mm_mullo_epi32(dis_b1_lo, dis_b1_lo);
            __m128i var_y_b1_hi = _mm_mullo_epi32(dis_b1_hi, dis_b1_hi);

            __m128i cov_xy_b1_lo = _mm_mullo_epi32(ref_b1_lo, dis_b1_lo);
            __m128i cov_xy_b1_hi = _mm_mullo_epi32(ref_b1_hi, dis_b1_hi);

            __m128i var_x_b2_lo = _mm_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m128i var_x_b2_hi = _mm_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m128i var_y_b2_lo = _mm_mullo_epi32(dis_b2_lo, dis_b2_lo);
            __m128i var_y_b2_hi = _mm_mullo_epi32(dis_b2_hi, dis_b2_hi);
            __m128i cov_xy_b2_lo = _mm_mullo_epi32(ref_b2_lo, dis_b2_lo);
            __m128i cov_xy_b2_hi = _mm_mullo_epi32(ref_b2_hi, dis_b2_hi);

            __m128i var_x_b3_lo = _mm_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m128i var_x_b3_hi = _mm_mullo_epi32(ref_b3_hi, ref_b3_hi);
            __m128i var_y_b3_lo = _mm_mullo_epi32(dis_b3_lo, dis_b3_lo);
            __m128i var_y_b3_hi = _mm_mullo_epi32(dis_b3_hi, dis_b3_hi);
            __m128i cov_xy_b3_lo = _mm_mullo_epi32(ref_b3_lo, dis_b3_lo);
            __m128i cov_xy_b3_hi = _mm_mullo_epi32(ref_b3_hi, dis_b3_hi);

            __m128i var_x_lo = _mm_add_epi32(var_x_b1_lo, var_x_b2_lo);
            __m128i var_x_hi = _mm_add_epi32(var_x_b1_hi, var_x_b2_hi);
            __m128i var_y_lo = _mm_add_epi32(var_y_b1_lo, var_y_b2_lo);
            __m128i var_y_hi = _mm_add_epi32(var_y_b1_hi, var_y_b2_hi);
            __m128i cov_xy_lo = _mm_add_epi32(cov_xy_b1_lo, cov_xy_b2_lo);
            __m128i cov_xy_hi = _mm_add_epi32(cov_xy_b1_hi, cov_xy_b2_hi);
            var_x_lo = _mm_add_epi32(var_x_lo, var_x_b3_lo);
            var_x_hi = _mm_add_epi32(var_x_hi, var_x_b3_hi);
            var_y_lo = _mm_add_epi32(var_y_lo, var_y_b3_lo);
            var_y_hi = _mm_add_epi32(var_y_hi, var_y_b3_hi);
            cov_xy_lo = _mm_add_epi32(cov_xy_lo, cov_xy_b3_lo);
            cov_xy_hi = _mm_add_epi32(cov_xy_hi, cov_xy_b3_hi);
        
            __m128i l_den_lo = _mm_add_epi32(var_x_b0_lo, var_y_b0_lo);
            __m128i l_den_hi = _mm_add_epi32(var_x_b0_hi, var_y_b0_hi);

            var_x_lo = _mm_srai_epi32(var_x_lo, SSIM_INTER_VAR_SHIFTS);
            var_x_hi = _mm_srai_epi32(var_x_hi, SSIM_INTER_VAR_SHIFTS);
            var_y_lo = _mm_srai_epi32(var_y_lo, SSIM_INTER_VAR_SHIFTS);
            var_y_hi = _mm_srai_epi32(var_y_hi, SSIM_INTER_VAR_SHIFTS);
            cov_xy_lo = _mm_srai_epi32(cov_xy_lo, SSIM_INTER_VAR_SHIFTS);
            cov_xy_hi = _mm_srai_epi32(cov_xy_hi, SSIM_INTER_VAR_SHIFTS);

            l_den_lo = _mm_srai_epi32(l_den_lo, SSIM_INTER_L_SHIFT);
            l_den_hi = _mm_srai_epi32(l_den_hi, SSIM_INTER_L_SHIFT);

            __m128i l_num_lo = _mm_add_epi32(cov_xy_b0_lo, C1_128);
            __m128i l_num_hi = _mm_add_epi32(cov_xy_b0_hi, C1_128);

            __m128i cs_den_lo = _mm_add_epi32(var_x_lo, var_y_lo);
            __m128i cs_den_hi = _mm_add_epi32(var_x_hi, var_y_hi);
            __m128i cs_num_lo = _mm_add_epi32(cov_xy_lo, C2_128);
            __m128i cs_num_hi = _mm_add_epi32(cov_xy_hi, C2_128);

            cs_den_lo = _mm_srai_epi32(cs_den_lo, SSIM_INTER_CS_SHIFT);
            cs_den_hi = _mm_srai_epi32(cs_den_hi, SSIM_INTER_CS_SHIFT);
            
            l_den_lo = _mm_add_epi32(l_den_lo, C1_128);
            l_den_hi = _mm_add_epi32(l_den_hi, C1_128);

            cs_den_lo = _mm_add_epi32(cs_den_lo, C2_128);
            cs_den_hi = _mm_add_epi32(cs_den_hi, C2_128);

            __m128i map_num_lo0, map_num_lo1, map_num_hi0, map_num_hi1;
            __m128i map_den_lo0, map_den_lo1, map_den_hi0, map_den_hi1;

            map_num_lo0 = _mm_mul_epi32(l_num_lo, cs_num_lo);
            map_num_lo1 = _mm_mul_epi32(_mm_srai_epi64(l_num_lo, 32), _mm_srai_epi64(cs_num_lo, 32));

            map_num_hi0 = _mm_mul_epi32(l_num_hi, cs_num_hi);
            map_num_hi1 = _mm_mul_epi32(_mm_srai_epi64(l_num_hi, 32), _mm_srai_epi64(cs_num_hi, 32));

            map_den_lo0 = _mm_mul_epi32(l_den_lo, cs_den_lo);
            map_den_lo1 = _mm_mul_epi32(_mm_srai_epi64(l_den_lo, 32), _mm_srai_epi64(cs_den_lo, 32));

            map_den_hi0 = _mm_mul_epi32(l_den_hi, cs_den_hi);
            map_den_hi1 = _mm_mul_epi32(_mm_srai_epi64(l_den_hi, 32), _mm_srai_epi64(cs_den_hi, 32));

            __m128i zcnt_lo0 = _mm_lzcnt_epi64(map_den_lo0);
            __m128i zcnt_lo1 = _mm_lzcnt_epi64(map_den_lo1);
            __m128i zcnt_hi0 = _mm_lzcnt_epi64(map_den_hi0);
            __m128i zcnt_hi1 = _mm_lzcnt_epi64(map_den_hi1);

            zcnt_lo0 = _mm_sub_epi64(_mm_set1_epi64x(49), zcnt_lo0);
            zcnt_lo1 = _mm_sub_epi64(_mm_set1_epi64x(49), zcnt_lo1);
            zcnt_hi0 = _mm_sub_epi64(_mm_set1_epi64x(49), zcnt_hi0);
            zcnt_hi1 = _mm_sub_epi64(_mm_set1_epi64x(49), zcnt_hi1);

            map_den_lo0 = _mm_srav_epi64(map_den_lo0, zcnt_lo0);
            map_num_lo0 = _mm_srav_epi64(map_num_lo0, zcnt_lo0);
            map_den_lo1 = _mm_srav_epi64(map_den_lo1, zcnt_lo1);
            map_num_lo1 = _mm_srav_epi64(map_num_lo1, zcnt_lo1);
            map_den_hi0 = _mm_srav_epi64(map_den_hi0, zcnt_hi0);
            map_num_hi0 = _mm_srav_epi64(map_num_hi0, zcnt_hi0);
            map_den_hi1 = _mm_srav_epi64(map_den_hi1, zcnt_hi1);
            map_num_hi1 = _mm_srav_epi64(map_num_hi1, zcnt_hi1);

            map_den_lo0 = _mm_add_epi64(map_den_lo0, _mm_set1_epi64x(32768));
            map_den_lo1 = _mm_add_epi64(map_den_lo1, _mm_set1_epi64x(32768));
            map_den_hi0 = _mm_add_epi64(map_den_hi0, _mm_set1_epi64x(32768));
            map_den_hi1 = _mm_add_epi64(map_den_hi1, _mm_set1_epi64x(32768));

            __m128i div_lookup_lo0 = _mm_i64gather_epi32(div_lookup, map_den_lo0, 4);
            __m128i div_lookup_lo1 = _mm_i64gather_epi32(div_lookup, map_den_lo1, 4);
            __m128i div_lookup_hi0 = _mm_i64gather_epi32(div_lookup, map_den_hi0, 4);
            __m128i div_lookup_hi1 = _mm_i64gather_epi32(div_lookup, map_den_hi1, 4);
            __m128i map_lo0, map_lo1, map_hi0, map_hi1;

            Multiply64Bit_128(map_num_lo0, _mm_cvtepi32_epi64(div_lookup_lo0), map_lo0);
            Multiply64Bit_128(map_num_lo1, _mm_cvtepi32_epi64(div_lookup_lo1), map_lo1);
            Multiply64Bit_128(map_num_hi0, _mm_cvtepi32_epi64(div_lookup_hi0), map_hi0);
            Multiply64Bit_128(map_num_hi1, _mm_cvtepi32_epi64(div_lookup_hi1), map_hi1);

            map_lo0 = _mm_srai_epi64(map_lo0, SSIM_SHIFT_DIV);
            map_lo1 = _mm_srai_epi64(map_lo1, SSIM_SHIFT_DIV);
            map_hi0 = _mm_srai_epi64(map_hi0, SSIM_SHIFT_DIV);
            map_hi1 = _mm_srai_epi64(map_hi1, SSIM_SHIFT_DIV);

#if ENABLE_MINK3POOL
            __m128i const1_minus_map_lo0 = _mm_sub_epi64(const_1_128, map_lo0);
            __m128i const1_minus_map_lo1 = _mm_sub_epi64(const_1_128, map_lo1);
            __m128i const1_minus_map_hi0 = _mm_sub_epi64(const_1_128, map_hi0);
            __m128i const1_minus_map_hi1 = _mm_sub_epi64(const_1_128, map_hi1);

            __m128i const1_minus_map_sq_lo0 = _mm_mul_epi32(const1_minus_map_lo0, const1_minus_map_lo0);
            __m128i const1_minus_map_sq_lo1 = _mm_mul_epi32(const1_minus_map_lo1, const1_minus_map_lo1);
            __m128i const1_minus_map_sq_hi0 = _mm_mul_epi32(const1_minus_map_hi0, const1_minus_map_hi0);
            __m128i const1_minus_map_sq_hi1 = _mm_mul_epi32(const1_minus_map_hi1, const1_minus_map_hi1);

            __m128i rowcube_1minus_map_lo0, rowcube_1minus_map_lo1, rowcube_1minus_map_hi0, rowcube_1minus_map_hi1;

            Multiply64Bit_128(const1_minus_map_sq_lo0, const1_minus_map_lo0, rowcube_1minus_map_lo0);
            Multiply64Bit_128(const1_minus_map_sq_lo1, const1_minus_map_lo1, rowcube_1minus_map_lo1);
            Multiply64Bit_128(const1_minus_map_sq_hi0, const1_minus_map_hi0, rowcube_1minus_map_hi0);
            Multiply64Bit_128(const1_minus_map_sq_hi1, const1_minus_map_hi1, rowcube_1minus_map_hi1);

            rowcube_1minus_map_lo0 = _mm_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_lo1);
            rowcube_1minus_map_hi0 = _mm_add_epi64(rowcube_1minus_map_hi0, rowcube_1minus_map_hi1);
            rowcube_1minus_map_lo0 = _mm_add_epi64(rowcube_1minus_map_lo0, rowcube_1minus_map_hi0);
            accum_rowcube_128 = _mm_add_epi64(accum_rowcube_128, rowcube_1minus_map_lo0);

#else
            __m128i  map_sq_lo0, map_sq_lo1, map_sq_hi0, map_sq_hi1;
            Multiply64Bit_128(map_lo0, map_lo0, map_sq_lo0);
            Multiply64Bit_128(map_lo1, map_lo1, map_sq_lo1);
            Multiply64Bit_128(map_hi0, map_hi0, map_sq_hi0);
            Multiply64Bit_128(map_hi1, map_hi1, map_sq_hi1);

            map_lo0 = _mm_add_epi64(map_lo0, map_lo1);
            map_hi0 = _mm_add_epi64(map_hi0, map_hi1);
            map_lo0 = _mm_add_epi64(map_lo0, map_hi0);
            accum_map_128 = _mm_add_epi64(accum_map_128, map_lo0);

            map_sq_lo0 = _mm_add_epi64(map_sq_lo0, map_sq_lo1);
            map_sq_hi0 = _mm_add_epi64(map_sq_hi0, map_sq_hi1);
            map_sq_lo0 = _mm_add_epi64(map_sq_lo0, map_sq_hi0);
            accum_map_sq_128 = _mm_add_epi64(accum_map_sq_128, map_sq_lo0);
#endif
        }

        for (; j < width; j++)
        {
            index = i * width + j;
            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;

            for (int k = 1; k < 4; k++)
            {
                var_x += ((ssim_inter_dtype)ref->bands[k][index] * ref->bands[k][index]);
                var_y += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index] * dist->bands[k][index]);
            }
            var_x_band0 = (ssim_inter_dtype)mx * mx;
            var_y_band0 = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;

            var_x = (var_x >> SSIM_INTER_VAR_SHIFTS);
            var_y = (var_y >> SSIM_INTER_VAR_SHIFTS);
            cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

            l_num = (cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);
            cs_num = (cov_xy + C2);
            cs_den = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            numVal[j] = (ssim_accum_dtype)l_num * cs_num;
            denVal[j] = (ssim_accum_dtype)l_den * cs_den;

            map_num = (ssim_accum_dtype)l_num * cs_num;
            map_den = (ssim_accum_dtype)l_den * cs_den;
            
            /**
             * l_den & cs_den are variance terms, hence they will always be +ve 
             * getting best 15bits and retaining one signed bit, using get_best_i16_from_u64
             * This is done to reuse ADM division LUT, which has LUT for values from -2^15 to 2^15
            */
            int power_val;
            i16_map_den = get_best_i16_from_u64((uint64_t) map_den, &power_val);
            /**
             * The actual equation of map is map_num/map_den
             * The division is done using LUT, results of div_lookup = 2^30/i16_map_den
             * map = map_num/map_den => map = map_num / (i16_map_den << power_val)
             * => map = (map_num >> power_val) / i16_map_den
             * => map = (map_num >> power_val) * (div_lookup[i16_map_den + 32768] >> 30) //since it has -ve vals in 1st half
             * => map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> 30
             * Shift by 30 might be very high even for 32 bits precision, hence shift only by 15 
            */
            map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;

#if ENABLE_MINK3POOL
            ssim_accum_dtype const1_minus_map = const_1 - map;
            rowcube_1minus_map += const1_minus_map * const1_minus_map * const1_minus_map;
#else
            accum_map += map;
            map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype)map * map));
#endif
        }
#if ENABLE_MINK3POOL
        __m256i r4 = _mm256_add_epi64( _mm512_castsi512_si256(accum_rowcube_512), _mm512_extracti64x4_epi64(accum_rowcube_512, 1));
        r4 = _mm256_add_epi64(r4, accum_rowcube_256);
        __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(r4), _mm256_extracti128_si256(r4, 1));
        r2 = _mm_add_epi64(r2, accum_rowcube_128);
        accumcube_1minus_map += (double)(_mm_extract_epi64(r2, 0) + _mm_extract_epi64(r2, 1) + rowcube_1minus_map);
        accum_rowcube_512 = _mm512_setzero_si512();
        accum_rowcube_256 = _mm256_setzero_si256();
        accum_rowcube_128 = _mm_setzero_si128();
        rowcube_1minus_map = 0;
#endif
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt(accumcube_1minus_map/(width*height))/const_1;
    *score = ssim_clip(ssim_val, 0, 1);
#else
    __m256i r4_map = _mm256_add_epi64(_mm512_castsi512_si256(accum_map_512), _mm512_extracti64x4_epi64(accum_map_512, 1));
    r4_map = _mm256_add_epi64(r4_map, accum_map_256);
    __m256i r4_map_sq = _mm256_add_epi64(_mm512_castsi512_si256(accum_map_sq_512), _mm512_extracti64x4_epi64(accum_map_sq_512, 1));
    r4_map_sq = _mm256_add_epi64(r4_map_sq, accum_map_sq_256);
    __m128i r2_map = _mm_add_epi64(_mm256_castsi256_si128(r4_map), _mm256_extracti64x2_epi64(r4_map, 1));
    r2_map = _mm_add_epi64(r2_map, accum_map_128);
    __m128i r2_map_sq = _mm_add_epi64(_mm256_castsi256_si128(r4_map_sq), _mm256_extracti64x2_epi64(r4_map_sq, 1));
    r2_map_sq = _mm_add_epi64(r2_map_sq, accum_map_sq_128);
    int64_t r1_map = _mm_extract_epi64(r2_map, 0) + _mm_extract_epi64(r2_map, 1);
    int64_t r1_map_sq = _mm_extract_epi64(r2_map_sq, 0) + _mm_extract_epi64(r2_map_sq, 1);
    
    accum_map += r1_map;
    map_sq_insum += r1_map_sq;
    accum_map_sq = map_sq_insum / (height * width);
    double ssim_mean = (double)accum_map / (height * width);
    double ssim_std; 
    ssim_std = sqrt(MAX(0, ((double) accum_map_sq - ssim_mean*ssim_mean)));
    *score = (ssim_std / ssim_mean);

#endif

    free(numVal);
    free(denVal);
    ret = 0;
    return ret;
}