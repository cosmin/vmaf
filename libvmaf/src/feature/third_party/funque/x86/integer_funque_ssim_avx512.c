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

#define cvt_1_32x8_to_2_64x4(a_32x8, r_64x4_lo, r_64x4_hi)                       \
    {                                                                            \
        r_64x4_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a_32x8));       \
        r_64x4_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(a_32x8, 1)); \
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

int integer_compute_ssim_funque_avx512(i_dwt2buffers *ref, i_dwt2buffers *dist, SsimScore_int *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;

    int width = ref->width;
    int height = ref->height;

    /**
     * C1 is constant is added to ref^2, dist^2,
     *  - hence we have to multiply by pending_div^2
     * As per floating point,C1 is added to 2*(mx/win_dim)*(my/win_dim) &
     * (mx/win_dim)*(mx/win_dim)+(my/win_dim)*(my/win_dim) win_dim = 1 << n_levels, where n_levels =
     * 1 Since win_dim division is avoided for mx & my, C1 is left shifted by 1
     */
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) *
                           ((pending_div * pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    /**
     * shifts are handled similar to C1
     * not shifted left because the other terms to which this is added undergoes equivalent right
     * shift
     */
    ssim_inter_dtype C2 =
        ((K2 * max_val) * (K2 * max_val) *
         ((pending_div * pending_div) << (2 - SSIM_INTER_VAR_SHIFTS + SSIM_INTER_CS_SHIFT)));

    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map;
    int16_t i16_map_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

    ssim_accum_dtype rowcube_1minus_map = 0;
    double accumcube_1minus_map = 0;
    const ssim_inter_dtype const_1 = 32768;  // div_Q_factor>>SSIM_SHIFT_DIV

    __m512i const_1_512 = _mm512_set1_epi64(32768);
    __m512i accum_rowcube_512 = _mm512_setzero_si512();

    __m256i const_1_256 = _mm256_set1_epi64x(32768);
    __m256i accum_rowcube_256 = _mm256_setzero_si256();

    __m128i const_1_128 = _mm_set1_epi64x(32768);
    __m128i accum_rowcube_128 = _mm_setzero_si128();

    ssim_accum_dtype accum_map = 0;

    __m512i C1_512 = _mm512_set1_epi32(C1);
    __m512i C2_512 = _mm512_set1_epi32(C2);
    __m512i constant_2 = _mm512_set1_epi32(2);

    int64_t *numVal = (int64_t *)malloc(width * sizeof(int64_t));
    int64_t *denVal = (int64_t *)malloc(width * sizeof(int64_t));

    int width_rem_size32 = width - (width % 32);
    int index = 0, j, k;

    for (int i = 0; i < height; i++)
    {
        for(j = 0; j < width_rem_size32; j += 32) {
            index = i * width + j;

            __m512i ref_b0 = _mm512_loadu_si512((__m512i *) (ref->bands[0] + index));
            __m512i dis_b0 = _mm512_loadu_si512((__m512i *) (dist->bands[0] + index));

            __m512i ref_b0_lo, ref_b0_hi, dis_b0_lo, dis_b0_hi;

            cvt_1_16x16_to_2_32x8_512(ref_b0, ref_b0_lo, ref_b0_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b0, dis_b0_lo, dis_b0_hi);

            __m512i var_x_b0_lo = _mm512_mullo_epi32(ref_b0_lo, ref_b0_lo);
            __m512i var_x_b0_hi = _mm512_mullo_epi32(ref_b0_hi, ref_b0_hi);
            __m512i var_y_b0_lo = _mm512_mullo_epi32(dis_b0_lo, dis_b0_lo);
            __m512i var_y_b0_hi = _mm512_mullo_epi32(dis_b0_hi, dis_b0_hi);
            __m512i cov_xy_b0_lo = _mm512_mullo_epi32(ref_b0_lo, dis_b0_lo);
            __m512i cov_xy_b0_hi = _mm512_mullo_epi32(ref_b0_hi, dis_b0_hi);

            __m512i ref_b1 = _mm512_loadu_si512((__m512i *) (ref->bands[1] + index));
            __m512i dis_b1 = _mm512_loadu_si512((__m512i *) (dist->bands[1] + index));
            __m512i ref_b2 = _mm512_loadu_si512((__m512i *) (ref->bands[2] + index));
            __m512i dis_b2 = _mm512_loadu_si512((__m512i *) (dist->bands[2] + index));
            __m512i ref_b3 = _mm512_loadu_si512((__m512i *) (ref->bands[3] + index));
            __m512i dis_b3 = _mm512_loadu_si512((__m512i *) (dist->bands[3] + index));

            __m512i ref_b1_lo, ref_b1_hi, dis_b1_lo, dis_b1_hi, ref_b2_lo, ref_b2_hi, dis_b2_lo,
                dis_b2_hi, ref_b3_lo, ref_b3_hi, dis_b3_lo, dis_b3_hi;
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

            __m512i l_num_lo =
                _mm512_add_epi32(_mm512_mullo_epi32(cov_xy_b0_lo, constant_2), C1_512);
            __m512i l_num_hi =
                _mm512_add_epi32(_mm512_mullo_epi32(cov_xy_b0_hi, constant_2), C1_512);

            __m512i cs_den_lo = _mm512_add_epi32(var_x_lo, var_y_lo);
            __m512i cs_den_hi = _mm512_add_epi32(var_x_hi, var_y_hi);
            __m512i cs_num_lo = _mm512_add_epi32(_mm512_mullo_epi32(cov_xy_lo, constant_2), C2_512);
            __m512i cs_num_hi = _mm512_add_epi32(_mm512_mullo_epi32(cov_xy_hi, constant_2), C2_512);

            cs_den_lo = _mm512_srai_epi32(cs_den_lo, SSIM_INTER_CS_SHIFT);
            cs_den_hi = _mm512_srai_epi32(cs_den_hi, SSIM_INTER_CS_SHIFT);

            l_den_lo = _mm512_add_epi32(l_den_lo, C1_512);
            l_den_hi = _mm512_add_epi32(l_den_hi, C1_512);

            cs_den_lo = _mm512_add_epi32(cs_den_lo, C2_512);
            cs_den_hi = _mm512_add_epi32(cs_den_hi, C2_512);

            __m512i l_num_lo0, l_num_lo1, l_num_hi0, l_num_hi1, cs_num_lo0, cs_num_lo1, cs_num_hi0,
                cs_num_hi1, l_den_lo0, l_den_lo1, l_den_hi0, l_den_hi1, cs_den_lo0, cs_den_lo1,
                cs_den_hi0, cs_den_hi1;

            cvt_1_32x8_to_2_64x4(l_num_lo, l_num_lo0, l_num_lo1);
            cvt_1_32x8_to_2_64x4(l_num_hi, l_num_hi0, l_num_hi1);

            cvt_1_32x8_to_2_64x4(cs_num_lo, cs_num_lo0, cs_num_lo1);
            cvt_1_32x8_to_2_64x4(cs_num_hi, cs_num_hi0, cs_num_hi1);

            cvt_1_32x8_to_2_64x4(l_den_lo, l_den_lo0, l_den_lo1);
            cvt_1_32x8_to_2_64x4(l_den_hi, l_den_hi0, l_den_hi1);

            cvt_1_32x8_to_2_64x4(cs_den_lo, cs_den_lo0, cs_den_lo1);
            cvt_1_32x8_to_2_64x4(cs_den_hi, cs_den_hi0, cs_den_hi1);

            __m512i map_num_lo0, map_num_lo1, map_num_hi0, map_num_hi1;
            __m512i map_den_lo0, map_den_lo1, map_den_hi0, map_den_hi1;

            Multiply64Bit_512(l_num_lo0, cs_num_lo0, map_num_lo0);
            Multiply64Bit_512(l_num_lo1, cs_num_lo1, map_num_lo1);
            Multiply64Bit_512(l_num_hi0, cs_num_hi0, map_num_hi0);
            Multiply64Bit_512(l_num_hi1, cs_num_hi1, map_num_hi1);

            Multiply64Bit_512(l_den_lo0, cs_den_lo0, map_den_lo0);
            Multiply64Bit_512(l_den_lo1, cs_den_lo1, map_den_lo1);
            Multiply64Bit_512(l_den_hi0, cs_den_hi0, map_den_hi0);
            Multiply64Bit_512(l_den_hi1, cs_den_hi1, map_den_hi1);

            _mm512_storeu_si512((__m512i *) (numVal + j), map_num_lo0);
            _mm512_storeu_si512((__m512i *) (numVal + j + 8), map_num_lo1);
            _mm512_storeu_si512((__m512i *) (numVal + j + 16), map_num_hi0);
            _mm512_storeu_si512((__m512i *) (numVal + j + 24), map_num_hi1);

            _mm512_storeu_si512((__m512i *) (denVal + j), map_den_lo0);
            _mm512_storeu_si512((__m512i *) (denVal + j + 8), map_den_lo1);
            _mm512_storeu_si512((__m512i *) (denVal + j + 16), map_den_hi0);
            _mm512_storeu_si512((__m512i *) (denVal + j + 24), map_den_hi1);
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

            l_num = (2 * cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);
            cs_num = (2 * cov_xy + C2);
            cs_den = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            numVal[j] = (ssim_accum_dtype)l_num * cs_num;
            denVal[j] = (ssim_accum_dtype)l_den * cs_den;
        }
        for(k = 0; k < width; k++) {
            int power_val;
            i16_map_den = get_best_i16_from_u64((uint64_t) denVal[k], &power_val);
            map = ((numVal[k] >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;

            ssim_accum_dtype const1_minus_map = const_1 - map;
            rowcube_1minus_map += const1_minus_map * const1_minus_map * const1_minus_map;

            accum_map += map;
            // map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype)map * map));
        }
        __m256i r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_rowcube_512),
                                      _mm512_extracti64x4_epi64(accum_rowcube_512, 1));
        r4 = _mm256_add_epi64(r4, accum_rowcube_256);
        __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(r4), _mm256_extracti128_si256(r4, 1));
        r2 = _mm_add_epi64(r2, accum_rowcube_128);
        accumcube_1minus_map += (double)(_mm_extract_epi64(r2, 0) + _mm_extract_epi64(r2, 1) + rowcube_1minus_map);
        accum_rowcube_512 = _mm512_setzero_si512();
        accum_rowcube_256 = _mm256_setzero_si256();
        accum_rowcube_128 = _mm_setzero_si128();
        rowcube_1minus_map = 0;
    }

    double ssim_val = 1 - cbrt(accumcube_1minus_map / (width * height)) / const_1;
    score->mink3 = ssim_clip(ssim_val, 0, 1);

    score->mean = (double) accum_map / (height * width) / (1 << SSIM_SHIFT_DIV);

    free(numVal);
    free(denVal);
    ret = 0;
    return ret;
}

static inline int16_t ms_ssim_get_best_i16_from_u32_avx512(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);

    if(k > 17) {
        k -= 17;
        // temp = temp << k;
        *x = 0;
    } else if(k < 16) {
        k = 17 - k;
        temp = temp >> k;
        *x = k;
    } else {
        *x = 0;
        if(temp >> 15) {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (int16_t) temp;
}

int integer_compute_ms_ssim_funque_avx512(i_dwt2buffers *ref, i_dwt2buffers *dist,
                                          MsSsimScore_int *score, int max_val, float K1, float K2,
                                          int pending_div, int32_t *div_lookup, int n_levels,
                                          int is_pyr)
{
    int ret = 1;

    int cum_array_width = (ref->width) * (1 << n_levels);
    int win_size = (n_levels << 1);
    int win_size_c2 = win_size;
    pending_div = pending_div >> (n_levels - 1);
    int pending_div_c1 = pending_div;
    int pending_div_c2 = pending_div;
    int pending_div_offset = 0;
    int pending_div_halfround = 0;
    int width = ref->width;
    int height = ref->height;

    int32_t *var_x_cum = *(score->var_x_cum);
    int32_t *var_y_cum = *(score->var_y_cum);
    int32_t *cov_xy_cum = *(score->cov_xy_cum);

    if(is_pyr) {
        win_size_c2 = 2;
        pending_div_c1 = (1 << i_nadenau_pending_div_factors[n_levels - 1][0]) * 255;
        pending_div_c2 =
            (1 << (i_nadenau_pending_div_factors[n_levels - 1][1] + (n_levels - 1))) * 255;
        pending_div_offset = 2 * (i_nadenau_pending_div_factors[n_levels - 1][3] -
                                  i_nadenau_pending_div_factors[n_levels - 1][1]);
        pending_div_halfround = (pending_div_offset == 0) ? 0 : (1 << (pending_div_offset - 1));
        if((n_levels > 1)) {
            int index_cum = 0;
            int shift_cums = 2 * (i_nadenau_pending_div_factors[n_levels - 2][1] -
                                  i_nadenau_pending_div_factors[n_levels - 1][1] - 1);
            for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                    var_x_cum[index_cum] =
                        (var_x_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    var_y_cum[index_cum] =
                        (var_y_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    cov_xy_cum[index_cum] =
                        (cov_xy_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    index_cum++;
                }
                index_cum += (cum_array_width - width);
            }
        }
    }

    int64_t c1_mul = (((int64_t) pending_div_c1 * pending_div_c1) >> (SSIM_INTER_L_SHIFT));
    int64_t c2_mul = (((int64_t) pending_div_c2 * pending_div_c2) >>
                      (SSIM_INTER_VAR_SHIFTS + SSIM_INTER_CS_SHIFT));

    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * c1_mul);

    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * c2_mul);

    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map, l, cs;
    int16_t i16_l_den;
    int16_t i16_cs_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;

    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_l = 0;
    ssim_accum_dtype accum_cs = 0;
    ssim_accum_dtype accum_sq_map = 0;
    ssim_accum_dtype map_sq = 0;

    ssim_inter_dtype mink3_const = 32768;
    ssim_inter_dtype mink3_const_map = (mink3_const * mink3_const) >> SSIM_R_SHIFT;
    ssim_inter_dtype mink3_const_l = mink3_const >> L_R_SHIFT;
    ssim_inter_dtype mink3_const_cs = mink3_const >> CS_R_SHIFT;

    ssim_inter_dtype map_r_shift = 0;
    ssim_inter_dtype l_r_shift = 0;
    ssim_inter_dtype cs_r_shift = 0;
    ssim_mink3_accum_dtype mink3_map = 0;
    ssim_mink3_accum_dtype mink3_l = 0;
    ssim_mink3_accum_dtype mink3_cs = 0;
    ssim_mink3_accum_dtype accum_mink3_map = 0;
    ssim_mink3_accum_dtype accum_mink3_l = 0;
    ssim_mink3_accum_dtype accum_mink3_cs = 0;

    int is_pyr_sft = (is_pyr == 1) ? 0 : 2;
    int const_factor = 2 >> SSIM_INTER_L_SHIFT;
    __m512i const_fact = _mm512_set1_epi32(const_factor);

    __m512i C1_512 = _mm512_set1_epi32(C1);
    __m512i C2_512 = _mm512_set1_epi32(C2);

    int32_t *lNumVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *csNumVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *lDenVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *csDenVal = (int32_t *) malloc(width * sizeof(int32_t));

    int width_rem_size16 = width - (width % 32);
    // int width_rem_size8 = width - (width % 8);
    int index = 0, j, k;
    int index_cum = 0;
    for(int i = 0; i < height; i++) {
        ssim_accum_dtype row_accum_sq_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_l = 0;
        ssim_mink3_accum_dtype row_accum_mink3_cs = 0;
        for(j = 0; j < width_rem_size16; j += 32) {
            index = i * width + j;

            __m512i ref_b0 = _mm512_loadu_si512((__m512i *) (ref->bands[0] + index));
            __m512i dis_b0 = _mm512_loadu_si512((__m512i *) (dist->bands[0] + index));

            __m512i ref_b0_lo, ref_b0_hi, dis_b0_lo, dis_b0_hi;
            cvt_1_16x16_to_2_32x8_512(ref_b0, ref_b0_lo, ref_b0_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b0, dis_b0_lo, dis_b0_hi);

            __m512i var_x_b0_lo = _mm512_mullo_epi32(ref_b0_lo, ref_b0_lo);
            __m512i var_x_b0_hi = _mm512_mullo_epi32(ref_b0_hi, ref_b0_hi);
            __m512i var_y_b0_lo = _mm512_mullo_epi32(dis_b0_lo, dis_b0_lo);
            __m512i var_y_b0_hi = _mm512_mullo_epi32(dis_b0_hi, dis_b0_hi);
            __m512i cov_xy_b0_lo = _mm512_mullo_epi32(ref_b0_lo, dis_b0_lo);
            __m512i cov_xy_b0_hi = _mm512_mullo_epi32(ref_b0_hi, dis_b0_hi);

            var_x_b0_lo = _mm512_srai_epi32(var_x_b0_lo, win_size);
            var_x_b0_hi = _mm512_srai_epi32(var_x_b0_hi, win_size);
            var_y_b0_lo = _mm512_srai_epi32(var_y_b0_lo, win_size);
            var_y_b0_hi = _mm512_srai_epi32(var_y_b0_hi, win_size);
            cov_xy_b0_lo = _mm512_srai_epi32(cov_xy_b0_lo, win_size);
            cov_xy_b0_hi = _mm512_srai_epi32(cov_xy_b0_hi, win_size);

            __m512i l_den_lo = _mm512_add_epi32(var_x_b0_lo, var_y_b0_lo);
            __m512i l_den_hi = _mm512_add_epi32(var_x_b0_hi, var_y_b0_hi);

            cov_xy_b0_lo = _mm512_mullo_epi32(cov_xy_b0_lo, const_fact);
            cov_xy_b0_hi = _mm512_mullo_epi32(cov_xy_b0_hi, const_fact);

            l_den_lo = _mm512_srai_epi32(l_den_lo, SSIM_INTER_L_SHIFT);
            l_den_hi = _mm512_srai_epi32(l_den_hi, SSIM_INTER_L_SHIFT);

            l_den_lo = _mm512_add_epi32(l_den_lo, C1_512);
            l_den_hi = _mm512_add_epi32(l_den_hi, C1_512);

            cov_xy_b0_lo = _mm512_add_epi32(cov_xy_b0_lo, C1_512);
            cov_xy_b0_hi = _mm512_add_epi32(cov_xy_b0_hi, C1_512);

            _mm512_storeu_si512((__m512i *) (lDenVal + j), l_den_lo);
            _mm512_storeu_si512((__m512i *) (lDenVal + j + 16), l_den_hi);

            _mm512_storeu_si512((__m512i *) (lNumVal + j), cov_xy_b0_lo);
            _mm512_storeu_si512((__m512i *) (lNumVal + j + 16), cov_xy_b0_hi);

            __m512i ref_b1 = _mm512_loadu_si512((__m512i *) (ref->bands[1] + index));
            __m512i dis_b1 = _mm512_loadu_si512((__m512i *) (dist->bands[1] + index));
            __m512i ref_b2 = _mm512_loadu_si512((__m512i *) (ref->bands[2] + index));
            __m512i dis_b2 = _mm512_loadu_si512((__m512i *) (dist->bands[2] + index));
            __m512i ref_b3 = _mm512_loadu_si512((__m512i *) (ref->bands[3] + index));
            __m512i dis_b3 = _mm512_loadu_si512((__m512i *) (dist->bands[3] + index));

            __m512i ref_b1_lo, ref_b1_hi, dis_b1_lo, dis_b1_hi, ref_b2_lo, ref_b2_hi, dis_b2_lo,
                dis_b2_hi, ref_b3_lo, ref_b3_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x16_to_2_32x8_512(ref_b1, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b1, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b2, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b2, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b3, ref_b3_lo, ref_b3_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b3, dis_b3_lo, dis_b3_hi);

            __m512i varXcum32x8_lb = _mm512_loadu_si512((__m512i *) (var_x_cum + index_cum));
            __m512i varXcum32x8_hb = _mm512_loadu_si512((__m512i *) (var_x_cum + index_cum + 16));
            __m512i varYcum32x8_lb = _mm512_loadu_si512((__m512i *) (var_y_cum + index_cum));
            __m512i varYcum32x8_hb = _mm512_loadu_si512((__m512i *) (var_y_cum + index_cum + 16));
            __m512i covXYcum32x8_lb = _mm512_loadu_si512((__m512i *) (cov_xy_cum + index_cum));
            __m512i covXYcum32x8_hb = _mm512_loadu_si512((__m512i *) (cov_xy_cum + index_cum + 16));

            varXcum32x8_lb = _mm512_srai_epi32(varXcum32x8_lb, is_pyr_sft);
            varXcum32x8_hb = _mm512_srai_epi32(varXcum32x8_hb, is_pyr_sft);
            varYcum32x8_lb = _mm512_srai_epi32(varYcum32x8_lb, is_pyr_sft);
            varYcum32x8_hb = _mm512_srai_epi32(varYcum32x8_hb, is_pyr_sft);
            covXYcum32x8_lb = _mm512_srai_epi32(covXYcum32x8_lb, is_pyr_sft);
            covXYcum32x8_hb = _mm512_srai_epi32(covXYcum32x8_hb, is_pyr_sft);

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

            var_x_lo = _mm512_srai_epi32(var_x_lo, win_size_c2);
            var_x_hi = _mm512_srai_epi32(var_x_hi, win_size_c2);
            var_y_lo = _mm512_srai_epi32(var_y_lo, win_size_c2);
            var_y_hi = _mm512_srai_epi32(var_y_hi, win_size_c2);
            cov_xy_lo = _mm512_srai_epi32(cov_xy_lo, win_size_c2);
            cov_xy_hi = _mm512_srai_epi32(cov_xy_hi, win_size_c2);

            var_x_lo = _mm512_add_epi32(var_x_lo, varXcum32x8_lb);
            var_x_hi = _mm512_add_epi32(var_x_hi, varXcum32x8_hb);
            var_y_lo = _mm512_add_epi32(var_y_lo, varYcum32x8_lb);
            var_y_hi = _mm512_add_epi32(var_y_hi, varYcum32x8_hb);
            cov_xy_lo = _mm512_add_epi32(cov_xy_lo, covXYcum32x8_lb);
            cov_xy_hi = _mm512_add_epi32(cov_xy_hi, covXYcum32x8_hb);

            _mm512_storeu_si512((__m512i *) (var_x_cum + index_cum), var_x_lo);
            _mm512_storeu_si512((__m512i *) (var_x_cum + index_cum + 16), var_x_hi);
            _mm512_storeu_si512((__m512i *) (var_y_cum + index_cum), var_y_lo);
            _mm512_storeu_si512((__m512i *) (var_y_cum + index_cum + 16), var_y_hi);
            _mm512_storeu_si512((__m512i *) (cov_xy_cum + index_cum), cov_xy_lo);
            _mm512_storeu_si512((__m512i *) (cov_xy_cum + index_cum + 16), cov_xy_hi);

            // __m256i l_num_lo = _mm256_add_epi32(cov_xy_b0_lo, C1_256);
            // __m256i l_num_hi = _mm256_add_epi32(cov_xy_b0_hi, C1_256);

            __m512i cs_den_lo = _mm512_add_epi32(var_x_lo, var_y_lo);
            __m512i cs_den_hi = _mm512_add_epi32(var_x_hi, var_y_hi);
            cs_den_lo = _mm512_srai_epi32(cs_den_lo, SSIM_INTER_CS_SHIFT);
            cs_den_hi = _mm512_srai_epi32(cs_den_hi, SSIM_INTER_CS_SHIFT);

            cov_xy_lo = _mm512_mullo_epi32(cov_xy_lo, const_fact);
            cov_xy_hi = _mm512_mullo_epi32(cov_xy_hi, const_fact);
            __m512i cs_num_lo = _mm512_add_epi32(cov_xy_lo, C2_512);
            __m512i cs_num_hi = _mm512_add_epi32(cov_xy_hi, C2_512);

            cs_den_lo = _mm512_add_epi32(cs_den_lo, C2_512);
            cs_den_hi = _mm512_add_epi32(cs_den_hi, C2_512);

            _mm512_storeu_si512((__m512i *) (csNumVal + j), cs_num_lo);
            _mm512_storeu_si512((__m512i *) (csNumVal + j + 16), cs_num_hi);

            _mm512_storeu_si512((__m512i *) (csDenVal + j), cs_den_lo);
            _mm512_storeu_si512((__m512i *) (csDenVal + j + 16), cs_den_hi);

            index_cum += 32;
        }

        for(; j < width; j++) {
            index = i * width + j;

            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;
            int k;
#if BAND_HVD_SAME_PENDING_DIV
            for(k = 1; k < 4; k++)
#else
            for(k = 1; k < 3; k++)
#endif
            {
                var_x += ((ssim_inter_dtype) ref->bands[k][index] * ref->bands[k][index]);
                var_y += ((ssim_inter_dtype) dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype) ref->bands[k][index] * dist->bands[k][index]);
            }
#if !(BAND_HVD_SAME_PENDING_DIV)
            // The extra right shift will be done for pyr since the upscale factors are different
            // for subbands
            var_x += (((ssim_inter_dtype) ref->bands[k][index] * ref->bands[k][index]) +
                      pending_div_halfround) >>
                     pending_div_offset;
            var_y += (((ssim_inter_dtype) dist->bands[k][index] * dist->bands[k][index]) +
                      pending_div_halfround) >>
                     pending_div_offset;
            cov_xy += (((ssim_inter_dtype) ref->bands[k][index] * dist->bands[k][index]) +
                       pending_div_halfround) >>
                      pending_div_offset;
#endif
            var_x_band0 = ((ssim_inter_dtype) mx * mx) >> win_size;
            var_y_band0 = ((ssim_inter_dtype) my * my) >> win_size;
            cov_xy_band0 = ((ssim_inter_dtype) mx * my) >> win_size;

            var_x_cum[index_cum] = var_x_cum[index_cum] >> is_pyr_sft;
            var_y_cum[index_cum] = var_y_cum[index_cum] >> is_pyr_sft;
            cov_xy_cum[index_cum] = cov_xy_cum[index_cum] >> is_pyr_sft;

            var_x_cum[index_cum] += (var_x >> win_size_c2);
            var_y_cum[index_cum] += (var_y >> win_size_c2);
            cov_xy_cum[index_cum] += (cov_xy >> win_size_c2);

            var_x = var_x_cum[index_cum];
            var_y = var_y_cum[index_cum];
            cov_xy = cov_xy_cum[index_cum];

            lNumVal[j] = ((2 >> SSIM_INTER_L_SHIFT) * cov_xy_band0 + C1);
            lDenVal[j] = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);

            csNumVal[j] = ((2 >> SSIM_INTER_CS_SHIFT) * cov_xy + C2);
            csDenVal[j] = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            index_cum++;
        }

        for(k = 0; k < width; k++) {
            int power_val_l;
            i16_l_den = ms_ssim_get_best_i16_from_u32_avx512((uint32_t) lDenVal[k], &power_val_l);

            int power_val_cs;
            i16_cs_den =
                ms_ssim_get_best_i16_from_u32_avx512((uint32_t) csDenVal[k], &power_val_cs);

            l = ((lNumVal[k] >> power_val_l) * div_lookup[i16_l_den + 32768]) >> SSIM_SHIFT_DIV;
            cs = ((csNumVal[k] >> power_val_cs) * div_lookup[i16_cs_den + 32768]) >> SSIM_SHIFT_DIV;
            map = l * cs;

            accum_l += l;
            accum_cs += cs;
            accum_map += map;
            map_sq = ((int64_t) map * map) >> SSIM_SQ_ROW_SHIFT;
            row_accum_sq_map += map_sq;

            l_r_shift = l >> L_R_SHIFT;
            cs_r_shift = cs >> CS_R_SHIFT;
            map_r_shift = map >> SSIM_R_SHIFT;

            mink3_l = pow((mink3_const_l - l_r_shift), 3);
            mink3_cs = pow((mink3_const_cs - cs_r_shift), 3);
            mink3_map = pow((mink3_const_map - map_r_shift), 3);

            row_accum_mink3_l += mink3_l;
            row_accum_mink3_cs += mink3_cs;
            row_accum_mink3_map += mink3_map;
        }
        accum_sq_map += (row_accum_sq_map >> SSIM_SQ_COL_SHIFT);

        accum_mink3_l += (row_accum_mink3_l >> L_MINK3_ROW_R_SHIFT);
        accum_mink3_cs += (row_accum_mink3_cs >> CS_MINK3_ROW_R_SHIFT);
        accum_mink3_map += (row_accum_mink3_map >> SSIM_MINK3_ROW_R_SHIFT);

        index_cum += (cum_array_width - width);
    }

    double l_mean = (double) accum_l / (height * width);
    double cs_mean = (double) accum_cs / (height * width);
    double ssim_mean = (double) accum_map / (height * width);

    double inter_shift_sq = 1 << (SSIM_SQ_ROW_SHIFT + SSIM_SQ_COL_SHIFT);
    double ssim_var =
        (((double) accum_sq_map / (height * width)) * inter_shift_sq) - ((ssim_mean * ssim_mean));


    double mink3_cbrt_const_l = pow(2, (39 / 3));
    double mink3_cbrt_const_cs = pow(2, (38.0 / 3));
    double mink3_cbrt_const_map = pow(2, (38.0 / 3));

    double l_mink3 = mink3_cbrt_const_l - (double) cbrt(accum_mink3_l / (width * height));
    double cs_mink3 = mink3_cbrt_const_cs - (double) cbrt(accum_mink3_cs / (width * height));
    double ssim_mink3 = mink3_cbrt_const_map - (double) cbrt(accum_mink3_map / (width * height));

    score->ssim_mean = ssim_mean / (1 << (SSIM_SHIFT_DIV * 2));
    score->l_mean = l_mean / (1 << SSIM_SHIFT_DIV);
    score->cs_mean = cs_mean / (1 << SSIM_SHIFT_DIV);
    score->l_mink3 = l_mink3 / pow(2, (39 / 3));
    score->cs_mink3 = cs_mink3 / pow(2, (38.0 / 3));
    score->ssim_mink3 = ssim_mink3 / pow(2, (38.0 / 3));

    ret = 0;
    return ret;
}

int integer_mean_2x2_ms_ssim_funque_avx512(int32_t *var_x_cum, int32_t *var_y_cum,
                                           int32_t *cov_xy_cum, int width, int height, int level)
{
    int ret = 1;

    int index = 0;
    int index_cum = 0;
    int cum_array_width = (width) * (1 << (level + 1));

    __m512i shuffleMask = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    __m512i permIndex = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    __m512i vec_constant = _mm512_set1_epi32(2);
    int i = 0;
    int j = 0;

    for(i = 0; i < (height >> 1); i++) {
        for(j = 0; j <= width - 16; j += 16) {
            index = i * cum_array_width + (j >> 1);

            __m512i var_x_cum_1 = _mm512_loadu_si512((__m512i *) (var_x_cum + index_cum));
            __m512i var_y_cum_1 = _mm512_loadu_si512((__m512i *) (var_y_cum + index_cum));
            __m512i cov_xy_cum_1 = _mm512_loadu_si512((__m512i *) (cov_xy_cum + index_cum));

            __m512i var_x_cum_2 =
                _mm512_loadu_si512((__m512i *) (var_x_cum + cum_array_width + index_cum));
            __m512i var_y_cum_2 =
                _mm512_loadu_si512((__m512i *) (var_y_cum + cum_array_width + index_cum));
            __m512i cov_xy_cum_2 =
                _mm512_loadu_si512((__m512i *) (cov_xy_cum + cum_array_width + index_cum));

            __m512i var_x_cum_sum = _mm512_add_epi32(var_x_cum_1, var_x_cum_2);
            __m512i var_y_cum_sum = _mm512_add_epi32(var_y_cum_1, var_y_cum_2);
            __m512i cov_xy_cum_sum = _mm512_add_epi32(cov_xy_cum_1, cov_xy_cum_2);

            __m512i var_x_cum_sum_reverse = _mm512_permutexvar_epi32(shuffleMask, var_x_cum_sum);

            __m512i var_y_cum_sum_reverse = _mm512_permutexvar_epi32(shuffleMask, var_y_cum_sum);

            __m512i cov_xy_cum_sum_reverse = _mm512_permutexvar_epi32(shuffleMask, cov_xy_cum_sum);

            __m512i var_x_cum_inter = _mm512_permutexvar_epi32(
                permIndex,
                _mm512_srai_epi32(
                    _mm512_add_epi32(_mm512_add_epi32(var_x_cum_sum, var_x_cum_sum_reverse),
                                     vec_constant),
                    2));

            __m512i var_y_cum_inter = _mm512_permutexvar_epi32(
                permIndex,
                _mm512_srai_epi32(
                    _mm512_add_epi32(_mm512_add_epi32(var_y_cum_sum, var_y_cum_sum_reverse),
                                     vec_constant),
                    2));

            __m512i cov_xy_cum_inter = _mm512_permutexvar_epi32(
                permIndex,
                _mm512_srai_epi32(
                    _mm512_add_epi32(_mm512_add_epi32(cov_xy_cum_sum, cov_xy_cum_sum_reverse),
                                     vec_constant),
                    2));

            _mm256_storeu_si256((__m256i *) (var_x_cum + index),
                                _mm512_extracti32x8_epi32(var_x_cum_inter, 1));
            _mm256_storeu_si256((__m256i *) (var_y_cum + index),
                                _mm512_extracti32x8_epi32(var_y_cum_inter, 1));
            _mm256_storeu_si256((__m256i *) (cov_xy_cum + index),
                                _mm512_extracti32x8_epi32(cov_xy_cum_inter, 1));

            index_cum += 16;
        }
        for(; j < width; j += 2) {
            index = i * cum_array_width + (j >> 1);
            var_x_cum[index] = var_x_cum[index_cum] + var_x_cum[index_cum + 1] +
                               var_x_cum[index_cum + (cum_array_width)] +
                               var_x_cum[index_cum + (cum_array_width) + 1];
            var_x_cum[index] = (var_x_cum[index] + 2) >> 2;

            var_y_cum[index] = var_y_cum[index_cum] + var_y_cum[index_cum + 1] +
                               var_y_cum[index_cum + (cum_array_width)] +
                               var_y_cum[index_cum + (cum_array_width) + 1];
            var_y_cum[index] = (var_y_cum[index] + 2) >> 2;

            cov_xy_cum[index] = cov_xy_cum[index_cum] + cov_xy_cum[index_cum + 1] +
                                cov_xy_cum[index_cum + (cum_array_width)] +
                                cov_xy_cum[index_cum + (cum_array_width) + 1];
            cov_xy_cum[index] = (cov_xy_cum[index] + 2) >> 2;

            index_cum += 2;
        }
        index_cum += ((cum_array_width << 1) - width);
    }
    ret = 0;
    return ret;
}

int integer_ms_ssim_shift_cum_buffer_funque_avx512(int32_t *var_x_cum, int32_t *var_y_cum, int32_t *cov_xy_cum,
                                                   int width, int height, int level, uint8_t csf_pending_div[4],
                                                   uint8_t csf_pending_div_lp1[4])
{
    int cum_array_width = width * (1 << (level + 1));
    int index_cum = 0;
    int shift_cums = 2 * (csf_pending_div[1] - csf_pending_div_lp1[1] - 1);
    int i = 0;
    int j = 0;
    __m512i add_constant = _mm512_set1_epi32(1 << (shift_cums - 1));

    for(i = 0; i < height; i++)
    {
        for(j = 0; j < width - 16; j += 16)
        {
            __m512i var_x_cum_buf = _mm512_loadu_si512((__m512i*) (var_x_cum + index_cum));
            __m512i var_y_cum_buf = _mm512_loadu_si512((__m512i*) (var_y_cum + index_cum));
            __m512i cov_xy_cum_buf = _mm512_loadu_si512((__m512i*) (cov_xy_cum + index_cum));

            __m512i var_x_cum_str = _mm512_srai_epi32(_mm512_add_epi32(var_x_cum_buf, add_constant), shift_cums);
            __m512i var_y_cum_str = _mm512_srai_epi32(_mm512_add_epi32(var_y_cum_buf, add_constant), shift_cums);
            __m512i cov_xy_cum_str = _mm512_srai_epi32(_mm512_add_epi32(cov_xy_cum_buf, add_constant), shift_cums);

            _mm512_store_si512((__m512i*)(var_x_cum + index_cum), var_x_cum_str);
            _mm512_store_si512((__m512i*)(var_y_cum + index_cum), var_y_cum_str);
            _mm512_store_si512((__m512i*)(cov_xy_cum + index_cum), cov_xy_cum_str);

            index_cum += 16;
        }
        for(; j < width; j++)
        {
            var_x_cum[index_cum] = (var_x_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
            var_y_cum[index_cum] = (var_y_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
            cov_xy_cum[index_cum] = (cov_xy_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
            index_cum++;
        }
        index_cum += (cum_array_width - width);
    }
}