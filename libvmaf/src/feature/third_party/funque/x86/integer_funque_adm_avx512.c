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
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "../integer_funque_adm.h"
#include "integer_funque_adm_avx512.h"
#include "integer_funque_adm_avx2.h"
#include "mem.h"
#include "../adm_tools.h"
#include "../integer_funque_filters.h"
#include <immintrin.h>

#define cvt_1_16x16_to_2_32x8_512(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a_16x16)); \
    r_32x8_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(a_16x16, 1)); \
}

#define cvt_1_16x16_to_2_32x8_256(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_16x16)); \
    r_32x8_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a_16x16, 1)); \
}

#define cvt_1_16x8_to_2_32x4_256(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm_cvtepi16_epi32(a_16x16); \
    r_32x8_hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(a_16x16, 0x0E)); \
}

#define shift15_64b_signExt_512(a, r)\
{ \
    r = _mm512_add_epi64( _mm512_srli_epi64(a, 15) , _mm512_and_si512(a, _mm512_set1_epi64(0xFFFE000000000000)));\
}

#define shift15_64b_signExt_256(a, r)\
{ \
    r = _mm256_add_epi64( _mm256_srli_epi64(a, 15) , _mm256_and_si256(a, _mm256_set1_epi64x(0xFFFE000000000000)));\
}

#define shift15_64b_signExt_128(a, r)\
{ \
    r = _mm_add_epi64( _mm_srli_epi64(a, 15) , _mm_and_si128(a, _mm_set1_epi64x(0xFFFE000000000000)));\
} 

void integer_adm_decouple_avx512(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den, float adm_pending_div)
{
    // const float cos_1deg_sq = COS_1DEG_SQ;

    size_t width = ref.width;
    size_t height = ref.height;
    int i, j, k, index, addIndex,restIndex;
    
    adm_i16_dtype tmp_val;
    int angle_flag;
    
    adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
    int border_h = (border_size * height);
    int border_w = (border_size * width);

	double den_sum[3] = {0};
    int64_t den_row_sum[3] = {0};
    int64_t col0_ref_cube[3] = {0};
    int loop_h, loop_w, dlm_width;
	int extra_sample_h = 0, extra_sample_w = 0;

	adm_i64_dtype den_cube[3] = {0};

	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	*/
#if ADM_REFLECT_PAD
    extra_sample_w = 0;
    extra_sample_h = 0;
#else
    extra_sample_w = 1;
    extra_sample_h = 1;

#endif
	
	border_h -= extra_sample_h;
	border_w -= extra_sample_w;

#if !ADM_REFLECT_PAD
    //If reflect pad is disabled & if border_size is 0, process 1 row,col pixels lesser
    border_h = MAX(1,border_h);
    border_w = MAX(1,border_w);
#endif

    loop_h = height - border_h;
    loop_w = width - border_w;
#if ADM_REFLECT_PAD
	int dlm_height = height - (border_h << 1);
#endif
	dlm_width = width - (border_w << 1);

	//The width of i_dlm_add buffer will be extra only if padding is enabled
    int dlm_add_w = dlm_width  + (ADM_REFLECT_PAD << 1);

    int loop_w_32 = loop_w - ((loop_w - border_w) % 32);
	int loop_w_16 = loop_w - ((loop_w - border_w) % 16);
    int loop_w_8 = loop_w - ((loop_w - border_w) % 8);

    __m512i perm_64_to_32_512 = _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0);
    __m512i packs_32_512 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    __m512i perm_for_64b_mul_512 = _mm512_set_epi32(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
    __m512i add_16384_512 = _mm512_set1_epi64(16384);
    __m512i add_32768_512 = _mm512_set1_epi64(32768);
    __m512i add_32768_32b_512 = _mm512_set1_epi32(32768);
    __m512i add_16384_32b_512 = _mm512_set1_epi32(16384);
    __m512i zero_512 = _mm512_setzero_si512();

    __m256i perm_64_to_32_256 = _mm256_set_epi32(14, 6, 12, 4, 10, 2, 8, 0);
    __m256i perm_for_64b_mul_256 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i add_16384_256 = _mm256_set1_epi64x(16384);
    __m256i add_32768_256 = _mm256_set1_epi64x(32768);
    __m256i add_32768_32b_256 = _mm256_set1_epi32(32768);
    __m256i add_16384_32b_256 = _mm256_set1_epi32(16384);
    __m256i zero_256 = _mm256_setzero_si256();

    __m128i perm_64_to_32_128 = _mm_set_epi32(6, 2, 4, 0);
    __m128i add_16384_128 = _mm_set1_epi64x(16384);
    __m128i add_32768_128 = _mm_set1_epi64x(32768);
    __m128i add_32768_32b_128 = _mm_set1_epi32(32768);
    __m128i add_16384_32b_128 = _mm_set1_epi32(16384);
    __m128i zero_128 = _mm_setzero_si128();

    for (i = border_h; i < loop_h; i++)
    {
        if(extra_sample_w)
        {
            for(k=1; k<4; k++)
            {
                int16_t ref_abs = abs(ref.bands[k][i*width + border_w]);
                col0_ref_cube[k-1] = (int64_t) ref_abs * ref_abs * ref_abs;
            }
        }
        j = border_w;
        for (; j < loop_w_32; j+=32)
        {
            index = i * width + j;
            //If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;
			restIndex = (i - border_h) * (dlm_width) + j - border_w;

            __m512i ref_b1_512 = _mm512_loadu_si512((__m512i*)(ref.bands[1] + index));
            __m512i dis_b1_512 = _mm512_loadu_si512((__m512i*)(dist.bands[1] + index));
            __m512i ref_b2_512 = _mm512_loadu_si512((__m512i*)(ref.bands[2] + index));
            __m512i dis_b2_512 = _mm512_loadu_si512((__m512i*)(dist.bands[2] + index));

            __m512i ref_b1b2_lo = _mm512_unpacklo_epi16(ref_b1_512, ref_b2_512);
            __m512i ref_b1b2_hi = _mm512_unpackhi_epi16(ref_b1_512, ref_b2_512);
            __m512i dis_b1b2_lo = _mm512_unpacklo_epi16(dis_b1_512, dis_b2_512);
            __m512i dis_b1b2_hi = _mm512_unpackhi_epi16(dis_b1_512, dis_b2_512);

            __m512i ot_dp_lo = _mm512_madd_epi16(ref_b1b2_lo, dis_b1b2_lo);
            __m512i ot_dp_hi = _mm512_madd_epi16(ref_b1b2_hi, dis_b1b2_hi);

            __m512i o_mag_sq_lo = _mm512_madd_epi16(ref_b1b2_lo, ref_b1b2_lo);
            __m512i o_mag_sq_hi = _mm512_madd_epi16(ref_b1b2_hi, ref_b1b2_hi);
            
            __m512i t_mag_sq_lo = _mm512_madd_epi16(dis_b1b2_lo, dis_b1b2_lo);
            __m512i t_mag_sq_hi = _mm512_madd_epi16(dis_b1b2_hi, dis_b1b2_hi);

            ot_dp_lo = _mm512_max_epi32(ot_dp_lo, zero_512);
            ot_dp_hi = _mm512_max_epi32(ot_dp_hi, zero_512);

            __m512i ot_dp_lo_0 = _mm512_mul_epi32(ot_dp_lo, ot_dp_lo);
            __m512i ot_dp_lo_1 = _mm512_mul_epi32(_mm512_srai_epi64(ot_dp_lo, 32), _mm512_srai_epi64(ot_dp_lo, 32));
            __m512i ot_dp_hi_0 = _mm512_mul_epi32(ot_dp_hi, ot_dp_hi);
            __m512i ot_dp_hi_1 = _mm512_mul_epi32(_mm512_srai_epi64(ot_dp_hi, 32), _mm512_srai_epi64(ot_dp_hi, 32));

            __m512i ot_mag_sq_lo_0 = _mm512_mul_epi32(o_mag_sq_lo, t_mag_sq_lo);
            __m512i ot_mag_sq_lo_1 = _mm512_mul_epi32(_mm512_srai_epi64(o_mag_sq_lo, 32), _mm512_srai_epi64(t_mag_sq_lo, 32));
            __m512i ot_mag_sq_hi_0 = _mm512_mul_epi32(o_mag_sq_hi, t_mag_sq_hi);
            __m512i ot_mag_sq_hi_1 = _mm512_mul_epi32(_mm512_srai_epi64(o_mag_sq_hi, 32), _mm512_srai_epi64(t_mag_sq_hi, 32));
            
            __mmask32 angle_mask32 = 0;
            for(int a = 0; a < 8; a+=2)
            {
                int a0 = ((adm_i64_dtype)ot_dp_lo_0[a] >= COS_1DEG_SQ * (adm_i64_dtype)ot_mag_sq_lo_0[a]) << a*4;
                int a2 = (ot_dp_lo_0[a + 1] >= COS_1DEG_SQ * ot_mag_sq_lo_0[a + 1]) << (a*4 + 2);
                int a1 = (ot_dp_lo_1[a] >= COS_1DEG_SQ * ot_mag_sq_lo_1[a]) << (a*4 + 1);
                int a3 = (ot_dp_lo_1[a + 1] >= COS_1DEG_SQ * ot_mag_sq_lo_1[a + 1]) << (a*4 + 3);
                int a4 = (ot_dp_hi_0[a] >= COS_1DEG_SQ * ot_mag_sq_hi_0[a]) << (a*4 + 4);
                int a6 = (ot_dp_hi_0[a + 1] >= COS_1DEG_SQ * ot_mag_sq_hi_0[a + 1]) << (a*4 + 6);
                int a5 = (ot_dp_hi_1[a] >= COS_1DEG_SQ * ot_mag_sq_hi_1[a]) << (a*4 + 5);
                int a7 = (ot_dp_hi_1[a + 1] >= COS_1DEG_SQ * ot_mag_sq_hi_1[a + 1]) << (a*4 + 7);
                angle_mask32 += a0 + a2 + a1 + a3 + a4 + a6 + a5 + a7;
            }

            __m512i dis_b3_512 = _mm512_loadu_si512((__m512i*)(dist.bands[3] + index));
            __m512i ref_b3_512 = _mm512_loadu_si512((__m512i*)(ref.bands[3] + index));

            __m512i ref_b1_lo, ref_b1_hi, ref_b2_lo, ref_b2_hi, ref_b3_lo, ref_b3_hi;
            cvt_1_16x16_to_2_32x8_512(ref_b1_512, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b2_512, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8_512(ref_b3_512, ref_b3_lo, ref_b3_hi);

            __m512i adm_div_b1_lo, adm_div_b1_hi, adm_div_b2_lo, adm_div_b2_hi, adm_div_b3_lo, adm_div_b3_hi;

            adm_div_b1_lo = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b1_lo, add_32768_32b_512), adm_div_lookup, 4);
            adm_div_b1_hi = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b1_hi, add_32768_32b_512), adm_div_lookup, 4);
            adm_div_b2_lo = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b2_lo, add_32768_32b_512), adm_div_lookup, 4);
            adm_div_b2_hi = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b2_hi, add_32768_32b_512), adm_div_lookup, 4);
            adm_div_b3_lo = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b3_lo, add_32768_32b_512), adm_div_lookup, 4);
            adm_div_b3_hi = _mm512_i32gather_epi32(_mm512_add_epi32(ref_b3_hi, add_32768_32b_512), adm_div_lookup, 4);

            __m512i dis_b1_lo, dis_b1_hi, dis_b2_lo, dis_b2_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x16_to_2_32x8_512(dis_b1_512, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b2_512, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8_512(dis_b3_512, dis_b3_lo, dis_b3_hi);

            __m512i adm_b1_dis_lo0 = _mm512_mul_epi32(adm_div_b1_lo, dis_b1_lo);
            __m512i adm_b1_dis_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b1_lo, 32), _mm512_srli_epi64(dis_b1_lo, 32));
            __m512i adm_b1_dis_hi8 = _mm512_mul_epi32(adm_div_b1_hi, dis_b1_hi);
            __m512i adm_b1_dis_hi9 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b1_hi, 32), _mm512_srli_epi64(dis_b1_hi, 32));

            __m512i adm_b2_dis_lo0 = _mm512_mul_epi32(adm_div_b2_lo, dis_b2_lo);
            __m512i adm_b2_dis_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b2_lo, 32), _mm512_srli_epi64(dis_b2_lo, 32));
            __m512i adm_b2_dis_hi8 = _mm512_mul_epi32(adm_div_b2_hi, dis_b2_hi);
            __m512i adm_b2_dis_hi9 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b2_hi, 32), _mm512_srli_epi64(dis_b2_hi, 32));

            __m512i adm_b3_dis_lo0 = _mm512_mul_epi32(adm_div_b3_lo, dis_b3_lo);
            __m512i adm_b3_dis_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b3_lo, 32), _mm512_srli_epi64(dis_b3_lo, 32));
            __m512i adm_b3_dis_hi8 = _mm512_mul_epi32(adm_div_b3_hi, dis_b3_hi);
            __m512i adm_b3_dis_hi9 = _mm512_mul_epi32(_mm512_srli_epi64(adm_div_b3_hi, 32), _mm512_srli_epi64(dis_b3_hi, 32));
 
            adm_b1_dis_lo0 = _mm512_add_epi64(adm_b1_dis_lo0, add_16384_512);
            adm_b1_dis_lo1 = _mm512_add_epi64(adm_b1_dis_lo1, add_16384_512);
            adm_b1_dis_hi8 = _mm512_add_epi64(adm_b1_dis_hi8, add_16384_512);
            adm_b1_dis_hi9 = _mm512_add_epi64(adm_b1_dis_hi9, add_16384_512);
            adm_b2_dis_lo0 = _mm512_add_epi64(adm_b2_dis_lo0, add_16384_512);
            adm_b2_dis_lo1 = _mm512_add_epi64(adm_b2_dis_lo1, add_16384_512);
            adm_b2_dis_hi8 = _mm512_add_epi64(adm_b2_dis_hi8, add_16384_512);
            adm_b2_dis_hi9 = _mm512_add_epi64(adm_b2_dis_hi9, add_16384_512);
            adm_b3_dis_lo0 = _mm512_add_epi64(adm_b3_dis_lo0, add_16384_512);
            adm_b3_dis_lo1 = _mm512_add_epi64(adm_b3_dis_lo1, add_16384_512);
            adm_b3_dis_hi8 = _mm512_add_epi64(adm_b3_dis_hi8, add_16384_512);
            adm_b3_dis_hi9 = _mm512_add_epi64(adm_b3_dis_hi9, add_16384_512);

            shift15_64b_signExt_512(adm_b1_dis_lo0, adm_b1_dis_lo0);
            shift15_64b_signExt_512(adm_b1_dis_lo1, adm_b1_dis_lo1);
            shift15_64b_signExt_512(adm_b1_dis_hi8, adm_b1_dis_hi8);
            shift15_64b_signExt_512(adm_b1_dis_hi9, adm_b1_dis_hi9);
            shift15_64b_signExt_512(adm_b2_dis_lo0, adm_b2_dis_lo0);
            shift15_64b_signExt_512(adm_b2_dis_lo1, adm_b2_dis_lo1);
            shift15_64b_signExt_512(adm_b2_dis_hi8, adm_b2_dis_hi8);
            shift15_64b_signExt_512(adm_b2_dis_hi9, adm_b2_dis_hi9);
            shift15_64b_signExt_512(adm_b3_dis_lo0, adm_b3_dis_lo0);
            shift15_64b_signExt_512(adm_b3_dis_lo1, adm_b3_dis_lo1);
            shift15_64b_signExt_512(adm_b3_dis_hi8, adm_b3_dis_hi8);
            shift15_64b_signExt_512(adm_b3_dis_hi9, adm_b3_dis_hi9);

            __mmask16 eqz_b1_lo = _mm512_cmpeq_epi32_mask(ref_b1_lo, zero_512);
            __mmask16 eqz_b1_hi = _mm512_cmpeq_epi32_mask(ref_b1_hi, zero_512);
            __mmask16 eqz_b2_lo = _mm512_cmpeq_epi32_mask(ref_b2_lo, zero_512);
            __mmask16 eqz_b2_hi = _mm512_cmpeq_epi32_mask(ref_b2_hi, zero_512);
            __mmask16 eqz_b3_lo = _mm512_cmpeq_epi32_mask(ref_b3_lo, zero_512);
            __mmask16 eqz_b3_hi = _mm512_cmpeq_epi32_mask(ref_b3_hi, zero_512);

            __m512i adm_b1_dis_lo = _mm512_permutex2var_epi32(adm_b1_dis_lo0, perm_64_to_32_512, adm_b1_dis_lo1);
            __m512i adm_b1_dis_hi = _mm512_permutex2var_epi32(adm_b1_dis_hi8, perm_64_to_32_512, adm_b1_dis_hi9);
            __m512i adm_b2_dis_lo = _mm512_permutex2var_epi32(adm_b2_dis_lo0, perm_64_to_32_512, adm_b2_dis_lo1);
            __m512i adm_b2_dis_hi = _mm512_permutex2var_epi32(adm_b2_dis_hi8, perm_64_to_32_512, adm_b2_dis_hi9);
            __m512i adm_b3_dis_lo = _mm512_permutex2var_epi32(adm_b3_dis_lo0, perm_64_to_32_512, adm_b3_dis_lo1);
            __m512i adm_b3_dis_hi = _mm512_permutex2var_epi32(adm_b3_dis_hi8, perm_64_to_32_512, adm_b3_dis_hi9);

            __m512i tmp_k_b1_lo = _mm512_mask_blend_epi32(eqz_b1_lo, adm_b1_dis_lo, add_32768_512);
            __m512i tmp_k_b1_hi = _mm512_mask_blend_epi32(eqz_b1_hi, adm_b1_dis_hi, add_32768_512);
            __m512i tmp_k_b2_lo = _mm512_mask_blend_epi32(eqz_b2_lo, adm_b2_dis_lo, add_32768_512);
            __m512i tmp_k_b2_hi = _mm512_mask_blend_epi32(eqz_b2_hi, adm_b2_dis_hi, add_32768_512);
            __m512i tmp_k_b3_lo = _mm512_mask_blend_epi32(eqz_b3_lo, adm_b3_dis_lo, add_32768_512);
            __m512i tmp_k_b3_hi = _mm512_mask_blend_epi32(eqz_b3_hi, adm_b3_dis_hi, add_32768_512);

            tmp_k_b1_lo = _mm512_max_epi32(tmp_k_b1_lo, zero_512);
            tmp_k_b1_hi = _mm512_max_epi32(tmp_k_b1_hi, zero_512);
            tmp_k_b2_lo = _mm512_max_epi32(tmp_k_b2_lo, zero_512);
            tmp_k_b2_hi = _mm512_max_epi32(tmp_k_b2_hi, zero_512);
            tmp_k_b3_lo = _mm512_max_epi32(tmp_k_b3_lo, zero_512);
            tmp_k_b3_hi = _mm512_max_epi32(tmp_k_b3_hi, zero_512);

            tmp_k_b1_lo = _mm512_min_epi32(tmp_k_b1_lo, add_32768_32b_512);
            tmp_k_b1_hi = _mm512_min_epi32(tmp_k_b1_hi, add_32768_32b_512);
            tmp_k_b2_lo = _mm512_min_epi32(tmp_k_b2_lo, add_32768_32b_512);
            tmp_k_b2_hi = _mm512_min_epi32(tmp_k_b2_hi, add_32768_32b_512);
            tmp_k_b3_lo = _mm512_min_epi32(tmp_k_b3_lo, add_32768_32b_512);
            tmp_k_b3_hi = _mm512_min_epi32(tmp_k_b3_hi, add_32768_32b_512);

            __m512i tmp_val_b1_lo = _mm512_mullo_epi32(tmp_k_b1_lo, ref_b1_lo);
            __m512i tmp_val_b1_hi = _mm512_mullo_epi32(tmp_k_b1_hi, ref_b1_hi);
            __m512i tmp_val_b2_lo = _mm512_mullo_epi32(tmp_k_b2_lo, ref_b2_lo);
            __m512i tmp_val_b2_hi = _mm512_mullo_epi32(tmp_k_b2_hi, ref_b2_hi);
            __m512i tmp_val_b3_lo = _mm512_mullo_epi32(tmp_k_b3_lo, ref_b3_lo);
            __m512i tmp_val_b3_hi = _mm512_mullo_epi32(tmp_k_b3_hi, ref_b3_hi);

            tmp_val_b1_lo = _mm512_add_epi32(tmp_val_b1_lo, add_16384_32b_512);
            tmp_val_b1_hi = _mm512_add_epi32(tmp_val_b1_hi, add_16384_32b_512);
            tmp_val_b2_lo = _mm512_add_epi32(tmp_val_b2_lo, add_16384_32b_512);
            tmp_val_b3_lo = _mm512_add_epi32(tmp_val_b3_lo, add_16384_32b_512);
            tmp_val_b2_hi = _mm512_add_epi32(tmp_val_b2_hi, add_16384_32b_512);
            tmp_val_b3_hi = _mm512_add_epi32(tmp_val_b3_hi, add_16384_32b_512);

            tmp_val_b1_lo = _mm512_srai_epi32(tmp_val_b1_lo, 15);
            tmp_val_b1_hi = _mm512_srai_epi32(tmp_val_b1_hi, 15);
            tmp_val_b2_lo = _mm512_srai_epi32(tmp_val_b2_lo, 15);
            tmp_val_b2_hi = _mm512_srai_epi32(tmp_val_b2_hi, 15);
            tmp_val_b3_lo = _mm512_srai_epi32(tmp_val_b3_lo, 15);
            tmp_val_b3_hi = _mm512_srai_epi32(tmp_val_b3_hi, 15);

            __m512i tmp_val_b1 = _mm512_packs_epi32(tmp_val_b1_lo, tmp_val_b1_hi);
            __m512i tmp_val_b2 = _mm512_packs_epi32(tmp_val_b2_lo, tmp_val_b2_hi);
            __m512i tmp_val_b3 = _mm512_packs_epi32(tmp_val_b3_lo, tmp_val_b3_hi);

            tmp_val_b1 = _mm512_permutexvar_epi64(packs_32_512, tmp_val_b1);
            tmp_val_b2 = _mm512_permutexvar_epi64(packs_32_512, tmp_val_b2);
            tmp_val_b3 = _mm512_permutexvar_epi64(packs_32_512, tmp_val_b3);
                       
            __m512i dlm_rest_b1_512 = _mm512_mask_blend_epi16(angle_mask32, tmp_val_b1, dis_b1_512);
            __m512i dlm_rest_b2_512 = _mm512_mask_blend_epi16(angle_mask32, tmp_val_b2, dis_b2_512);
            __m512i dlm_rest_b3_512 = _mm512_mask_blend_epi16(angle_mask32, tmp_val_b3, dis_b3_512);

            __m512i dist_m_dlm_rest_b1 = _mm512_abs_epi16(_mm512_sub_epi16(dis_b1_512, dlm_rest_b1_512));
            __m512i dist_m_dlm_rest_b2 = _mm512_abs_epi16(_mm512_sub_epi16(dis_b2_512, dlm_rest_b2_512));
            __m512i dlm_add_512 = _mm512_adds_epu16(dist_m_dlm_rest_b1, dist_m_dlm_rest_b2);
            __m512i dist_m_dlm_rest_b3 = _mm512_abs_epi16(_mm512_sub_epi16(dis_b3_512, dlm_rest_b3_512));            
            dlm_add_512 = _mm512_adds_epu16(dlm_add_512, dist_m_dlm_rest_b3);

            _mm512_storeu_si512((__m512i*)(i_dlm_rest.bands[1] + restIndex), dlm_rest_b1_512);
            _mm512_storeu_si512((__m512i*)(i_dlm_rest.bands[2] + restIndex), dlm_rest_b2_512);
            _mm512_storeu_si512((__m512i*)(i_dlm_rest.bands[3] + restIndex), dlm_rest_b3_512);

            __m512i dlm_add_lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(dlm_add_512));
            __m512i dlm_add_hi = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(dlm_add_512, 1));

            ref_b1_lo = _mm512_abs_epi32(ref_b1_lo);
            ref_b1_hi = _mm512_abs_epi32(ref_b1_hi);
            ref_b2_lo = _mm512_abs_epi32(ref_b2_lo);
            ref_b2_hi = _mm512_abs_epi32(ref_b2_hi);
            ref_b3_lo = _mm512_abs_epi32(ref_b3_lo);
            ref_b3_hi = _mm512_abs_epi32(ref_b3_hi);

            _mm512_storeu_si512((__m512i*)(i_dlm_add + addIndex), dlm_add_lo);
            _mm512_storeu_si512((__m512i*)(i_dlm_add + addIndex + 16), dlm_add_hi);

            __m512i ref_b_ref_b1_lo = _mm512_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m512i ref_b_ref_b1_hi = _mm512_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m512i ref_b_ref_b2_lo = _mm512_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m512i ref_b_ref_b2_hi = _mm512_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m512i ref_b_ref_b3_lo = _mm512_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m512i ref_b_ref_b3_hi = _mm512_mullo_epi32(ref_b3_hi, ref_b3_hi);

            ref_b_ref_b1_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b1_lo);
            ref_b1_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b1_lo);
            ref_b_ref_b1_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b1_hi);
            ref_b1_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b1_hi);
            ref_b_ref_b2_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b2_lo);
            ref_b2_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b2_lo);
            ref_b_ref_b2_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b2_hi);
            ref_b2_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b2_hi);
            ref_b_ref_b3_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b3_lo);
            ref_b3_lo = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b3_lo);
            ref_b_ref_b3_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b_ref_b3_hi);
            ref_b3_hi = _mm512_permutexvar_epi32(perm_for_64b_mul_512, ref_b3_hi);

            __m512i ref_b_ref_b1_lo0 = _mm512_mul_epi32(ref_b_ref_b1_lo, ref_b1_lo);
            __m512i ref_b_ref_b1_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b1_lo, 32), _mm512_srli_epi64(ref_b1_lo, 32));
            __m512i ref_b_ref_b1_hi0 = _mm512_mul_epi32(ref_b_ref_b1_hi, ref_b1_hi);
            __m512i ref_b_ref_b1_hi1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b1_hi, 32), _mm512_srli_epi64(ref_b1_hi, 32));
            __m512i ref_b_ref_b2_lo0 = _mm512_mul_epi32(ref_b_ref_b2_lo, ref_b2_lo);
            __m512i ref_b_ref_b2_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b2_lo, 32), _mm512_srli_epi64(ref_b2_lo, 32));
            __m512i ref_b_ref_b2_hi0 = _mm512_mul_epi32(ref_b_ref_b2_hi, ref_b2_hi);
            __m512i ref_b_ref_b2_hi1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b2_hi, 32), _mm512_srli_epi64(ref_b2_hi, 32));
            __m512i ref_b_ref_b3_lo0 = _mm512_mul_epi32(ref_b_ref_b3_lo, ref_b3_lo);
            __m512i ref_b_ref_b3_lo1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b3_lo, 32), _mm512_srli_epi64(ref_b3_lo, 32));
            __m512i ref_b_ref_b3_hi0 = _mm512_mul_epi32(ref_b_ref_b3_hi, ref_b3_hi);
            __m512i ref_b_ref_b3_hi1 = _mm512_mul_epi32(_mm512_srli_epi64(ref_b_ref_b3_hi, 32), _mm512_srli_epi64(ref_b3_hi, 32));

            __m512i b1_r8_lo = _mm512_add_epi64(ref_b_ref_b1_lo0, ref_b_ref_b1_lo1);
            __m512i b1_r8_hi = _mm512_add_epi64(ref_b_ref_b1_hi0, ref_b_ref_b1_hi1);
            __m512i b2_r8_lo = _mm512_add_epi64(ref_b_ref_b2_lo0, ref_b_ref_b2_lo1);
            __m512i b2_r8_hi = _mm512_add_epi64(ref_b_ref_b2_hi0, ref_b_ref_b2_hi1);
            __m512i b3_r8_lo = _mm512_add_epi64(ref_b_ref_b3_lo0, ref_b_ref_b3_lo1);
            __m512i b3_r8_hi = _mm512_add_epi64(ref_b_ref_b3_hi0, ref_b_ref_b3_hi1);
            __m512i b1_r8 = _mm512_add_epi64(b1_r8_lo, b1_r8_hi);
            __m512i b2_r8 = _mm512_add_epi64(b2_r8_lo, b2_r8_hi);
            __m512i b3_r8 = _mm512_add_epi64(b3_r8_lo, b3_r8_hi);

            __m256i b1_r4 = _mm256_add_epi64(_mm512_castsi512_si256(b1_r8), _mm512_extracti64x4_epi64(b1_r8, 1));
            __m256i b2_r4 = _mm256_add_epi64(_mm512_castsi512_si256(b2_r8), _mm512_extracti64x4_epi64(b2_r8, 1));
            __m256i b3_r4 = _mm256_add_epi64(_mm512_castsi512_si256(b3_r8), _mm512_extracti64x4_epi64(b3_r8, 1));

            __m128i b1_r2 = _mm_add_epi64(_mm256_castsi256_si128(b1_r4), _mm256_extractf128_si256(b1_r4, 1));
            __m128i b2_r2 = _mm_add_epi64(_mm256_castsi256_si128(b2_r4), _mm256_extractf128_si256(b2_r4, 1));
            __m128i b3_r2 = _mm_add_epi64(_mm256_castsi256_si128(b3_r4), _mm256_extractf128_si256(b3_r4, 1));

            int64_t r_b1 = _mm_extract_epi64(b1_r2, 0) + _mm_extract_epi64(b1_r2, 1);
            int64_t r_b2 = _mm_extract_epi64(b2_r2, 0) + _mm_extract_epi64(b2_r2, 1);
            int64_t r_b3 = _mm_extract_epi64(b3_r2, 0) + _mm_extract_epi64(b3_r2, 1);

            den_row_sum[0] += r_b1;
            den_row_sum[1] += r_b2;
            den_row_sum[2] += r_b3;
       }

        for (; j < loop_w_16; j+=16)
        {
            index = i * width + j;

            //If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;
			restIndex = (i - border_h) * (dlm_width) + j - border_w;

            __m256i ref_b1_256 = _mm256_loadu_si256((__m256i*)(ref.bands[1] + index));
            __m256i dis_b1_256 = _mm256_loadu_si256((__m256i*)(dist.bands[1] + index));
            __m256i ref_b2_256 = _mm256_loadu_si256((__m256i*)(ref.bands[2] + index));
            __m256i dis_b2_256 = _mm256_loadu_si256((__m256i*)(dist.bands[2] + index));

            __m256i ref_b1b2_lo = _mm256_unpacklo_epi16(ref_b1_256, ref_b2_256);
            __m256i ref_b1b2_hi = _mm256_unpackhi_epi16(ref_b1_256, ref_b2_256);
            __m256i dis_b1b2_lo = _mm256_unpacklo_epi16(dis_b1_256, dis_b2_256);
            __m256i dis_b1b2_hi = _mm256_unpackhi_epi16(dis_b1_256, dis_b2_256);

            __m256i ot_dp_lo = _mm256_madd_epi16(ref_b1b2_lo, dis_b1b2_lo);
            __m256i ot_dp_hi = _mm256_madd_epi16(ref_b1b2_hi, dis_b1b2_hi);

            __m256i o_mag_sq_lo = _mm256_madd_epi16(ref_b1b2_lo, ref_b1b2_lo);
            __m256i o_mag_sq_hi = _mm256_madd_epi16(ref_b1b2_hi, ref_b1b2_hi);
            
            __m256i t_mag_sq_lo = _mm256_madd_epi16(dis_b1b2_lo, dis_b1b2_lo);
            __m256i t_mag_sq_hi = _mm256_madd_epi16(dis_b1b2_hi, dis_b1b2_hi);

            ot_dp_lo = _mm256_max_epi32(ot_dp_lo, zero_256);
            ot_dp_hi = _mm256_max_epi32(ot_dp_hi, zero_256);

            __m256i ot_dp_lo_0 = _mm256_mul_epi32(ot_dp_lo, ot_dp_lo);
            __m256i ot_dp_lo_1 = _mm256_mul_epi32(_mm256_srai_epi64(ot_dp_lo, 32), _mm256_srai_epi64(ot_dp_lo, 32));
            __m256i ot_dp_hi_0 = _mm256_mul_epi32(ot_dp_hi, ot_dp_hi);
            __m256i ot_dp_hi_1 = _mm256_mul_epi32(_mm256_srai_epi64(ot_dp_hi, 32), _mm256_srai_epi64(ot_dp_hi, 32));

            __m256i ot_mag_sq_lo_0 = _mm256_mul_epi32(o_mag_sq_lo, t_mag_sq_lo);
            __m256i ot_mag_sq_lo_1 = _mm256_mul_epi32(_mm256_srai_epi64(o_mag_sq_lo, 32), _mm256_srai_epi64(t_mag_sq_lo, 32));
            __m256i ot_mag_sq_hi_0 = _mm256_mul_epi32(o_mag_sq_hi, t_mag_sq_hi);
            __m256i ot_mag_sq_hi_1 = _mm256_mul_epi32(_mm256_srai_epi64(o_mag_sq_hi, 32), _mm256_srai_epi64(t_mag_sq_hi, 32));
            
            __mmask16 angle_mask16 = 0;
            for(int a = 0; a < 4; a+=2)
            {
                int a0 = ((adm_i64_dtype)ot_dp_lo_0[a] >= COS_1DEG_SQ * (adm_i64_dtype)ot_mag_sq_lo_0[a]) << a*4;
                int a2 = (ot_dp_lo_0[a + 1] >= COS_1DEG_SQ * ot_mag_sq_lo_0[a + 1]) << (a*4 + 2);
                int a1 = (ot_dp_lo_1[a] >= COS_1DEG_SQ * ot_mag_sq_lo_1[a]) << (a*4 + 1);
                int a3 = (ot_dp_lo_1[a + 1] >= COS_1DEG_SQ * ot_mag_sq_lo_1[a + 1]) << (a*4 + 3);
                int a4 = (ot_dp_hi_0[a] >= COS_1DEG_SQ * ot_mag_sq_hi_0[a]) << (a*4 + 4);
                int a6 = (ot_dp_hi_0[a + 1] >= COS_1DEG_SQ * ot_mag_sq_hi_0[a + 1]) << (a*4 + 6);
                int a5 = (ot_dp_hi_1[a] >= COS_1DEG_SQ * ot_mag_sq_hi_1[a]) << (a*4 + 5);
                int a7 = (ot_dp_hi_1[a + 1] >= COS_1DEG_SQ * ot_mag_sq_hi_1[a + 1]) << (a*4 + 7);
                angle_mask16 += a0 + a2 + a1 + a3 + a4 + a6 + a5 + a7;
            }
            
            __m256i dis_b3_256 = _mm256_loadu_si256((__m256i*)(dist.bands[3] + index));
            __m256i ref_b3_256 = _mm256_loadu_si256((__m256i*)(ref.bands[3] + index));

            __m256i ref_b1_lo, ref_b1_hi, ref_b2_lo, ref_b2_hi, ref_b3_lo, ref_b3_hi;
            cvt_1_16x16_to_2_32x8_256(ref_b1_256, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8_256(ref_b2_256, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8_256(ref_b3_256, ref_b3_lo, ref_b3_hi);

            __m256i adm_div_b1_lo, adm_div_b1_hi, adm_div_b2_lo, adm_div_b2_hi, adm_div_b3_lo, adm_div_b3_hi;
            adm_div_b1_lo = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b1_lo, add_32768_32b_256), adm_div_lookup, 4);
            adm_div_b1_hi = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b1_hi, add_32768_32b_256), adm_div_lookup, 4);
            adm_div_b2_lo = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b2_lo, add_32768_32b_256), adm_div_lookup, 4);
            adm_div_b2_hi = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b2_hi, add_32768_32b_256), adm_div_lookup, 4);
            adm_div_b3_lo = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b3_lo, add_32768_32b_256), adm_div_lookup, 4);
            adm_div_b3_hi = _mm256_mmask_i32gather_epi32(zero_256, 0xFF, _mm256_add_epi32(ref_b3_hi, add_32768_32b_256), adm_div_lookup, 4);

            __m256i dis_b1_lo, dis_b1_hi, dis_b2_lo, dis_b2_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x16_to_2_32x8_256(dis_b1_256, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b2_256, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8_256(dis_b3_256, dis_b3_lo, dis_b3_hi);

            __m256i adm_b1_dis_lo0 = _mm256_mul_epi32(adm_div_b1_lo, dis_b1_lo);
            __m256i adm_b1_dis_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b1_lo, 32), _mm256_srli_epi64(dis_b1_lo, 32));
            __m256i adm_b1_dis_hi8 = _mm256_mul_epi32(adm_div_b1_hi, dis_b1_hi);
            __m256i adm_b1_dis_hi9 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b1_hi, 32), _mm256_srli_epi64(dis_b1_hi, 32));

            __m256i adm_b2_dis_lo0 = _mm256_mul_epi32(adm_div_b2_lo, dis_b2_lo);
            __m256i adm_b2_dis_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b2_lo, 32), _mm256_srli_epi64(dis_b2_lo, 32));
            __m256i adm_b2_dis_hi8 = _mm256_mul_epi32(adm_div_b2_hi, dis_b2_hi);
            __m256i adm_b2_dis_hi9 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b2_hi, 32), _mm256_srli_epi64(dis_b2_hi, 32));

            __m256i adm_b3_dis_lo0 = _mm256_mul_epi32(adm_div_b3_lo, dis_b3_lo);
            __m256i adm_b3_dis_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b3_lo, 32), _mm256_srli_epi64(dis_b3_lo, 32));
            __m256i adm_b3_dis_hi8 = _mm256_mul_epi32(adm_div_b3_hi, dis_b3_hi);
            __m256i adm_b3_dis_hi9 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b3_hi, 32), _mm256_srli_epi64(dis_b3_hi, 32));
 
            adm_b1_dis_lo0 = _mm256_add_epi64(adm_b1_dis_lo0, add_16384_256);
            adm_b1_dis_lo1 = _mm256_add_epi64(adm_b1_dis_lo1, add_16384_256);
            adm_b1_dis_hi8 = _mm256_add_epi64(adm_b1_dis_hi8, add_16384_256);
            adm_b1_dis_hi9 = _mm256_add_epi64(adm_b1_dis_hi9, add_16384_256);
            adm_b2_dis_lo0 = _mm256_add_epi64(adm_b2_dis_lo0, add_16384_256);
            adm_b2_dis_lo1 = _mm256_add_epi64(adm_b2_dis_lo1, add_16384_256);
            adm_b2_dis_hi8 = _mm256_add_epi64(adm_b2_dis_hi8, add_16384_256);
            adm_b2_dis_hi9 = _mm256_add_epi64(adm_b2_dis_hi9, add_16384_256);
            adm_b3_dis_lo0 = _mm256_add_epi64(adm_b3_dis_lo0, add_16384_256);
            adm_b3_dis_lo1 = _mm256_add_epi64(adm_b3_dis_lo1, add_16384_256);
            adm_b3_dis_hi8 = _mm256_add_epi64(adm_b3_dis_hi8, add_16384_256);
            adm_b3_dis_hi9 = _mm256_add_epi64(adm_b3_dis_hi9, add_16384_256);

            shift15_64b_signExt_256(adm_b1_dis_lo0, adm_b1_dis_lo0);
            shift15_64b_signExt_256(adm_b1_dis_lo1, adm_b1_dis_lo1);
            shift15_64b_signExt_256(adm_b1_dis_hi8, adm_b1_dis_hi8);
            shift15_64b_signExt_256(adm_b1_dis_hi9, adm_b1_dis_hi9);
            shift15_64b_signExt_256(adm_b2_dis_lo0, adm_b2_dis_lo0);
            shift15_64b_signExt_256(adm_b2_dis_lo1, adm_b2_dis_lo1);
            shift15_64b_signExt_256(adm_b2_dis_hi8, adm_b2_dis_hi8);
            shift15_64b_signExt_256(adm_b2_dis_hi9, adm_b2_dis_hi9);
            shift15_64b_signExt_256(adm_b3_dis_lo0, adm_b3_dis_lo0);
            shift15_64b_signExt_256(adm_b3_dis_lo1, adm_b3_dis_lo1);
            shift15_64b_signExt_256(adm_b3_dis_hi8, adm_b3_dis_hi8);
            shift15_64b_signExt_256(adm_b3_dis_hi9, adm_b3_dis_hi9);
                        
            __mmask8 eqz_b1_lo = _mm256_cmpeq_epi32_mask(ref_b1_lo, zero_256);
            __mmask8 eqz_b1_hi = _mm256_cmpeq_epi32_mask(ref_b1_hi, zero_256);
            __mmask8 eqz_b2_lo = _mm256_cmpeq_epi32_mask(ref_b2_lo, zero_256);
            __mmask8 eqz_b2_hi = _mm256_cmpeq_epi32_mask(ref_b2_hi, zero_256);
            __mmask8 eqz_b3_lo = _mm256_cmpeq_epi32_mask(ref_b3_lo, zero_256);
            __mmask8 eqz_b3_hi = _mm256_cmpeq_epi32_mask(ref_b3_hi, zero_256);

            __m256i adm_b1_dis_lo = _mm256_permutex2var_epi32(adm_b1_dis_lo0, perm_64_to_32_256, adm_b1_dis_lo1);
            __m256i adm_b1_dis_hi = _mm256_permutex2var_epi32(adm_b1_dis_hi8, perm_64_to_32_256, adm_b1_dis_hi9);
            __m256i adm_b2_dis_lo = _mm256_permutex2var_epi32(adm_b2_dis_lo0, perm_64_to_32_256, adm_b2_dis_lo1);
            __m256i adm_b2_dis_hi = _mm256_permutex2var_epi32(adm_b2_dis_hi8, perm_64_to_32_256, adm_b2_dis_hi9);
            __m256i adm_b3_dis_lo = _mm256_permutex2var_epi32(adm_b3_dis_lo0, perm_64_to_32_256, adm_b3_dis_lo1);
            __m256i adm_b3_dis_hi = _mm256_permutex2var_epi32(adm_b3_dis_hi8, perm_64_to_32_256, adm_b3_dis_hi9);

            __m256i tmp_k_b1_lo = _mm256_mask_blend_epi32(eqz_b1_lo, adm_b1_dis_lo, add_32768_256);
            __m256i tmp_k_b1_hi = _mm256_mask_blend_epi32(eqz_b1_hi, adm_b1_dis_hi, add_32768_256);
            __m256i tmp_k_b2_lo = _mm256_mask_blend_epi32(eqz_b2_lo, adm_b2_dis_lo, add_32768_256);
            __m256i tmp_k_b2_hi = _mm256_mask_blend_epi32(eqz_b2_hi, adm_b2_dis_hi, add_32768_256);
            __m256i tmp_k_b3_lo = _mm256_mask_blend_epi32(eqz_b3_lo, adm_b3_dis_lo, add_32768_256);
            __m256i tmp_k_b3_hi = _mm256_mask_blend_epi32(eqz_b3_hi, adm_b3_dis_hi, add_32768_256);

            tmp_k_b1_lo = _mm256_max_epi32(tmp_k_b1_lo, zero_256);
            tmp_k_b1_hi = _mm256_max_epi32(tmp_k_b1_hi, zero_256);
            tmp_k_b2_lo = _mm256_max_epi32(tmp_k_b2_lo, zero_256);
            tmp_k_b2_hi = _mm256_max_epi32(tmp_k_b2_hi, zero_256);
            tmp_k_b3_lo = _mm256_max_epi32(tmp_k_b3_lo, zero_256);
            tmp_k_b3_hi = _mm256_max_epi32(tmp_k_b3_hi, zero_256);

            tmp_k_b1_lo = _mm256_min_epi32(tmp_k_b1_lo, add_32768_32b_256);
            tmp_k_b1_hi = _mm256_min_epi32(tmp_k_b1_hi, add_32768_32b_256);
            tmp_k_b2_lo = _mm256_min_epi32(tmp_k_b2_lo, add_32768_32b_256);
            tmp_k_b2_hi = _mm256_min_epi32(tmp_k_b2_hi, add_32768_32b_256);
            tmp_k_b3_lo = _mm256_min_epi32(tmp_k_b3_lo, add_32768_32b_256);
            tmp_k_b3_hi = _mm256_min_epi32(tmp_k_b3_hi, add_32768_32b_256);

            __m256i tmp_val_b1_lo = _mm256_mullo_epi32(tmp_k_b1_lo, ref_b1_lo);
            __m256i tmp_val_b1_hi = _mm256_mullo_epi32(tmp_k_b1_hi, ref_b1_hi);
            __m256i tmp_val_b2_lo = _mm256_mullo_epi32(tmp_k_b2_lo, ref_b2_lo);
            __m256i tmp_val_b2_hi = _mm256_mullo_epi32(tmp_k_b2_hi, ref_b2_hi);
            __m256i tmp_val_b3_lo = _mm256_mullo_epi32(tmp_k_b3_lo, ref_b3_lo);
            __m256i tmp_val_b3_hi = _mm256_mullo_epi32(tmp_k_b3_hi, ref_b3_hi);

            tmp_val_b1_lo = _mm256_add_epi32(tmp_val_b1_lo, add_16384_32b_256);
            tmp_val_b1_hi = _mm256_add_epi32(tmp_val_b1_hi, add_16384_32b_256);
            tmp_val_b2_lo = _mm256_add_epi32(tmp_val_b2_lo, add_16384_32b_256);
            tmp_val_b3_lo = _mm256_add_epi32(tmp_val_b3_lo, add_16384_32b_256);
            tmp_val_b2_hi = _mm256_add_epi32(tmp_val_b2_hi, add_16384_32b_256);
            tmp_val_b3_hi = _mm256_add_epi32(tmp_val_b3_hi, add_16384_32b_256);

            tmp_val_b1_lo = _mm256_srai_epi32(tmp_val_b1_lo, 15);
            tmp_val_b1_hi = _mm256_srai_epi32(tmp_val_b1_hi, 15);
            tmp_val_b2_lo = _mm256_srai_epi32(tmp_val_b2_lo, 15);
            tmp_val_b2_hi = _mm256_srai_epi32(tmp_val_b2_hi, 15);
            tmp_val_b3_lo = _mm256_srai_epi32(tmp_val_b3_lo, 15);
            tmp_val_b3_hi = _mm256_srai_epi32(tmp_val_b3_hi, 15);

            __m256i tmp_val_b1 = _mm256_packs_epi32(tmp_val_b1_lo, tmp_val_b1_hi);
            __m256i tmp_val_b2 = _mm256_packs_epi32(tmp_val_b2_lo, tmp_val_b2_hi);
            __m256i tmp_val_b3 = _mm256_packs_epi32(tmp_val_b3_lo, tmp_val_b3_hi);
            tmp_val_b1 = _mm256_permute4x64_epi64(tmp_val_b1, 0xD8);
            tmp_val_b2 = _mm256_permute4x64_epi64(tmp_val_b2, 0xD8);
            tmp_val_b3 = _mm256_permute4x64_epi64(tmp_val_b3, 0xD8);

            __m256i dlm_rest_b1_256 = _mm256_mask_blend_epi16(angle_mask16, tmp_val_b1, dis_b1_256);
            __m256i dlm_rest_b2_256 = _mm256_mask_blend_epi16(angle_mask16, tmp_val_b2, dis_b2_256);
            __m256i dlm_rest_b3_256 = _mm256_mask_blend_epi16(angle_mask16, tmp_val_b3, dis_b3_256);

            __m256i dist_m_dlm_rest_b1 = _mm256_abs_epi16(_mm256_sub_epi16(dis_b1_256, dlm_rest_b1_256));
            __m256i dist_m_dlm_rest_b2 = _mm256_abs_epi16(_mm256_sub_epi16(dis_b2_256, dlm_rest_b2_256));
            __m256i dlm_add_256 = _mm256_adds_epu16(dist_m_dlm_rest_b1, dist_m_dlm_rest_b2);
            __m256i dist_m_dlm_rest_b3 = _mm256_abs_epi16(_mm256_sub_epi16(dis_b3_256, dlm_rest_b3_256));            
            dlm_add_256 = _mm256_adds_epu16(dlm_add_256, dist_m_dlm_rest_b3);

            _mm256_storeu_si256((__m256i*)(i_dlm_rest.bands[1] + restIndex), dlm_rest_b1_256);
            _mm256_storeu_si256((__m256i*)(i_dlm_rest.bands[2] + restIndex), dlm_rest_b2_256);
            _mm256_storeu_si256((__m256i*)(i_dlm_rest.bands[3] + restIndex), dlm_rest_b3_256);

            __m256i dlm_add_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dlm_add_256));
            __m256i dlm_add_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dlm_add_256, 1));

            ref_b1_lo = _mm256_abs_epi32(ref_b1_lo);
            ref_b1_hi = _mm256_abs_epi32(ref_b1_hi);
            ref_b2_lo = _mm256_abs_epi32(ref_b2_lo);
            ref_b2_hi = _mm256_abs_epi32(ref_b2_hi);
            ref_b3_lo = _mm256_abs_epi32(ref_b3_lo);
            ref_b3_hi = _mm256_abs_epi32(ref_b3_hi);

            _mm256_storeu_si256((__m256i*)(i_dlm_add + addIndex), dlm_add_lo);
            _mm256_storeu_si256((__m256i*)(i_dlm_add + addIndex + 8), dlm_add_hi);

            __m256i ref_b_ref_b1_lo = _mm256_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m256i ref_b_ref_b1_hi = _mm256_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m256i ref_b_ref_b2_lo = _mm256_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m256i ref_b_ref_b2_hi = _mm256_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m256i ref_b_ref_b3_lo = _mm256_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m256i ref_b_ref_b3_hi = _mm256_mullo_epi32(ref_b3_hi, ref_b3_hi);

            ref_b_ref_b1_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b1_lo, perm_for_64b_mul_256);
            ref_b1_lo = _mm256_permutevar8x32_epi32( ref_b1_lo, perm_for_64b_mul_256);
            ref_b_ref_b1_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b1_hi, perm_for_64b_mul_256);
            ref_b1_hi = _mm256_permutevar8x32_epi32( ref_b1_hi, perm_for_64b_mul_256);
            ref_b_ref_b2_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b2_lo, perm_for_64b_mul_256);
            ref_b2_lo = _mm256_permutevar8x32_epi32( ref_b2_lo, perm_for_64b_mul_256);
            ref_b_ref_b2_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b2_hi, perm_for_64b_mul_256);
            ref_b2_hi = _mm256_permutevar8x32_epi32( ref_b2_hi, perm_for_64b_mul_256);
            ref_b_ref_b3_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b3_lo, perm_for_64b_mul_256);
            ref_b3_lo = _mm256_permutevar8x32_epi32( ref_b3_lo, perm_for_64b_mul_256);
            ref_b_ref_b3_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b3_hi, perm_for_64b_mul_256);
            ref_b3_hi = _mm256_permutevar8x32_epi32( ref_b3_hi, perm_for_64b_mul_256);

            __m256i ref_b_ref_b1_lo0 = _mm256_mul_epi32(ref_b_ref_b1_lo, ref_b1_lo);
            __m256i ref_b_ref_b1_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b1_lo, 32), _mm256_srli_epi64(ref_b1_lo, 32));
            __m256i ref_b_ref_b1_hi0 = _mm256_mul_epi32(ref_b_ref_b1_hi, ref_b1_hi);
            __m256i ref_b_ref_b1_hi1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b1_hi, 32), _mm256_srli_epi64(ref_b1_hi, 32));
            __m256i ref_b_ref_b2_lo0 = _mm256_mul_epi32(ref_b_ref_b2_lo, ref_b2_lo);
            __m256i ref_b_ref_b2_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b2_lo, 32), _mm256_srli_epi64(ref_b2_lo, 32));
            __m256i ref_b_ref_b2_hi0 = _mm256_mul_epi32(ref_b_ref_b2_hi, ref_b2_hi);
            __m256i ref_b_ref_b2_hi1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b2_hi, 32), _mm256_srli_epi64(ref_b2_hi, 32));
            __m256i ref_b_ref_b3_lo0 = _mm256_mul_epi32(ref_b_ref_b3_lo, ref_b3_lo);
            __m256i ref_b_ref_b3_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b3_lo, 32), _mm256_srli_epi64(ref_b3_lo, 32));
            __m256i ref_b_ref_b3_hi0 = _mm256_mul_epi32(ref_b_ref_b3_hi, ref_b3_hi);
            __m256i ref_b_ref_b3_hi1 = _mm256_mul_epi32(_mm256_srli_epi64(ref_b_ref_b3_hi, 32), _mm256_srli_epi64(ref_b3_hi, 32));

            __m256i b1_r4_lo = _mm256_add_epi64(ref_b_ref_b1_lo0, ref_b_ref_b1_lo1);
            __m256i b1_r4_hi = _mm256_add_epi64(ref_b_ref_b1_hi0, ref_b_ref_b1_hi1);
            __m256i b2_r4_lo = _mm256_add_epi64(ref_b_ref_b2_lo0, ref_b_ref_b2_lo1);
            __m256i b2_r4_hi = _mm256_add_epi64(ref_b_ref_b2_hi0, ref_b_ref_b2_hi1);
            __m256i b3_r4_lo = _mm256_add_epi64(ref_b_ref_b3_lo0, ref_b_ref_b3_lo1);
            __m256i b3_r4_hi = _mm256_add_epi64(ref_b_ref_b3_hi0, ref_b_ref_b3_hi1);
            __m256i b1_r4 = _mm256_add_epi64(b1_r4_lo, b1_r4_hi);
            __m256i b2_r4 = _mm256_add_epi64(b2_r4_lo, b2_r4_hi);
            __m256i b3_r4 = _mm256_add_epi64(b3_r4_lo, b3_r4_hi);
            __m128i b1_r2 = _mm_add_epi64(_mm256_castsi256_si128(b1_r4), _mm256_extractf128_si256(b1_r4, 1));
            __m128i b2_r2 = _mm_add_epi64(_mm256_castsi256_si128(b2_r4), _mm256_extractf128_si256(b2_r4, 1));
            __m128i b3_r2 = _mm_add_epi64(_mm256_castsi256_si128(b3_r4), _mm256_extractf128_si256(b3_r4, 1));
            int64_t r_b1 = _mm_extract_epi64(b1_r2, 0) + _mm_extract_epi64(b1_r2, 1);
            int64_t r_b2 = _mm_extract_epi64(b2_r2, 0) + _mm_extract_epi64(b2_r2, 1);
            int64_t r_b3 = _mm_extract_epi64(b3_r2, 0) + _mm_extract_epi64(b3_r2, 1);

            den_row_sum[0] += r_b1;
            den_row_sum[1] += r_b2;
            den_row_sum[2] += r_b3;
        }

        for (; j < loop_w_8; j+=8)
        {
            index = i * width + j;
            //If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;
			restIndex = (i - border_h) * (dlm_width) + j - border_w;

            __m128i ref_b1_128 = _mm_loadu_si128((__m128i*)(ref.bands[1] + index));
            __m128i dis_b1_128 = _mm_loadu_si128((__m128i*)(dist.bands[1] + index));
            __m128i ref_b2_128 = _mm_loadu_si128((__m128i*)(ref.bands[2] + index));
            __m128i dis_b2_128 = _mm_loadu_si128((__m128i*)(dist.bands[2] + index));

            __m128i ref_b1b2_lo = _mm_unpacklo_epi16(ref_b1_128, ref_b2_128);
            __m128i ref_b1b2_hi = _mm_unpackhi_epi16(ref_b1_128, ref_b2_128);
            __m128i dis_b1b2_lo = _mm_unpacklo_epi16(dis_b1_128, dis_b2_128);
            __m128i dis_b1b2_hi = _mm_unpackhi_epi16(dis_b1_128, dis_b2_128);

            __m128i ot_dp_lo = _mm_madd_epi16(ref_b1b2_lo, dis_b1b2_lo);
            __m128i ot_dp_hi = _mm_madd_epi16(ref_b1b2_hi, dis_b1b2_hi);

            __m128i o_mag_sq_lo = _mm_madd_epi16(ref_b1b2_lo, ref_b1b2_lo);
            __m128i o_mag_sq_hi = _mm_madd_epi16(ref_b1b2_hi, ref_b1b2_hi);
            
            __m128i t_mag_sq_lo = _mm_madd_epi16(dis_b1b2_lo, dis_b1b2_lo);
            __m128i t_mag_sq_hi = _mm_madd_epi16(dis_b1b2_hi, dis_b1b2_hi);

            ot_dp_lo = _mm_max_epi32(ot_dp_lo, zero_128);
            ot_dp_hi = _mm_max_epi32(ot_dp_hi, zero_128);

            __m128i ot_dp_lo_0 = _mm_mul_epi32(ot_dp_lo, ot_dp_lo);
            __m128i ot_dp_lo_1 = _mm_mul_epi32(_mm_srai_epi64(ot_dp_lo, 32), _mm_srai_epi64(ot_dp_lo, 32));
            __m128i ot_dp_hi_0 = _mm_mul_epi32(ot_dp_hi, ot_dp_hi);
            __m128i ot_dp_hi_1 = _mm_mul_epi32(_mm_srai_epi64(ot_dp_hi, 32), _mm_srai_epi64(ot_dp_hi, 32));

            __m128i ot_mag_sq_lo_0 = _mm_mul_epi32(o_mag_sq_lo, t_mag_sq_lo);
            __m128i ot_mag_sq_lo_1 = _mm_mul_epi32(_mm_srai_epi64(o_mag_sq_lo, 32), _mm_srai_epi64(t_mag_sq_lo, 32));
            __m128i ot_mag_sq_hi_0 = _mm_mul_epi32(o_mag_sq_hi, t_mag_sq_hi);
            __m128i ot_mag_sq_hi_1 = _mm_mul_epi32(_mm_srai_epi64(o_mag_sq_hi, 32), _mm_srai_epi64(t_mag_sq_hi, 32));
            
            __mmask8 angle_mask8 = 0;
            int a0 = ((adm_i64_dtype)ot_dp_lo_0[0] >= COS_1DEG_SQ * (adm_i64_dtype)ot_mag_sq_lo_0[0]);
            int a2 = (ot_dp_lo_0[1] >= COS_1DEG_SQ * ot_mag_sq_lo_0[1]) << 2;
            int a1 = (ot_dp_lo_1[0] >= COS_1DEG_SQ * ot_mag_sq_lo_1[0]) << 1;
            int a3 = (ot_dp_lo_1[1] >= COS_1DEG_SQ * ot_mag_sq_lo_1[1]) << 3;
            int a4 = (ot_dp_hi_0[0] >= COS_1DEG_SQ * ot_mag_sq_hi_0[0]) << 4;
            int a6 = (ot_dp_hi_0[1] >= COS_1DEG_SQ * ot_mag_sq_hi_0[1]) << 6;
            int a5 = (ot_dp_hi_1[0] >= COS_1DEG_SQ * ot_mag_sq_hi_1[0]) << 5;
            int a7 = (ot_dp_hi_1[1] >= COS_1DEG_SQ * ot_mag_sq_hi_1[1]) << 7;
            angle_mask8 += a0 + a2 + a1 + a3 + a4 + a6 + a5 + a7;
            
            __m128i dis_b3_128 = _mm_loadu_si128((__m128i*)(dist.bands[3] + index));
            __m128i ref_b3_128 = _mm_loadu_si128((__m128i*)(ref.bands[3] + index));

            __m128i ref_b1_lo, ref_b1_hi, ref_b2_lo, ref_b2_hi, ref_b3_lo, ref_b3_hi;
            cvt_1_16x8_to_2_32x4_256(ref_b1_128, ref_b1_lo, ref_b1_hi);
            cvt_1_16x8_to_2_32x4_256(ref_b2_128, ref_b2_lo, ref_b2_hi);
            cvt_1_16x8_to_2_32x4_256(ref_b3_128, ref_b3_lo, ref_b3_hi);

            __m128i adm_div_b1_lo, adm_div_b1_hi, adm_div_b2_lo, adm_div_b2_hi, adm_div_b3_lo, adm_div_b3_hi;
            adm_div_b1_lo = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b1_lo, add_32768_32b_128), 4);
            adm_div_b1_hi = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b1_hi, add_32768_32b_128), 4);
            adm_div_b2_lo = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b2_lo, add_32768_32b_128), 4);
            adm_div_b2_hi = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b2_hi, add_32768_32b_128), 4);
            adm_div_b3_lo = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b3_lo, add_32768_32b_128), 4);
            adm_div_b3_hi = _mm_i32gather_epi32(adm_div_lookup, _mm_add_epi32(ref_b3_hi, add_32768_32b_128), 4);
            
            __m128i dis_b1_lo, dis_b1_hi, dis_b2_lo, dis_b2_hi, dis_b3_lo, dis_b3_hi;
            cvt_1_16x8_to_2_32x4_256(dis_b1_128, dis_b1_lo, dis_b1_hi);
            cvt_1_16x8_to_2_32x4_256(dis_b2_128, dis_b2_lo, dis_b2_hi);
            cvt_1_16x8_to_2_32x4_256(dis_b3_128, dis_b3_lo, dis_b3_hi);

            __m128i adm_b1_dis_lo0 = _mm_mul_epi32(adm_div_b1_lo, dis_b1_lo);
            __m128i adm_b1_dis_lo1 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b1_lo, 32), _mm_srli_epi64(dis_b1_lo, 32));
            __m128i adm_b1_dis_hi8 = _mm_mul_epi32(adm_div_b1_hi, dis_b1_hi);
            __m128i adm_b1_dis_hi9 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b1_hi, 32), _mm_srli_epi64(dis_b1_hi, 32));

            __m128i adm_b2_dis_lo0 = _mm_mul_epi32(adm_div_b2_lo, dis_b2_lo);
            __m128i adm_b2_dis_lo1 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b2_lo, 32), _mm_srli_epi64(dis_b2_lo, 32));
            __m128i adm_b2_dis_hi8 = _mm_mul_epi32(adm_div_b2_hi, dis_b2_hi);
            __m128i adm_b2_dis_hi9 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b2_hi, 32), _mm_srli_epi64(dis_b2_hi, 32));

            __m128i adm_b3_dis_lo0 = _mm_mul_epi32(adm_div_b3_lo, dis_b3_lo);
            __m128i adm_b3_dis_lo1 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b3_lo, 32), _mm_srli_epi64(dis_b3_lo, 32));
            __m128i adm_b3_dis_hi8 = _mm_mul_epi32(adm_div_b3_hi, dis_b3_hi);
            __m128i adm_b3_dis_hi9 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b3_hi, 32), _mm_srli_epi64(dis_b3_hi, 32));
 
            adm_b1_dis_lo0 = _mm_add_epi64(adm_b1_dis_lo0, add_16384_128);
            adm_b1_dis_lo1 = _mm_add_epi64(adm_b1_dis_lo1, add_16384_128);
            adm_b1_dis_hi8 = _mm_add_epi64(adm_b1_dis_hi8, add_16384_128);
            adm_b1_dis_hi9 = _mm_add_epi64(adm_b1_dis_hi9, add_16384_128);
            adm_b2_dis_lo0 = _mm_add_epi64(adm_b2_dis_lo0, add_16384_128);
            adm_b2_dis_lo1 = _mm_add_epi64(adm_b2_dis_lo1, add_16384_128);
            adm_b2_dis_hi8 = _mm_add_epi64(adm_b2_dis_hi8, add_16384_128);
            adm_b2_dis_hi9 = _mm_add_epi64(adm_b2_dis_hi9, add_16384_128);
            adm_b3_dis_lo0 = _mm_add_epi64(adm_b3_dis_lo0, add_16384_128);
            adm_b3_dis_lo1 = _mm_add_epi64(adm_b3_dis_lo1, add_16384_128);
            adm_b3_dis_hi8 = _mm_add_epi64(adm_b3_dis_hi8, add_16384_128);
            adm_b3_dis_hi9 = _mm_add_epi64(adm_b3_dis_hi9, add_16384_128);

            shift15_64b_signExt_128(adm_b1_dis_lo0, adm_b1_dis_lo0);
            shift15_64b_signExt_128(adm_b1_dis_lo1, adm_b1_dis_lo1);
            shift15_64b_signExt_128(adm_b1_dis_hi8, adm_b1_dis_hi8);
            shift15_64b_signExt_128(adm_b1_dis_hi9, adm_b1_dis_hi9);
            shift15_64b_signExt_128(adm_b2_dis_lo0, adm_b2_dis_lo0);
            shift15_64b_signExt_128(adm_b2_dis_lo1, adm_b2_dis_lo1);
            shift15_64b_signExt_128(adm_b2_dis_hi8, adm_b2_dis_hi8);
            shift15_64b_signExt_128(adm_b2_dis_hi9, adm_b2_dis_hi9);
            shift15_64b_signExt_128(adm_b3_dis_lo0, adm_b3_dis_lo0);
            shift15_64b_signExt_128(adm_b3_dis_lo1, adm_b3_dis_lo1);
            shift15_64b_signExt_128(adm_b3_dis_hi8, adm_b3_dis_hi8);
            shift15_64b_signExt_128(adm_b3_dis_hi9, adm_b3_dis_hi9);

            __mmask8 eqz_b1_lo = _mm_cmpeq_epi32_mask(ref_b1_lo, _mm_setzero_si128());
            __mmask8 eqz_b1_hi = _mm_cmpeq_epi32_mask(ref_b1_hi, _mm_setzero_si128());
            __mmask8 eqz_b2_lo = _mm_cmpeq_epi32_mask(ref_b2_lo, _mm_setzero_si128());
            __mmask8 eqz_b2_hi = _mm_cmpeq_epi32_mask(ref_b2_hi, _mm_setzero_si128());
            __mmask8 eqz_b3_lo = _mm_cmpeq_epi32_mask(ref_b3_lo, _mm_setzero_si128());
            __mmask8 eqz_b3_hi = _mm_cmpeq_epi32_mask(ref_b3_hi, _mm_setzero_si128());

            __m128i adm_b1_dis_lo = _mm_permutex2var_epi32(adm_b1_dis_lo0, perm_64_to_32_128, adm_b1_dis_lo1);
            __m128i adm_b1_dis_hi = _mm_permutex2var_epi32(adm_b1_dis_hi8, perm_64_to_32_128, adm_b1_dis_hi9);
            __m128i adm_b2_dis_lo = _mm_permutex2var_epi32(adm_b2_dis_lo0, perm_64_to_32_128, adm_b2_dis_lo1);
            __m128i adm_b2_dis_hi = _mm_permutex2var_epi32(adm_b2_dis_hi8, perm_64_to_32_128, adm_b2_dis_hi9);
            __m128i adm_b3_dis_lo = _mm_permutex2var_epi32(adm_b3_dis_lo0, perm_64_to_32_128, adm_b3_dis_lo1);
            __m128i adm_b3_dis_hi = _mm_permutex2var_epi32(adm_b3_dis_hi8, perm_64_to_32_128, adm_b3_dis_hi9);

            __m128i tmp_k_b1_lo = _mm_mask_blend_epi32(eqz_b1_lo, adm_b1_dis_lo, add_32768_128);
            __m128i tmp_k_b1_hi = _mm_mask_blend_epi32(eqz_b1_hi, adm_b1_dis_hi, add_32768_128);
            __m128i tmp_k_b2_lo = _mm_mask_blend_epi32(eqz_b2_lo, adm_b2_dis_lo, add_32768_128);
            __m128i tmp_k_b2_hi = _mm_mask_blend_epi32(eqz_b2_hi, adm_b2_dis_hi, add_32768_128);
            __m128i tmp_k_b3_lo = _mm_mask_blend_epi32(eqz_b3_lo, adm_b3_dis_lo, add_32768_128);
            __m128i tmp_k_b3_hi = _mm_mask_blend_epi32(eqz_b3_hi, adm_b3_dis_hi, add_32768_128);
            
            tmp_k_b1_lo = _mm_max_epi32(tmp_k_b1_lo, zero_128);
            tmp_k_b1_hi = _mm_max_epi32(tmp_k_b1_hi, zero_128);
            tmp_k_b2_lo = _mm_max_epi32(tmp_k_b2_lo, zero_128);
            tmp_k_b2_hi = _mm_max_epi32(tmp_k_b2_hi, zero_128);
            tmp_k_b3_lo = _mm_max_epi32(tmp_k_b3_lo, zero_128);
            tmp_k_b3_hi = _mm_max_epi32(tmp_k_b3_hi, zero_128);

            tmp_k_b1_lo = _mm_min_epi32(tmp_k_b1_lo, add_32768_32b_128);
            tmp_k_b1_hi = _mm_min_epi32(tmp_k_b1_hi, add_32768_32b_128);
            tmp_k_b2_lo = _mm_min_epi32(tmp_k_b2_lo, add_32768_32b_128);
            tmp_k_b2_hi = _mm_min_epi32(tmp_k_b2_hi, add_32768_32b_128);
            tmp_k_b3_lo = _mm_min_epi32(tmp_k_b3_lo, add_32768_32b_128);
            tmp_k_b3_hi = _mm_min_epi32(tmp_k_b3_hi, add_32768_32b_128);
                        
            __m128i tmp_val_b1_lo = _mm_mullo_epi32(tmp_k_b1_lo, ref_b1_lo);
            __m128i tmp_val_b1_hi = _mm_mullo_epi32(tmp_k_b1_hi, ref_b1_hi);
            __m128i tmp_val_b2_lo = _mm_mullo_epi32(tmp_k_b2_lo, ref_b2_lo);
            __m128i tmp_val_b2_hi = _mm_mullo_epi32(tmp_k_b2_hi, ref_b2_hi);
            __m128i tmp_val_b3_lo = _mm_mullo_epi32(tmp_k_b3_lo, ref_b3_lo);
            __m128i tmp_val_b3_hi = _mm_mullo_epi32(tmp_k_b3_hi, ref_b3_hi);

            tmp_val_b1_lo = _mm_add_epi32(tmp_val_b1_lo, add_16384_32b_128);
            tmp_val_b1_hi = _mm_add_epi32(tmp_val_b1_hi, add_16384_32b_128);
            tmp_val_b2_lo = _mm_add_epi32(tmp_val_b2_lo, add_16384_32b_128);
            tmp_val_b3_lo = _mm_add_epi32(tmp_val_b3_lo, add_16384_32b_128);
            tmp_val_b2_hi = _mm_add_epi32(tmp_val_b2_hi, add_16384_32b_128);
            tmp_val_b3_hi = _mm_add_epi32(tmp_val_b3_hi, add_16384_32b_128);

            tmp_val_b1_lo = _mm_srai_epi32(tmp_val_b1_lo, 15);
            tmp_val_b1_hi = _mm_srai_epi32(tmp_val_b1_hi, 15);
            tmp_val_b2_lo = _mm_srai_epi32(tmp_val_b2_lo, 15);
            tmp_val_b2_hi = _mm_srai_epi32(tmp_val_b2_hi, 15);
            tmp_val_b3_lo = _mm_srai_epi32(tmp_val_b3_lo, 15);
            tmp_val_b3_hi = _mm_srai_epi32(tmp_val_b3_hi, 15);

            __m128i tmp_val_b1 = _mm_packs_epi32(tmp_val_b1_lo, tmp_val_b1_hi);
            __m128i tmp_val_b2 = _mm_packs_epi32(tmp_val_b2_lo, tmp_val_b2_hi);
            __m128i tmp_val_b3 = _mm_packs_epi32(tmp_val_b3_lo, tmp_val_b3_hi);

            __m128i dlm_rest_b1_128 = _mm_mask_blend_epi16(angle_mask8, tmp_val_b1, dis_b1_128);
            __m128i dlm_rest_b2_128 = _mm_mask_blend_epi16(angle_mask8, tmp_val_b2, dis_b2_128);
            __m128i dlm_rest_b3_128 = _mm_mask_blend_epi16(angle_mask8, tmp_val_b3, dis_b3_128);

            __m128i dist_m_dlm_rest_b1 = _mm_abs_epi16(_mm_sub_epi16(dis_b1_128, dlm_rest_b1_128));
            __m128i dist_m_dlm_rest_b2 = _mm_abs_epi16(_mm_sub_epi16(dis_b2_128, dlm_rest_b2_128));
            __m128i dlm_add_256 = _mm_adds_epu16(dist_m_dlm_rest_b1, dist_m_dlm_rest_b2);
            __m128i dist_m_dlm_rest_b3 = _mm_abs_epi16(_mm_sub_epi16(dis_b3_128, dlm_rest_b3_128));            
            dlm_add_256 = _mm_adds_epu16(dlm_add_256, dist_m_dlm_rest_b3);

            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[1] + restIndex), dlm_rest_b1_128);
            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[2] + restIndex), dlm_rest_b2_128);
            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[3] + restIndex), dlm_rest_b3_128);

            __m128i dlm_add_lo = _mm_cvtepu16_epi32(dlm_add_256);
            __m128i dlm_add_hi = _mm_cvtepu16_epi32(_mm_shuffle_epi32(dlm_add_256, 0x0E));

            ref_b1_lo = _mm_abs_epi32(ref_b1_lo);
            ref_b1_hi = _mm_abs_epi32(ref_b1_hi);
            ref_b2_lo = _mm_abs_epi32(ref_b2_lo);
            ref_b2_hi = _mm_abs_epi32(ref_b2_hi);
            ref_b3_lo = _mm_abs_epi32(ref_b3_lo);
            ref_b3_hi = _mm_abs_epi32(ref_b3_hi);

            _mm_storeu_si128((__m128i*)(i_dlm_add + addIndex), dlm_add_lo);
            _mm_storeu_si128((__m128i*)(i_dlm_add + addIndex + 4), dlm_add_hi);

            __m128i ref_b_ref_b1_lo = _mm_mullo_epi32(ref_b1_lo, ref_b1_lo);
            __m128i ref_b_ref_b1_hi = _mm_mullo_epi32(ref_b1_hi, ref_b1_hi);
            __m128i ref_b_ref_b2_lo = _mm_mullo_epi32(ref_b2_lo, ref_b2_lo);
            __m128i ref_b_ref_b2_hi = _mm_mullo_epi32(ref_b2_hi, ref_b2_hi);
            __m128i ref_b_ref_b3_lo = _mm_mullo_epi32(ref_b3_lo, ref_b3_lo);
            __m128i ref_b_ref_b3_hi = _mm_mullo_epi32(ref_b3_hi, ref_b3_hi);

            ref_b_ref_b1_lo = _mm_shuffle_epi32( ref_b_ref_b1_lo, 0xD8);          
            ref_b1_lo = _mm_shuffle_epi32( ref_b1_lo, 0xD8);
            ref_b_ref_b1_hi = _mm_shuffle_epi32( ref_b_ref_b1_hi, 0xD8);
            ref_b1_hi = _mm_shuffle_epi32( ref_b1_hi, 0xD8);
            ref_b_ref_b2_lo = _mm_shuffle_epi32( ref_b_ref_b2_lo, 0xD8);
            ref_b2_lo = _mm_shuffle_epi32( ref_b2_lo, 0xD8);
            ref_b_ref_b2_hi = _mm_shuffle_epi32( ref_b_ref_b2_hi, 0xD8);
            ref_b2_hi = _mm_shuffle_epi32( ref_b2_hi, 0xD8);
            ref_b_ref_b3_lo = _mm_shuffle_epi32( ref_b_ref_b3_lo, 0xD8);
            ref_b3_lo = _mm_shuffle_epi32( ref_b3_lo, 0xD8);
            ref_b_ref_b3_hi = _mm_shuffle_epi32( ref_b_ref_b3_hi, 0xD8);
            ref_b3_hi = _mm_shuffle_epi32( ref_b3_hi, 0xD8);

            __m128i ref_b_ref_b1_lo0 = _mm_mul_epi32(ref_b_ref_b1_lo, ref_b1_lo);
            __m128i ref_b_ref_b1_lo1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b1_lo, 32), _mm_srli_epi64(ref_b1_lo, 32));
            __m128i ref_b_ref_b1_hi0 = _mm_mul_epi32(ref_b_ref_b1_hi, ref_b1_hi);
            __m128i ref_b_ref_b1_hi1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b1_hi, 32), _mm_srli_epi64(ref_b1_hi, 32));
            __m128i ref_b_ref_b2_lo0 = _mm_mul_epi32(ref_b_ref_b2_lo, ref_b2_lo);
            __m128i ref_b_ref_b2_lo1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b2_lo, 32), _mm_srli_epi64(ref_b2_lo, 32));
            __m128i ref_b_ref_b2_hi0 = _mm_mul_epi32(ref_b_ref_b2_hi, ref_b2_hi);
            __m128i ref_b_ref_b2_hi1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b2_hi, 32), _mm_srli_epi64(ref_b2_hi, 32));
            __m128i ref_b_ref_b3_lo0 = _mm_mul_epi32(ref_b_ref_b3_lo, ref_b3_lo);
            __m128i ref_b_ref_b3_lo1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b3_lo, 32), _mm_srli_epi64(ref_b3_lo, 32));
            __m128i ref_b_ref_b3_hi0 = _mm_mul_epi32(ref_b_ref_b3_hi, ref_b3_hi);
            __m128i ref_b_ref_b3_hi1 = _mm_mul_epi32(_mm_srli_epi64(ref_b_ref_b3_hi, 32), _mm_srli_epi64(ref_b3_hi, 32));

            __m128i b1_r2_lo = _mm_add_epi64(ref_b_ref_b1_lo0, ref_b_ref_b1_lo1);
            __m128i b1_r2_hi = _mm_add_epi64(ref_b_ref_b1_hi0, ref_b_ref_b1_hi1);
            __m128i b2_r2_lo = _mm_add_epi64(ref_b_ref_b2_lo0, ref_b_ref_b2_lo1);
            __m128i b2_r2_hi = _mm_add_epi64(ref_b_ref_b2_hi0, ref_b_ref_b2_hi1);
            __m128i b3_r2_lo = _mm_add_epi64(ref_b_ref_b3_lo0, ref_b_ref_b3_lo1);
            __m128i b3_r2_hi = _mm_add_epi64(ref_b_ref_b3_hi0, ref_b_ref_b3_hi1);
            __m128i b1_r2 = _mm_add_epi64(b1_r2_lo, b1_r2_hi);
            __m128i b2_r2 = _mm_add_epi64(b2_r2_lo, b2_r2_hi);
            __m128i b3_r2 = _mm_add_epi64(b3_r2_lo, b3_r2_hi);
            int64_t r_b1 = _mm_extract_epi64(b1_r2, 0) + _mm_extract_epi64(b1_r2, 1);
            int64_t r_b2 = _mm_extract_epi64(b2_r2, 0) + _mm_extract_epi64(b2_r2, 1);
            int64_t r_b3 = _mm_extract_epi64(b3_r2, 0) + _mm_extract_epi64(b3_r2, 1);

            den_row_sum[0] += r_b1;
            den_row_sum[1] += r_b2;
            den_row_sum[2] += r_b3;
        }

        for (; j < loop_w; j++)
        {
            index = i * width + j;

            //If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;

			restIndex = (i - border_h) * (dlm_width) + j - border_w;
            ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
            o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
            t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);
            angle_flag = ((ot_dp >= 0) && (((adm_i64_dtype)ot_dp * ot_dp) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_sq * t_mag_sq)));
            i_dlm_add[addIndex] = 0;
            for (k = 1; k < 4; k++)
            {
                /**
                 * Division dist/ref is carried using lookup table adm_div_lookup and converted to multiplication
                 */
                adm_i32_dtype tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((adm_i64_dtype)adm_div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
                adm_u16_dtype kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
                /**
                 * kh is in Q15 type and ref.bands[k][index] is in Q16 type hence shifted by
                 * 15 to make result Q16
                 */
                tmp_val = (((adm_i32_dtype)kh * ref.bands[k][index]) + 16384) >> 15;
                
                i_dlm_rest.bands[k][restIndex] = angle_flag ? dist.bands[k][index] : tmp_val;
                /**
                 * Absolute is taken here for the difference value instead of 
                 * taking absolute of pyr_2 in integer_dlm_contrast_mask_one_way function
                 */
                i_dlm_add[addIndex] += (int32_t)abs(dist.bands[k][index] - i_dlm_rest.bands[k][restIndex]);

                //Accumulating denominator score to avoid load in next stage
                int16_t ref_abs = abs(ref.bands[k][index]);
                den_cube[k-1] = (adm_i64_dtype)ref_abs * ref_abs * ref_abs;
                
                den_row_sum[k-1] += den_cube[k-1];
            }
        }
        if(extra_sample_w)
        {
            for(k = 0; k < 3; k++)
            {
                den_row_sum[k] -= den_cube[k];
                den_row_sum[k] -= col0_ref_cube[k];
            }
        }
        if((i != border_h && i != (loop_h - 1)) || !extra_sample_h)
        {
            for(k=0; k<3; k++)
            {
                den_sum[k] += den_row_sum[k];
            }
        }
        den_row_sum[0] = 0;
        den_row_sum[1] = 0;
        den_row_sum[2] = 0;
#if ADM_REFLECT_PAD
        if(!extra_sample_w)
		{
			addIndex = (i + 1 - border_h) * (dlm_add_w);
			i_dlm_add[addIndex + 0] = i_dlm_add[addIndex + 2];
			i_dlm_add[addIndex + dlm_width + 1] = i_dlm_add[addIndex + dlm_width - 1];
		}
#endif
    }
#if ADM_REFLECT_PAD
	if(!extra_sample_h)
	{
		int row2Idx = 2 * (dlm_add_w);
		int rowLast2Idx = (dlm_height - 1) * (dlm_add_w);
		int rowLastPadIdx = (dlm_height + 1) * (dlm_add_w);

		memcpy(&i_dlm_add[0], &i_dlm_add[row2Idx], sizeof(int32_t) * (dlm_add_w));

		memcpy(&i_dlm_add[rowLastPadIdx], &i_dlm_add[rowLast2Idx], sizeof(int32_t) * (dlm_width+2));
	}
#endif 
    //Calculating denominator score
    double den_band = 0;
    for(k=0; k<3; k++)
    {
        double accum_den = (double) den_sum[k] / ADM_CUBE_DIV;
        den_band += powf((double)(accum_den), 1.0 / 3.0);
    }
    // compensation for the division by thirty in the numerator
    *adm_score_den = ((den_band) / (adm_pending_div / powf((double)(256), 1.0 / 3.0))) + 1e-4;

}
