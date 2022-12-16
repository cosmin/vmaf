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
#include "integer_funque_adm_avx2.h"
#include "mem.h"
#include "../adm_tools.h"
#include "../integer_funque_filters.h"
#include <immintrin.h>


#define cvt_1_16x16_to_2_32x8(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_16x16)); \
    r_32x8_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a_16x16, 1)); \
}

#define cvt_1_32x8_to_2_64x4(a_32x8, r_64x4_lo, r_64x4_hi) \
{ \
    r_64x4_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a_32x8)); \
    r_64x4_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(a_32x8, 1)); \
}

#define cvt_1_16x8_to_2_32x4(a_16x16, r_32x8_lo, r_32x8_hi) \
{ \
    r_32x8_lo = _mm_cvtepi16_epi32(a_16x16); \
    r_32x8_hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(a_16x16, 0x0E)); \
}

#define cvt_1_32x4_to_2_64x2(a_32x8, r_64x4_lo, r_64x4_hi) \
{ \
    r_64x4_lo = _mm_cvtepi32_epi64(a_32x8); \
    r_64x4_hi = _mm_cvtepi32_epi64(_mm_shuffle_epi32(a_32x8, 0x0E)); \
}

#define calc_angle(ot_dp, o_mag, t_mag, angle, n) \
{ \
    int ot_dp_int = _mm256_extract_epi32(ot_dp, n); \
    int o_mag_int = _mm256_extract_epi32(o_mag, n); \
    int t_mag_int = _mm256_extract_epi32(t_mag, n); \
    angle = ((ot_dp_int >= 0) && (((adm_i64_dtype)ot_dp_int * ot_dp_int) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_int * t_mag_int))); \
}

#define calc_angle_128(ot_dp, o_mag, t_mag, angle, n) \
{ \
    int ot_dp_int = _mm_extract_epi32(ot_dp, n); \
    int o_mag_int = _mm_extract_epi32(o_mag, n); \
    int t_mag_int = _mm_extract_epi32(t_mag, n); \
    angle = ((ot_dp_int >= 0) && (((adm_i64_dtype)ot_dp_int * ot_dp_int) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_int * t_mag_int))); \
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

#define shift15_64b_signExt(a, r)\
{ \
    r = _mm256_add_epi64( _mm256_srli_epi64(a, 15) , _mm256_and_si256(a, _mm256_set1_epi64x(0xFFFE000000000000)));\
}

#define shift15_64b_signExt_128(a, r)\
{ \
    r = _mm_add_epi64( _mm_srli_epi64(a, 15) , _mm_and_si128(a, _mm_set1_epi64x(0xFFFE000000000000)));\
} 

void integer_adm_decouple_avx2(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den)
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

    uint16_t angle_flag_table[16];

	int loop_w_16 = loop_w - ((loop_w - border_w) % 16);
    int loop_w_8 = loop_w - ((loop_w - border_w) % 8);

    __m256i perm_dis = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i add_16384_256 = _mm256_set1_epi64x(16384);
    __m256i add_32768_256 = _mm256_set1_epi64x(32768);
    __m256i add_32768_32b_256 = _mm256_set1_epi32(32768);
    __m256i add_16384_32b_256 = _mm256_set1_epi32(16384);

    __m128i add_16384_128 = _mm_set1_epi64x(16384);
    __m128i add_32768_128 = _mm_set1_epi64x(32768);
    __m128i add_32768_32b_128 = _mm_set1_epi32(32768);
    __m128i add_16384_32b_128 = _mm_set1_epi32(16384);

    __m256i perm_64b_to_32b = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    __m256i perm_for_64b_mul = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

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
        for (j = border_w; j < loop_w_16; j+=16)
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

            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[0], 0);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[1], 1);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[2], 2);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[3], 3);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[4], 0);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[5], 1);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[6], 2);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[7], 3);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[8], 4);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[9], 5);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[10], 6);
            calc_angle(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[11], 7);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[12], 4);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[13], 5);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[14], 6);
            calc_angle(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[15], 7);

            __m256i angle_256 = _mm256_set_epi16(angle_flag_table[15], angle_flag_table[14], angle_flag_table[13], angle_flag_table[12], \
            angle_flag_table[11], angle_flag_table[10], angle_flag_table[9], angle_flag_table[8], angle_flag_table[7], angle_flag_table[6], \
            angle_flag_table[5], angle_flag_table[4], angle_flag_table[3], angle_flag_table[2], angle_flag_table[1], angle_flag_table[0]);
            __m256i dlm_add_select = _mm256_mullo_epi16(angle_256, _mm256_set1_epi16((int16_t)0xFFFF));
            
            __m256i dis_b3_256 = _mm256_loadu_si256((__m256i*)(dist.bands[3] + index));
            __m256i ref_b3_256 = _mm256_loadu_si256((__m256i*)(ref.bands[3] + index));

            __m256i adm_div_b1_lo, adm_div_b1_hi, adm_div_b2_lo, adm_div_b2_hi, adm_div_b3_lo, adm_div_b3_hi;

            // 0 4 1 5 2 6 3 7
            adm_div_b1_lo = _mm256_set_epi32(adm_div_lookup[ref.bands[1][index + 7] + 32768], adm_div_lookup[ref.bands[1][index + 3] + 32768], 
            adm_div_lookup[ref.bands[1][index + 6] + 32768], adm_div_lookup[ref.bands[1][index + 2] + 32768], \
            adm_div_lookup[ref.bands[1][index + 5] + 32768], adm_div_lookup[ref.bands[1][index + 1] + 32768], \
            adm_div_lookup[ref.bands[1][index + 4] + 32768], adm_div_lookup[ref.bands[1][index] + 32768]);

            // 8 12 9 13 10 14 11 15
            adm_div_b1_hi = _mm256_set_epi32(adm_div_lookup[ref.bands[1][index + 15] + 32768], adm_div_lookup[ref.bands[1][index + 11] + 32768], 
            adm_div_lookup[ref.bands[1][index + 14] + 32768], adm_div_lookup[ref.bands[1][index + 10] + 32768], \
            adm_div_lookup[ref.bands[1][index + 13] + 32768], adm_div_lookup[ref.bands[1][index + 9] + 32768], \
            adm_div_lookup[ref.bands[1][index + 12] + 32768], adm_div_lookup[ref.bands[1][index + 8] + 32768]);

            adm_div_b2_lo = _mm256_set_epi32(adm_div_lookup[ref.bands[2][index + 7] + 32768], adm_div_lookup[ref.bands[2][index + 3] + 32768], 
            adm_div_lookup[ref.bands[2][index + 6] + 32768], adm_div_lookup[ref.bands[2][index + 2] + 32768], \
            adm_div_lookup[ref.bands[2][index + 5] + 32768], adm_div_lookup[ref.bands[2][index + 1] + 32768], \
            adm_div_lookup[ref.bands[2][index + 4] + 32768], adm_div_lookup[ref.bands[2][index] + 32768]);

            adm_div_b2_hi = _mm256_set_epi32(adm_div_lookup[ref.bands[2][index + 15] + 32768], adm_div_lookup[ref.bands[2][index + 11] + 32768], 
            adm_div_lookup[ref.bands[2][index + 14] + 32768], adm_div_lookup[ref.bands[2][index + 10] + 32768], \
            adm_div_lookup[ref.bands[2][index + 13] + 32768], adm_div_lookup[ref.bands[2][index + 9] + 32768], \
            adm_div_lookup[ref.bands[2][index + 12] + 32768], adm_div_lookup[ref.bands[2][index + 8] + 32768]);

            adm_div_b3_lo = _mm256_set_epi32(adm_div_lookup[ref.bands[3][index + 7] + 32768], adm_div_lookup[ref.bands[3][index + 3] + 32768], 
            adm_div_lookup[ref.bands[3][index + 6] + 32768], adm_div_lookup[ref.bands[3][index + 2] + 32768], \
            adm_div_lookup[ref.bands[3][index + 5] + 32768], adm_div_lookup[ref.bands[3][index + 1] + 32768], \
            adm_div_lookup[ref.bands[3][index + 4] + 32768], adm_div_lookup[ref.bands[3][index] + 32768]);

            adm_div_b3_hi = _mm256_set_epi32(adm_div_lookup[ref.bands[3][index + 15] + 32768], adm_div_lookup[ref.bands[3][index + 11] + 32768], 
            adm_div_lookup[ref.bands[3][index + 14] + 32768], adm_div_lookup[ref.bands[3][index + 10] + 32768], \
            adm_div_lookup[ref.bands[3][index + 13] + 32768], adm_div_lookup[ref.bands[3][index + 9] + 32768], \
            adm_div_lookup[ref.bands[3][index + 12] + 32768], adm_div_lookup[ref.bands[3][index + 8] + 32768]);

            __m256i dis_b1_lo, dis_b1_hi, dis_b2_lo, dis_b2_hi, dis_b3_lo, dis_b3_hi;
            // 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
            cvt_1_16x16_to_2_32x8(dis_b1_256, dis_b1_lo, dis_b1_hi);
            cvt_1_16x16_to_2_32x8(dis_b2_256, dis_b2_lo, dis_b2_hi);
            cvt_1_16x16_to_2_32x8(dis_b3_256, dis_b3_lo, dis_b3_hi);

            // 0 4 1 5 2 6 3 7 | 8 12 9 13 10 14 11 15
            dis_b1_lo = _mm256_permutevar8x32_epi32(dis_b1_lo, perm_dis);
            dis_b1_hi = _mm256_permutevar8x32_epi32(dis_b1_hi, perm_dis);
            dis_b2_lo = _mm256_permutevar8x32_epi32(dis_b2_lo, perm_dis);
            dis_b2_hi = _mm256_permutevar8x32_epi32(dis_b2_hi, perm_dis);
            dis_b3_lo = _mm256_permutevar8x32_epi32(dis_b3_lo, perm_dis);
            dis_b3_hi = _mm256_permutevar8x32_epi32(dis_b3_hi, perm_dis);

            // 0 1 2 3
            __m256i adm_b1_dis_lo0 = _mm256_mul_epi32(adm_div_b1_lo, dis_b1_lo);
            // 4 5 6 7
            __m256i adm_b1_dis_lo1 = _mm256_mul_epi32(_mm256_srli_epi64(adm_div_b1_lo, 32), _mm256_srli_epi64(dis_b1_lo, 32));
            // 8 9 10 11
            __m256i adm_b1_dis_hi8 = _mm256_mul_epi32(adm_div_b1_hi, dis_b1_hi);
            // 12 13 14 15
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

            shift15_64b_signExt(adm_b1_dis_lo0, adm_b1_dis_lo0);
            shift15_64b_signExt(adm_b1_dis_lo1, adm_b1_dis_lo1);
            shift15_64b_signExt(adm_b1_dis_hi8, adm_b1_dis_hi8);
            shift15_64b_signExt(adm_b1_dis_hi9, adm_b1_dis_hi9);
            shift15_64b_signExt(adm_b2_dis_lo0, adm_b2_dis_lo0);
            shift15_64b_signExt(adm_b2_dis_lo1, adm_b2_dis_lo1);
            shift15_64b_signExt(adm_b2_dis_hi8, adm_b2_dis_hi8);
            shift15_64b_signExt(adm_b2_dis_hi9, adm_b2_dis_hi9);
            shift15_64b_signExt(adm_b3_dis_lo0, adm_b3_dis_lo0);
            shift15_64b_signExt(adm_b3_dis_lo1, adm_b3_dis_lo1);
            shift15_64b_signExt(adm_b3_dis_hi8, adm_b3_dis_hi8);
            shift15_64b_signExt(adm_b3_dis_hi9, adm_b3_dis_hi9);
            
            __m256i ref_b1_lo, ref_b1_hi, ref_b2_lo, ref_b2_hi, ref_b3_lo, ref_b3_hi;
            // 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
            cvt_1_16x16_to_2_32x8(ref_b1_256, ref_b1_lo, ref_b1_hi);
            cvt_1_16x16_to_2_32x8(ref_b2_256, ref_b2_lo, ref_b2_hi);
            cvt_1_16x16_to_2_32x8(ref_b3_256, ref_b3_lo, ref_b3_hi);

            __m256i eqz_b1_lo = _mm256_cmpeq_epi32(ref_b1_lo, _mm256_setzero_si256());
            __m256i eqz_b1_hi = _mm256_cmpeq_epi32(ref_b1_hi, _mm256_setzero_si256());
            __m256i eqz_b2_lo = _mm256_cmpeq_epi32(ref_b2_lo, _mm256_setzero_si256());
            __m256i eqz_b2_hi = _mm256_cmpeq_epi32(ref_b2_hi, _mm256_setzero_si256());
            __m256i eqz_b3_lo = _mm256_cmpeq_epi32(ref_b3_lo, _mm256_setzero_si256());
            __m256i eqz_b3_hi = _mm256_cmpeq_epi32(ref_b3_hi, _mm256_setzero_si256());
            
            __m256i eqz_b1_lo0, eqz_b1_lo1, eqz_b1_hi0, eqz_b1_hi1, eqz_b2_lo0, eqz_b2_lo1, eqz_b2_hi0, eqz_b2_hi1, eqz_b3_lo0, eqz_b3_lo1, eqz_b3_hi0, eqz_b3_hi1;
            // 0 1 2 3 | 4 5 6 7
            cvt_1_32x8_to_2_64x4(eqz_b1_lo, eqz_b1_lo0, eqz_b1_lo1);
            cvt_1_32x8_to_2_64x4(eqz_b1_hi, eqz_b1_hi0, eqz_b1_hi1);
            cvt_1_32x8_to_2_64x4(eqz_b2_lo, eqz_b2_lo0, eqz_b2_lo1);
            cvt_1_32x8_to_2_64x4(eqz_b2_hi, eqz_b2_hi0, eqz_b2_hi1);
            cvt_1_32x8_to_2_64x4(eqz_b3_lo, eqz_b3_lo0, eqz_b3_lo1);
            cvt_1_32x8_to_2_64x4(eqz_b3_hi, eqz_b3_hi0, eqz_b3_hi1);

            adm_b1_dis_lo0 = _mm256_andnot_si256(eqz_b1_lo0, adm_b1_dis_lo0);
            adm_b1_dis_lo1 = _mm256_andnot_si256(eqz_b1_lo1, adm_b1_dis_lo1);
            adm_b1_dis_hi8 = _mm256_andnot_si256(eqz_b1_hi0, adm_b1_dis_hi8);
            adm_b1_dis_hi9 = _mm256_andnot_si256(eqz_b1_hi1, adm_b1_dis_hi9);

            adm_b2_dis_lo0 = _mm256_andnot_si256(eqz_b2_lo0, adm_b2_dis_lo0);
            adm_b2_dis_lo1 = _mm256_andnot_si256(eqz_b2_lo1, adm_b2_dis_lo1);
            adm_b2_dis_hi8 = _mm256_andnot_si256(eqz_b2_hi0, adm_b2_dis_hi8);
            adm_b2_dis_hi9 = _mm256_andnot_si256(eqz_b2_hi1, adm_b2_dis_hi9);

            adm_b3_dis_lo0 = _mm256_andnot_si256(eqz_b3_lo0, adm_b3_dis_lo0);
            adm_b3_dis_lo1 = _mm256_andnot_si256(eqz_b3_lo1, adm_b3_dis_lo1);
            adm_b3_dis_hi8 = _mm256_andnot_si256(eqz_b3_hi0, adm_b3_dis_hi8);
            adm_b3_dis_hi9 = _mm256_andnot_si256(eqz_b3_hi1, adm_b3_dis_hi9);

            // ref.bands[k][index] == 0 ? b[k]_add_[index] = add_32768 : 0
            __m256i b1_add_lo0 = _mm256_and_si256(eqz_b1_lo0, add_32768_256);
            __m256i b1_add_lo1 = _mm256_and_si256(eqz_b1_lo1, add_32768_256);
            __m256i b1_add_hi0 = _mm256_and_si256(eqz_b1_hi0, add_32768_256);
            __m256i b1_add_hi1 = _mm256_and_si256(eqz_b1_hi1, add_32768_256);

            __m256i b2_add_lo0 = _mm256_and_si256(eqz_b2_lo0, add_32768_256);
            __m256i b2_add_lo1 = _mm256_and_si256(eqz_b2_lo1, add_32768_256);
            __m256i b2_add_hi0 = _mm256_and_si256(eqz_b2_hi0, add_32768_256);
            __m256i b2_add_hi1 = _mm256_and_si256(eqz_b2_hi1, add_32768_256);

            __m256i b3_add_lo0 = _mm256_and_si256(eqz_b3_lo0, add_32768_256);
            __m256i b3_add_lo1 = _mm256_and_si256(eqz_b3_lo1, add_32768_256);
            __m256i b3_add_hi0 = _mm256_and_si256(eqz_b3_hi0, add_32768_256);
            __m256i b3_add_hi1 = _mm256_and_si256(eqz_b3_hi1, add_32768_256);

            // ref.bands[k][index] == 0 ? b[k]_add_[index] = add_32768 : b[k]_add_[index]
            adm_b1_dis_lo0 = _mm256_add_epi64(b1_add_lo0, adm_b1_dis_lo0);
            adm_b1_dis_lo1 = _mm256_add_epi64(b1_add_lo1, adm_b1_dis_lo1);
            adm_b1_dis_hi8 = _mm256_add_epi64(b1_add_hi0, adm_b1_dis_hi8);
            adm_b1_dis_hi9 = _mm256_add_epi64(b1_add_hi1, adm_b1_dis_hi9);

            adm_b2_dis_lo0 = _mm256_add_epi64(b2_add_lo0, adm_b2_dis_lo0);
            adm_b2_dis_lo1 = _mm256_add_epi64(b2_add_lo1, adm_b2_dis_lo1);
            adm_b2_dis_hi8 = _mm256_add_epi64(b2_add_hi0, adm_b2_dis_hi8);
            adm_b2_dis_hi9 = _mm256_add_epi64(b2_add_hi1, adm_b2_dis_hi9);

            adm_b3_dis_lo0 = _mm256_add_epi64(b3_add_lo0, adm_b3_dis_lo0);
            adm_b3_dis_lo1 = _mm256_add_epi64(b3_add_lo1, adm_b3_dis_lo1);
            adm_b3_dis_hi8 = _mm256_add_epi64(b3_add_hi0, adm_b3_dis_hi8);
            adm_b3_dis_hi9 = _mm256_add_epi64(b3_add_hi1, adm_b3_dis_hi9);

            __m256i tmp_k_b1_lo = _mm256_permutevar8x32_epi32( adm_b1_dis_lo0, perm_64b_to_32b);
            __m256i tmp_k_b1_hi = _mm256_permutevar8x32_epi32( adm_b1_dis_hi8, perm_64b_to_32b);
            __m256i tmp_k_b2_lo = _mm256_permutevar8x32_epi32( adm_b2_dis_lo0, perm_64b_to_32b);
            __m256i tmp_k_b2_hi = _mm256_permutevar8x32_epi32( adm_b2_dis_hi8, perm_64b_to_32b);
            __m256i tmp_k_b3_lo = _mm256_permutevar8x32_epi32( adm_b3_dis_lo0, perm_64b_to_32b);
            __m256i tmp_k_b3_hi = _mm256_permutevar8x32_epi32( adm_b3_dis_hi8, perm_64b_to_32b);
            tmp_k_b1_lo = _mm256_inserti128_si256(tmp_k_b1_lo, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b1_dis_lo1, perm_64b_to_32b)), 1);
            tmp_k_b1_hi = _mm256_inserti128_si256(tmp_k_b1_hi, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b1_dis_hi9, perm_64b_to_32b)), 1);
            tmp_k_b2_lo = _mm256_inserti128_si256(tmp_k_b2_lo, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b2_dis_lo1, perm_64b_to_32b)), 1);
            tmp_k_b2_hi = _mm256_inserti128_si256(tmp_k_b2_hi, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b2_dis_hi9, perm_64b_to_32b)), 1);
            tmp_k_b3_lo = _mm256_inserti128_si256(tmp_k_b3_lo, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b3_dis_lo1, perm_64b_to_32b)), 1);
            tmp_k_b3_hi = _mm256_inserti128_si256(tmp_k_b3_hi, _mm256_castsi256_si128(_mm256_permutevar8x32_epi32( adm_b3_dis_hi9, perm_64b_to_32b)), 1);

            __m256i tmp_k_b1_lo_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b1_lo);
            __m256i tmp_k_b1_hi_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b1_hi);
            __m256i tmp_k_b2_lo_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b2_lo);
            __m256i tmp_k_b2_hi_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b2_hi);
            __m256i tmp_k_b3_lo_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b3_lo);
            __m256i tmp_k_b3_hi_eqz = _mm256_cmpgt_epi32(_mm256_setzero_si256(), tmp_k_b3_hi);

            tmp_k_b1_lo = _mm256_andnot_si256(tmp_k_b1_lo_eqz, tmp_k_b1_lo);
            tmp_k_b1_hi = _mm256_andnot_si256(tmp_k_b1_hi_eqz, tmp_k_b1_hi);
            tmp_k_b2_lo = _mm256_andnot_si256(tmp_k_b2_lo_eqz, tmp_k_b2_lo);
            tmp_k_b2_hi = _mm256_andnot_si256(tmp_k_b2_hi_eqz, tmp_k_b2_hi);
            tmp_k_b3_lo = _mm256_andnot_si256(tmp_k_b3_lo_eqz, tmp_k_b3_lo);
            tmp_k_b3_hi = _mm256_andnot_si256(tmp_k_b3_hi_eqz, tmp_k_b3_hi);

            __m256i tmp_k_b1_lo_gt = _mm256_cmpgt_epi32(tmp_k_b1_lo, add_32768_32b_256);
            __m256i tmp_k_b1_hi_gt = _mm256_cmpgt_epi32(tmp_k_b1_hi, add_32768_32b_256);
            __m256i tmp_k_b2_lo_gt = _mm256_cmpgt_epi32(tmp_k_b2_lo, add_32768_32b_256);
            __m256i tmp_k_b2_hi_gt = _mm256_cmpgt_epi32(tmp_k_b2_hi, add_32768_32b_256);
            __m256i tmp_k_b3_lo_gt = _mm256_cmpgt_epi32(tmp_k_b3_lo, add_32768_32b_256);
            __m256i tmp_k_b3_hi_gt = _mm256_cmpgt_epi32(tmp_k_b3_hi, add_32768_32b_256);

            tmp_k_b1_lo = _mm256_andnot_si256(tmp_k_b1_lo_gt, tmp_k_b1_lo);
            tmp_k_b1_hi = _mm256_andnot_si256(tmp_k_b1_hi_gt, tmp_k_b1_hi);
            tmp_k_b2_lo = _mm256_andnot_si256(tmp_k_b2_lo_gt, tmp_k_b2_lo);
            tmp_k_b2_hi = _mm256_andnot_si256(tmp_k_b2_hi_gt, tmp_k_b2_hi);
            tmp_k_b3_lo = _mm256_andnot_si256(tmp_k_b3_lo_gt, tmp_k_b3_lo);
            tmp_k_b3_hi = _mm256_andnot_si256(tmp_k_b3_hi_gt, tmp_k_b3_hi);

            __m256i add_b1_lo = _mm256_and_si256(tmp_k_b1_lo_gt, add_32768_32b_256);
            __m256i add_b1_hi = _mm256_and_si256(tmp_k_b1_hi_gt, add_32768_32b_256);
            __m256i add_b2_lo = _mm256_and_si256(tmp_k_b2_lo_gt, add_32768_32b_256);
            __m256i add_b2_hi = _mm256_and_si256(tmp_k_b2_hi_gt, add_32768_32b_256);
            __m256i add_b3_lo = _mm256_and_si256(tmp_k_b3_lo_gt, add_32768_32b_256);
            __m256i add_b3_hi = _mm256_and_si256(tmp_k_b3_hi_gt, add_32768_32b_256);
         
            tmp_k_b1_lo = _mm256_add_epi32(tmp_k_b1_lo, add_b1_lo);
            tmp_k_b1_hi = _mm256_add_epi32(tmp_k_b1_hi, add_b1_hi);
            tmp_k_b2_lo = _mm256_add_epi32(tmp_k_b2_lo, add_b2_lo);
            tmp_k_b2_hi = _mm256_add_epi32(tmp_k_b2_hi, add_b2_hi);
            tmp_k_b3_lo = _mm256_add_epi32(tmp_k_b3_lo, add_b3_lo);
            tmp_k_b3_hi = _mm256_add_epi32(tmp_k_b3_hi, add_b3_hi);

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

            __m256i dis_and_angle_b1 = _mm256_and_si256(dis_b1_256, dlm_add_select);
            __m256i dis_and_angle_b2 = _mm256_and_si256(dis_b2_256, dlm_add_select);
            __m256i dis_and_angle_b3 = _mm256_and_si256(dis_b3_256, dlm_add_select);
            __m256i tmp_val_and_angle_b1 = _mm256_andnot_si256(dlm_add_select, tmp_val_b1);
            __m256i tmp_val_and_angle_b2 = _mm256_andnot_si256(dlm_add_select, tmp_val_b2);
            __m256i tmp_val_and_angle_b3 = _mm256_andnot_si256(dlm_add_select, tmp_val_b3);
            __m256i dlm_rest_b1_256 = _mm256_add_epi16(dis_and_angle_b1, tmp_val_and_angle_b1);
            __m256i dlm_rest_b2_256 = _mm256_add_epi16(dis_and_angle_b2, tmp_val_and_angle_b2);
            __m256i dlm_rest_b3_256 = _mm256_add_epi16(dis_and_angle_b3, tmp_val_and_angle_b3);

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

            ref_b_ref_b1_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b1_lo, perm_for_64b_mul);
            ref_b1_lo = _mm256_permutevar8x32_epi32( ref_b1_lo, perm_for_64b_mul);
            ref_b_ref_b1_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b1_hi, perm_for_64b_mul);
            ref_b1_hi = _mm256_permutevar8x32_epi32( ref_b1_hi, perm_for_64b_mul);
            ref_b_ref_b2_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b2_lo, perm_for_64b_mul);
            ref_b2_lo = _mm256_permutevar8x32_epi32( ref_b2_lo, perm_for_64b_mul);
            ref_b_ref_b2_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b2_hi, perm_for_64b_mul);
            ref_b2_hi = _mm256_permutevar8x32_epi32( ref_b2_hi, perm_for_64b_mul);
            ref_b_ref_b3_lo = _mm256_permutevar8x32_epi32( ref_b_ref_b3_lo, perm_for_64b_mul);
            ref_b3_lo = _mm256_permutevar8x32_epi32( ref_b3_lo, perm_for_64b_mul);
            ref_b_ref_b3_hi = _mm256_permutevar8x32_epi32( ref_b_ref_b3_hi, perm_for_64b_mul);
            ref_b3_hi = _mm256_permutevar8x32_epi32( ref_b3_hi, perm_for_64b_mul);

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

            calc_angle_128(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[0], 0);
            calc_angle_128(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[1], 1);
            calc_angle_128(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[2], 2);
            calc_angle_128(ot_dp_lo, o_mag_sq_lo, t_mag_sq_lo, angle_flag_table[3], 3);
            calc_angle_128(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[4], 0);
            calc_angle_128(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[5], 1);
            calc_angle_128(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[6], 2);
            calc_angle_128(ot_dp_hi, o_mag_sq_hi, t_mag_sq_hi, angle_flag_table[7], 3);
            
            __m128i angle_128 = _mm_set_epi16(  angle_flag_table[7], angle_flag_table[6], angle_flag_table[5], angle_flag_table[4], \
                                                angle_flag_table[3], angle_flag_table[2], angle_flag_table[1], angle_flag_table[0]);
            __m128i dlm_add_select = _mm_mullo_epi16(angle_128, _mm_set1_epi16((int16_t)0xFFFF));
            
            __m128i dis_b3_128 = _mm_loadu_si128((__m128i*)(dist.bands[3] + index));
            __m128i ref_b3_128 = _mm_loadu_si128((__m128i*)(ref.bands[3] + index));

            __m128i adm_div_b1_lo, adm_div_b1_hi, adm_div_b2_lo, adm_div_b2_hi, adm_div_b3_lo, adm_div_b3_hi;

            // 0 2 1 3
            adm_div_b1_lo = _mm_set_epi32(  adm_div_lookup[ref.bands[1][index + 3] + 32768], adm_div_lookup[ref.bands[1][index + 1] + 32768], \
                                            adm_div_lookup[ref.bands[1][index + 2] + 32768], adm_div_lookup[ref.bands[1][index] + 32768]);
            // 4 6 5 7
            adm_div_b1_hi = _mm_set_epi32(  adm_div_lookup[ref.bands[1][index + 7] + 32768], adm_div_lookup[ref.bands[1][index + 5] + 32768], \
                                            adm_div_lookup[ref.bands[1][index + 6] + 32768], adm_div_lookup[ref.bands[1][index + 4] + 32768]);

            adm_div_b2_lo = _mm_set_epi32(  adm_div_lookup[ref.bands[2][index + 3] + 32768], adm_div_lookup[ref.bands[2][index + 1] + 32768], \
                                            adm_div_lookup[ref.bands[2][index + 2] + 32768], adm_div_lookup[ref.bands[2][index] + 32768]);

            adm_div_b2_hi = _mm_set_epi32(  adm_div_lookup[ref.bands[2][index + 7] + 32768], adm_div_lookup[ref.bands[2][index + 5] + 32768], \
                                            adm_div_lookup[ref.bands[2][index + 6] + 32768], adm_div_lookup[ref.bands[2][index + 4] + 32768]);

            adm_div_b3_lo = _mm_set_epi32(  adm_div_lookup[ref.bands[3][index + 3] + 32768], adm_div_lookup[ref.bands[3][index + 1] + 32768], \
                                            adm_div_lookup[ref.bands[3][index + 2] + 32768], adm_div_lookup[ref.bands[3][index] + 32768]);

            adm_div_b3_hi = _mm_set_epi32(  adm_div_lookup[ref.bands[3][index + 7] + 32768], adm_div_lookup[ref.bands[3][index + 5] + 32768], \
                                            adm_div_lookup[ref.bands[3][index + 6] + 32768], adm_div_lookup[ref.bands[3][index + 4] + 32768]);

            __m128i dis_b1_lo, dis_b1_hi, dis_b2_lo, dis_b2_hi, dis_b3_lo, dis_b3_hi;
            // 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
            cvt_1_16x8_to_2_32x4(dis_b1_128, dis_b1_lo, dis_b1_hi);
            cvt_1_16x8_to_2_32x4(dis_b2_128, dis_b2_lo, dis_b2_hi);
            cvt_1_16x8_to_2_32x4(dis_b3_128, dis_b3_lo, dis_b3_hi);

            // 0 2 1 3 | 4 6 5 7
            dis_b1_lo = _mm_shuffle_epi32(dis_b1_lo, 0xD8);
            dis_b1_hi = _mm_shuffle_epi32(dis_b1_hi, 0xD8);
            dis_b2_lo = _mm_shuffle_epi32(dis_b2_lo, 0xD8);
            dis_b2_hi = _mm_shuffle_epi32(dis_b2_hi, 0xD8);
            dis_b3_lo = _mm_shuffle_epi32(dis_b3_lo, 0xD8);
            dis_b3_hi = _mm_shuffle_epi32(dis_b3_hi, 0xD8);

            // 0 1
            __m128i adm_b1_dis_lo0 = _mm_mul_epi32(adm_div_b1_lo, dis_b1_lo);
            // 2 3
            __m128i adm_b1_dis_lo1 = _mm_mul_epi32(_mm_srli_epi64(adm_div_b1_lo, 32), _mm_srli_epi64(dis_b1_lo, 32));
            // 4 5
            __m128i adm_b1_dis_hi8 = _mm_mul_epi32(adm_div_b1_hi, dis_b1_hi);
            // 6 7
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
            
            __m128i ref_b1_lo, ref_b1_hi, ref_b2_lo, ref_b2_hi, ref_b3_lo, ref_b3_hi;
            // 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
            cvt_1_16x8_to_2_32x4(ref_b1_128, ref_b1_lo, ref_b1_hi);
            cvt_1_16x8_to_2_32x4(ref_b2_128, ref_b2_lo, ref_b2_hi);
            cvt_1_16x8_to_2_32x4(ref_b3_128, ref_b3_lo, ref_b3_hi);

            __m128i eqz_b1_lo = _mm_cmpeq_epi32(ref_b1_lo, _mm_setzero_si128());
            __m128i eqz_b1_hi = _mm_cmpeq_epi32(ref_b1_hi, _mm_setzero_si128());
            __m128i eqz_b2_lo = _mm_cmpeq_epi32(ref_b2_lo, _mm_setzero_si128());
            __m128i eqz_b2_hi = _mm_cmpeq_epi32(ref_b2_hi, _mm_setzero_si128());
            __m128i eqz_b3_lo = _mm_cmpeq_epi32(ref_b3_lo, _mm_setzero_si128());
            __m128i eqz_b3_hi = _mm_cmpeq_epi32(ref_b3_hi, _mm_setzero_si128());
            
            __m128i eqz_b1_lo0, eqz_b1_lo1, eqz_b1_hi0, eqz_b1_hi1, eqz_b2_lo0, eqz_b2_lo1, eqz_b2_hi0, eqz_b2_hi1, eqz_b3_lo0, eqz_b3_lo1, eqz_b3_hi0, eqz_b3_hi1;
            // 0 1 2 3 | 4 5 6 7
            cvt_1_32x4_to_2_64x2(eqz_b1_lo, eqz_b1_lo0, eqz_b1_lo1);
            cvt_1_32x4_to_2_64x2(eqz_b1_hi, eqz_b1_hi0, eqz_b1_hi1);
            cvt_1_32x4_to_2_64x2(eqz_b2_lo, eqz_b2_lo0, eqz_b2_lo1);
            cvt_1_32x4_to_2_64x2(eqz_b2_hi, eqz_b2_hi0, eqz_b2_hi1);
            cvt_1_32x4_to_2_64x2(eqz_b3_lo, eqz_b3_lo0, eqz_b3_lo1);
            cvt_1_32x4_to_2_64x2(eqz_b3_hi, eqz_b3_hi0, eqz_b3_hi1);

            adm_b1_dis_lo0 = _mm_andnot_si128(eqz_b1_lo0, adm_b1_dis_lo0);
            adm_b1_dis_lo1 = _mm_andnot_si128(eqz_b1_lo1, adm_b1_dis_lo1);
            adm_b1_dis_hi8 = _mm_andnot_si128(eqz_b1_hi0, adm_b1_dis_hi8);
            adm_b1_dis_hi9 = _mm_andnot_si128(eqz_b1_hi1, adm_b1_dis_hi9);

            adm_b2_dis_lo0 = _mm_andnot_si128(eqz_b2_lo0, adm_b2_dis_lo0);
            adm_b2_dis_lo1 = _mm_andnot_si128(eqz_b2_lo1, adm_b2_dis_lo1);
            adm_b2_dis_hi8 = _mm_andnot_si128(eqz_b2_hi0, adm_b2_dis_hi8);
            adm_b2_dis_hi9 = _mm_andnot_si128(eqz_b2_hi1, adm_b2_dis_hi9);

            adm_b3_dis_lo0 = _mm_andnot_si128(eqz_b3_lo0, adm_b3_dis_lo0);
            adm_b3_dis_lo1 = _mm_andnot_si128(eqz_b3_lo1, adm_b3_dis_lo1);
            adm_b3_dis_hi8 = _mm_andnot_si128(eqz_b3_hi0, adm_b3_dis_hi8);
            adm_b3_dis_hi9 = _mm_andnot_si128(eqz_b3_hi1, adm_b3_dis_hi9);

            // ref.bands[k][index] == 0 ? b[k]_add_[index] = add_32768 : 0
            __m128i b1_add_lo0 = _mm_and_si128(eqz_b1_lo0, add_32768_128);
            __m128i b1_add_lo1 = _mm_and_si128(eqz_b1_lo1, add_32768_128);
            __m128i b1_add_hi0 = _mm_and_si128(eqz_b1_hi0, add_32768_128);
            __m128i b1_add_hi1 = _mm_and_si128(eqz_b1_hi1, add_32768_128);

            __m128i b2_add_lo0 = _mm_and_si128(eqz_b2_lo0, add_32768_128);
            __m128i b2_add_lo1 = _mm_and_si128(eqz_b2_lo1, add_32768_128);
            __m128i b2_add_hi0 = _mm_and_si128(eqz_b2_hi0, add_32768_128);
            __m128i b2_add_hi1 = _mm_and_si128(eqz_b2_hi1, add_32768_128);

            __m128i b3_add_lo0 = _mm_and_si128(eqz_b3_lo0, add_32768_128);
            __m128i b3_add_lo1 = _mm_and_si128(eqz_b3_lo1, add_32768_128);
            __m128i b3_add_hi0 = _mm_and_si128(eqz_b3_hi0, add_32768_128);
            __m128i b3_add_hi1 = _mm_and_si128(eqz_b3_hi1, add_32768_128);

            // ref.bands[k][index] == 0 ? b[k]_add_[index] = add_32768 : b[k]_add_[index]
            adm_b1_dis_lo0 = _mm_add_epi64(b1_add_lo0, adm_b1_dis_lo0);
            adm_b1_dis_lo1 = _mm_add_epi64(b1_add_lo1, adm_b1_dis_lo1);
            adm_b1_dis_hi8 = _mm_add_epi64(b1_add_hi0, adm_b1_dis_hi8);
            adm_b1_dis_hi9 = _mm_add_epi64(b1_add_hi1, adm_b1_dis_hi9);

            adm_b2_dis_lo0 = _mm_add_epi64(b2_add_lo0, adm_b2_dis_lo0);
            adm_b2_dis_lo1 = _mm_add_epi64(b2_add_lo1, adm_b2_dis_lo1);
            adm_b2_dis_hi8 = _mm_add_epi64(b2_add_hi0, adm_b2_dis_hi8);
            adm_b2_dis_hi9 = _mm_add_epi64(b2_add_hi1, adm_b2_dis_hi9);

            adm_b3_dis_lo0 = _mm_add_epi64(b3_add_lo0, adm_b3_dis_lo0);
            adm_b3_dis_lo1 = _mm_add_epi64(b3_add_lo1, adm_b3_dis_lo1);
            adm_b3_dis_hi8 = _mm_add_epi64(b3_add_hi0, adm_b3_dis_hi8);
            adm_b3_dis_hi9 = _mm_add_epi64(b3_add_hi1, adm_b3_dis_hi9);

            __m128i tmp_k_b1_lo = _mm_shuffle_epi32( adm_b1_dis_lo0, 0x58);
            __m128i tmp_k_b1_hi = _mm_shuffle_epi32( adm_b1_dis_hi8, 0x58);
            __m128i tmp_k_b2_lo = _mm_shuffle_epi32( adm_b2_dis_lo0, 0x58);
            __m128i tmp_k_b2_hi = _mm_shuffle_epi32( adm_b2_dis_hi8, 0x58);
            __m128i tmp_k_b3_lo = _mm_shuffle_epi32( adm_b3_dis_lo0, 0x58);
            __m128i tmp_k_b3_hi = _mm_shuffle_epi32( adm_b3_dis_hi8, 0x58);

            tmp_k_b1_lo = _mm_add_epi32(tmp_k_b1_lo, _mm_shuffle_epi32( adm_b1_dis_lo1, 0x85));
            tmp_k_b1_hi = _mm_add_epi32(tmp_k_b1_hi, _mm_shuffle_epi32( adm_b1_dis_hi9, 0x85));
            tmp_k_b2_lo = _mm_add_epi32(tmp_k_b2_lo, _mm_shuffle_epi32( adm_b2_dis_lo1, 0x85));
            tmp_k_b2_hi = _mm_add_epi32(tmp_k_b2_hi, _mm_shuffle_epi32( adm_b2_dis_hi9, 0x85));
            tmp_k_b3_lo = _mm_add_epi32(tmp_k_b3_lo, _mm_shuffle_epi32( adm_b3_dis_lo1, 0x85));
            tmp_k_b3_hi = _mm_add_epi32(tmp_k_b3_hi, _mm_shuffle_epi32( adm_b3_dis_hi9, 0x85));

            __m128i tmp_k_b1_lo_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b1_lo);
            __m128i tmp_k_b1_hi_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b1_hi);
            __m128i tmp_k_b2_lo_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b2_lo);
            __m128i tmp_k_b2_hi_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b2_hi);
            __m128i tmp_k_b3_lo_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b3_lo);
            __m128i tmp_k_b3_hi_eqz = _mm_cmpgt_epi32(_mm_setzero_si128(), tmp_k_b3_hi);

            tmp_k_b1_lo = _mm_andnot_si128(tmp_k_b1_lo_eqz, tmp_k_b1_lo);
            tmp_k_b1_hi = _mm_andnot_si128(tmp_k_b1_hi_eqz, tmp_k_b1_hi);
            tmp_k_b2_lo = _mm_andnot_si128(tmp_k_b2_lo_eqz, tmp_k_b2_lo);
            tmp_k_b2_hi = _mm_andnot_si128(tmp_k_b2_hi_eqz, tmp_k_b2_hi);
            tmp_k_b3_lo = _mm_andnot_si128(tmp_k_b3_lo_eqz, tmp_k_b3_lo);
            tmp_k_b3_hi = _mm_andnot_si128(tmp_k_b3_hi_eqz, tmp_k_b3_hi);



            __m128i tmp_k_b1_lo_gt = _mm_cmpgt_epi32(tmp_k_b1_lo, add_32768_32b_128);
            __m128i tmp_k_b1_hi_gt = _mm_cmpgt_epi32(tmp_k_b1_hi, add_32768_32b_128);
            __m128i tmp_k_b2_lo_gt = _mm_cmpgt_epi32(tmp_k_b2_lo, add_32768_32b_128);
            __m128i tmp_k_b2_hi_gt = _mm_cmpgt_epi32(tmp_k_b2_hi, add_32768_32b_128);
            __m128i tmp_k_b3_lo_gt = _mm_cmpgt_epi32(tmp_k_b3_lo, add_32768_32b_128);
            __m128i tmp_k_b3_hi_gt = _mm_cmpgt_epi32(tmp_k_b3_hi, add_32768_32b_128);

            tmp_k_b1_lo = _mm_andnot_si128(tmp_k_b1_lo_gt, tmp_k_b1_lo);
            tmp_k_b1_hi = _mm_andnot_si128(tmp_k_b1_hi_gt, tmp_k_b1_hi);
            tmp_k_b2_lo = _mm_andnot_si128(tmp_k_b2_lo_gt, tmp_k_b2_lo);
            tmp_k_b2_hi = _mm_andnot_si128(tmp_k_b2_hi_gt, tmp_k_b2_hi);
            tmp_k_b3_lo = _mm_andnot_si128(tmp_k_b3_lo_gt, tmp_k_b3_lo);
            tmp_k_b3_hi = _mm_andnot_si128(tmp_k_b3_hi_gt, tmp_k_b3_hi);

            __m128i add_b1_lo = _mm_and_si128(tmp_k_b1_lo_gt, add_32768_32b_128);
            __m128i add_b1_hi = _mm_and_si128(tmp_k_b1_hi_gt, add_32768_32b_128);
            __m128i add_b2_lo = _mm_and_si128(tmp_k_b2_lo_gt, add_32768_32b_128);
            __m128i add_b2_hi = _mm_and_si128(tmp_k_b2_hi_gt, add_32768_32b_128);
            __m128i add_b3_lo = _mm_and_si128(tmp_k_b3_lo_gt, add_32768_32b_128);
            __m128i add_b3_hi = _mm_and_si128(tmp_k_b3_hi_gt, add_32768_32b_128);
         
            tmp_k_b1_lo = _mm_add_epi32(tmp_k_b1_lo, add_b1_lo);
            tmp_k_b1_hi = _mm_add_epi32(tmp_k_b1_hi, add_b1_hi);
            tmp_k_b2_lo = _mm_add_epi32(tmp_k_b2_lo, add_b2_lo);
            tmp_k_b2_hi = _mm_add_epi32(tmp_k_b2_hi, add_b2_hi);
            tmp_k_b3_lo = _mm_add_epi32(tmp_k_b3_lo, add_b3_lo);
            tmp_k_b3_hi = _mm_add_epi32(tmp_k_b3_hi, add_b3_hi);

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

            __m128i dis_and_angle_b1 = _mm_and_si128(dis_b1_128, dlm_add_select);
            __m128i dis_and_angle_b2 = _mm_and_si128(dis_b2_128, dlm_add_select);
            __m128i dis_and_angle_b3 = _mm_and_si128(dis_b3_128, dlm_add_select);
            __m128i tmp_val_and_angle_b1 = _mm_andnot_si128(dlm_add_select, tmp_val_b1);
            __m128i tmp_val_and_angle_b2 = _mm_andnot_si128(dlm_add_select, tmp_val_b2);
            __m128i tmp_val_and_angle_b3 = _mm_andnot_si128(dlm_add_select, tmp_val_b3);
            __m128i dlm_rest_b1_256 = _mm_add_epi16(dis_and_angle_b1, tmp_val_and_angle_b1);
            __m128i dlm_rest_b2_256 = _mm_add_epi16(dis_and_angle_b2, tmp_val_and_angle_b2);
            __m128i dlm_rest_b3_256 = _mm_add_epi16(dis_and_angle_b3, tmp_val_and_angle_b3);

            __m128i dist_m_dlm_rest_b1 = _mm_abs_epi16(_mm_sub_epi16(dis_b1_128, dlm_rest_b1_256));
            __m128i dist_m_dlm_rest_b2 = _mm_abs_epi16(_mm_sub_epi16(dis_b2_128, dlm_rest_b2_256));
            __m128i dlm_add_256 = _mm_adds_epu16(dist_m_dlm_rest_b1, dist_m_dlm_rest_b2);
            __m128i dist_m_dlm_rest_b3 = _mm_abs_epi16(_mm_sub_epi16(dis_b3_128, dlm_rest_b3_256));            
            dlm_add_256 = _mm_adds_epu16(dlm_add_256, dist_m_dlm_rest_b3);

            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[1] + restIndex), dlm_rest_b1_256);
            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[2] + restIndex), dlm_rest_b2_256);
            _mm_storeu_si128((__m128i*)(i_dlm_rest.bands[3] + restIndex), dlm_rest_b3_256);

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
    *adm_score_den = (den_band * 30) + 1e-4;

}