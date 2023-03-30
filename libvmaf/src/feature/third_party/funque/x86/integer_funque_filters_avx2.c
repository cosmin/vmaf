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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mem.h"
#include "../offset.h"
#include "../integer_funque_filters.h"
#include <immintrin.h>

void integer_funque_dwt2_avx2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);

    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    const int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    const int8_t filter_shift_rnd = 1<<(filter_shift - 1);

    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    const int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    const int8_t filter_shift_lcpad_rnd = 1<<(filter_shift_lcpad - 1);

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];

    int16_t row_idx0, row_idx1, col_idx0;
	
	int row0_offset, row1_offset;
    
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;

    int i, j;

	int width_rem_size16 = width_div_2 - (width_div_2 % 16);
	int width_rem_size8 = width_div_2 - (width_div_2 % 8);
	int width_rem_size4 = width_div_2 - (width_div_2 % 4);

	__m256i filter_shift_256 = _mm256_set1_epi32(filter_shift);
	__m256i idx_perm = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
	__m256i zero_256 = _mm256_setzero_si256();

	__m128i filter_shift_128 = _mm_set1_epi32(filter_shift);
	__m128i zero_128 = _mm_setzero_si128();

    for (i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;
		row0_offset = (row_idx0)*width;
		row1_offset = (row_idx1)*width;
        j = 0;
		for(; j< width_rem_size16; j+=16)
		{
			int col_idx0 = (j << 1);

			__m256i src_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0));
			__m256i src_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0));
			__m256i src2_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0 + 16));
			__m256i src2_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0 + 16));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_a_256));
			__m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_a_256, 1));
			__m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_b_256));
			__m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_b_256, 1));
			__m256i a2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src2_a_256));
			__m256i a2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src2_a_256, 1));
			__m256i b2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src2_b_256));
			__m256i b2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src2_b_256, 1));

			__m256i a_p_b_c_p_d_lo = _mm256_add_epi32(a_lo, b_lo);
			__m256i a_p_b_c_p_d_hi = _mm256_add_epi32(a_hi, b_hi);
			__m256i a_m_b_c_m_d_lo = _mm256_sub_epi32(a_lo, b_lo);
			__m256i a_m_b_c_m_d_hi = _mm256_sub_epi32(a_hi, b_hi);
			__m256i a_p_b_c_p_d_2_lo = _mm256_add_epi32(a2_lo, b2_lo);
			__m256i a_p_b_c_p_d_2_hi = _mm256_add_epi32(a2_hi, b2_hi);
			__m256i a_m_b_c_m_d_2_lo = _mm256_sub_epi32(a2_lo, b2_lo);
			__m256i a_m_b_c_m_d_2_hi = _mm256_sub_epi32(a2_hi, b2_hi);;

			__m256i band_a_256 = _mm256_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m256i band_v_256 = _mm256_hsub_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m256i band_h_256 = _mm256_hadd_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);
			__m256i band_d_256 = _mm256_hsub_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);
			__m256i band_a2_256 = _mm256_hadd_epi32(a_p_b_c_p_d_2_lo, a_p_b_c_p_d_2_hi);
			__m256i band_v2_256 = _mm256_hsub_epi32(a_p_b_c_p_d_2_lo, a_p_b_c_p_d_2_hi);
			__m256i band_h2_256 = _mm256_hadd_epi32(a_m_b_c_m_d_2_lo, a_m_b_c_m_d_2_hi);
			__m256i band_d2_256 = _mm256_hsub_epi32(a_m_b_c_m_d_2_lo, a_m_b_c_m_d_2_hi);

			band_a_256 = _mm256_add_epi32(band_a_256, filter_shift_256);
			band_v_256 = _mm256_add_epi32(band_v_256, filter_shift_256);
			band_h_256 = _mm256_add_epi32(band_h_256, filter_shift_256);
			band_d_256 = _mm256_add_epi32(band_d_256, filter_shift_256);
			band_a2_256 = _mm256_add_epi32(band_a2_256, filter_shift_256);
			band_h2_256 = _mm256_add_epi32(band_h2_256, filter_shift_256);
			band_v2_256 = _mm256_add_epi32(band_v2_256, filter_shift_256);
			band_d2_256 = _mm256_add_epi32(band_d2_256, filter_shift_256);

			band_a_256 = _mm256_srai_epi32(band_a_256, filter_shift_rnd);
			band_a2_256 = _mm256_srai_epi32(band_a2_256, filter_shift_rnd);
			band_h_256 = _mm256_srai_epi32(band_h_256, filter_shift_rnd);
			band_h2_256 = _mm256_srai_epi32(band_h2_256, filter_shift_rnd);
			band_v_256 = _mm256_srai_epi32(band_v_256, filter_shift_rnd);
			band_v2_256 = _mm256_srai_epi32(band_v2_256, filter_shift_rnd);
			band_d_256 = _mm256_srai_epi32(band_d_256, filter_shift_rnd);
			band_d2_256 = _mm256_srai_epi32(band_d2_256, filter_shift_rnd);

			band_a_256 = _mm256_packs_epi32(band_a_256, band_a2_256);
			band_h_256 = _mm256_packs_epi32(band_h_256, band_h2_256);
			band_v_256 = _mm256_packs_epi32(band_v_256, band_v2_256);
			band_d_256 = _mm256_packs_epi32(band_d_256, band_d2_256);

			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm);
			band_h_256 = _mm256_permutevar8x32_epi32(band_h_256, idx_perm);
			band_v_256 = _mm256_permutevar8x32_epi32(band_v_256, idx_perm);
			band_d_256 = _mm256_permutevar8x32_epi32(band_d_256, idx_perm);
			
			_mm256_storeu_si256((__m256i*)(band_a + i * dst_px_stride + j), band_a_256);
			_mm256_storeu_si256((__m256i*)(band_h + i * dst_px_stride + j), band_h_256);
			_mm256_storeu_si256((__m256i*)(band_v + i * dst_px_stride + j), band_v_256);
			_mm256_storeu_si256((__m256i*)(band_d + i * dst_px_stride + j), band_d_256);
        }

		for(; j< width_rem_size8; j+=8)
		{
			int col_idx0 = (j << 1);

			__m256i src_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0));
			__m256i src_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_a_256));
			__m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_a_256, 1));
			__m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_b_256));
			__m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_b_256, 1));

			__m256i a_p_b_c_p_d_lo = _mm256_add_epi32(a_lo, b_lo);
			__m256i a_p_b_c_p_d_hi = _mm256_add_epi32(a_hi, b_hi);
			__m256i a_m_b_c_m_d_lo = _mm256_sub_epi32(a_lo, b_lo);
			__m256i a_m_b_c_m_d_hi = _mm256_sub_epi32(a_hi, b_hi);

			__m256i band_a_256 = _mm256_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m256i band_v_256 = _mm256_hsub_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m256i band_h_256 = _mm256_hadd_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);
			__m256i band_d_256 = _mm256_hsub_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);

			band_a_256 = _mm256_add_epi32(band_a_256, filter_shift_256);
			band_v_256 = _mm256_add_epi32(band_v_256, filter_shift_256);
			band_h_256 = _mm256_add_epi32(band_h_256, filter_shift_256);
			band_d_256 = _mm256_add_epi32(band_d_256, filter_shift_256);

			band_a_256 = _mm256_srai_epi32(band_a_256, filter_shift_rnd);
			band_h_256 = _mm256_srai_epi32(band_h_256, filter_shift_rnd);
			band_v_256 = _mm256_srai_epi32(band_v_256, filter_shift_rnd);
			band_d_256 = _mm256_srai_epi32(band_d_256, filter_shift_rnd);

			band_a_256 = _mm256_packs_epi32(band_a_256, zero_256);
			band_h_256 = _mm256_packs_epi32(band_h_256, zero_256);
			band_v_256 = _mm256_packs_epi32(band_v_256, zero_256);
			band_d_256 = _mm256_packs_epi32(band_d_256, zero_256);

			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm);
			band_h_256 = _mm256_permutevar8x32_epi32(band_h_256, idx_perm);
			band_v_256 = _mm256_permutevar8x32_epi32(band_v_256, idx_perm);
			band_d_256 = _mm256_permutevar8x32_epi32(band_d_256, idx_perm);
			
			_mm_storeu_si128((__m128i*)(band_a + i * dst_px_stride + j), _mm256_castsi256_si128(band_a_256));
			_mm_storeu_si128((__m128i*)(band_h + i * dst_px_stride + j), _mm256_castsi256_si128(band_h_256));
			_mm_storeu_si128((__m128i*)(band_v + i * dst_px_stride + j), _mm256_castsi256_si128(band_v_256));
			_mm_storeu_si128((__m128i*)(band_d + i * dst_px_stride + j), _mm256_castsi256_si128(band_d_256));
        }

		for(; j< width_rem_size4; j+=4)
		{
			int col_idx0 = (j << 1);

			__m128i src_a_128 = _mm_loadu_si128((__m128i*)(src + row0_offset + col_idx0));
			__m128i src_b_128 = _mm_loadu_si128((__m128i*)(src + row1_offset + col_idx0));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m128i a_lo = _mm_cvtepi16_epi32( _mm_unpacklo_epi64(src_a_128, zero_128));
			__m128i a_hi = _mm_cvtepi16_epi32( _mm_unpackhi_epi64(src_a_128, zero_128));
			__m128i b_lo = _mm_cvtepi16_epi32( _mm_unpacklo_epi64(src_b_128, zero_128));
			__m128i b_hi = _mm_cvtepi16_epi32( _mm_unpackhi_epi64(src_b_128, zero_128));

			__m128i a_p_b_c_p_d_lo = _mm_add_epi32(a_lo, b_lo);
			__m128i a_p_b_c_p_d_hi = _mm_add_epi32(a_hi, b_hi);
			__m128i a_m_b_c_m_d_lo = _mm_sub_epi32(a_lo, b_lo);
			__m128i a_m_b_c_m_d_hi = _mm_sub_epi32(a_hi, b_hi);

			__m128i band_a_128 = _mm_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m128i band_h_128 = _mm_hadd_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);
			__m128i band_v_128 = _mm_hsub_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m128i band_d_128 = _mm_hsub_epi32(a_m_b_c_m_d_lo, a_m_b_c_m_d_hi);

			band_a_128 = _mm_add_epi32(band_a_128, filter_shift_128);
			band_h_128 = _mm_add_epi32(band_h_128, filter_shift_128);
			band_v_128 = _mm_add_epi32(band_v_128, filter_shift_128);
			band_d_128 = _mm_add_epi32(band_d_128, filter_shift_128);

			band_a_128 = _mm_srai_epi32(band_a_128, filter_shift_rnd);
			band_h_128 = _mm_srai_epi32(band_h_128, filter_shift_rnd);
			band_v_128 = _mm_srai_epi32(band_v_128, filter_shift_rnd);
			band_d_128 = _mm_srai_epi32(band_d_128, filter_shift_rnd);

			band_a_128 = _mm_packs_epi32(band_a_128, zero_128);
			band_h_128 = _mm_packs_epi32(band_h_128, zero_128);
			band_v_128 = _mm_packs_epi32(band_v_128, zero_128);
			band_d_128 = _mm_packs_epi32(band_d_128, zero_128);
			
			_mm_storel_epi64((__m128i*)(band_a + i * dst_px_stride + j), band_a_128);
			_mm_storel_epi64((__m128i*)(band_h + i * dst_px_stride + j), band_h_128);
			_mm_storel_epi64((__m128i*)(band_v + i * dst_px_stride + j), band_v_128);
			_mm_storel_epi64((__m128i*)(band_d + i * dst_px_stride + j), band_d_128);
        }

		for(; j< width_div_2; ++j)
		{
			int col_idx0 = (j << 1);
			int col_idx1 = (j << 1) + 1;
			
			// a & b 2 values in adjacent rows at the same coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			// c & d are adjacent values to a & b in teh same row
			spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
			spat_fil_output_dtype src_d = src[row1_offset + col_idx1];

			//a + b	& a - b	
			int32_t src_a_p_b = src_a + src_b;
			int32_t src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int32_t src_c_p_d = src_c + src_d;
			int32_t src_c_m_d = src_c - src_d;

			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift);
			
			//F* F (a - b + c - d) - band H  (F*F is 1/2)
            band_h[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_m_b + src_c_m_d) + filter_shift_rnd) >> filter_shift);
			
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
            band_v[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b - src_c_p_d) + filter_shift_rnd) >> filter_shift);

			//F* F (a - b - c - d) - band D  (F*F is 1/2)
            band_d[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_m_b - src_c_m_d) + filter_shift_rnd) >> filter_shift);
        }

        if(last_col)
        {
			col_idx0 = width_div_2 << 1;
			j = width_div_2;
			
			// a & b 2 values in adjacent rows at the last coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			int src_a_m_b = src_a - src_b;
			
            //F* F (a + b + a + b) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_p_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
			
			//F* F (a - b + a - b) - band H  (F*F is 1/2)
            band_h[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_m_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
			
			//F* F (a + b - (a + b)) - band V, Last column V will always be 0            
            band_v[i*dst_px_stride+j] = 0;

			//F* F (a - b - (a -b)) - band D,  Last column D will always be 0
            band_d[i*dst_px_stride+j] = 0;
        }
    }
}

void integer_funque_vifdwt2_band0_avx2(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);

    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    const int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    const int8_t filter_shift_rnd = 1<<(filter_shift - 1);

    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    const int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    const int8_t filter_shift_lcpad_rnd = 1<<(filter_shift_lcpad - 1);

    int16_t row_idx0, row_idx1, col_idx0;
	// int16_t col_idx1;
	int row0_offset, row1_offset;
    // int64_t accum;
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;

    int i, j;

	int width_rem_size16 = width_div_2 - (width_div_2 % 16);
	int width_rem_size8 = width_div_2 - (width_div_2 % 8);
	int width_rem_size4 = width_div_2 - (width_div_2 % 4);

	__m256i filter_shift_256 = _mm256_set1_epi32(filter_shift);
	__m256i idx_perm = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
	__m256i zero_256 = _mm256_setzero_si256();

	__m128i filter_shift_128 = _mm_set1_epi32(filter_shift);
	__m128i zero_128 = _mm_setzero_si128();
    
	for (i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;
		row0_offset = (row_idx0)*width;
		row1_offset = (row_idx1)*width;
        j = 0;
		for(; j< width_rem_size16; j+=16)
		{
			int col_idx0 = (j << 1);

			__m256i src_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0));
			__m256i src_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0));
			__m256i src2_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0 + 16));
			__m256i src2_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0 + 16));

			__m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_a_256));
			__m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_a_256, 1));
			__m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_b_256));
			__m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_b_256, 1));
			__m256i a2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src2_a_256));
			__m256i a2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src2_a_256, 1));
			__m256i b2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src2_b_256));
			__m256i b2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src2_b_256, 1));

			__m256i a_p_b_c_p_d_lo = _mm256_add_epi32(a_lo, b_lo);
			__m256i a_p_b_c_p_d_hi = _mm256_add_epi32(a_hi, b_hi);
			__m256i a_p_b_c_p_d_2_lo = _mm256_add_epi32(a2_lo, b2_lo);
			__m256i a_p_b_c_p_d_2_hi = _mm256_add_epi32(a2_hi, b2_hi);

			__m256i band_a_256 = _mm256_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			__m256i band_a2_256 = _mm256_hadd_epi32(a_p_b_c_p_d_2_lo, a_p_b_c_p_d_2_hi);

			band_a_256 = _mm256_add_epi32(band_a_256, filter_shift_256);
			band_a2_256 = _mm256_add_epi32(band_a2_256, filter_shift_256);
			band_a_256 = _mm256_srai_epi32(band_a_256, filter_shift_rnd);
			band_a2_256 = _mm256_srai_epi32(band_a2_256, filter_shift_rnd);

			band_a_256 = _mm256_packs_epi32(band_a_256, band_a2_256);
			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm);

			_mm256_storeu_si256((__m256i*)(band_a + i * dst_px_stride + j), band_a_256);
        }

		for(; j< width_rem_size8; j+=8)
		{
			int col_idx0 = (j << 1);

			__m256i src_a_256 = _mm256_loadu_si256((__m256i*)(src + row0_offset + col_idx0));
			__m256i src_b_256 = _mm256_loadu_si256((__m256i*)(src + row1_offset + col_idx0));

			__m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_a_256));
			__m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_a_256, 1));
			__m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src_b_256));
			__m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src_b_256, 1));

			__m256i a_p_b_c_p_d_lo = _mm256_add_epi32(a_lo, b_lo);
			__m256i a_p_b_c_p_d_hi = _mm256_add_epi32(a_hi, b_hi);

			__m256i band_a_256 = _mm256_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);

			band_a_256 = _mm256_add_epi32(band_a_256, filter_shift_256);
			band_a_256 = _mm256_srai_epi32(band_a_256, filter_shift_rnd);

			band_a_256 = _mm256_packs_epi32(band_a_256, zero_256);
			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm);

			_mm256_storeu_si256((__m256i*)(band_a + i * dst_px_stride + j), band_a_256);
        }

		for(; j< width_rem_size4; j+=4)
		{
			int col_idx0 = (j << 1);

			__m128i src_a_128 = _mm_loadu_si128((__m128i*)(src + row0_offset + col_idx0));
			__m128i src_b_128 = _mm_loadu_si128((__m128i*)(src + row1_offset + col_idx0));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m128i a_lo = _mm_cvtepi16_epi32( _mm_unpacklo_epi64(src_a_128, zero_128));
			__m128i a_hi = _mm_cvtepi16_epi32( _mm_unpackhi_epi64(src_a_128, zero_128));
			__m128i b_lo = _mm_cvtepi16_epi32( _mm_unpacklo_epi64(src_b_128, zero_128));
			__m128i b_hi = _mm_cvtepi16_epi32( _mm_unpackhi_epi64(src_b_128, zero_128));

			__m128i a_p_b_c_p_d_lo = _mm_add_epi32(a_lo, b_lo);
			__m128i a_p_b_c_p_d_hi = _mm_add_epi32(a_hi, b_hi);

			__m128i band_a_128 = _mm_hadd_epi32(a_p_b_c_p_d_lo, a_p_b_c_p_d_hi);
			
			band_a_128 = _mm_add_epi32(band_a_128, filter_shift_128);
			band_a_128 = _mm_srai_epi32(band_a_128, filter_shift_rnd);

			band_a_128 = _mm_packs_epi32(band_a_128, zero_128);
			
			_mm_storel_epi64((__m128i*)(band_a + i * dst_px_stride + j), band_a_128);
        }

		for(; j< width_div_2; ++j)
		{
			int col_idx0 = (j << 1);
			int col_idx1 = (j << 1) + 1;
			
			// a & b 2 values in adjacent rows at the same coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			// c & d are adjacent values to a & b in teh same row
			spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
			spat_fil_output_dtype src_d = src[row1_offset + col_idx1];
			
			//a + b	& a - b	
			int32_t src_a_p_b = src_a + src_b;
			// int32_t src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int32_t src_c_p_d = src_c + src_d;
			// int32_t src_c_m_d = src_c - src_d;

			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift);
        }

        if(last_col)
        {
			col_idx0 = width_div_2 << 1;
			j = width_div_2;
			
			// a & b 2 values in adjacent rows at the last coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			
            //F* F (a + b + a + b) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_p_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
        }
    }
}

/**
 * This function applies intermediate horizontal pass filter inside spatial filter
 */

static inline void integer_horizontal_filter_avx2(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst, const spat_fil_coeff_dtype *i_filter_coeffs, int width, int fwidth, int dst_row_idx, int half_fw)
{
    int j, fj, jj1, jj2;
    __m256i res0_256, res4_256, res8_256, res12_256;
	__m128i res0_128, res4_128, res8_128, res12_128;

	int width_rem_size32 = (width - half_fw) - ((width - 2*half_fw) % 32);
	int width_rem_size16 = (width - half_fw) - ((width - 2*half_fw) % 16);
	int width_rem_size8 = (width - half_fw) - ((width - 2*half_fw) % 8);

	const spat_fil_coeff_dtype i_filter_coeffs_with_zeros[51] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -900, -1054, -1239, -1452, -1669, -1798, -1547, -66, 4677, 14498, 21495,
        14498, 4677, -66, -1547, -1798, -1669, -1452, -1239, -1054, -900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
	const spat_fil_accum_dtype i32_filter_coeffs[11] = {
        -900 + (spat_fil_accum_dtype)(((unsigned int)-1054) << 16) + (1 << 16),
		-1239 + (spat_fil_accum_dtype)(((unsigned int)-1452) << 16) + (1 << 16),
		-1669 + (spat_fil_accum_dtype)(((unsigned int)-1798) << 16) + (1 << 16),
		-1547 + (spat_fil_accum_dtype)(((unsigned int)-66) << 16) + (1 << 16),
		4677 + (14498 << 16) /* + (1 << 16) */,
		21495 + (14498 << 16) /* + (1 << 16) */,
		4677 + (spat_fil_accum_dtype)(((unsigned int)-66) << 16) /* + (1 << 16) */,
		-1547 + (spat_fil_accum_dtype)(((unsigned int)-1798) << 16) + (1 << 16),
		-1669 + (spat_fil_accum_dtype)(((unsigned int)-1452) << 16) + (1 << 16),
		-1239 + (spat_fil_accum_dtype)(((unsigned int)-1054) << 16) + (1 << 16),
		-900 + (1 << 16)
    };
	(void)fwidth;

	__m256i d0 = _mm256_load_si256((__m256i*)(tmp));
	__m256i d1 = _mm256_load_si256((__m256i*)(tmp + 16));

	int half_filter_table_w = 25;
	for (j = 0; j < (half_fw / 2) + 1; j++)
	{
		int fi0 = half_filter_table_w - j;
		int fi1 =  j + half_filter_table_w + 1;
		__m256i coef0 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi0));
		__m256i coef1 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi1));
			
		__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef0);
		__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef0);

		__m256i mul1_lo_256 = _mm256_mullo_epi16(d0, coef1);
		__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0, coef1);

		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
		
		__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
		__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

		tmp0_lo = _mm256_add_epi32(tmp0_lo, tmp0_hi);
		tmp0_hi = _mm256_add_epi32(tmp1_lo, tmp1_hi);
		
		res0_256 = _mm256_add_epi32(tmp0_lo, tmp0_hi);

		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(res0_256), _mm256_extracti128_si256(res0_256, 1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);
		int r = _mm_cvtsi128_si32(r1);
		dst[dst_row_idx + j] = (spat_fil_output_dtype) ((r + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}

	for (; j < half_fw; j++)
	{
		int fi0 = half_filter_table_w - j;
		int fi1 =  j + half_filter_table_w + 1;
		int fi2 =  fi0 + 16;

		__m256i coef0 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi0));
		__m256i coef1 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi1));
		__m256i coef2 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi2));

		coef0 = _mm256_add_epi16(coef0, coef1);

		__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef0);
		__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef0);

		__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef2);
		__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef2);

		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
		__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

		tmp0_lo = _mm256_add_epi32(tmp0_lo, tmp0_hi);
		tmp0_hi = _mm256_add_epi32(tmp1_lo, tmp1_hi);

		res0_256 = _mm256_add_epi32(tmp0_lo, tmp0_hi);
		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(res0_256), _mm256_extracti128_si256(res0_256, 1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);
		int r = _mm_cvtsi128_si32(r1);
		dst[dst_row_idx + j] = (spat_fil_output_dtype) ((r + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}
	    
	//This is the core loop
	__m256i coef0_256 = _mm256_set1_epi16(i_filter_coeffs[0]);
	for (; j < width_rem_size32; j+=32)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(SPAT_FILTER_OUT_RND);

		for (fj = 0; fj < half_fw; fj+=2){
			jj1 = f_l_j + fj*2;

			__m256i coef0 = _mm256_set1_epi32(i32_filter_coeffs[fj]);
			__m256i coef1 = _mm256_set1_epi32(i32_filter_coeffs[fj+1]);

			__m256i d0 = _mm256_loadu_si256((__m256i*)(tmp + jj1));
			__m256i d2 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 2));
			__m256i d1 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 1));
			__m256i d3 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 3));

			__m256i d0_16 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 16));
			__m256i d2_16 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 18));
			__m256i d1_16 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 17));
			__m256i d3_16 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 19));

			res0_256 = _mm256_add_epi32(res0_256, _mm256_madd_epi16(d0, coef0));
			res0_256 = _mm256_add_epi32(res0_256, _mm256_madd_epi16(d2, coef1));
			res4_256 = _mm256_add_epi32(res4_256, _mm256_madd_epi16(d1, coef0));
			res4_256 = _mm256_add_epi32(res4_256, _mm256_madd_epi16(d3, coef1));

			res8_256 = _mm256_add_epi32(res8_256, _mm256_madd_epi16(d0_16, coef0));
			res8_256 = _mm256_add_epi32(res8_256, _mm256_madd_epi16(d2_16, coef1));
			res12_256 = _mm256_add_epi32(res12_256, _mm256_madd_epi16(d1_16, coef0));
			res12_256 = _mm256_add_epi32(res12_256, _mm256_madd_epi16(d3_16, coef1));
		}
		__m256i d0 = _mm256_loadu_si256((__m256i*)(tmp + f_r_j));
		__m256i d16 = _mm256_loadu_si256((__m256i*)(tmp + f_r_j + 16));
		
		__m256i tmp0 = _mm256_unpacklo_epi32(res0_256, res4_256);
		__m256i tmp4 = _mm256_unpackhi_epi32(res0_256, res4_256);
		__m256i tmp16 = _mm256_unpacklo_epi32(res8_256, res12_256);
		__m256i tmp20 = _mm256_unpackhi_epi32(res8_256, res12_256);
		
		__m256i mul0_lo = _mm256_mullo_epi16(d0, coef0_256);
		__m256i mul0_hi = _mm256_mulhi_epi16(d0, coef0_256);
		__m256i mul16_lo = _mm256_mullo_epi16(d16, coef0_256);
		__m256i mul16_hi = _mm256_mulhi_epi16(d16, coef0_256);

		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);
		__m256i tmp16_lo = _mm256_unpacklo_epi16(mul16_lo, mul16_hi);
		__m256i tmp16_hi = _mm256_unpackhi_epi16(mul16_lo, mul16_hi);

		tmp0 = _mm256_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm256_add_epi32(tmp4, tmp0_hi);
		tmp16 = _mm256_add_epi32(tmp16, tmp16_lo);
		tmp20 = _mm256_add_epi32(tmp20, tmp16_hi);

		tmp0 = _mm256_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm256_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		tmp16 = _mm256_srai_epi32(tmp16, SPAT_FILTER_OUT_SHIFT);
		tmp20 = _mm256_srai_epi32(tmp20, SPAT_FILTER_OUT_SHIFT);
		
		res0_256 = _mm256_packs_epi32(tmp0, tmp4);
		res8_256 = _mm256_packs_epi32(tmp16, tmp20);

		_mm256_storeu_si256((__m256i*)(dst + dst_row_idx + j), res0_256);
		_mm256_storeu_si256((__m256i*)(dst + dst_row_idx + j + 16), res8_256);
	}
	
	for (; j < width_rem_size16; j+=16)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(SPAT_FILTER_OUT_RND);

		for (fj = 0; fj < half_fw; fj+=2){
			jj1 = f_l_j + fj*2;

			__m256i coef0 = _mm256_set1_epi32(i32_filter_coeffs[fj]);
			__m256i coef1 = _mm256_set1_epi32(i32_filter_coeffs[fj+1]);

			__m256i d0 = _mm256_loadu_si256((__m256i*)(tmp + jj1));
			__m256i d2 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 2));
			__m256i d1 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 1));
			__m256i d3 = _mm256_loadu_si256((__m256i*)(tmp + jj1 + 3));

			res0_256 = _mm256_add_epi32(res0_256, _mm256_madd_epi16(d0, coef0));
			res0_256 = _mm256_add_epi32(res0_256, _mm256_madd_epi16(d2, coef1));
			res4_256 = _mm256_add_epi32(res4_256, _mm256_madd_epi16(d1, coef0));
			res4_256 = _mm256_add_epi32(res4_256, _mm256_madd_epi16(d3, coef1));
		}
		__m256i d0 = _mm256_loadu_si256((__m256i*)(tmp + f_r_j));
		__m256i tmp0 = _mm256_unpacklo_epi32(res0_256, res4_256);
		__m256i tmp4 = _mm256_unpackhi_epi32(res0_256, res4_256);
		__m256i mul0_lo = _mm256_mullo_epi16(d0, coef0_256);
		__m256i mul0_hi = _mm256_mulhi_epi16(d0, coef0_256);
		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);

		tmp0 = _mm256_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm256_add_epi32(tmp4, tmp0_hi);

		tmp0 = _mm256_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm256_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		
		res0_256 = _mm256_packs_epi32(tmp0, tmp4);
		_mm256_storeu_si256((__m256i*)(dst + dst_row_idx + j), res0_256);
	}

	__m128i coef0_128 = _mm_set1_epi16(i_filter_coeffs[0]);
	for (; j < width_rem_size8; j+=8)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(SPAT_FILTER_OUT_RND);

		for (fj = 0; fj < half_fw; fj+=2){
			jj1 = f_l_j + fj*2;
            
			__m128i coef0 = _mm_set1_epi32(i32_filter_coeffs[fj]);
			__m128i coef1 = _mm_set1_epi32(i32_filter_coeffs[fj+1]);

			__m128i d0 = _mm_loadu_si128((__m128i*)(tmp + jj1));
			__m128i d2 = _mm_loadu_si128((__m128i*)(tmp + jj1 + 2));
			__m128i d1 = _mm_loadu_si128((__m128i*)(tmp + jj1 + 1));
			__m128i d3 = _mm_loadu_si128((__m128i*)(tmp + jj1 + 3));

			res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(d0, coef0));
			res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(d2, coef1));
			res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(d1, coef0));
			res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(d3, coef1));
        }
		__m128i d0 = _mm_loadu_si128((__m128i*)(tmp + f_r_j));
		__m128i tmp0 = _mm_unpacklo_epi32(res0_128, res4_128);
		__m128i tmp4 = _mm_unpackhi_epi32(res0_128, res4_128);
		__m128i mul0_lo = _mm_mullo_epi16(d0, coef0_128);
		__m128i mul0_hi = _mm_mulhi_epi16(d0, coef0_128);
		__m128i tmp0_lo = _mm_unpacklo_epi16(mul0_lo, mul0_hi);
		__m128i tmp0_hi = _mm_unpackhi_epi16(mul0_lo, mul0_hi);

		tmp0 = _mm_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm_add_epi32(tmp4, tmp0_hi);
		tmp0 = _mm_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		res0_128 = _mm_packs_epi32(tmp0, tmp4);

		_mm_storeu_si128((__m128i*)(dst + dst_row_idx + j), res0_128);
	}	
	
    for (; j < (width - half_fw); j++)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
        spat_fil_accum_dtype accum = 0;
        /**
         * The filter coefficients are symmetric, 
         * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
         * The centre pixel is multiplied and accumulated outside the loop
        */
        for (fj = 0; fj < half_fw; fj++){

            jj1 = f_l_j + fj;
            jj2 = f_r_j - fj;
            accum += i_filter_coeffs[fj] * ((spat_fil_accum_dtype)tmp[jj1] + tmp[jj2]); //Since filter coefficients are symmetric
        }
        accum += (spat_fil_inter_dtype) i_filter_coeffs[half_fw] * tmp[j];
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
    }
	
	/**
     * This loop is to handle virtual padding of the right border pixels
     */
	d0 = _mm256_loadu_si256((__m256i*)(tmp + j - 6));
	d1 = _mm256_loadu_si256((__m256i*)(tmp + j - 22));

	for (; j < (width - ((half_fw / 2) + 1)); j++)
	{
		int fi0 = half_filter_table_w + width - half_fw - j - 6;
		int fi1 =  j - width + half_fw;
		int fi2 =  3 + width - half_fw - j;
		
		__m256i coef0 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi0));
		__m256i coef1 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi1));
		__m256i coef2 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi2));

		coef0 = _mm256_add_epi16(coef0, coef1);

		__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef0);
		__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef0);

		__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef2);
		__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef2);

		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
		__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

		tmp0_lo = _mm256_add_epi32(tmp0_lo, tmp0_hi);
		tmp1_lo = _mm256_add_epi32(tmp1_lo, tmp1_hi);
		
		res0_256 = _mm256_add_epi32(tmp0_lo, tmp1_lo);

		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(res0_256), _mm256_extracti128_si256(res0_256,1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);

		int r = _mm_cvtsi128_si32(r1);
		dst[dst_row_idx + j] = (spat_fil_output_dtype) ((r + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}
	
	for (; j < width; j++)
	{
		int fi0 = half_filter_table_w + width - half_fw - j - 6;
		int fi1 =  j - width + half_fw;
		
		__m256i coef0 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi0));
		__m256i coef1 = _mm256_loadu_si256((__m256i*)(i_filter_coeffs_with_zeros + fi1));

		__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef0);
		__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef0);

		__m256i mul1_lo_256 = _mm256_mullo_epi16(d0, coef1);
		__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0, coef1);

		__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
		__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);

		__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
		__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

		tmp0_lo = _mm256_add_epi32(tmp0_lo, tmp0_hi);
		tmp0_hi = _mm256_add_epi32(tmp1_lo, tmp1_hi);

		res0_256 = _mm256_add_epi32(tmp0_lo, tmp0_hi);

		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(res0_256), _mm256_extracti128_si256(res0_256,1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);
		int r = _mm_cvtsi128_si32(r1);
		dst[dst_row_idx + j] = (spat_fil_output_dtype) ((r + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}	
}

void integer_spatial_filter_avx2(void *src, spat_fil_output_dtype *dst, int width, int height, int bitdepth)
{
    const spat_fil_coeff_dtype i_filter_coeffs[21] = {
        -900, -1054, -1239, -1452, -1669, -1798, -1547, -66, 4677, 14498, 21495,
        14498, 4677, -66, -1547, -1798, -1669, -1452, -1239, -1054, -900
    };

	// For madd version
	const spat_fil_accum_dtype i32_filter_coeffs[11] = {
        -900 + (spat_fil_accum_dtype)(((unsigned int)-1054) << 16) + (1 << 16),
		-1239 + (spat_fil_accum_dtype)(((unsigned int)-1452) << 16) + (1 << 16),
		-1669 + (spat_fil_accum_dtype)(((unsigned int)-1798) << 16) + (1 << 16),
		-1547 + (spat_fil_accum_dtype)(((unsigned int)-66) << 16) + (1 << 16),
		4677 + (14498 << 16) /* + (1 << 16) */,
		21495 + (14498 << 16) /* + (1 << 16) */,
		4677 + (spat_fil_accum_dtype)(((unsigned int)-66) << 16) /* + (1 << 16) */,
		-1547 + (spat_fil_accum_dtype)(((unsigned int)-1798) << 16) + (1 << 16),
		-1669 + (spat_fil_accum_dtype)(((unsigned int)-1452) << 16) + (1 << 16),
		-1239 + (spat_fil_accum_dtype)(((unsigned int)-1054) << 16) + (1 << 16),
		-900 + (1 << 16)
    };

    int src_px_stride = width;
    int dst_px_stride = width;
	int width_rem_size64 = width - (width % 64);
	int width_rem_size32 = width - (width % 32);
	int width_rem_size16 = width - (width % 16);
	int width_rem_size8 = width - (width % 8);	

    spat_fil_inter_dtype *tmp = aligned_malloc(ALIGN_CEIL(src_px_stride * sizeof(spat_fil_inter_dtype)), MAX_ALIGN);

    // spat_fil_inter_dtype imgcoeff;
	uint8_t *src_8b = NULL;
	uint16_t *src_hbd = NULL;
	
	int interim_rnd = 0, interim_shift = 0;

    int i, j, fi, ii, ii1, ii2;
	//unsigned int i, j, fi, ii, ii1, ii2;
	// int fj, jj, jj1, jj;
    // spat_fil_coeff_dtype *coeff_ptr;
    int fwidth = 21;
    int half_fw = fwidth / 2;
	
	if(8 == bitdepth)
	{
		src_8b = (uint8_t*)src;
		src_hbd = NULL;
		interim_rnd = SPAT_FILTER_INTER_RND;
		interim_shift = SPAT_FILTER_INTER_SHIFT;
	}
	else // HBD case
	{
		src_8b = NULL;
		src_hbd = (uint16_t*)src;		
		interim_shift = SPAT_FILTER_INTER_SHIFT + (bitdepth - 8);
		interim_rnd = (1 << (interim_shift - 1));
	}

	__m256i res0_256, res4_256, res8_256, res12_256;
	__m256i res0_32_256, res4_32_256, res8_32_256, res12_32_256;
	__m256i zero_256 = _mm256_setzero_si256();
	__m128i res0_128, res4_128, res8_128, res12_128;
	__m128i zero_128 = _mm_setzero_si128();

    /**
     * The loop i=0 to height is split into 3 parts
     * This is to avoid the if conditions used for virtual padding
     */
    for (i = 0; i < half_fw; i++){

        int diff_i_halffw = i - half_fw;
        int pro_mir_end = -diff_i_halffw - 1;
		j = 0;
        /* Vertical pass. */

		if(8 == bitdepth)
		{
			for (; j < width_rem_size64; j+=64){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				res0_32_256 = res4_32_256 = res8_32_256 = res12_32_256 = _mm256_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j + 32));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
					__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);
					
					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32_lo, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32_lo, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d0_32_hi, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d0_32_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j + 32));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);
					
					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
					__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32_lo, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32_lo, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d0_32_hi, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d0_32_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);

				res0_32_256 = _mm256_srai_epi32(res0_32_256, interim_shift);	
				res4_32_256 = _mm256_srai_epi32(res4_32_256, interim_shift);
				res8_32_256 = _mm256_srai_epi32(res8_32_256, interim_shift);
				res12_32_256 = _mm256_srai_epi32(res12_32_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);
				res0_32_256 = _mm256_packs_epi32(res0_32_256, res4_32_256);
				res8_32_256 = _mm256_packs_epi32(res8_32_256, res12_32_256);

				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				__m256i r0_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x20);
				__m256i r8_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x31);

				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
				_mm256_store_si256((__m256i*)(tmp + j + 32), r0_32);
				_mm256_store_si256((__m256i*)(tmp + j + 48), r8_32);
			}

			for (; j < width_rem_size32; j+=32){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					
					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);
					
					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}
				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);

				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){

					ii = pro_mir_end - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d0_hi, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp1_lo_128 = _mm_unpacklo_epi16(mul1_lo_128, mul1_hi_128);
					__m128i tmp1_hi_128 = _mm_unpackhi_epi16(mul1_lo_128, mul1_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
					res8_128 = _mm_add_epi32(tmp1_lo_128, res8_128);
					res12_128 = _mm_add_epi32(tmp1_hi_128, res12_128);

				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d0_hi, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp1_lo_128 = _mm_unpacklo_epi16(mul1_lo_128, mul1_hi_128);
					__m128i tmp1_hi_128 = _mm_unpackhi_epi16(mul1_lo_128, mul1_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
					res8_128 = _mm_add_epi32(tmp1_lo_128, res8_128);
					res12_128 = _mm_add_epi32(tmp1_hi_128, res12_128);
				}

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res8_128 = _mm_srai_epi32(res8_128, interim_shift);
				res12_128 = _mm_srai_epi32(res12_128, interim_shift);
								
				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				res8_128 = _mm_packs_epi32(res8_128, res12_128);

				__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res0_128), res8_128, 1);
				_mm256_store_si256((__m256i*)(tmp + j), res);
			}

			for (; j < width_rem_size8; j+=8){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					
					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++){
				spat_fil_accum_dtype accum = 0;

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++)
				{
					ii = pro_mir_end - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (; j < width_rem_size64; j+=64)
			{
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				res0_32_256 = res4_32_256 = res8_32_256 = res12_32_256 = _mm256_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 32));
					__m256i d1_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 48));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d1_32, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d1_32, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 32));
					__m256i d1_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 48));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d1_32, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d1_32, coef);

					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}

				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);

				res0_32_256 = _mm256_srai_epi32(res0_32_256, interim_shift);
				res4_32_256 = _mm256_srai_epi32(res4_32_256, interim_shift);
				res8_32_256 = _mm256_srai_epi32(res8_32_256, interim_shift);
				res12_32_256 = _mm256_srai_epi32(res12_32_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);

				res0_32_256 = _mm256_packs_epi32(res0_32_256, res4_32_256);
				res8_32_256 = _mm256_packs_epi32(res8_32_256, res12_32_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
				_mm256_store_si256((__m256i*)(tmp + j + 16), res8_256);
				_mm256_store_si256((__m256i*)(tmp + j + 32), res0_32_256);
				_mm256_store_si256((__m256i*)(tmp + j + 48), res8_32_256);
			}

			for (; j < width_rem_size32; j+=32)
			{
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}

				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
				_mm256_store_si256((__m256i*)(tmp + j + 16), res8_256);
			}

			for (; j < width_rem_size16; j+=16)
			{
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);

					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				}

				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
			}

			for (; j < width_rem_size8; j+=8)
			{
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);

					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
						
					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);

				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++)
			{

				spat_fil_accum_dtype accum = 0;

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){

					ii = pro_mir_end - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}

        /* Horizontal pass. common for 8bit and hbd cases */
		integer_horizontal_filter_avx2(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);

    }
    //This is the core loop
    for ( ; i < (height - half_fw); i++){

        int f_l_i = i - half_fw;
        int f_r_i = i + half_fw;
        /* Vertical pass. */
		j = 0;
		if(8 == bitdepth)
		{
			for (; j < width_rem_size64; j+=64){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				res0_32_256 = res4_32_256 = res8_32_256 = res12_32_256 = _mm256_set1_epi32(interim_rnd);
				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2)
				{
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii1 * src_px_stride + j));
					__m256i d20 = _mm256_loadu_si256((__m256i*)(src_8b + ii2 * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m256i d19 = _mm256_loadu_si256((__m256i*)(src_8b + (ii2 - 1) * src_px_stride + j));
					
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii1 * src_px_stride + j + 32));
					__m256i d20_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii2 * src_px_stride + j + 32));
					__m256i d1_32 = _mm256_loadu_si256((__m256i*)(src_8b + (ii1 + 1) * src_px_stride + j + 32));
					__m256i d19_32 = _mm256_loadu_si256((__m256i*)(src_8b + (ii2 - 1) * src_px_stride + j + 32));
					
					__m256i f0_1 = _mm256_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d20_lo = _mm256_unpacklo_epi8(d20, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d20_hi = _mm256_unpackhi_epi8(d20, zero_256);
					__m256i d1_lo = _mm256_unpacklo_epi8(d1, zero_256);
					__m256i d19_lo = _mm256_unpacklo_epi8(d19, zero_256);
					__m256i d1_hi = _mm256_unpackhi_epi8(d1, zero_256);
					__m256i d19_hi = _mm256_unpackhi_epi8(d19, zero_256);

					__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
					__m256i d20_32_lo = _mm256_unpacklo_epi8(d20_32, zero_256);
					__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);
					__m256i d20_32_hi = _mm256_unpackhi_epi8(d20_32, zero_256);
					__m256i d1_32_lo = _mm256_unpacklo_epi8(d1_32, zero_256);
					__m256i d19_32_lo = _mm256_unpacklo_epi8(d19_32, zero_256);
					__m256i d1_32_hi = _mm256_unpackhi_epi8(d1_32, zero_256);
					__m256i d19_32_hi = _mm256_unpackhi_epi8(d19_32, zero_256);
					
					d0_lo = _mm256_add_epi16(d0_lo, d20_lo);
					d1_lo = _mm256_add_epi16(d1_lo, d19_lo);
					d0_hi = _mm256_add_epi16(d0_hi, d20_hi);
					d1_hi = _mm256_add_epi16(d1_hi, d19_hi);

					d0_32_lo = _mm256_add_epi16(d0_32_lo, d20_32_lo);
					d1_32_lo = _mm256_add_epi16(d1_32_lo, d19_32_lo);
					d0_32_hi = _mm256_add_epi16(d0_32_hi, d20_32_hi);
					d1_32_hi = _mm256_add_epi16(d1_32_hi, d19_32_hi);

					__m256i l0_20_1_19_0 = _mm256_unpacklo_epi16(d0_lo, d1_lo);
					__m256i l0_20_1_19_4 = _mm256_unpackhi_epi16(d0_lo, d1_lo);
					__m256i l0_20_1_19_8 = _mm256_unpacklo_epi16(d0_hi, d1_hi);
					__m256i l0_20_1_19_12 = _mm256_unpackhi_epi16(d0_hi, d1_hi);

					__m256i l0_20_1_19_0_32 = _mm256_unpacklo_epi16(d0_32_lo, d1_32_lo);
					__m256i l0_20_1_19_4_32 = _mm256_unpackhi_epi16(d0_32_lo, d1_32_lo);
					__m256i l0_20_1_19_8_32 = _mm256_unpacklo_epi16(d0_32_hi, d1_32_hi);
					__m256i l0_20_1_19_12_32 = _mm256_unpackhi_epi16(d0_32_hi, d1_32_hi);

					res0_256 = _mm256_add_epi32(res0_256,_mm256_madd_epi16(l0_20_1_19_0, f0_1));
					res4_256 = _mm256_add_epi32(res4_256,_mm256_madd_epi16(l0_20_1_19_4, f0_1));
					res8_256 = _mm256_add_epi32(res8_256,_mm256_madd_epi16(l0_20_1_19_8, f0_1));
					res12_256 = _mm256_add_epi32(res12_256,_mm256_madd_epi16(l0_20_1_19_12, f0_1));

					res0_32_256 = _mm256_add_epi32(res0_32_256,_mm256_madd_epi16(l0_20_1_19_0_32, f0_1));
					res4_32_256 = _mm256_add_epi32(res4_32_256,_mm256_madd_epi16(l0_20_1_19_4_32, f0_1));
					res8_32_256 = _mm256_add_epi32(res8_32_256,_mm256_madd_epi16(l0_20_1_19_8_32, f0_1));
					res12_32_256 = _mm256_add_epi32(res12_32_256,_mm256_madd_epi16(l0_20_1_19_12_32, f0_1));
				}
				__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + i * src_px_stride + j));
				__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + i * src_px_stride + j + 32));
				__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

				__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
				__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
				__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
				__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);
				
				__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
				__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
				__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
				__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

				__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32_lo, coef);
				__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32_lo, coef);
				__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d0_32_hi, coef);
				__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d0_32_hi, coef);
				
				__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
				__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

				__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
				__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
				__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
				__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);
				
				res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
				res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
				res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

				res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
				res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
				res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
				res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);

				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);

				res0_32_256 = _mm256_srai_epi32(res0_32_256, interim_shift);	
				res4_32_256 = _mm256_srai_epi32(res4_32_256, interim_shift);
				res8_32_256 = _mm256_srai_epi32(res8_32_256, interim_shift);
				res12_32_256 = _mm256_srai_epi32(res12_32_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);

				res0_32_256 = _mm256_packs_epi32(res0_32_256, res4_32_256);
				res8_32_256 = _mm256_packs_epi32(res8_32_256, res12_32_256);
				
				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				__m256i r0_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x20);
				__m256i r8_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x31);
				
				_mm256_storeu_si256((__m256i*)(tmp + j), r0);
				_mm256_storeu_si256((__m256i*)(tmp + j + 16), r8);
				_mm256_storeu_si256((__m256i*)(tmp + j + 32), r0_32);
				_mm256_storeu_si256((__m256i*)(tmp + j + 48), r8_32);
			}

			for (; j < width_rem_size32; j+=32){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){

					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii1 * src_px_stride + j));
					__m256i d20 = _mm256_loadu_si256((__m256i*)(src_8b + ii2 * src_px_stride + j));

					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m256i d19 = _mm256_loadu_si256((__m256i*)(src_8b + (ii2 - 1) * src_px_stride + j));
					__m256i f0_1 = _mm256_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d20_lo = _mm256_unpacklo_epi8(d20, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d20_hi = _mm256_unpackhi_epi8(d20, zero_256);
					
					__m256i d1_lo = _mm256_unpacklo_epi8(d1, zero_256);
					__m256i d19_lo = _mm256_unpacklo_epi8(d19, zero_256);
					__m256i d1_hi = _mm256_unpackhi_epi8(d1, zero_256);
					__m256i d19_hi = _mm256_unpackhi_epi8(d19, zero_256);
					
					d0_lo = _mm256_add_epi16(d0_lo, d20_lo);
					d1_lo = _mm256_add_epi16(d1_lo, d19_lo);
					d0_hi = _mm256_add_epi16(d0_hi, d20_hi);
					d1_hi = _mm256_add_epi16(d1_hi, d19_hi);

					__m256i l0_20_1_19_0 = _mm256_unpacklo_epi16(d0_lo, d1_lo);
					__m256i l0_20_1_19_4 = _mm256_unpackhi_epi16(d0_lo, d1_lo);
					__m256i l0_20_1_19_8 = _mm256_unpacklo_epi16(d0_hi, d1_hi);
					__m256i l0_20_1_19_12 = _mm256_unpackhi_epi16(d0_hi, d1_hi);

					res0_256 = _mm256_add_epi32(res0_256,_mm256_madd_epi16(l0_20_1_19_0, f0_1));
					res4_256 = _mm256_add_epi32(res4_256,_mm256_madd_epi16(l0_20_1_19_4, f0_1));
					res8_256 = _mm256_add_epi32(res8_256,_mm256_madd_epi16(l0_20_1_19_8, f0_1));
					res12_256 = _mm256_add_epi32(res12_256,_mm256_madd_epi16(l0_20_1_19_12, f0_1));
				}
				__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + i * src_px_stride + j));
				__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

				__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
				__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
				
				__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
				__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
				__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
				__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);
				
				__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
				__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
				
				res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
				res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
				res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);
				
				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				_mm256_storeu_si256((__m256i*)(tmp + j), r0);
				_mm256_storeu_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii1 * src_px_stride + j));
					__m128i d20 = _mm_loadu_si128((__m128i*)(src_8b + ii2 * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m128i d19 = _mm_loadu_si128((__m128i*)(src_8b + (ii2 - 1) * src_px_stride + j));
					__m128i f0_1 = _mm_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);
					__m128i d20_lo = _mm_unpacklo_epi8(d20, zero_128);
					__m128i d20_hi = _mm_unpackhi_epi8(d20, zero_128);
					
					__m128i d1_lo = _mm_unpacklo_epi8(d1, zero_128);
					__m128i d1_hi = _mm_unpackhi_epi8(d1, zero_128);
					__m128i d19_lo = _mm_unpacklo_epi8(d19, zero_128);
					__m128i d19_hi = _mm_unpackhi_epi8(d19, zero_128);
					
					d0_lo = _mm_add_epi16(d0_lo, d20_lo);
					d0_hi = _mm_add_epi16(d0_hi, d20_hi);
					d1_lo = _mm_add_epi16(d1_lo, d19_lo);
					d1_hi = _mm_add_epi16(d1_hi, d19_hi);

					__m128i l0_20_1_19_0 = _mm_unpacklo_epi16(d0_lo, d1_lo);
					__m128i l0_20_1_19_4 = _mm_unpackhi_epi16(d0_lo, d1_lo);
					__m128i l0_20_1_19_8 = _mm_unpacklo_epi16(d0_hi, d1_hi);
					__m128i l0_20_1_19_12 = _mm_unpackhi_epi16(d0_hi, d1_hi);

					res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(l0_20_1_19_0, f0_1));
					res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(l0_20_1_19_4, f0_1));
					res8_128 = _mm_add_epi32(res8_128, _mm_madd_epi16(l0_20_1_19_8, f0_1));
					res12_128 = _mm_add_epi32(res12_128, _mm_madd_epi16(l0_20_1_19_12, f0_1));
				}
				__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + i * src_px_stride + j));
				__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
				__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
				__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);

				__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
				__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
				__m128i mul1_lo_128 = _mm_mullo_epi16(d0_hi, coef);
				__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_hi, coef);
				
				__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
				__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
				__m128i tmp1_lo_128 = _mm_unpacklo_epi16(mul1_lo_128, mul1_hi_128);
				__m128i tmp1_hi_128 = _mm_unpackhi_epi16(mul1_lo_128, mul1_hi_128);
				
				res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
				res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				res8_128 = _mm_add_epi32(tmp1_lo_128, res8_128);
				res12_128 = _mm_add_epi32(tmp1_hi_128, res12_128);

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res8_128 = _mm_srai_epi32(res8_128, interim_shift);
				res12_128 = _mm_srai_epi32(res12_128, interim_shift);
								
				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				res8_128 = _mm_packs_epi32(res8_128, res12_128);

				__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res0_128), res8_128, 1);
				_mm256_store_si256((__m256i*)(tmp + j), res);
			}

			for (; j < width_rem_size8; j+=8){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii1 * src_px_stride + j));
					__m128i d20 = _mm_loadu_si128((__m128i*)(src_8b + ii2 * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m128i d19 = _mm_loadu_si128((__m128i*)(src_8b + (ii2 - 1) * src_px_stride + j));
					__m128i f0_1 = _mm_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d20_lo = _mm_unpacklo_epi8(d20, zero_128);
					__m128i d1_lo = _mm_unpacklo_epi8(d1, zero_128);
					__m128i d19_lo = _mm_unpacklo_epi8(d19, zero_128);
					
					d0_lo = _mm_add_epi16(d0_lo, d20_lo);
					d1_lo = _mm_add_epi16(d1_lo, d19_lo);

					__m128i l0_20_1_19_0 = _mm_unpacklo_epi16(d0_lo, d1_lo);
					__m128i l0_20_1_19_4 = _mm_unpackhi_epi16(d0_lo, d1_lo);

					res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(l0_20_1_19_0, f0_1));
					res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(l0_20_1_19_4, f0_1));
				}
				__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + i * src_px_stride + j));
				__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
				__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);

				__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
				__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
				
				__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
				__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
				
				res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
				res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res0_128 = _mm_packs_epi32(res0_128, res4_128);

				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++){
				spat_fil_accum_dtype accum = 0;
				for (fi = 0; fi < (half_fw); fi++){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src_8b[ii1 * src_px_stride + j] + src_8b[ii2 * src_px_stride + j]);
				}
				accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[i * src_px_stride + j];
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (; j < width_rem_size32; j+=32){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii1 * src_px_stride + j));
					__m256i d20 = _mm256_loadu_si256((__m256i*)(src_hbd + ii2 * src_px_stride + j));
					__m256i d0_16 = _mm256_loadu_si256((__m256i*)(src_hbd + ii1 * src_px_stride + j + 16));
					__m256i d20_16 = _mm256_loadu_si256((__m256i*)(src_hbd + ii2 * src_px_stride + j + 16));

					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m256i d19 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m256i d1_16 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii1 + 1) * src_px_stride + j + 16));
					__m256i d19_16 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii2 - 1) * src_px_stride + j + 16));
					__m256i f0_1 = _mm256_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm256_add_epi16(d0, d20);
					d0_16 = _mm256_add_epi16(d0_16, d20_16);
					d1 = _mm256_add_epi16(d1, d19);
					d1_16 = _mm256_add_epi16(d1_16, d19_16);

					__m256i l0_20_1_19_0 = _mm256_unpacklo_epi16(d0, d1);
					__m256i l0_20_1_19_4 = _mm256_unpackhi_epi16(d0, d1);
					__m256i l0_20_1_19_16 = _mm256_unpacklo_epi16(d0_16, d1_16);
					__m256i l0_20_1_19_20 = _mm256_unpackhi_epi16(d0_16, d1_16);
					
					res0_256 = _mm256_add_epi32(res0_256,_mm256_madd_epi16(l0_20_1_19_0, f0_1));
					res4_256 = _mm256_add_epi32(res4_256,_mm256_madd_epi16(l0_20_1_19_4, f0_1));
					res8_256 = _mm256_add_epi32(res8_256,_mm256_madd_epi16(l0_20_1_19_16, f0_1));
					res12_256 = _mm256_add_epi32(res12_256,_mm256_madd_epi16(l0_20_1_19_20, f0_1));
				}
				__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + i * src_px_stride + j));
				__m256i d0_16 = _mm256_loadu_si256((__m256i*)(src_hbd + i * src_px_stride + j + 16));
				__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

				__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
				__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
				__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_16, coef);
				__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_16, coef);
				
				__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
				__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
				
				res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
				res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
				res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
				_mm256_store_si256((__m256i*)(tmp + j + 16), res8_256);
			}

			for (; j < width_rem_size16; j+=16){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii1 * src_px_stride + j));
					__m256i d20 = _mm256_loadu_si256((__m256i*)(src_hbd + ii2 * src_px_stride + j));

					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m256i d19 = _mm256_loadu_si256((__m256i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m256i f0_1 = _mm256_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm256_add_epi16(d0, d20);
					d1 = _mm256_add_epi16(d1, d19);

					__m256i l0_20_1_19_0 = _mm256_unpacklo_epi16(d0, d1);
					__m256i l0_20_1_19_4 = _mm256_unpackhi_epi16(d0, d1);
					
					res0_256 = _mm256_add_epi32(res0_256,_mm256_madd_epi16(l0_20_1_19_0, f0_1));
					res4_256 = _mm256_add_epi32(res4_256,_mm256_madd_epi16(l0_20_1_19_4, f0_1));
				}
				__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + i * src_px_stride + j));
				__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

				__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
				__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
				
				__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
				__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
				
				res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
				res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);

				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
			}

			for (; j < width_rem_size8; j+=8){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii1 * src_px_stride + j));
					__m128i d20 = _mm_loadu_si128((__m128i*)(src_hbd + ii2 * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m128i d19 = _mm_loadu_si128((__m128i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m128i f0_1 = _mm_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm_add_epi16(d0, d20);
					d1 = _mm_add_epi16(d1, d19);

					__m128i l0_20_1_19_0 = _mm_unpacklo_epi16(d0, d1);
					__m128i l0_20_1_19_4 = _mm_unpackhi_epi16(d0, d1);
					
					res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(l0_20_1_19_0, f0_1));
					res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(l0_20_1_19_4, f0_1));
				}
				__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + i * src_px_stride + j));
				__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

				__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
				__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
				
				__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
				__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
				
				res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
				res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res0_128 = _mm_packs_epi32(res0_128, res4_128);

				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++){

				spat_fil_accum_dtype accum = 0;

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi++){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src_hbd[ii1 * src_px_stride + j] + src_hbd[ii2 * src_px_stride + j]);
				}
				accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[i * src_px_stride + j];
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}

		/* Horizontal pass. common for 8bit and hbd cases */
        integer_horizontal_filter_avx2(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);

    }
    /**
     * This loop is to handle virtual padding of the bottom border pixels
     */
    for (; i < height; i++){

        int diff_i_halffw = i - half_fw;
        int epi_mir_i = 2 * height - diff_i_halffw - 1;
        int epi_last_i  = height - diff_i_halffw;
        
        /* Vertical pass. */
		j = 0;
		if(8 == bitdepth)
		{
			for (; j < width_rem_size64; j+=64){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				res0_32_256 = res4_32_256 = res8_32_256 = res12_32_256 = _mm256_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j + 32));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);
					
					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
					__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32_lo, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32_lo, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d0_32_hi, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d0_32_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j + 32));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					__m256i d0_32_lo = _mm256_unpacklo_epi8(d0_32, zero_256);
					__m256i d0_32_hi = _mm256_unpackhi_epi8(d0_32, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32_lo, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32_lo, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d0_32_hi, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d0_32_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				
				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);

				res0_32_256 = _mm256_srai_epi32(res0_32_256, interim_shift);	
				res4_32_256 = _mm256_srai_epi32(res4_32_256, interim_shift);
				res8_32_256 = _mm256_srai_epi32(res8_32_256, interim_shift);
				res12_32_256 = _mm256_srai_epi32(res12_32_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);
				res0_32_256 = _mm256_packs_epi32(res0_32_256, res4_32_256);
				res8_32_256 = _mm256_packs_epi32(res8_32_256, res12_32_256);
				
				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				__m256i r0_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x20);
				__m256i r8_32 = _mm256_permute2x128_si256(res0_32_256, res8_32_256, 0x31);
				
				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
				_mm256_store_si256((__m256i*)(tmp + j + 32), r0_32);
				_mm256_store_si256((__m256i*)(tmp + j + 48), r8_32);
			}

			for (; j < width_rem_size32; j+=32){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);
					
					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}
				
				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);
				
				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
					
					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d0_hi, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp1_lo_128 = _mm_unpacklo_epi16(mul1_lo_128, mul1_hi_128);
					__m128i tmp1_hi_128 = _mm_unpackhi_epi16(mul1_lo_128, mul1_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
					res8_128 = _mm_add_epi32(tmp1_lo_128, res8_128);
					res12_128 = _mm_add_epi32(tmp1_hi_128, res12_128);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);
					__m128i d0_hi = _mm_unpackhi_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d0_hi, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp1_lo_128 = _mm_unpacklo_epi16(mul1_lo_128, mul1_hi_128);
					__m128i tmp1_hi_128 = _mm_unpackhi_epi16(mul1_lo_128, mul1_hi_128);
						
					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
					res8_128 = _mm_add_epi32(tmp1_lo_128, res8_128);
					res12_128 = _mm_add_epi32(tmp1_hi_128, res12_128);
				}
				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);
				res8_128 = _mm_srai_epi32(res8_128, interim_shift);
				res12_128 = _mm_srai_epi32(res12_128, interim_shift);
								
				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				res8_128 = _mm_packs_epi32(res8_128, res12_128);

				__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res0_128), res8_128, 1);
				_mm256_store_si256((__m256i*)(tmp + j), res);
			}

			for (; j < width_rem_size8; j+=8){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_8b + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i d0_lo = _mm_unpacklo_epi8(d0, zero_128);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0_lo, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0_lo, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
						
					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}
				
				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);

				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++)
			{
				spat_fil_accum_dtype accum = 0;
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (; j < width_rem_size64; j+=64){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);
				res0_32_256 = res4_32_256 = res8_32_256 = res12_32_256 = _mm256_set1_epi32(interim_rnd);

				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 32));
					__m256i d1_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 48));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d1_32, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d1_32, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d1 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i d0_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 32));
					__m256i d1_32 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 48));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d1, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d1, coef);

					__m256i mul0_32_lo_256 = _mm256_mullo_epi16(d0_32, coef);
					__m256i mul0_32_hi_256 = _mm256_mulhi_epi16(d0_32, coef);
					__m256i mul1_32_lo_256 = _mm256_mullo_epi16(d1_32, coef);
					__m256i mul1_32_hi_256 = _mm256_mulhi_epi16(d1_32, coef);

					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					__m256i tmp0_32_lo = _mm256_unpacklo_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp0_32_hi = _mm256_unpackhi_epi16(mul0_32_lo_256, mul0_32_hi_256);
					__m256i tmp1_32_lo = _mm256_unpacklo_epi16(mul1_32_lo_256, mul1_32_hi_256);
					__m256i tmp1_32_hi = _mm256_unpackhi_epi16(mul1_32_lo_256, mul1_32_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

					res0_32_256 = _mm256_add_epi32(tmp0_32_lo, res0_32_256);
					res4_32_256 = _mm256_add_epi32(tmp0_32_hi, res4_32_256);
					res8_32_256 = _mm256_add_epi32(tmp1_32_lo, res8_32_256);
					res12_32_256 = _mm256_add_epi32(tmp1_32_hi, res12_32_256);
				}
				
				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);

				res0_32_256 = _mm256_srai_epi32(res0_32_256, interim_shift);
				res4_32_256 = _mm256_srai_epi32(res4_32_256, interim_shift);
				res8_32_256 = _mm256_srai_epi32(res8_32_256, interim_shift);
				res12_32_256 = _mm256_srai_epi32(res12_32_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);

				res0_32_256 = _mm256_packs_epi32(res0_32_256, res4_32_256);
				res8_32_256 = _mm256_packs_epi32(res8_32_256, res12_32_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
				_mm256_store_si256((__m256i*)(tmp + j + 16), res8_256);
				_mm256_store_si256((__m256i*)(tmp + j + 32), res0_32_256);
				_mm256_store_si256((__m256i*)(tmp + j + 48), res8_32_256);
			}

			for (; j < width_rem_size32; j+=32){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d16 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d16, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d16, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}
				
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i d16 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j + 16));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d16, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d16, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);
				}

				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256,interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);
				res8_256 = _mm256_packs_epi32(res8_256,res12_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
				_mm256_store_si256((__m256i*)(tmp + j + 16), res8_256);
			}

			for (; j < width_rem_size16; j+=16){
				res0_256 = res4_256 = res8_256 = res12_256 = _mm256_set1_epi32(interim_rnd);

				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				}
				
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_hbd + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				}

				res0_256 = _mm256_srai_epi32(res0_256,interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256,interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256,res4_256);

				_mm256_store_si256((__m256i*)(tmp + j), res0_256);
			}

			for (; j < width_rem_size8; j+=8){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);

					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}
				
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);

					__m128i tmp0_lo_128 = _mm_unpacklo_epi16(mul0_lo_128, mul0_hi_128);
					__m128i tmp0_hi_128 = _mm_unpackhi_epi16(mul0_lo_128, mul0_hi_128);
						
					res0_128 = _mm_add_epi32(tmp0_lo_128, res0_128);
					res4_128 = _mm_add_epi32(tmp0_hi_128, res4_128);
				}

				res0_128 = _mm_srai_epi32(res0_128, interim_shift);
				res4_128 = _mm_srai_epi32(res4_128, interim_shift);

				res0_128 = _mm_packs_epi32(res0_128, res4_128);
				_mm_store_si128((__m128i*)(tmp + j), res0_128);
			}

			for (; j < width; j++){
				spat_fil_accum_dtype accum = 0;
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
        /* Horizontal pass. common for 8bit and hbd cases */
        integer_horizontal_filter_avx2(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }

    aligned_free(tmp);

    return;
}