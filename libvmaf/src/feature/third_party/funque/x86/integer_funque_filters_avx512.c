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

#if 0
void integer_funque_dwt2_avx512(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
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

	int width_rem_size32 = width_div_2 - (width_div_2 % 32);
	int width_rem_size16 = width_div_2 - (width_div_2 % 16);
	int width_rem_size8 = width_div_2 - (width_div_2 % 8);
	int width_rem_size4 = width_div_2 - (width_div_2 % 4);

	__m512i filter_shift_512 = _mm512_set1_epi32(filter_shift);
	__m512i idx_perm_512 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
	__m512i idx_extract_ab_512 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
	__m512i idx_extract_cd_512 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
	__m512i zero_512 = _mm512_setzero_si512();

	__m256i idx_perm_256 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
	__m256i filter_shift_256 = _mm256_set1_epi32(filter_shift);
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
		for(; j< width_rem_size32; j+=32)
		{
			int col_idx0 = (j << 1);

			__m512i src_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0));
			__m512i src_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0));
			__m512i src2_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0 + 32));
			__m512i src2_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0 + 32));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m512i a_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_a_512));
			__m512i a_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_a_512, 1));
			__m512i b_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_b_512));
			__m512i b_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_b_512, 1));
			__m512i a2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src2_a_512));
			__m512i a2_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src2_a_512, 1));
			__m512i b2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src2_b_512));
			__m512i b2_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src2_b_512, 1));

			__m512i a_p_b_c_p_d_lo = _mm512_add_epi32(a_lo, b_lo);
			__m512i a_p_b_c_p_d_hi = _mm512_add_epi32(a_hi, b_hi);
			__m512i a_m_b_c_m_d_lo = _mm512_sub_epi32(a_lo, b_lo);
			__m512i a_m_b_c_m_d_hi = _mm512_sub_epi32(a_hi, b_hi);
			__m512i a_p_b_c_p_d_2_lo = _mm512_add_epi32(a2_lo, b2_lo);
			__m512i a_p_b_c_p_d_2_hi = _mm512_add_epi32(a2_hi, b2_hi);
			__m512i a_m_b_c_m_d_2_lo = _mm512_sub_epi32(a2_lo, b2_lo);
			__m512i a_m_b_c_m_d_2_hi = _mm512_sub_epi32(a2_hi, b2_hi);;

			__m512i a_p_b_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_ab_512, a_p_b_c_p_d_hi);
			__m512i c_p_d_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_cd_512, a_p_b_c_p_d_hi);
			__m512i a_m_b_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_lo, idx_extract_ab_512, a_m_b_c_m_d_hi);
			__m512i c_m_d_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_lo, idx_extract_cd_512, a_m_b_c_m_d_hi);
			__m512i a_p_b_2_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_2_lo, idx_extract_ab_512, a_p_b_c_p_d_2_hi);
			__m512i c_p_d_2_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_2_lo, idx_extract_cd_512, a_p_b_c_p_d_2_hi);
			__m512i a_m_b_2_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_2_lo, idx_extract_ab_512, a_m_b_c_m_d_2_hi);
			__m512i c_m_d_2_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_2_lo, idx_extract_cd_512, a_m_b_c_m_d_2_hi);
			
			__m512i band_a_512 = _mm512_add_epi32(a_p_b_512, c_p_d_512);
			__m512i band_v_512 = _mm512_sub_epi32(a_p_b_512, c_p_d_512);
			__m512i band_h_512 = _mm512_add_epi32(a_m_b_512, c_m_d_512);
			__m512i band_d_512 = _mm512_sub_epi32(a_m_b_512, c_m_d_512);
			__m512i band_a2_512 = _mm512_add_epi32(a_p_b_2_512, c_p_d_2_512);
			__m512i band_v2_512 = _mm512_sub_epi32(a_p_b_2_512, c_p_d_2_512);
			__m512i band_h2_512 = _mm512_add_epi32(a_m_b_2_512, c_m_d_2_512);
			__m512i band_d2_512 = _mm512_sub_epi32(a_m_b_2_512, c_m_d_2_512);

			band_a_512 = _mm512_add_epi32(band_a_512, filter_shift_512);
			band_v_512 = _mm512_add_epi32(band_v_512, filter_shift_512);
			band_h_512 = _mm512_add_epi32(band_h_512, filter_shift_512);
			band_d_512 = _mm512_add_epi32(band_d_512, filter_shift_512);
			band_a2_512 = _mm512_add_epi32(band_a2_512, filter_shift_512);
			band_h2_512 = _mm512_add_epi32(band_h2_512, filter_shift_512);
			band_v2_512 = _mm512_add_epi32(band_v2_512, filter_shift_512);
			band_d2_512 = _mm512_add_epi32(band_d2_512, filter_shift_512);

			band_a_512 = _mm512_srai_epi32(band_a_512, filter_shift_rnd);
			band_a2_512 = _mm512_srai_epi32(band_a2_512, filter_shift_rnd);
			band_h_512 = _mm512_srai_epi32(band_h_512, filter_shift_rnd);
			band_h2_512 = _mm512_srai_epi32(band_h2_512, filter_shift_rnd);
			band_v_512 = _mm512_srai_epi32(band_v_512, filter_shift_rnd);
			band_v2_512 = _mm512_srai_epi32(band_v2_512, filter_shift_rnd);
			band_d_512 = _mm512_srai_epi32(band_d_512, filter_shift_rnd);
			band_d2_512 = _mm512_srai_epi32(band_d2_512, filter_shift_rnd);

			band_a_512 = _mm512_packs_epi32(band_a_512, band_a2_512);
			band_h_512 = _mm512_packs_epi32(band_h_512, band_h2_512);
			band_v_512 = _mm512_packs_epi32(band_v_512, band_v2_512);
			band_d_512 = _mm512_packs_epi32(band_d_512, band_d2_512);

			band_a_512 = _mm512_permutexvar_epi64(idx_perm_512, band_a_512);
			band_h_512 = _mm512_permutexvar_epi64(idx_perm_512, band_h_512);
			band_v_512 = _mm512_permutexvar_epi64(idx_perm_512, band_v_512);
			band_d_512 = _mm512_permutexvar_epi64(idx_perm_512, band_d_512);
			
			_mm512_storeu_si512((__m512i*)(band_a + i * dst_px_stride + j), band_a_512);
			_mm512_storeu_si512((__m512i*)(band_h + i * dst_px_stride + j), band_h_512);
			_mm512_storeu_si512((__m512i*)(band_v + i * dst_px_stride + j), band_v_512);
			_mm512_storeu_si512((__m512i*)(band_d + i * dst_px_stride + j), band_d_512);
        }
		
		for(; j< width_rem_size16; j+=16)
		{
			int col_idx0 = (j << 1);

			__m512i src_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0));
			__m512i src_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0));

			// Original
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			//F* F (a - b + c - d) - band H  (F*F is 1/2)		
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
			//F* F (a - b - c - d) - band D  (F*F is 1/2)

			__m512i a_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_a_512));
			__m512i a_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_a_512, 1));
			__m512i b_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_b_512));
			__m512i b_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_b_512, 1));

			__m512i a_p_b_c_p_d_lo = _mm512_add_epi32(a_lo, b_lo);
			__m512i a_p_b_c_p_d_hi = _mm512_add_epi32(a_hi, b_hi);
			__m512i a_m_b_c_m_d_lo = _mm512_sub_epi32(a_lo, b_lo);
			__m512i a_m_b_c_m_d_hi = _mm512_sub_epi32(a_hi, b_hi);

			__m512i a_p_b_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_ab_512, a_p_b_c_p_d_hi);
			__m512i c_p_d_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_cd_512, a_p_b_c_p_d_hi);
			__m512i a_m_b_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_lo, idx_extract_ab_512, a_m_b_c_m_d_hi);
			__m512i c_m_d_512 = _mm512_permutex2var_epi32(a_m_b_c_m_d_lo, idx_extract_cd_512, a_m_b_c_m_d_hi);

			__m512i band_a_512 = _mm512_add_epi32(a_p_b_512, c_p_d_512);
			__m512i band_v_512 = _mm512_sub_epi32(a_p_b_512, c_p_d_512);
			__m512i band_h_512 = _mm512_add_epi32(a_m_b_512, c_m_d_512);
			__m512i band_d_512 = _mm512_sub_epi32(a_m_b_512, c_m_d_512);

			band_a_512 = _mm512_add_epi32(band_a_512, filter_shift_512);
			band_v_512 = _mm512_add_epi32(band_v_512, filter_shift_512);
			band_h_512 = _mm512_add_epi32(band_h_512, filter_shift_512);
			band_d_512 = _mm512_add_epi32(band_d_512, filter_shift_512);

			band_a_512 = _mm512_srai_epi32(band_a_512, filter_shift_rnd);
			band_h_512 = _mm512_srai_epi32(band_h_512, filter_shift_rnd);
			band_v_512 = _mm512_srai_epi32(band_v_512, filter_shift_rnd);
			band_d_512 = _mm512_srai_epi32(band_d_512, filter_shift_rnd);

			band_a_512 = _mm512_packs_epi32(band_a_512, zero_512);
			band_h_512 = _mm512_packs_epi32(band_h_512, zero_512);
			band_v_512 = _mm512_packs_epi32(band_v_512, zero_512);
			band_d_512 = _mm512_packs_epi32(band_d_512, zero_512);

			band_a_512 = _mm512_permutexvar_epi64(idx_perm_512, band_a_512);
			band_h_512 = _mm512_permutexvar_epi64(idx_perm_512, band_h_512);
			band_v_512 = _mm512_permutexvar_epi64(idx_perm_512, band_v_512);
			band_d_512 = _mm512_permutexvar_epi64(idx_perm_512, band_d_512);
			
			_mm256_storeu_si256((__m256i*)(band_a + i * dst_px_stride + j), _mm512_castsi512_si256(band_a_512));
			_mm256_storeu_si256((__m256i*)(band_h + i * dst_px_stride + j), _mm512_castsi512_si256(band_h_512));
			_mm256_storeu_si256((__m256i*)(band_v + i * dst_px_stride + j), _mm512_castsi512_si256(band_v_512));
			_mm256_storeu_si256((__m256i*)(band_d + i * dst_px_stride + j), _mm512_castsi512_si256(band_d_512));
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

			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm_256);
			band_h_256 = _mm256_permutevar8x32_epi32(band_h_256, idx_perm_256);
			band_v_256 = _mm256_permutevar8x32_epi32(band_v_256, idx_perm_256);
			band_d_256 = _mm256_permutevar8x32_epi32(band_d_256, idx_perm_256);
			
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
#endif
void integer_funque_dwt2_avx512(spat_fil_output_dtype *src, ptrdiff_t src_stride,
                                i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width,
                                int height, int spatial_csf, int level)
{
    int src_px_stride = src_stride / sizeof(dwt2_dtype);
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    int8_t const_2_wl0 = 1;

    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    int8_t filter_shift_rnd__val = 1 << (filter_shift - 1);

    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    int8_t filter_shift_lcpad_rnd = 1 << (filter_shift_lcpad - 1);

    if(spatial_csf == 0)
    {
        if(level != 3)
        {
            filter_shift = 0;
            filter_shift_rnd__val = 0;
            const_2_wl0 = 2;
            filter_shift_lcpad = 0;
            filter_shift_lcpad_rnd = 0;
        }
    }

    __m512i filter_shift_rnd = _mm512_set1_epi32(filter_shift_rnd__val);

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];

    int16_t row_idx0, row_idx1, col_idx0;

    int row0_offset, row1_offset;

    int last_col = width & 1;

    int i, j, k;
    __m512i multi_const_1 = _mm512_set_epi16(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                             1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
    __m512i multi_const_2 = _mm512_set_epi16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                             0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);

    for(i = 0; i < (height + 1) >> 1; ++i)
    {
        row_idx0 = i << 1;
        row_idx1 = (i << 1) + 1;
        row_idx1 = row_idx1 < height ? row_idx1 : i << 1;
        row0_offset = (row_idx0) *src_px_stride;
        row1_offset = (row_idx1) *src_px_stride;

        for(j = 0; j <= width - 32; j += 32)
        {
            col_idx0 = j;
            k = (j >> 1);

            __m512i loaded_data_ac = _mm512_loadu_si512((__m512i *) (src + row0_offset + col_idx0));

            __m512i a_32bit = _mm512_madd_epi16(loaded_data_ac, multi_const_2);
            __m512i c_32bit = _mm512_madd_epi16(loaded_data_ac, multi_const_1);

            __m512i a_add_c = _mm512_add_epi32(a_32bit, c_32bit);
            __m512i a_sub_c = _mm512_sub_epi32(a_32bit, c_32bit);

            __m512i loaded_data_bd = _mm512_loadu_si512((__m512i *) (src + row1_offset + col_idx0));

            __m512i b_32bit = _mm512_madd_epi16(loaded_data_bd, multi_const_2);
            __m512i d_32bit = _mm512_madd_epi16(loaded_data_bd, multi_const_1);

            __m512i b_add_d = _mm512_add_epi32(b_32bit, d_32bit);
            __m512i b_sub_d = _mm512_sub_epi32(b_32bit, d_32bit);

            __m512i inter_a = _mm512_srai_epi32(
                _mm512_add_epi32(_mm512_add_epi32(a_add_c, b_add_d), filter_shift_rnd),
                filter_shift);
            _mm256_storeu_si256((__m256i *) (band_a + i * dst_px_stride + k),
                                _mm512_cvtepi32_epi16(inter_a));

            __m512i inter_h = _mm512_srai_epi32(
                _mm512_add_epi32(_mm512_sub_epi32(a_add_c, b_add_d), filter_shift_rnd),
                filter_shift);
            _mm256_storeu_si256((__m256i *) (band_h + i * dst_px_stride + k),
                                _mm512_cvtepi32_epi16(inter_h));

            __m512i inter_v = _mm512_srai_epi32(
                _mm512_add_epi32(_mm512_add_epi32(a_sub_c, b_sub_d), filter_shift_rnd),
                filter_shift);
            _mm256_storeu_si256((__m256i *) (band_v + i * dst_px_stride + k),
                                _mm512_cvtepi32_epi16(inter_v));

            __m512i inter_d = _mm512_srai_epi32(
                _mm512_add_epi32(_mm512_sub_epi32(a_sub_c, b_sub_d), filter_shift_rnd),
                filter_shift);
            _mm256_storeu_si256((__m256i *) (band_d + i * dst_px_stride + k),
                                _mm512_cvtepi32_epi16(inter_d));
        }
        for(; j < width - 1; j += 2)
        {
            col_idx0 = j;
            int col_idx1 = j + 1;
            int k = (j >> 1);

            // a & b 2 values in adjacent rows at the same coloumn
            spat_fil_output_dtype src_a = src[row0_offset + col_idx0];
            spat_fil_output_dtype src_b = src[row1_offset + col_idx0];

            // c & d are adjacent values to a & b in teh same row
            spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
            spat_fil_output_dtype src_d = src[row1_offset + col_idx1];

            // a + b & a - b
            int32_t src_a_p_b = src_a + src_b;
            int32_t src_a_m_b = src_a - src_b;

            // c + d & c - d
            int32_t src_c_p_d = src_c + src_d;
            int32_t src_c_m_d = src_c - src_d;

            // F* F (a + b + c + d) - band A  (F*F is 1/2)
            band_a[i * dst_px_stride + k] =
                (dwt2_dtype) (((src_a_p_b + src_c_p_d) + filter_shift_rnd__val) >> filter_shift);

            // F* F (a - b + c - d) - band H  (F*F is 1/2)
            band_h[i * dst_px_stride + k] =
                (dwt2_dtype) (((src_a_m_b + src_c_m_d) + filter_shift_rnd__val) >> filter_shift);

            // F* F (a + b - c + d) - band V  (F*F is 1/2)
            band_v[i * dst_px_stride + k] =
                (dwt2_dtype) (((src_a_p_b - src_c_p_d) + filter_shift_rnd__val) >> filter_shift);

            // F* F (a - b - c - d) - band D  (F*F is 1/2)
            band_d[i * dst_px_stride + k] =
                (dwt2_dtype) (((src_a_m_b - src_c_m_d) + filter_shift_rnd__val) >> filter_shift);
        }
        if(last_col)
        {
            col_idx0 = j;
            int k = j >> 1;

            // a & b 2 values in adjacent rows at the last coloumn
            spat_fil_output_dtype src_a = src[row0_offset + col_idx0];
            spat_fil_output_dtype src_b = src[row1_offset + col_idx0];

            // a + b	& a - b
            int32_t src_a_p_b = src_a + src_b;
            int32_t src_a_m_b = src_a - src_b;

            // F* F (a + b + a + b) - band A  (F*F is 1/2)
            band_a[i * dst_px_stride + k] =
                (dwt2_dtype) ((src_a_p_b * const_2_wl0 + filter_shift_lcpad_rnd) >>
                              filter_shift_lcpad);

            // F* F (a - b + a - b) - band H  (F*F is 1/2)
            band_h[i * dst_px_stride + k] =
                (dwt2_dtype) ((src_a_m_b * const_2_wl0 + filter_shift_lcpad_rnd) >>
                              filter_shift_lcpad);

            // F* F (a + b - (a + b)) - band V, Last column V will always be 0
            band_v[i * dst_px_stride + k] = 0;

            // F* F (a - b - (a -b)) - band D,  Last column D will always be 0
            band_d[i * dst_px_stride + k] = 0;
        }
    }
}

void integer_funque_vifdwt2_band0_avx512(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height)
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
	
	int width_rem_size32 = width_div_2 - (width_div_2 % 32);
	int width_rem_size16 = width_div_2 - (width_div_2 % 16);
	int width_rem_size8 = width_div_2 - (width_div_2 % 8);
	int width_rem_size4 = width_div_2 - (width_div_2 % 4);
	
	__m512i idx_extract_ab_512 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
	__m512i idx_extract_cd_512 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
	__m512i filter_shift_512 = _mm512_set1_epi32(filter_shift);
	__m512i idx_perm_512 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
	__m512i zero_512 = _mm512_setzero_si512();

	__m256i filter_shift_256 = _mm256_set1_epi32(filter_shift);
	__m256i idx_perm_256 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
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
		j=0;

		for(; j< width_rem_size32; j+=32)
		{
			int col_idx0 = (j << 1);
			__m512i src_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0));
			__m512i src_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0));
			__m512i src2_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0 + 32));
			__m512i src2_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0 + 32));

			__m512i a_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_a_512));
			__m512i a_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_a_512, 1));
			__m512i b_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_b_512));
			__m512i b_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_b_512, 1));
			__m512i a2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src2_a_512));
			__m512i a2_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src2_a_512, 1));
			__m512i b2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src2_b_512));
			__m512i b2_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src2_b_512, 1));

			__m512i a_p_b_c_p_d_lo = _mm512_add_epi32(a_lo, b_lo);
			__m512i a_p_b_c_p_d_hi = _mm512_add_epi32(a_hi, b_hi);
			__m512i a_p_b_c_p_d_2_lo = _mm512_add_epi32(a2_lo, b2_lo);
			__m512i a_p_b_c_p_d_2_hi = _mm512_add_epi32(a2_hi, b2_hi);

			__m512i band_a_ab_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_ab_512, a_p_b_c_p_d_hi);
			__m512i band_a_cd_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_cd_512, a_p_b_c_p_d_hi);
			__m512i band_a_ab_2_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_2_lo, idx_extract_ab_512, a_p_b_c_p_d_2_hi);
			__m512i band_a_cd_2_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_2_lo, idx_extract_cd_512, a_p_b_c_p_d_2_hi);

			__m512i band_a_512 = _mm512_add_epi32(band_a_ab_512, band_a_cd_512);
			__m512i band_a_2_512 = _mm512_add_epi32(band_a_ab_2_512, band_a_cd_2_512);

			band_a_512 = _mm512_add_epi32(band_a_512, filter_shift_512);
			band_a_2_512 = _mm512_add_epi32(band_a_2_512, filter_shift_512);

			band_a_512 = _mm512_srai_epi32(band_a_512, filter_shift_rnd);
			band_a_2_512 = _mm512_srai_epi32(band_a_2_512, filter_shift_rnd);
			
			band_a_512 = _mm512_packs_epi32(band_a_512, band_a_2_512);
			band_a_512 = _mm512_permutexvar_epi64(idx_perm_512, band_a_512);

			_mm512_storeu_si512((__m512i*)(band_a + i * dst_px_stride + j), band_a_512);
        }
        
		for(; j< width_rem_size16; j+=16)
		{
			int col_idx0 = (j << 1);
			__m512i src_a_512 = _mm512_loadu_si512((__m512i*)(src + row0_offset + col_idx0));
			__m512i src_b_512 = _mm512_loadu_si512((__m512i*)(src + row1_offset + col_idx0));

			__m512i a_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_a_512));
			__m512i a_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_a_512, 1));
			__m512i b_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(src_b_512));
			__m512i b_hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(src_b_512, 1));

			__m512i a_p_b_c_p_d_lo = _mm512_add_epi32(a_lo, b_lo);
			__m512i a_p_b_c_p_d_hi = _mm512_add_epi32(a_hi, b_hi);

			__m512i band_a_ab_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_ab_512, a_p_b_c_p_d_hi);
			__m512i band_a_cd_512 = _mm512_permutex2var_epi32(a_p_b_c_p_d_lo, idx_extract_cd_512, a_p_b_c_p_d_hi);
			__m512i band_a_512 = _mm512_add_epi32(band_a_ab_512, band_a_cd_512);
			
			band_a_512 = _mm512_add_epi32(band_a_512, filter_shift_512);
			band_a_512 = _mm512_srai_epi32(band_a_512, filter_shift_rnd);
			band_a_512 = _mm512_packs_epi32(band_a_512, zero_512);
			band_a_512 = _mm512_permutexvar_epi64(idx_perm_512, band_a_512);

			_mm256_storeu_si256((__m256i*)(band_a + i * dst_px_stride + j), _mm512_castsi512_si256(band_a_512));
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

			band_a_256 = _mm256_permutevar8x32_epi32(band_a_256, idx_perm_256);
			_mm_storeu_si128((__m128i*)(band_a + i * dst_px_stride + j), _mm256_castsi256_si128(band_a_256));
        }

		for(; j< width_rem_size4; j+=4)
		{
			int col_idx0 = (j << 1);

			__m128i src_a_128 = _mm_loadu_si128((__m128i*)(src + row0_offset + col_idx0));
			__m128i src_b_128 = _mm_loadu_si128((__m128i*)(src + row1_offset + col_idx0));

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

static void integer_horizontal_filter_avx512(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst, const spat_fil_coeff_dtype *i_filter_coeffs, int width, int fwidth, int dst_row_idx, int half_fw)
{
    int j, fj, jj1, jj2;
	__m512i res0_512, res4_512, res8_512, res12_512;
    __m256i res0_256, res4_256, res8_256, res12_256;
	__m128i res0_128, res4_128, res8_128, res12_128;
	
	int width_rem_size128 = (width - half_fw) - ((width - 2*half_fw) % 128);
	int width_rem_size64 = (width - half_fw) - ((width - 2*half_fw) % 64);
	int width_rem_size32 = (width - half_fw) - ((width - 2*half_fw) % 32);
	int width_rem_size16 = (width - half_fw) - ((width - 2*half_fw) % 16);
	int width_rem_size8 = (width - half_fw) - ((width - 2*half_fw) % 8);

	const spat_fil_coeff_dtype i_filter_coeffs_with_zeros[83] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -900, -1054, -1239, -1452, -1669, -1798, -1547, -66, 4677, 14498, 21495,
        14498, 4677, -66, -1547, -1798, -1669, -1452, -1239, -1054, -900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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
	
	__m512i d0_512 = _mm512_loadu_si512((__m512i*)(tmp));
	int half_filter_table_w2 = 41;

	for(j = 0; j < half_fw; j++)
	{
		int fi0 = half_filter_table_w2 - j;
		int fi1 =  j + half_filter_table_w2 + 1;
		__m512i coef1 = _mm512_loadu_si512((__m512i*)(i_filter_coeffs_with_zeros + fi1));
		__m512i coef0 = _mm512_loadu_si512((__m512i*)(i_filter_coeffs_with_zeros + fi0));

		__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_512, coef0);
		__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_512, coef0);

		__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_512, coef1);
		__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_512, coef1);

		__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
		__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
		
		__m512i tmp1_lo = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
		__m512i tmp1_hi = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

		tmp0_lo = _mm512_add_epi32(tmp0_lo, tmp0_hi);
		tmp0_hi = _mm512_add_epi32(tmp1_lo, tmp1_hi);
		
		__m512i res0 = _mm512_add_epi32(tmp0_lo, tmp0_hi);
		__m256i r8 = _mm256_add_epi32(_mm512_castsi512_si256(res0), _mm512_extracti32x8_epi32(res0, 1));
		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(r8), _mm256_extracti128_si256(r8, 1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);
		dst[dst_row_idx + j] =  ((_mm_cvtsi128_si32(r1) + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}

    __m512i coef0_512 = _mm512_set1_epi16(i_filter_coeffs[0]);
	//This is the core loop
	for (; j < width_rem_size128; j+=128)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(SPAT_FILTER_OUT_RND);
		__m512i res16_512, res20_512, res24_512, res28_512;
		res16_512 = res20_512 = res24_512 = res28_512 = _mm512_set1_epi32(SPAT_FILTER_OUT_RND);

		for (fj = 0; fj < half_fw; fj+=2){
            jj1 = f_l_j + fj*2;

			__m512i coef0 = _mm512_set1_epi32(i32_filter_coeffs[fj]);
			__m512i coef1 = _mm512_set1_epi32(i32_filter_coeffs[fj+1]);

			__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + jj1));
			__m512i d2 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 2));
			__m512i d1 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 1));
			__m512i d3 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 3));
			__m512i d0_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 32));
			__m512i d2_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 34));
			__m512i d1_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 33));
			__m512i d3_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 35));

			__m512i d0_64 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 64));
			__m512i d2_64 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 2 + 64));
			__m512i d1_64 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 1 + 64));
			__m512i d3_64 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 3 + 64));
			__m512i d0_96 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 32 + 64));
			__m512i d2_96 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 34 + 64));
			__m512i d1_96 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 33 + 64));
			__m512i d3_96 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 35 + 64));

			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d0, coef0));
			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d2, coef1));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d1, coef0));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d3, coef1));

			res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(d0_32, coef0));
			res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(d2_32, coef1));
			res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(d1_32, coef0));
			res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(d3_32, coef1));

			res16_512 = _mm512_add_epi32(res16_512, _mm512_madd_epi16(d0_64, coef0));
			res16_512 = _mm512_add_epi32(res16_512, _mm512_madd_epi16(d2_64, coef1));
			res20_512 = _mm512_add_epi32(res20_512, _mm512_madd_epi16(d1_64, coef0));
			res20_512 = _mm512_add_epi32(res20_512, _mm512_madd_epi16(d3_64, coef1));

			res24_512 = _mm512_add_epi32(res24_512, _mm512_madd_epi16(d0_96, coef0));
			res24_512 = _mm512_add_epi32(res24_512, _mm512_madd_epi16(d2_96, coef1));
			res28_512 = _mm512_add_epi32(res28_512, _mm512_madd_epi16(d1_96, coef0));
			res28_512 = _mm512_add_epi32(res28_512, _mm512_madd_epi16(d3_96, coef1));

		}
		__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j));
		__m512i d0_32 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j + 32));
		__m512i d0_64 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j + 64));
		__m512i d0_96 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j + 96));

		__m512i tmp0 = _mm512_unpacklo_epi32(res0_512, res4_512);
		__m512i tmp4 = _mm512_unpackhi_epi32(res0_512, res4_512);
		__m512i tmp8 = _mm512_unpacklo_epi32(res8_512, res12_512);
		__m512i tmp12 = _mm512_unpackhi_epi32(res8_512, res12_512);
		__m512i tmp16 = _mm512_unpacklo_epi32(res16_512, res20_512);
		__m512i tmp20 = _mm512_unpackhi_epi32(res16_512, res20_512);
		__m512i tmp24 = _mm512_unpacklo_epi32(res24_512, res28_512);
		__m512i tmp28 = _mm512_unpackhi_epi32(res24_512, res28_512);
		
		__m512i mul0_lo = _mm512_mullo_epi16(d0, coef0_512);
		__m512i mul0_hi = _mm512_mulhi_epi16(d0, coef0_512);
		__m512i mul0_32_lo = _mm512_mullo_epi16(d0_32, coef0_512);
		__m512i mul0_32_hi = _mm512_mulhi_epi16(d0_32, coef0_512);
		__m512i mul0_64_lo = _mm512_mullo_epi16(d0_64, coef0_512);
		__m512i mul0_64_hi = _mm512_mulhi_epi16(d0_64, coef0_512);
		__m512i mul0_96_lo = _mm512_mullo_epi16(d0_96, coef0_512);
		__m512i mul0_96_hi = _mm512_mulhi_epi16(d0_96, coef0_512);

		__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo, mul0_hi);
		__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo, mul0_hi);
		__m512i tmp0_32_lo = _mm512_unpacklo_epi16(mul0_32_lo, mul0_32_hi);
		__m512i tmp0_32_hi = _mm512_unpackhi_epi16(mul0_32_lo, mul0_32_hi);

		__m512i tmp0_64_lo = _mm512_unpacklo_epi16(mul0_64_lo, mul0_64_hi);
		__m512i tmp0_64_hi = _mm512_unpackhi_epi16(mul0_64_lo, mul0_64_hi);
		__m512i tmp0_96_lo = _mm512_unpacklo_epi16(mul0_96_lo, mul0_96_hi);
		__m512i tmp0_96_hi = _mm512_unpackhi_epi16(mul0_96_lo, mul0_96_hi);
		
		tmp0 = _mm512_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm512_add_epi32(tmp4, tmp0_hi);
		tmp8 = _mm512_add_epi32(tmp8, tmp0_32_lo);
		tmp12 = _mm512_add_epi32(tmp12, tmp0_32_hi);
		tmp16 = _mm512_add_epi32(tmp16, tmp0_64_lo);
		tmp20 = _mm512_add_epi32(tmp20, tmp0_64_hi);
		tmp24 = _mm512_add_epi32(tmp24, tmp0_96_lo);
		tmp28 = _mm512_add_epi32(tmp28, tmp0_96_hi);

		tmp0 = _mm512_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm512_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		tmp8 = _mm512_srai_epi32(tmp8, SPAT_FILTER_OUT_SHIFT);
		tmp12 = _mm512_srai_epi32(tmp12, SPAT_FILTER_OUT_SHIFT);
		tmp16 = _mm512_srai_epi32(tmp16, SPAT_FILTER_OUT_SHIFT);
		tmp20 = _mm512_srai_epi32(tmp20, SPAT_FILTER_OUT_SHIFT);
		tmp24 = _mm512_srai_epi32(tmp24, SPAT_FILTER_OUT_SHIFT);
		tmp28 = _mm512_srai_epi32(tmp28, SPAT_FILTER_OUT_SHIFT);

		res0_512 = _mm512_packs_epi32(tmp0, tmp4);
		res8_512 = _mm512_packs_epi32(tmp8, tmp12);
		res16_512 = _mm512_packs_epi32(tmp16, tmp20);
		res24_512 = _mm512_packs_epi32(tmp24, tmp28);

		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j), res0_512);
		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j + 32), res8_512);
		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j + 64), res16_512);
		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j + 96), res24_512);
	}

	for (; j < width_rem_size64; j+=64)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(SPAT_FILTER_OUT_RND);

		for (fj = 0; fj < half_fw; fj+=2){
            jj1 = f_l_j + fj*2;

			__m512i coef0 = _mm512_set1_epi32(i32_filter_coeffs[fj]);
			__m512i coef1 = _mm512_set1_epi32(i32_filter_coeffs[fj+1]);

			__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + jj1));
			__m512i d2 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 2));
			__m512i d1 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 1));
			__m512i d3 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 3));

			__m512i d0_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 32));
			__m512i d2_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 34));
			__m512i d1_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 33));
			__m512i d3_32 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 35));

			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d0, coef0));
			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d2, coef1));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d1, coef0));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d3, coef1));

			res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(d0_32, coef0));
			res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(d2_32, coef1));
			res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(d1_32, coef0));
			res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(d3_32, coef1));
		}
		__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j));
		__m512i d0_32 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j + 32));
		
		__m512i tmp0 = _mm512_unpacklo_epi32(res0_512, res4_512);
		__m512i tmp4 = _mm512_unpackhi_epi32(res0_512, res4_512);
		__m512i tmp8 = _mm512_unpacklo_epi32(res8_512, res12_512);
		__m512i tmp12 = _mm512_unpackhi_epi32(res8_512, res12_512);

		__m512i mul0_lo = _mm512_mullo_epi16(d0, coef0_512);
		__m512i mul0_hi = _mm512_mulhi_epi16(d0, coef0_512);
		__m512i mul0_32_lo = _mm512_mullo_epi16(d0_32, coef0_512);
		__m512i mul0_32_hi = _mm512_mulhi_epi16(d0_32, coef0_512);

		__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo, mul0_hi);
		__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo, mul0_hi);
		__m512i tmp0_32_lo = _mm512_unpacklo_epi16(mul0_32_lo, mul0_32_hi);
		__m512i tmp0_32_hi = _mm512_unpackhi_epi16(mul0_32_lo, mul0_32_hi);
		
		tmp0 = _mm512_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm512_add_epi32(tmp4, tmp0_hi);
		tmp8 = _mm512_add_epi32(tmp8, tmp0_32_lo);
		tmp12 = _mm512_add_epi32(tmp12, tmp0_32_hi);

		tmp0 = _mm512_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm512_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		tmp8 = _mm512_srai_epi32(tmp8, SPAT_FILTER_OUT_SHIFT);
		tmp12 = _mm512_srai_epi32(tmp12, SPAT_FILTER_OUT_SHIFT);

		res0_512 = _mm512_packs_epi32(tmp0, tmp4);
		res8_512 = _mm512_packs_epi32(tmp8, tmp12);

		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j), res0_512);
		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j + 32), res8_512);
	}
	
    for (; j < width_rem_size32; j+=32)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
		
		res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(SPAT_FILTER_OUT_RND);
		for (fj = 0; fj < half_fw; fj+=2){
            jj1 = f_l_j + fj*2;

			__m512i coef0 = _mm512_set1_epi32(i32_filter_coeffs[fj]);
			__m512i coef1 = _mm512_set1_epi32(i32_filter_coeffs[fj+1]);

			__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + jj1));
			__m512i d2 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 2));
			__m512i d1 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 1));
			__m512i d3 = _mm512_loadu_si512((__m512i*)(tmp + jj1 + 3));

			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d0, coef0));
			res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(d2, coef1));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d1, coef0));
			res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(d3, coef1));
        }
		__m512i d0 = _mm512_loadu_si512((__m512i*)(tmp + f_r_j));
		__m512i tmp0 = _mm512_unpacklo_epi32(res0_512, res4_512);
		__m512i tmp4 = _mm512_unpackhi_epi32(res0_512, res4_512);		
		__m512i mul0_lo = _mm512_mullo_epi16(d0, coef0_512);
		__m512i mul0_hi = _mm512_mulhi_epi16(d0, coef0_512);
		__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo, mul0_hi);
		__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo, mul0_hi);

		tmp0 = _mm512_add_epi32(tmp0, tmp0_lo);
		tmp4 = _mm512_add_epi32(tmp4, tmp0_hi);

		tmp0 = _mm512_srai_epi32(tmp0, SPAT_FILTER_OUT_SHIFT);
		tmp4 = _mm512_srai_epi32(tmp4, SPAT_FILTER_OUT_SHIFT);
		
		res0_512 = _mm512_packs_epi32(tmp0, tmp4);
		_mm512_storeu_si512((__m512i*)(dst + dst_row_idx + j), res0_512);
	}
	
	__m256i coef0_256 = _mm256_set1_epi16(i_filter_coeffs[0]);
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
		
	d0_512 = _mm512_loadu_si512((__m512i*)(tmp + j - 22));
	/**
     * This loop is to handle virtual padding of the right border pixels
     */
	for(; j < width; j++)
	{
		int fi0 = half_filter_table_w2 + width - half_fw*3 - j - 2;
		int fi1 =  j - width + half_fw;
		__m512i coef1 = _mm512_loadu_si512((__m512i*)(i_filter_coeffs_with_zeros + fi1));
		__m512i coef0 = _mm512_loadu_si512((__m512i*)(i_filter_coeffs_with_zeros + fi0));		
		
		__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_512, coef0);
		__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_512, coef0);

		__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_512, coef1);
		__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_512, coef1);

		__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
		__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
		
		__m512i tmp1_lo = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
		__m512i tmp1_hi = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

		tmp0_lo = _mm512_add_epi32(tmp0_lo, tmp0_hi);
		tmp0_hi = _mm512_add_epi32(tmp1_lo, tmp1_hi);
		
		__m512i res0 = _mm512_add_epi32(tmp0_lo, tmp0_hi);
		__m256i r8 = _mm256_add_epi32(_mm512_castsi512_si256(res0), _mm512_extracti32x8_epi32(res0, 1));
		__m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(r8), _mm256_extracti128_si256(r8, 1));
		__m128i r2 = _mm_hadd_epi32(r4, r4);
		__m128i r1 = _mm_hadd_epi32(r2, r2);
		dst[dst_row_idx + j] =  ((_mm_cvtsi128_si32(r1) + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
	}
}

void integer_spatial_filter_avx512(void *src, spat_fil_output_dtype *dst, int width, int height, int bitdepth)
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
	int width_rem_size128 = width - (width % 128);
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

	__m512i res0_512, res4_512, res8_512, res12_512;
	__m512i res0_64_512, res4_64_512, res8_64_512, res12_64_512;
	__m512i zero_512 = _mm512_setzero_si512();
	__m256i res0_256, res4_256, res8_256, res12_256;
	__m256i zero_256 = _mm256_setzero_si256();
	__m128i res0_128, res4_128, res8_128, res12_128;
	__m128i zero_128 = _mm_setzero_si128();

	__m512i perm0 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
	__m512i perm32 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);

    /**
     * The loop i=0 to height is split into 3 parts
     * This is to avoid the if conditions used for virtual padding
     */
    for (i = 0; i < half_fw; i++){

        int diff_i_halffw = i - half_fw;
        int pro_mir_end = -diff_i_halffw - 1;

        /* Vertical pass. */
		j = 0;
		if(8 == bitdepth)
		{
			for (; j < width_rem_size128; j+= 128)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				res0_64_512 = res4_64_512 = res8_64_512 = res12_64_512 = _mm512_set1_epi32(interim_rnd);
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i d64 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j + 64));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d64_lo = _mm512_unpacklo_epi8(d64, zero_512);
					__m512i d64_hi = _mm512_unpackhi_epi8(d64, zero_512);
					
					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					__m512i mul2_lo_512 = _mm512_mullo_epi16(d64_lo, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d64_lo, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d64_hi, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d64_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i d64 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j + 64));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d64_lo = _mm512_unpacklo_epi8(d64, zero_512);
					__m512i d64_hi = _mm512_unpackhi_epi8(d64, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					__m512i mul2_lo_512 = _mm512_mullo_epi16(d64_lo, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d64_lo, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d64_hi, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d64_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);
						
					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_64_512 = _mm512_srai_epi32(res0_64_512, interim_shift);	
				res4_64_512 = _mm512_srai_epi32(res4_64_512, interim_shift);
				res8_64_512 = _mm512_srai_epi32(res8_64_512, interim_shift);
				res12_64_512 = _mm512_srai_epi32(res12_64_512, interim_shift);
					
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				res0_64_512 = _mm512_packs_epi32(res0_64_512, res4_64_512);
				res8_64_512 = _mm512_packs_epi32(res8_64_512, res12_64_512);

				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);
				__m512i r32 = _mm512_permutex2var_epi64(res0_64_512, perm0, res8_64_512);
				__m512i r48 = _mm512_permutex2var_epi64(res0_64_512, perm32, res8_64_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
				_mm512_storeu_si512((__m512i*)(tmp + j + 64), r32);
				_mm512_storeu_si512((__m512i*)(tmp + j + 96), r48);
			}

			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					
					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}
				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
						
					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}
				
				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);

				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
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
					__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + ii * src_px_stride + j));
					__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

					__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
					__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);
					
					__m256i mul0_lo_256 = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi_256 = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo_256 = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi_256 = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo_256 = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi_256 = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo_256 = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi_256 = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);

					res0_256 = _mm256_add_epi32(tmp0_lo_256, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi_256, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo_256, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi_256, res12_256);
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
					__m256i tmp0_lo_256 = _mm256_unpacklo_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp0_hi_256 = _mm256_unpackhi_epi16(mul0_lo_256, mul0_hi_256);
					__m256i tmp1_lo_256 = _mm256_unpacklo_epi16(mul1_lo_256, mul1_hi_256);
					__m256i tmp1_hi_256 = _mm256_unpackhi_epi16(mul1_lo_256, mul1_hi_256);
						
					res0_256 = _mm256_add_epi32(tmp0_lo_256, res0_256);
					res4_256 = _mm256_add_epi32(tmp0_hi_256, res4_256);
					res8_256 = _mm256_add_epi32(tmp1_lo_256, res8_256);
					res12_256 = _mm256_add_epi32(tmp1_hi_256, res12_256);
				}
				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);
								
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);

				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				
				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16)
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
			for (; j < width_rem_size128; j+=128)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				res0_64_512 = res4_64_512 = res8_64_512 = res12_64_512 = _mm512_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i d2 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 64));
					__m512i d3 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 96));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);
					__m512i mul2_lo_512 = _mm512_mullo_epi16(d2, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d2, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d3, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d3, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i d2 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 64));
					__m512i d3 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 96));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);
					__m512i mul2_lo_512 = _mm512_mullo_epi16(d2, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d2, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d3, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d3, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);
				res0_64_512 = _mm512_srai_epi32(res0_64_512, interim_shift);
				res4_64_512 = _mm512_srai_epi32(res4_64_512, interim_shift);
				res8_64_512 = _mm512_srai_epi32(res8_64_512, interim_shift);
				res12_64_512 = _mm512_srai_epi32(res12_64_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				res0_64_512 = _mm512_packs_epi32(res0_64_512, res4_64_512);
				res8_64_512 = _mm512_packs_epi32(res8_64_512, res12_64_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), res8_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 64), res0_64_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 96), res8_64_512);
			}

			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), res8_512);
			}

			for (; j < width_rem_size32; j+=32)
			{
				res0_512 = res4_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi <= pro_mir_end; fi++){
					ii = pro_mir_end - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
			}

			for (; j < width_rem_size16; j+=16)
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
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j + 8));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d1, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d1, coef);

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
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j + 8));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d1, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d1, coef);

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
		integer_horizontal_filter_avx512(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }
    //This is the core loop
    for ( ; i < (height - half_fw); i++){

        int f_l_i = i - half_fw;
        int f_r_i = i + half_fw;
        /* Vertical pass. */
		j = 0;
		if(8 == bitdepth)
		{
			for (; j < width_rem_size128; j+=128)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				res0_64_512 = res4_64_512 = res8_64_512 = res12_64_512 = _mm512_set1_epi32(interim_rnd);

				for (fi = 0; fi < (half_fw); fi+=2)
				{
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii1 * src_px_stride + j));
					__m512i d20 = _mm512_loadu_si512((__m512i*)(src_8b + ii2 * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m512i d19 = _mm512_loadu_si512((__m512i*)(src_8b + (ii2 - 1) * src_px_stride + j));

					__m512i d0_64 = _mm512_loadu_si512((__m512i*)(src_8b + ii1 * src_px_stride + j + 64));
					__m512i d20_64 = _mm512_loadu_si512((__m512i*)(src_8b + ii2 * src_px_stride + j + 64));
					__m512i d1_64 = _mm512_loadu_si512((__m512i*)(src_8b + (ii1 + 1) * src_px_stride + j + 64));
					__m512i d19_64 = _mm512_loadu_si512((__m512i*)(src_8b + (ii2 - 1) * src_px_stride + j + 64));
					
					__m512i f0_1 = _mm512_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d20_lo = _mm512_unpacklo_epi8(d20, zero_512);
					__m512i d20_hi = _mm512_unpackhi_epi8(d20, zero_512);
					__m512i d1_lo = _mm512_unpacklo_epi8(d1, zero_512);
					__m512i d1_hi = _mm512_unpackhi_epi8(d1, zero_512);
					__m512i d19_lo = _mm512_unpacklo_epi8(d19, zero_512);
					__m512i d19_hi = _mm512_unpackhi_epi8(d19, zero_512);

					__m512i d0_64_lo = _mm512_unpacklo_epi8(d0_64, zero_512);
					__m512i d0_64_hi = _mm512_unpackhi_epi8(d0_64, zero_512);
					__m512i d20_64_lo = _mm512_unpacklo_epi8(d20_64, zero_512);
					__m512i d20_64_hi = _mm512_unpackhi_epi8(d20_64, zero_512);
					__m512i d1_64_lo = _mm512_unpacklo_epi8(d1_64, zero_512);
					__m512i d1_64_hi = _mm512_unpackhi_epi8(d1_64, zero_512);
					__m512i d19_64_lo = _mm512_unpacklo_epi8(d19_64, zero_512);
					__m512i d19_64_hi = _mm512_unpackhi_epi8(d19_64, zero_512);

					d0_lo = _mm512_add_epi16(d0_lo, d20_lo);
					d0_hi = _mm512_add_epi16(d0_hi, d20_hi);
					d1_lo = _mm512_add_epi16(d1_lo, d19_lo);
					d1_hi = _mm512_add_epi16(d1_hi, d19_hi);

					d0_64_lo = _mm512_add_epi16(d0_64_lo, d20_64_lo);
					d0_64_hi = _mm512_add_epi16(d0_64_hi, d20_64_hi);
					d1_64_lo = _mm512_add_epi16(d1_64_lo, d19_64_lo);
					d1_64_hi = _mm512_add_epi16(d1_64_hi, d19_64_hi);

					__m512i l0_20_1_19_0 = _mm512_unpacklo_epi16(d0_lo, d1_lo);
					__m512i l0_20_1_19_4 = _mm512_unpackhi_epi16(d0_lo, d1_lo);
					__m512i l0_20_1_19_8 = _mm512_unpacklo_epi16(d0_hi, d1_hi);
					__m512i l0_20_1_19_12 = _mm512_unpackhi_epi16(d0_hi, d1_hi);

					__m512i l0_20_1_19_0_64 = _mm512_unpacklo_epi16(d0_64_lo, d1_64_lo);
					__m512i l0_20_1_19_4_64 = _mm512_unpackhi_epi16(d0_64_lo, d1_64_lo);
					__m512i l0_20_1_19_8_64 = _mm512_unpacklo_epi16(d0_64_hi, d1_64_hi);
					__m512i l0_20_1_19_12_64 = _mm512_unpackhi_epi16(d0_64_hi, d1_64_hi);

					res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(l0_20_1_19_0, f0_1));
					res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(l0_20_1_19_4, f0_1));
					res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(l0_20_1_19_8, f0_1));
					res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(l0_20_1_19_12, f0_1));

					res0_64_512 = _mm512_add_epi32(res0_64_512, _mm512_madd_epi16(l0_20_1_19_0_64, f0_1));
					res4_64_512 = _mm512_add_epi32(res4_64_512, _mm512_madd_epi16(l0_20_1_19_4_64, f0_1));
					res8_64_512 = _mm512_add_epi32(res8_64_512, _mm512_madd_epi16(l0_20_1_19_8_64, f0_1));
					res12_64_512 = _mm512_add_epi32(res12_64_512, _mm512_madd_epi16(l0_20_1_19_12_64, f0_1));
				}

				__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + i * src_px_stride + j));
				__m512i d0_64 = _mm512_loadu_si512((__m512i*)(src_8b + i * src_px_stride + j + 64));
				__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

				__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
				__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
				__m512i d0_64_lo = _mm512_unpacklo_epi8(d0_64, zero_512);
				__m512i d0_64_hi = _mm512_unpackhi_epi8(d0_64, zero_512);
				
				__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
				__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
				__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
				__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

				__m512i mul0_64_lo_512 = _mm512_mullo_epi16(d0_64_lo, coef);
				__m512i mul0_64_hi_512 = _mm512_mulhi_epi16(d0_64_lo, coef);
				__m512i mul1_64_lo_512 = _mm512_mullo_epi16(d0_64_hi, coef);
				__m512i mul1_64_hi_512 = _mm512_mulhi_epi16(d0_64_hi, coef);
				
				__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
				__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

				__m512i tmp0_64_lo_512 = _mm512_unpacklo_epi16(mul0_64_lo_512, mul0_64_hi_512);
				__m512i tmp0_64_hi_512 = _mm512_unpackhi_epi16(mul0_64_lo_512, mul0_64_hi_512);
				__m512i tmp1_64_lo_512 = _mm512_unpacklo_epi16(mul1_64_lo_512, mul1_64_hi_512);
				__m512i tmp1_64_hi_512 = _mm512_unpackhi_epi16(mul1_64_lo_512, mul1_64_hi_512);

				res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
				res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
				res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

				res0_64_512 = _mm512_add_epi32(tmp0_64_lo_512, res0_64_512);
				res4_64_512 = _mm512_add_epi32(tmp0_64_hi_512, res4_64_512);
				res8_64_512 = _mm512_add_epi32(tmp1_64_lo_512, res8_64_512);
				res12_64_512 = _mm512_add_epi32(tmp1_64_hi_512, res12_64_512);

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_64_512 = _mm512_srai_epi32(res0_64_512, interim_shift);	
				res4_64_512 = _mm512_srai_epi32(res4_64_512, interim_shift);
				res8_64_512 = _mm512_srai_epi32(res8_64_512, interim_shift);
				res12_64_512 = _mm512_srai_epi32(res12_64_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);

				res0_64_512 = _mm512_packs_epi32(res0_64_512, res4_64_512);
				res8_64_512 = _mm512_packs_epi32(res8_64_512, res12_64_512);

				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);
				__m512i r32 = _mm512_permutex2var_epi64(res0_64_512, perm0, res8_64_512);
				__m512i r48 = _mm512_permutex2var_epi64(res0_64_512, perm32, res8_64_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
				_mm512_storeu_si512((__m512i*)(tmp + j + 64), r32);
				_mm512_storeu_si512((__m512i*)(tmp + j + 96), r48);
			}

			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				
				for (fi = 0; fi < (half_fw); fi+=2){

					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii1 * src_px_stride + j));
					__m512i d20 = _mm512_loadu_si512((__m512i*)(src_8b + ii2 * src_px_stride + j));

					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_8b + (ii1 + 1) * src_px_stride + j));
					__m512i d19 = _mm512_loadu_si512((__m512i*)(src_8b + (ii2 - 1) * src_px_stride + j));
					__m512i f0_1 = _mm512_set1_epi32(i32_filter_coeffs[fi / 2]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d20_lo = _mm512_unpacklo_epi8(d20, zero_512);
					__m512i d20_hi = _mm512_unpackhi_epi8(d20, zero_512);

					__m512i d1_lo = _mm512_unpacklo_epi8(d1, zero_512);
					__m512i d1_hi = _mm512_unpackhi_epi8(d1, zero_512);
					__m512i d19_lo = _mm512_unpacklo_epi8(d19, zero_512);
					__m512i d19_hi = _mm512_unpackhi_epi8(d19, zero_512);

					d0_lo = _mm512_add_epi16(d0_lo, d20_lo);
					d0_hi = _mm512_add_epi16(d0_hi, d20_hi);
					d1_lo = _mm512_add_epi16(d1_lo, d19_lo);
					d1_hi = _mm512_add_epi16(d1_hi, d19_hi);

					__m512i l0_20_1_19_0 = _mm512_unpacklo_epi16(d0_lo, d1_lo);
					__m512i l0_20_1_19_4 = _mm512_unpackhi_epi16(d0_lo, d1_lo);
					__m512i l0_20_1_19_8 = _mm512_unpacklo_epi16(d0_hi, d1_hi);
					__m512i l0_20_1_19_12 = _mm512_unpackhi_epi16(d0_hi, d1_hi);

					res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(l0_20_1_19_0, f0_1));
					res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(l0_20_1_19_4, f0_1));
					res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(l0_20_1_19_8, f0_1));
					res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(l0_20_1_19_12, f0_1));
				}

				__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + i * src_px_stride + j));
				__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
				__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
				__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
				
				__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
				__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
				__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
				__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);
				
				__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
				__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

				res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
				res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
				res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				
				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
			}

			for (; j < width_rem_size32; j+=32)
			{
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

					res0_256 = _mm256_add_epi32(res0_256, _mm256_madd_epi16(l0_20_1_19_0, f0_1));
					res4_256 = _mm256_add_epi32(res4_256, _mm256_madd_epi16(l0_20_1_19_4, f0_1));
					res8_256 = _mm256_add_epi32(res8_256, _mm256_madd_epi16(l0_20_1_19_8, f0_1));
					res12_256 = _mm256_add_epi32(res12_256, _mm256_madd_epi16(l0_20_1_19_12, f0_1));
				}				
				__m256i d0 = _mm256_loadu_si256((__m256i*)(src_8b + i * src_px_stride + j));
				__m256i coef = _mm256_set1_epi16(i_filter_coeffs[fi]);

				__m256i d0_lo = _mm256_unpacklo_epi8(d0, zero_256);
				__m256i d0_hi = _mm256_unpackhi_epi8(d0, zero_256);

				__m256i mul0_lo = _mm256_mullo_epi16(d0_lo, coef);
				__m256i mul0_hi = _mm256_mulhi_epi16(d0_lo, coef);
				__m256i mul1_lo = _mm256_mullo_epi16(d0_hi, coef);
				__m256i mul1_hi = _mm256_mulhi_epi16(d0_hi, coef);
				
				__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
				__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);
				__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo, mul1_hi);
				__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo, mul1_hi);
				
				res0_256 = _mm256_add_epi32(tmp0_lo, res0_256);
				res4_256 = _mm256_add_epi32(tmp0_hi, res4_256);
				res8_256 = _mm256_add_epi32(tmp1_lo, res8_256);
				res12_256 = _mm256_add_epi32(tmp1_hi, res12_256);

				res0_256 = _mm256_srai_epi32(res0_256, interim_shift);	
				res4_256 = _mm256_srai_epi32(res4_256, interim_shift);
				res8_256 = _mm256_srai_epi32(res8_256, interim_shift);
				res12_256 = _mm256_srai_epi32(res12_256, interim_shift);
				
				res0_256 = _mm256_packs_epi32(res0_256, res4_256);
				res8_256 = _mm256_packs_epi32(res8_256, res12_256);
				
				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				_mm256_storeu_si256((__m256i*)(tmp + j), r0);
				_mm256_storeu_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16)
			{
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

			for (; j < width_rem_size8; j+=8)
			{
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

			for (; j < width; j++)
			{
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
			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2)
				{
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii1 * src_px_stride + j));
					__m512i d20 = _mm512_loadu_si512((__m512i*)(src_hbd + ii2 * src_px_stride + j));
					__m512i d0_32 = _mm512_loadu_si512((__m512i*)(src_hbd + ii1 * src_px_stride + j + 32));
					__m512i d20_32 = _mm512_loadu_si512((__m512i*)(src_hbd + ii2 * src_px_stride + j + 32));

					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m512i d19 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m512i d1_32 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii1 + 1) * src_px_stride + j + 32));
					__m512i d19_32 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii2 - 1) * src_px_stride + j + 32));

					__m512i f0_1 = _mm512_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm512_add_epi16(d0, d20);
					d0_32 = _mm512_add_epi16(d0_32, d20_32);
					d1 = _mm512_add_epi16(d1, d19);
					d1_32 = _mm512_add_epi16(d1_32, d19_32);

					__m512i l0_20_1_19_0 = _mm512_unpacklo_epi16(d0, d1);
					__m512i l0_20_1_19_4 = _mm512_unpackhi_epi16(d0, d1);
					__m512i l0_20_1_19_16 = _mm512_unpacklo_epi16(d0_32, d1_32);
					__m512i l0_20_1_19_20 = _mm512_unpackhi_epi16(d0_32, d1_32);
					
					res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(l0_20_1_19_0, f0_1));
					res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(l0_20_1_19_4, f0_1));
					res8_512 = _mm512_add_epi32(res8_512, _mm512_madd_epi16(l0_20_1_19_16, f0_1));
					res12_512 = _mm512_add_epi32(res12_512, _mm512_madd_epi16(l0_20_1_19_20, f0_1));
				}

				__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + i * src_px_stride + j));
				__m512i d0_32 = _mm512_loadu_si512((__m512i*)(src_hbd + i * src_px_stride + j + 32));
				__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
				__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
				__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
				__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_32, coef);
				__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_32, coef);
				
				__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
				__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
				
				res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
				res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
				res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), res8_512);
			}
		
			for (; j < width_rem_size32; j+=32)
			{
				res0_512 = res4_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi+=2)
				{
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii1 * src_px_stride + j));
					__m512i d20 = _mm512_loadu_si512((__m512i*)(src_hbd + ii2 * src_px_stride + j));

					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m512i d19 = _mm512_loadu_si512((__m512i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m512i f0_1 = _mm512_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm512_add_epi16(d0, d20);
					d1 = _mm512_add_epi16(d1, d19);

					__m512i l0_20_1_19_0 = _mm512_unpacklo_epi16(d0, d1);
					__m512i l0_20_1_19_4 = _mm512_unpackhi_epi16(d0, d1);
					
					res0_512 = _mm512_add_epi32(res0_512, _mm512_madd_epi16(l0_20_1_19_0, f0_1));
					res4_512 = _mm512_add_epi32(res4_512, _mm512_madd_epi16(l0_20_1_19_4, f0_1));
				}

				__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + i * src_px_stride + j));
				__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);
				__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
				__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
				
				__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
				__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
				
				res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
				res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
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
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii1 * src_px_stride + j));
					__m128i d20 = _mm_loadu_si128((__m128i*)(src_hbd + ii2 * src_px_stride + j));
					__m128i d0_8 = _mm_loadu_si128((__m128i*)(src_hbd + ii1 * src_px_stride + j + 8));
					__m128i d20_8 = _mm_loadu_si128((__m128i*)(src_hbd + ii2 * src_px_stride + j + 8));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + (ii1 + 1) * src_px_stride + j));
					__m128i d19 = _mm_loadu_si128((__m128i*)(src_hbd + (ii2 - 1) * src_px_stride + j));
					__m128i d1_8 = _mm_loadu_si128((__m128i*)(src_hbd + (ii1 + 1) * src_px_stride + j + 8));
					__m128i d19_8 = _mm_loadu_si128((__m128i*)(src_hbd + (ii2 - 1) * src_px_stride + j + 8));
					__m128i f0_1 = _mm_set1_epi32(i32_filter_coeffs[fi / 2]);

					d0 = _mm_add_epi16(d0, d20);
					d0_8 = _mm_add_epi16(d0_8, d20_8);
					d1 = _mm_add_epi16(d1, d19);
					d1_8 = _mm_add_epi16(d1_8, d19_8);

					__m128i l0_20_1_19_0 = _mm_unpacklo_epi16(d0, d1);
					__m128i l0_20_1_19_4 = _mm_unpackhi_epi16(d0, d1);
					__m128i l0_20_1_19_8 = _mm_unpacklo_epi16(d0_8, d1_8);
					__m128i l0_20_1_19_16 = _mm_unpackhi_epi16(d0_8, d1_8);

					res0_128 = _mm_add_epi32(res0_128, _mm_madd_epi16(l0_20_1_19_0, f0_1));
					res4_128 = _mm_add_epi32(res4_128, _mm_madd_epi16(l0_20_1_19_4, f0_1));
					res8_128 = _mm_add_epi32(res8_128, _mm_madd_epi16(l0_20_1_19_8, f0_1));
					res12_128 = _mm_add_epi32(res12_128, _mm_madd_epi16(l0_20_1_19_16, f0_1));
				}
				__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + i * src_px_stride + j));
				__m128i d0_16 = _mm_loadu_si128((__m128i*)(src_hbd + i * src_px_stride + j + 8));
				__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
				__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
				__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
				__m128i mul1_lo_128 = _mm_mullo_epi16(d0_16, coef);
				__m128i mul1_hi_128 = _mm_mulhi_epi16(d0_16, coef);
				
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
        integer_horizontal_filter_avx512(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }
    /**
     * This loop is to handle virtual padding of the bottom border pixels
     */
    for (; i < height; i++){

        int diff_i_halffw = i - half_fw;
        int epi_mir_i = 2 * height - diff_i_halffw - 1;
        int epi_last_i  = height - diff_i_halffw;
        j = 0;
        /* Vertical pass. */

		if(8 == bitdepth)
		{
			for (; j < width_rem_size128; j+=128)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				res0_64_512 = res4_64_512 = res8_64_512 = res12_64_512 = _mm512_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i d64 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j + 64));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d64_lo = _mm512_unpacklo_epi8(d64, zero_512);
					__m512i d64_hi = _mm512_unpackhi_epi8(d64, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					__m512i mul2_lo_512 = _mm512_mullo_epi16(d64_lo, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d64_lo, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d64_hi, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d64_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);

					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i d64 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j + 64));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);
					__m512i d64_lo = _mm512_unpacklo_epi8(d64, zero_512);
					__m512i d64_hi = _mm512_unpackhi_epi8(d64, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					__m512i mul2_lo_512 = _mm512_mullo_epi16(d64_lo, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d64_lo, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d64_hi, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d64_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);
						
					res0_512 = _mm512_add_epi32(tmp0_lo, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi, res12_512);

					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}
				
				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_64_512 = _mm512_srai_epi32(res0_64_512, interim_shift);	
				res4_64_512 = _mm512_srai_epi32(res4_64_512, interim_shift);
				res8_64_512 = _mm512_srai_epi32(res8_64_512, interim_shift);
				res12_64_512 = _mm512_srai_epi32(res12_64_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				res0_64_512 = _mm512_packs_epi32(res0_64_512, res4_64_512);
				res8_64_512 = _mm512_packs_epi32(res8_64_512, res12_64_512);

				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);
				__m512i r32 = _mm512_permutex2var_epi64(res0_64_512, perm0, res8_64_512);
				__m512i r48 = _mm512_permutex2var_epi64(res0_64_512, perm32, res8_64_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
				_mm512_storeu_si512((__m512i*)(tmp + j + 64), r32);
				_mm512_storeu_si512((__m512i*)(tmp + j + 96), r48);
			}

			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}
				//This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_8b + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i d0_lo = _mm512_unpacklo_epi8(d0, zero_512);
					__m512i d0_hi = _mm512_unpackhi_epi8(d0, zero_512);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0_lo, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0_lo, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d0_hi, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
						
					res0_512 = _mm512_add_epi32(tmp0_lo, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi, res12_512);
				}
				
				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				
				__m512i r0 = _mm512_permutex2var_epi64(res0_512, perm0, res8_512);
				__m512i r16 = _mm512_permutex2var_epi64(res0_512, perm32, res8_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), r0);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), r16);
				
			}

			for (; j < width_rem_size32; j+=32)
			{
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

					__m256i mul0_lo = _mm256_mullo_epi16(d0_lo, coef);
					__m256i mul0_hi = _mm256_mulhi_epi16(d0_lo, coef);
					__m256i mul1_lo = _mm256_mullo_epi16(d0_hi, coef);
					__m256i mul1_hi = _mm256_mulhi_epi16(d0_hi, coef);

					// regroup the 2 parts of the result
					__m256i tmp0_lo = _mm256_unpacklo_epi16(mul0_lo, mul0_hi);
					__m256i tmp0_hi = _mm256_unpackhi_epi16(mul0_lo, mul0_hi);
					__m256i tmp1_lo = _mm256_unpacklo_epi16(mul1_lo, mul1_hi);
					__m256i tmp1_hi = _mm256_unpackhi_epi16(mul1_lo, mul1_hi);
						
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

				__m256i r0 = _mm256_permute2x128_si256(res0_256, res8_256, 0x20);
				__m256i r8 = _mm256_permute2x128_si256(res0_256, res8_256, 0x31);
				_mm256_store_si256((__m256i*)(tmp + j), r0);
				_mm256_store_si256((__m256i*)(tmp + j + 16), r8);
			}

			for (; j < width_rem_size16; j+=16)
			{
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

			for (; j < width_rem_size8; j+=8)
			{
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
			for (; j < width_rem_size128; j+=128)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);
				res0_64_512 = res4_64_512 = res8_64_512 = res12_64_512 = _mm512_set1_epi32(interim_rnd);
				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi < epi_last_i; fi++){
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i d2 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 64));
					__m512i d3 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 96));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);
					__m512i mul2_lo_512 = _mm512_mullo_epi16(d2, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d2, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d3, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d3, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i d2 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 64));
					__m512i d3 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 96));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);
					__m512i mul2_lo_512 = _mm512_mullo_epi16(d2, coef);
					__m512i mul2_hi_512 = _mm512_mulhi_epi16(d2, coef);
					__m512i mul3_lo_512 = _mm512_mullo_epi16(d3, coef);
					__m512i mul3_hi_512 = _mm512_mulhi_epi16(d3, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp2_lo_512 = _mm512_unpacklo_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp2_hi_512 = _mm512_unpackhi_epi16(mul2_lo_512, mul2_hi_512);
					__m512i tmp3_lo_512 = _mm512_unpacklo_epi16(mul3_lo_512, mul3_hi_512);
					__m512i tmp3_hi_512 = _mm512_unpackhi_epi16(mul3_lo_512, mul3_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
					res0_64_512 = _mm512_add_epi32(tmp2_lo_512, res0_64_512);
					res4_64_512 = _mm512_add_epi32(tmp2_hi_512, res4_64_512);
					res8_64_512 = _mm512_add_epi32(tmp3_lo_512, res8_64_512);
					res12_64_512 = _mm512_add_epi32(tmp3_hi_512, res12_64_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);
				res0_64_512 = _mm512_srai_epi32(res0_64_512, interim_shift);
				res4_64_512 = _mm512_srai_epi32(res4_64_512, interim_shift);
				res8_64_512 = _mm512_srai_epi32(res8_64_512, interim_shift);
				res12_64_512 = _mm512_srai_epi32(res12_64_512, interim_shift);
								
				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);
				res0_64_512 = _mm512_packs_epi32(res0_64_512, res4_64_512);
				res8_64_512 = _mm512_packs_epi32(res8_64_512, res12_64_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), res8_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 64), res0_64_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 96), res8_64_512);
			}

			for (; j < width_rem_size64; j+=64)
			{
				res0_512 = res4_512 = res8_512 = res12_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi < epi_last_i; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i d1 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j + 32));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);
					__m512i mul1_lo_512 = _mm512_mullo_epi16(d1, coef);
					__m512i mul1_hi_512 = _mm512_mulhi_epi16(d1, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp1_lo_512 = _mm512_unpacklo_epi16(mul1_lo_512, mul1_hi_512);
					__m512i tmp1_hi_512 = _mm512_unpackhi_epi16(mul1_lo_512, mul1_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
					res8_512 = _mm512_add_epi32(tmp1_lo_512, res8_512);
					res12_512 = _mm512_add_epi32(tmp1_hi_512, res12_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);
				res8_512 = _mm512_srai_epi32(res8_512, interim_shift);
				res12_512 = _mm512_srai_epi32(res12_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);
				res8_512 = _mm512_packs_epi32(res8_512, res12_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
				_mm512_storeu_si512((__m512i*)(tmp + j + 32), res8_512);
			}

			for (; j < width_rem_size32; j+=32)
			{
				res0_512 = res4_512 = _mm512_set1_epi32(interim_rnd);

				/**
				 * The full loop is from fi = 0 to fwidth
				 * During the loop when the centre pixel is at i, 
				 * the top part is available only till i-(fwidth/2) >= 0, 
				 * hence padding (border mirroring) is required when i-fwidth/2 < 0
				 */
				//This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
				for (fi = 0; fi < epi_last_i; fi++)
				{
					ii = diff_i_halffw + fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				}

				//Here the normal loop is executed where ii = i - fwidth / 2 + fi
				for ( ; fi < fwidth; fi++)
				{
					ii = epi_mir_i - fi;
					__m512i d0 = _mm512_loadu_si512((__m512i*)(src_hbd + ii * src_px_stride + j));
					__m512i coef = _mm512_set1_epi16(i_filter_coeffs[fi]);

					__m512i mul0_lo_512 = _mm512_mullo_epi16(d0, coef);
					__m512i mul0_hi_512 = _mm512_mulhi_epi16(d0, coef);

					// regroup the 2 parts of the result
					__m512i tmp0_lo_512 = _mm512_unpacklo_epi16(mul0_lo_512, mul0_hi_512);
					__m512i tmp0_hi_512 = _mm512_unpackhi_epi16(mul0_lo_512, mul0_hi_512);

					res0_512 = _mm512_add_epi32(tmp0_lo_512, res0_512);
					res4_512 = _mm512_add_epi32(tmp0_hi_512, res4_512);
				}

				res0_512 = _mm512_srai_epi32(res0_512, interim_shift);	
				res4_512 = _mm512_srai_epi32(res4_512, interim_shift);

				res0_512 = _mm512_packs_epi32(res0_512, res4_512);

				_mm512_storeu_si512((__m512i*)(tmp + j), res0_512);
			}

			for (; j < width_rem_size16; j+=16){
				res0_128 = res4_128 = res8_128 = res12_128 = _mm_set1_epi32(interim_rnd);

				//Here the normal loop is executed where ii = i - fwidth/2 + fi
				for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j + 8));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);
					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d1, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d1, coef);

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
					__m128i d0 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j));
					__m128i d1 = _mm_loadu_si128((__m128i*)(src_hbd + ii * src_px_stride + j + 8));
					__m128i coef = _mm_set1_epi16(i_filter_coeffs[fi]);

					__m128i mul0_lo_128 = _mm_mullo_epi16(d0, coef);
					__m128i mul0_hi_128 = _mm_mulhi_epi16(d0, coef);
					__m128i mul1_lo_128 = _mm_mullo_epi16(d1, coef);
					__m128i mul1_hi_128 = _mm_mulhi_epi16(d1, coef);

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
        integer_horizontal_filter_avx512(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }

    aligned_free(tmp);

    return;
}

void integer_funque_dwt2_inplace_csf_avx512(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4],
                                     int min_theta, int max_theta, uint16_t interim_rnd_factors[4],
                                     uint8_t interim_shift_factors[4], int level)

{
	UNUSED(min_theta);
	UNUSED(max_theta);
	UNUSED(level);

    dwt2_dtype *src_ptr0=src->bands[0];
	dwt2_dtype *src_ptr1=src->bands[2];
	dwt2_dtype *src_ptr2=src->bands[3];
	dwt2_dtype *src_ptr3=src->bands[1];

    dwt2_dtype *dst_ptr0=src->bands[0];
	dwt2_dtype *dst_ptr1=src->bands[2];
	dwt2_dtype *dst_ptr2=src->bands[3];
	dwt2_dtype *dst_ptr3=src->bands[1];

    //dwt2_dtype *angles[4] = {src->bands[0], src->bands[2], src->bands[3], src->bands[1]};
    int px_stride = src->stride / sizeof(dwt2_dtype);

    int left = 0;
    int top = 0;

    /*  changes made wrt to my code will be changed later */
    min_theta++;min_theta--;max_theta++;max_theta--;interim_shift_factors[0]++;interim_shift_factors[0]--;interim_rnd_factors[0]++;interim_rnd_factors[0]--;level++;level--;
    /*upto here*/

    /*
    int right = src->crop_width; 
    int bottom = src->crop_height;
    */

    int right = src->width; 
    int bottom = src->height;

    int i, j, theta, src_offset, dst_offset;
    spat_fil_accum_dtype mul_val;
    dwt2_dtype dst_val;
    int width_rem=src->width - (src->width)%32;
// 

    // x86 variables
    __m512i d0,mul0_lo,mul0_hi,tmp0_lo,tmp0_hi;
    __m512i d1,mul1_lo,mul1_hi,tmp1_lo,tmp1_hi;
    __m512i d2,mul2_lo,mul2_hi,tmp2_lo,tmp2_hi;
    __m512i d3,mul3_lo,mul3_hi,tmp3_lo,tmp3_hi;
    __m512i res0,res1,res2,res3,fres0,fres1,fres2,fres3;
    __m512i result0_lo,result0_hi,result1_lo,result1_hi,result2_lo,result2_hi,result3_lo,result3_hi;
    __m512i mask= _mm512_set_epi16(0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF,0,0xFFFF);
   

    __m512i coef_0 = _mm512_set1_epi16(factors[0]);
    __m512i coef_1 = _mm512_set1_epi16(factors[1]);
    __m512i coef_2 = _mm512_set1_epi16(factors[2]);
    __m512i coef_3 = _mm512_set1_epi16(factors[3]);

    __m512i rnd_0= _mm512_set1_epi32(interim_rnd_factors[0]);
    __m512i rnd_1= _mm512_set1_epi32(interim_rnd_factors[1]);
    __m512i rnd_2= _mm512_set1_epi32(interim_rnd_factors[2]);
    __m512i rnd_3= _mm512_set1_epi32(interim_rnd_factors[3]);
    

        for(i = top; i < bottom; ++i) {
      
            src_offset = px_stride*i;
            dst_offset =  px_stride*i;
         

   
            for(j = left; j < width_rem; j=j+32) {
                
           
            d0 = _mm512_loadu_si512((__m512i*)(src_ptr0+src_offset+j ));
            d1 = _mm512_loadu_si512((__m512i*)(src_ptr1+src_offset+j ));
            d2 = _mm512_loadu_si512((__m512i*)(src_ptr2+src_offset+j ));
            d3 = _mm512_loadu_si512((__m512i*)(src_ptr3+src_offset+j ));

            mul0_lo = _mm512_mullo_epi16(d0, coef_0);
			mul0_hi = _mm512_mulhi_epi16(d0, coef_0);
            mul1_lo = _mm512_mullo_epi16(d1, coef_1);
			mul1_hi = _mm512_mulhi_epi16(d1, coef_1);
            mul2_lo = _mm512_mullo_epi16(d2, coef_2);
			mul2_hi = _mm512_mulhi_epi16(d2, coef_2);
            mul3_lo = _mm512_mullo_epi16(d3, coef_3);
			mul3_hi = _mm512_mulhi_epi16(d3, coef_3);

            tmp0_lo = _mm512_unpacklo_epi16(mul0_lo, mul0_hi);
			tmp0_hi = _mm512_unpackhi_epi16(mul0_lo, mul0_hi);
            tmp1_lo = _mm512_unpacklo_epi16(mul1_lo, mul1_hi);
			tmp1_hi = _mm512_unpackhi_epi16(mul1_lo, mul1_hi);
            tmp2_lo = _mm512_unpacklo_epi16(mul2_lo, mul2_hi);
			tmp2_hi = _mm512_unpackhi_epi16(mul2_lo, mul2_hi);
            tmp3_lo = _mm512_unpacklo_epi16(mul3_lo, mul3_hi);
			tmp3_hi = _mm512_unpackhi_epi16(mul3_lo, mul3_hi);

           tmp0_lo= _mm512_add_epi32(tmp0_lo, rnd_0);
           tmp0_hi= _mm512_add_epi32(tmp0_hi, rnd_0);
           
           tmp1_lo= _mm512_add_epi32(tmp1_lo, rnd_1);
           tmp1_hi= _mm512_add_epi32(tmp1_hi, rnd_1);
           
           tmp2_lo= _mm512_add_epi32(tmp2_lo, rnd_2);
           tmp2_hi= _mm512_add_epi32(tmp2_hi, rnd_2);

           tmp3_lo= _mm512_add_epi32(tmp3_lo, rnd_3);
           tmp3_hi= _mm512_add_epi32(tmp3_hi, rnd_3);

       
        
           tmp0_lo= _mm512_srai_epi32(tmp0_lo, interim_shift_factors[0]);
           tmp0_hi= _mm512_srai_epi32(tmp0_hi, interim_shift_factors[0]);

           tmp1_lo= _mm512_srai_epi32(tmp1_lo, interim_shift_factors[1]);
           tmp1_hi= _mm512_srai_epi32(tmp1_hi, interim_shift_factors[1]);

           tmp2_lo= _mm512_srai_epi32(tmp2_lo, interim_shift_factors[2]);
           tmp2_hi= _mm512_srai_epi32(tmp2_hi, interim_shift_factors[2]);

           tmp3_lo= _mm512_srai_epi32(tmp3_lo, interim_shift_factors[3]);
           tmp3_hi= _mm512_srai_epi32(tmp3_hi, interim_shift_factors[3]);
           


   
         result0_lo= _mm512_and_si512(tmp0_lo, mask); 
         result0_hi= _mm512_and_si512(tmp0_hi, mask);
         result1_lo= _mm512_and_si512(tmp1_lo, mask); 
         result1_hi= _mm512_and_si512(tmp1_hi, mask); 
         result2_lo= _mm512_and_si512(tmp2_lo, mask); 
         result2_hi= _mm512_and_si512(tmp2_hi, mask); 
         result3_lo= _mm512_and_si512(tmp3_lo, mask); 
         result3_hi= _mm512_and_si512(tmp3_hi, mask);  

          res0 = _mm512_packus_epi32(result0_lo, result0_hi);
          res1 = _mm512_packus_epi32(result1_lo, result1_hi);
          res2 = _mm512_packus_epi32(result2_lo, result2_hi);
          res3 = _mm512_packus_epi32(result3_lo, result3_hi);
          


           
        _mm512_storeu_si512((__m512i*)(dst_ptr0+dst_offset+j), res0);
        _mm512_storeu_si512((__m512i*)(dst_ptr1+dst_offset+j), res1);
        _mm512_storeu_si512((__m512i*)(dst_ptr2+dst_offset+j), res2);
        _mm512_storeu_si512((__m512i*)(dst_ptr3+dst_offset+j), res3);
        

        }

        for(; j < right; ++j) {
                mul_val = (spat_fil_accum_dtype) factors[0] * src_ptr0[src_offset+j];
                dst_val = (dwt2_dtype) ((mul_val + interim_rnd_factors[0]) >>
                                        interim_shift_factors[0]);
                dst_ptr0[dst_offset+j] = dst_val;

                mul_val = (spat_fil_accum_dtype) factors[1] * src_ptr1[src_offset+j];
                dst_val = (dwt2_dtype) ((mul_val + interim_rnd_factors[1]) >>
                                        interim_shift_factors[1]);
                dst_ptr1[dst_offset+j] = dst_val;

                
                mul_val = (spat_fil_accum_dtype) factors[2] * src_ptr2[src_offset+j];
                dst_val = (dwt2_dtype) ((mul_val + interim_rnd_factors[2]) >>
                                        interim_shift_factors[2]);
                dst_ptr2[dst_offset+j] = dst_val;

                
                mul_val = (spat_fil_accum_dtype) factors[3] * src_ptr3[src_offset+j];
                dst_val = (dwt2_dtype) ((mul_val + interim_rnd_factors[3]) >>
                                        interim_shift_factors[3]);
                dst_ptr3[dst_offset+j] = dst_val;
            }
        }
}