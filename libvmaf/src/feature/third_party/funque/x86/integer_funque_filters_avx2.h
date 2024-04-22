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
#include "../integer_funque_filters.h"
#include <immintrin.h>

#define hor_sum_and_store(addr, r) \
{ \
    __m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1)); \
    __m128i r2 = _mm_hadd_epi32(r4, r4); \
    __m128i r1 = _mm_hadd_epi32(r2, r2); \
    int r = _mm_cvtsi128_si32(r1); \
    dst[dst_row_idx + j] = (spat_fil_output_dtype) ((r + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT); \
}

#define shuffle_and_store(addr, v0, v8) \
{ \
    __m256i r0 = _mm256_permute2x128_si256(v0, v8, 0x20); \
    __m256i r8 = _mm256_permute2x128_si256(v0, v8, 0x31); \
    _mm256_store_si256((__m256i*)(addr), r0); \
    _mm256_store_si256((__m256i*)(addr + 16), r8); \
}

void integer_funque_dwt2_avx2(spat_fil_output_dtype *src, ptrdiff_t src_stride,
                              i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height,
                              int spatial_csf_flag, int level);

void integer_funque_vifdwt2_band0_avx2(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride,
                                       int width, int height);

void integer_spatial_filter_avx2(void *src, spat_fil_output_dtype *dst, int dst_stride, int width,
                                 int height, int bitdepth, spat_fil_inter_dtype *tmp, int num_taps);

void integer_spatial_5tap_filter_avx2(void *src, spat_fil_output_dtype *dst, int dst_stride,
                                      int width, int height, int bitdepth,
                                      spat_fil_inter_dtype *tmp, char *spatial_csf_filter);

void integer_funque_dwt2_inplace_csf_avx2(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4],
                                          int min_theta, int max_theta,
                                          uint16_t interim_rnd_factors[4],
                                          uint8_t interim_shift_factors[4], int level);