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
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>

#include "integer_funque_motion_avx2.h"

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
double integer_funque_image_mad_avx2(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor)
{
    motion_accum_dtype accum = 0;
    int width_32 = width - (width % 32);
    int width_16 = width - (width % 16);
    int width_8 = width - (width % 8);
    __m256i accum_256 = _mm256_setzero_si256();

    for (int i = 0; i < height; ++i) {
        int j = 0;
    	motion_interaccum_dtype accum_line = 0;
        __m256i accum_line_256 = _mm256_setzero_si256();

        for (; j < width_32; j+=32) {
            __m256i img1_x0 = _mm256_loadu_si256((__m256i*)(img1 + i * img1_stride + j));
            __m256i img2_x0 = _mm256_loadu_si256((__m256i*)(img2 + i * img1_stride + j));
            __m256i img1_x16 = _mm256_loadu_si256((__m256i*)(img1 + i * img1_stride + j + 16));
            __m256i img2_x16 = _mm256_loadu_si256((__m256i*)(img2 + i * img1_stride + j + 16));

            __m256i sub_x0 = _mm256_sub_epi16(img1_x0, img2_x0);
            __m256i sub_x16 = _mm256_sub_epi16(img1_x16, img2_x16);
            __m256i abs_x0 = _mm256_abs_epi16(sub_x0);
            __m256i abs_x16 = _mm256_abs_epi16(sub_x16);

            // 16 to 32 bits
            __m256i abs_x8 = _mm256_unpackhi_epi16(abs_x0, _mm256_setzero_si256());
            abs_x0 = _mm256_unpacklo_epi16(abs_x0, _mm256_setzero_si256());
            __m256i abs_x24 = _mm256_unpackhi_epi16(abs_x16, _mm256_setzero_si256());
            abs_x16 = _mm256_unpacklo_epi16(abs_x16, _mm256_setzero_si256());
            __m256i abs_sum0 = _mm256_add_epi32(abs_x0, abs_x8);
            __m256i abs_sum8 = _mm256_add_epi32(abs_x16, abs_x24);
            abs_sum0 = _mm256_add_epi32(abs_sum0, abs_sum8);
            accum_line_256 = _mm256_add_epi32(accum_line_256, abs_sum0);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }

        for (; j < width_16; j+=16) {
            __m256i img1_x0 = _mm256_loadu_si256((__m256i*)(img1 + i * img1_stride + j));
            __m256i img2_x0 = _mm256_loadu_si256((__m256i*)(img2 + i * img1_stride + j));

            __m256i sub_x0 = _mm256_sub_epi16(img1_x0, img2_x0);
            __m256i abs_x0 = _mm256_abs_epi16(sub_x0);

            // 16 to 32 bits
            __m256i abs_x8 = _mm256_unpackhi_epi16(abs_x0, _mm256_setzero_si256());
            abs_x0 = _mm256_unpacklo_epi16(abs_x0, _mm256_setzero_si256());
            __m256i abs_sum0 = _mm256_add_epi32(abs_x0, abs_x8);
            accum_line_256 = _mm256_add_epi32(accum_line_256, abs_sum0);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }

        for (; j < width_8; j+=8) {
            __m128i img1_x0 = _mm_loadu_si128((__m128i*)(img1 + i * img1_stride + j));
            __m128i img2_x0 = _mm_loadu_si128((__m128i*)(img2 + i * img1_stride + j));

            __m128i sub_x0 = _mm_sub_epi16(img1_x0, img2_x0);
            __m128i abs_x0 = _mm_abs_epi16(sub_x0);

            // 16 to 32 bits
            __m128i abs_x4 = _mm_unpackhi_epi16(abs_x0, _mm_setzero_si128());
            abs_x0 = _mm_unpacklo_epi16(abs_x0, _mm_setzero_si128());
            __m128i abs_sum0 = _mm_add_epi32(abs_x0, abs_x4);
            accum_line_256 = _mm256_add_epi32(accum_line_256, _mm256_castsi128_si256(abs_sum0));
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }

        // 0 1 2 3 x x x x 4 5 6 7 x x x x
        accum_line_256 = _mm256_hadd_epi32(accum_line_256, accum_line_256);
        for (; j < width; ++j) {
            dwt2_dtype img1px = img1[i * img1_stride + j];
            dwt2_dtype img2px = img2[i * img2_stride + j];

            accum_line += (motion_interaccum_dtype) abs(img1px - img2px);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (motion_accum_dtype) accum_line;
        // 0 1 2 3 4 5 6 7 -> 32bits
        accum_line_256 = _mm256_unpacklo_epi32(accum_line_256, _mm256_setzero_si256());
        accum_256 = _mm256_add_epi64(accum_256, accum_line_256);
        //assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }
    __m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(accum_256), _mm256_extracti128_si256(accum_256, 1));
    __m128i r2 = _mm_hadd_epi32(r4, r4);
    __m128i r1 = _mm_hadd_epi32(r2, r2);
    accum += _mm_extract_epi32(r1, 0);

    double d_accum = (double) accum / pending_div_factor;
    return (d_accum / (width * height));
}