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
#include <mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#include "integer_funque_strred_avx2.h"
#include "../integer_funque_strred.h"

void integer_subract_subbands_avx2(const dwt2_dtype *ref_src, const dwt2_dtype *ref_prev_src,
                                   dwt2_dtype *ref_dst, const dwt2_dtype *dist_src,
                                   const dwt2_dtype *dist_prev_src, dwt2_dtype *dist_dst, int width,
                                   int height)
{
    int i, j;
    // i , j , height , width changed to int from size_t
    for(i = 0; i < height; i++) {
        for(j = 0; j <= width - 16; j += 16) {
            __m256i ref_src_v = _mm256_loadu_si256((__m256i *) (ref_src + j + i * width));
            __m256i ref_prev_src_v = _mm256_loadu_si256((__m256i *) (ref_prev_src + j + i * width));
            __m256i dist_src_v = _mm256_loadu_si256((__m256i *) (dist_src + j + i * width));
            __m256i dist_prev_src_v =
                _mm256_loadu_si256((__m256i *) (dist_prev_src + j + i * width));

            __m256i ref_dst_final = _mm256_sub_epi16(ref_src_v, ref_prev_src_v);
            __m256i dist_dst_final = _mm256_sub_epi16(dist_src_v, dist_prev_src_v);

            _mm256_storeu_si256((__m256i *) (ref_dst + j + i * width), ref_dst_final);
            _mm256_storeu_si256((__m256i *) (dist_dst + j + i * width), dist_dst_final);
        }
        for(; j < width; j++) {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

float integer_rred_entropies_and_scales_avx2(const dwt2_dtype *x_t, const dwt2_dtype *y_t,
                                             size_t width, size_t height,
                                             uint32_t *log_lut, double sigma_nsq_arg,
                                             int32_t shift_val, uint8_t enable_temporal,
                                             float *spat_scales_x, float *spat_scales_y,
                                             uint8_t check_enable_spatial_csf)
{
    int kh = STRRED_WINDOW_SIZE;
    int kw = STRRED_WINDOW_SIZE;

    /* amount of reflecting */
    int x_reflect = (int) ((STRRED_WINDOW_SIZE - 1) / 2);
    int y_reflect = (int) ((STRRED_WINDOW_SIZE - 1) / 2);
    size_t strred_width, strred_height;

#if STRRED_REFLECT_PAD
    strred_width = width;
    strred_height = height;
#else
    strred_width = width - (2 * x_reflect);
    strred_height = height - (2 * x_reflect);
#endif

    size_t r_width = strred_width + (2 * x_reflect);
    size_t r_height = strred_height + (2 * x_reflect);
    dwt2_dtype *x_pad_t;
    dwt2_dtype *y_pad_t;

#if STRRED_REFLECT_PAD
    x_pad_t = (dwt2_dtype *) malloc(sizeof(dwt2_dtype *) * (strred_width + (2 * x_reflect)) *
                                    (strred_height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype *) malloc(sizeof(dwt2_dtype *) * (strred_width + (2 * y_reflect)) *
                                    (strred_height + (2 * y_reflect)));

    strred_integer_reflect_pad(x_t, strred_width, strred_height, x_reflect, x_pad_t);
    strred_integer_reflect_pad(y_t, strred_width, strred_height, y_reflect, y_pad_t);

#else
    x_pad_t = x_t;
    y_pad_t = y_t;
#endif

    float agg_abs_accum = 0;
    int16_t knorm_fact =
        25891;  // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21;
    uint32_t entr_const =
        (uint32_t) (log2f(2 * PI_CONSTANT * EULERS_CONSTANT) * TWO_POWER_Q_FACTOR);

    {
        int width_p1 = r_width + 1;
        int height_p1 = r_height + 1;

        int32_t *interim_1_x = (int32_t *) calloc(width_p1, sizeof(int32_t));
        int32_t *interim_1_y = (int32_t *) calloc(width_p1, sizeof(int32_t));
        int64_t *interim_2_x = (int64_t *) calloc(width_p1, sizeof(int64_t));
        int64_t *interim_2_y = (int64_t *) calloc(width_p1, sizeof(int64_t));

        int i = 0;
        int j = 0;

        // The height loop is broken into 2 parts,
        // 1st loop, prev kh row is not available to subtract during vertical summation
        for(i = 1; i < kh + 1; i++) {
            int src_offset = (i - 1) * r_width;
            for(j = 1; j <= width_p1 - 16; j += 16) {
                int j_minus1 = j - 1;
                __m256i src_x_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (x_pad_t + src_offset + j_minus1));
                __m256i src_x_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_val_16x16, 0));
                __m256i src_x_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_val_16x16, 1));

                __m256i src_y_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (y_pad_t + src_offset + j_minus1));
                __m256i src_y_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_val_16x16, 0));
                __m256i src_y_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_val_16x16, 1));

                __m256i src_xx_val_lo = _mm256_mullo_epi32(src_x_val_32x8_lo, src_x_val_32x8_lo);
                __m256i src_xx_val_hi = _mm256_mullo_epi32(src_x_val_32x8_hi, src_x_val_32x8_hi);
                __m256i src_yy_val_lo = _mm256_mullo_epi32(src_y_val_32x8_lo, src_y_val_32x8_lo);
                __m256i src_yy_val_hi = _mm256_mullo_epi32(src_y_val_32x8_hi, src_y_val_32x8_hi);

                __m256i interim_1_x_8val_lo = _mm256_loadu_si256((__m256i *) (interim_1_x + j));
                __m256i to_be_stored_interim_1_x_lo =
                    _mm256_add_epi32(interim_1_x_8val_lo, src_x_val_32x8_lo);
                _mm256_storeu_si256((__m256i *) (interim_1_x + j), to_be_stored_interim_1_x_lo);

                __m256i interim_1_x_8val_hi = _mm256_loadu_si256((__m256i *) (interim_1_x + j + 8));
                __m256i to_be_stored_interim_1_x_hi =
                    _mm256_add_epi32(interim_1_x_8val_hi, src_x_val_32x8_hi);
                _mm256_storeu_si256((__m256i *) (interim_1_x + j + 8), to_be_stored_interim_1_x_hi);

                __m256i interim_1_y_8val_lo = _mm256_loadu_si256((__m256i *) (interim_1_y + j));
                __m256i to_be_stored_interim_1_y_lo =
                    _mm256_add_epi32(interim_1_y_8val_lo, src_y_val_32x8_lo);
                _mm256_storeu_si256((__m256i *) (interim_1_y + j), to_be_stored_interim_1_y_lo);

                __m256i interim_1_y_8val_hi = _mm256_loadu_si256((__m256i *) (interim_1_y + j + 8));
                __m256i to_be_stored_interim_1_y_hi =
                    _mm256_add_epi32(interim_1_y_8val_hi, src_y_val_32x8_hi);
                _mm256_storeu_si256((__m256i *) (interim_1_y + j + 8), to_be_stored_interim_1_y_hi);

                __m256i interim_2_x_4val_lo_lo = _mm256_loadu_si256((__m256i *) (interim_2_x + j));
                __m256i to_be_stored_interim_2_x_lo_lo = _mm256_add_epi64(
                    interim_2_x_4val_lo_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_lo, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j), to_be_stored_interim_2_x_lo_lo);

                __m256i interim_2_x_4val_lo_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 4));
                __m256i to_be_stored_interim_2_x_lo_hi = _mm256_add_epi64(
                    interim_2_x_4val_lo_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_lo, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 4),
                                    to_be_stored_interim_2_x_lo_hi);

                __m256i interim_2_x_4val_hi_lo =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 8));
                __m256i to_be_stored_interim_2_x_hi_lo = _mm256_add_epi64(
                    interim_2_x_4val_hi_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_hi, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 8),
                                    to_be_stored_interim_2_x_hi_lo);

                __m256i interim_2_x_4val_hi_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 12));
                __m256i to_be_stored_interim_2_x_hi_hi = _mm256_add_epi64(
                    interim_2_x_4val_hi_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_hi, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 12),
                                    to_be_stored_interim_2_x_hi_hi);

                __m256i interim_2_y_4val_lo_lo = _mm256_loadu_si256((__m256i *) (interim_2_y + j));
                __m256i to_be_stored_interim_2_y_lo_lo = _mm256_add_epi64(
                    interim_2_y_4val_lo_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_lo, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j), to_be_stored_interim_2_y_lo_lo);

                __m256i interim_2_y_4val_lo_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 4));
                __m256i to_be_stored_interim_2_y_lo_hi = _mm256_add_epi64(
                    interim_2_y_4val_lo_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_lo, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 4),
                                    to_be_stored_interim_2_y_lo_hi);

                __m256i interim_2_y_4val_hi_lo =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 8));
                __m256i to_be_stored_interim_2_y_hi_lo = _mm256_add_epi64(
                    interim_2_y_4val_hi_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_hi, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 8),
                                    to_be_stored_interim_2_y_hi_lo);

                __m256i interim_2_y_4val_hi_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 12));
                __m256i to_be_stored_interim_2_y_hi_hi = _mm256_add_epi64(
                    interim_2_y_4val_hi_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_hi, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 12),
                                    to_be_stored_interim_2_y_hi_hi);
            }
            for(; j < width_p1; j++) {
                int j_minus1 = j - 1;
                int16_t src_x_val = x_pad_t[src_offset + j_minus1];
                int16_t src_y_val = y_pad_t[src_offset + j_minus1];
                int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t) src_y_val * src_y_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val;
            }
        }
        /**
         * The strred score calculations would start from the kh,kw index of var & covar
         * Hence horizontal sum of first kh rows are not used, hence that computation is avoided
         */
        // score computation for 1st row of variance & covariance i.e. kh row of padded img

        if(check_enable_spatial_csf == 1)
            agg_abs_accum += strred_horz_integralsum_spatial_csf(
                kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_lut,
                interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x,
                spat_scales_y, i - kh, shift_val);
        else
            agg_abs_accum += strred_horz_integralsum_wavelet(
                kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg, log_lut,
                interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal, spat_scales_x,
                spat_scales_y, i - kh, shift_val);

        // 2nd loop, core loop
        for(; i < height_p1; i++) {
            int src_offset = (i - 1) * r_width;
            int pre_kh_src_offset = (i - 1 - kh) * r_width;
            for(j = 1; j <= width_p1 - 16; j += 16) {
                int j_minus1 = j - 1;

                __m256i src_x_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (x_pad_t + src_offset + j_minus1));
                __m256i src_x_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_val_16x16, 0));
                __m256i src_x_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_val_16x16, 1));

                __m256i src_y_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (y_pad_t + src_offset + j_minus1));
                __m256i src_y_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_val_16x16, 0));
                __m256i src_y_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_val_16x16, 1));

                __m256i src_x_prekh_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (x_pad_t + pre_kh_src_offset + j_minus1));
                __m256i src_x_prekh_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_prekh_val_16x16, 0));
                __m256i src_x_prekh_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_x_prekh_val_16x16, 1));

                __m256i src_y_prekh_val_16x16 =
                    _mm256_loadu_si256((__m256i *) (y_pad_t + pre_kh_src_offset + j_minus1));
                __m256i src_y_prekh_val_32x8_lo =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_prekh_val_16x16, 0));
                __m256i src_y_prekh_val_32x8_hi =
                    _mm256_cvtepi16_epi32(_mm256_extractf128_si256(src_y_prekh_val_16x16, 1));

                __m256i src_xx_val_lo = _mm256_mullo_epi32(src_x_val_32x8_lo, src_x_val_32x8_lo);
                __m256i src_xx_val_hi = _mm256_mullo_epi32(src_x_val_32x8_hi, src_x_val_32x8_hi);
                __m256i src_yy_val_lo = _mm256_mullo_epi32(src_y_val_32x8_lo, src_y_val_32x8_lo);
                __m256i src_yy_val_hi = _mm256_mullo_epi32(src_y_val_32x8_hi, src_y_val_32x8_hi);

                __m256i src_xx_prekh_val_lo =
                    _mm256_mullo_epi32(src_x_prekh_val_32x8_lo, src_x_prekh_val_32x8_lo);
                __m256i src_xx_prekh_val_hi =
                    _mm256_mullo_epi32(src_x_prekh_val_32x8_hi, src_x_prekh_val_32x8_hi);
                __m256i src_yy_prekh_val_lo =
                    _mm256_mullo_epi32(src_y_prekh_val_32x8_lo, src_y_prekh_val_32x8_lo);
                __m256i src_yy_prekh_val_hi =
                    _mm256_mullo_epi32(src_y_prekh_val_32x8_hi, src_y_prekh_val_32x8_hi);

                __m256i src_x_val_32x8_lo_minus =
                    _mm256_sub_epi32(src_x_val_32x8_lo, src_x_prekh_val_32x8_lo);
                __m256i src_x_val_32x8_hi_minus =
                    _mm256_sub_epi32(src_x_val_32x8_hi, src_x_prekh_val_32x8_hi);
                __m256i src_y_val_32x8_lo_minus =
                    _mm256_sub_epi32(src_y_val_32x8_lo, src_y_prekh_val_32x8_lo);
                __m256i src_y_val_32x8_hi_minus =
                    _mm256_sub_epi32(src_y_val_32x8_hi, src_y_prekh_val_32x8_hi);

                __m256i src_xx_val_lo_minus = _mm256_sub_epi32(src_xx_val_lo, src_xx_prekh_val_lo);
                __m256i src_xx_val_hi_minus = _mm256_sub_epi32(src_xx_val_hi, src_xx_prekh_val_hi);
                __m256i src_yy_val_lo_minus = _mm256_sub_epi32(src_yy_val_lo, src_yy_prekh_val_lo);
                __m256i src_yy_val_hi_minus = _mm256_sub_epi32(src_yy_val_hi, src_yy_prekh_val_hi);

                __m256i interim_1_x_8val_lo = _mm256_loadu_si256((__m256i *) (interim_1_x + j));
                __m256i to_be_stored_interim_1_x_lo =
                    _mm256_add_epi32(interim_1_x_8val_lo, src_x_val_32x8_lo_minus);
                _mm256_storeu_si256((__m256i *) (interim_1_x + j), to_be_stored_interim_1_x_lo);

                __m256i interim_1_x_8val_hi = _mm256_loadu_si256((__m256i *) (interim_1_x + j + 8));
                __m256i to_be_stored_interim_1_x_hi =
                    _mm256_add_epi32(interim_1_x_8val_hi, src_x_val_32x8_hi_minus);
                _mm256_storeu_si256((__m256i *) (interim_1_x + j + 8), to_be_stored_interim_1_x_hi);

                __m256i interim_1_y_8val_lo = _mm256_loadu_si256((__m256i *) (interim_1_y + j));
                __m256i to_be_stored_interim_1_y_lo =
                    _mm256_add_epi32(interim_1_y_8val_lo, src_y_val_32x8_lo_minus);
                _mm256_storeu_si256((__m256i *) (interim_1_y + j), to_be_stored_interim_1_y_lo);

                __m256i interim_1_y_8val_hi = _mm256_loadu_si256((__m256i *) (interim_1_y + j + 8));
                __m256i to_be_stored_interim_1_y_hi =
                    _mm256_add_epi32(interim_1_y_8val_hi, src_y_val_32x8_hi_minus);
                _mm256_storeu_si256((__m256i *) (interim_1_y + j + 8), to_be_stored_interim_1_y_hi);

                __m256i interim_2_x_4val_lo_lo = _mm256_loadu_si256((__m256i *) (interim_2_x + j));
                __m256i to_be_stored_interim_2_x_lo_lo = _mm256_add_epi64(
                    interim_2_x_4val_lo_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_lo_minus, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j), to_be_stored_interim_2_x_lo_lo);

                __m256i interim_2_x_4val_lo_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 4));
                __m256i to_be_stored_interim_2_x_lo_hi = _mm256_add_epi64(
                    interim_2_x_4val_lo_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_lo_minus, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 4),
                                    to_be_stored_interim_2_x_lo_hi);

                __m256i interim_2_x_4val_hi_lo =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 8));
                __m256i to_be_stored_interim_2_x_hi_lo = _mm256_add_epi64(
                    interim_2_x_4val_hi_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_hi_minus, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 8),
                                    to_be_stored_interim_2_x_hi_lo);

                __m256i interim_2_x_4val_hi_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_x + j + 12));
                __m256i to_be_stored_interim_2_x_hi_hi = _mm256_add_epi64(
                    interim_2_x_4val_hi_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_xx_val_hi_minus, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_x + j + 12),
                                    to_be_stored_interim_2_x_hi_hi);

                __m256i interim_2_y_4val_lo_lo = _mm256_loadu_si256((__m256i *) (interim_2_y + j));
                __m256i to_be_stored_interim_2_y_lo_lo = _mm256_add_epi64(
                    interim_2_y_4val_lo_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_lo_minus, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j), to_be_stored_interim_2_y_lo_lo);

                __m256i interim_2_y_4val_lo_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 4));
                __m256i to_be_stored_interim_2_y_lo_hi = _mm256_add_epi64(
                    interim_2_y_4val_lo_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_lo_minus, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 4),
                                    to_be_stored_interim_2_y_lo_hi);

                __m256i interim_2_y_4val_hi_lo =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 8));
                __m256i to_be_stored_interim_2_y_hi_lo = _mm256_add_epi64(
                    interim_2_y_4val_hi_lo,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_hi_minus, 0)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 8),
                                    to_be_stored_interim_2_y_hi_lo);

                __m256i interim_2_y_4val_hi_hi =
                    _mm256_loadu_si256((__m256i *) (interim_2_y + j + 12));
                __m256i to_be_stored_interim_2_y_hi_hi = _mm256_add_epi64(
                    interim_2_y_4val_hi_hi,
                    _mm256_cvtepi32_epi64(_mm256_extractf128_si256(src_yy_val_hi_minus, 1)));
                _mm256_storeu_si256((__m256i *) (interim_2_y + j + 12),
                                    to_be_stored_interim_2_y_hi_hi);
            }
            for(; j < width_p1; j++) {
                int j_minus1 = j - 1;
                int16_t src_x_val = x_pad_t[src_offset + j_minus1];
                int16_t src_y_val = y_pad_t[src_offset + j_minus1];

                int16_t src_x_prekh_val = x_pad_t[pre_kh_src_offset + j_minus1];
                int16_t src_y_prekh_val = y_pad_t[pre_kh_src_offset + j_minus1];
                int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
                int32_t src_xx_prekh_val = (int32_t) src_x_prekh_val * src_x_prekh_val;
                int32_t src_yy_prekh_val = (int32_t) src_y_prekh_val * src_y_prekh_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val - src_x_prekh_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val - src_y_prekh_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val - src_xx_prekh_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val - src_yy_prekh_val;
            }

            // horizontal summation and score compuations
            if(check_enable_spatial_csf == 1)
                agg_abs_accum += strred_horz_integralsum_spatial_csf(
                    kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg,
                    log_lut, interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal,
                    spat_scales_x, spat_scales_y, (i - kh) * width_p1, shift_val);
            else
                agg_abs_accum += strred_horz_integralsum_wavelet(
                    kw, width_p1, knorm_fact, knorm_shift, entr_const, sigma_nsq_arg,
                    log_lut, interim_1_x, interim_2_x, interim_1_y, interim_2_y, enable_temporal,
                    spat_scales_x, spat_scales_y, (i - kh) * width_p1, shift_val);
        }

        free(interim_1_x);
        free(interim_1_y);
        free(interim_2_x);
        free(interim_2_y);
    }

#if STRRED_REFLECT_PAD
    free(x_pad_t);
    free(y_pad_t);
#endif
    return agg_abs_accum;
}

int integer_compute_srred_funque_avx2(const struct i_dwt2buffers *ref,
                                      const struct i_dwt2buffers *dist, size_t width, size_t height,
                                      float **spat_scales_ref, float **spat_scales_dist,
                                      struct strred_results *strred_scores, int block_size, int level,
                                      uint32_t *log_lut, int32_t shift_val_arg,
                                      double sigma_nsq_t, uint8_t check_enable_spatial_csf)
{
    int ret;
    UNUSED(block_size);
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband;
    float spat_values[DEFAULT_STRRED_SUBBANDS], fspat_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    for(subband = 1; subband < total_subbands; subband++) {
        enable_temp = 0;
        spat_values[subband] = 0;

        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else {
            shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
        }
        spat_values[subband] = integer_rred_entropies_and_scales_avx2(
            ref->bands[subband], dist->bands[subband], width, height, log_lut, sigma_nsq_t,
            shift_val, enable_temp, spat_scales_ref[subband], spat_scales_dist[subband],
            check_enable_spatial_csf);
        fspat_val[subband] = spat_values[subband] / (width * height);
    }

    strred_scores->spat_vals[level] = (fspat_val[1] + fspat_val[2] + fspat_val[3]) / 3;

    // Add equations to compute S-RRED using norm factors
    int norm_factor = 1, num_level;
    for(num_level = 0; num_level <= level; num_level++) norm_factor = num_level + 1;

    strred_scores->spat_vals_cumsum += strred_scores->spat_vals[level];

    strred_scores->srred_vals[level] = strred_scores->spat_vals_cumsum / norm_factor;

    ret = 0;
    return ret;
}

int integer_compute_strred_funque_avx2(const struct i_dwt2buffers *ref,
                                       const struct i_dwt2buffers *dist,
                                       struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist,
                                       size_t width, size_t height, float **spat_scales_ref,
                                       float **spat_scales_dist, struct strred_results *strred_scores,
                                       int block_size, int level, uint32_t *log_lut,
                                       int32_t shift_val_arg, double sigma_nsq_t,
                                       uint8_t check_enable_spatial_csf)
{
    int ret;
    UNUSED(block_size);
    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t subband;
    float temp_values[DEFAULT_STRRED_SUBBANDS], ftemp_val[DEFAULT_STRRED_SUBBANDS];
    uint8_t enable_temp = 0;
    int32_t shift_val;

    for(subband = 1; subband < total_subbands; subband++) {
        if(check_enable_spatial_csf == 1)
            shift_val = 2 * shift_val_arg;
        else {
            shift_val = 2 * i_nadenau_pending_div_factors[level][subband];
        }

        if(prev_ref != NULL && prev_dist != NULL) {
            enable_temp = 1;
            dwt2_dtype *ref_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            dwt2_dtype *dist_temporal = (dwt2_dtype *) calloc(width * height, sizeof(dwt2_dtype));
            temp_values[subband] = 0;

            integer_subract_subbands_avx2(ref->bands[subband], prev_ref->bands[subband], ref_temporal,
                                       dist->bands[subband], prev_dist->bands[subband],
                                       dist_temporal, width, height);
            temp_values[subband] = integer_rred_entropies_and_scales_avx2(
                ref_temporal, dist_temporal, width, height, log_lut, sigma_nsq_t, shift_val,
                enable_temp, spat_scales_ref[subband], spat_scales_dist[subband],
                check_enable_spatial_csf);
            ftemp_val[subband] = temp_values[subband] / (width * height);

            free(ref_temporal);
            free(dist_temporal);
        }
    }
    strred_scores->temp_vals[level] = (ftemp_val[1] + ftemp_val[2] + ftemp_val[3]) / 3;
    strred_scores->spat_temp_vals[level] =
        strred_scores->spat_vals[level] * strred_scores->temp_vals[level];

    // Add equations to compute ST-RRED using norm factors
    int norm_factor = 1, num_level;
    for(num_level = 0; num_level <= level; num_level++) norm_factor = num_level + 1;

    strred_scores->temp_vals_cumsum += strred_scores->temp_vals[level];
    strred_scores->spat_temp_vals_cumsum += strred_scores->spat_temp_vals[level];

    strred_scores->trred_vals[level] = strred_scores->temp_vals_cumsum / norm_factor;
    strred_scores->strred_vals[level] = strred_scores->spat_temp_vals_cumsum / norm_factor;

    ret = 0;
    return ret;
}