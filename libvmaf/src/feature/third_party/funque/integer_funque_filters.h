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
#ifndef FILTERS_FUNQUE_H_
#define FILTERS_FUNQUE_H_
#include <stddef.h>
#include <stdint.h>

#include "config.h"

#define SPAT_FILTER_COEFF_SHIFT 16
#define SPAT_FILTER_INTER_SHIFT  9
#define SPAT_FILTER_INTER_RND (1 << (SPAT_FILTER_INTER_SHIFT - 1))
#define SPAT_FILTER_OUT_SHIFT   16
#define SPAT_FILTER_OUT_RND (1 << (SPAT_FILTER_OUT_SHIFT - 1))
typedef int16_t spat_fil_coeff_dtype;
typedef int16_t spat_fil_inter_dtype;
typedef int32_t spat_fil_accum_dtype;
typedef int16_t spat_fil_output_dtype;

#define DWT2_COEFF_UPSHIFT 0
#define DWT2_INTER_SHIFT   0  //Shifting to make the intermediate have Q16 format
#define DWT2_OUT_SHIFT     1  //Shifting to make the output have Q16 format
typedef int16_t dwt2_dtype;
typedef int32_t dwt2_accum_dtype;
typedef int16_t dwt2_inter_dtype;

typedef int32_t motion_interaccum_dtype;
typedef int64_t motion_accum_dtype;

typedef int32_t ssim_inter_dtype;
typedef int64_t ssim_accum_dtype;
#define SSIM_SHIFT_DIV 15 //Depends on ssim_accum_dtype datatype
#define SSIM_INTER_VAR_SHIFTS 1
#define SSIM_INTER_L_SHIFT 1 //If this is updated, the usage has to be changed in integer_ssim.c(currently 2>>SSIM_INTER_L_SHIFT) is used for readability
#define SSIM_INTER_CS_SHIFT 1 //If this is updated, the usage has to be changed in integer_ssim.c(currently 2>>SSIM_INTER_CS_SHIFT) is used for readability

typedef struct i_dwt2buffers {
    dwt2_dtype *bands[4];
    int width;
    int height;
}i_dwt2buffers;

void integer_spatial_filter(uint8_t *src, spat_fil_output_dtype *dst, int width, int height);

void integer_funque_dwt2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height);

#endif /* FILTERS_FUNQUE_H_ */