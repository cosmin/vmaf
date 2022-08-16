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
#ifndef FEATURE_ADM_H_
#define FEATURE_ADM_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "integer_funque_filters.h"

typedef int16_t adm_i16_dtype;
typedef int32_t adm_i32_dtype;
typedef uint16_t adm_u16_dtype;
typedef int64_t adm_i64_dtype;

typedef struct i_adm_buffers
{
  adm_i32_dtype *bands[4];
  int width;
  int height;
} i_adm_buffers;

typedef struct u_adm_buffers
{
    adm_u16_dtype *bands[4];
    int width;
    int height;
} u_adm_buffers;

#define ADM_CUBE_SHIFT 8
#define ADM_CUBE_SHIFT_ROUND (1 << (ADM_CUBE_SHIFT - 1))

#define K_INTEGRALIMG_ADM 3
#define ADM_CUBE_DIV pow(2,ADM_CUBE_SHIFT)

/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

#define SHIFT_ADM_DECOUPLE_FINAL 16

#define EXTRA_SAMPLE_BORDER 1

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif
#define COS_1DEG_SQ cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0)

static inline int clip(int value, int low, int high);
void integer_reflect_pad_adm(const adm_u16_dtype *src, size_t width, size_t height, int reflect, adm_u16_dtype *dest);
int integer_compute_adm_funque(ModuleFunqueState m, i_dwt2buffers ref, i_dwt2buffers dist, double *adm_score, 
                               double *adm_score_num, double *adm_score_den, size_t width, size_t height, 
                               float border_size, int16_t shift_val, int32_t* adm_div_lookup);
void integer_dlm_decouple_c(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den);

void div_lookup_generator(int32_t* adm_div_lookup);
#endif /* _FEATURE_ADM_H_ */