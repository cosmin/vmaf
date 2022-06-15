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

#include "config.h"

#if FUNQUE_DOUBLE_DTYPE
typedef double funque_dtype;
#else
typedef float funque_dtype;
#endif
typedef struct dwt2buffers {
    funque_dtype *bands[4];
    int width;
    int height;
}dwt2buffers;

void spatial_filter(funque_dtype *src, funque_dtype *dst, ptrdiff_t dst_stride, int width, int height);

void funque_dwt2(funque_dtype *src, dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);

void normalize_bitdepth(funque_dtype *src, funque_dtype *dst, int scaler, ptrdiff_t dst_stride, int width, int height);

#endif /* FILTERS_FUNQUE_H_ */