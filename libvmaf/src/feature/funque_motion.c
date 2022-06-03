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
#include "funque_filters.h"
/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
funque_dtype funque_image_sad_c(const funque_dtype *img1, const funque_dtype *img2, int width, int height, int img1_stride, int img2_stride)
{
    funque_dtype accum = (funque_dtype)0.0;

    for (int i = 0; i < height; ++i) {
                funque_dtype accum_line = (funque_dtype)0.0;
        for (int j = 0; j < width; ++j) {
            funque_dtype img1px = img1[i * img1_stride + j];
            funque_dtype img2px = img2[i * img2_stride + j];

            accum_line += fabs(img1px - img2px);
        }
                accum += accum_line;
    }

    return (funque_dtype) (accum / (width * height));
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion_funque(const funque_dtype *ref, const funque_dtype *dis, int w, int h, int ref_stride, int dis_stride, double *score)
{

    if (ref_stride % sizeof(funque_dtype) != 0)
    {
        printf("error: ref_stride %% sizeof(funque_dtype) != 0, ref_stride = %d, sizeof(funque_dtype) = %zu.\n", ref_stride, sizeof(funque_dtype));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(funque_dtype) != 0)
    {
        printf("error: dis_stride %% sizeof(funque_dtype) != 0, dis_stride = %d, sizeof(funque_dtype) = %zu.\n", dis_stride, sizeof(funque_dtype));
        fflush(stdout);
        goto fail;
    }
    // stride for funque_image_sad_c is in terms of (sizeof(funque_dtype) bytes)
    *score = funque_image_sad_c(ref, dis, w, h, ref_stride / sizeof(funque_dtype), dis_stride / sizeof(funque_dtype));

    return 0;

fail:
    return 1;
}
