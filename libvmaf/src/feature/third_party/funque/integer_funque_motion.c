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

#include "integer_funque_motion.h"

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
double integer_funque_image_mad_c(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, int pending_div_factor)
{
    motion_accum_dtype accum = 0;

    for (int i = 0; i < height; ++i) {
        motion_interaccum_dtype accum_line = 0;
        for (int j = 0; j < width; ++j) {
            dwt2_dtype img1px = img1[i * img1_stride + j];
            dwt2_dtype img2px = img2[i * img2_stride + j];

            accum_line += (motion_interaccum_dtype) abs(img1px - img2px);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (motion_accum_dtype) accum_line;
        //assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }

    double d_accum = (double) accum / pending_div_factor;
    return (d_accum / (width * height));
}

/**
 * Note: prev_stride and curr_stride are in terms of bytes
 */

int integer_compute_motion_funque_c(const dwt2_dtype *prev, const dwt2_dtype *curr, int w, int h, int prev_stride, int curr_stride, int pending_div_factor_arg, double *score)
{
    int pending_div_factor = (1 << pending_div_factor_arg) * 255;

    if (prev_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: prev_stride %% sizeof(dwt2_dtype) != 0, prev_stride = %d, sizeof(dwt2_dtype) = %zu.\n", prev_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    if (curr_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: curr_stride %% sizeof(dwt2_dtype) != 0, curr_stride = %d, sizeof(dwt2_dtype) = %zu.\n", curr_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    // stride for integer_funque_image_mad_c is in terms of (sizeof(dwt2_dtype) bytes)

    *score = integer_funque_image_mad_c(prev, curr, w, h, prev_stride / sizeof(dwt2_dtype), curr_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}

int integer_compute_mad_funque_c(const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, int pending_div_factor_arg, double *score)
{
    int pending_div_factor = (1 << pending_div_factor_arg) * 255;

    if (ref_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: ref_stride %% sizeof(dwt2_dtype) != 0, ref_stride = %d, sizeof(dwt2_dtype) = %zu.\n", ref_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: dis_stride %% sizeof(dwt2_dtype) != 0, dis_stride = %d, sizeof(dwt2_dtype) = %zu.\n", dis_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    // stride for integer_funque_image_mad_c is in terms of (sizeof(dwt2_dtype) bytes)

    *score = integer_funque_image_mad_c(ref, dis, w, h, ref_stride / sizeof(dwt2_dtype), dis_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}
