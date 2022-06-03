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

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
double funque_image_sad_c(const double *img1, const double *img2, int width, int height, int img1_stride, int img2_stride)
{
    double accum = (double)0.0;

    for (int i = 0; i < height; ++i) {
                double accum_line = (double)0.0;
        for (int j = 0; j < width; ++j) {
            double img1px = img1[i * img1_stride + j];
            double img2px = img2[i * img2_stride + j];

            accum_line += fabs(img1px - img2px);
        }
                accum += accum_line;
    }

    return (double) (accum / (width * height));
}

/**
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion_funque(const double *ref, const double *dis, int w, int h, int ref_stride, int dis_stride, double *score)
{

    if (ref_stride % sizeof(double) != 0)
    {
        printf("error: ref_stride %% sizeof(double) != 0, ref_stride = %d, sizeof(double) = %zu.\n", ref_stride, sizeof(double));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(double) != 0)
    {
        printf("error: dis_stride %% sizeof(double) != 0, dis_stride = %d, sizeof(double) = %zu.\n", dis_stride, sizeof(double));
        fflush(stdout);
        goto fail;
    }
    // stride for funque_image_sad_c is in terms of (sizeof(double) bytes)
    *score = funque_image_sad_c(ref, dis, w, h, ref_stride / sizeof(double), dis_stride / sizeof(double));

    return 0;

fail:
    return 1;
}
