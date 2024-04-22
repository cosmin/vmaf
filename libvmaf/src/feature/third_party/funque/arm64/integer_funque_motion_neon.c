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

#include <arm_neon.h>
#include <stdlib.h>
#include "../integer_funque_filters.h"
#include "../integer_funque_motion.h"

double integer_funque_image_mad_neon(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor)
{
    int img1_stride2 = (img1_stride << 1);
    int img2_stride2 = (img2_stride << 1);

    int16x8_t img16x8_1, img16x8_2, img16x8_3, img16x8_4;
    int16x8_t img16x8_5, img16x8_6, img16x8_7, img16x8_8;
    int64x2_t res64x2_1, res64x2_2, res64x2_3, res64x2_4;
    int64x2_t dupZero = vdupq_n_s64(0);
    res64x2_1 = dupZero;
    res64x2_2 = dupZero;
    res64x2_3 = dupZero;
    res64x2_4 = dupZero;

    motion_accum_dtype accum = 0, accumOddWidth = 0;
    int heightDiv2 = (height >> 1);
    int lastRow = (height & 1);
    int i, j;

    for (i = 0; i < heightDiv2; i++)
    {
        for (j = 0; j <= width - 16; j += 16)
        {
            img16x8_1 = vld1q_s16(img1 + j);
            img16x8_2 = vld1q_s16(img2 + j);
            img16x8_3 = vld1q_s16(img1 + j + 8);
            img16x8_4 = vld1q_s16(img2 + j + 8);
            img16x8_5 = vld1q_s16(img1 + j + img1_stride);
            img16x8_6 = vld1q_s16(img2 + j + img2_stride);
            img16x8_7 = vld1q_s16(img1 + j + img1_stride + 8);
            img16x8_8 = vld1q_s16(img2 + j + img2_stride + 8);

            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_1), vget_low_s16(img16x8_2)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_3), vget_low_s16(img16x8_4)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_1, img16x8_2));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_3, img16x8_4));
            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_5), vget_low_s16(img16x8_6)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_7), vget_low_s16(img16x8_8)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_5, img16x8_6));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_7, img16x8_8));
        }
        for (; j < width; j++)
        {
            dwt2_dtype img1px = img1[j];
            dwt2_dtype img2px = img2[j];
            dwt2_dtype img3px = img1[j + img1_stride];
            dwt2_dtype img4px = img2[j + img2_stride];
            accumOddWidth += (motion_interaccum_dtype)abs(img1px - img2px) + (motion_interaccum_dtype)abs(img3px - img4px);
        }
        img1 += img1_stride2;
        img2 += img2_stride2;
    }

    if (lastRow)
    {
        for (j = 0; j <= width - 16; j += 16)
        {
            img16x8_1 = vld1q_s16(img1 + j);
            img16x8_2 = vld1q_s16(img2 + j);
            img16x8_3 = vld1q_s16(img1 + j + 8);
            img16x8_4 = vld1q_s16(img2 + j + 8);

            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_1), vget_low_s16(img16x8_2)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_3), vget_low_s16(img16x8_4)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_1, img16x8_2));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_3, img16x8_4));
        }
        for (; j < width; ++j)
        {
            dwt2_dtype img1px = img1[j];
            dwt2_dtype img2px = img2[j];
            accumOddWidth += (motion_interaccum_dtype)abs(img1px - img2px);
        }
    }

    accum = vaddvq_s64(res64x2_1) + vaddvq_s64(res64x2_2) + vaddvq_s64(res64x2_3) + vaddvq_s64(res64x2_4) + accumOddWidth;
    double d_accum = (double)accum / pending_div_factor;
    return (d_accum / (width * height));
}

int integer_compute_motion_funque_neon(const dwt2_dtype *prev, const dwt2_dtype *curr, int w, int h, int prev_stride, int curr_stride, int pending_div_factor_arg, double *score)
{
    float pending_div_factor = (1 << pending_div_factor_arg) * 255;

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

    *score = integer_funque_image_mad_neon(prev, curr, w, h, prev_stride / sizeof(dwt2_dtype), curr_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}

int integer_compute_mad_funque_neon(const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, int pending_div_factor_arg, double *score)
{

    float pending_div_factor = (1 << pending_div_factor_arg) * 255;

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

    *score = integer_funque_image_mad_neon(ref, dis, w, h, ref_stride / sizeof(dwt2_dtype), dis_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}