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
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <arm_neon.h>

#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

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

void integer_integral_image_adm_sums_neon(i_dwt2buffers pyr_1, adm_u16_dtype *x, int k, int stride, i_adm_buffers masked_pyr, int width, int height, int band_index)
{
    adm_u16_dtype *x_pad;
    int i, j, index;
    adm_i32_dtype pyr_abs;

    int x_reflect = (int)((k - stride) / 2);

    x_pad = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));

    integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);

    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    size_t int_stride = r_width + 1;

    adm_i64_dtype *sum;
    adm_i64_dtype *temp_sum;
    sum = (adm_i64_dtype *)malloc((r_width + 1) * (r_height + 1) * sizeof(adm_i64_dtype));
    temp_sum = (adm_i64_dtype *)malloc((r_width + 1) * (r_height + 1) * sizeof(adm_i64_dtype));

    /*
    ** Setting the first row values to 0
    */
    memset(sum, 0, int_stride * sizeof(adm_i64_dtype));

    for (size_t i = 1; i < (k + 1); i++)
    {
        temp_sum[i * int_stride] = 0; // Setting the first column value to 0
        for (size_t j = 1; j < (k + 1); j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)];
        }
        for (size_t j = k + 1; j < int_stride; j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)] - x_pad[(i - 1) * r_width + j - k - 1];
        }
        for (size_t j = 1; j < int_stride; j++)
        {
            sum[i * int_stride + j] = temp_sum[i * int_stride + j] + sum[(i - 1) * int_stride + j];
        }
    }

    for (size_t i = (k + 1); i < (r_height + 1); i++)
    {
        temp_sum[i * int_stride] = 0; // Setting the first column value to 0
        for (size_t j = 1; j < (k + 1); j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)];
        }
        for (size_t j = k + 1; j < int_stride; j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)] - x_pad[(i - 1) * r_width + j - k - 1];
        }
        for (size_t j = 1; j < int_stride; j++)
        {
            sum[i * int_stride + j] = temp_sum[i * int_stride + j] + sum[(i - 1) * int_stride + j] - temp_sum[(i - k) * int_stride + j];
        }
    }
    /*
    ** For band 1 loop the pyr_1 value is multiplied by
    ** 30 to avaoid the precision loss that would happen
    ** due to the division by 30 of masking_threshold
    */
    if (band_index == 1)
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width;)
            {
                index = i * width + j;

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                int32x4_t mt2 = vld1q_s32(sum + ((i + k) * int_stride + j + k) + 4);
                uint16x8_t xt = vld1q_u16(x + index);
                uint32x4_t xt_m = vmovl_u16(vget_low_u16(xt));
                uint32x4_t xt_m_2 = vmovl_u16(vget_high_u16(xt));
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t masking_threshold_2 = vaddq_s32(vreinterpretq_s32_u32(xt_m_2), mt2);

                int16x4_t val_1_t = vld1_s16(pyr_1.bands[1] + index);
                int16x4_t val_2_t = vld1_s16(pyr_1.bands[2] + index);
                int16x4_t val_3_t = vld1_s16(pyr_1.bands[3] + index);
                int16x4_t val_1_t_2 = vld1_s16(pyr_1.bands[1] + index + 4);
                int16x4_t val_2_t_2 = vld1_s16(pyr_1.bands[2] + index + 4);
                int16x4_t val_3_t_2 = vld1_s16(pyr_1.bands[3] + index + 4);

                int16x4_t val_1abs = vabs_s16(val_1_t);
                int16x4_t val_2abs = vabs_s16(val_2_t);
                int16x4_t val_3abs = vabs_s16(val_3_t);
                int16x4_t val_1abs_2 = vabs_s16(val_1_t_2);
                int16x4_t val_2abs_2 = vabs_s16(val_2_t_2);
                int16x4_t val_3abs_2 = vabs_s16(val_3_t_2);

                int32x4_t mull1 = vmull_n_s16(val_1abs, 30);
                int32x4_t mull2 = vmull_n_s16(val_2abs, 30);
                int32x4_t mull3 = vmull_n_s16(val_3abs, 30);
                int32x4_t mull1_2 = vmull_n_s16(val_1abs_2, 30);
                int32x4_t mull2_2 = vmull_n_s16(val_2abs_2, 30);
                int32x4_t mull3_2 = vmull_n_s16(val_3abs_2, 30);

                int32x4_t sub_1 = vsubq_s32(mull1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(mull2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(mull3, masking_threshold);
                int32x4_t sub_1_2 = vsubq_s32(mull1_2, masking_threshold_2);
                int32x4_t sub_2_2 = vsubq_s32(mull2_2, masking_threshold_2);
                int32x4_t sub_3_2 = vsubq_s32(mull3_2, masking_threshold_2);

                vst1q_s32(masked_pyr.bands[1] + index, sub_1);
                vst1q_s32(masked_pyr.bands[2] + index, sub_2);
                vst1q_s32(masked_pyr.bands[3] + index, sub_3);
                vst1q_s32(masked_pyr.bands[1] + index + 4, sub_1_2);
                vst1q_s32(masked_pyr.bands[2] + index + 4, sub_2_2);
                vst1q_s32(masked_pyr.bands[3] + index + 4, sub_3_2);

                j += 8;
            }
        }
    }

    if (band_index == 2)
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width;)
            {
                index = i * width + j;

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                int32x4_t mt2 = vld1q_s32(sum + ((i + k) * int_stride + j + k) + 4);
                uint16x8_t xt = vld1q_u16(x + index);
                uint32x4_t xt_m = vmovl_u16(vget_low_u16(xt));
                uint32x4_t xt_m_2 = vmovl_u16(vget_high_u16(xt));
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t masking_threshold_2 = vaddq_s32(vreinterpretq_s32_u32(xt_m_2), mt2);

                int32x4_t val_1 = vld1q_s32(masked_pyr.bands[1] + index);
                int32x4_t val_2 = vld1q_s32(masked_pyr.bands[2] + index);
                int32x4_t val_3 = vld1q_s32(masked_pyr.bands[3] + index);
                int32x4_t val_1_2 = vld1q_s32(masked_pyr.bands[1] + index + 4);
                int32x4_t val_2_2 = vld1q_s32(masked_pyr.bands[2] + index + 4);
                int32x4_t val_3_2 = vld1q_s32(masked_pyr.bands[3] + index + 4);

                int32x4_t sub_1 = vsubq_s32(val_1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(val_2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(val_3, masking_threshold);
                int32x4_t sub_1_2 = vsubq_s32(val_1_2, masking_threshold_2);
                int32x4_t sub_2_2 = vsubq_s32(val_2_2, masking_threshold_2);
                int32x4_t sub_3_2 = vsubq_s32(val_3_2, masking_threshold_2);

                vst1q_s32(masked_pyr.bands[1] + index, sub_1);
                vst1q_s32(masked_pyr.bands[2] + index, sub_2);
                vst1q_s32(masked_pyr.bands[3] + index, sub_3);
                vst1q_s32(masked_pyr.bands[1] + index + 4, sub_1_2);
                vst1q_s32(masked_pyr.bands[2] + index + 4, sub_2_2);
                vst1q_s32(masked_pyr.bands[3] + index + 4, sub_3_2);

                j += 8;
            }
        }
    }
    /*
    ** For band 3 loop the final value is clipped
    ** to minimum of zero.
    */
    if (band_index == 3)
    {
        int32x4_t lower = vdupq_n_s32(0); // do this only once before the loops
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width;)
            {
                index = i * width + j;

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                int32x4_t mt2 = vld1q_s32(sum + ((i + k) * int_stride + j + k) + 4);
                uint16x8_t xt = vld1q_u16(x + index);
                uint32x4_t xt_m = vmovl_u16(vget_low_u16(xt));
                uint32x4_t xt_m_2 = vmovl_u16(vget_high_u16(xt));
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t masking_threshold_2 = vaddq_s32(vreinterpretq_s32_u32(xt_m_2), mt2);

                int32x4_t val_1 = vld1q_s32(masked_pyr.bands[1] + index);
                int32x4_t val_2 = vld1q_s32(masked_pyr.bands[2] + index);
                int32x4_t val_3 = vld1q_s32(masked_pyr.bands[3] + index);
                int32x4_t val_1_2 = vld1q_s32(masked_pyr.bands[1] + index + 4);
                int32x4_t val_2_2 = vld1q_s32(masked_pyr.bands[2] + index + 4);
                int32x4_t val_3_2 = vld1q_s32(masked_pyr.bands[3] + index + 4);

                int32x4_t sub_1 = vsubq_s32(val_1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(val_2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(val_3, masking_threshold);
                int32x4_t sub_1_2 = vsubq_s32(val_1_2, masking_threshold_2);
                int32x4_t sub_2_2 = vsubq_s32(val_2_2, masking_threshold_2);
                int32x4_t sub_3_2 = vsubq_s32(val_3_2, masking_threshold_2);

                int32x4_t x1 = vmaxq_s32(sub_1, lower);
                int32x4_t x2 = vmaxq_s32(sub_2, lower);
                int32x4_t x3 = vmaxq_s32(sub_3, lower);
                int32x4_t x1_2 = vmaxq_s32(sub_1_2, lower);
                int32x4_t x2_2 = vmaxq_s32(sub_2_2, lower);
                int32x4_t x3_2 = vmaxq_s32(sub_3_2, lower);

                vst1q_s32(masked_pyr.bands[1] + index, x1);
                vst1q_s32(masked_pyr.bands[2] + index, x2);
                vst1q_s32(masked_pyr.bands[3] + index, x3);
                vst1q_s32(masked_pyr.bands[1] + index + 4, x1_2);
                vst1q_s32(masked_pyr.bands[2] + index + 4, x2_2);
                vst1q_s32(masked_pyr.bands[3] + index + 4, x3_2);

                j += 8;
            }
        }
    }

    free(temp_sum);
    free(sum);
    free(x_pad);
}

