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

void integer_dlm_decouple_neon(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, u_adm_buffers i_dlm_add, int32_t *adm_div_lookup)
{
  size_t width = ref.width;
  size_t height = ref.height;
  int i, j, k, l, index;

  int angle_flagC;
  adm_i16_dtype tmp_val;
  uint16_t angle_flag[8];

  adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
  int16x8_t src16x8_rH, src16x8_rV, src16x8_rD, src16x8_dH, src16x8_dV, src16x8_dD;
  int16x4_t src16x8_rH_lo, src16x8_rV_lo, src16x8_rD_lo, src16x8_dH_lo, src16x8_dV_lo;
  int32x4_t mul32x4_rdH0, mul32x4_rdH1, mul32x4_rrH0, mul32x4_rrH1, mul32x4_ddH0, mul32x4_ddH1;
  int64x2_t otdp_64x2_rdH00, otdp_64x2_rdH01, otdp_64x2_rdH10, otdp_64x2_rdH11;
  int64x2_t otmag_64x2_rrH00, otmag_64x2_rrH01, otmag_64x2_rrH10, otmag_64x2_rrH11;
  uint32x4_t chkZero32x4_rHlo, chkZero32x4_rHhi, chkZero32x4_rVlo, chkZero32x4_rVhi, chkZero32x4_rDlo, chkZero32x4_rDhi;
  int32x4_t src32x4_b1lo, src32x4_b1hi, src32x4_b2lo, src32x4_b2hi, src32x4_b3lo, src32x4_b3hi;
  int32x4_t src32x4_dH_lo, src32x4_dH_hi, src32x4_dV_lo, src32x4_dV_hi, src32x4_dD_lo, src32x4_dD_hi;
  int64x2_t mul64x2_H0, mul64x2_H1, mul64x2_H2, mul64x2_H3;
  int64x2_t mul64x2_V0, mul64x2_V1, mul64x2_V2, mul64x2_V3;
  int64x2_t mul64x2_D0, mul64x2_D1, mul64x2_D2, mul64x2_D3;
  int32x4_t sft32x4_Hlo, sft32x4_Hhi, sft32x4_Vlo, sft32x4_Vhi, sft32x4_Dlo, sft32x4_Dhi;
  int32x4_t tmp_Hlo, tmp_Hhi, tmp_Vlo, tmp_Vhi, tmp_Dlo, tmp_Dhi;
  int32x4_t tmpVal_Hlo, tmpVal_Hhi, tmpVal_Vlo, tmpVal_Vhi, tmpVal_Dlo, tmpVal_Dhi;
  uint16x4_t kh16x4_Hlo, kh16x4_Hhi, kh16x4_Vlo, kh16x4_Vhi, kh16x4_Dlo, kh16x4_Dhi;
  int16x8_t sft16x8_H, sft16x8_V, sft16x8_D, dlmRest_H, dlmRest_V, dlmRest_D, admAdd_H, admAdd_V, admAdd_D;
  uint16x8_t angFlagBuf, angBuf0, angBuf1;
  int16x8_t sftModH, srcModH, sftModV, srcModV, sftModD, srcModD;

  int32x4_t dupVal0 = vdupq_n_s32(32768);
  uint16x4_t dupVal1 = vdup_n_u16(32768);
  uint16x8_t dupConst1 = vdupq_n_u16(1);
  int32x4_t dupConstZero = vdupq_n_s32(0);

  int32_t *buf1_adm_div = (int32_t *)malloc(8 * sizeof(int32_t *));
  int32_t *buf2_adm_div = (int32_t *)malloc(8 * sizeof(int32_t *));
  int32_t *buf3_adm_div = (int32_t *)malloc(8 * sizeof(int32_t *));
  int32_t *buf_ot_dp = (int32_t *)malloc(8 * sizeof(int32_t *));
  int64_t *buf_ot_dp_sq = (int64_t *)malloc(8 * sizeof(int64_t *));
  int64_t *buf_ot_mag = (int64_t *)malloc(8 * sizeof(int64_t *));

  dwt2_dtype *refBandH = ref.bands[1];
  dwt2_dtype *refBandV = ref.bands[2];
  dwt2_dtype *refBandD = ref.bands[3];

  dwt2_dtype *distBandH = dist.bands[1];
  dwt2_dtype *distBandV = dist.bands[2];
  dwt2_dtype *distBandD = dist.bands[3];

  for (i = 0; i < height; i++)
  {
    for (j = 0; j <= width - 8; j += 8)
    {
      index = i * width + j;

      src16x8_rH = vld1q_s16(refBandH + index);
      src16x8_rV = vld1q_s16(refBandV + index);
      src16x8_dH = vld1q_s16(distBandH + index);
      src16x8_dV = vld1q_s16(distBandV + index);

      src16x8_rH_lo = vget_low_s16(src16x8_rH);
      src16x8_rV_lo = vget_low_s16(src16x8_rV);
      src16x8_dH_lo = vget_low_s16(src16x8_dH);
      src16x8_dV_lo = vget_low_s16(src16x8_dV);

      mul32x4_rdH0 = vmull_s16(src16x8_rH_lo, src16x8_dH_lo);
      mul32x4_rdH0 = vmlal_s16(mul32x4_rdH0, src16x8_rV_lo, src16x8_dV_lo);
      mul32x4_rdH1 = vmull_high_s16(src16x8_rH, src16x8_dH);
      mul32x4_rdH1 = vmlal_high_s16(mul32x4_rdH1, src16x8_rV, src16x8_dV);

      mul32x4_rrH0 = vmull_s16(src16x8_rH_lo, src16x8_rH_lo);
      mul32x4_rrH0 = vmlal_s16(mul32x4_rrH0, src16x8_rV_lo, src16x8_rV_lo);
      mul32x4_rrH1 = vmull_high_s16(src16x8_rH, src16x8_rH);
      mul32x4_rrH1 = vmlal_high_s16(mul32x4_rrH1, src16x8_rV, src16x8_rV);

      mul32x4_ddH0 = vmull_s16(src16x8_dH_lo, src16x8_dH_lo);
      mul32x4_ddH0 = vmlal_s16(mul32x4_ddH0, src16x8_dV_lo, src16x8_dV_lo);
      mul32x4_ddH1 = vmull_high_s16(src16x8_dH, src16x8_dH);
      mul32x4_ddH1 = vmlal_high_s16(mul32x4_ddH1, src16x8_dV, src16x8_dV);

      otdp_64x2_rdH00 = vmull_s32(vget_low_s32(mul32x4_rdH0), vget_low_s32(mul32x4_rdH0));
      otdp_64x2_rdH01 = vmull_high_s32(mul32x4_rdH0, mul32x4_rdH0);
      otdp_64x2_rdH10 = vmull_s32(vget_low_s32(mul32x4_rdH1), vget_low_s32(mul32x4_rdH1));
      otdp_64x2_rdH11 = vmull_high_s32(mul32x4_rdH1, mul32x4_rdH1);

      otmag_64x2_rrH00 = vmull_s32(vget_low_s32(mul32x4_rrH0), vget_low_s32(mul32x4_ddH0));
      otmag_64x2_rrH01 = vmull_high_s32(mul32x4_rrH0, mul32x4_ddH0);
      otmag_64x2_rrH10 = vmull_s32(vget_low_s32(mul32x4_rrH1), vget_low_s32(mul32x4_ddH1));
      otmag_64x2_rrH11 = vmull_high_s32(mul32x4_rrH1, mul32x4_ddH1);

      vst1q_s32(buf_ot_dp, mul32x4_rdH0);
      vst1q_s32(buf_ot_dp + 4, mul32x4_rdH1);

      vst1q_s64(buf_ot_dp_sq, otdp_64x2_rdH00);
      vst1q_s64(buf_ot_dp_sq + 2, otdp_64x2_rdH01);
      vst1q_s64(buf_ot_dp_sq + 4, otdp_64x2_rdH10);
      vst1q_s64(buf_ot_dp_sq + 6, otdp_64x2_rdH11);

      vst1q_s64(buf_ot_mag, otmag_64x2_rrH00);
      vst1q_s64(buf_ot_mag + 2, otmag_64x2_rrH01);
      vst1q_s64(buf_ot_mag + 4, otmag_64x2_rrH10);
      vst1q_s64(buf_ot_mag + 6, otmag_64x2_rrH11);

      for (l = 0; l < 8; l++)
      {
        angle_flag[l] = (uint16_t)((buf_ot_dp[l] >= 0) && (buf_ot_dp_sq[l] >= COS_1DEG_SQ * buf_ot_mag[l]));
        buf1_adm_div[l] = adm_div_lookup[ref.bands[1][index + l] + 32768];
        buf2_adm_div[l] = adm_div_lookup[ref.bands[2][index + l] + 32768];
        buf3_adm_div[l] = adm_div_lookup[ref.bands[3][index + l] + 32768];
      }

      src16x8_dD = vld1q_s16(distBandD + index);
      src16x8_rD = vld1q_s16(refBandD + index);
      src16x8_rD_lo = vget_low_s16(src16x8_rD);

      tmp_Hlo = vmovl_s16(src16x8_rH_lo);
      tmp_Hhi = vmovl_high_s16(src16x8_rH);
      tmp_Vlo = vmovl_s16(src16x8_rV_lo);
      tmp_Vhi = vmovl_high_s16(src16x8_rV);
      tmp_Dlo = vmovl_s16(src16x8_rD_lo);
      tmp_Dhi = vmovl_high_s16(src16x8_rD);

      src32x4_b1lo = vld1q_s32(buf1_adm_div);
      src32x4_b1hi = vld1q_s32(buf1_adm_div + 4);
      src32x4_dH_lo = vmovl_s16(src16x8_dH_lo);
      src32x4_dH_hi = vmovl_high_s16(src16x8_dH);

      src32x4_b2lo = vld1q_s32(buf2_adm_div);
      src32x4_b2hi = vld1q_s32(buf2_adm_div + 4);
      src32x4_dV_lo = vmovl_s16(src16x8_dV_lo);
      src32x4_dV_hi = vmovl_high_s16(src16x8_dV);

      src32x4_b3lo = vld1q_s32(buf3_adm_div);
      src32x4_b3hi = vld1q_s32(buf3_adm_div + 4);
      src32x4_dD_lo = vmovl_s16(vget_low_s16(src16x8_dD));
      src32x4_dD_hi = vmovl_high_s16(src16x8_dD);

      mul64x2_H0 = vmull_s32(vget_low_s32(src32x4_b1lo), vget_low_s32(src32x4_dH_lo));
      mul64x2_H1 = vmull_high_s32(src32x4_b1lo, src32x4_dH_lo);
      mul64x2_H2 = vmull_s32(vget_low_s32(src32x4_b1hi), vget_low_s32(src32x4_dH_hi));
      mul64x2_H3 = vmull_high_s32(src32x4_b1hi, src32x4_dH_hi);

      mul64x2_V0 = vmull_s32(vget_low_s32(src32x4_b2lo), vget_low_s32(src32x4_dV_lo));
      mul64x2_V1 = vmull_high_s32(src32x4_b2lo, src32x4_dV_lo);
      mul64x2_V2 = vmull_s32(vget_low_s32(src32x4_b2hi), vget_low_s32(src32x4_dV_hi));
      mul64x2_V3 = vmull_high_s32(src32x4_b2hi, src32x4_dV_hi);

      mul64x2_D0 = vmull_s32(vget_low_s32(src32x4_b3lo), vget_low_s32(src32x4_dD_lo));
      mul64x2_D1 = vmull_high_s32(src32x4_b3lo, src32x4_dD_lo);
      mul64x2_D2 = vmull_s32(vget_low_s32(src32x4_b3hi), vget_low_s32(src32x4_dD_hi));
      mul64x2_D3 = vmull_high_s32(src32x4_b3hi, src32x4_dD_hi);

      sft32x4_Hlo = vcombine_s32(vrshrn_n_s64(mul64x2_H0, 15), vrshrn_n_s64(mul64x2_H1, 15));
      sft32x4_Hhi = vcombine_s32(vrshrn_n_s64(mul64x2_H2, 15), vrshrn_n_s64(mul64x2_H3, 15));
      sft32x4_Vlo = vcombine_s32(vrshrn_n_s64(mul64x2_V0, 15), vrshrn_n_s64(mul64x2_V1, 15));
      sft32x4_Vhi = vcombine_s32(vrshrn_n_s64(mul64x2_V2, 15), vrshrn_n_s64(mul64x2_V3, 15));
      sft32x4_Dlo = vcombine_s32(vrshrn_n_s64(mul64x2_D0, 15), vrshrn_n_s64(mul64x2_D1, 15));
      sft32x4_Dhi = vcombine_s32(vrshrn_n_s64(mul64x2_D2, 15), vrshrn_n_s64(mul64x2_D3, 15));

      chkZero32x4_rHlo = vceqq_s32(tmp_Hlo, dupConstZero);
      chkZero32x4_rHhi = vceqq_s32(tmp_Hhi, dupConstZero);
      chkZero32x4_rVlo = vceqq_s32(tmp_Vlo, dupConstZero);
      chkZero32x4_rVhi = vceqq_s32(tmp_Vhi, dupConstZero);
      chkZero32x4_rDlo = vceqq_s32(tmp_Dlo, dupConstZero);
      chkZero32x4_rDhi = vceqq_s32(tmp_Dhi, dupConstZero);

      tmp_Hlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rHlo, vreinterpretq_u32_s32(dupVal0)));
      tmp_Hhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rHhi, vreinterpretq_u32_s32(dupVal0)));
      tmp_Vlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rVlo, vreinterpretq_u32_s32(dupVal0)));
      tmp_Vhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rVhi, vreinterpretq_u32_s32(dupVal0)));
      tmp_Dlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rDlo, vreinterpretq_u32_s32(dupVal0)));
      tmp_Dhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rDhi, vreinterpretq_u32_s32(dupVal0)));

      tmpVal_Hlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rHlo), vreinterpretq_u32_s32(sft32x4_Hlo)));
      tmpVal_Hhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rHhi), vreinterpretq_u32_s32(sft32x4_Hhi)));
      tmpVal_Vlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rVlo), vreinterpretq_u32_s32(sft32x4_Vlo)));
      tmpVal_Vhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rVhi), vreinterpretq_u32_s32(sft32x4_Vhi)));
      tmpVal_Dlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rDlo), vreinterpretq_u32_s32(sft32x4_Dlo)));
      tmpVal_Dhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rDhi), vreinterpretq_u32_s32(sft32x4_Dhi)));

      tmp_Hlo = vaddq_s32(tmp_Hlo, tmpVal_Hlo);
      tmp_Hhi = vaddq_s32(tmp_Hhi, tmpVal_Hhi);
      tmp_Vlo = vaddq_s32(tmp_Vlo, tmpVal_Vlo);
      tmp_Vhi = vaddq_s32(tmp_Vhi, tmpVal_Vhi);
      tmp_Dlo = vaddq_s32(tmp_Dlo, tmpVal_Dlo);
      tmp_Dhi = vaddq_s32(tmp_Dhi, tmpVal_Dhi);

      kh16x4_Hlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Hlo));
      kh16x4_Hhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Hhi));
      kh16x4_Vlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Vlo));
      kh16x4_Vhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Vhi));
      kh16x4_Dlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Dlo));
      kh16x4_Dhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Dhi));

      tmpVal_Hlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Hlo)), vmovl_s16(src16x8_rH_lo));
      tmpVal_Hhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Hhi)), vmovl_high_s16(src16x8_rH));
      tmpVal_Vlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Vlo)), vmovl_s16(src16x8_rV_lo));
      tmpVal_Vhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Vhi)), vmovl_high_s16(src16x8_rV));
      tmpVal_Dlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Dlo)), vmovl_s16(src16x8_rD_lo));
      tmpVal_Dhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Dhi)), vmovl_high_s16(src16x8_rD));

      sft16x8_H = vcombine_s16(vrshrn_n_s32((tmpVal_Hlo), 15), vrshrn_n_s32((tmpVal_Hhi), 15));
      sft16x8_V = vcombine_s16(vrshrn_n_s32((tmpVal_Vlo), 15), vrshrn_n_s32((tmpVal_Vhi), 15));
      sft16x8_D = vcombine_s16(vrshrn_n_s32((tmpVal_Dlo), 15), vrshrn_n_s32((tmpVal_Dhi), 15));

      angFlagBuf = vld1q_u16(angle_flag);
      angBuf1 = vcltq_u16(angFlagBuf, dupConst1);
      angBuf0 = vcgtzq_s16(vreinterpretq_s16_u16(angFlagBuf));

      sftModH = vandq_s16(sft16x8_H, vreinterpretq_s16_u16(angBuf1));
      srcModH = vandq_s16(src16x8_dH, vreinterpretq_s16_u16(angBuf0));
      sftModV = vandq_s16(sft16x8_V, vreinterpretq_s16_u16(angBuf1));
      srcModV = vandq_s16(src16x8_dV, vreinterpretq_s16_u16(angBuf0));
      sftModD = vandq_s16(sft16x8_D, vreinterpretq_s16_u16(angBuf1));
      srcModD = vandq_s16(src16x8_dD, vreinterpretq_s16_u16(angBuf0));

      dlmRest_H = vaddq_s16(sftModH, srcModH);
      dlmRest_V = vaddq_s16(sftModV, srcModV);
      dlmRest_D = vaddq_s16(sftModD, srcModD);

      admAdd_H = vabdq_s16(src16x8_dH, dlmRest_H);
      admAdd_V = vabdq_s16(src16x8_dV, dlmRest_V);
      admAdd_D = vabdq_s16(src16x8_dD, dlmRest_D);

      vst1q_s16((i_dlm_rest.bands[1] + index), dlmRest_H);
      vst1q_s16((i_dlm_rest.bands[2] + index), dlmRest_V);
      vst1q_s16((i_dlm_rest.bands[3] + index), dlmRest_D);

      vst1q_u16((i_dlm_add.bands[1] + index), vreinterpretq_u16_s16(admAdd_H));
      vst1q_u16((i_dlm_add.bands[2] + index), vreinterpretq_u16_s16(admAdd_V));
      vst1q_u16((i_dlm_add.bands[3] + index), vreinterpretq_u16_s16(admAdd_D));
    }
    for (; j < width; j++)
    {
      index = i * width + j;
      ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
      o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
      t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);
      angle_flagC = ((ot_dp >= 0) && (((int64_t)ot_dp * ot_dp) >= COS_1DEG_SQ * ((int64_t)o_mag_sq * t_mag_sq)));

      for (k = 1; k < 4; k++)
      {
        adm_i32_dtype tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((adm_i64_dtype)adm_div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
        adm_u16_dtype kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
        tmp_val = (((adm_i32_dtype)kh * ref.bands[k][index]) + 16384) >> 15;

        i_dlm_rest.bands[k][index] = angle_flagC ? dist.bands[k][index] : tmp_val;
        i_dlm_add.bands[k][index] = abs(dist.bands[k][index] - i_dlm_rest.bands[k][index]); // to avoid abs in integer_dlm_contrast_mask_one_way function
      }
    }
  }

  free(buf_ot_dp);
  free(buf_ot_dp_sq);
  free(buf_ot_mag);
  free(buf1_adm_div);
  free(buf2_adm_div);
  free(buf3_adm_div);
}
