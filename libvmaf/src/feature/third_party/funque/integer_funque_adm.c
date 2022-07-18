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
#include <stddef.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mem.h"
#include "adm_tools.h"
#include "integer_funque_filters.h"
#include "integer_funque_adm.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

typedef struct i_adm_buffers {
    adm_i32_dtype *bands[4];
    int width;
    int height;
}i_adm_buffers;

typedef struct u_adm_buffers {
    adm_u16_dtype *bands[4];
    int width;
    int height;
}u_adm_buffers;


static const int32_t div_Q_factor = 1073741824; // 2^30

void div_lookup_generator(int32_t* adm_div_lookup)
{
    for (int i = 1; i <= 32768; ++i)
    {
        int32_t recip = (int32_t)(div_Q_factor / i);
        adm_div_lookup[32768 + i] = recip;
        adm_div_lookup[32768 - i] = 0 - recip;
    }
}

static inline int clip(int value, int low, int high)
{
  return value < low ? low : (value > high ? high : value);
}

void integer_reflect_pad_adm(const adm_u16_dtype *src, size_t width, size_t height, int reflect, adm_u16_dtype *dest)
{
  size_t out_width = width + 2 * reflect;
  size_t out_height = height + 2 * reflect;

  for (size_t i = reflect; i != (out_height - reflect); i++)
  {

    for (int j = 0; j != reflect; j++)
    {
      dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
    }

    memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(adm_u16_dtype) * width);

    for (int j = 0; j != reflect; j++)
      dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
  }

  for (int i = 0; i != reflect; i++)
  {
    memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(adm_u16_dtype) * out_width);
    memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(adm_u16_dtype) * out_width);
  }
}

void integer_integral_image_adm(const adm_u16_dtype *src, size_t width, size_t height, adm_i64_dtype *sum)
{
  double st1, st2, st3;

  for (size_t i = 0; i < (height + 1); ++i)
  {
    for (size_t j = 0; j < (width + 1); ++j)
    {
      if (i == 0 || j == 0)
        continue;

      adm_i64_dtype val = (adm_i64_dtype)(src[(i - 1) * width + (j - 1)]); // 64 to avoid overflow

      val += (adm_i64_dtype)(sum[(i - 1) * (width + 1) + j]);
      val += (adm_i64_dtype)(sum[i * (width + 1) + j - 1]) - (adm_i64_dtype)(sum[(i - 1) * (width + 1) + j - 1]);
      sum[i * (width + 1) + j] = val;
    }
  }
}

void integer_integral_image_adm_sums(adm_u16_dtype *x, int k, int stride, adm_i32_dtype *mx, adm_i32_dtype *masking_threshold_int, int width, int height)
{
  dwt2_dtype *x_pad;
  adm_i64_dtype *int_x;
  int i, j, index;

  int x_reflect = (int)((k - stride) / 2);

  x_pad = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));

  integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);

  size_t r_width = width + (2 * x_reflect);
  size_t r_height = height + (2 * x_reflect);

  int_x = (adm_i64_dtype *)malloc((r_width + 1) * (r_height + 1), sizeof(adm_i64_dtype));

  integer_integral_image_adm(x_pad, r_width, r_height, int_x);

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      index = i * width + j;
      mx[index] = (int_x[i * (width + 3) + j] - int_x[i * (width + 3) + j + k] - int_x[(i + k) * (width + 3) + j] + int_x[(i + k) * (width + 3) + j + k]);
      masking_threshold_int[index] = (adm_i32_dtype)x[index] + mx[index];
    }
  }

  free(x_pad);
  free(int_x);
}

void integer_dlm_contrast_mask_one_way(i_dwt2buffers pyr_1, u_adm_buffers pyr_2, i_adm_buffers masked_pyr, size_t width, size_t height)
{
  int i, k, j, index;
  adm_i32_dtype val = 0;
  adm_i32_dtype pyr_abs;
  adm_i32_dtype *masking_threshold, *masking_threshold_int;
  adm_i32_dtype *integral_sum;

  masking_threshold_int = (adm_i32_dtype *)malloc(width * height, sizeof(adm_i32_dtype));
  masking_threshold = (adm_i32_dtype *)malloc(width * height, sizeof(adm_i32_dtype));
  integral_sum = (adm_i32_dtype *)malloc(width * height, sizeof(adm_i32_dtype));

  integer_integral_image_adm_sums(pyr_2.bands[1], 3, 1, integral_sum, masking_threshold_int, width, height);
  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      index = i * width + j;
      masking_threshold[index] = masking_threshold_int[index];
    }
  }
  
  for (k = 2; k < 4; k++)
  {
    integer_integral_image_adm_sums(pyr_2.bands[k], 3, 1, integral_sum, masking_threshold_int, width, height);
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
        masking_threshold[index] += masking_threshold_int[index];
      }
    }
  }

  for (k = 1; k < 4; k++)
  {
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
        // compensation for the division by 30 of masking_threshold
        pyr_abs = abs((adm_i32_dtype)pyr_1.bands[k][index]) * 30;
        val = pyr_abs - masking_threshold[index];
        masked_pyr.bands[k][index] = (adm_i32_dtype)clip(val, 0.0, val);
      }
    }
  }
  free(masking_threshold);
  free(masking_threshold_int);
  free(integral_sum);
}

void integer_dlm_decouple(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, u_adm_buffers i_dlm_add, int32_t *adm_div_lookup)
{
  const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
  size_t width = ref.width;
  size_t height = ref.height;
  int i, j, k, index;

  adm_i16_dtype tmp_val;
  int angle_flag;

  adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      index = i * width + j;
      ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
      o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
      t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);

      /** angle_flag is calculated in floating-point by converting fixed-point variables back to floating-point  */
      angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) && ((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >= cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

      for (k = 1; k < 4; k++)
      {
        /**
         * Division dist/ref is carried using lookup table and converted to multiplication
         */
        adm_i32_dtype tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((adm_i64_dtype)adm_div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
        adm_u16_dtype kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
        /**
         * kh is in Q15 type and ref.bands[k][index] is in Q16 type hence shifted by
         * 15 to make result Q16
         */
        tmp_val = (((adm_i32_dtype)kh * ref.bands[k][index]) + 16384) >> 15;

        i_dlm_rest.bands[k][index] = angle_flag ? dist.bands[k][index] : tmp_val;
        i_dlm_add.bands[k][index] = abs(dist.bands[k][index] - i_dlm_rest.bands[k][index]); // to avoid abs in cotrast_mask function
      }
    }
  }

  free(ot_dp);
  free(o_mag_sq);
  free(t_mag_sq);
}

int integer_compute_adm_funque(i_dwt2buffers i_ref, i_dwt2buffers i_dist, double *adm_score, double *adm_score_num, double *adm_score_den, size_t width, size_t height, float border_size, int16_t shift_val, int32_t *adm_div_lookup)
{

  int i, j, k, index;
  adm_i64_dtype num_sum = 0, den_sum = 0;
  adm_i32_dtype ref_abs;
  adm_i64_dtype num_cube = 0, den_cube = 0;
  double num_band = 0, den_band = 0;
  i_dwt2buffers i_dlm_rest;
  u_adm_buffers i_dlm_add;
  i_adm_buffers i_pyr_rest;
  i_dlm_rest.bands[1] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
  i_dlm_rest.bands[2] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
  i_dlm_rest.bands[3] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
  i_dlm_add.bands[1] = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
  i_dlm_add.bands[2] = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
  i_dlm_add.bands[3] = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
  i_pyr_rest.bands[1] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);
  i_pyr_rest.bands[2] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);
  i_pyr_rest.bands[3] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);

  integer_dlm_decouple(i_ref, i_dist, i_dlm_rest, i_dlm_add, adm_div_lookup);

  integer_dlm_contrast_mask_one_way(i_dlm_rest, i_dlm_add, i_pyr_rest, width, height);

  int border_h = (border_size * height);
  int border_w = (border_size * width);
  int loop_h = height - border_h;
  int loop_w = width - border_w;
  double row_num, row_den, accum_num = 0, accum_den = 0;

  for (k = 1; k < 4; k++)
  {
    for (i = border_h; i < loop_h; i++)
    {
      for (j = border_w; j < loop_w; j++)
      {
        index = i * width + j;
        num_cube = (adm_i64_dtype)i_pyr_rest.bands[k][index] * i_pyr_rest.bands[k][index] * i_pyr_rest.bands[k][index];
        num_sum += ((num_cube + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
        // compensation for the division by thirty in the numerator
        ref_abs = abs((adm_i64_dtype)i_ref.bands[k][index]) * 30;
        den_cube = (adm_i64_dtype)ref_abs * ref_abs * ref_abs;
        den_sum += ((den_cube + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT);
      }
      row_num = (double)num_sum ;
      row_den = (double)den_sum ;
      accum_num += row_num;
      accum_den += row_den;
      num_sum = 0;
      den_sum = 0;
    }

    den_band += powf((double)(accum_den), 1.0 / 3.0);
    num_band += powf((double)(accum_num), 1.0 / 3.0);
    accum_num = 0;
    accum_den = 0;
  }

  *adm_score_num = num_band + 1e-4;
  *adm_score_den = den_band + 1e-4;
  *adm_score = (*adm_score_num) / (*adm_score_den);

  for (int i = 1; i < 4; i++)
  {
    free(i_dlm_rest.bands[i]);
    free(i_dlm_add.bands[i]);
    free(i_pyr_rest.bands[i]);
  }

  int ret = 0;
  return ret;
}
