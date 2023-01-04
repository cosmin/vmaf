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
#include <stdlib.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mem.h"
#include "funque_adm_options.h"
#include "adm_tools.h"
#include "funque_filters.h"
#include "funque_adm.h"
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline float clip(float value, float low, float high)
{
  return value < low ? low : (value > high ? high : value);
}

#ifdef __SSE2__
#ifdef ADM_OPT_RECIP_DIVISION

#include <emmintrin.h>

// static float rcp_s(float x)
// {
//   float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
//   return xi + xi * (1.0f - x * xi);
// }

#define DIVS(n, d) ((n)*rcp_s(d))
#endif // ADM_OPT_RECIP_DIVISION
#else
#define DIVS(n, d) ((n) / (d))
#endif // __SSE2__

void reflect_pad_adm(const float *src, int width, int height, int reflect, float *dest)
{
  int out_width = width + 2 * reflect;
  int out_height = height + 2 * reflect;
  int i, j;
  static int cnt = 0;
  cnt++;
  // Skip first `reflect` rows, iterate through next height rows
  for (i = reflect; i != (out_height - reflect); i++)
  {

    // Mirror first `reflect` columns
    for (j = 0; j != reflect; j++)
    {
      dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
    }

    // Copy corresponding row values from input image
    memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(float) * width);

    // Mirror last `reflect` columns
    for (j = 0; j != reflect; j++)
      dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
  }

  // Mirror first `reflect` and last `reflect` rows
  for (i = 0; i != reflect; i++)
  {
    memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(float) * out_width);
    memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(float) * out_width);
  }
}

void integral_image_adm(const float *src, int width, int height, float *sum)
{
  int i, j;
  for (i = 0; i < (height + 1); ++i)
  {
    for (j = 0; j < (width + 1); ++j)
    {
      if (i == 0 || j == 0)
        continue;

      float val = src[(i - 1) * width + (j - 1)];

      if (i >= 1)
      {
        val += sum[(i - 1) * (width + 1) + j];
        if (j >= 1)
        {
          val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
        }
      }
      else
      {
        if (j >= 1)
        {
          val += sum[i * width + j - 1];
        }
      }
      sum[i * (width + 1) + j] = val;
    }
  }
}

void integral_image_adm_sums(float *x, int k, int stride, float *mx, int width, int height)
{
  float *x_pad, *int_x;
  int i, j;

  int x_reflect = (int)((k - stride) / 2);

  x_pad = (float *)malloc(sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));

  reflect_pad_adm(x, width, height, x_reflect, x_pad);

  int r_width = width + (2 * x_reflect);
  int r_height = height + (2 * x_reflect);

  int_x = (float *)calloc((r_width + 1) * (r_height + 1), sizeof(float));

  integral_image_adm(x_pad, r_width, r_height, int_x);

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      mx[i * width + j] = (int_x[i * (width + 3) + j] - int_x[i * (width + 3) + j + k] - int_x[(i + k) * (width + 3) + j] + int_x[(i + k) * (width + 3) + j + k]);
    }
  }
  free(x_pad);
  free(int_x);
}

void dlm_decouple(dwt2buffers ref, dwt2buffers dist, dwt2buffers dlm_rest, dwt2buffers dlm_add)
{
#ifdef ADM_OPT_AVOID_ATAN
  const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
  float eps = 1e-30;
  int width = ref.width;
  int height = ref.height;
  int i, j, k, index;

  // float *var_k;
  float val;
  float tmp_val;
  int angle_flag;

#ifdef ADM_OPT_AVOID_ATAN
  float *ot_dp = (float *)calloc(width * height, sizeof(float));
  float *o_mag_sq = (float *)calloc(width * height, sizeof(float));
  float *t_mag_sq = (float *)calloc(width * height, sizeof(float));
#else
  float *psi_ref = (float *)calloc(width * height, sizeof(float));
  float *psi_dist = (float *)calloc(width * height, sizeof(float));
  float *psi_diff = (float *)calloc(width * height, sizeof(float));
#endif

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      index = i * width + j;
#ifdef ADM_OPT_AVOID_ATAN
      ot_dp[index] = (ref.bands[1][index] * dist.bands[1][index]) + (ref.bands[2][index] * dist.bands[2][index]);
      o_mag_sq[index] = (ref.bands[1][index] * ref.bands[1][index]) + (ref.bands[2][index] * ref.bands[2][index]);
      t_mag_sq[index] = (dist.bands[1][index] * dist.bands[1][index]) + (dist.bands[2][index] * dist.bands[2][index]);
      angle_flag = (ot_dp[index] >= 0.0f) && (ot_dp[index] * ot_dp[index] >= cos_1deg_sq * o_mag_sq[index] * t_mag_sq[index]);
#else
      psi_ref[index] = atanf(ref.bands[2][index] / (ref.bands[1][index] + eps)) + M_PI * ((ref.bands[1][index] <= 0));
      psi_dist[index] = atanf(dist.bands[2][index] / (dist.bands[1][index] + eps)) + M_PI * ((dist.bands[1][index] <= 0));
      psi_diff[index] = 180 * fabsf(psi_ref[index] - psi_dist[index]) / M_PI;
      angle_flag = psi_diff[index] < 1;
#endif
      for (k = 1; k < 4; k++)
      {
        val = clip(dist.bands[k][index] / (ref.bands[k][index] + eps), 0.0, 1.0);
        tmp_val = (val * (ref.bands[k][index]));

        dlm_rest.bands[k][index] = angle_flag ? (dist.bands[k][index]) : tmp_val;
        dlm_add.bands[k][index] = dist.bands[k][index] - dlm_rest.bands[k][index];
      }
    }
  }

#ifdef ADM_OPT_AVOID_ATAN
  free(ot_dp);
  free(o_mag_sq);
  free(t_mag_sq);
#else
  free(psi_ref);
  free(psi_dist);
  free(psi_diff);
#endif
}

void dlm_contrast_mask_one_way(dwt2buffers pyr_1, dwt2buffers pyr_2, dwt2buffers masked_pyr, int width, int height)
{
  int i, k, j, index;
  float val = 0;
  float *masking_signal, *masking_threshold, *integral_sum;

  masking_signal = (float *)calloc(width * height, sizeof(float));
  masking_threshold = (float *)calloc(width * height, sizeof(float));
  integral_sum = (float *)calloc(width * height, sizeof(float));

  for (k = 1; k < 4; k++)
  {
    // printf("printing masking signal in band %d\n", k);
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
        masking_signal[index] = fabsf(pyr_2.bands[k][index]);
      }
    }
    integral_image_adm_sums(masking_signal, 3, 1, integral_sum, width, height);
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
        masking_threshold[index] += (integral_sum[index] + masking_signal[index]) / 30;
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
        val = fabsf(pyr_1.bands[k][index]) - masking_threshold[index];
        masked_pyr.bands[k][index] = clip(val, 0.0, val);
      }
    }
  }
  free(masking_signal);
  free(masking_threshold);
  free(integral_sum);
}

int compute_adm_funque(dwt2buffers ref, dwt2buffers dist, double *adm_score, double *adm_score_num, double *adm_score_den, float border_size)
{
  int width = ref.width;
  int height = ref.height;
  int i, j, k, index;
  double num_sum = 0, den_sum = 0, num_band = 0, den_band = 0;
  dwt2buffers dlm_rest, dlm_add, pyr_rest;
  dlm_rest.bands[0] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[1] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[2] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[3] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[0] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[1] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[2] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[3] = (float *)malloc(sizeof(float) * height * width);
  pyr_rest.bands[0] = (float *)malloc(sizeof(float) * height * width);
  pyr_rest.bands[1] = (float *)malloc(sizeof(float) * height * width);
  pyr_rest.bands[2] = (float *)malloc(sizeof(float) * height * width);
  pyr_rest.bands[3] = (float *)malloc(sizeof(float) * height * width);

  dlm_decouple(ref, dist, dlm_rest, dlm_add);

  dlm_contrast_mask_one_way(dlm_rest, dlm_add, pyr_rest, width, height);

  int border_h = (border_size * height);
  int border_w = (border_size * width);
  int loop_h = height - border_h;
  int loop_w = width - border_w;

  for (k = 1; k < 4; k++)
  {
    for (i = border_h; i < loop_h; i++)
    {
      for (j = border_w; j < loop_w; j++)
      {
        index = i * width + j;
        num_sum += powf(pyr_rest.bands[k][index], 3.0);
        den_sum += powf(fabsf(ref.bands[k][index]), 3.0);
      }
    }
    den_band += powf(den_sum, 1.0 / 3.0);
    num_band += powf(num_sum, 1.0 / 3.0);
    num_sum = 0;
    den_sum = 0;
  }

  *adm_score_num = num_band + 1e-4;
  *adm_score_den = den_band + 1e-4;
  *adm_score = (*adm_score_num) / (*adm_score_den);

  for (int i = 0; i < 4; i++)
  {
    free(dlm_rest.bands[i]);
    free(dlm_add.bands[i]);
    free(pyr_rest.bands[i]);
  }

  int ret = 0;
  return ret;
}
