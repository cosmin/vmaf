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
#include <mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "funque_filters.h"
#include "funque_strred_options.h"
#include "funque_global_options.h"

void strred_reflect_pad(const float* src, size_t width, size_t height, int reflect, float* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(float) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(float) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(float) * out_width);
    }
}

void strred_strred_integral_image_2(const float* src1, const float* src2, size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            double val = (double)src1[(i - 1) * width + (j - 1)] * (double)src2[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void strred_integral_image(const float* src, size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            double val = (double)src[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void strred_compute_entropy_scale(const double* int_1_x, const double* int_2_x, size_t width, size_t height, size_t kh, size_t kw, double kNorm, double* entropy, double* scale, double entr_const)
{
    double mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = (int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw]) / kNorm;
            vx = ((int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) / kNorm) - (mx * mx);

            vx = (vx < 0) ? 0 : vx; /* Add CLIP macro */

            entropy[i * width + j] = log(vx + STRRED_SIGMA_NSQ) + entr_const;
            scale[i * width + j] = log(1 + vx);

        }
    }
}

void rred_entropies_and_scales(const float* src, int block_size, size_t width, size_t height, size_t stride, float *entropy, float *scale)
{
    float sigma_nsq = STRRED_SIGMA_NSQ;
    float tol = 1e-10;

    float entr_const;

    if(block_size == 1)
    {
        entr_const = log(2 * M_PI * EULERS_CONSTANT);
        int k = STRRED_WINDOW_SIZE;
        int k_norm = k * k;
        int kw = k;
        int kh = k;
        int wd, ht;
        int i, j;

        float* x_pad;

        int x_reflect = (int)((kw - 1) / 2);
        x_pad = (float*) malloc (sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));

        strred_reflect_pad(src, width, height, x_reflect, x_pad);
        size_t r_width = width + (2 * x_reflect);
        size_t r_height = height + (2 * x_reflect);

        double *int_1_x, *int_2_x;
        double *var_x;

        int_1_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        int_2_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

        strred_integral_image(x_pad, r_width, r_height, int_1_x);
        strred_strred_integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
        strred_compute_entropy_scale(int_1_x, int_2_x, r_width + 1, r_height + 1, kw, kh, k_norm, entropy, scale, entr_const);

        free(int_1_x);
        free(int_2_x);
    }
}

void subract_subbands(const float* ref_src, const float* ref_prev_src, float* ref_dst,
                      const float* dist_src, const float* dist_prev_src, float* dist_dst,
                      size_t width, size_t height)
{
    int i, j;

    for(i = 0; i < height; i++)
    {
        for(j = 0; j < width; j++)
        {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

int copy_prev_frame_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist, struct dwt2buffers* prev_ref, struct dwt2buffers* prev_dist, size_t width, size_t height)
{
    int subband;
    int total_subbands = DEFAULT_STRRED_SUBBANDS;

    for(subband = 1; subband < total_subbands; subband++)
    {
        prev_ref->bands[subband] = (double*)calloc(width * height, sizeof(double));
        prev_dist->bands[subband] = (double*)calloc(width * height, sizeof(double));

        memcpy(prev_ref->bands[subband], ref->bands[subband], width * height * sizeof(double));
        memcpy(prev_dist->bands[subband], dist->bands[subband], width * height * sizeof(double));
    }

    //prev_ref->width = ref->width;
    //prev_ref->height = ref->height;
    //prev_dist->width = dist->width;
    //prev_dist->height = dist->height;

    return 0;
}


int compute_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist, struct dwt2buffers* prev_ref, struct dwt2buffers* prev_dist,
                        size_t width, size_t height, double* srred_vals, double* trred_vals, double* strred_vals,
                        double* srred_approx_vals, double* trred_approx_vals, double* strred_approx_vals,
                        double* spat_vals, double* temp_vals, double* spat_temp_vals,
                        int k, int stride, double sigma_nsq_arg, int index, int level)
{
    int x_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    int subband;
    int total_subbands = DEFAULT_STRRED_SUBBANDS;

    
    int compute_temporal;

    // TODO: Insert an assert to check whether details ref and details dist are of same length
    int n_levels = sizeof(ref->bands) / sizeof(ref->bands[0]) - 1;

    double *entropies_ref = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    double *entropies_dist = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    double *scales_ref = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    double *scales_dist = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    double *temp_scales_ref = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    double *temp_scales_dist = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

    double *spat_aggregate = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

    float ref_entropy, ref_scale, dist_entropy, dist_scale;
    double spat_abs, spat_mean, spat_values[DEFAULT_STRRED_SUBBANDS];

    for(subband = 1; subband < total_subbands; subband++)
    {
        spat_abs = 0;

        rred_entropies_and_scales(ref->bands[subband], BLOCK_SIZE, width, height, stride, entropies_ref, scales_ref);
        rred_entropies_and_scales(dist->bands[subband], BLOCK_SIZE, width, height, stride, entropies_dist, scales_dist);

        for (int i = 0; i < r_height; i++)
        {
            for (int j = 0; j < r_width; j++)
            {
                spat_aggregate[i * r_width + j] = entropies_ref[i * r_width + j] * scales_ref[i * r_width + j] - entropies_dist[i * r_width + j] * scales_dist[i * r_width + j];
            }
        }

        for (int i = 0; i < r_height; i++)
        {
            for (int j = 0; j < r_width; j++)
            {
                spat_abs += fabs(spat_aggregate[i * r_width + j]);
            }
            spat_mean = spat_abs / (height * width);
        }
        spat_values[subband] = spat_mean;

        if(prev_ref != NULL && prev_dist != NULL)
        {
            float *ref_temporal = (float*)calloc((width) * (height), sizeof(float));
            float *dist_temporal = (float*)calloc((width) * (height), sizeof(float));
            double *temp_aggregate = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
            double temp_abs, temp_mean, temp_values[DEFAULT_STRRED_SUBBANDS];

            temp_abs = 0;

            subract_subbands(ref->bands[subband], prev_ref->bands[subband], ref_temporal,
                             dist->bands[subband], prev_dist->bands[subband], dist_temporal, width, height);

            rred_entropies_and_scales(ref_temporal, BLOCK_SIZE, width, height, stride, entropies_ref, temp_scales_ref);
            rred_entropies_and_scales(dist_temporal, BLOCK_SIZE, width, height, stride, entropies_dist, temp_scales_dist);

            for (int i = 0; i < r_height; i++)
            {
                for (int j = 0; j < r_width; j++)
                {
                    temp_aggregate[i * r_width + j] = entropies_ref[i * r_width + j] * scales_ref[i * r_width + j] * temp_scales_ref[i * r_width + j] -
                                                      entropies_dist[i * r_width + j] * scales_dist[i * r_width + j] * temp_scales_dist[i * r_width + j];
                }
            }

            for (int i = 0; i < r_height; i++)
            {
                for (int j = 0; j < r_width; j++)
                {
                    temp_abs += fabs(temp_aggregate[i * r_width + j]);
                }
                temp_mean = temp_abs / (height * width);
            }
            temp_values[subband] = temp_mean;
        }
    }

    return 0;
}