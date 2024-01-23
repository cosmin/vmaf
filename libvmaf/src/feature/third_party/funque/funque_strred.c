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
#include "funque_strred.h"

void strred_reflect_pad(const float* src, size_t width, size_t height, int reflect, float* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for(size_t i = reflect; i != (out_height - reflect); i++) {
        for(int j = 0; j != reflect; j++) {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(float) * width);

        for(int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] =
                dest[i * out_width + out_width - reflect - 2 - j];
    }

    for(int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width],
               &dest[reflect * out_width + (i + 1) * out_width], sizeof(float) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width],
               &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width],
               sizeof(float) * out_width);
    }
}

void strred_strred_integral_image_2(const float* src1, const float* src2, size_t width,
                                    size_t height, double* sum)
{
    for(size_t i = 0; i < (height + 1); ++i) {
        for(size_t j = 0; j < (width + 1); ++j) {
            if(i == 0 || j == 0)
                continue;

            double val =
                (double) src1[(i - 1) * width + (j - 1)] * (double) src2[(i - 1) * width + (j - 1)];

            if(i >= 1) {
                val += sum[(i - 1) * (width + 1) + j];
                if(j >= 1) {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            } else {
                if(j >= 1) {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void strred_integral_image(const float* src, size_t width, size_t height, double* sum)
{
    for(size_t i = 0; i < (height + 1); ++i) {
        for(size_t j = 0; j < (width + 1); ++j) {
            if(i == 0 || j == 0)
                continue;

            double val = (double) src[(i - 1) * width + (j - 1)];

            if(i >= 1) {
                val += sum[(i - 1) * (width + 1) + j];
                if(j >= 1) {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            } else {
                if(j >= 1) {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void strred_compute_entropy_scale(const double* int_1_x, const double* int_2_x, size_t width,
                                  size_t height, size_t kh, size_t kw, int kNorm, float* entropy,
                                  float* scale)
{
    float mu_x, var_x;
    float entr_const = log(2 * PI_CONSTANT * EULERS_CONSTANT);
    float sigma_nsq = STRRED_SIGMA_NSQ;

    for(size_t i = 0; i < (height - kh); i++) {
        for(size_t j = 0; j < (width - kw); j++) {
            mu_x = (int_1_x[i * width + j] - int_1_x[i * width + j + kw] -
                    int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw]) /
                    kNorm;
            var_x = ((int_2_x[i * width + j] - int_2_x[i * width + j + kw] -
                      int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) /
                      kNorm) - (mu_x * mu_x);

            var_x = (var_x < 0) ? 0 : var_x; /* Add CLIP macro */

            entropy[i * width + j] = log(var_x + sigma_nsq) + entr_const;
            scale[i * width + j] = log(1 + var_x);
        }
    }
}

void rred_entropies_and_scales(const float* src, int block_size, size_t width, size_t height,
                                float* entropy, float* scale)
{
    if(block_size == 1) {
        int k = STRRED_WINDOW_SIZE;
        int k_norm = k * k;
        int kw = k;
        int kh = k;

        int x_reflect = (int) ((kw - 1) / 2);
        size_t r_width = width + (2 * x_reflect);
        size_t r_height = height + (2 * x_reflect);

        float* x_pad =
            (float*) malloc(sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
        double* int_1_x = (double*) calloc((r_width + 1) * (r_height + 1), sizeof(double));
        double* int_2_x = (double*) calloc((r_width + 1) * (r_height + 1), sizeof(double));

        strred_reflect_pad(src, width, height, x_reflect, x_pad);
        strred_integral_image(x_pad, r_width, r_height, int_1_x);
        strred_strred_integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
        strred_compute_entropy_scale(int_1_x, int_2_x, r_width + 1, r_height + 1, kw, kh, k_norm,
                                     entropy, scale);

        free(x_pad);
        free(int_1_x);
        free(int_2_x);
    }
}

void subract_subbands(const float* ref_src, const float* ref_prev_src, float* ref_dst,
                      const float* dist_src, const float* dist_prev_src, float* dist_dst,
                      size_t width, size_t height)
{
    size_t i, j;

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            ref_dst[i * width + j] = ref_src[i * width + j] - ref_prev_src[i * width + j];
            dist_dst[i * width + j] = dist_src[i * width + j] - dist_prev_src[i * width + j];
        }
    }
}

int compute_srred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist,
                         size_t width, size_t height, float** spat_scales_ref, float** spat_scales_dist,
                         struct strred_results* strred_scores, int block_size, int level)
{
    size_t subband;
    float spat_abs, spat_values[DEFAULT_STRRED_SUBBANDS];

    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t x_reflect = (size_t) ((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    float* entropies_ref = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));
    float* entropies_dist = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));
    float* spat_aggregate = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));

    for(subband = 1; subband < total_subbands; subband++) {
        size_t i, j;
        spat_abs = 0;

        rred_entropies_and_scales(ref->bands[subband], block_size, width, height, entropies_ref,
                                  spat_scales_ref[subband]);
        rred_entropies_and_scales(dist->bands[subband], block_size, width, height, entropies_dist,
                                  spat_scales_dist[subband]);

        for(i = 0; i < r_height; i++) {
            for(j = 0; j < r_width; j++) {
                spat_aggregate[i * r_width + j] =
                    entropies_ref[i * r_width + j] * spat_scales_ref[subband][i * r_width + j] -
                    entropies_dist[i * r_width + j] * spat_scales_dist[subband][i * r_width + j];
            }
        }

        for(i = 0; i < r_height; i++) {
            for(j = 0; j < r_width; j++) {
                spat_abs += fabs(spat_aggregate[i * r_width + j]);
            }
        }
        spat_values[subband] = spat_abs / (height * width);
    }

    strred_scores->spat_vals[level] = (spat_values[1] + spat_values[2] + spat_values[3]) / 3;

    // Add equations to compute S-RRED using norm factors
    int norm_factor, num_level;
    for(num_level = 0; num_level <= level; num_level++)
        norm_factor = num_level + 1;

    strred_scores->spat_vals_cumsum += strred_scores->spat_vals[level];

    strred_scores->srred_vals[level] = strred_scores->spat_vals_cumsum / norm_factor;

    free(entropies_ref);
    free(entropies_dist);
    free(spat_aggregate);

    return 0;
}

int compute_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist,
                          struct strredbuffers* prev_ref, struct strredbuffers* prev_dist,
                          size_t width, size_t height, float** spat_scales_ref, float** spat_scales_dist, 
                          struct strred_results* strred_scores, int block_size, int level)
{
    size_t subband;
    float temp_abs, temp_values[DEFAULT_STRRED_SUBBANDS];

    size_t total_subbands = DEFAULT_STRRED_SUBBANDS;
    size_t x_reflect = (size_t) ((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    float* entropies_ref = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));
    float* entropies_dist = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));
    float* temp_scales_ref = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));
    float* temp_scales_dist = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));

    for(subband = 1; subband < total_subbands; subband++) {
        size_t i, j;

        if(prev_ref != NULL && prev_dist != NULL) {
            float* ref_temporal = (float*) calloc((width) * (height), sizeof(float));
            float* dist_temporal = (float*) calloc((width) * (height), sizeof(float));
            float* temp_aggregate = (float*) calloc((r_width + 1) * (r_height + 1), sizeof(float));

            temp_abs = 0;

            subract_subbands(ref->bands[subband], prev_ref->bands[subband], ref_temporal,
                             dist->bands[subband], prev_dist->bands[subband], dist_temporal, width,
                             height);

            rred_entropies_and_scales(ref_temporal, block_size, width, height, entropies_ref,
                                      temp_scales_ref);
            rred_entropies_and_scales(dist_temporal, block_size, width, height, entropies_dist,
                                      temp_scales_dist);

            for(i = 0; i < r_height; i++) {
                for(j = 0; j < r_width; j++) {
                    temp_aggregate[i * r_width + j] =
                        entropies_ref[i * r_width + j] * spat_scales_ref[subband][i * r_width + j] *
                            temp_scales_ref[i * r_width + j] -
                        entropies_dist[i * r_width + j] * spat_scales_dist[subband][i * r_width + j] *
                            temp_scales_dist[i * r_width + j];
                }
            }

            for(i = 0; i < r_height; i++) {
                for(j = 0; j < r_width; j++) {
                    temp_abs += fabs(temp_aggregate[i * r_width + j]);
                }
            }
            temp_values[subband] = temp_abs / (height * width);

            free(ref_temporal);
            free(dist_temporal);
            free(temp_aggregate);
        } else {
            strred_scores->temp_vals[level] = 0;
            strred_scores->spat_temp_vals[level] = 0;
        }
    }

    strred_scores->temp_vals[level] = (temp_values[1] + temp_values[2] + temp_values[3]) / 3;
    strred_scores->spat_temp_vals[level] =
        strred_scores->spat_vals[level] * strred_scores->temp_vals[level];

    // Add equations to compute ST-RRED using norm factors
    int norm_factor, num_level;
    for(num_level = 0; num_level <= level; num_level++)
        norm_factor = num_level + 1;

    strred_scores->temp_vals_cumsum += strred_scores->temp_vals[level];
    strred_scores->spat_temp_vals_cumsum += strred_scores->spat_temp_vals[level];

    strred_scores->trred_vals[level] = strred_scores->temp_vals_cumsum / norm_factor;
    strred_scores->strred_vals[level] = strred_scores->spat_temp_vals_cumsum / norm_factor;

    free(entropies_ref);
    free(entropies_dist);
    free(temp_scales_ref);
    free(temp_scales_dist);

    return 0;
}