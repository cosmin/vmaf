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

void rred_entropies_and_scales(const float* x, int block_size, size_t width, size_t height, size_t stride, float *entropy, float *scale)
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

        strred_reflect_pad(x, width, height, x_reflect, x_pad);
        size_t r_width = width + (2 * x_reflect);
        size_t r_height = height + (2 * x_reflect);


        double *int_1_x, *int_2_x, *entropies, *scales;
        double *var_x;

        int_1_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        int_2_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        entropies = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        scales = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

        strred_integral_image(x_pad, r_width, r_height, int_1_x);
        strred_strred_integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
        strred_compute_entropy_scale(int_1_x, int_2_x, r_width + 1, r_height + 1, kw, kh, k_norm, entropy, scale, entr_const);

        free(int_1_x);
        free(int_2_x);
    }
}

void strred_strred_integral_image_2_temporal(const float* src1, const float* src2, size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            double val = (double)src1[(i - 1) * width + (j - 1)] - (double)src2[(i - 1) * width + (j - 1)];
            val = (double) val * val;

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

void strred_integral_image_temporal(const float* src1, const float* src2,  size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            double val = (double)src1[(i - 1) * width + (j - 1)] - (double)src2[(i - 1) * width + (j - 1)];

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

void rred_entropies_and_scales_temporal(const float* x, const float* prev_x, int block_size, size_t width, size_t height, size_t stride, double *entropy, double *scale)
{
    float sigma_nsq = STRRED_SIGMA_NSQ;
    float tol = 1e-10;

    float entr_const;

    if(block_size == 1)
    {
        float exp_val = EULERS_CONSTANT;
        entr_const = log(2 * M_PI * EULERS_CONSTANT);
        int k = STRRED_WINDOW_SIZE;
        int k_norm = k * k;
        int kw = k;
        int kh = k;
        int wd, ht;
        int i, j;

        float* x_pad;
        float* prev_x_pad;

        int x_reflect = (int)((kw - 1) / 2);
        int prev_x_reflect = (int)((kw - 1) / 2);
        x_pad = (float*) malloc (sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
        prev_x_pad = (float*) malloc (sizeof(float) * (width + (2 * prev_x_reflect)) * (height + (2 * prev_x_reflect)));

        strred_reflect_pad(x, width, height, x_reflect, x_pad);
        strred_reflect_pad(prev_x, width, height, prev_x_reflect, prev_x_pad);
        size_t r_width = width + (2 * x_reflect);
        size_t r_height = height + (2 * x_reflect);


        double *int_1_x, *int_2_x;
        double *var_x;

        int_1_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        int_2_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

        strred_integral_image_temporal(x_pad, prev_x_pad, r_width, r_height, int_1_x);
        strred_strred_integral_image_2_temporal(x_pad, prev_x_pad, r_width, r_height, int_2_x);
        strred_compute_entropy_scale(int_1_x, int_2_x, width, height, kw, kh, k_norm, entropy, scale, entr_const);

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

int compute_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist, size_t width, size_t height, 
                        double* srred_vals, double* trred_vals, double* strred_vals,
                        double* srred_approx_vals, double* trred_approx_vals, double* strred_approx_vals,
                        double* spat_vals, double* temp_vals, double* spat_temp_vals,
                        int k, int stride, double sigma_nsq_arg, int index)
{
#if 1
    // For frame 0. The code will not enter strred function in python and will copy the contents to prev pyr
    // Here insert an if condition for the same. If frame != 0 then enter the loop else just copy the contents into prev_frame
    // Need to pass frame number in the caller of the function
    int x_reflect = (int)((STRRED_WINDOW_SIZE - 1) / 2);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    int subband;
    int total_subbands = 4;

    static struct dwt2buffers *prev_ref = NULL;
    static struct dwt2buffers *prev_dist = NULL;
    int ret;

    // Pass frame index to the function argument 
    if (index == 0)
    {
        prev_ref = (struct dwt2buffers*)calloc(1, sizeof(struct dwt2buffers));
        prev_dist = (struct dwt2buffers*)calloc(1, sizeof(struct dwt2buffers));

        for(subband = 0; subband < total_subbands; subband++)
        {
            prev_ref->bands[subband] = (double*)calloc(width * height, sizeof(double));
            prev_dist->bands[subband] = (double*)calloc(width * height, sizeof(double));

            // Use memcpy to copy the contents of ref and dist to prev_ref and prev_dist
            memcpy(prev_ref->bands[subband], ref->bands[subband], width * height);
            memcpy(prev_dist->bands[subband], dist->bands[subband], width * height);
        }

        prev_ref->width = ref->width;
        prev_ref->height = ref->height;
        prev_dist->width = dist->width;
        prev_dist->height = dist->height;
    }
    else
    {
        int compute_temporal;

        // TODO: Insert an assert to check whether details ref and details dist are of same length
        int n_levels = sizeof(ref->bands) / sizeof(ref->bands[0]) - 1;

        double *entropies_ref = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        double *entropies_dist = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        double *scales_ref = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
        double *scales_dist = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

        double *spat_aggregate = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

        float ref_entropy, ref_scale, dist_entropy, dist_scale;
        double spat_temp_vals, spat_vals, temp_vals;
        double spat_mean;
        double check_val;

        for(subband = 1; subband < total_subbands; subband++)
        {
            spat_mean = 0;

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
                    spat_mean += fabs(spat_aggregate[i * r_width + j]);
                }
                spat_vals = spat_mean / (height * width);
            }
        }

        if(prev_ref != NULL && prev_dist != NULL)
        {
            struct dwt2buffers *ref_temporal = (struct dwt2buffers*)calloc((width) * (height), sizeof(struct dwt2buffers));
            struct dwt2buffers *dist_temporal = (struct dwt2buffers*)calloc((width) * (height), sizeof(struct dwt2buffers));
            double *temp_aggregate = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
            double temp_mean;

            for(subband = 1; subband < total_subbands; subband++)
            {
                subract_subbands(ref->bands[subband], prev_ref->bands[subband], ref_temporal->bands[subband], dist->bands[subband], prev_dist->bands[subband], dist_temporal->bands [subband], width, height);

                rred_entropies_and_scales(ref_temporal->bands[subband], BLOCK_SIZE, width, height, stride, entropies_ref, scales_ref);
                rred_entropies_and_scales(dist_temporal->bands[subband], BLOCK_SIZE, width, height, stride, entropies_dist, scales_dist);

                for (int i = 0; i < r_height; i++)
                {
                    for (int j = 0; j < r_width; j++)
                    {
                        temp_aggregate[i * r_width + j] = entropies_ref[i * r_width + j] * scales_ref[i * r_width + j] - entropies_dist[i * r_width + j] * scales_dist[i * r_width + j];
                    }
                }

                for (int i = 0; i < r_height; i++)
                {
                    for (int j = 0; j < r_width; j++)
                    {
                        temp_mean += fabs(temp_aggregate[i * r_width + j]);
                    }
                    temp_vals = temp_mean / (height * width);
                    check_val = temp_vals;
                }
            }
        }



        //compute_temporal = (prev_ref->bands[0] != 'NULL') ? 1 : 0;

        //if(compute_temporal)
        //{
        //    rred_entropies_and_scales_temporal(ref->bands[subband], prev_ref->bands[subband], BLOCK_SIZE, width, height, stride, &ref_entropy, &ref_scale);
        //    rred_entropies_and_scales_temporal(dist->bands[subband], prev_dist->bands[subband], BLOCK_SIZE, width, height, stride, &dist_entropy, &dist_scale);
        //}


        // Agrregate functions from Ajat
        // Call the Aggregator function/MACRO and pass entropies, scales to get spat, temp, spat-temporal values


        prev_ref = ref;
        prev_dist = dist;
    }

    ret = 0;

    return ret;
#else

    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = k * k;

    float* x_pad, * y_pad;

    int x_reflect = (int)((kh - stride) / 2);
    int y_reflect = (int)((kw - stride) / 2);

    x_pad = (float*)malloc(sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (float*)malloc(sizeof(float) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

    strred_reflect_pad(x, width, height, x_reflect, x_pad);
    strred_reflect_pad(y, width, height, y_reflect, y_pad);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    double* int_1_x, * int_1_y, * int_2_x, * int_2_y, * int_xy;
    double* var_x, * var_y, * cov_xy;

    int_1_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_1_y = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_2_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_2_y = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_xy = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

    strred_integral_image(x_pad, r_width, r_height, int_1_x);
    strred_integral_image(y_pad, r_width, r_height, int_1_y);
    strred_strred_integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
    strred_strred_integral_image_2(y_pad, y_pad, r_width, r_height, int_2_y);
    strred_strred_integral_image_2(x_pad, y_pad, r_width, r_height, int_xy);

    var_x = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));

    strred_compute_metrics(int_1_x, int_1_y, int_2_x, int_2_y, int_xy, r_width + 1, r_height + 1, kh, kw, k_norm, var_x, var_y, cov_xy);

    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;

    double* g = (double*)malloc(sizeof(double) * s_width * s_height);
    double* sv_sq = (double*)malloc(sizeof(double) * s_width * s_height);
    double exp = (double)1e-10;
    int index;
	
	double sigma_nsq = sigma_nsq_arg;
#if strred_STABILITY
	double sigma_nsq_base = sigma_nsq_arg / (255.0*255.0);	
	double sigma_max_inv = 4.0;
	sigma_nsq = sigma_nsq_base;
#if USE_DYNAMIC_SIGMA_NSQ
	sigma_nsq = sigma_nsq_base * (2 << (strred_level + 1));
#endif
#endif


    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;
	
	double num_val, den_val;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;

            g[index] = cov_xy[index] / (var_x[index] + exp);
            sv_sq[index] = var_y[index] - g[index] * cov_xy[index];

            if (var_x[index] < exp)
            {
                g[index] = (double)0;
                sv_sq[index] = var_y[index];
                var_x[index] = (double)0;
            }

            if (var_y[index] < exp)
            {
                g[index] = (double)0;
                sv_sq[index] = (double)0;
            }

            if (g[index] < 0)
            {
                sv_sq[index] = var_x[index];
                g[index] = (double)0;
            }

            if (sv_sq[index] < exp)
                sv_sq[index] = exp;

#if strred_STABILITY
			num_val = (log2((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)));
			den_val = (log2((double)1 + var_x[index] / sigma_nsq));

			if (cov_xy[index] < 0.0f) {
				num_val = 0.0f;
			}

			if (var_x[index] < sigma_nsq) {
				num_val = 1.0f - var_y[index] * sigma_max_inv;
				den_val = 1.0f;
			}
#else

			num_val = (log2((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)) + (double)1e-4);
			den_val = (log2((double)1 + var_x[index] / sigma_nsq) + (double)1e-4);

#endif

			*score_num += num_val;
			*score_den += den_val;
        }
    }

#if strred_STABILITY
	*score += ((*score_den) == 0.0) ? 1.0 : ((*score_num) / (*score_den));
#else
    *score += ((*score_num) / (*score_den));
#endif

    free(x_pad);
    free(y_pad);
    free(int_1_x);
    free(int_1_y);
    free(int_2_x);
    free(int_2_y);
    free(int_xy);
    free(var_x);
    free(var_y);
    free(cov_xy);
    free(g);
    free(sv_sq);
    ret = 0;

    return ret;
#endif
}