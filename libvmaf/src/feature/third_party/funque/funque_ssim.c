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
#include "funque_filters.h"
#include "funque_ssim_options.h"
#include <complex.h>
#include <stdio.h>
#include <assert.h>

int compute_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, double *score, int max_val, float K1, float K2)
{
    //TODO: Assert checks to make sure src_ref, src_dist same in qty and nlevels = 1
    int ret = 1;

    *score = 0;

    int n_levels = 1;

    int width = ref->crop_width;
    int height = ref->crop_height;

    float C1 = (K1 * max_val) * (K1 * max_val);
    float C2 = (K2 * max_val) * (K2 * max_val);

    float* var_x = (float*)calloc(width * height, sizeof(float));
    float* var_y = (float*)calloc(width * height, sizeof(float));
    float* cov_xy = (float*)calloc(width * height, sizeof(float));

#if ENABLE_MINK3POOL
    float cube_1minus_map = 0;
#endif

    int win_dim = 1 << n_levels;
    int win_size = (1 << (n_levels << 1));

    float mx, my, l, cs;
    double sum = 0;
    int index = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            index = i * width + j;
            mx = ref->bands[0][index] / win_dim;
            my = dist->bands[0][index] / win_dim;

            for (int k = 1; k < 4; k++)
            {
                var_x[index] += ref->bands[k][index] * ref->bands[k][index];
                var_y[index] += dist->bands[k][index] * dist->bands[k][index];
                cov_xy[index] += ref->bands[k][index] * dist->bands[k][index];
            }

            var_x[index] /= win_size;
            var_y[index] /= win_size;
            cov_xy[index] /= win_size;

            l = (2 * mx * my + C1) / ((mx * mx) + (my * my) + C1);
            cs = (2 * cov_xy[index] + C2) / (var_x[index] + var_y[index] + C2);
#if ENABLE_MINK3POOL
            cube_1minus_map += pow((1 - (l * cs)), 3);
#else
            sum += (l * cs);
#endif
        }
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt((double)cube_1minus_map/(width*height));
    *score = ssim_clip(ssim_val, 0, 1);
#else
    float ssim_mean = sum / (height * width);
    *score = ssim_mean;
#endif
    free(var_x);
    free(var_y);
    free(cov_xy);

    ret = 0;

    return ret;
}

const double exps[5] = {0.0448000000, 0.2856000000, 0.3001000000, 0.2363000000, 0.1333000000};

int compute_ms_ssim_funque(dwt2buffers* ref, dwt2buffers* dist, MsSsimScore* score, int max_val,
                           float K1, float K2, int n_levels)
{
    int ret = 1;

    int cum_array_width = (ref->crop_width) * (1 << n_levels);

    int width = ref->crop_width;
    int height = ref->crop_height;

    float C1 = (K1 * max_val) * (K1 * max_val);
    float C2 = (K2 * max_val) * (K2 * max_val);

    float var_x = 0;
    float var_y = 0;
    float cov_xy = 0;

    float* var_x_cum = *(score->var_x_cum);
    float* var_y_cum = *(score->var_y_cum);
    float* cov_xy_cum = *(score->cov_xy_cum);


    double cube_1minus_l = 0;
    double cube_1minus_cs = 0;
    double cube_1minus_map = 0;


    int win_dim = (1 << n_levels);          // 2^L
    int win_size = (1 << (n_levels << 1));  // 2^(2L), i.e., a win_dim X win_dim square

    double mx, my, l, cs;
    double ssim_sum = 0;
    double l_sum = 0;
    double cs_sum = 0;
    double ssim_sq_sum = 0;
    double l_sq_sum = 0;
    double cs_sq_sum = 0;
    int index = 0;
    int index_cum = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            index = i * width + j;
            mx = ref->bands[0][index] / win_dim;
            my = dist->bands[0][index] / win_dim;

            for(int k = 1; k < 4; k++) {
                var_x_cum[index_cum] += ((ref->bands[k][index] * ref->bands[k][index]) / win_size);
                var_y_cum[index_cum] += ((dist->bands[k][index] * dist->bands[k][index]) / win_size);
                cov_xy_cum[index_cum] += ((ref->bands[k][index] * dist->bands[k][index]) / win_size);
            }

            var_x = var_x_cum[index_cum];
            var_y = var_y_cum[index_cum];
            cov_xy = cov_xy_cum[index_cum];

            l = (2 * mx * my + C1) / ((mx * mx) + (my * my) + C1);
            cs = (2 * cov_xy + C2) / (var_x + var_y + C2);

            cube_1minus_l += pow((1 - l), 3);
            cube_1minus_cs += pow((1 - cs), 3);
            cube_1minus_map += pow((1 - (l * cs)), 3);

            ssim_sum += (l * cs);
            l_sum += l;
            cs_sum += cs;
            ssim_sq_sum += (l * cs) * (l * cs);
            l_sq_sum += l * l;
            cs_sq_sum += cs * cs;

            index_cum++;
        }
        index_cum += (cum_array_width - width);
    }


    double l_mink3 = 1 - (double)cbrt(cube_1minus_l / (width * height));
    double cs_mink3 = 1 - (double)cbrt(cube_1minus_cs / (width * height));
    double ssim_mink3 = 1 - (double)cbrt(cube_1minus_map / (width * height));
    score->ssim_mink3 = ssim_mink3;
    score->l_mink3 = l_mink3;
    score->cs_mink3 = cs_mink3;
    //score->ssim_mean = ssim_clip(ssim_val, 0, 1);

    double ssim_mean = ssim_sum / (height * width);
    double l_mean = l_sum / (height * width);
    double cs_mean = cs_sum / (height * width);
    score->ssim_mean = ssim_mean;
    score->l_mean = l_mean;
    score->cs_mean = cs_mean;

    double l_var = (l_sq_sum / (height * width)) - (l_mean * l_mean);
    double cs_var = (cs_sq_sum / (height * width)) - (cs_mean * cs_mean);
    double ssim_var = (ssim_sq_sum / (height * width)) - (ssim_mean * ssim_mean);

    assert(l_var >= 0);
    assert(cs_var >= 0);
    assert(ssim_var >= 0);

    double l_std = sqrt(l_var);
    double cs_std = sqrt(cs_var);
    double ssim_std = sqrt(ssim_var);

    double l_cov = l_std / l_mean;
    double cs_cov = cs_std / cs_mean;
    double ssim_cov = ssim_std / ssim_mean;

    score->ssim_cov = ssim_cov;
    score->l_cov = l_cov;
    score->cs_cov = cs_cov;


    ret = 0;

    return ret;
}

int compute_ms_ssim_mean_scales(MsSsimScore* score, int n_levels)
{
    int ret = 1;

    double cum_prod_mean[5] = {0};
    double cum_prod_concat_mean[5] = {0};

    double cum_prod_cov[5] = {0};
    double cum_prod_concat_cov[5] = {0};

    double cum_prod_mink3[5] = {0};
    double cum_prod_concat_mink3[5] = {0};

    double sign_cum_prod_mean = (score[0].cs_mean) >= 0 ? 1 : -1;  
    double sign_cum_prod_cov = (score[0].cs_cov) >= 0 ? 1 : -1;
    double sign_cum_prod_mink3 = (score[0].cs_mink3) >= 0 ? 1 : -1;

    cum_prod_mean[0] = pow(fabs(score[0].cs_mean), exps[0]) * sign_cum_prod_mean;
    cum_prod_cov[0] = pow(fabs(score[0].cs_cov), exps[0]) * sign_cum_prod_cov;
    cum_prod_mink3[0] = pow(fabs(score[0].cs_mink3), exps[0]) * sign_cum_prod_mink3;

    for(int i = 1; i < n_levels; i++) {
        sign_cum_prod_mean = (score[i].cs_mean) >= 0 ? 1 : -1;  
        sign_cum_prod_cov = (score[i].cs_cov) >= 0 ? 1 : -1;
        sign_cum_prod_mink3 = (score[i].cs_mink3) >= 0 ? 1 : -1;

        cum_prod_mean[i] = cum_prod_mean[i-1] * pow(fabs(score[i].cs_mean), exps[i]) * sign_cum_prod_mean;
        cum_prod_cov[i] = cum_prod_cov[i - 1] * pow(fabs(score[i].cs_cov), exps[i]) * sign_cum_prod_cov;
        cum_prod_mink3[i] = cum_prod_mink3[i - 1] * pow(fabs(score[i].cs_mink3), exps[i]) * sign_cum_prod_mink3;        
    }

    cum_prod_concat_mean[0] = 1;
    cum_prod_concat_cov[0] = 1;
    cum_prod_concat_mink3[0] = 1;
    for(int i = 1; i < n_levels; i++) {
        cum_prod_concat_mean[i] = cum_prod_mean[i - 1];
        cum_prod_concat_cov[i] = cum_prod_cov[i - 1];
        cum_prod_concat_mink3[i] = cum_prod_mink3[i - 1];
    }

    for(int i = 0; i < n_levels; i++) {
        
        float sign_mssim_mean = (score[i].ssim_mean) >= 0 ? 1 : -1;  
        float sign_mssim_cov = (score[i].ssim_cov) >= 0 ? 1 : -1;
        float sign_mssim_mink3 = (score[i].ssim_mink3) >= 0 ? 1 : -1;
        score[i].ms_ssim_mean = cum_prod_concat_mean[i] * pow(fabs(score[i].ssim_mean), exps[i]) * sign_mssim_mean;
        score[i].ms_ssim_cov = cum_prod_concat_cov[i] * pow(fabs(score[i].ssim_cov), exps[i])* sign_mssim_cov;
        score[i].ms_ssim_mink3 = cum_prod_concat_mink3[i] * pow(fabs(score[i].ssim_mink3), exps[i])* sign_mssim_mink3;
        
    }

    ret = 0;

    return ret;
}