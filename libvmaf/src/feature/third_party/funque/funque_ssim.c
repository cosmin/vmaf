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

int compute_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, double *score, int max_val, float K1, float K2)
{
    //TODO: Assert checks to make sure src_ref, src_dist same in qty and nlevels = 1
    int ret = 1;

    *score = 0;

    int n_levels = 1;

    int width = ref->width;
    int height = ref->height;

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

const float exps[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};

int compute_ms_ssim_funque(dwt2buffers* ref, dwt2buffers* dist, MsSsimScore* score, int max_val,
                           float K1, float K2, int n_levels)
{
    int ret = 1;
    MsSsimScore ms_ssim_score;
    ms_ssim_score = *score;

    int width = ref->width;
    int height = ref->height;

    double C1 = (K1 * max_val) * (K1 * max_val);
    double C2 = (K2 * max_val) * (K2 * max_val);

    double* var_x_add = (double*) calloc(width * height, sizeof(double));
    double* var_y_add = (double*) calloc(width * height, sizeof(double));
    double* cov_xy_add = (double*) calloc(width * height, sizeof(double));

    double* var_x = (double*) calloc(width * height, sizeof(double));
    double* var_y = (double*) calloc(width * height, sizeof(double));
    double* cov_xy = (double*) calloc(width * height, sizeof(double));

    double* l_arr = (double*) calloc(width * height, sizeof(double));
    double* cs_arr = (double*) calloc(width * height, sizeof(double));
    double* ssim_arr = (double*) calloc(width * height, sizeof(double));

    double* var_x_cum = *(score->var_x_cum);
    double* var_y_cum = *(score->var_y_cum);
    double* cov_xy_cum = *(score->cov_xy_cum);

#if ENABLE_MINK3POOL
    float cube_1minus_map = 0;
#endif

    int win_dim = (1 << n_levels);          // 2^L
    int win_size = (1 << (n_levels << 1));  // 2^(2L), i.e., a win_dim X win_dim square

    double mx, my, l, cs;
    double sum = 0;
    double l_sum = 0;
    double cs_sum = 0;
    int index = 0;
    int index_cum = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            index = i * width + j;
            mx = ref->bands[0][index] / win_dim;
            my = dist->bands[0][index] / win_dim;

            for(int k = 1; k < 4; k++) {
                var_x_add[index] += ref->bands[k][index] * ref->bands[k][index];
                var_y_add[index] += dist->bands[k][index] * dist->bands[k][index];
                cov_xy_add[index] += ref->bands[k][index] * dist->bands[k][index];
            }

            var_x_cum[index] = var_x_cum[index_cum] + var_x_cum[index_cum + 1] +
                               var_x_cum[index_cum + (width * 2)] +
                               var_x_cum[index_cum + (width * 2) + 1] + var_x_add[index];
            var_y_cum[index] = var_y_cum[index_cum] + var_y_cum[index_cum + 1] +
                               var_y_cum[index_cum + (width * 2)] +
                               var_y_cum[index_cum + (width * 2) + 1] + var_y_add[index];
            cov_xy_cum[index] = cov_xy_cum[index_cum] + cov_xy_cum[index_cum + 1] +
                                cov_xy_cum[index_cum + (width * 2)] +
                                cov_xy_cum[index_cum + (width * 2) + 1] + cov_xy_add[index];

            var_x[index] = var_x_cum[index] / win_size;
            var_y[index] = var_y_cum[index] / win_size;
            cov_xy[index] = cov_xy_cum[index] / win_size;

            l_arr[index] = (2 * mx * my + C1) / ((mx * mx) + (my * my) + C1);
            cs_arr[index] = (2 * cov_xy[index] + C2) / (var_x[index] + var_y[index] + C2);
            ssim_arr[index] = l_arr[index] * cs_arr[index];
#if ENABLE_MINK3POOL
            cube_1minus_map += pow((1 - (l_arr[index] * cs_arr[index])), 3);
#else
            sum += (l_arr[index] * cs_arr[index]);
            l_sum += l_arr[index];
            cs_sum += cs_arr[index];
#endif
            index_cum += 2;
        }
        index_cum += (width * 2);
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt((double) cube_1minus_map / (width * height));
    score->ssim_mean = ssim_clip(ssim_val, 0, 1);
#else
    double ssim_mean = sum / (height * width);
    double l_mean = l_sum / (height * width);
    double cs_mean = cs_sum / (height * width);
    score->ssim_mean = ssim_mean;
    score->l_mean = l_mean;
    score->cs_mean = cs_mean;

    double l_var = 0;
    double cs_var = 0;
    double ssim_var = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            index = i * width + j;
            l_var += (l_arr[index] - score->l_mean) * (l_arr[index] - score->l_mean);
            cs_var += (cs_arr[index] - score->cs_mean) * (cs_arr[index] - score->cs_mean);
            ssim_var += (ssim_arr[index] - score->ssim_mean) * (ssim_arr[index] - score->ssim_mean);
        }
    }
    l_var = l_var / (height * width);
    cs_var = cs_var / (height * width);
    ssim_var = ssim_var / (height * width);

    double l_std = sqrt(l_var);
    double cs_std = sqrt(cs_var);
    double ssim_std = sqrt(ssim_var);

    double l_cov = l_std / l_mean;
    double cs_cov = cs_std / cs_mean;
    double ssim_cov = ssim_std / ssim_mean;

    score->ssim_cov = ssim_cov;
    score->l_cov = l_cov;
    score->cs_cov = cs_cov;
#endif

    free(var_x);
    free(var_y);
    free(cov_xy);
    free(var_x_add);
    free(var_y_add);
    free(cov_xy_add);
    free(l_arr);
    free(cs_arr);
    free(ssim_arr);

    ret = 0;

    return ret;
}

int compute_ms_ssim_mean_scales(MsSsimScore* score, int n_levels)
{
    int ret = 1;

    double cum_prod_mean[5] = {0};
    double cum_prod_concat_mean[5] = {0};
    double ms_ssim_mean_scales[5] = {0};

    double cum_prod_cov[5] = {0};
    double cum_prod_concat_cov[5] = {0};
    double ms_ssim_cov_scales[5] = {0};

    cum_prod_mean[0] = pow(score[0].cs_mean, exps[0]);
    cum_prod_cov[0] = pow(score[0].cs_cov, exps[0]);
    for(int i = 1; i < (n_levels - 1); i++) {
        cum_prod_mean[i] = cum_prod_mean[i - 1] + pow(score[i].cs_mean, exps[i]);
        cum_prod_cov[i] = cum_prod_cov[i - 1] + pow(score[i].cs_cov, exps[i]);
    }

    cum_prod_concat_mean[0] = 1;
    cum_prod_concat_cov[0] = 1;
    for(int i = 1; i < n_levels; i++) {
        cum_prod_concat_mean[i] = cum_prod_mean[i - 1];
        cum_prod_concat_cov[i] = cum_prod_cov[i - 1];
    }

    for(int i = 0; i < n_levels; i++) {
        score[i].ms_ssim_mean = cum_prod_concat_mean[i] * pow(score[i].ssim_mean, exps[i]);
        score[i].ms_ssim_cov = cum_prod_concat_cov[i] * pow(score[i].ssim_cov, exps[i]);
    }

    ret = 0;

    return ret;
}