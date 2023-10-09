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

int compute_ms_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, MsSsimScore *score, int max_val, float K1, float K2, int n_levels)
{
    //TODO: Assert checks to make sure src_ref, src_dist same in qty and nlevels = 1
    int ret = 1;
    MsSsimScore ms_ssim_score;
    ms_ssim_score = *score;

    float exps[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};

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

    int win_dim = (1 << n_levels);  // 2^L
    int win_size = (1 << (n_levels << 1));  // 2^(2L), i.e., a win_dim X win_dim square

    float mx, my, l, cs;
    double sum = 0;
    double l_sum = 0;
    double cs_sum = 0;
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
            l_sum += l;
            cs_sum += cs;
#endif
        }
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt((double)cube_1minus_map/(width*height));
    ms_ssim_score.ssim_mean = ssim_clip(ssim_val, 0, 1);
#else
    float ssim_mean = sum / (height * width);
    float l_mean = l_sum / (height * width);
    float cs_mean = cs_sum / (height * width);
    ms_ssim_score.ssim_mean = ssim_mean;
    ms_ssim_score.l_mean = l_mean;
    ms_ssim_score.cs_mean = cs_mean;
#endif
    free(var_x);
    free(var_y);
    free(cov_xy);

    ret = 0;

    return ret;

}
