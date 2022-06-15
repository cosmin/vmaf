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
#include "integer_filters.h"

int compute_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, double *score, int max_val, funque_dtype K1, funque_dtype K2)
{
    //TODO: Assert checks to make sure src_ref, src_dist same in qty and nlevels = 1
    int ret = 1;

    *score = 0;

    int n_levels = 1;

    size_t width = ref->width;
    size_t height = ref->height;

    funque_dtype C1 = (K1 * max_val) * (K1 * max_val);
    funque_dtype C2 = (K2 * max_val) * (K2 * max_val);

   /* funque_dtype* mu_x = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* mu_y = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));*/
    funque_dtype* var_x = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* var_y = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* cov_xy = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));

    // memset(var_x, 0, width * height * sizeof(var_x[0]));
    // memset(var_y, 0, width * height * sizeof(var_y[0]));
    // memset(cov_xy, 0, width * height * sizeof(cov_xy[0]));

    //funque_dtype* l = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);
    //funque_dtype* cs = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);
    funque_dtype* map = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);

    int win_dim = 1 << n_levels;
    int win_size = (1 << (n_levels << 1));

    funque_dtype mx, my, l, cs;
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

            //TODO: Implemenet generic loop for n_levels > 1

            var_x[index] /= win_size;
            var_y[index] /= win_size;
            cov_xy[index] /= win_size;

            l = (2 * mx * my + C1) / ((mx * mx) + (my * my) + C1);
            cs = (2 * cov_xy[index] + C2) / (var_x[index] + var_y[index] + C2);
            map[index] = l * cs;
            sum += (l * cs);
        }
    }

    funque_dtype ssim_mean = sum / (height * width);
    funque_dtype sd = 0;
    for (int i = 0; i < (height * width); i++)
    {
        sd += pow(map[i] - ssim_mean, 2);
    }

    funque_dtype ssim_std = sqrt(sd / (height * width));

    /*if (strcmp(pool, "mean"))
        return ssim_mean;
    else if (strcmp(pool, "cov"))*/
    *score = (ssim_std / ssim_mean);

    free(var_x);
    free(var_y);
    free(cov_xy);
    free(map);

    ret = 0;

    return ret;
}