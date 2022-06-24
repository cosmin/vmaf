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

int integer_compute_ssim_funque(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, funque_dtype K1, funque_dtype K2, int pending_div)
{
    //TODO: Assert checks to make sure src_ref, src_dist same in qty and nlevels = 1
    int ret = 1;

    *score = 0;

    int n_levels = 1;

    size_t width = ref->width;
    size_t height = ref->height;
    int win_dim = 1 << n_levels;
    int win_size = (1 << (n_levels << 1));
    int shift_win_size =  1 << n_levels;

    funque_dtype f_C1 = (K1 * max_val) * (K1 * max_val);
    funque_dtype f_C2 = (K2 * max_val) * (K2 * max_val);
    //pending_div is remaining for ref, dist
    //and this constant is added to ref^2, dist^2
    //hence we have to multiply by pending_div^2
    //C1 is added to 2*mx*my & mx*mx+my*my -> They are shifted by 1 to make them 31 bits
    //Hence C1 is shifted right by 1
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div*pending_div) << 1));
    //but ref^2, dist^2 are shifted right by shift_win_size
    //hence ((pending_div * pending_div) >> shift_win_size) for C2, since it is added to ref^2 & dist^2
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div*pending_div)));// >> shift_win_size));
   /* funque_dtype* mu_x = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* mu_y = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));*/
    funque_dtype* f_var_x = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* f_var_y = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));
    funque_dtype* f_cov_xy = (funque_dtype*)calloc(width * height, sizeof(funque_dtype));

    ssim_inter_dtype *var_x  = (ssim_inter_dtype*) calloc(width * height, sizeof(ssim_inter_dtype));
    ssim_inter_dtype *var_y  = (ssim_inter_dtype*) calloc(width * height, sizeof(ssim_inter_dtype));
    ssim_inter_dtype *cov_xy = (ssim_inter_dtype*) calloc(width * height, sizeof(ssim_inter_dtype));

    // memset(var_x, 0, width * height * sizeof(var_x[0]));
    // memset(var_y, 0, width * height * sizeof(var_y[0]));
    // memset(cov_xy, 0, width * height * sizeof(cov_xy[0]));

    // funque_dtype* f_l = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);
    // funque_dtype* f_cs = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);
    funque_dtype* map = (funque_dtype*)malloc(sizeof(funque_dtype) * width * height);

    
    funque_dtype f_mx, f_my, f_l, f_cs;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;
    float l, cs;
    double sum = 0;
    int index = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            index = i * width + j;
            f_mx = (float)ref->bands[0][index] / (win_dim * pending_div);
            f_my = (float)dist->bands[0][index] / (win_dim * pending_div);
            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            for (int k = 1; k < 4; k++)
            {
                f_var_x[index] += ((float)ref->bands[k][index]/pending_div) * ((float)ref->bands[k][index]/pending_div);
                f_var_y[index] += ((float)dist->bands[k][index]/pending_div) * ((float)dist->bands[k][index]/pending_div);
                f_cov_xy[index] += ((float)ref->bands[k][index]/pending_div) * ((float)dist->bands[k][index]/pending_div);

                //If precision loss is seen, here shift can be done by 1 & 1 more shift after accumulation completes
                var_x[index] += ((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index])  >> 1;
                var_y[index] += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]) >> 1;
                cov_xy[index]+= ((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]) >> 1;
            }
            var_x_band0  = (ssim_inter_dtype)mx * mx;
            var_y_band0  = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;
            //TODO: Implemenet generic loop for n_levels > 1

            f_var_x[index] /= win_size;
            f_var_y[index] /= win_size;
            f_cov_xy[index] /= win_size;
            
            
            //l = (2*mx*my + C1) / (mx*mx + my*my + C1)
            // Splitting this into 2 variables l_num, l_den
            // l_num = (2*mx*my)>>shift_win_size + C1
            //This is because, 2*mx*my takes full 32 bits (mx holds 16bits-> 1sign 15bit for value)
            //After mul, mx*my takes 31bits including sign
            //Hence 2*mx*my takes full 32 bits, for addition with C1 right shifted by 1
            l_num = cov_xy_band0 + C1;
            l_den = ((var_x_band0 + var_y_band0)>>1) + C1;

            // l_num >>= 16;
            // l_den >>= 16;
            //cs = (2*cov_xy[index]+C2)/(var_x[index]+var_y[index]+C2)
            //Similar to l, cs is split to cs_num cs_den
            //Shifts are alrady done for var_x, var_y, cov_xy
            cs_num = (cov_xy[index]+C2);
            cs_den = (((var_x[index]+var_y[index])>>1)+C2);
            // cs_num >>= 16;
            // cs_den >>= 16;
            f_l = (2*f_mx*f_my+f_C1) / (f_mx*f_mx + f_my*f_my + f_C1);
            f_cs = (2 * f_cov_xy[index] + f_C2) / (f_var_x[index] + f_var_y[index] + f_C2);
            // l = (2 * mx * my + C1) / ((mx * mx) + (my * my) + C1);
            // cs = (2 * cov_xy[index] + C2) / (var_x[index] + var_y[index] + C2);
            // l = ((float) l_num) / l_den;
            // cs = ((float) cs_num) / cs_den;
            // l = f_l;
            // cs = f_cs;
            l= (float)l_num/l_den;
            cs = (float) cs_num/cs_den;
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