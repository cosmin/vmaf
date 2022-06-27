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
#include <stdlib.h>
#include "integer_filters.h"

int integer_compute_ssim_funque(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, funque_dtype K1, funque_dtype K2, int pending_div)
{
    int ret = 1;

    size_t width = ref->width;
    size_t height = ref->height;

    /**
     * @brief 
     * C1 is constant is added to ref^2, dist^2, 
     *  - hence we have to multiply by pending_div^2
     * As per floating point,C1 is added to 2*(mx/win_dim)*(my/win_dim) & (mx/win_dim)*(mx/win_dim)+(my/win_dim)*(my/win_dim)
     * win_dim = 1 << n_levels, where n_levels = 1
     * Since win_dim division is avoided for mx & my, C1 is left shifted by 1
     */
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div*pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    /**
     * @brief 
     * shifts are handled similar to C1
     * not shifted left because the other terms to which this is added undergoes equivalent right shift 
     */
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div*pending_div) >> (SSIM_INTER_VAR_SHIFTS+SSIM_INTER_CS_SHIFT-2)));
    
    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_map_sq = 0;
    
    int index = 0;
    for (int i = 0; i < height; i++)
    {
        ssim_accum_dtype map_sq_insum = 0;
        for (int j = 0; j < width; j++)
        {
            index = i * width + j;

            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x  = 0;
            var_y  = 0;
            cov_xy = 0;

            for (int k = 1; k < 4; k++)
            {
                /**
                 * @brief 
                 * ref, dist in Q15 => (Q15*Q15)>>1 = Q29 
                 * num_bands = 3(for accumulation) => Q29+Q29+Q29 = Q31
                 */
                var_x  += ((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index])  >> SSIM_INTER_VAR_SHIFTS;
                var_y  += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]) >> SSIM_INTER_VAR_SHIFTS;
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]) >> SSIM_INTER_VAR_SHIFTS;
            }
            var_x_band0  = (ssim_inter_dtype)mx * mx;
            var_y_band0  = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;
            
            
            //l = (2*mx*my + C1) / (mx*mx + my*my + C1)
            // Splitting this into 2 variables l_num, l_den
            // l_num = (2*mx*my)>>1 + C1
            //This is because, 2*mx*my takes full 32 bits (mx holds 16bits-> 1sign 15bit for value)
            //After mul, mx*my takes 31bits including sign
            //Hence 2*mx*my takes full 32 bits, for addition with C1 right shifted by 1
            l_num = ((2>>SSIM_INTER_L_SHIFT)*cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0)>>SSIM_INTER_L_SHIFT) + C1);

            //cs = (2*cov_xy+C2)/(var_x+var_y+C2)
            //Similar to l, cs is split to cs_num cs_den
            //One extra shift is done here since more accumulation is happening across bands
            //Hence the extra left shift is avoided for C2 unlike C1
            cs_num = ((2>>SSIM_INTER_CS_SHIFT)*cov_xy+C2);
            cs_den = (((var_x+var_y)>>SSIM_INTER_CS_SHIFT)+C2);

            //right shift by SSIM_SHIFT_DIV for denominator to avoid precision loss
            //adding 1 to avoid divide by 0
            //This shift cancels when ssim_std / ssim_mean ratio is taken
            map = (ssim_inter_dtype) (((ssim_accum_dtype)l_num * cs_num) / (((ssim_accum_dtype)l_den * cs_den) >> SSIM_SHIFT_DIV + 1));
            accum_map += map;
            map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype) map * map)/width);
        }
        accum_map_sq += (ssim_accum_dtype) (map_sq_insum/height);
    }

    double ssim_mean = (double)accum_map / (height * width);


    double ssim_std; 
    ssim_std = sqrt((double) accum_map_sq - ssim_mean*ssim_mean);

    *score = (ssim_std / ssim_mean);

    ret = 0;

    return ret;
}