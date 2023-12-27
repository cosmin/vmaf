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
#include <assert.h>
#include "integer_funque_filters.h"
#include "integer_funque_ssim.h"
#include "funque_ssim_options.h"

static inline int16_t get_best_i16_from_u64(uint64_t temp, int *power)
{
    //assert(temp >= 0x20000);
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t) temp;
}

static inline int16_t get_best_i16_from_u32(uint32_t temp, int *power)
{
    assert(temp >= 0x20000);
    int k = __builtin_clz(temp);
    k = 17 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t) temp;
}

static inline int16_t ms_ssim_get_best_i16_from_u32(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);

    if(k > 17) {
        k -= 17;
        //temp = temp << k;
        *x = 0;

    } else if(k < 16) {
        k = 17 - k;
        temp = temp >> k;
        *x = k;
    } else {
        *x = 0;
        if(temp >> 15) {
            temp = temp >> 1;
            *x = 1;
        }
    }

    return (int16_t) temp;
}

const double int_exps[5] = {0.0448000000, 0.2856000000, 0.3001000000, 0.2363000000, 0.1333000000};

int integer_compute_ssim_funque(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;

    int width = ref->crop_width;
    int height = ref->crop_height;

    /**
     * C1 is constant is added to ref^2, dist^2, 
     *  - hence we have to multiply by pending_div^2
     * As per floating point,C1 is added to 2*(mx/win_dim)*(my/win_dim) & (mx/win_dim)*(mx/win_dim)+(my/win_dim)*(my/win_dim)
     * win_dim = 1 << n_levels, where n_levels = 1
     * Since win_dim division is avoided for mx & my, C1 is left shifted by 1
     */
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div*pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    /**
     * shifts are handled similar to C1
     * not shifted left because the other terms to which this is added undergoes equivalent right shift 
     */
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div*pending_div) << (2 - SSIM_INTER_VAR_SHIFTS+SSIM_INTER_CS_SHIFT)));
    
    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map;
    ssim_accum_dtype map_num;
    ssim_accum_dtype map_den;
    int16_t i16_map_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

#if ENABLE_MINK3POOL
    ssim_accum_dtype rowcube_1minus_map = 0;
    double accumcube_1minus_map = 0;
    const ssim_inter_dtype const_1 = 32768;  //div_Q_factor>>SSIM_SHIFT_DIV
#else
    ssim_accum_dtype accum_map = 0;
#endif

    int index = 0;
    for (int i = 0; i < height; i++)
    {
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
                var_x  += ((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index]);
                var_y  += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]);
            }
            var_x_band0  = (ssim_inter_dtype)mx * mx;
            var_y_band0  = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;

            /**
             * ref, dist in Q15 with range [-23875, 23875]
             * num_bands = 3(for accumulation) where max value requires 31 bits
             * Since additions are present in next stage right shifted by SSIM_INTER_VAR_SHIFTS
            */
            var_x  = (var_x  >> SSIM_INTER_VAR_SHIFTS);
            var_y  = (var_y  >> SSIM_INTER_VAR_SHIFTS);
            cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

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
            
            map_num = (ssim_accum_dtype)l_num * cs_num;
            map_den = (ssim_accum_dtype)l_den * cs_den;
            
            /**
             * l_den & cs_den are variance terms, hence they will always be +ve 
             * getting best 15bits and retaining one signed bit, using get_best_i16_from_u64
             * This is done to reuse ADM division LUT, which has LUT for values from -2^15 to 2^15
            */
            int power_val;
            i16_map_den = get_best_i16_from_u64((uint64_t) map_den, &power_val);
            /**
             * The actual equation of map is map_num/map_den
             * The division is done using LUT, results of div_lookup = 2^30/i16_map_den
             * map = map_num/map_den => map = map_num / (i16_map_den << power_val)
             * => map = (map_num >> power_val) / i16_map_den
             * => map = (map_num >> power_val) * (div_lookup[i16_map_den + 32768] >> 30) //since it has -ve vals in 1st half
             * => map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> 30
             * Shift by 30 might be very high even for 32 bits precision, hence shift only by 15 
            */
            map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;

#if ENABLE_MINK3POOL
            ssim_accum_dtype const1_minus_map = const_1 - map;
            rowcube_1minus_map += const1_minus_map * const1_minus_map * const1_minus_map;
#else
            accum_map += map;
#endif
        }
#if ENABLE_MINK3POOL
        accumcube_1minus_map += (double) rowcube_1minus_map;
        rowcube_1minus_map = 0;
#endif
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt(accumcube_1minus_map/(width*height))/const_1;
    *score = ssim_clip(ssim_val, 0, 1);
#else
    *score = (double)accum_map / (height * width)/ (1 << SSIM_SHIFT_DIV);
#endif
    ret = 0;

    return ret;
}

int integer_compute_ms_ssim_funque(i_dwt2buffers *ref, i_dwt2buffers *dist, MsSsimScore_int *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup, int n_levels, int is_pyr)
{
    int ret = 1;

    int cum_array_width = (ref->crop_width) * (1 << n_levels);
    //int win_dim = (1 << n_levels);          // 2^L
    int win_size = (n_levels << 1); 
    int win_size_c2 = win_size;
    pending_div = pending_div >> (n_levels -1);
    int pending_div_c1 = pending_div;
    int pending_div_c2 = pending_div;
    int pending_div_offset = 0;
    int pending_div_halfround = 0;
    int width = ref->crop_width;
    int height = ref->crop_height;

    int32_t* var_x_cum = *(score->var_x_cum);
    int32_t* var_y_cum = *(score->var_y_cum);
    int32_t* cov_xy_cum = *(score->cov_xy_cum);

    if (is_pyr)
    {
        win_size_c2 = 2;
        pending_div_c1 = (1<<i_nadenau_pending_div_factors[n_levels-1][0]) * 255;
        pending_div_c2 = (1<<(i_nadenau_pending_div_factors[n_levels-1][1] + (n_levels - 1))) * 255;
        pending_div_offset = 2 * (i_nadenau_pending_div_factors[n_levels-1][3] - i_nadenau_pending_div_factors[n_levels-1][1]);
        pending_div_halfround = (pending_div_offset == 0) ? 0 : (1 << (pending_div_offset-1));
        if ((n_levels > 1))
        {
            int index_cum = 0;
            int shift_cums = 2 * (i_nadenau_pending_div_factors[n_levels-2][1] - i_nadenau_pending_div_factors[n_levels-1][1] - 1);
            for (int i = 0; i < height; i++)
            {   
                for (int j = 0; j < width; j++)
                {
                    
                    var_x_cum[index_cum] = (var_x_cum[index_cum] + (1<<(shift_cums-1))) >> shift_cums;
                    var_y_cum[index_cum] = (var_y_cum[index_cum] + (1<<(shift_cums-1))) >> shift_cums;
                    cov_xy_cum[index_cum] = (cov_xy_cum[index_cum] + (1<<(shift_cums-1))) >> shift_cums;
                    index_cum++;
                }
                index_cum += (cum_array_width - width);
            }
        }
    }

    int64_t c1_mul = (((int64_t) pending_div_c1*pending_div_c1) >> (SSIM_INTER_L_SHIFT));
    int64_t c2_mul = (((int64_t) pending_div_c2*pending_div_c2) >> (SSIM_INTER_VAR_SHIFTS+SSIM_INTER_CS_SHIFT));

    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * c1_mul);

    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * c2_mul);

    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map, l, cs;
    int16_t i16_l_den;
    int16_t i16_cs_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_l = 0;
    ssim_accum_dtype accum_cs = 0;
    ssim_accum_dtype accum_sq_map = 0;
    ssim_accum_dtype accum_sq_l = 0;
    ssim_accum_dtype accum_sq_cs = 0;
    ssim_accum_dtype map_sq = 0;

    ssim_inter_dtype mink3_const = 32768; // 2^15
    //ssim_inter_dtype mink3_const = pending_div * pending_div;
    ssim_inter_dtype mink3_const_map = (mink3_const * mink3_const) >> SSIM_R_SHIFT;
    ssim_inter_dtype mink3_const_l = mink3_const >> L_R_SHIFT;
    ssim_inter_dtype mink3_const_cs = mink3_const >> CS_R_SHIFT;

    ssim_inter_dtype map_r_shift = 0;
    ssim_inter_dtype l_r_shift = 0;
    ssim_inter_dtype cs_r_shift = 0;
    ssim_mink3_accum_dtype mink3_map = 0;
    ssim_mink3_accum_dtype mink3_l = 0;
    ssim_mink3_accum_dtype mink3_cs = 0;
    ssim_mink3_accum_dtype accum_mink3_map = 0;
    ssim_mink3_accum_dtype accum_mink3_l = 0;
    ssim_mink3_accum_dtype accum_mink3_cs = 0;

    int index = 0;
    int index_cum = 0;
    for(int i = 0; i < height; i++) {
        ssim_accum_dtype row_accum_sq_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_l = 0;
        ssim_mink3_accum_dtype row_accum_mink3_cs = 0;
        for(int j = 0; j < width; j++) {
            index = i * width + j;

            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;
            int k;
#if BAND_HVD_SAME_PENDING_DIV
            for (k = 1; k < 4; k++)
#else
            for (k = 1; k < 3; k++)
#endif
            {
                var_x  += ((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index]);
                var_y  += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]);
            }
#if !(BAND_HVD_SAME_PENDING_DIV)
            //The extra right shift will be done for pyr since the upscale factors are different for subbands
            var_x  += (((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index]) + pending_div_halfround) >> pending_div_offset;
            var_y  += (((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]) + pending_div_halfround) >> pending_div_offset;
            cov_xy += (((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]) + pending_div_halfround) >> pending_div_offset;
#endif            
            var_x_band0  = ((ssim_inter_dtype)mx * mx) >> win_size;
            var_y_band0  = ((ssim_inter_dtype)my * my) >> win_size;
            cov_xy_band0 = ((ssim_inter_dtype)mx * my) >> win_size;

            //var_x = (var_x >> SSIM_INTER_VAR_SHIFTS);
            //var_y = (var_y >> SSIM_INTER_VAR_SHIFTS);
            //cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

            if (!is_pyr)
            {
                var_x_cum[index_cum] = var_x_cum[index_cum] >> 2;
                var_y_cum[index_cum] = var_y_cum[index_cum] >> 2;
                cov_xy_cum[index_cum] = cov_xy_cum[index_cum] >> 2;
            }

            var_x_cum[index_cum] += (var_x >> win_size_c2);
            var_y_cum[index_cum] += (var_y >> win_size_c2);
            cov_xy_cum[index_cum] += (cov_xy >> win_size_c2);

            var_x = var_x_cum[index_cum];
            var_y = var_y_cum[index_cum];
            cov_xy = cov_xy_cum[index_cum];

            l_num = ((2 >> SSIM_INTER_L_SHIFT) * cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);

            cs_num = ((2 >> SSIM_INTER_CS_SHIFT) * cov_xy + C2);
            cs_den = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            int power_val_l;
            i16_l_den = ms_ssim_get_best_i16_from_u32((uint32_t) l_den, &power_val_l);

            int power_val_cs;
            i16_cs_den = ms_ssim_get_best_i16_from_u32((uint32_t) cs_den, &power_val_cs);

            /**
             * The actual equation of map is map_num/map_den
             * The division is done using LUT, results of div_lookup = 2^30/i16_map_den
             * map = map_num/map_den => map = map_num / (i16_map_den << power_val)
             * => map = (map_num >> power_val) / i16_map_den
             * => map = (map_num >> power_val) * (div_lookup[i16_map_den + 32768] >> 30) //since it
             * has -ve vals in 1st half
             * => map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> 30
             * Shift by 30 might be very high even for 32 bits precision, hence shift only by 15
             */

            l = ((l_num >> power_val_l) * div_lookup[i16_l_den + 32768]) >> SSIM_SHIFT_DIV;
            cs = ((cs_num >> power_val_cs) * div_lookup[i16_cs_den + 32768]) >> SSIM_SHIFT_DIV;
            // map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;
            map = l * cs;

            accum_l += l;
            accum_cs += cs;
            accum_map += map;
            accum_sq_l += (l * l);
            accum_sq_cs += (cs * cs);
            map_sq = ((int64_t) map * map) >> SSIM_SQ_ROW_SHIFT;
            row_accum_sq_map += map_sq;


            l_r_shift = l >> L_R_SHIFT;
            cs_r_shift = cs >> CS_R_SHIFT;
            map_r_shift = map >> SSIM_R_SHIFT;

            mink3_l = pow((mink3_const_l - l_r_shift), 3);
            mink3_cs = pow((mink3_const_cs - cs_r_shift), 3);
            mink3_map = pow((mink3_const_map - map_r_shift), 3);

            row_accum_mink3_l += mink3_l;
            row_accum_mink3_cs += mink3_cs;
            row_accum_mink3_map += mink3_map;

            index_cum++;
        }
        accum_sq_map += (row_accum_sq_map >> SSIM_SQ_COL_SHIFT);

        accum_mink3_l += (row_accum_mink3_l >> L_MINK3_ROW_R_SHIFT);
        accum_mink3_cs += (row_accum_mink3_cs >> CS_MINK3_ROW_R_SHIFT);
        accum_mink3_map += (row_accum_mink3_map >> SSIM_MINK3_ROW_R_SHIFT);

        index_cum += (cum_array_width - width);
    }

    double l_mean = (double) accum_l / (height * width);
    double cs_mean = (double) accum_cs / (height * width);
    double ssim_mean = (double) accum_map / (height * width);

    double l_var = ((double) accum_sq_l / (height * width)) - (l_mean * l_mean);
    double cs_var = ((double) accum_sq_cs / (height * width)) - (cs_mean * cs_mean);
    double inter_shift_sq = 1 << (SSIM_SQ_ROW_SHIFT + SSIM_SQ_COL_SHIFT);
    double ssim_var =
        (((double) accum_sq_map / (height * width)) * inter_shift_sq) - ((ssim_mean * ssim_mean));

    double l_std = sqrt(l_var);
    double cs_std = sqrt(cs_var);
    double ssim_std = sqrt(ssim_var);

    double l_cov = l_std / l_mean;
    double cs_cov = cs_std / cs_mean;
    double ssim_cov = ssim_std / ssim_mean;

    double mink3_cbrt_const_l = pow(2,(39/3));
    double mink3_cbrt_const_cs = pow(2,(38.0/3));
    double mink3_cbrt_const_map = pow(2,(38.0/3));

    double l_mink3 = mink3_cbrt_const_l - (double)cbrt(accum_mink3_l / (width * height));
    double cs_mink3 = mink3_cbrt_const_cs - (double)cbrt(accum_mink3_cs / (width * height));
    double ssim_mink3 = mink3_cbrt_const_map - (double)cbrt(accum_mink3_map / (width * height));

    score->ssim_mean = ssim_mean / (1 << (SSIM_SHIFT_DIV * 2));
    score->l_mean = l_mean / (1 << SSIM_SHIFT_DIV);
    score->cs_mean = cs_mean / (1 << SSIM_SHIFT_DIV);
    score->ssim_cov = ssim_cov;
    score->l_cov = l_cov;
    score->cs_cov = cs_cov;
    score->l_mink3 = l_mink3 / pow(2,(39/3));
    score->cs_mink3 = cs_mink3 / pow(2,(38.0/3));
    score->ssim_mink3 = ssim_mink3 / pow(2,(38.0/3));

    ret = 0;

    return ret;
}

int integer_mean_2x2_ms_ssim_funque_c(int32_t *var_x_cum, int32_t *var_y_cum, int32_t *cov_xy_cum, 
                                    int width, int height, int level)
{
    int ret = 1;

    int index = 0;
    int index_cum = 0;
    int cum_array_width = (width) * (1 << (level + 1));

    for(int i = 0; i < (height / 2); i++) {
        for(int j = 0; j < (width / 2); j++) {
            /* Accumulations are done using mean computations of 2x2 pixels */
            index = i * cum_array_width + j;
            var_x_cum[index] = var_x_cum[index_cum] + var_x_cum[index_cum + 1] +
                               var_x_cum[index_cum + (cum_array_width)] +
                               var_x_cum[index_cum + (cum_array_width) + 1];
            var_x_cum[index] = (var_x_cum[index] + 2) >> 2;

            var_y_cum[index] = var_y_cum[index_cum] + var_y_cum[index_cum + 1] +
                               var_y_cum[index_cum + (cum_array_width)] +
                               var_y_cum[index_cum + (cum_array_width) + 1];
            var_y_cum[index] = (var_y_cum[index] + 2) >> 2;

            cov_xy_cum[index] = cov_xy_cum[index_cum] + cov_xy_cum[index_cum + 1] +
                                cov_xy_cum[index_cum + (cum_array_width)] +
                                cov_xy_cum[index_cum + (cum_array_width) + 1];
            cov_xy_cum[index] = (cov_xy_cum[index] + 2) >> 2;

            index_cum += 2;
        }
        index_cum += ((cum_array_width * 2) - width);
    }

    ret = 0;

    return ret;
}

int integer_compute_ms_ssim_mean_scales(MsSsimScore_int *score, int n_levels)
{
    int ret = 1;

    double cum_prod_mean[5] = {0};
    double cum_prod_concat_mean[5] = {0};

    double cum_prod_cov[5] = {0};
    double cum_prod_concat_cov[5] = {0};

    double cum_prod_mink3[5] = {0};
    double cum_prod_concat_mink3[5] = {0};
    double ms_ssim_mink3_scales[5] = {0};

    float sign_cum_prod_mean = (score[0].cs_mean) >= 0 ? 1 : -1;  
    float sign_cum_prod_cov = (score[0].cs_cov) >= 0 ? 1 : -1;
    float sign_cum_prod_mink3 = (score[0].cs_mink3) >= 0 ? 1 : -1;

    cum_prod_mean[0] = pow(fabs(score[0].cs_mean), int_exps[0]) * sign_cum_prod_mean;
    cum_prod_cov[0] = pow(fabs(score[0].cs_cov), int_exps[0]) * sign_cum_prod_cov;
    cum_prod_mink3[0] = pow(fabs(score[0].cs_mink3), int_exps[0]) * sign_cum_prod_mink3;
    for(int i = 1; i < n_levels; i++) {
        sign_cum_prod_mean = (score[i].cs_mean) >= 0 ? 1 : -1;  
        sign_cum_prod_cov = (score[i].cs_cov) >= 0 ? 1 : -1;
        sign_cum_prod_mink3 = (score[i].cs_mink3) >= 0 ? 1 : -1;

        cum_prod_mean[i] = cum_prod_mean[i-1] * pow(fabs(score[i].cs_mean), int_exps[i]) * sign_cum_prod_mean;
        cum_prod_cov[i] = cum_prod_cov[i - 1] * pow(fabs(score[i].cs_cov), int_exps[i]) * sign_cum_prod_cov;
        cum_prod_mink3[i] = cum_prod_mink3[i - 1] * pow(fabs(score[i].cs_mink3), int_exps[i]) * sign_cum_prod_mink3;
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
        
        score[i].ms_ssim_mean = cum_prod_concat_mean[i] * pow(fabs(score[i].ssim_mean), int_exps[i]) * sign_mssim_mean;
        score[i].ms_ssim_cov = cum_prod_concat_cov[i] * pow(fabs(score[i].ssim_cov), int_exps[i])* sign_mssim_cov;
        
        score[i].ms_ssim_mink3 = cum_prod_concat_mink3[i] * pow(fabs(score[i].ssim_mink3), int_exps[i])* sign_mssim_mink3;
    }

    ret = 0;

    return ret;
}