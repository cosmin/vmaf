
#include "../funque_ssim_options.h"

#include "../integer_funque_filters.h"
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>

// #define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline int16_t ssim_get_best_i16_from_u64(uint64_t temp, int *power)
{
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t)temp;
}

static inline int16_t ms_ssim_get_best_i16_from_u32_neon(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);

    if(k > 17) {
        k -= 17;
        // temp = temp << k;
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

int integer_compute_ssim_funque_neon(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;
    int width = ref->width;
    int height = ref->height;

    dwt2_dtype mx, my;
    ssim_inter_dtype var_x, var_y, cov_xy, var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype map, l_num, l_den, cs_num, cs_den;
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div * pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div * pending_div) << (2 - SSIM_INTER_VAR_SHIFTS + SSIM_INTER_CS_SHIFT)));

#if ENABLE_MINK3POOL
    ssim_accum_dtype rowcube_1minus_map = 0;
    double accumcube_1minus_map = 0;
    const ssim_inter_dtype const_1 = 32768;  //div_Q_factor>>SSIM_SHIFT_DIV
#else
    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_map_sq = 0;
    ssim_accum_dtype map_sq_insum = 0;
#endif

    int index = 0, i, j, k;
    int16_t i16_map_den;

    int16x8_t ref16x8_b0, dist16x8_b0, ref16x8_b1, dist16x8_b1;
    int16x8_t ref16x8_b2, dist16x8_b2, ref16x8_b3, dist16x8_b3;
    int16x4_t tmpRef16x8_low, tmpDist16x8_low;
    int32x4_t var32x4_lb0, var32x4_hb0, cov32x4_lb0, cov32x4_hb0;
    int32x4_t lDen32x4_lb0, lDen32x4_hb0, csDen32x4_lb1, csDen32x4_hb1;
    int32x4_t addlNum32x4_lb0, addlNum32x4_hb0, addlDen32x4_lb0, addlDen32x4_hb0;
    int32x4_t varSftX32x4_lb1, varSftX32x4_hb1, varSftY32x4_lb1, varSftY32x4_hb1;
    int32x4_t varSft32x4_lb1, varSft32x4_hb1, covSft32x4_lb1, covSft32x4_hb1;
    int32x4_t addcsNum32x4_lb1, addcsNum32x4_hb1, addcsDen32x4_lb1, addcsDen32x4_hb1;
    int64x2_t mulNum64x2_0lo, mulNum64x2_0hi, mulNum64x2_1lo, mulNum64x2_1hi, mulDen64x2_0lo, mulDen64x2_0hi, mulDen64x2_1lo, mulDen64x2_1hi;
    int32x4_t varX32x4_lb1, varX32x4_hb1, varY32x4_lb1, varY32x4_hb1, cov32x4_lb1, cov32x4_hb1;
    int32x4_t dupC1 = vdupq_n_s32(C1);
    int32x4_t dupC2 = vdupq_n_s32(C2);

    int64_t *numVal = (int64_t *)malloc(width * sizeof(int64_t *));
    int64_t *denVal = (int64_t *)malloc(width * sizeof(int64_t *));

    for (i = 0; i < height; i++)
    {
        for (j = 0; j <= width - 8; j += 8)
        {
            index = i * width + j;
            ref16x8_b0 = vld1q_s16(ref->bands[0] + index);
            dist16x8_b0 = vld1q_s16(dist->bands[0] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b0);
            tmpDist16x8_low = vget_low_s16(dist16x8_b0);

            var32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            var32x4_hb0 = vmull_high_s16(ref16x8_b0, ref16x8_b0);
            var32x4_lb0 = vmlal_s16(var32x4_lb0, tmpDist16x8_low, tmpDist16x8_low);
            var32x4_hb0 = vmlal_high_s16(var32x4_hb0, dist16x8_b0, dist16x8_b0);
            cov32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb0 = vmull_high_s16(ref16x8_b0, dist16x8_b0);

            lDen32x4_lb0 = vshrq_n_s32(var32x4_lb0, SSIM_INTER_L_SHIFT);
            lDen32x4_hb0 = vshrq_n_s32(var32x4_hb0, SSIM_INTER_L_SHIFT);
            cov32x4_lb0 = vaddq_s32(cov32x4_lb0, cov32x4_lb0);
            cov32x4_hb0 = vaddq_s32(cov32x4_hb0, cov32x4_hb0);

            addlDen32x4_lb0 = vaddq_s32(lDen32x4_lb0, dupC1);
            addlDen32x4_hb0 = vaddq_s32(lDen32x4_hb0, dupC1);
            addlNum32x4_lb0 = vaddq_s32(cov32x4_lb0, dupC1);
            addlNum32x4_hb0 = vaddq_s32(cov32x4_hb0, dupC1);

            ref16x8_b1 = vld1q_s16(ref->bands[1] + index);
            dist16x8_b1 = vld1q_s16(dist->bands[1] + index);
            ref16x8_b2 = vld1q_s16(ref->bands[2] + index);
            dist16x8_b2 = vld1q_s16(dist->bands[2] + index);
            ref16x8_b3 = vld1q_s16(ref->bands[3] + index);
            dist16x8_b3 = vld1q_s16(dist->bands[3] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b1);
            tmpDist16x8_low = vget_low_s16(dist16x8_b1);
            varX32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmull_high_s16(ref16x8_b1, ref16x8_b1);
            varY32x4_lb1 = vmull_s16(tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmull_high_s16(dist16x8_b1, dist16x8_b1);
            cov32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmull_high_s16(ref16x8_b1, dist16x8_b1);

            tmpRef16x8_low = vget_low_s16(ref16x8_b2);
            tmpDist16x8_low = vget_low_s16(dist16x8_b2);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b2, ref16x8_b2);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b2, dist16x8_b2);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b2, dist16x8_b2);

            tmpRef16x8_low = vget_low_s16(ref16x8_b3);
            tmpDist16x8_low = vget_low_s16(dist16x8_b3);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b3, ref16x8_b3);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b3, dist16x8_b3);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b3, dist16x8_b3);

            covSft32x4_lb1 = vshrq_n_s32(cov32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            covSft32x4_hb1 = vshrq_n_s32(cov32x4_hb1, SSIM_INTER_VAR_SHIFTS);
            varSftX32x4_lb1 = vshrq_n_s32(varX32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            varSftX32x4_hb1 = vshrq_n_s32(varX32x4_hb1, SSIM_INTER_VAR_SHIFTS);
            varSftY32x4_lb1 = vshrq_n_s32(varY32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            varSftY32x4_hb1 = vshrq_n_s32(varY32x4_hb1, SSIM_INTER_VAR_SHIFTS);

            covSft32x4_lb1 = vaddq_s32(covSft32x4_lb1, covSft32x4_lb1);
            covSft32x4_hb1 = vaddq_s32(covSft32x4_hb1, covSft32x4_hb1);
            varSft32x4_lb1 = vaddq_s32(varSftX32x4_lb1, varSftY32x4_lb1);
            varSft32x4_hb1 = vaddq_s32(varSftX32x4_hb1, varSftY32x4_hb1);
            csDen32x4_lb1 = vshrq_n_s32(varSft32x4_lb1, SSIM_INTER_CS_SHIFT);
            csDen32x4_hb1 = vshrq_n_s32(varSft32x4_hb1, SSIM_INTER_CS_SHIFT);

            addcsNum32x4_lb1 = vaddq_s32(covSft32x4_lb1, dupC2);
            addcsNum32x4_hb1 = vaddq_s32(covSft32x4_hb1, dupC2);
            addcsDen32x4_lb1 = vaddq_s32(csDen32x4_lb1, dupC2);
            addcsDen32x4_hb1 = vaddq_s32(csDen32x4_hb1, dupC2);

            mulNum64x2_0lo = vmull_s32(vget_low_s32(addlNum32x4_lb0), vget_low_s32(addcsNum32x4_lb1));
            mulNum64x2_0hi = vmull_high_s32(addlNum32x4_lb0, addcsNum32x4_lb1);
            mulNum64x2_1lo = vmull_s32(vget_low_s32(addlNum32x4_hb0), vget_low_s32(addcsNum32x4_hb1));
            mulNum64x2_1hi = vmull_high_s32(addlNum32x4_hb0, addcsNum32x4_hb1);

            mulDen64x2_0lo = vmull_s32(vget_low_s32(addlDen32x4_lb0), vget_low_s32(addcsDen32x4_lb1));
            mulDen64x2_0hi = vmull_high_s32(addlDen32x4_lb0, addcsDen32x4_lb1);
            mulDen64x2_1lo = vmull_s32(vget_low_s32(addlDen32x4_hb0), vget_low_s32(addcsDen32x4_hb1));
            mulDen64x2_1hi = vmull_high_s32(addlDen32x4_hb0, addcsDen32x4_hb1);

            vst1q_s64(numVal + j, mulNum64x2_0lo);
            vst1q_s64(numVal + j + 2, mulNum64x2_0hi);
            vst1q_s64(numVal + j + 4, mulNum64x2_1lo);
            vst1q_s64(numVal + j + 6, mulNum64x2_1hi);

            vst1q_s64(denVal + j, mulDen64x2_0lo);
            vst1q_s64(denVal + j + 2, mulDen64x2_0hi);
            vst1q_s64(denVal + j + 4, mulDen64x2_1lo);
            vst1q_s64(denVal + j + 6, mulDen64x2_1hi);
        }
        for (; j < width; j++)
        {
            index = i * width + j;
            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;

            for (int k = 1; k < 4; k++)
            {
                var_x += ((ssim_inter_dtype)ref->bands[k][index] * ref->bands[k][index]);
                var_y += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index] * dist->bands[k][index]);
            }
            var_x_band0 = (ssim_inter_dtype)mx * mx;
            var_y_band0 = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;

            var_x = (var_x >> SSIM_INTER_VAR_SHIFTS);
            var_y = (var_y >> SSIM_INTER_VAR_SHIFTS);
            cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

            l_num = (2 * cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);
            cs_num = (2 * cov_xy + C2);
            cs_den = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            numVal[j] = (ssim_accum_dtype)l_num * cs_num;
            denVal[j] = (ssim_accum_dtype)l_den * cs_den;
        }
        for (k = 0; k < width; k++)
        {
            int power_val;
            i16_map_den = ssim_get_best_i16_from_u64((uint64_t)denVal[k], &power_val);
            map = ((numVal[k] >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;

#if ENABLE_MINK3POOL
            ssim_accum_dtype const1_minus_map = const_1 - map;
            rowcube_1minus_map += const1_minus_map * const1_minus_map * const1_minus_map;
#else
            accum_map += map;
            // map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype)map * map));
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

    // accum_map_sq = map_sq_insum / (height * width);
    // double ssim_mean = (double)accum_map / (height * width);
    // double ssim_std;
    // ssim_std = sqrt(MAX(0, ((double)accum_map_sq - ssim_mean * ssim_mean)));
    // *score = (ssim_std / ssim_mean);

    *score = (double) accum_map / (height * width) / (1 << SSIM_SHIFT_DIV);

#endif

    free(numVal);
    free(denVal);
    ret = 0;
    return ret;
}

int integer_compute_ms_ssim_funque_neon(i_dwt2buffers *ref, i_dwt2buffers *dist,
                                        MsSsimScore_int *score, int max_val, float K1, float K2,
                                        int pending_div, int32_t *div_lookup, int n_levels,
                                        int is_pyr)
{
    int cum_array_width = (ref->width) * (1 << n_levels);

    int win_size = (n_levels << 1);
    int win_size_c2 = win_size;
    pending_div = pending_div >> (n_levels - 1);
    int pending_div_c1 = pending_div;
    int pending_div_c2 = pending_div;
    int pending_div_offset = 0;
    int pending_div_halfround = 0;
    int width = ref->width;
    int height = ref->height;

    int32_t *var_x_cum = *(score->var_x_cum);
    int32_t *var_y_cum = *(score->var_y_cum);
    int32_t *cov_xy_cum = *(score->cov_xy_cum);

    if(is_pyr) {
        win_size_c2 = 2;
        pending_div_c1 = (1 << i_nadenau_pending_div_factors[n_levels - 1][0]) * 255;
        pending_div_c2 =
            (1 << (i_nadenau_pending_div_factors[n_levels - 1][1] + (n_levels - 1))) * 255;
        pending_div_offset = 2 * (i_nadenau_pending_div_factors[n_levels - 1][3] -
                                  i_nadenau_pending_div_factors[n_levels - 1][1]);
        pending_div_halfround = (pending_div_offset == 0) ? 0 : (1 << (pending_div_offset - 1));
        if((n_levels > 1)) {
            int index_cum = 0;
            int shift_cums = 2 * (i_nadenau_pending_div_factors[n_levels - 2][1] -
                                  i_nadenau_pending_div_factors[n_levels - 1][1] - 1);
            for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                    var_x_cum[index_cum] =
                        (var_x_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    var_y_cum[index_cum] =
                        (var_y_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    cov_xy_cum[index_cum] =
                        (cov_xy_cum[index_cum] + (1 << (shift_cums - 1))) >> shift_cums;
                    index_cum++;
                }
                index_cum += (cum_array_width - width);
            }
        }
    }

    int64_t c1_mul = (((int64_t) pending_div_c1 * pending_div_c1) >> (SSIM_INTER_L_SHIFT));
    int64_t c2_mul = (((int64_t) pending_div_c2 * pending_div_c2) >>
                      (SSIM_INTER_VAR_SHIFTS + SSIM_INTER_CS_SHIFT));

    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * c1_mul);
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * c2_mul);

    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map, l, cs;
    int16_t i16_l_den;
    int16_t i16_cs_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;

    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_l = 0;
    ssim_accum_dtype accum_cs = 0;
    ssim_accum_dtype accum_sq_map = 0;
    ssim_accum_dtype accum_sq_l = 0;
    ssim_accum_dtype accum_sq_cs = 0;
    ssim_accum_dtype map_sq = 0;

    ssim_inter_dtype mink3_const = 32768;  // 2^15
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

    int16x8_t ref16x8_b0, dist16x8_b0, ref16x8_b1, dist16x8_b1;
    int16x8_t ref16x8_b2, dist16x8_b2, ref16x8_b3, dist16x8_b3;
    int16x4_t tmpRef16x8_low, tmpDist16x8_low;
    int32x4_t varX32x4_lb0, varX32x4_hb0, varY32x4_lb0, varY32x4_hb0, cov32x4_lb0, cov32x4_hb0;
    int32x4_t lDen32x4_lb0, lDen32x4_hb0, csDen32x4_lb1, csDen32x4_hb1;
    int32x4_t addlNum32x4_lb0, addlNum32x4_hb0, addlDen32x4_lb0, addlDen32x4_hb0;
    int32x4_t varSftX32x4_lb1, varSftX32x4_hb1, varSftY32x4_lb1, varSftY32x4_hb1;
    int32x4_t varSft32x4_lb1, varSft32x4_hb1, covSft32x4_lb1, covSft32x4_hb1;
    int32x4_t addcsNum32x4_lb1, addcsNum32x4_hb1, addcsDen32x4_lb1, addcsDen32x4_hb1;
    int32x4_t varX32x4_lb1, varX32x4_hb1, varY32x4_lb1, varY32x4_hb1, cov32x4_lb1, cov32x4_hb1;
    int32x4_t varXcum32x4_lb, varXcum32x4_hb, varYcum32x4_lb, varYcum32x4_hb, covXYcum32x4_lb,
        covXYcum32x4_hb;

    int32x4_t dupC1 = vdupq_n_s32(C1);
    int32x4_t dupC2 = vdupq_n_s32(C2);

    int32_t *lNumVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *csNumVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *lDenVal = (int32_t *) malloc(width * sizeof(int32_t));
    int32_t *csDenVal = (int32_t *) malloc(width * sizeof(int32_t));

    int neg_win_size = -win_size;
    int neg_win_size_c2 = -win_size_c2;
    int sft_if_pyr = (is_pyr == 1) ? 0 : -2;

    int index = 0, i, j, k;
    int index_cum = 0;
    for(i = 0; i < height; i++) {
        ssim_accum_dtype row_accum_sq_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_map = 0;
        ssim_mink3_accum_dtype row_accum_mink3_l = 0;
        ssim_mink3_accum_dtype row_accum_mink3_cs = 0;
        for(j = 0; j <= width - 8; j += 8) {
            index = i * width + j;
            ref16x8_b0 = vld1q_s16(ref->bands[0] + index);
            dist16x8_b0 = vld1q_s16(dist->bands[0] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b0);
            tmpDist16x8_low = vget_low_s16(dist16x8_b0);

            varX32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb0 = vmull_high_s16(ref16x8_b0, ref16x8_b0);
            varY32x4_lb0 = vmull_s16(tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb0 = vmull_high_s16(dist16x8_b0, dist16x8_b0);
            cov32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb0 = vmull_high_s16(ref16x8_b0, dist16x8_b0);

            varX32x4_lb0 = vshlq_n_s32(varX32x4_lb0, neg_win_size);
            varX32x4_hb0 = vshlq_n_s32(varX32x4_hb0, neg_win_size);
            varY32x4_lb0 = vshlq_n_s32(varY32x4_lb0, neg_win_size);
            varY32x4_hb0 = vshlq_n_s32(varY32x4_hb0, neg_win_size);
            cov32x4_lb0 = vshlq_n_s32(cov32x4_lb0, neg_win_size);
            cov32x4_hb0 = vshlq_n_s32(cov32x4_hb0, neg_win_size);

            varX32x4_lb0 = vaddq_s32(varX32x4_lb0, varY32x4_lb0);
            varX32x4_hb0 = vaddq_s32(varX32x4_hb0, varY32x4_hb0);
            lDen32x4_lb0 = vshrq_n_s32(varX32x4_lb0, SSIM_INTER_L_SHIFT);
            lDen32x4_hb0 = vshrq_n_s32(varX32x4_hb0, SSIM_INTER_L_SHIFT);

            cov32x4_lb0 = vaddq_s32(cov32x4_lb0, cov32x4_lb0);
            cov32x4_hb0 = vaddq_s32(cov32x4_hb0, cov32x4_hb0);
            addlNum32x4_lb0 = vaddq_s32(cov32x4_lb0, dupC1);
            addlNum32x4_hb0 = vaddq_s32(cov32x4_hb0, dupC1);
            addlDen32x4_lb0 = vaddq_s32(lDen32x4_lb0, dupC1);
            addlDen32x4_hb0 = vaddq_s32(lDen32x4_hb0, dupC1);

            vst1q_s32(lNumVal + j, addlNum32x4_lb0);
            vst1q_s32(lNumVal + j + 4, addlNum32x4_hb0);
            vst1q_s32(lDenVal + j, addlDen32x4_lb0);
            vst1q_s32(lDenVal + j + 4, addlDen32x4_hb0);

            ref16x8_b1 = vld1q_s16(ref->bands[1] + index);
            dist16x8_b1 = vld1q_s16(dist->bands[1] + index);
            ref16x8_b2 = vld1q_s16(ref->bands[2] + index);
            dist16x8_b2 = vld1q_s16(dist->bands[2] + index);
            ref16x8_b3 = vld1q_s16(ref->bands[3] + index);
            dist16x8_b3 = vld1q_s16(dist->bands[3] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b1);
            tmpDist16x8_low = vget_low_s16(dist16x8_b1);
            varX32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmull_high_s16(ref16x8_b1, ref16x8_b1);
            varY32x4_lb1 = vmull_s16(tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmull_high_s16(dist16x8_b1, dist16x8_b1);
            cov32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmull_high_s16(ref16x8_b1, dist16x8_b1);

            tmpRef16x8_low = vget_low_s16(ref16x8_b2);
            tmpDist16x8_low = vget_low_s16(dist16x8_b2);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b2, ref16x8_b2);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b2, dist16x8_b2);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b2, dist16x8_b2);

            tmpRef16x8_low = vget_low_s16(ref16x8_b3);
            tmpDist16x8_low = vget_low_s16(dist16x8_b3);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b3, ref16x8_b3);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b3, dist16x8_b3);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b3, dist16x8_b3);

            varXcum32x4_lb = vld1q_s32(var_x_cum + index_cum);
            varXcum32x4_hb = vld1q_s32(var_x_cum + index_cum + 4);
            varYcum32x4_lb = vld1q_s32(var_y_cum + index_cum);
            varYcum32x4_hb = vld1q_s32(var_y_cum + index_cum + 4);
            covXYcum32x4_lb = vld1q_s32(cov_xy_cum + index_cum);
            covXYcum32x4_hb = vld1q_s32(cov_xy_cum + index_cum + 4);

            covSft32x4_lb1 = vshlq_n_s32(cov32x4_lb1, neg_win_size_c2);
            covSft32x4_hb1 = vshlq_n_s32(cov32x4_hb1, neg_win_size_c2);
            varSftX32x4_lb1 = vshlq_n_s32(varX32x4_lb1, neg_win_size_c2);
            varSftX32x4_hb1 = vshlq_n_s32(varX32x4_hb1, neg_win_size_c2);
            varSftY32x4_lb1 = vshlq_n_s32(varY32x4_lb1, neg_win_size_c2);
            varSftY32x4_hb1 = vshlq_n_s32(varY32x4_hb1, neg_win_size_c2);

            varXcum32x4_lb = vshlq_n_s32(varXcum32x4_lb, sft_if_pyr);
            varXcum32x4_hb = vshlq_n_s32(varXcum32x4_hb, sft_if_pyr);
            varYcum32x4_lb = vshlq_n_s32(varYcum32x4_lb, sft_if_pyr);
            varYcum32x4_hb = vshlq_n_s32(varYcum32x4_hb, sft_if_pyr);
            covXYcum32x4_lb = vshlq_n_s32(covXYcum32x4_lb, sft_if_pyr);
            covXYcum32x4_hb = vshlq_n_s32(covXYcum32x4_hb, sft_if_pyr);

            varSftX32x4_lb1 = vaddq_s32(varXcum32x4_lb, varSftX32x4_lb1);
            varSftX32x4_hb1 = vaddq_s32(varXcum32x4_hb, varSftX32x4_hb1);
            varSftY32x4_lb1 = vaddq_s32(varYcum32x4_lb, varSftY32x4_lb1);
            varSftY32x4_hb1 = vaddq_s32(varYcum32x4_hb, varSftY32x4_hb1);
            covSft32x4_lb1 = vaddq_s32(covXYcum32x4_lb, covSft32x4_lb1);
            covSft32x4_hb1 = vaddq_s32(covXYcum32x4_hb, covSft32x4_hb1);

            vst1q_s32(var_x_cum + index_cum, varSftX32x4_lb1);
            vst1q_s32(var_x_cum + index_cum + 4, varSftX32x4_hb1);
            vst1q_s32(var_y_cum + index_cum, varSftY32x4_lb1);
            vst1q_s32(var_y_cum + index_cum + 4, varSftY32x4_hb1);
            vst1q_s32(cov_xy_cum + index_cum, covSft32x4_lb1);
            vst1q_s32(cov_xy_cum + index_cum + 4, covSft32x4_hb1);

            varSft32x4_lb1 = vaddq_s32(varSftX32x4_lb1, varSftY32x4_lb1);
            varSft32x4_hb1 = vaddq_s32(varSftX32x4_hb1, varSftY32x4_hb1);
            csDen32x4_lb1 = vshrq_n_s32(varSft32x4_lb1, SSIM_INTER_CS_SHIFT);
            csDen32x4_hb1 = vshrq_n_s32(varSft32x4_hb1, SSIM_INTER_CS_SHIFT);

            addcsNum32x4_lb1 = vmlaq_n_s32(dupC2, covSft32x4_lb1, 2);
            addcsNum32x4_hb1 = vmlaq_n_s32(dupC2, covSft32x4_hb1, 2);
            addcsDen32x4_lb1 = vaddq_s32(csDen32x4_lb1, dupC2);
            addcsDen32x4_hb1 = vaddq_s32(csDen32x4_hb1, dupC2);

            vst1q_s32(csNumVal + j, addcsNum32x4_lb1);
            vst1q_s32(csNumVal + j + 4, addcsNum32x4_hb1);
            vst1q_s32(csDenVal + j, addcsDen32x4_lb1);
            vst1q_s32(csDenVal + j + 4, addcsDen32x4_hb1);
            index_cum += 8;
        }
        for(; j < width; j++) {
            index = i * width + j;

            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;
            int k;
#if BAND_HVD_SAME_PENDING_DIV
            for(k = 1; k < 4; k++)
#else
            for(k = 1; k < 3; k++)
#endif
            {
                var_x += ((ssim_inter_dtype) ref->bands[k][index] * ref->bands[k][index]);
                var_y += ((ssim_inter_dtype) dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype) ref->bands[k][index] * dist->bands[k][index]);
            }
#if !(BAND_HVD_SAME_PENDING_DIV)
            // The extra right shift will be done for pyr since the upscale factors are different
            // for subbands
            var_x += (((ssim_inter_dtype) ref->bands[k][index] * ref->bands[k][index]) +
                      pending_div_halfround) >>
                     pending_div_offset;
            var_y += (((ssim_inter_dtype) dist->bands[k][index] * dist->bands[k][index]) +
                      pending_div_halfround) >>
                     pending_div_offset;
            cov_xy += (((ssim_inter_dtype) ref->bands[k][index] * dist->bands[k][index]) +
                       pending_div_halfround) >>
                      pending_div_offset;
#endif
            var_x_band0 = ((ssim_inter_dtype) mx * mx) >> win_size;
            var_y_band0 = ((ssim_inter_dtype) my * my) >> win_size;
            cov_xy_band0 = ((ssim_inter_dtype) mx * my) >> win_size;

            var_x_cum[index_cum] = var_x_cum[index_cum] << sft_if_pyr;
            var_y_cum[index_cum] = var_y_cum[index_cum] << sft_if_pyr;
            cov_xy_cum[index_cum] = cov_xy_cum[index_cum] << sft_if_pyr;

            var_x_cum[index_cum] += (var_x >> win_size_c2);
            var_y_cum[index_cum] += (var_y >> win_size_c2);
            cov_xy_cum[index_cum] += (cov_xy >> win_size_c2);

            var_x = var_x_cum[index_cum];
            var_y = var_y_cum[index_cum];
            cov_xy = cov_xy_cum[index_cum];

            lNumVal[j] = ((2 >> SSIM_INTER_L_SHIFT) * cov_xy_band0 + C1);
            lDenVal[j] = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);

            csNumVal[j] = ((2 >> SSIM_INTER_CS_SHIFT) * cov_xy + C2);
            csDenVal[j] = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);
            index_cum++;
        }

        for(k = 0; k < width; k++) {
            int power_val_l;
            i16_l_den = ms_ssim_get_best_i16_from_u32_neon((uint32_t) lDenVal[k], &power_val_l);

            int power_val_cs;
            i16_cs_den = ms_ssim_get_best_i16_from_u32_neon((uint32_t) csDenVal[k], &power_val_cs);

            l = ((lNumVal[k] >> power_val_l) * div_lookup[i16_l_den + 32768]) >> SSIM_SHIFT_DIV;
            cs = ((csNumVal[k] >> power_val_cs) * div_lookup[i16_cs_den + 32768]) >> SSIM_SHIFT_DIV;
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

    double mink3_cbrt_const_l = pow(2, (39 / 3));
    double mink3_cbrt_const_cs = pow(2, (38.0 / 3));
    double mink3_cbrt_const_map = pow(2, (38.0 / 3));

    double l_mink3 = mink3_cbrt_const_l - (double) cbrt(accum_mink3_l / (width * height));
    double cs_mink3 = mink3_cbrt_const_cs - (double) cbrt(accum_mink3_cs / (width * height));
    double ssim_mink3 = mink3_cbrt_const_map - (double) cbrt(accum_mink3_map / (width * height));

    score->ssim_mean = ssim_mean / (1 << (SSIM_SHIFT_DIV * 2));
    score->l_mean = l_mean / (1 << SSIM_SHIFT_DIV);
    score->cs_mean = cs_mean / (1 << SSIM_SHIFT_DIV);
    score->ssim_cov = ssim_cov;
    score->l_cov = l_cov;
    score->cs_cov = cs_cov;
    score->l_mink3 = l_mink3 / pow(2, (39 / 3));
    score->cs_mink3 = cs_mink3 / pow(2, (38.0 / 3));
    score->ssim_mink3 = ssim_mink3 / pow(2, (38.0 / 3));

    ret = 0;

    return ret;
}

int integer_mean_2x2_ms_ssim_funque_neon(int32_t *var_x_cum, int32_t *var_y_cum,
                                         int32_t *cov_xy_cum, int width, int height, int level)
{
    int ret = 1;
    int cum_array_width = (width) * (1 << (level + 1));
    int index = 0;
    int index_cum = 0;
    int i = 0;
    int j = 0;

    for(i = 0; i < (height / 2); i++) {
        for(j = 0; j <= width - 8; j = j + 8) {
            index = i * cum_array_width + j / 2;

            int32x4_t var_x_1_1 = vld1q_s32(&var_x_cum[index_cum]);
            int32x4_t var_x_1_2 = vld1q_s32(&var_x_cum[index_cum + 4]);
            int32x4_t var_x_2_1 = vld1q_s32(&var_x_cum[index_cum + (cum_array_width)]);
            int32x4_t var_x_2_2 = vld1q_s32(&var_x_cum[index_cum + (cum_array_width) + 4]);
            int32x4_t var1_x = vpaddq_s32(var_x_1_1, var_x_1_2);
            int32x4_t var2_x = vpaddq_s32(var_x_2_1, var_x_2_2);
            int32x4_t var3_x = vaddq_s32(var1_x, var2_x);
            vst1q_s32(&var_x_cum[index], vrshrq_n_s32(var3_x, 2));

            int32x4_t var_y_1_1 = vld1q_s32(&var_y_cum[index_cum]);
            int32x4_t var_y_1_2 = vld1q_s32(&var_y_cum[index_cum + 4]);
            int32x4_t var_y_2_1 = vld1q_s32(&var_y_cum[index_cum + (cum_array_width)]);
            int32x4_t var_y_2_2 = vld1q_s32(&var_y_cum[index_cum + (cum_array_width) + 4]);
            int32x4_t var1_y = vpaddq_s32(var_y_1_1, var_y_1_2);
            int32x4_t var2_y = vpaddq_s32(var_y_2_1, var_y_2_2);
            int32x4_t var3_y = vaddq_s32(var1_y, var2_y);
            vst1q_s32(&var_y_cum[index], vrshrq_n_s32(var3_y, 2));

            int32x4_t cov_xy_1_1 = vld1q_s32(&cov_xy_cum[index_cum]);
            int32x4_t cov_xy_1_2 = vld1q_s32(&cov_xy_cum[index_cum + 4]);
            int32x4_t cov_xy_2_1 = vld1q_s32(&cov_xy_cum[index_cum + (cum_array_width)]);
            int32x4_t cov_xy_2_2 = vld1q_s32(&cov_xy_cum[index_cum + (cum_array_width) + 4]);
            int32x4_t var1_xy = vpaddq_s32(cov_xy_1_1, cov_xy_1_2);
            int32x4_t var2_xy = vpaddq_s32(cov_xy_2_1, cov_xy_2_2);
            int32x4_t var3_xy = vaddq_s32(var1_xy, var2_xy);
            vst1q_s32(&cov_xy_cum[index], vrshrq_n_s32(var3_xy, 2));

            index_cum += 8;
        }
        for(; j < width; j = j + 2) {
            index = i * cum_array_width + j / 2;
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
