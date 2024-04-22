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
#ifndef FILTERS_FUNQUE_H_
#define FILTERS_FUNQUE_H_
#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "funque_global_options.h"
#include "funque_vif_options.h"
#define ADM_REFLECT_PAD 1
#define VIF_REFLECT_PAD 1

#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
#define UNUSED(x) (void) (x)

// Spatial Filters
#define NGAN_21_TAP_FILTER 21
#define NADENAU_SPAT_5_TAP_FILTER 5

// Wavelet Filters
#define NADENAU_WEIGHT_FILTER 1  // Default set to nadenau_weight
#define LI_FILTER 2
#define HILL_FILTER 3
#define WATSON_FILTER 4
#define MANNOS_WEIGHT_FILTER 5

#define BAND_HVD_SAME_PENDING_DIV 1
#define SPAT_FILTER_COEFF_SHIFT 16
#define SPAT_FILTER_INTER_SHIFT  9
#define SPAT_FILTER_INTER_RND (1 << (SPAT_FILTER_INTER_SHIFT - 1))
#define SPAT_FILTER_OUT_SHIFT   16
#define SPAT_FILTER_OUT_RND (1 << (SPAT_FILTER_OUT_SHIFT - 1))
typedef int16_t spat_fil_coeff_dtype;
typedef int16_t spat_fil_inter_dtype;
typedef int32_t spat_fil_accum_dtype;
typedef int16_t spat_fil_output_dtype;

#define DWT2_COEFF_UPSHIFT 0
#define DWT2_INTER_SHIFT   0  //Shifting to make the intermediate have Q16 format
#define DWT2_OUT_SHIFT     1  //Shifting to make the output have Q16 format

typedef int16_t dwt2_dtype;
typedef int8_t dwt2_input_dtype;
typedef int32_t dwt2_accum_dtype;
typedef int16_t dwt2_inter_dtype;

typedef int32_t motion_interaccum_dtype;
typedef int64_t motion_accum_dtype;

typedef int32_t ssim_inter_dtype;
typedef int64_t ssim_accum_dtype;
typedef uint32_t ssim_mink3_inter_dtype;
typedef uint64_t ssim_mink3_accum_dtype;
#define SSIM_SHIFT_DIV 15 //Depends on ssim_accum_dtype datatype
#define SSIM_SQ_ROW_SHIFT 9
#define SSIM_SQ_COL_SHIFT 11
#define SSIM_INTER_VAR_SHIFTS 0
#define SSIM_INTER_L_SHIFT \
    0  // If this is updated, the usage has to be changed in integer_ssim.c(currently
       // 2>>SSIM_INTER_L_SHIFT) is used for readability
#define SSIM_INTER_CS_SHIFT \
    0  // If this is updated, the usage has to be changed in integer_ssim.c(currently
       // 2>>SSIM_INTER_CS_SHIFT) is used for readability
#define L_R_SHIFT 0
#define CS_R_SHIFT 0
#define SSIM_R_SHIFT 14
#define L_MINK3_ROW_R_SHIFT 6
#define CS_MINK3_ROW_R_SHIFT 7
#define SSIM_MINK3_ROW_R_SHIFT 10

#define STRRED_Q_FORMAT 26
#define BITS_USED_BY_VIF_AND_STRRED_LUT 22
#define TWO_POW_BITS_USED (1 << BITS_USED_BY_VIF_AND_STRRED_LUT)

typedef struct i_dwt2buffers {
    dwt2_dtype *bands[4];
    int width;
    int height;
    int stride;
}i_dwt2buffers;

typedef struct SsimScore_int {
    double mean;
    double mink3;
} SsimScore_int;

typedef struct MsSsimScore_int {
    double ssim_mean;
    double l_mean;
    double cs_mean;
    double ssim_cov;
    double l_cov;
    double cs_cov;
    double l_mink3;
    double cs_mink3;
    double ssim_mink3;

    double ms_ssim_mean;
    double ms_ssim_cov;
    double ms_ssim_mink3;

    int32_t **var_x_cum;
    int32_t **var_y_cum;
    int32_t **cov_xy_cum;
} MsSsimScore_int;

typedef struct strred_results {
    double srred_vals[MAX_LEVELS];
    double trred_vals[MAX_LEVELS];
    double strred_vals[MAX_LEVELS];
    double spat_vals[MAX_LEVELS];
    double temp_vals[MAX_LEVELS];
    double spat_temp_vals[MAX_LEVELS];
    double spat_vals_cumsum, temp_vals_cumsum, spat_temp_vals_cumsum;

} strred_results;

typedef struct ModuleFunqueState
{
    //function pointers
    void (*integer_funque_picture_copy)(void *src, spat_fil_output_dtype *dst, int dst_stride,
                                        int width, int height, int bitdepth);
    void (*integer_spatial_filter)(void *src, spat_fil_output_dtype *dst, int dst_stride, int width,
                                   int height, int bitdepth, spat_fil_inter_dtype *tmp,
                                   char *spatial_csf_filter);
    void (*integer_funque_dwt2)(spat_fil_output_dtype *src, ptrdiff_t src_stride,
                                i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width,
                                int height, int spatial_csf_flag, int level);
    void (*integer_funque_dwt2_inplace_csf)(const i_dwt2buffers *src,
                                            spat_fil_coeff_dtype factors[4], int min_theta,
                                            int max_theta, uint16_t interim_rnd_factors[4],
                                            uint8_t interim_shift_factors[4], int level);
    void (*integer_funque_vifdwt2_band0)(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height);
    int (*integer_compute_ssim_funque)(i_dwt2buffers *ref, i_dwt2buffers *dist, SsimScore_int *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup);
    int (*integer_compute_ms_ssim_funque)(i_dwt2buffers *ref, i_dwt2buffers *dist,
                                          MsSsimScore_int *score, int max_val, float K1, float K2,
                                          int pending_div_c1, int pending_div_c2, int pending_div_offset, 
                                          int pending_div_halfround, int32_t *div_lookup, int n_levels,
                                          int is_pyr);
    int (*integer_mean_2x2_ms_ssim_funque)(int32_t *var_x_cum, int32_t *var_y_cum,
                                           int32_t *cov_xy_cum, int width, int height, int level);
    int (*integer_ms_ssim_shift_cum_buffer_funque)(int32_t *var_x_cum, int32_t *var_y_cum,
                                           int32_t *cov_xy_cum, int width, int height, int level, uint8_t csf_pending_div[4], uint8_t csf_pending_div_lp1[4]);
    int (*integer_compute_motion_funque)(const dwt2_dtype *prev, const dwt2_dtype *curr, int w, int h, int prev_stride, int curr_stride, int pending_div_factor, double *score);
    int (*integer_compute_mad_funque)(const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, int pending_div_factor, double *score);
    void (*integer_funque_adm_decouple)(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, int32_t *i_dlm_add, 
                                 int32_t *adm_div_lookup, float border_size, double *adm_score_den, float adm_pending_div);
    void (*integer_adm_integralimg_numscore)(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                                             int stride, int width, int height, int32_t *interim_x, 
                                             float border_size, double *adm_score_num, float adm_pending_div);

#if USE_DYNAMIC_SIGMA_NSQ
    int (*integer_compute_vif_funque)(const dwt2_dtype* x_t, const dwt2_dtype* y_t, 
                                           size_t width, size_t height, 
                                           double* score, double* score_num, double* score_den, 
                                           int k, int stride, double sigma_nsq_arg, 
                                           int64_t shift_val, uint32_t* log_lut, int vif_level);
#else
    int (*integer_compute_vif_funque)(const dwt2_dtype* x_t, const dwt2_dtype* y_t, 
                                           size_t width, size_t height, 
                                           double* score, double* score_num, double* score_den, 
                                           int k, int stride, double sigma_nsq_arg, 
                                           int64_t shift_val, uint32_t* log_lut);
#endif
    // void (*resizer_step)(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax);

    int (*integer_compute_srred_funque)(
        const struct i_dwt2buffers *ref, const struct i_dwt2buffers *dist, size_t width,
        size_t height, float **spat_scales_ref, float **spat_scales_dist,
        struct strred_results *strred_scores, int block_size, int level,
        uint32_t *log_lut, int32_t shift_val, double sigma_nsq_t, uint8_t enable_spatial_csf, uint8_t csf_pending_div[4]);

    int (*integer_compute_strred_funque)(
        const struct i_dwt2buffers *ref, const struct i_dwt2buffers *dist,
        struct i_dwt2buffers *prev_ref, struct i_dwt2buffers *prev_dist, size_t width,
        size_t height, float **spat_scales_ref, float **spat_scales_dist,
        struct strred_results *strred_scores, int block_size, int level,
        uint32_t *log_lut, int32_t shift_val, double sigma_nsq_t, uint8_t enable_spatial_csf, uint8_t csf_pending_div[4]);

    int (*integer_copy_prev_frame_strred_funque)(const struct i_dwt2buffers *ref,
                                                 const struct i_dwt2buffers *dist,
                                                 struct i_dwt2buffers *prev_ref,
                                                 struct i_dwt2buffers *prev_dist, size_t width,
                                                 size_t height);

}ModuleFunqueState;

/* filter format where 0 = approx, 1 = vertical, 2 = diagonal, 3 = horizontal as in
 * funque_dwt2_inplace_csf */
/* All the coefficients are in Q15 format*/
static const spat_fil_coeff_dtype i_nadenau_weight_coeffs[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {16384, 22544, 23331, 22544},
    {16384, 27836, 24297, 27836},
#else
    {16384, 22544, 23331, 22544},
    {16384, 27836, 24297, 27836},
#endif
    {16384, 26081, 20876, 26081},
    {16384, 30836, 29061, 30836}
};

static const uint8_t i_nadenau_pending_div_factors[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {6, 11, 11, 11},  // L0
    {5, 7, 7, 7},     // L1
#else
    {6, 11, 11, 14},  // L0
    {5, 7, 7, 8},     // L1
#endif
    {4, 5, 5, 5},  // L2
    {3, 4, 4, 4}  // L3
};

static const uint8_t i_nadenau_interim_shift[4][4] = {
    {9, 9, 12, 9},
    {11, 11, 12, 11},
    {13, 13, 13, 13},
    {13, 13, 13, 13}
};

static const spat_fil_coeff_dtype i_li_coeffs[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {16384, 22842, 30944, 22842},
    {16384, 21867, 21362, 21867},
    {16384, 21885, 25508, 21885},
#else
    {16384, 22842, 30944, 22842},
    {16384, 21867, 21362, 21867},
    {16384, 21885, 25508, 21885},
#endif
    {16384, 32318, 28749, 32318}
};

static const uint8_t i_li_interim_shift[4][4] = {
    {9, 12, 17, 12},
    {11, 12, 14, 12},
    {13, 13, 14, 13},
    {13, 13, 13, 13}
};

static const uint8_t i_li_pending_div_factors[4][4] = {
    {6, 11, 11, 11},  // L0
    {5, 7, 7, 7},   // L1
    {4, 5, 5, 5},  // L2
    {3, 4, 4, 4}  // L3
};

static const spat_fil_coeff_dtype i_hill_coeffs[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {16384, 22691, 20082, 22691},
    {16384, -22164, 28535, -22164},
    {16384, -17920, -26774, -17920},
#else
    {16384, 22691, 20082, 22691},
    {16384, -22164, 28535, -22164},
    {16384, -17920, -26774, -17920},
#endif
    {16384, -22347, -32484, -22347}
};

static const uint8_t i_hill_interim_shift[4][4] = {
    {9, 9, 10, 9},
    {11, 11, 13, 11},
    {13, 15, 14, 15},
    {13, 16, 16, 16},
};
 
static const uint8_t i_hill_pending_div_factors[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {6, 10, 10, 10},  // L0
    {5, 7, 7, 7},     // L1
    {4, 6, 6, 6},     // L2
#else
    {6, 10, 10, 11},  // L0
    {5, 7, 7, 9},     // L1
    {4, 8, 8, 7},     // L2
#endif
    {3, 5, 5, 5}  // L3
};

static const spat_fil_coeff_dtype i_watson_coeffs[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {16384, 18226, 24707, 18226},
    {16384, 16769, 29987, 16769},
#else
    {16384, 22544, 23331, 22544},
    {16384, 27836, 24297, 27836},
#endif
    {16384, 22740, 25582, 22740},
    {16384, 23946, 16417, 23946}
};

static const uint8_t i_watson_interim_shift[4][4] = {
    {9, 10, 12, 10},
    {11, 11, 13, 11},
    {13, 13, 14, 13},
    {13, 13, 13, 13}
};

static const uint8_t i_watson_pending_div_factors[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {6, 11, 11, 11},  // L0
    {5, 10, 10, 10},  // L1
#else
    {6, 12, 12, 14},  // L0
    {5, 10, 10, 11},  // L1
#endif
    {4, 9, 9, 9},  // L2
    {3, 8, 8, 8},  // L3
};

static const spat_fil_coeff_dtype i_mannos_weight_coeffs[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {16384, 29856, 32663, 29856},
    {16384, 29492, 29501, 29492},
#else
    {16384, 29856, 32663, 29856},
    {16384, 29492, 29501, 29492},
#endif
    {16384, 25627, 32387, 25627},
    {16384, 32145, 32145, 32145}
};

static const uint8_t i_mannos_weight_interim_shift[4][4] = {
    {9, 12, 17, 12},
    {11, 12, 14, 12},
    {13, 13, 14, 13},
    {13, 13, 13, 13}
};

static const uint8_t i_mannos_weight_pending_div_factors[4][4] = {
#if BAND_HVD_SAME_PENDING_DIV
    {6, 11, 11, 11},  // L0
    {5, 7, 7, 7},     // L1
#else
    {6, 14, 14, 19},  // L0
    {5, 8, 8, 10},    // L1
#endif
    {4, 5, 5, 5},  // L2
    {3, 4, 4, 4},  // L3
};

void funque_log_generate(uint32_t *log_lut);

void integer_spatial_filter_c(void *src, spat_fil_output_dtype *dst, int dst_stride, int width,
                              int height, int bitdepth, spat_fil_inter_dtype *tmp,
                              char *spatial_csf_filter);

void integer_funque_dwt2_c(spat_fil_output_dtype *src, ptrdiff_t src_stride,
                           i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height,
                           int spatial_csf_flag, int level);

void integer_funque_dwt2_wavelet(void *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride,
                                 int width, int height);

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height);

void integer_funque_dwt2_inplace_csf_c(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4],
                                       int min_theta, int max_theta,
                                       uint16_t interim_rnd_factors[4],
                                       uint8_t interim_shift_factors[4], int level);

void integer_reflect_pad_for_input_hbd(void *src, void *dst, int width, int height,
                                       int reflect_width, int reflect_height);
void integer_reflect_pad_for_input(void *src, void *dst, int width, int height, int reflect_width,
                                   int reflect_height, int bpc);

#endif /* FILTERS_FUNQUE_H_ */
