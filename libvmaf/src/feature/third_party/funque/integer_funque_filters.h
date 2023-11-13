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

#include "funque_vif_options.h"
#define ADM_REFLECT_PAD 0
#define VIF_REFLECT_PAD 1

#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)

#define NGAN_21_TAP_FILTER 21
#define NADENAU_SPAT_5_TAP_FILTER 5
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
#define SSIM_SHIFT_DIV 15 //Depends on ssim_accum_dtype datatype
#define SSIM_INTER_VAR_SHIFTS 1
#define SSIM_INTER_L_SHIFT 1 //If this is updated, the usage has to be changed in integer_ssim.c(currently 2>>SSIM_INTER_L_SHIFT) is used for readability
#define SSIM_INTER_CS_SHIFT 1 //If this is updated, the usage has to be changed in integer_ssim.c(currently 2>>SSIM_INTER_CS_SHIFT) is used for readability

typedef struct i_dwt2buffers {
    dwt2_dtype *bands[4];
    int width;
    int height;
    int stride;
}i_dwt2buffers;

typedef struct ModuleFunqueState
{
    //function pointers
    void (*integer_funque_picture_copy)(void *src, spat_fil_output_dtype *dst, int dst_stride, int width, int height, int bitdepth);
    void (*integer_spatial_filter)(void *src, spat_fil_output_dtype *dst, int dst_stride, int width, int height, int bitdepth, spat_fil_inter_dtype *tmp, int num_taps);
    void (*integer_funque_dwt2)(spat_fil_output_dtype *src, ptrdiff_t src_stride, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height, int spatial_csf_flag, int level);
    void (*integer_funque_dwt2_inplace_csf)(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4], int min_theta, int max_theta, uint16_t interim_rnd_factors[4], uint8_t interim_shift_factors[4], int level); 
    void (*integer_funque_vifdwt2_band0)(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height);
    int (*integer_compute_ssim_funque)(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup);
    int (*integer_compute_ms_ssim_funque)(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup);
    double (*integer_funque_image_mad)(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor);
    void (*integer_funque_adm_decouple)(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, int32_t *i_dlm_add, 
                                 int32_t *adm_div_lookup, float border_size, double *adm_score_den);
    void (*integer_adm_integralimg_numscore)(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                                             int stride, int width, int height, int32_t *interim_x, 
                                             float border_size, double *adm_score_num);

#if USE_DYNAMIC_SIGMA_NSQ
    int (*integer_compute_vif_funque)(const dwt2_dtype* x_t, const dwt2_dtype* y_t, 
                                           size_t width, size_t height, 
                                           double* score, double* score_num, double* score_den, 
                                           int k, int stride, double sigma_nsq_arg, 
                                           int64_t shift_val, uint32_t* log_18, int vif_level);
#else
    int (*integer_compute_vif_funque)(const dwt2_dtype* x_t, const dwt2_dtype* y_t, 
                                           size_t width, size_t height, 
                                           double* score, double* score_num, double* score_den, 
                                           int k, int stride, double sigma_nsq_arg, 
                                           int64_t shift_val, uint32_t* log_18);
#endif
    // void (*resizer_step)(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax);
}ModuleFunqueState;

/* filter format where 0 = approx, 1 = vertical, 2 = diagonal, 3 = horizontal as in funque_dwt2_inplace_csf */
/* All the coefficients are in Q15 format*/
static const spat_fil_coeff_dtype i_nadenau_weight_coeffs[4][4] = {
    {32767, 22544, 23331, 22544},
    {32767, 27836, 24297, 27836},
    {32767, 26081, 20876, 26081},
    {32767, 30836, 29061, 30836},
    /*{ 1, 0.98396102, 0.96855064, 0.98396102},*/
};

static const uint8_t i_nadenau_weight_interim_shift[4][4] = {
    {10, 9, 9, 9},
    {15, 14, 14, 14},
    {15, 14, 14, 14},
    {15, 14, 14, 14},
};
void integer_spatial_filter(void *src, spat_fil_output_dtype *dst, int dst_stride, int width, int height, int bitdepth, spat_fil_inter_dtype *tmp, int num_taps);

void integer_funque_dwt2(spat_fil_output_dtype *src, ptrdiff_t src_stride, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height, int spatial_csf_flag, int level);

void integer_funque_dwt2_wavelet(void *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height);

void integer_funque_dwt2_inplace_csf(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4], int min_theta, int max_theta, uint16_t interim_rnd_factors[4], uint8_t interim_shift_factors[4], int level);

#endif /* FILTERS_FUNQUE_H_ */
