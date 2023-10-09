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

#include "config.h"
#include <math.h>
#include "macros.h"

struct funque_dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

// 0 -> Y, 1 -> Cb, 2 -> Cr
static const struct funque_dwt_model_params funque_dwt_7_9_YCbCr_threshold[3] = {
        { .a = 0.495, .k = 0.466, .f0 = 0.401, .g = { 1.501, 1.0, 0.534, 1.0} },
        { .a = 1.633, .k = 0.353, .f0 = 0.209, .g = { 1.520, 1.0, 0.502, 1.0} },
        { .a = 0.944, .k = 0.521, .f0 = 0.404, .g = { 1.868, 1.0, 0.516, 1.0} }
};

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static const float funque_dwt_7_9_basis_function_amplitudes[6][4] = {
        { 0.62171,  0.67234,  0.72709,  0.67234  },
        { 0.34537,  0.41317,  0.49428,  0.41317  },
        { 0.18004,  0.22727,  0.28688,  0.22727  },
        { 0.091401, 0.11792,  0.15214,  0.11792  },
};

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static FORCE_INLINE inline float funque_dwt_quant_step(const struct funque_dwt_model_params *params,
                                                int lambda, int theta, double norm_view_dist, int ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = norm_view_dist * ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0,lambda+1)*params->f0*params->g[theta]/r);
    float Q = 2.0*params->a*pow(10.0,params->k*temp*temp)/funque_dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

typedef struct dwt2buffers {
    float *bands[4];
    int width;
    int height;
    ptrdiff_t stride;
}dwt2buffers;

/* filter format where 0 = approx, 1 = vertical, 2 = diagonal, 3 = horizontal as in funque_dwt2_inplace_csf */
static const float nadenau_weight_coeffs[4][4] = 
{
    { 1, 0.04299846, 0.00556257, 0.04299846},
    { 1, 0.42474743, 0.18536903, 0.42474743},
    { 1, 0.79592083, 0.63707782, 0.79592083},
    { 1, 0.94104990, 0.88686180, 0.94104990},
    //{ 1, 0.98396102, 0.96855064, 0.98396102},
};
 
static const float li_coeffs[4][4] = 
{
    { 1, 0.00544585178,  0.00023055401, 0.00544585178},
    { 1, 0.16683506215,  0.04074566701, 0.16683506215},
    { 1, 0.66786346496,  0.38921962529, 0.66786346496},
    { 1, 0.98626459244,  0.87735995465, 0.98626459244},
    //{ 1, 0.91608864363, 0.91608864363, 0.98675189575},
};

static const float hill_coeffs[4][4] = 
{
    {  1,  0.08655904,  0.03830355,   0.08655904},
    {  1, -0.33820268,  0.10885236,  -0.33820268},
    {  1, -0.06836095, -0.20426743,  -0.06836095},
    {  1, -0.04262307, -0.0619592 ,  -0.04262307},
    //{ -0.03159788, -0.03394834, -0.03394834, -0.04074054},
};

static const float watson_coeffs[4][4] = 
{
    {  1, 0.01738153, 0.00589069, 0.01738153},
    {  1, 0.03198481, 0.01429907, 0.03198481},
    {  1, 0.04337266, 0.02439691, 0.04337266},
    {  1, 0.04567341, 0.03131274, 0.04567341},
    //{ 0.03669688, 0.03867382, 0.03867382, 0.03187392},
};

static const float mannos_weight_coeffs[4][4] = 
{
    {  1, 7.11828321e-03, 2.43358422e-04, 7.11828321e-03},
    {  1, 2.25003123e-01, 5.62679733e-02, 2.25003123e-01},
    {  1, 7.82068784e-01, 4.94193706e-01, 7.82068784e-01},
    {  1, 9.81000000e-01, 9.81000000e-01, 9.81000000e-01},
    //{ 9.81000000e-01, 9.81000000e-01, 9.81000000e-01, 9.81000000e-01},
};  

void spatial_csfs(float *src, float *dst, int width, int height, int num_taps);

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, int width, int height);

void funque_vifdwt2_band0(float *src, float *band_a, ptrdiff_t dst_stride, int width, int height);

void normalize_bitdepth(float *src, float *dst, int scaler, ptrdiff_t dst_stride, int width, int height);

void funque_dwt2_inplace_csf(const dwt2buffers *src, float factors[4], int min_theta, int max_theta);

#endif /* FILTERS_FUNQUE_H_ */