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

void spatial_csfs(float *src, float *dst, int width, int height, int num_taps);

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, int width, int height);

void funque_vifdwt2_band0(float *src, float *band_a, ptrdiff_t dst_stride, int width, int height);

void normalize_bitdepth(float *src, float *dst, int scaler, ptrdiff_t dst_stride, int width, int height);

void funque_dwt2_inplace_csf(const dwt2buffers *src, float factors[4], int min_theta, int max_theta);

#endif /* FILTERS_FUNQUE_H_ */