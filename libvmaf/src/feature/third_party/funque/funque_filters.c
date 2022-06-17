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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "mem.h"
#include "offset.h"
#include "funque_filters.h"

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(float);
    float filter_coeff_lo[2] = {0.707106781,  0.707106781};
    float filter_coeff_hi[2] = {0.707106781, -0.707106781};

    float *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);

    float *band_a = dwt2_dst->bands[0];
    float *band_h = dwt2_dst->bands[1];
    float *band_v = dwt2_dst->bands[2];
    float *band_d = dwt2_dst->bands[3];
    float accum;
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = 0;
            accum += filter_coeff_lo[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_lo[1] * src[(row_idx1)*width+j];
            tmplo[j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_hi[1] * src[(row_idx1)*width+j];
            tmphi[j] = accum;
        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            accum = 0;
            accum += filter_coeff_lo[0] * tmplo[col_idx0];
            accum += filter_coeff_lo[1] * tmplo[col_idx1];
            band_a[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_lo[0] * tmphi[col_idx0];
            accum += filter_coeff_lo[1] * tmphi[col_idx1];
            band_h[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmplo[col_idx0];
            accum += filter_coeff_hi[1] * tmplo[col_idx1];
            band_v[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmphi[col_idx0];
            accum += filter_coeff_hi[1] * tmphi[col_idx1];
            band_d[i*dst_px_stride+j] = accum;
        }
    }
    aligned_free(tmplo);
    aligned_free(tmphi);
}

//This is dwt function used to get inputs only to VIF higher levels
//This is because only band0 is used as input to VIF
void funque_vifdwt2_band0(float *src, float *band_a, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(float);
    float filter_coeff_lo[2] = {0.707106781,  0.707106781};

    float *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);

    float accum;
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = 0;
            accum += filter_coeff_lo[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_lo[1] * src[(row_idx1)*width+j];
            tmplo[j] = accum;

        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            accum = 0;
            accum += filter_coeff_lo[0] * tmplo[col_idx0];
            accum += filter_coeff_lo[1] * tmplo[col_idx1];
            band_a[i*dst_px_stride+j] = accum;
        }
    }
    aligned_free(tmplo);
}

//Convolution using coefficients from python workspace
void spatial_filter(float *src, float *dst, ptrdiff_t dst_stride, int width, int height)
{
    //Copied the coefficients from python coefficients
    float filter_coeffs[21] = {-0.01373464, -0.01608515, -0.01890698, -0.02215702, -0.02546262, 
                             -0.02742965, -0.02361034, -0.00100996,  0.07137023,  0.22121922,
                             0.3279824 ,  0.22121922,  0.07137023, -0.00100996, -0.02361034,
                             -0.02742965, -0.02546262, -0.02215702, -0.01890698, -0.01608515,
                             -0.01373464};
    
    int src_px_stride = width;
    int dst_px_stride = width;

    float *tmp = aligned_malloc(ALIGN_CEIL(src_px_stride * sizeof(float)), MAX_ALIGN);
    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;
    int fwidth = 21;
    for (i = 0; i < height; ++i) {

        /* Vertical pass. */
        for (j = 0; j < width; ++j) {
            double accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = filter_coeffs[fi];
                
                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -(ii+1)  : (ii >= height ? 2 * height - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += (double) fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < width; ++j) {
            float accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = filter_coeffs[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -(jj+1) : (jj >= width ? 2 * width - jj - 1 : jj);

                imgcoeff = tmp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);

    return;
}

void normalize_bitdepth(float *src, float *dst, int scaler, ptrdiff_t dst_stride, int width, int height)
{
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            dst[j] = src[j] / scaler;
        }
        dst += dst_stride / sizeof(float);
        src += dst_stride / sizeof(float);
    }
    return;
}
