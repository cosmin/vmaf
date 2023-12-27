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

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, int width, int height)
{
    int dst_px_stride = dwt2_dst->stride / sizeof(float);
    float filter_coeff = 1/sqrtf(2);

    float *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);

    float *band_a = dwt2_dst->bands[0];
    float *band_h = dwt2_dst->bands[1];
    float *band_v = dwt2_dst->bands[2];
    float *band_d = dwt2_dst->bands[3];

    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
    for (int i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(int j=0; j<width; ++j){
            tmplo[j] = filter_coeff * (src[(row_idx0)*width+j] + src[(row_idx1)*width+j]);
            tmphi[j] = filter_coeff * (src[(row_idx0)*width+j] - src[(row_idx1)*width+j]);
        }

        /* Horizontal pass (lo and hi). */
        for(int j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            band_a[i*dst_px_stride+j] = filter_coeff * (tmplo[col_idx0] + tmplo[col_idx1]);
            band_h[i*dst_px_stride+j] = filter_coeff * (tmphi[col_idx0] + tmphi[col_idx1]);
            band_v[i*dst_px_stride+j] = filter_coeff * (tmplo[col_idx0] - tmplo[col_idx1]);
            band_d[i*dst_px_stride+j] = filter_coeff * (tmphi[col_idx0] - tmphi[col_idx1]);
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
    float filter_coeff = 1/sqrtf(2);

    float *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);

    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
    for (int i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(int j=0; j<width; ++j){

            tmplo[j] = filter_coeff * (src[(row_idx0)*width+j] + src[(row_idx1)*width+j]);
        }

        /* Horizontal pass (lo and hi). */
        for(int j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            band_a[i*dst_px_stride+j] = filter_coeff * (tmplo[col_idx0] + tmplo[col_idx1]);
        }
    }
    aligned_free(tmplo);
}

const float nadeanu_filter_coeffs[5] = {0.0253133196, 0.2310067710, 0.4759767950, 0.2310067710,
                                            0.0253133196 };

const float ngan_filter_coeffs[21] = {
        -0.01373463642215844680849 ,
        -0.01608514932055564797264 ,
        -0.01890698454168517061991 ,
        -0.02215701978091480159327 ,
        -0.02546262290256656735110 ,
        -0.02742964751138579973522 ,
        -0.02361034173470941133210 ,
        -0.00100995586322411641696 ,
         0.07137023192756482281585 ,
         0.22121922236871877087694 ,
         0.32798240221262831006754 ,
         0.22121922236871877087694 ,
         0.07137023192756482281585 ,
        -0.00100995586322411641696 ,
        -0.02361034173470941133210 ,
        -0.02742964751138579973522 ,
        -0.02546262290256656735110 ,
        -0.02215701978091480159327 ,
        -0.01890698454168517061991 ,
        -0.01608514932055564797264 ,
        -0.01373463642215844680849 };
    
void spatial_csfs(float *src, float *dst, int width, int height, float *tmp_buf, char *spatial_csf_filter)
{
    const float *filter_coeffs;
    int filter_size;
    if(strcmp(spatial_csf_filter, "nadenau_spat") == 0) {
        /*coefficients for 5 tap nadeanu_spat filter */
        filter_coeffs = nadeanu_filter_coeffs;
        filter_size = 5;
    } else {
        filter_coeffs = ngan_filter_coeffs;
        filter_size = 21;
    }
    int src_px_stride = width;
    int dst_px_stride = width;

    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for(i = 0; i < height; ++i) {
        /* Vertical pass. */
        for(j = 0; j < width; ++j) {
            double accum = 0;

            for(fi = 0; fi < filter_size; ++fi) {
                fcoeff = filter_coeffs[fi];

                ii = i - filter_size / 2 + fi;
                ii = ii < 0 ? -(ii + 1) : (ii >= height ? 2 * height - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += (double) fcoeff * imgcoeff;
            }
            tmp_buf[j] = accum;
        }

        /* Horizontal pass. */
        for(j = 0; j < width; ++j) {
            double accum = 0;

            for(fj = 0; fj < filter_size; ++fj) {
                fcoeff = filter_coeffs[fj];

                jj = j - filter_size / 2 + fj;
                jj = jj < 0 ? -(jj + 1) : (jj >= width ? 2 * width - jj - 1 : jj);

                imgcoeff = tmp_buf[jj];

                accum += (double) fcoeff * imgcoeff;
            }
            dst[i * dst_px_stride + j] = accum;
        }
    }

    return;
}

void normalize_bitdepth(float *src, float *dst, int scaler, ptrdiff_t dst_stride, int width, int height)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst[j] = src[j] / scaler;
        }
        dst += dst_stride / sizeof(float);
        src += dst_stride / sizeof(float);
    }
    return;
}

void reflect_pad_for_input(const float *src, float *dst, int width, int height, int reflect_width, int reflect_height)
{
   size_t out_width = width + 2 * reflect_width;
   size_t out_height = height + 2 * reflect_height;

   for (size_t i = reflect_height; i != (out_height - reflect_height); i++) {

       for (int j = 0; j != reflect_width; j++)
       {
           dst[i * out_width + (reflect_width - 1 - j)] = src[(i - reflect_height) * width + j + 1];
       }

       memcpy(&dst[i * out_width + reflect_width], &src[(i - reflect_height) * width], sizeof(float) * width);

       for (int j = 0; j != reflect_width; j++)
           dst[i * out_width + out_width - reflect_width + j] = dst[i * out_width + out_width - reflect_width - 2 - j];
   }

  for (int i = 0; i != reflect_height; i++) {
      memcpy(&dst[(reflect_height - 1) * out_width - i * out_width], &dst[reflect_height * out_width + (i + 1) * out_width], sizeof(float) * out_width);
      memcpy(&dst[(out_height - reflect_height) * out_width + i * out_width], &dst[(out_height - reflect_height - 1) * out_width - (i + 1) * out_width], sizeof(float) * out_width);
  }
}

void funque_dwt2_inplace_csf(const dwt2buffers *src, float factors[4], int min_theta, int max_theta)
{
    float *src_ptr;
    float *dst_ptr;

    /* put these in theta format where 0 = approx, 1 = vertical, 2 = diagonal, 3 = horizontal */
    float *angles[4] = { src->bands[0], src->bands[2], src->bands[3], src->bands[1]};

    int px_stride = src->stride / sizeof(float);

    /* The computation of the csf values is not required for the regions which lie outside the frame borders */
    int left = 0;
    int top = 0;
    int right = src->width;
    int bottom = src->height;

    int i, j, theta, src_offset, dst_offset;
    float dst_val;

    for (theta = min_theta; theta <= max_theta; ++theta) {
        src_ptr = angles[theta];
        dst_ptr = angles[theta];

        for (i = top; i < bottom; ++i) {
            src_offset = i * px_stride;
            dst_offset = i * px_stride;

            for (j = left; j < right; ++j) {
                dst_val = factors[theta] * src_ptr[src_offset + j];
                dst_ptr[dst_offset + j] = dst_val;
            }
        }
    }
}
