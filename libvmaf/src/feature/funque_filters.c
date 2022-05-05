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
// #include "vif_options.h"
// #include "vif_tools.h"
// #include <libvmaf/picture.h>
#include "funque_filters.h"

void funque_convolution(const float *f, const float *src, float *dst, float *tmpbuf, int w, int h, int src_stride, int dst_stride, int fwidth)
{

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    /* fall back */

    float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {

        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            double accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = f[fi];
                
                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -(ii+1)  : (ii >= h ? 2 * h - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += (double) fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = f[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -(jj+1) : (jj >= w ? 2 * w - jj - 1 : jj);

                imgcoeff = tmp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);
}

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    float filter_coeff_lo[2] = {0.707106781,  0.707106781};
    float filter_coeff_hi[2] = {0.707106781, -0.707106781};
    // float tmplo[10], tmphi[10];
    float *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(float)), MAX_ALIGN);
    
    float *band_a = dwt2_dst->bands[0];
    float *band_h = dwt2_dst->bands[1];
    float *band_v = dwt2_dst->bands[2];
    float *band_d = dwt2_dst->bands[3];
    float accum;
    // dst_stride = 5;
    // src->w[0] = 10;
    // src->h[0] = 10;
    // temp_a[11] = 0;
    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = 0;
            accum += filter_coeff_lo[0] * src[(2*i)*width+j];
            accum += filter_coeff_lo[1] * src[(2*i+1)*width+j];
            tmplo[j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * src[(2*i)*width+j];
            accum += filter_coeff_hi[1] * src[(2*i+1)*width+j];
            tmphi[j] = accum;
        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            accum = 0;
            accum += filter_coeff_lo[0] * tmplo[2*j];
            accum += filter_coeff_lo[1] * tmplo[2*j+1];
            band_a[i*dst_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_lo[0] * tmphi[2*j];
            accum += filter_coeff_lo[1] * tmphi[2*j+1];
            band_h[i*dst_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmplo[2*j];
            accum += filter_coeff_hi[1] * tmplo[2*j+1];
            band_v[i*dst_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmphi[2*j];
            accum += filter_coeff_hi[1] * tmphi[2*j+1];
            band_d[i*dst_stride+j] = accum;
        }
    }
}

void spatial_filter(float *src, float *dst, ptrdiff_t dst_stride, int offset, int width, int height)
{
    //Copied the coefficients from python
    float filtercoeff[21] = {-0.01373464, -0.01608515, -0.01890698, -0.02215702, -0.02546262, 
                             -0.02742965, -0.02361034, -0.00100996,  0.07137023,  0.22121922,
                             0.3279824 ,  0.22121922,  0.07137023, -0.00100996, -0.02361034,
                             -0.02742965, -0.02546262, -0.02215702, -0.01890698, -0.01608515,
                             -0.01373464};
    // float *float_data = dst;
    // uint8_t *data = src->data[0];
    // src->w[0] = src->w[0]/2;
    // src->h[0] = src->h[0]/2;
    int buf_stride = ALIGN_CEIL(width * sizeof(float));
    size_t buf_sz_one = (size_t)buf_stride * height;

    float *tmpbuf;
    char *data_top;
    float *data_buf = 0;
    float *tmp_srcbuf;
    if (!(data_buf = aligned_malloc(buf_sz_one * 2, 32)))
	{
		printf("error: aligned_malloc failed for data_buf.\n");
		fflush(stdout);
		return;
	}
    data_top = (char *)data_buf;
    tmpbuf = (float *)data_top; data_top+=buf_sz_one;
    tmp_srcbuf = (float *)data_top;

    // for (unsigned i = 0; i < src->h[0]; i++) {
    //     for (unsigned j = 0; j < src->w[0]; j++) {
    //         tmp_srcbuf[j] = (float) data[j] + offset;
    //     }
    //     tmp_srcbuf += dst_stride / sizeof(float);
    //     data += src->stride[0];
    // }

    funque_convolution(filtercoeff, src, dst, tmpbuf, width, height, buf_stride, buf_stride, 21);
    float temp_a[100];
    for(unsigned i = 0; i < 10; i++)
    {
        for(unsigned j=0; j<10; j++)
        {
            temp_a[i*10+j] = j;
        }
    }
    // float_data = temp_a;
    
    // printf("Printing band_a\n");
    // for(unsigned i=0; i<5; i++)
    // {
    //     for(unsigned j=0; j<5; j++)
    //     {
    //         printf("%f,", band_a[i*5+j]);
    //     }
    //     printf("\n");
    // }
    // printf("Printing band_h\n");
    // for(unsigned i=0; i<5; i++)
    // {
    //     for(unsigned j=0; j<5; j++)
    //     {
    //         printf("%f,", band_h[i*5+j]);
    //     }
    //     printf("\n");
    // }
    // printf("Printing band_v\n");
    // for(unsigned i=0; i<5; i++)
    // {
    //     for(unsigned j=0; j<5; j++)
    //     {
    //         printf("%f,", band_v[i*5+j]);
    //     }
    //     printf("\n");
    // }
    // printf("Printing band_d\n");
    // for(unsigned i=0; i<5; i++)
    // {
    //     for(unsigned j=0; j<5; j++)
    //     {
    //         printf("%f,", band_d[i*5+j]);
    //     }
    //     printf("\n");
    // }
    return;
}
