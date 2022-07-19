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
#include "mem.h"
#include "offset.h"
#include "integer_funque_filters.h"

void integer_funque_dwt2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT
    const dwt2_dtype filter_coeff = 23170;

    dwt2_dtype *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)), MAX_ALIGN);
    dwt2_dtype *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)), MAX_ALIGN);

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];
    dwt2_accum_dtype accum;
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;

    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = (dwt2_accum_dtype)filter_coeff * (src[(row_idx0)*width+j] + src[(row_idx1)*width+j]);
            tmplo[j] = (dwt2_dtype) (accum >> DWT2_INTER_SHIFT);

            accum = (dwt2_accum_dtype)filter_coeff * (src[(row_idx0)*width+j] - src[(row_idx1)*width+j]);
            tmphi[j] = (dwt2_dtype) (accum >> DWT2_INTER_SHIFT);
        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            accum = (dwt2_accum_dtype)filter_coeff * (tmplo[col_idx0] + tmplo[col_idx1]);
            band_a[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = (dwt2_accum_dtype)filter_coeff * (tmphi[col_idx0] + tmphi[col_idx1]);
            band_h[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = (dwt2_accum_dtype) filter_coeff * (tmplo[col_idx0] - tmplo[col_idx1]);
            band_v[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = (dwt2_accum_dtype) filter_coeff * (tmphi[col_idx0] - tmphi[col_idx1]);
            band_d[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);
        }
    }
    aligned_free(tmplo);
    aligned_free(tmphi);
}

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT
    const dwt2_dtype filter_coeff = 23170;

    dwt2_dtype *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)), MAX_ALIGN);

    dwt2_accum_dtype accum;
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;

    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = (dwt2_accum_dtype)filter_coeff * (src[(row_idx0)*width+j] + src[(row_idx1)*width+j]);
            tmplo[j] = (dwt2_dtype) (accum >> DWT2_INTER_SHIFT);
        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            accum = (dwt2_accum_dtype)filter_coeff * (tmplo[col_idx0] + tmplo[col_idx1]);
            band_a[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);
        }
    }
    aligned_free(tmplo);
}

/**
 * This function applies intermediate horizontal pass filter inside spatial filter
 */
static inline void integer_horizontal_filter(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst, const spat_fil_coeff_dtype *i_filter_coeffs, int width, int height, int fwidth, int dst_row_idx, int half_fw)
{
    int j, fj, jj, jj1, jj2;

    for (j = 0; j < 10; j++)
    {
        int pro_j_end  = half_fw - j - 1;
        int diff_j_hfw = j - half_fw;
        spat_fil_accum_dtype accum = 0;
        //Mirroring using for loop
        for (fj = 0; fj <= pro_j_end; fj++){

            jj = pro_j_end - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        for ( ; fj < fwidth; fj++)
        {
            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) (accum >> SPAT_FILTER_OUT_SHIFT);
    }
    for ( ; j < (width - 10); j++)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
        spat_fil_accum_dtype accum = 0;
        for (fj = 0; fj < 10; fj++){

            jj1 = f_l_j + fj;
            jj2 = f_r_j - fj;
            accum += i_filter_coeffs[fj] * ((spat_fil_accum_dtype)tmp[jj1] + tmp[jj2]); //Since filter coefficients are symmetric
        }
        accum += (spat_fil_inter_dtype) i_filter_coeffs[10] * tmp[j];
        dst[dst_row_idx + j] = (spat_fil_output_dtype) (accum >> SPAT_FILTER_OUT_SHIFT);
    }
    for ( ; j < width; j++)
    {
        int diff_j_hfw = j - half_fw;
        int epi_last_j = width - diff_j_hfw;
        int epi_mirr_j = (width<<1) - diff_j_hfw - 1;
        spat_fil_accum_dtype accum = 0;
        for (fj = 0; fj < epi_last_j; fj++){

            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        for ( ; fj < fwidth; fj++)
        {
            jj = epi_mirr_j - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) (accum >> SPAT_FILTER_OUT_SHIFT);

    }

}

void integer_spatial_filter(uint8_t *src, spat_fil_output_dtype *dst, int width, int height)
{

    const spat_fil_coeff_dtype i_filter_coeffs[21] = {
        -900, -1054, -1239, -1452, -1669, -1798, -1547, -66, 4677, 14498, 21495,
        14498, 4677, -66, -1547, -1798, -1669, -1452, -1239, -1054, -900
    };

    int src_px_stride = width;
    int dst_px_stride = width;

    spat_fil_inter_dtype *tmp = aligned_malloc(ALIGN_CEIL(src_px_stride * sizeof(spat_fil_inter_dtype)), MAX_ALIGN);
    spat_fil_inter_dtype imgcoeff;

    int i, j, fi, fj, ii, jj, jj1, jj2, ii1, ii2;
    spat_fil_coeff_dtype *coeff_ptr;
    int fwidth = 21;
    int half_fw = fwidth / 2;
    for (i = 0; i < half_fw; i++){

        int diff_i_halffw = i - half_fw;
        int pro_mir_end = -diff_i_halffw - 1;

        /* Vertical pass. */
        for (j = 0; j < width; j++){

            spat_fil_accum_dtype accum = 0;

            //Mirroring using for loop itself
            for (fi = 0; fi <= pro_mir_end; fi++){

                ii = pro_mir_end - fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            for ( ; fi < fwidth; fi++)
            {
                ii = diff_i_halffw + fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            tmp[j] = (spat_fil_inter_dtype) (accum >> SPAT_FILTER_INTER_SHIFT);
        }

        /* Horizontal pass. */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, height, fwidth, i*dst_px_stride, half_fw);
    }
    for ( ; i < (height - half_fw); i++){

        int f_l_i = i - half_fw;
        int f_r_i = i + half_fw;
        /* Vertical pass. */
        for (j = 0; j < width; j++){

            spat_fil_accum_dtype accum = 0;

            //Mirroring using for loop itself
            for (fi = 0; fi < (half_fw); fi++){
                ii1 = f_l_i + fi;
                ii2 = f_r_i - fi;
                accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src[ii1 * src_px_stride + j] + src[ii2 * src_px_stride + j]);
            }
            accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[i * src_px_stride + j];
            tmp[j] = (spat_fil_inter_dtype) (accum >> SPAT_FILTER_INTER_SHIFT);
        }

        /* Horizontal pass. */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, height, fwidth, i*dst_px_stride, half_fw);
    }
    for (; i < height; i++){

        int diff_i_halffw = i - half_fw;
        int epi_mir_i = 2 * height - diff_i_halffw - 1;
        int epi_last_i  = height - diff_i_halffw;
        
        /* Vertical pass. */
        for (j = 0; j < width; j++){

            spat_fil_accum_dtype accum = 0;

            //Mirroring using for loop itself
            for (fi = 0; fi < epi_last_i; fi++){

                ii = diff_i_halffw + fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            for ( ; fi < fwidth; fi++)
            {
                ii = epi_mir_i - fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            tmp[j] = (spat_fil_inter_dtype) (accum >> SPAT_FILTER_INTER_SHIFT);
        }

        /* Horizontal pass. */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, height, fwidth, i*dst_px_stride, half_fw);
    }

    aligned_free(tmp);

    return;
}
