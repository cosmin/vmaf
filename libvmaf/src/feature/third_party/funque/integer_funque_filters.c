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

void int16_frame_to_csv(int16_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%d,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void integer_funque_dwt2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT
    const int64_t filter_coeff_sq = 23170 * 23170; // square is used in the final stage

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
	int row0_offset, row1_offset;
    int64_t accum;
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;
    unsigned i, j;
    for (i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;
		row0_offset = (row_idx0)*width;
		row1_offset = (row_idx1)*width;
        
        for(j=0; j< width_div_2; ++j)
		{
			
			int col_idx0 = (j << 1);
			int col_idx1 = (j << 1) + 1;
			
			// a & b 2 values in adjacent rows at the same coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			// c & d are adjacent values to a & b in teh same row
			spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
			spat_fil_output_dtype src_d = src[row1_offset + col_idx1];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			int src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int src_c_p_d = src_c + src_d;
			int src_c_m_d = src_c - src_d;
			
			//F* F (a + b + c + d) - band A
			accum = filter_coeff_sq * (src_a_p_b + src_c_p_d);
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);
			
			//F* F (a - b + c - d) - band H
            accum = filter_coeff_sq * (src_a_m_b + src_c_m_d);
            band_h[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);
			
			//F* F (a + b - c + d) - band V
            accum = filter_coeff_sq * (src_a_p_b - src_c_p_d);
            band_v[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);

			//F* F (a - b - c - d) - band D
            accum = filter_coeff_sq * (src_a_m_b - src_c_m_d);
            band_d[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);		
        }

        if(last_col)
        {
			col_idx0 = width_div_2 << 1;
			j = width_div_2;
			
			// a & b 2 values in adjacent rows at the last coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			int src_a_m_b = src_a - src_b;
			
            //F* F (a + b + a + b) - band A
			accum = filter_coeff_sq * (src_a_p_b << 1);
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);
			
			//F* F (a - b + a - b) - band H
            accum = filter_coeff_sq * (src_a_m_b << 1);
            band_h[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);
			
			//F* F (a + b - (a + b)) - band V            
            band_v[i*dst_px_stride+j] = 0;

			//F* F (a - b - (a -b)) - band D           
            band_d[i*dst_px_stride+j] = 0;
        }
    }
}

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT
    const int64_t filter_coeff_sq = 23170 * 23170; // square is used in the final stage

    int16_t row_idx0, row_idx1, col_idx0, col_idx1;
	int row0_offset, row1_offset;
    int64_t accum;
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;
    unsigned i, j;
    
    for (i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;
		row0_offset = (row_idx0)*width;
		row1_offset = (row_idx1)*width;
        
        for(j=0; j< width_div_2; ++j)
		{
			
			int col_idx0 = (j << 1);
			int col_idx1 = (j << 1) + 1;
			
			// a & b 2 values in adjacent rows at the same coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			// c & d are adjacent values to a & b in teh same row
			spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
			spat_fil_output_dtype src_d = src[row1_offset + col_idx1];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			int src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int src_c_p_d = src_c + src_d;
			int src_c_m_d = src_c - src_d;
			
			//F* F (a + b + c + d) - band A
			accum = filter_coeff_sq * (src_a_p_b + src_c_p_d);
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);	
        }

        if(last_col)
        {
			col_idx0 = width_div_2 << 1;
			j = width_div_2;
			
			// a & b 2 values in adjacent rows at the last coloumn
			spat_fil_output_dtype src_a = src[row0_offset+ col_idx0];
			spat_fil_output_dtype src_b = src[row1_offset+ col_idx0];
			
			//a + b	& a - b	
			int src_a_p_b = src_a + src_b;
			int src_a_m_b = src_a - src_b;
			
            //F* F (a + b + a + b) - band A
			accum = filter_coeff_sq * (src_a_p_b << 1);
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((accum + DWT2_OUT_RND)>> DWT2_OUT_SHIFT);
        }
    }
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
