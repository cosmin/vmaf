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
#include <time.h>

void integer_funque_dwt2(spat_fil_output_dtype *src, ptrdiff_t src_stride, i_dwt2buffers *dwt2_dst,
                         ptrdiff_t dst_stride, int width, int height, int spatial_csf, int level)
{
    int src_px_stride = src_stride / sizeof(dwt2_dtype);
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    int8_t const_2_wl0 = 1;
    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    int8_t filter_shift_rnd = 1 << (filter_shift - 1);
    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    int8_t filter_shift_lcpad_rnd = 1 << (filter_shift_lcpad - 1);

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];

    int16_t row_idx0, row_idx1, col_idx0;
	// int16_t col_idx1;
	int row0_offset, row1_offset;
    // int64_t accum;
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;

    int i, j;

    /* In Wavelet function level 0, 1, 2 would fit in 16-bit variable. For level 3 one right shift
     * is required */
    if(spatial_csf == 0) {
        if(level != 3) {
            filter_shift = 0;
            filter_shift_rnd = 0;
            const_2_wl0 = 2;
            filter_shift_lcpad = 0;
            filter_shift_lcpad_rnd = 0;
        }
    }

    for (i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;
        row0_offset = (row_idx0) *src_px_stride;
        row1_offset = (row_idx1) *src_px_stride;

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
			int32_t src_a_p_b = src_a + src_b;
			int32_t src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int32_t src_c_p_d = src_c + src_d;
			int32_t src_c_m_d = src_c - src_d;
			
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift);
			
			//F* F (a - b + c - d) - band H  (F*F is 1/2)
            band_h[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_m_b + src_c_m_d) + filter_shift_rnd) >> filter_shift);
			
			//F* F (a + b - c + d) - band V  (F*F is 1/2)
            band_v[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b - src_c_p_d) + filter_shift_rnd) >> filter_shift);

			//F* F (a - b - c - d) - band D  (F*F is 1/2)
            band_d[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_m_b - src_c_m_d) + filter_shift_rnd) >> filter_shift);		
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
			
            //F* F (a + b + a + b) - band A  (F*F is 1/2)
            band_a[i * dst_px_stride + j] =
                (dwt2_dtype) ((src_a_p_b * const_2_wl0 + filter_shift_lcpad_rnd) >>
                              filter_shift_lcpad);

            //F* F (a - b + a - b) - band H  (F*F is 1/2)
            band_h[i * dst_px_stride + j] =
                (dwt2_dtype) ((src_a_m_b * const_2_wl0 + filter_shift_lcpad_rnd) >>
                              filter_shift_lcpad);

            //F* F (a + b - (a + b)) - band V, Last column V will always be 0            
            band_v[i*dst_px_stride+j] = 0;

			//F* F (a - b - (a -b)) - band D,  Last column D will always be 0
            band_d[i*dst_px_stride+j] = 0;
        }
    }
}

void integer_funque_vifdwt2_band0(dwt2_dtype *src, dwt2_dtype *band_a, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);

    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    const int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    const int8_t filter_shift_rnd = 1<<(filter_shift - 1);
    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    const int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    const int8_t filter_shift_lcpad_rnd = 1<<(filter_shift_lcpad - 1);

    int16_t row_idx0, row_idx1, col_idx0;
	// int16_t col_idx1;
	int row0_offset, row1_offset;
    // int64_t accum;
	int width_div_2 = width >> 1; // without rounding (last value is handle outside)
	int last_col = width & 1;

    int i, j;

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
			int32_t src_a_p_b = src_a + src_b;

			// int32_t src_a_m_b = src_a - src_b;
			
			//c + d	& c - d
			int32_t src_c_p_d = src_c + src_d;
			// int32_t src_c_m_d = src_c - src_d;

			
			//F* F (a + b + c + d) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) (((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift);
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
			
            //F* F (a + b + a + b) - band A  (F*F is 1/2)
			band_a[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_p_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
        }
    }
}

/**
 * This function applies intermediate horizontal pass filter inside spatial filter
 */

static inline void integer_horizontal_filter(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst,
                                             const spat_fil_coeff_dtype *i_filter_coeffs, int width,
                                             int filter_size, int dst_row_idx, int half_fw)
{
    int j, fj, jj, jj1, jj2;
    /**
     * Similar to vertical pass the loop is split into 3 parts
     * This is to avoid the if conditions used for virtual padding
     */
    for (j = 0; j < half_fw; j++)
    {
        int pro_j_end  = half_fw - j - 1;
        int diff_j_hfw = j - half_fw;
        spat_fil_accum_dtype accum = 0;
        /**
         * The full loop is from fj = 0 to filter_size
         * During the loop when the centre pixel is at j,
         * the left part is available only till j-(filter_size/2) >= 0,
         * hence padding (border mirroring) is required when j-filter_size/2 < 0
         */
        // This loop does border mirroring (jj = -(j - filter_size/2 + fj + 1))
        for (fj = 0; fj <= pro_j_end; fj++){

            jj = pro_j_end - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        // Here the normal loop is executed where jj = j - filter_size/2 + fj
        for(; fj < filter_size; fj++) {
            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
    }

    //This is the core loop
    for ( ; j < (width - half_fw); j++)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
        spat_fil_accum_dtype accum = 0;
        /**
         * The filter coefficients are symmetric, 
         * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
         * The centre pixel is multiplied and accumulated outside the loop
        */
        for (fj = 0; fj < half_fw; fj++){

            jj1 = f_l_j + fj;
            jj2 = f_r_j - fj;
            accum += i_filter_coeffs[fj] * ((spat_fil_accum_dtype)tmp[jj1] + tmp[jj2]); //Since filter coefficients are symmetric
        }
        accum += (spat_fil_inter_dtype) i_filter_coeffs[half_fw] * tmp[j];
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
    }
    /**
     * This loop is to handle virtual padding of the right border pixels
     */
    for ( ; j < width; j++)
    {
        int diff_j_hfw = j - half_fw;
        int epi_last_j = width - diff_j_hfw;
        int epi_mirr_j = (width<<1) - diff_j_hfw - 1;
        spat_fil_accum_dtype accum = 0;
        /**
         * The full loop is from fj = 0 to filter_size
         * During the loop when the centre pixel is at j,
         * the right pixels are available only till j+(filter_size/2) < width,
         * hence padding (border mirroring) is required when j+(filter_size/2) >= width
         */
        // Here the normal loop is executed where jj = j - filter_size/2 + fj
        for (fj = 0; fj < epi_last_j; fj++){

            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        // This loop does border mirroring (jj = 2*width - (j - filter_size/2 + fj) - 1)
        for(; fj < filter_size; fj++) {
            jj = epi_mirr_j - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);

    }

}

const spat_fil_coeff_dtype i_ngan_filter_coeffs[21] = {
    -900,  -1054, -1239, -1452, -1669, -1798, -1547, -66,   4677,  14498, 21495,
    14498, 4677,  -66,   -1547, -1798, -1669, -1452, -1239, -1054, -900};

const spat_fil_coeff_dtype i_nadeanu_filter_coeffs[5] = {1658, 15139, 31193, 15139, 1658};

void integer_spatial_filter(void *src, spat_fil_output_dtype *dst, int dst_stride, int width,
                            int height, int bitdepth, spat_fil_inter_dtype *tmp,
                            char *spatial_csf_filter)
{
    int filter_size;
    const spat_fil_coeff_dtype *i_filter_coeffs;

    if(strcmp(spatial_csf_filter, "nadenau_spat") == 0) {
        filter_size = 5;
        i_filter_coeffs = i_nadeanu_filter_coeffs;
    } else if(strcmp(spatial_csf_filter, "ngan_spat") == 0) {
        filter_size = 21;
        i_filter_coeffs = i_ngan_filter_coeffs;
    }

    int src_px_stride = width;
    int dst_px_stride = dst_stride / sizeof(spat_fil_output_dtype);

    // spat_fil_inter_dtype imgcoeff;
	uint8_t *src_8b = NULL;
	uint16_t *src_hbd = NULL;
	
	int interim_rnd = 0, interim_shift = 0;

    int i, j, fi, ii, ii1, ii2;
	// int fj, jj, jj1, jj;
    // spat_fil_coeff_dtype *coeff_ptr;
    int half_fw = filter_size / 2;

    if(8 == bitdepth)
	{
		src_8b = (uint8_t*)src;
		src_hbd = NULL;
		interim_rnd = SPAT_FILTER_INTER_RND;
		interim_shift = SPAT_FILTER_INTER_SHIFT;
	}
	else // HBD case
	{
		src_8b = NULL;
		src_hbd = (uint16_t*)src;		
		interim_shift = SPAT_FILTER_INTER_SHIFT + (bitdepth - 8);
		interim_rnd = (1 << (interim_shift - 1));
	}

    /**
     * The loop i=0 to height is split into 3 parts
     * This is to avoid the if conditions used for virtual padding
     */
    for (i = 0; i < half_fw; i++){

        int diff_i_halffw = i - half_fw;
        int pro_mir_end = -diff_i_halffw - 1;

        /* Vertical pass. */

		if(8 == bitdepth)
		{
			for (j = 0; j < width; j++){

				spat_fil_accum_dtype accum = 0;

                /**
                 * The full loop is from fi = 0 to filter_size
                 * During the loop when the centre pixel is at i,
                 * the top part is available only till i-(filter_size/2) >= 0,
                 * hence padding (border mirroring) is required when i-filter_size/2 < 0
                 */
                // This loop does border mirroring (ii = -(i - filter_size/2 + fi + 1))
                for (fi = 0; fi <= pro_mir_end; fi++){

					ii = pro_mir_end - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
                // Here the normal loop is executed where ii = i - filter_size / 2 + fi
                for(; fi < filter_size; fi++) {
                    ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                }
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (j = 0; j < width; j++)
			{

				spat_fil_accum_dtype accum = 0;

                /**
                 * The full loop is from fi = 0 to filter_size
                 * During the loop when the centre pixel is at i,
                 * the top part is available only till i-(filter_size/2) >= 0,
                 * hence padding (border mirroring) is required when i-filter_size/2 < 0
                 */
                // This loop does border mirroring (ii = -(i - filter_size/2 + fi + 1))
                for (fi = 0; fi <= pro_mir_end; fi++){

					ii = pro_mir_end - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}
                // Here the normal loop is executed where ii = i - filter_size / 2 + fi
                for(; fi < filter_size; fi++) {
                    ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
                }
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}

        /* Horizontal pass. common for 8bit and hbd cases */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, filter_size, i * dst_px_stride,
                                  half_fw);
    }
    //This is the core loop
    for ( ; i < (height - half_fw); i++){

        int f_l_i = i - half_fw;
        int f_r_i = i + half_fw;
        /* Vertical pass. */

		if(8 == bitdepth)
		{
			for (j = 0; j < width; j++){

				spat_fil_accum_dtype accum = 0;

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi++){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src_8b[ii1 * src_px_stride + j] + src_8b[ii2 * src_px_stride + j]);
				}
				accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[i * src_px_stride + j];
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (j = 0; j < width; j++){

				spat_fil_accum_dtype accum = 0;

				/**
				 * The filter coefficients are symmetric, 
				 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				 * The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi++){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src_hbd[ii1 * src_px_stride + j] + src_hbd[ii2 * src_px_stride + j]);
				}
				accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[i * src_px_stride + j];
				tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
			
		}

        /* Horizontal pass. common for 8bit and hbd cases */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, filter_size, i * dst_px_stride,
                                  half_fw);
    }
    /**
     * This loop is to handle virtual padding of the bottom border pixels
     */
    for (; i < height; i++){

        int diff_i_halffw = i - half_fw;
        int epi_mir_i = 2 * height - diff_i_halffw - 1;
        int epi_last_i  = height - diff_i_halffw;
        
        /* Vertical pass. */

		if(8 == bitdepth)
		{
			for (j = 0; j < width; j++){

				spat_fil_accum_dtype accum = 0;

                /**
                 * The full loop is from fi = 0 to filter_size
                 * During the loop when the centre pixel is at i,
                 * the bottom pixels are available only till i+(filter_size/2) < height,
                 * hence padding (border mirroring) is required when i+(filter_size/2) >= height
                 */
                // Here the normal loop is executed where ii = i - filter_size/2 + fi
                for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
				}
                // This loop does border mirroring (ii = 2*height - (i - filter_size/2 + fi) - 1)
                for(; fi < filter_size; fi++) {
                    ii = epi_mir_i - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                }
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
		}
		else
		{
			for (j = 0; j < width; j++){

				spat_fil_accum_dtype accum = 0;

                /**
                 * The full loop is from fi = 0 to filter_size
                 * During the loop when the centre pixel is at i,
                 * the bottom pixels are available only till i+(filter_size/2) < height,
                 * hence padding (border mirroring) is required when i+(filter_size/2) >= height
                 */
                // Here the normal loop is executed where ii = i - filter_size/2 + fi
                for (fi = 0; fi < epi_last_i; fi++){

					ii = diff_i_halffw + fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
				}
                // This loop does border mirroring (ii = 2*height - (i - filter_size/2 + fi) - 1)
                for(; fi < filter_size; fi++) {
                    ii = epi_mir_i - fi;
					accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_hbd[ii * src_px_stride + j];
                }
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
			}
			
		}

        /* Horizontal pass. common for 8bit and hbd cases */
        integer_horizontal_filter(tmp, dst, i_filter_coeffs, width, filter_size, i * dst_px_stride,
                                  half_fw);
    }

    return;
}

void integer_reflect_pad_for_input_hbd(void *src, void *dst, int width, int height,
                                       int reflect_width, int reflect_height)
{
    size_t out_width = width + 2 * reflect_width;
    size_t out_height = height + 2 * reflect_height;
    uint16_t *src_hbd = (uint16_t *) src;
    uint16_t *dst_hbd = (uint16_t *) dst;

    for(size_t i = reflect_height; i != (out_height - reflect_height); i++) {
        for(int j = 0; j != reflect_width; j++) {
            dst_hbd[i * out_width + (reflect_width - 1 - j)] =
                src_hbd[(i - reflect_height) * width + j + 1];
        }

        memcpy(&dst_hbd[i * out_width + reflect_width], &src_hbd[(i - reflect_height) * width],
               sizeof(uint16_t) * width);

        for(int j = 0; j != reflect_width; j++)
            dst_hbd[i * out_width + out_width - reflect_width + j] =
                dst_hbd[i * out_width + out_width - reflect_width - 2 - j];
    }

    for(int i = 0; i != reflect_height; i++) {
        memcpy(&dst_hbd[(reflect_height - 1) * out_width - i * out_width],
               &dst_hbd[reflect_height * out_width + (i + 1) * out_width],
               sizeof(uint16_t) * out_width);
        memcpy(&dst_hbd[(out_height - reflect_height) * out_width + i * out_width],
               &dst_hbd[(out_height - reflect_height - 1) * out_width - (i + 1) * out_width],
               sizeof(uint16_t) * out_width);
    }
}

void integer_reflect_pad_for_input(void *src, void *dst, int width, int height, int reflect_width,
                                   int reflect_height, int bpc)
{
    if(bpc > 8) {
        integer_reflect_pad_for_input_hbd(src, dst, width, height, reflect_width, reflect_height);
        return;
    }
    size_t out_width = width + 2 * reflect_width;
    size_t out_height = height + 2 * reflect_height;
    uint8_t *src_8bd = (uint8_t *) src;
    uint8_t *dst_8bd = (uint8_t *) dst;

    for(size_t i = reflect_height; i != (out_height - reflect_height); i++) {
        for(int j = 0; j != reflect_width; j++) {
            dst_8bd[i * out_width + (reflect_width - 1 - j)] =
                src_8bd[(i - reflect_height) * width + j + 1];
        }

        memcpy(&dst_8bd[i * out_width + reflect_width], &src_8bd[(i - reflect_height) * width],
               sizeof(uint8_t) * width);

        for(int j = 0; j != reflect_width; j++)
            dst_8bd[i * out_width + out_width - reflect_width + j] =
                dst_8bd[i * out_width + out_width - reflect_width - 2 - j];
    }

    for(int i = 0; i != reflect_height; i++) {
        memcpy(&dst_8bd[(reflect_height - 1) * out_width - i * out_width],
               &dst_8bd[reflect_height * out_width + (i + 1) * out_width],
               sizeof(uint8_t) * out_width);
        memcpy(&dst_8bd[(out_height - reflect_height) * out_width + i * out_width],
               &dst_8bd[(out_height - reflect_height - 1) * out_width - (i + 1) * out_width],
               sizeof(uint8_t) * out_width);
    }
}

void integer_funque_dwt2_inplace_csf_c(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4],
                                     int min_theta, int max_theta, uint16_t interim_rnd_factors[4],
                                     uint8_t interim_shift_factors[4], int level)
{
    UNUSED(level);
    dwt2_dtype *src_ptr;
    dwt2_dtype *dst_ptr;

    /* put these in theta format where 0 = approx, 1 = vertical, 2 = diagonal, 3 = horizontal */
    dwt2_dtype *angles[4] = {src->bands[0], src->bands[2], src->bands[3], src->bands[1]};

    int px_stride = src->stride / sizeof(dwt2_dtype);

    /* The computation of the csf values is not required for the regions which lie outside the frame
     * borders */
    int left = 0;
    int top = 0;
    int right = src->width;
    int bottom = src->height;

    int i, j, theta, src_offset, dst_offset;
    spat_fil_accum_dtype mul_val;
    dwt2_dtype dst_val;

    for(theta = min_theta; theta <= max_theta; ++theta) {
        src_ptr = angles[theta];
        dst_ptr = angles[theta];

        for(i = top; i < bottom; ++i) {
            src_offset = i * px_stride;
            dst_offset = i * px_stride;

            for(j = left; j < right; ++j) {
                mul_val = (spat_fil_accum_dtype) factors[theta] * src_ptr[src_offset + j];
                dst_val = (dwt2_dtype) ((mul_val + interim_rnd_factors[theta]) >>
                                        interim_shift_factors[theta]);
                dst_ptr[dst_offset + j] = dst_val;
            }
        }
    }
}
