#include "../integer_funque_filters.h"
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

void integer_funque_dwt2_neon(spat_fil_output_dtype *src, ptrdiff_t src_stride,
                              i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height,
                              int spatial_csf, int level)
{
    int8_t FILTER_SHIFT = 1 + DWT2_OUT_SHIFT;
    int8_t FILTER_SHIFT_RND = 1 << (FILTER_SHIFT - 1);
    int8_t FILTER_SHIFT_LCPAD = 1 + DWT2_OUT_SHIFT - 1;
    int8_t FILTER_SHIFT_LCPAD_RND = 1 << (FILTER_SHIFT_LCPAD - 1);
    int8_t const_2_wl0 = 1;

    if(spatial_csf == 0)
    {
        if(level != 3)
        {
            FILTER_SHIFT = 0;
            FILTER_SHIFT_RND = 0;
            const_2_wl0 = 2;
            FILTER_SHIFT_LCPAD = 0;
            FILTER_SHIFT_LCPAD_RND = 0;
        }
    }

    int32x4_t const_2 = vdupq_n_s32(FILTER_SHIFT);
    int32x4_t shift_vector = vdupq_n_s32(-FILTER_SHIFT);

    // const int8_t FILTER_SHIFT = 1 + DWT2_OUT_SHIFT;
    // const int8_t FILTER_SHIFT_RND = 1 << (FILTER_SHIFT - 1);
    // const int8_t FILTER_SHIFT_LCPAD = 1 + DWT2_OUT_SHIFT - 1;
    // const int8_t FILTER_SHIFT_LCPAD_RND = 1 << (FILTER_SHIFT_LCPAD - 1);
    // int8_t const_2_wl0 = 1;

    int heightDiv2 = (height + 1) >> 1;
    int i, j, k, bandOffsetIdx;
    int row0_offset, row1_offset;
    int16_t row_idx0, row_idx1;
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    int src_px_stride = src_stride / sizeof(dwt2_dtype);
    // printf("this is there 362");
    // dst_px_stride = 2*dst_px_stride - src_px_stride;
    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];

    int32x4_t addAB_32x4_lo, addAB_32x4_hi, addCD_32x4_lo, addCD_32x4_hi;
    int32x4_t subAB_32x4_lo, subAB_32x4_hi, subCD_32x4_lo, subCD_32x4_hi;
    int32x4_t addA_32x4_lo, addA_32x4_hi, addH_32x4_lo, addH_32x4_hi;
    int32x4_t subV_32x4_lo, subV_32x4_hi, subD_32x4_lo, subD_32x4_hi;
    int16x4_t bandA_16x4_lo, bandA_16x4_hi, bandH_16x4_lo, bandH_16x4_hi;
    int16x4_t bandV_16x4_lo, bandV_16x4_hi, bandD_16x4_lo, bandD_16x4_hi;
    int16x8_t src0_16x8, src1_16x8, src2_16x8, src3_16x8;
    int16x8x2_t srcAC_16x8, srcBD_16x8;

    for(i = 0; i < heightDiv2; i++)
    {
        row_idx0 = 2 * i;
        row_idx1 = 2 * i + 1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2 * i;
        row0_offset = (row_idx0) *src_px_stride;
        row1_offset = (row_idx1) *src_px_stride;

        for(j = 0; j <= width - 16; j += 16)
        {
            src0_16x8 = vld1q_s16(src + row0_offset + j);      // A and C
            src1_16x8 = vld1q_s16(src + row1_offset + j);      // B and D
            src2_16x8 = vld1q_s16(src + row0_offset + j + 8);  // A and C with offset 8
            src3_16x8 = vld1q_s16(src + row1_offset + j + 8);  // B and D with offset 8

            srcAC_16x8 = vuzpq_s16(src0_16x8, src2_16x8);  // A in val[0] and C in val[1]
            srcBD_16x8 = vuzpq_s16(src1_16x8, src3_16x8);  // B in val[0] and D in val[1]

            addAB_32x4_lo =
                vaddl_s16(vget_low_s16(srcAC_16x8.val[0]), vget_low_s16(srcBD_16x8.val[0]));
            addAB_32x4_hi = vaddl_high_s16(srcAC_16x8.val[0], srcBD_16x8.val[0]);
            addCD_32x4_lo =
                vaddl_s16(vget_low_s16(srcAC_16x8.val[1]), vget_low_s16(srcBD_16x8.val[1]));
            addCD_32x4_hi = vaddl_high_s16(srcAC_16x8.val[1], srcBD_16x8.val[1]);

            subAB_32x4_lo =
                vsubl_s16(vget_low_s16(srcAC_16x8.val[0]), vget_low_s16(srcBD_16x8.val[0]));
            subAB_32x4_hi = vsubl_high_s16(srcAC_16x8.val[0], srcBD_16x8.val[0]);
            subCD_32x4_lo =
                vsubl_s16(vget_low_s16(srcAC_16x8.val[1]), vget_low_s16(srcBD_16x8.val[1]));
            subCD_32x4_hi = vsubl_high_s16(srcAC_16x8.val[1], srcBD_16x8.val[1]);

            addA_32x4_lo = vaddq_s32(addAB_32x4_lo, addCD_32x4_lo);
            addA_32x4_hi = vaddq_s32(addAB_32x4_hi, addCD_32x4_hi);
            addH_32x4_lo = vaddq_s32(subAB_32x4_lo, subCD_32x4_lo);
            addH_32x4_hi = vaddq_s32(subAB_32x4_hi, subCD_32x4_hi);

            bandA_16x4_lo =
                vmovn_s32(vshlq_s32(vaddq_s32(addA_32x4_lo, const_2),
                                    shift_vector));  // earlier commented out code was in use.
            bandA_16x4_hi = vmovn_s32(vshlq_s32(vaddq_s32(addA_32x4_hi, const_2), shift_vector));
            bandH_16x4_lo = vmovn_s32(vshlq_s32(vaddq_s32(addH_32x4_lo, const_2), shift_vector));
            bandH_16x4_hi = vmovn_s32(vshlq_s32(vaddq_s32(addH_32x4_hi, const_2), shift_vector));

            // bandA_16x4_lo = vqrshrn_n_s32(addA_32x4_lo, FILTER_SHIFT);
            // bandA_16x4_hi = vqrshrn_n_s32(addA_32x4_hi, FILTER_SHIFT);
            // bandH_16x4_lo = vqrshrn_n_s32(addH_32x4_lo, FILTER_SHIFT);
            // bandH_16x4_hi = vqrshrn_n_s32(addH_32x4_hi, FILTER_SHIFT);

            subV_32x4_lo = vsubq_s32(addAB_32x4_lo, addCD_32x4_lo);
            subV_32x4_hi = vsubq_s32(addAB_32x4_hi, addCD_32x4_hi);
            subD_32x4_lo = vsubq_s32(subAB_32x4_lo, subCD_32x4_lo);
            subD_32x4_hi = vsubq_s32(subAB_32x4_hi, subCD_32x4_hi);

            // bandV_16x4_lo = vqrshrn_n_s32(subV_32x4_lo, FILTER_SHIFT);
            // bandV_16x4_hi = vqrshrn_n_s32(subV_32x4_hi, FILTER_SHIFT);
            // bandD_16x4_lo = vqrshrn_n_s32(subD_32x4_lo, FILTER_SHIFT);
            // bandD_16x4_hi = vqrshrn_n_s32(subD_32x4_hi, FILTER_SHIFT);

            bandV_16x4_lo = vmovn_s32(vshlq_s32(vaddq_s32(subV_32x4_lo, const_2), shift_vector));
            bandV_16x4_hi = vmovn_s32(vshlq_s32(vaddq_s32(subV_32x4_hi, const_2), shift_vector));
            bandD_16x4_lo = vmovn_s32(vshlq_s32(vaddq_s32(subD_32x4_lo, const_2), shift_vector));
            bandD_16x4_hi = vmovn_s32(vshlq_s32(vaddq_s32(subD_32x4_hi, const_2), shift_vector));

            bandOffsetIdx = i * dst_px_stride + (j >> 1);
            vst1_s16(band_a + bandOffsetIdx, bandA_16x4_lo);
            vst1_s16(band_a + bandOffsetIdx + 4, bandA_16x4_hi);

            vst1_s16(band_h + bandOffsetIdx, bandH_16x4_lo);
            vst1_s16(band_h + bandOffsetIdx + 4, bandH_16x4_hi);

            vst1_s16(band_v + bandOffsetIdx, bandV_16x4_lo);
            vst1_s16(band_v + bandOffsetIdx + 4, bandV_16x4_hi);

            vst1_s16(band_d + bandOffsetIdx, bandD_16x4_lo);
            vst1_s16(band_d + bandOffsetIdx + 4, bandD_16x4_hi);
        }
        for(; j < width; j += 2)
        {
            if(j == width - 1)
            {
                int col_idx0 = j;
                k = (j >> 1);

                // a & b 2 values in adjacent rows at the last coloumn
                spat_fil_output_dtype src_a = src[row0_offset + col_idx0];
                spat_fil_output_dtype src_b = src[row1_offset + col_idx0];

                // a + b    & a - b
                int src_a_p_b = src_a + src_b;
                int src_a_m_b = src_a - src_b;

                // F* F (a + b + a + b) - band A  (F*F is 1/2)
                band_a[i * dst_px_stride + k] =
                    (dwt2_dtype) ((src_a_p_b * const_2_wl0 + FILTER_SHIFT_LCPAD_RND) >>
                                  FILTER_SHIFT_LCPAD);

                // F* F (a - b + a - b) - band H  (F*F is 1/2)
                band_h[i * dst_px_stride + k] =
                    (dwt2_dtype) ((src_a_m_b * const_2_wl0 + FILTER_SHIFT_LCPAD_RND) >>
                                  FILTER_SHIFT_LCPAD);

                // F* F (a + b - (a + b)) - band V, Last column V will always be 0
                band_v[i * dst_px_stride + k] = 0;

                // F* F (a - b - (a -b)) - band D,  Last column D will always be 0
                band_d[i * dst_px_stride + k] = 0;
            }
            else
            {
                int col_idx0 = j;
                int col_idx1 = j + 1;
                k = (j >> 1);

                // a & b 2 values in adjacent rows at the same coloumn
                spat_fil_output_dtype src_a = src[row0_offset + col_idx0];
                spat_fil_output_dtype src_b = src[row1_offset + col_idx0];

                // c & d are adjacent values to a & b in teh same row
                spat_fil_output_dtype src_c = src[row0_offset + col_idx1];
                spat_fil_output_dtype src_d = src[row1_offset + col_idx1];

                // a + b    & a - b
                int32_t src_a_p_b = src_a + src_b;
                int32_t src_a_m_b = src_a - src_b;

                // c + d    & c - d
                int32_t src_c_p_d = src_c + src_d;
                int32_t src_c_m_d = src_c - src_d;

                // F* F (a + b + c + d) - band A  (F*F is 1/2)
                band_a[i * dst_px_stride + k] =
                    (dwt2_dtype) (((src_a_p_b + src_c_p_d) + FILTER_SHIFT_RND) >> FILTER_SHIFT);

                // F* F (a - b + c - d) - band H  (F*F is 1/2)
                band_h[i * dst_px_stride + k] =
                    (dwt2_dtype) (((src_a_m_b + src_c_m_d) + FILTER_SHIFT_RND) >> FILTER_SHIFT);

                // F* F (a + b - c + d) - band V  (F*F is 1/2)
                band_v[i * dst_px_stride + k] =
                    (dwt2_dtype) (((src_a_p_b - src_c_p_d) + FILTER_SHIFT_RND) >> FILTER_SHIFT);

                // F* F (a - b - c - d) - band D  (F*F is 1/2)
                band_d[i * dst_px_stride + k] =
                    (dwt2_dtype) (((src_a_m_b - src_c_m_d) + FILTER_SHIFT_RND) >> FILTER_SHIFT);
            }
        }
    }
}

static inline void integer_horizontal_filter_neon(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst, const spat_fil_coeff_dtype *i_filter_coeffs, int width, int fwidth, int dst_row_idx, int half_fw)
{
    int j, fj, jj, jj1, jj2;
	int ker_wid = width - 2 * half_fw;
	// int ker_wid_8 = ker_wid - ker_wid%8;
	int ker_wid_rem = ker_wid%8;
	
	// int16x8_t src16x8_in0, src16x8_in1, src16x8_in2, src16x8_in3;
	int16x8_t src16x8_0, src16x8_1, src16x8_2, src16x8_3, src16x8_4, src16x8_5;
	int16x8_t src16x8_6, src16x8_7, src16x8_8, src16x8_9, src16x8_10, src16x8_11;
	int16x8_t src16x8_12, src16x8_13, src16x8_14, src16x8_15, src16x8_16;
	int16x8_t src16x8_17, src16x8_18, src16x8_19, src16x8_20;
	int32x4_t sum32x4_0_lo, sum32x4_0_hi, sum32x4_1_lo, sum32x4_1_hi;
	int32x4_t sum32x4_2_lo, sum32x4_2_hi, sum32x4_3_lo, sum32x4_3_hi;
	int32x4_t sum32x4_4_lo, sum32x4_4_hi, sum32x4_5_lo, sum32x4_5_hi;
	int32x4_t sum32x4_6_lo, sum32x4_6_hi, sum32x4_7_lo, sum32x4_7_hi;
	int32x4_t sum32x4_8_lo, sum32x4_8_hi, sum32x4_9_lo, sum32x4_9_hi;
	int32x4_t sum32x4_10_lo, sum32x4_10_hi, accum_0_lo, accum_0_hi;
	int16x4_t dst16x4_0, dst16x4_1;
	
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
         * The full loop is from fj = 0 to fwidth
         * During the loop when the centre pixel is at j, 
         * the left part is available only till j-(fwidth/2) >= 0, 
         * hence padding (border mirroring) is required when j-fwidth/2 < 0
         */
        //This loop does border mirroring (jj = -(j - fwidth/2 + fj + 1))
        for (fj = 0; fj <= pro_j_end; fj++){

            jj = pro_j_end - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        //Here the normal loop is executed where jj = j - fwidth/2 + fj
        for ( ; fj < fwidth; fj++)
        {
            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
    }

    //This is the core loop
	for ( ; j < (width - half_fw - ker_wid_rem); j += 8)
    {
        int f_l_j = j - half_fw;

		src16x8_0 = vld1q_s16(tmp + f_l_j);
		src16x8_20 = vld1q_s16(tmp + f_l_j + 20);
		src16x8_1 = vld1q_s16(tmp + f_l_j + 1);
		src16x8_19 = vld1q_s16(tmp + f_l_j + 19);
		src16x8_2 = vld1q_s16(tmp + f_l_j + 2);
		src16x8_18 = vld1q_s16(tmp + f_l_j + 18);
		
		sum32x4_0_lo = vaddl_s16(vget_low_s16(src16x8_0), vget_low_s16(src16x8_20));
		sum32x4_0_hi = vaddl_high_s16(src16x8_0, src16x8_20);
		sum32x4_1_lo = vaddl_s16(vget_low_s16(src16x8_1), vget_low_s16(src16x8_19));
		sum32x4_1_hi = vaddl_high_s16(src16x8_1, src16x8_19);
		sum32x4_2_lo = vaddl_s16(vget_low_s16(src16x8_2), vget_low_s16(src16x8_18));
		sum32x4_2_hi = vaddl_high_s16(src16x8_2, src16x8_18);
		
		src16x8_3 = vld1q_s16(tmp + f_l_j + 3);
		src16x8_17 = vld1q_s16(tmp + f_l_j + 17);
		src16x8_4 = vld1q_s16(tmp + f_l_j + 4);
		src16x8_16 = vld1q_s16(tmp + f_l_j + 16);
		src16x8_5 = vld1q_s16(tmp + f_l_j + 5);
		src16x8_15 = vld1q_s16(tmp + f_l_j + 15);
		
		sum32x4_3_lo = vaddl_s16(vget_low_s16(src16x8_3), vget_low_s16(src16x8_17));
		sum32x4_3_hi = vaddl_high_s16(src16x8_3, src16x8_17);
		sum32x4_4_lo = vaddl_s16(vget_low_s16(src16x8_4), vget_low_s16(src16x8_16));
		sum32x4_4_hi = vaddl_high_s16(src16x8_4, src16x8_16);
		sum32x4_5_lo = vaddl_s16(vget_low_s16(src16x8_5), vget_low_s16(src16x8_15));
		sum32x4_5_hi = vaddl_high_s16(src16x8_5, src16x8_15);
		
		src16x8_6 = vld1q_s16(tmp + f_l_j + 6);
		src16x8_14 = vld1q_s16(tmp + f_l_j + 14);
		src16x8_7 = vld1q_s16(tmp + f_l_j + 7);
		src16x8_13 = vld1q_s16(tmp + f_l_j + 13);
		src16x8_8 = vld1q_s16(tmp + f_l_j + 8);
		src16x8_12 = vld1q_s16(tmp + f_l_j + 12);
		
		sum32x4_6_lo = vaddl_s16(vget_low_s16(src16x8_6), vget_low_s16(src16x8_14));
		sum32x4_6_hi = vaddl_high_s16(src16x8_6, src16x8_14);
		sum32x4_7_lo = vaddl_s16(vget_low_s16(src16x8_7), vget_low_s16(src16x8_13));
		sum32x4_7_hi = vaddl_high_s16(src16x8_7, src16x8_13);
		sum32x4_8_lo = vaddl_s16(vget_low_s16(src16x8_8), vget_low_s16(src16x8_12));
		sum32x4_8_hi = vaddl_high_s16(src16x8_8, src16x8_12);
		
		
		src16x8_9 =  vld1q_s16(tmp + f_l_j + 9);
		src16x8_11 = vld1q_s16(tmp + f_l_j + 11);
		src16x8_10 = vld1q_s16(tmp + f_l_j + 10);

		sum32x4_9_lo = vaddl_s16(vget_low_s16(src16x8_9), vget_low_s16(src16x8_11));
		sum32x4_9_hi = vaddl_high_s16(src16x8_9, src16x8_11);
		sum32x4_10_lo = vmovl_s16(vget_low_s16(src16x8_10));
		sum32x4_10_hi = vmovl_high_s16(src16x8_10);
		
		accum_0_lo = vmulq_n_s32(sum32x4_10_lo, i_filter_coeffs[10]);
		accum_0_hi = vmulq_n_s32(sum32x4_10_hi, i_filter_coeffs[10]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_0_lo, i_filter_coeffs[0]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_0_hi, i_filter_coeffs[0]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_1_lo, i_filter_coeffs[1]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_1_hi, i_filter_coeffs[1]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_2_lo, i_filter_coeffs[2]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_2_hi, i_filter_coeffs[2]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_3_lo, i_filter_coeffs[3]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_3_hi, i_filter_coeffs[3]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_4_lo, i_filter_coeffs[4]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_4_hi, i_filter_coeffs[4]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_5_lo, i_filter_coeffs[5]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_5_hi, i_filter_coeffs[5]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_6_lo, i_filter_coeffs[6]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_6_hi, i_filter_coeffs[6]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_7_lo, i_filter_coeffs[7]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_7_hi, i_filter_coeffs[7]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_8_lo, i_filter_coeffs[8]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_8_hi, i_filter_coeffs[8]);
		accum_0_lo = vmlaq_n_s32(accum_0_lo, sum32x4_9_lo, i_filter_coeffs[9]);
		accum_0_hi = vmlaq_n_s32(accum_0_hi, sum32x4_9_hi, i_filter_coeffs[9]);
		
		dst16x4_0 = vqrshrn_n_s32(accum_0_lo, SPAT_FILTER_OUT_SHIFT);
		dst16x4_1 = vqrshrn_n_s32(accum_0_hi, SPAT_FILTER_OUT_SHIFT);
		
		vst1_s16(dst + dst_row_idx + j, dst16x4_0);
		vst1_s16(dst + dst_row_idx + j + 4, dst16x4_1);
		    
    }
	//scalar part of the inner loop
	
    for (j = width - half_fw - ker_wid_rem ; j < (width - half_fw); j++)
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
         * The full loop is from fj = 0 to fwidth
         * During the loop when the centre pixel is at j, 
         * the right pixels are available only till j+(fwidth/2) < width, 
         * hence padding (border mirroring) is required when j+(fwidth/2) >= width
         */
        //Here the normal loop is executed where jj = j - fwidth/2 + fj
        for (fj = 0; fj < epi_last_j; fj++){

            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        //This loop does border mirroring (jj = 2*width - (j - fwidth/2 + fj) - 1)
        for ( ; fj < fwidth; fj++)
        {
            jj = epi_mirr_j - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);

    }

}

void integer_spatial_filter_neon(void *void_src, spat_fil_output_dtype *dst, int width, int height, int bitdepth)
{
    const spat_fil_coeff_dtype i_filter_coeffs[21] = {
        -900, -1054, -1239, -1452, -1669, -1798, -1547, -66, 4677, 14498, 21495,
        14498, 4677, -66, -1547, -1798, -1669, -1452, -1239, -1054, -900
    };

    int src_px_stride = width;
    int dst_px_stride = width;
	int width_rem_size = width - (width%16);
    if (bitdepth != 8)
    {
        printf("ERROR: arm simd support for spatial filter is only for 8bit contents\n");
        return;
    }
    uint8_t *src = (uint8_t*) void_src;
    spat_fil_inter_dtype *tmp = malloc(src_px_stride * sizeof(spat_fil_inter_dtype));
    // spat_fil_inter_dtype imgcoeff;
    // spat_fil_inter_dtype *tmp_int = tmp;
    int i, j, fi, ii, ii1, ii2;
    // int fj, jj, jj1, jj2, 
    // spat_fil_coeff_dtype *coeff_ptr;
    int fwidth = 21;
    int half_fw = fwidth / 2;
	
	int16x8_t src16x8_0_lo, src16x8_0_hi, src16x8_1_lo, src16x8_1_hi;
	int16x8_t src16x8_2_lo, src16x8_2_hi, src16x8_3_lo, src16x8_3_hi;
	int16x8_t src16x8_4_lo, src16x8_4_hi, src16x8_5_lo, src16x8_5_hi;
	int16x8_t src16x8_6_lo, src16x8_6_hi, src16x8_7_lo, src16x8_7_hi;
	int16x8_t src16x8_8_lo, src16x8_8_hi, src16x8_9_lo, src16x8_9_hi;
	int16x8_t src16x8_i_lo, src16x8_i_hi;
	int32x4_t accum_0_lo, accum_0_hi, accum_1_lo, accum_1_hi;
    int16x4_t tmp16x4_0, tmp16x4_1, tmp16x4_2, tmp16x4_3 ;
	// uint8x16_t src8x16_00, src8x16_01, src8x16_10, src8x16_11;
	uint8x8_t src8x8_00_lo, src8x8_01_lo, src8x8_10_lo, src8x8_11_lo;
	uint8x8_t src8x8_20_lo, src8x8_21_lo, src8x8_30_lo, src8x8_31_lo;
	uint8x8_t src8x8_40_lo, src8x8_41_lo, src8x8_50_lo, src8x8_51_lo;
	uint8x8_t src8x8_60_lo, src8x8_61_lo, src8x8_70_lo, src8x8_71_lo;
	uint8x8_t src8x8_80_lo, src8x8_81_lo, src8x8_90_lo, src8x8_91_lo;
	uint8x8_t src8x8_00_hi, src8x8_01_hi, src8x8_10_hi, src8x8_11_hi;
	uint8x8_t src8x8_20_hi, src8x8_21_hi, src8x8_30_hi, src8x8_31_hi;
	uint8x8_t src8x8_40_hi, src8x8_41_hi, src8x8_50_hi, src8x8_51_hi;
	uint8x8_t src8x8_60_hi, src8x8_61_hi, src8x8_70_hi, src8x8_71_hi;
	uint8x8_t src8x8_80_hi, src8x8_81_hi, src8x8_90_hi, src8x8_91_hi;
	uint8x8_t src8x8_i_lo, src8x8_i_hi;
	
    /**
     * The loop i=0 to height is split into 3 parts
     * This is to avoid the if conditions used for virtual padding
     */
    for (i = 0; i < half_fw; i++){

        int diff_i_halffw = i - half_fw;
        int pro_mir_end = -diff_i_halffw - 1;

        /* Vertical pass. */
        for (j = 0; j < width; j++){

            spat_fil_accum_dtype accum = 0;

            /**
             * The full loop is from fi = 0 to fwidth
             * During the loop when the centre pixel is at i, 
             * the top part is available only till i-(fwidth/2) >= 0, 
             * hence padding (border mirroring) is required when i-fwidth/2 < 0
             */
            //This loop does border mirroring (ii = -(i - fwidth/2 + fi + 1))
            for (fi = 0; fi <= pro_mir_end; fi++){

                ii = pro_mir_end - fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            //Here the normal loop is executed where ii = i - fwidth / 2 + fi
            for ( ; fi < fwidth; fi++)
            {
                ii = diff_i_halffw + fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            tmp[j] = (spat_fil_inter_dtype) ((accum + SPAT_FILTER_INTER_RND) >> SPAT_FILTER_INTER_SHIFT);
        }
        /* Horizontal pass. */
        integer_horizontal_filter_neon(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }
	
    //This is the core loop
    for ( ; i < (height - half_fw); i++){

        int f_l_i = i - half_fw;
        int f_r_i = i + half_fw;
        /* Vertical pass. */
        for (j = 0; j < width_rem_size; j += 16){

            // spat_fil_accum_dtype accum = 0;
			
			src8x8_i_lo  = vld1_u8(src + i * src_px_stride + j); 
			src8x8_i_hi  = vld1_u8(src + i * src_px_stride + j + 8); 
			src16x8_i_lo = vreinterpretq_s16_u16(vmovl_u8(src8x8_i_lo));
			src16x8_i_hi = vreinterpretq_s16_u16(vmovl_u8(src8x8_i_hi));
			
            src8x8_00_lo = vld1_u8(src + f_l_i * src_px_stride + j); 
			src8x8_01_lo = vld1_u8(src + f_r_i * src_px_stride + j);
            src8x8_10_lo = vld1_u8(src + (f_l_i + 1) * src_px_stride + j); 
			src8x8_11_lo = vld1_u8(src + (f_r_i - 1) * src_px_stride + j);
			src8x8_00_hi = vld1_u8(src + f_l_i * src_px_stride + j + 8); 
			src8x8_01_hi = vld1_u8(src + f_r_i * src_px_stride + j + 8);
            src8x8_10_hi = vld1_u8(src + (f_l_i + 1) * src_px_stride + j + 8); 
			src8x8_11_hi = vld1_u8(src + (f_r_i - 1) * src_px_stride + j + 8);
			
			src16x8_0_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_00_lo, src8x8_01_lo));
			src16x8_0_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_00_hi, src8x8_01_hi));
			src16x8_1_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_10_lo, src8x8_11_lo));
			src16x8_1_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_10_hi, src8x8_11_hi));
			
			src8x8_20_lo = vld1_u8(src + (f_l_i + 2) * src_px_stride + j); 
			src8x8_21_lo = vld1_u8(src + (f_r_i - 2) * src_px_stride + j); 
			src8x8_30_lo = vld1_u8(src + (f_l_i + 3) * src_px_stride + j); 
			src8x8_31_lo = vld1_u8(src + (f_r_i - 3) * src_px_stride + j); 
			src8x8_20_hi = vld1_u8(src + (f_l_i + 2) * src_px_stride + j + 8); 
			src8x8_21_hi = vld1_u8(src + (f_r_i - 2) * src_px_stride + j + 8); 
			src8x8_30_hi = vld1_u8(src + (f_l_i + 3) * src_px_stride + j + 8); 
			src8x8_31_hi = vld1_u8(src + (f_r_i - 3) * src_px_stride + j + 8);
			
			src16x8_2_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_20_lo, src8x8_21_lo));
			src16x8_2_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_20_hi, src8x8_21_hi));
			src16x8_3_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_30_lo, src8x8_31_lo));
			src16x8_3_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_30_hi, src8x8_31_hi));

			src8x8_40_lo = vld1_u8(src + (f_l_i + 4) * src_px_stride + j); 
			src8x8_41_lo = vld1_u8(src + (f_r_i - 4) * src_px_stride + j); 
			src8x8_50_lo = vld1_u8(src + (f_l_i + 5) * src_px_stride + j); 
			src8x8_51_lo = vld1_u8(src + (f_r_i - 5) * src_px_stride + j);
			src8x8_40_hi = vld1_u8(src + (f_l_i + 4) * src_px_stride + j + 8); 
			src8x8_41_hi = vld1_u8(src + (f_r_i - 4) * src_px_stride + j + 8); 
			src8x8_50_hi = vld1_u8(src + (f_l_i + 5) * src_px_stride + j + 8); 
			src8x8_51_hi = vld1_u8(src + (f_r_i - 5) * src_px_stride + j + 8);

			src16x8_4_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_40_lo, src8x8_41_lo));
			src16x8_4_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_40_hi, src8x8_41_hi));
			src16x8_5_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_50_lo, src8x8_51_lo));
			src16x8_5_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_50_hi, src8x8_51_hi));
			
			src8x8_60_lo = vld1_u8(src + (f_l_i + 6) * src_px_stride + j); 
			src8x8_61_lo = vld1_u8(src + (f_r_i - 6) * src_px_stride + j); 	
			src8x8_70_lo = vld1_u8(src + (f_l_i + 7) * src_px_stride + j); 
			src8x8_71_lo = vld1_u8(src + (f_r_i - 7) * src_px_stride + j); 
			src8x8_60_hi = vld1_u8(src + (f_l_i + 6) * src_px_stride + j + 8); 
			src8x8_61_hi = vld1_u8(src + (f_r_i - 6) * src_px_stride + j + 8); 	
			src8x8_70_hi = vld1_u8(src + (f_l_i + 7) * src_px_stride + j + 8); 
			src8x8_71_hi = vld1_u8(src + (f_r_i - 7) * src_px_stride + j + 8); 
			
			src16x8_6_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_60_lo, src8x8_61_lo));
			src16x8_6_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_60_hi, src8x8_61_hi));
			src16x8_7_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_70_lo, src8x8_71_lo));
			src16x8_7_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_70_hi, src8x8_71_hi));
			
			src8x8_80_lo = vld1_u8(src + (f_l_i + 8) * src_px_stride + j); 
			src8x8_81_lo = vld1_u8(src + (f_r_i - 8) * src_px_stride + j); 
			src8x8_90_lo = vld1_u8(src + (f_l_i + 9) * src_px_stride + j); 
			src8x8_91_lo = vld1_u8(src + (f_r_i - 9) * src_px_stride + j); 
			src8x8_80_hi = vld1_u8(src + (f_l_i + 8) * src_px_stride + j + 8); 
			src8x8_81_hi = vld1_u8(src + (f_r_i - 8) * src_px_stride + j + 8); 
			src8x8_90_hi = vld1_u8(src + (f_l_i + 9) * src_px_stride + j + 8); 
			src8x8_91_hi = vld1_u8(src + (f_r_i - 9) * src_px_stride + j + 8); 
			
			src16x8_8_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_80_lo, src8x8_81_lo));
			src16x8_8_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_80_hi, src8x8_81_hi));
			src16x8_9_lo = vreinterpretq_s16_u16(vaddl_u8(src8x8_90_lo, src8x8_91_lo));
			src16x8_9_hi = vreinterpretq_s16_u16(vaddl_u8(src8x8_90_hi, src8x8_91_hi));
			
			accum_0_lo = vmull_n_s16(vget_low_s16(src16x8_i_lo), i_filter_coeffs[10]);
			accum_0_hi = vmull_high_n_s16(src16x8_i_lo, i_filter_coeffs[10]);
			accum_1_lo = vmull_n_s16(vget_low_s16(src16x8_i_hi), i_filter_coeffs[10]);
			accum_1_hi = vmull_high_n_s16(src16x8_i_hi, i_filter_coeffs[10]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_0_lo), i_filter_coeffs[0]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_0_lo, i_filter_coeffs[0]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_0_hi), i_filter_coeffs[0]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_0_hi, i_filter_coeffs[0]);
	
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_1_lo), i_filter_coeffs[1]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_1_lo, i_filter_coeffs[1]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_1_hi), i_filter_coeffs[1]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_1_hi, i_filter_coeffs[1]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_2_lo), i_filter_coeffs[2]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_2_lo, i_filter_coeffs[2]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_2_hi), i_filter_coeffs[2]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_2_hi, i_filter_coeffs[2]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_3_lo), i_filter_coeffs[3]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_3_lo, i_filter_coeffs[3]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_3_hi), i_filter_coeffs[3]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_3_hi, i_filter_coeffs[3]);
	
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_4_lo), i_filter_coeffs[4]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_4_lo, i_filter_coeffs[4]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_4_hi), i_filter_coeffs[4]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_4_hi, i_filter_coeffs[4]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_5_lo), i_filter_coeffs[5]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_5_lo, i_filter_coeffs[5]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_5_hi), i_filter_coeffs[5]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_5_hi, i_filter_coeffs[5]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_6_lo), i_filter_coeffs[6]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_6_lo, i_filter_coeffs[6]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_6_hi), i_filter_coeffs[6]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_6_hi, i_filter_coeffs[6]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_7_lo), i_filter_coeffs[7]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_7_lo, i_filter_coeffs[7]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_7_hi), i_filter_coeffs[7]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_7_hi, i_filter_coeffs[7]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_8_lo), i_filter_coeffs[8]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_8_lo, i_filter_coeffs[8]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_8_hi), i_filter_coeffs[8]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_8_hi, i_filter_coeffs[8]);
			
			accum_0_lo = vmlal_n_s16(accum_0_lo, vget_low_s16(src16x8_9_lo), i_filter_coeffs[9]);
			accum_0_hi = vmlal_high_n_s16(accum_0_hi, src16x8_9_lo, i_filter_coeffs[9]);
			accum_1_lo = vmlal_n_s16(accum_1_lo, vget_low_s16(src16x8_9_hi), i_filter_coeffs[9]);
			accum_1_hi = vmlal_high_n_s16(accum_1_hi, src16x8_9_hi, i_filter_coeffs[9]);
			
			tmp16x4_0 = vqrshrn_n_s32(accum_0_lo, SPAT_FILTER_INTER_SHIFT);
			tmp16x4_1 = vqrshrn_n_s32(accum_0_hi, SPAT_FILTER_INTER_SHIFT);
			tmp16x4_2 = vqrshrn_n_s32(accum_1_lo, SPAT_FILTER_INTER_SHIFT);
			tmp16x4_3 = vqrshrn_n_s32(accum_1_hi, SPAT_FILTER_INTER_SHIFT);
			
			vst1_s16(tmp + j, tmp16x4_0);
			vst1_s16(tmp + j + 4, tmp16x4_1);
			vst1_s16(tmp + j + 8, tmp16x4_2);
			vst1_s16(tmp + j + 12, tmp16x4_3);
		}
		//loop for the remaining pixels if width is not multiple of 16
		for (j = width_rem_size; j < width; j++)
		{
				spat_fil_accum_dtype accum = 0;
					/**
				* The filter coefficients are symmetric, 
				* hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
				* The centre pixel is multiplied and accumulated outside the loop
				*/
				for (fi = 0; fi < (half_fw); fi++){
					ii1 = f_l_i + fi;
					ii2 = f_r_i - fi;
					accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src[ii1 * src_px_stride + j] + src[ii2 * src_px_stride + j]);
				}
				accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[i * src_px_stride + j];
				tmp[j] = (spat_fil_inter_dtype) ((accum + SPAT_FILTER_INTER_RND) >> SPAT_FILTER_INTER_SHIFT);
		}
        /* Horizontal pass. */
        integer_horizontal_filter_neon(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }
    /**
     * This loop is to handle virtual padding of the bottom border pixels
     */
    for (; i < height; i++){

        int diff_i_halffw = i - half_fw;
        int epi_mir_i = 2 * height - diff_i_halffw - 1;
        int epi_last_i  = height - diff_i_halffw;
        
        /* Vertical pass. */
        for (j = 0; j < width; j++){

            spat_fil_accum_dtype accum = 0;

            /**
             * The full loop is from fi = 0 to fwidth
             * During the loop when the centre pixel is at i, 
             * the bottom pixels are available only till i+(fwidth/2) < height, 
             * hence padding (border mirroring) is required when i+(fwidth/2) >= height
             */
            //Here the normal loop is executed where ii = i - fwidth/2 + fi
            for (fi = 0; fi < epi_last_i; fi++){

                ii = diff_i_halffw + fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            //This loop does border mirroring (ii = 2*height - (i - fwidth/2 + fi) - 1)
            for ( ; fi < fwidth; fi++)
            {
                ii = epi_mir_i - fi;
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }
            tmp[j] = (spat_fil_inter_dtype) ((accum + SPAT_FILTER_INTER_RND) >> SPAT_FILTER_INTER_SHIFT);
        }
        /* Horizontal pass. */
        integer_horizontal_filter_neon(tmp, dst, i_filter_coeffs, width, fwidth, i*dst_px_stride, half_fw);
    }
	
    free(tmp);

    return;
}

static inline void integer_horizontal_5tap_filter_neon(spat_fil_inter_dtype *tmp, spat_fil_output_dtype *dst, const spat_fil_coeff_dtype *i_filter_coeffs, int width, int fwidth, int dst_row_idx, int half_fw)
{
    int j, fj, jj, jj1, jj2;    

    for (j = 0; j < half_fw; j++)
    {
        int pro_j_end  = half_fw - j - 1;
        int diff_j_hfw = j - half_fw;
        spat_fil_accum_dtype accum = 0;

        for (fj = 0; fj <= pro_j_end; fj++){

            jj = pro_j_end - fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
  
        for ( ; fj < fwidth; fj++)
        {
            jj = diff_j_hfw + fj;
            accum += (spat_fil_accum_dtype) i_filter_coeffs[fj] * tmp[jj];
        }
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
    }
    
    // neon variables
    int16x8_t src16x8_0,src16x8_1;
    int32x4_t add_l,add_h;
    int16x4_t half_round16x4_lb, half_round16x4_hb;
    
    int wid = (width - 2 * half_fw)%8;
    
    //This is the core loop
    for ( ; j<(width - half_fw - wid); j=j+8)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
        int32x4_t accum_lb=vdupq_n_s32(0);
        int32x4_t accum_hb=vdupq_n_s32(0); 
   
        for (fj = 0; fj < half_fw; fj++){

            jj1 = f_l_j + fj;
            jj2 = f_r_j - fj;

            src16x8_0 = vld1q_s16(tmp + jj1);
            src16x8_1 = vld1q_s16(tmp + jj2);
            
            add_l = vaddl_s16(vget_low_s16(src16x8_0), vget_low_s16(src16x8_1));
	        add_h = vaddl_high_s16(src16x8_0, src16x8_1);
            
         
            accum_lb = vmlaq_n_s32(accum_lb, add_l, i_filter_coeffs[fj]);
            accum_hb = vmlaq_n_s32(accum_hb, add_h, i_filter_coeffs[fj]);

        }
        src16x8_0 = vld1q_s16(tmp + j);
        accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_0), i_filter_coeffs[half_fw]);
        accum_hb = vmlal_high_n_s16(accum_hb, src16x8_0, i_filter_coeffs[half_fw]);
      
       
        half_round16x4_lb = vqrshrn_n_s32(accum_lb, SPAT_FILTER_OUT_SHIFT);
        half_round16x4_hb = vqrshrn_n_s32(accum_hb, SPAT_FILTER_OUT_SHIFT);
        vst1_s16(dst + dst_row_idx + j, half_round16x4_lb);
		vst1_s16(dst + dst_row_idx + j + 4, half_round16x4_hb);

    }
    
     for (j = width - half_fw - wid ; j < (width - half_fw); j++)
    {
        int f_l_j = j - half_fw;
        int f_r_j = j + half_fw;
        spat_fil_accum_dtype accum = 0;
         for (fj = 0; fj < half_fw; fj++){

            jj1 = f_l_j + fj;
            jj2 = f_r_j - fj;
            accum += i_filter_coeffs[fj] * ((spat_fil_accum_dtype)tmp[jj1] + tmp[jj2]); 
        }
        accum += (spat_fil_inter_dtype) i_filter_coeffs[half_fw] * tmp[j];
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);
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
        dst[dst_row_idx + j] = (spat_fil_output_dtype) ((accum + SPAT_FILTER_OUT_RND) >> SPAT_FILTER_OUT_SHIFT);

    }

}

const spat_fil_coeff_dtype i_ngan_filter_coeffs_neon[21] = {
    -900,  -1054, -1239, -1452, -1669, -1798, -1547, -66,   4677,  14498, 21495,
    14498, 4677,  -66,   -1547, -1798, -1669, -1452, -1239, -1054, -900};

const spat_fil_coeff_dtype i_nadeanu_filter_coeffs_neon[5] = {1658, 15139, 31193, 15139, 1658};

void integer_spatial_5tap_filter_neon(void *src, spat_fil_output_dtype *dst, int dst_stride, int width, int height, int bitdepth, 
                                            spat_fil_inter_dtype *tmp, char *spatial_csf_filter)
{
    if(bitdepth==8)
    {
        int filter_size;
        const spat_fil_coeff_dtype *i_filter_coeffs;
        if(strcmp(spatial_csf_filter, "nadenau_spat") == 0) {
            filter_size = 5;
            i_filter_coeffs = i_nadeanu_filter_coeffs_neon;
        } else if(strcmp(spatial_csf_filter, "ngan_spat") == 0) {
            filter_size = 21;
            i_filter_coeffs = i_ngan_filter_coeffs_neon;
        }

        int src_px_stride = width;
        int dst_px_stride = dst_stride/sizeof(spat_fil_output_dtype);  // changes made by me here
        uint8_t *src_8b = NULL;
        int i, j, fi, ii,increment,increment1,increment2,increment3,increment4;
        int half_fw = filter_size / 2;
        src_8b = (uint8_t*)src;
        int eff_width=width-(width%8);

        int interim_rnd = SPAT_FILTER_INTER_RND,interim_shift = SPAT_FILTER_INTER_SHIFT;

        // declare all the neon variables
        uint8x8_t src8x8_0,src8x8_1,src8x8_2,src8x8_3,src8x8_4;
        int16x8_t src16x8_0,src16x8_1,src16x8_2,src16x8_3;
        int16x4_t half_round16x4_lb, half_round16x4_hb;

        for (i = 0; i < half_fw; i++){

            int diff_i_halffw = i - half_fw;
            int pro_mir_end = -diff_i_halffw - 1;

            for (j = 0; j < eff_width; j=j+8)
               {
                    int32x4_t accum_lb=vdupq_n_s32(0);
                    int32x4_t accum_hb=vdupq_n_s32(0); 

                    for (fi = 0; fi <= pro_mir_end; fi++)
                    {

                        ii = pro_mir_end - fi;
                        increment=ii * src_px_stride + j;
                        src8x8_0=vld1_u8(src_8b + increment);
                        src16x8_0=vreinterpretq_s16_u16(vmovl_u8(src8x8_0));
                        accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_0), i_filter_coeffs[fi]);
                        accum_hb = vmlal_high_n_s16(accum_hb, src16x8_0, i_filter_coeffs[fi]);

                    }
                    for ( ; fi < filter_size; fi++)
                    {
                        ii = diff_i_halffw + fi;
                        
                        increment=ii * src_px_stride + j;
                        src8x8_0=vld1_u8(src_8b + increment);
                        src16x8_0=vreinterpretq_s16_u16(vmovl_u8(src8x8_0));
                        accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_0), i_filter_coeffs[fi]);
                        accum_hb = vmlal_high_n_s16(accum_hb, src16x8_0, i_filter_coeffs[fi]);
                    }
                    half_round16x4_lb = vqrshrn_n_s32(accum_lb, SPAT_FILTER_INTER_SHIFT);
                    half_round16x4_hb = vqrshrn_n_s32(accum_hb, SPAT_FILTER_INTER_SHIFT);
                    vst1_s16(tmp + j,half_round16x4_lb);
                    vst1_s16(tmp + j+4,half_round16x4_hb);
                }

               // for the last columns  of each row
                for (j = eff_width; j < width; j++){

                    spat_fil_accum_dtype accum = 0;

                    /**
                     * The full loop is from fi = 0 to filter_size
                     * During the loop when the centre pixel is at i, 
                     * the top part is available only till i-(filter_size/2) >= 0, 
                     * hence padding (border mirroring) is required when i-filter_size/2 < 0
                     */
                    //This loop does border mirroring (ii = -(i - filter_size/2 + fi + 1))
                    for (fi = 0; fi <= pro_mir_end; fi++){

                        ii = pro_mir_end - fi;
                        accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                    }
                    //Here the normal loop is executed where ii = i - filter_size / 2 + fi
                    for ( ; fi < filter_size; fi++)
                    {
                        ii = diff_i_halffw + fi;
                        accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                    }
                    tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
                }
            integer_horizontal_5tap_filter_neon(tmp, dst, i_filter_coeffs, width, filter_size, i*dst_px_stride, half_fw);
        }

        for ( ; i < (height - half_fw); i++){
            int f_l_i = i - half_fw;
            int f_r_i = i + half_fw;

            for (j = 0; j < eff_width; j=j+8)
            {
                int32x4_t accum_lb=vdupq_n_s32(0);
                int32x4_t accum_hb=vdupq_n_s32(0);


                increment=(i-2)*src_px_stride +j;
                increment1=(i-1)*src_px_stride +j;
                increment2=(i)*src_px_stride +j;
                increment3=(i+1)*src_px_stride +j;
                increment4=(i+2)*src_px_stride +j;

                src8x8_0=vld1_u8(src_8b + increment);
                src8x8_4=vld1_u8(src_8b + increment4);
                src8x8_1=vld1_u8(src_8b + increment1);
                src8x8_3=vld1_u8(src_8b + increment3);
                src8x8_2=vld1_u8(src_8b + increment2);

               src16x8_2 = vreinterpretq_s16_u16(vmovl_u8(src8x8_2));
               src16x8_1 = vreinterpretq_s16_u16(vaddl_u8(src8x8_0, src8x8_4));
               src16x8_3 = vreinterpretq_s16_u16(vaddl_u8(src8x8_1, src8x8_3));

                accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_1), i_filter_coeffs[0]);
                accum_hb = vmlal_high_n_s16(accum_hb, src16x8_1, i_filter_coeffs[0]);
                accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_3), i_filter_coeffs[1]);
                accum_hb = vmlal_high_n_s16(accum_hb, src16x8_3, i_filter_coeffs[1]);

                accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_2), i_filter_coeffs[2]);
                accum_hb = vmlal_high_n_s16(accum_hb, src16x8_2, i_filter_coeffs[2]);
                half_round16x4_lb = vqrshrn_n_s32(accum_lb, SPAT_FILTER_INTER_SHIFT);
                half_round16x4_hb = vqrshrn_n_s32(accum_hb, SPAT_FILTER_INTER_SHIFT);

                vst1_s16(tmp + j,half_round16x4_lb);
                vst1_s16(tmp + j+4,half_round16x4_hb);
            }

            for (j = eff_width; j < width; j++){

                spat_fil_accum_dtype accum = 0;

                /**
                 * The filter coefficients are symmetric, 
                 * hence the corresponding pixels for whom coefficient values would be same are added first & then multiplied by coeff
                 * The centre pixel is multiplied and accumulated outside the loop
                */
                for (fi = 0; fi < (half_fw); fi++){
                    int ii1 = f_l_i + fi;
                    int ii2 = f_r_i - fi;
                    accum += i_filter_coeffs[fi] * ((spat_fil_inter_dtype)src_8b[ii1 * src_px_stride + j] + src_8b[ii2 * src_px_stride + j]);
                }
                accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[i * src_px_stride + j];
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
            }
            integer_horizontal_5tap_filter_neon(tmp, dst, i_filter_coeffs, width, filter_size, i*dst_px_stride, half_fw);
        }

        for (; i < height; i++){

            int diff_i_halffw = i - half_fw;
            int epi_mir_i = 2 * height - diff_i_halffw - 1;
            int epi_last_i  = height - diff_i_halffw;

            for (j = 0; j < eff_width; j=j+8)
            {
                int32x4_t accum_lb=vdupq_n_s32(0);
                int32x4_t accum_hb=vdupq_n_s32(0);

                for (fi = 0; fi < epi_last_i; fi++){

                    ii = diff_i_halffw + fi;
                    increment=ii * src_px_stride + j;
                    src8x8_0=vld1_u8(src_8b + increment);
                    src16x8_0=vreinterpretq_s16_u16(vmovl_u8(src8x8_0));

                    accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_0), i_filter_coeffs[fi]);
                    accum_hb = vmlal_high_n_s16(accum_hb, src16x8_0, i_filter_coeffs[fi]);
                }
                for ( ; fi < filter_size; fi++)
                {
                    ii = epi_mir_i - fi;

                    increment=ii * src_px_stride + j;
                    src8x8_0=vld1_u8(src_8b + increment);
                    src16x8_0=vreinterpretq_s16_u16(vmovl_u8(src8x8_0));

                    accum_lb = vmlal_n_s16(accum_lb, vget_low_s16(src16x8_0), i_filter_coeffs[fi]);
                    accum_hb = vmlal_high_n_s16(accum_hb, src16x8_0, i_filter_coeffs[fi]);
                }

                half_round16x4_lb = vqrshrn_n_s32(accum_lb, SPAT_FILTER_INTER_SHIFT);
                half_round16x4_hb = vqrshrn_n_s32(accum_hb, SPAT_FILTER_INTER_SHIFT);
                vst1_s16(tmp + j,half_round16x4_lb);
                vst1_s16(tmp + j+4,half_round16x4_hb); 
            }
            for (j = eff_width; j < width; j++){
                spat_fil_accum_dtype accum = 0;

                /**
                 * The full loop is from fi = 0 to filter_size
                 * During the loop when the centre pixel is at i, 
                 * the bottom pixels are available only till i+(filter_size/2) < height, 
                 * hence padding (border mirroring) is required when i+(filter_size/2) >= height
                 */
                //Here the normal loop is executed where ii = i - filter_size/2 + fi
                for (fi = 0; fi < epi_last_i; fi++){

                    ii = diff_i_halffw + fi;
                    accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                }
                //This loop does border mirroring (ii = 2*height - (i - filter_size/2 + fi) - 1)
                for ( ; fi < filter_size; fi++)
                {
                    ii = epi_mir_i - fi;
                    accum += (spat_fil_inter_dtype) i_filter_coeffs[fi] * src_8b[ii * src_px_stride + j];
                }
                tmp[j] = (spat_fil_inter_dtype) ((accum + interim_rnd) >> interim_shift);
            }
            integer_horizontal_5tap_filter_neon(tmp, dst, i_filter_coeffs, width, filter_size, i*dst_px_stride, half_fw);
        }
    }

    return;
}


void integer_funque_dwt2_inplace_csf_neon(const i_dwt2buffers *src, spat_fil_coeff_dtype factors[4],
                                          int min_theta, int max_theta, uint16_t interim_rnd_factors[4],
                                          uint8_t interim_shift_factors[4], int level)
{

    dwt2_dtype *angles[4] = {src->bands[0], src->bands[2], src->bands[3], src->bands[1]};

    int px_stride = src->stride / sizeof(dwt2_dtype);

    int left = 0;
    int top = 0;

    int right = src->width - ((src->width) % 32);
    int bottom = src->height;

    int i, j, src_offset, dst_offset, theta;

    // intialization required for extra width processing
    spat_fil_accum_dtype mul_val;
    dwt2_dtype dst_val;
    dwt2_dtype *src_ptr;
    dwt2_dtype *dst_ptr;

    // initializing neon variables
    int16x8_t src_0, src_1, src_2, src_3, src_4, src_5, src_6, src_7;
    int16x8_t src_8, src_9, src_10, src_11, src_12, src_13, src_14, src_15;

    int16x4_t src_0_l, src_1_l, src_2_l, src_3_l, src_4_l, src_5_l, src_6_l, src_7_l;
    int16x4_t src_8_l, src_9_l, src_10_l, src_11_l, src_12_l, src_13_l, src_14_l, src_15_l;

    int32x4_t mul_val_0_h, mul_val_1_h, mul_val_2_h, mul_val_3_h, mul_val_0_l, mul_val_1_l, mul_val_2_l, mul_val_3_l;
    int32x4_t mul_val_4_h, mul_val_5_h, mul_val_6_h, mul_val_7_h, mul_val_4_l, mul_val_5_l, mul_val_6_l, mul_val_7_l;

    int32x4_t mul_val_9_h, mul_val_10_h, mul_val_11_h, mul_val_8_h, mul_val_8_l, mul_val_9_l, mul_val_10_l, mul_val_11_l;
    int32x4_t mul_val_12_h, mul_val_13_h, mul_val_14_h, mul_val_15_h, mul_val_12_l, mul_val_13_l, mul_val_14_l, mul_val_15_l;

    int16x4_t dst_val_0_h, dst_val_1_h, dst_val_2_h, dst_val_3_h, dst_val_0_l, dst_val_1_l, dst_val_2_l, dst_val_3_l;
    int16x4_t dst_val_4_h, dst_val_5_h, dst_val_6_h, dst_val_7_h, dst_val_4_l, dst_val_5_l, dst_val_6_l, dst_val_7_l;

    int16x4_t dst_val_8_h, dst_val_9_h, dst_val_10_h, dst_val_11_h, dst_val_8_l, dst_val_9_l, dst_val_10_l, dst_val_11_l;
    int16x4_t dst_val_12_h, dst_val_13_h, dst_val_14_h, dst_val_15_h, dst_val_12_l, dst_val_13_l, dst_val_14_l, dst_val_15_l;

    int32x4_t rnd_factor1, rnd_factor2, rnd_factor3, rnd_factor4, a1, a2, a3, a4;

    rnd_factor1 = vdupq_n_s32(interim_rnd_factors[0]);
    rnd_factor2 = vdupq_n_s32(interim_rnd_factors[1]);
    rnd_factor3 = vdupq_n_s32(interim_rnd_factors[2]);
    rnd_factor4 = vdupq_n_s32(interim_rnd_factors[3]);

    a1 = vdupq_n_s32(-interim_shift_factors[0]);
    a2 = vdupq_n_s32(-interim_shift_factors[1]);
    a3 = vdupq_n_s32(-interim_shift_factors[2]);
    a4 = vdupq_n_s32(-interim_shift_factors[3]);

    for (i = top; i < bottom; ++i)
    {
        src_offset = i * px_stride;
        dst_offset = i * px_stride;

        for (j = left; j < right; j = j + 32)
        {

            // loading from all 4 bands 8 pixels
            src_0 = vld1q_s16(angles[0] + src_offset + j);
            src_1 = vld1q_s16(angles[1] + src_offset + j);
            src_2 = vld1q_s16(angles[2] + src_offset + j);
            src_3 = vld1q_s16(angles[3] + src_offset + j);

            src_4 = vld1q_s16(angles[0] + src_offset + 8 + j);
            src_5 = vld1q_s16(angles[1] + src_offset + 8 + j);
            src_6 = vld1q_s16(angles[2] + src_offset + 8 + j);
            src_7 = vld1q_s16(angles[3] + src_offset + 8 + j);

            src_8 = vld1q_s16(angles[0] + src_offset + 16 + j);
            src_9 = vld1q_s16(angles[1] + src_offset + 16 + j);
            src_10 = vld1q_s16(angles[2] + src_offset + 16 + j);
            src_11 = vld1q_s16(angles[3] + src_offset + 16 + j);

            src_12 = vld1q_s16(angles[0] + src_offset + 24 + j);
            src_13 = vld1q_s16(angles[1] + src_offset + 24 + j);
            src_14 = vld1q_s16(angles[2] + src_offset + 24 + j);
            src_15 = vld1q_s16(angles[3] + src_offset + 24 + j);

            // getting the lower value
            src_0_l = vget_low_s16(src_0);
            src_1_l = vget_low_s16(src_1);
            src_2_l = vget_low_s16(src_2);
            src_3_l = vget_low_s16(src_3);

            src_7_l = vget_low_s16(src_7);
            src_6_l = vget_low_s16(src_6);
            src_5_l = vget_low_s16(src_5);
            src_4_l = vget_low_s16(src_4);

            src_8_l = vget_low_s16(src_8);
            src_9_l = vget_low_s16(src_9);
            src_10_l = vget_low_s16(src_10);
            src_11_l = vget_low_s16(src_11);

            src_12_l = vget_low_s16(src_12);
            src_13_l = vget_low_s16(src_13);
            src_14_l = vget_low_s16(src_14);
            src_15_l = vget_low_s16(src_15);

            // multiplying
            mul_val_0_h = vmull_high_n_s16(src_0, factors[0]);
            mul_val_1_h = vmull_high_n_s16(src_1, factors[1]);
            mul_val_2_h = vmull_high_n_s16(src_2, factors[2]);
            mul_val_3_h = vmull_high_n_s16(src_3, factors[3]);

            mul_val_0_l = vmull_n_s16(src_0_l, factors[0]);
            mul_val_1_l = vmull_n_s16(src_1_l, factors[1]);
            mul_val_2_l = vmull_n_s16(src_2_l, factors[2]);
            mul_val_3_l = vmull_n_s16(src_3_l, factors[3]);

            mul_val_4_h = vmull_high_n_s16(src_4, factors[0]);
            mul_val_5_h = vmull_high_n_s16(src_5, factors[1]);
            mul_val_6_h = vmull_high_n_s16(src_6, factors[2]);
            mul_val_7_h = vmull_high_n_s16(src_7, factors[3]);

            mul_val_4_l = vmull_n_s16(src_4_l, factors[0]);
            mul_val_5_l = vmull_n_s16(src_5_l, factors[1]);
            mul_val_6_l = vmull_n_s16(src_6_l, factors[2]);
            mul_val_7_l = vmull_n_s16(src_7_l, factors[3]);

            mul_val_8_h = vmull_high_n_s16(src_8, factors[0]);
            mul_val_9_h = vmull_high_n_s16(src_9, factors[1]);
            mul_val_10_h = vmull_high_n_s16(src_10, factors[2]);
            mul_val_11_h = vmull_high_n_s16(src_11, factors[3]);

            mul_val_8_l = vmull_n_s16(src_8_l, factors[0]);
            mul_val_9_l = vmull_n_s16(src_9_l, factors[1]);
            mul_val_10_l = vmull_n_s16(src_10_l, factors[2]);
            mul_val_11_l = vmull_n_s16(src_11_l, factors[3]);

            mul_val_12_h = vmull_high_n_s16(src_12, factors[0]);
            mul_val_13_h = vmull_high_n_s16(src_13, factors[1]);
            mul_val_14_h = vmull_high_n_s16(src_14, factors[2]);
            mul_val_15_h = vmull_high_n_s16(src_15, factors[3]);

            mul_val_12_l = vmull_n_s16(src_12_l, factors[0]);
            mul_val_13_l = vmull_n_s16(src_13_l, factors[1]);
            mul_val_14_l = vmull_n_s16(src_14_l, factors[2]);
            mul_val_15_l = vmull_n_s16(src_15_l, factors[3]);

            // half_rounding_off
            dst_val_0_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_0_h, rnd_factor1), a1));
            dst_val_1_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_1_h, rnd_factor2), a2));
            dst_val_2_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_2_h, rnd_factor3), a3));
            dst_val_3_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_3_h, rnd_factor4), a4));

            dst_val_0_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_0_l, rnd_factor1), a1));
            dst_val_1_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_1_l, rnd_factor2), a2));
            dst_val_2_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_2_l, rnd_factor3), a3));
            dst_val_3_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_3_l, rnd_factor4), a4));

            dst_val_4_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_4_h, rnd_factor1), a1));
            dst_val_5_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_5_h, rnd_factor2), a2));
            dst_val_6_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_6_h, rnd_factor3), a3));
            dst_val_7_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_7_h, rnd_factor4), a4));

            dst_val_4_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_4_l, rnd_factor1), a1));
            dst_val_5_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_5_l, rnd_factor2), a2));
            dst_val_6_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_6_l, rnd_factor3), a3));
            dst_val_7_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_7_l, rnd_factor4), a4));

            dst_val_8_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_8_l, rnd_factor1), a1));
            dst_val_9_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_9_l, rnd_factor2), a2));
            dst_val_10_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_10_l, rnd_factor3), a3));
            dst_val_11_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_11_l, rnd_factor4), a4));

            dst_val_8_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_8_h, rnd_factor1), a1));
            dst_val_9_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_9_h, rnd_factor2), a2));
            dst_val_10_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_10_h, rnd_factor3), a3));
            dst_val_11_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_11_h, rnd_factor4), a4));

            dst_val_12_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_12_h, rnd_factor1), a1));
            dst_val_13_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_13_h, rnd_factor2), a2));
            dst_val_14_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_14_h, rnd_factor3), a3));
            dst_val_15_h = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_15_h, rnd_factor4), a4));

            dst_val_12_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_12_l, rnd_factor1), a1));
            dst_val_13_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_13_l, rnd_factor2), a2));
            dst_val_14_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_14_l, rnd_factor3), a3));
            dst_val_15_l = vmovn_s32(vshlq_s32(vaddq_s32(mul_val_15_l, rnd_factor4), a4));

            // storing the data
            vst1_s16(src->bands[0] + dst_offset + j, dst_val_0_l);
            vst1_s16(src->bands[2] + dst_offset + j, dst_val_1_l);
            vst1_s16(src->bands[3] + dst_offset + j, dst_val_2_l);
            vst1_s16(src->bands[1] + dst_offset + j, dst_val_3_l);

            vst1_s16(src->bands[0] + dst_offset + j + 4, dst_val_0_h);
            vst1_s16(src->bands[2] + dst_offset + j + 4, dst_val_1_h);
            vst1_s16(src->bands[3] + dst_offset + j + 4, dst_val_2_h);
            vst1_s16(src->bands[1] + dst_offset + j + 4, dst_val_3_h);

            vst1_s16(src->bands[0] + dst_offset + j + 8, dst_val_4_l);
            vst1_s16(src->bands[2] + dst_offset + j + 8, dst_val_5_l);
            vst1_s16(src->bands[3] + dst_offset + j + 8, dst_val_6_l);
            vst1_s16(src->bands[1] + dst_offset + j + 8, dst_val_7_l);

            vst1_s16(src->bands[0] + dst_offset + j + 12, dst_val_4_h);
            vst1_s16(src->bands[2] + dst_offset + j + 12, dst_val_5_h);
            vst1_s16(src->bands[3] + dst_offset + j + 12, dst_val_6_h);
            vst1_s16(src->bands[1] + dst_offset + j + 12, dst_val_7_h);

            vst1_s16(src->bands[0] + dst_offset + j + 16, dst_val_8_l);
            vst1_s16(src->bands[2] + dst_offset + j + 16, dst_val_9_l);
            vst1_s16(src->bands[3] + dst_offset + j + 16, dst_val_10_l);
            vst1_s16(src->bands[1] + dst_offset + j + 16, dst_val_11_l);

            vst1_s16(src->bands[0] + dst_offset + j + 20, dst_val_8_h);
            vst1_s16(src->bands[2] + dst_offset + j + 20, dst_val_9_h);
            vst1_s16(src->bands[3] + dst_offset + j + 20, dst_val_10_h);
            vst1_s16(src->bands[1] + dst_offset + j + 20, dst_val_11_h);

            vst1_s16(src->bands[0] + dst_offset + j + 24, dst_val_12_l);
            vst1_s16(src->bands[2] + dst_offset + j + 24, dst_val_13_l);
            vst1_s16(src->bands[3] + dst_offset + j + 24, dst_val_14_l);
            vst1_s16(src->bands[1] + dst_offset + j + 24, dst_val_15_l);

            vst1_s16(src->bands[0] + dst_offset + j + 28, dst_val_12_h);
            vst1_s16(src->bands[2] + dst_offset + j + 28, dst_val_13_h);
            vst1_s16(src->bands[3] + dst_offset + j + 28, dst_val_14_h);
            vst1_s16(src->bands[1] + dst_offset + j + 28, dst_val_15_h);
        }
    }
    // processing the last few columns of each row
    if (right != src->width)
    {

        for (theta = min_theta; theta <= max_theta; ++theta)
        {
            src_ptr = angles[theta];
            dst_ptr = angles[theta];

            for (i = top; i < bottom; ++i)
            {
                src_offset = i * px_stride;
                dst_offset = i * px_stride;

                for (j = right; j < src->width; ++j)
                {
                    mul_val = (spat_fil_accum_dtype)factors[theta] * src_ptr[src_offset + j];
                    dst_val = (dwt2_dtype)((mul_val + interim_rnd_factors[theta]) >>
                                           interim_shift_factors[theta]);
                    dst_ptr[dst_offset + j] = dst_val;
                }
            }
        }
    }
}
