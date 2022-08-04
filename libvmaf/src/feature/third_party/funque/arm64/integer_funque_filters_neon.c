#include "../integer_funque_filters.h"
#include <arm_neon.h>

void integer_funque_dwt2_neon(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
#if 0

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
            
            //a + b    & a - b    
            int32_t src_a_p_b = src_a + src_b;
            int32_t src_a_m_b = src_a - src_b;
            
            //c + d    & c - d
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
            
            //a + b    & a - b    
            int src_a_p_b = src_a + src_b;
            int src_a_m_b = src_a - src_b;
            
            //F* F (a + b + a + b) - band A  (F*F is 1/2)
            band_a[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_p_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
            
            //F* F (a - b + a - b) - band H  (F*F is 1/2)
            band_h[i*dst_px_stride+j] = (dwt2_dtype) ((src_a_m_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);
            
            //F* F (a + b - (a + b)) - band V, Last column V will always be 0            
            band_v[i*dst_px_stride+j] = 0;

            //F* F (a - b - (a -b)) - band D,  Last column D will always be 0
            band_d[i*dst_px_stride+j] = 0;
        }
    }

#else

    /**
     * Absolute value of filter coefficients are 1/sqrt(2)
     * The filter is handled by multiplying square of coefficients in final stage
     * Hence the value becomes 1/2, and this is handled using shifts
     * Also extra required out shift is done along with filter shift itself
     */
    const int8_t filter_shift = 1 + DWT2_OUT_SHIFT;
    const int8_t filter_shift_rnd = 1 << (filter_shift - 1);
    /**
     * Last column due to padding the values are left shifted and then right shifted
     * Hence using updated shifts. Subtracting 1 due to left shift
     */
    const int8_t filter_shift_lcpad = 1 + DWT2_OUT_SHIFT - 1;
    const int8_t filter_shift_lcpad_rnd = 1 << (filter_shift_lcpad - 1);

    int16x8_t src16x8_0, src16x8_1;
    int32x4_t add32x4_lo, add32x4_hi, sub32x4_lo, sub32x4_hi;
    int32x4_t mul32x4_0_lo, mul32x4_0_hi, mul32x4_1_lo, mul32x4_1_hi;
    int32x4x2_t arr32x4_lo, arr32x4_hi;
    int16x4_t sft16x4_0, sft16x4_1, sft16x4_2, sft16x4_3;
    int row0_offset, row1_offset;
    int16_t row_idx0, row_idx1;
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];

    int heightDiv2 = (height + 1) >> 1;
    int lastRow = height & 1;
    int i, j;
    int width_div_2 = width >> 1; // without rounding (last value is handle outside)
    int last_col = width & 1;
    // int32x4_t addAC_32x4_0, addAC_32x4_1, addBD_32x4_0, addBD_32x4_1, addABCD_32x4_0, addABCD_32x4_1, subABCD_32x4_0, subABCD_32x4_1;
    int32x4_t addAB_32x4_lo, addAB_32x4_hi, addCD_32x4_lo, addCD_32x4_hi;
    int32x4_t subAB_32x4_lo, subAB_32x4_hi, subCD_32x4_lo, subCD_32x4_hi;
    int32x4_t addA_32x4_lo, addA_32x4_hi, addH_32x4_lo, addH_32x4_hi;
    int32x4_t subV_32x4_lo, subV_32x4_hi, subD_32x4_lo, subD_32x4_hi;

    int16x4_t bandA_16x4_lo, bandA_16x4_hi, bandH_16x4_lo, bandH_16x4_hi;
    int16x4_t bandV_16x4_lo, bandV_16x4_hi, bandD_16x4_lo, bandD_16x4_hi;

    int16x8_t src0_16x8, src1_16x8, src2_16x8, src3_16x8;
    int16x8x2_t srcAC_16x8, srcBD_16x8;
    int bandOffset;

    for (i = 0; i < heightDiv2; i++)
    {
        row_idx0 = 2 * i;
        row_idx1 = 2 * i + 1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2 * i;
        row0_offset = (row_idx0)*width;
        row1_offset = (row_idx1)*width;

        for (j = 0; j <= width - 16; j += 16)
        {
            src0_16x8 = vld1q_s16(src + row0_offset + j);     // A and C
            src1_16x8 = vld1q_s16(src + row1_offset + j);     // B and D
            src2_16x8 = vld1q_s16(src + row0_offset + j + 8); // A and C with offset 8
            src3_16x8 = vld1q_s16(src + row1_offset + j + 8); // B and D with offset 8

            srcAC_16x8 = vuzpq_s16(src0_16x8, src2_16x8); // A in val[0] and C in val[1]
            srcBD_16x8 = vuzpq_s16(src1_16x8, src3_16x8); // B in val[0] and D in val[1]

            addAB_32x4_lo = vaddl_s16(vget_low_s16(srcAC_16x8.val[0]), vget_low_s16(srcBD_16x8.val[0]));
            addAB_32x4_hi = vaddl_high_s16(srcAC_16x8.val[0], srcBD_16x8.val[0]);
            addCD_32x4_lo = vaddl_s16(vget_low_s16(srcAC_16x8.val[1]), vget_low_s16(srcBD_16x8.val[1]));
            addCD_32x4_hi = vaddl_high_s16(srcAC_16x8.val[1], srcBD_16x8.val[1]);

            subAB_32x4_lo = vsubl_s16(vget_low_s16(srcAC_16x8.val[0]), vget_low_s16(srcBD_16x8.val[0]));
            subAB_32x4_hi = vsubl_high_s16(srcAC_16x8.val[0], srcBD_16x8.val[0]);
            subCD_32x4_lo = vsubl_s16(vget_low_s16(srcAC_16x8.val[1]), vget_low_s16(srcBD_16x8.val[1]));
            subCD_32x4_hi = vsubl_high_s16(srcAC_16x8.val[1], srcBD_16x8.val[1]);

#if 0
            addAC_32x4_0 = vpaddlq_s16(src0_16x8);
            addBD_32x4_0 = vpaddlq_s16(src1_16x8);
            addAC_32x4_1 = vpaddlq_s16(src2_16x8);
            addBD_32x4_1 = vpaddlq_s16(src3_16x8);

            addABCD_32x4_0 = vaddq_s32(addAC_32x4_0, addBD_32x4_0);
            addABCD_32x4_1 = vaddq_s32(addAC_32x4_1, addBD_32x4_1);
            subABCD_32x4_0 = vsubq_s32(addAC_32x4_0, addBD_32x4_0);
            subABCD_32x4_1 = vsubq_s32(addAC_32x4_1, addBD_32x4_1);

            bandA_16x4_lo = vrshrn_n_s32(addABCD_32x4_0, 8);
            bandA_16x4_hi = vrshrn_n_s32(addABCD_32x4_1, 8);
            bandH_16x4_lo = vrshrn_n_s32(subABCD_32x4_0, 8);
            bandH_16x4_hi = vrshrn_n_s32(subABCD_32x4_1, 8);
#else

            addA_32x4_lo = vaddq_s32(addAB_32x4_lo, addCD_32x4_lo);
            addA_32x4_hi = vaddq_s32(addAB_32x4_hi, addCD_32x4_hi);
            addH_32x4_lo = vaddq_s32(subAB_32x4_lo, subCD_32x4_lo);
            addH_32x4_hi = vaddq_s32(subAB_32x4_hi, subCD_32x4_hi);

            bandA_16x4_lo = vqrshrn_n_s32(addA_32x4_lo, filter_shift);
            bandA_16x4_hi = vqrshrn_n_s32(addA_32x4_hi, filter_shift);
            bandH_16x4_lo = vqrshrn_n_s32(addH_32x4_lo, filter_shift);
            bandH_16x4_hi = vqrshrn_n_s32(addH_32x4_hi, filter_shift);

#endif

            subV_32x4_lo = vsubq_s32(addAB_32x4_lo, addCD_32x4_lo);
            subV_32x4_hi = vsubq_s32(addAB_32x4_hi, addCD_32x4_hi);
            subD_32x4_lo = vsubq_s32(subAB_32x4_lo, subCD_32x4_lo);
            subD_32x4_hi = vsubq_s32(subAB_32x4_hi, subCD_32x4_hi);

            bandV_16x4_lo = vqrshrn_n_s32(subV_32x4_lo, filter_shift);
            bandV_16x4_hi = vqrshrn_n_s32(subV_32x4_hi, filter_shift);
            bandD_16x4_lo = vqrshrn_n_s32(subD_32x4_lo, filter_shift);
            bandD_16x4_hi = vqrshrn_n_s32(subD_32x4_hi, filter_shift);

            bandOffset = i * dst_px_stride + (j >> 1);
            vst1_s16(band_a + bandOffset, bandA_16x4_lo);
            vst1_s16(band_a + bandOffset + 4, bandA_16x4_hi);
            vst1_s16(band_h + bandOffset, bandH_16x4_lo);
            vst1_s16(band_h + bandOffset + 4, bandH_16x4_hi);

            vst1_s16(band_v + bandOffset, bandV_16x4_lo);
            vst1_s16(band_v + bandOffset + 4, bandV_16x4_hi);
            vst1_s16(band_d + bandOffset, bandD_16x4_lo);
            vst1_s16(band_d + bandOffset + 4, bandD_16x4_hi);
        }
        for (; j < width; ++j)
        {
            /*
            if (last_col)
            {
                int col_idx0 = width_div_2 << 1;
                j = width_div_2;

                // a & b 2 values in adjacent rows at the last coloumn
                spat_fil_output_dtype src_a = src[row0_offset + col_idx0];
                spat_fil_output_dtype src_b = src[row1_offset + col_idx0];

                // a + b    & a - b
                int src_a_p_b = src_a + src_b;
                int src_a_m_b = src_a - src_b;

                // F* F (a + b + a + b) - band A  (F*F is 1/2)
                band_a[i * dst_px_stride + j] = (dwt2_dtype)((src_a_p_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);

                // F* F (a - b + a - b) - band H  (F*F is 1/2)
                band_h[i * dst_px_stride + j] = (dwt2_dtype)((src_a_m_b + filter_shift_lcpad_rnd) >> filter_shift_lcpad);

                // F* F (a + b - (a + b)) - band V, Last column V will always be 0
                band_v[i * dst_px_stride + j] = 0;

                // F* F (a - b - (a -b)) - band D,  Last column D will always be 0
                band_d[i * dst_px_stride + j] = 0;
            }
            else
            */
            {
                int col_idx0 = (j << 1);
                int col_idx1 = (j << 1) + 1;

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
                band_a[i * dst_px_stride + j] = (dwt2_dtype)(((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift);
                // printf("%d\t", (dwt2_dtype)(((src_a_p_b + src_c_p_d) + filter_shift_rnd) >> filter_shift));

                // F* F (a - b + c - d) - band H  (F*F is 1/2)
                band_h[i * dst_px_stride + j] = (dwt2_dtype)(((src_a_m_b + src_c_m_d) + filter_shift_rnd) >> filter_shift);
                // printf("%d\t", (dwt2_dtype)(((src_a_m_b + src_c_m_d) + filter_shift_rnd) >> filter_shift));

                // F* F (a + b - c + d) - band V  (F*F is 1/2)
                band_v[i * dst_px_stride + j] = (dwt2_dtype)(((src_a_p_b - src_c_p_d) + filter_shift_rnd) >> filter_shift);
                // printf("%d\t", (dwt2_dtype)(((src_a_p_b - src_c_p_d) + filter_shift_rnd) >> filter_shift));

                // F* F (a - b - c - d) - band D  (F*F is 1/2)
                band_d[i * dst_px_stride + j] = (dwt2_dtype)(((src_a_m_b - src_c_m_d) + filter_shift_rnd) >> filter_shift);
                // printf("%d\t", (dwt2_dtype)(((src_a_m_b - src_c_m_d) + filter_shift_rnd) >> filter_shift));
            }
        }
    }

#endif
}