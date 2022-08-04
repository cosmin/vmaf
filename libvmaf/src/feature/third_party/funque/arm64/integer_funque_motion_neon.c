#include "../integer_funque_filters.h"
#include <arm_neon.h>

#define C_IN_SIMD 0
#define OLD_SIMD 1
#define MOD_SIMD 0

double integer_funque_image_mad_neon(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor)
{
#if C_IN_SIMD
    motion_accum_dtype accum = 0;

    for (int i = 0; i < height; ++i)
    {
        motion_interaccum_dtype accum_line = 0;
        for (int j = 0; j < width; ++j)
        {
            dwt2_dtype img1px = img1[i * img1_stride + j];
            dwt2_dtype img2px = img2[i * img2_stride + j];

            accum_line += (motion_interaccum_dtype)abs(img1px - img2px);
            // assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (motion_accum_dtype)accum_line;
        // assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }

    double d_accum = (double)accum / pending_div_factor;
    return (d_accum / (width * height));
#endif

#if OLD_SIMD

    int img1_stride2 = (img1_stride << 1);
    int img2_stride2 = (img2_stride << 1);

    int16x8_t img16x8_1, img16x8_2, img16x8_3, img16x8_4;
    int16x8_t img16x8_5, img16x8_6, img16x8_7, img16x8_8;
    int64x2_t res64x2_1, res64x2_2, res64x2_3, res64x2_4;
    int64x2_t res64x2_5, res64x2_6, res64x2_7, res64x2_8;
    int64x2_t dupZero = vdupq_n_s64(0);
    res64x2_1 = dupZero;
    res64x2_2 = dupZero;
    res64x2_3 = dupZero;
    res64x2_4 = dupZero;

    motion_accum_dtype accum = 0, accumOddWidth = 0;
    int heightDiv2 = (height >> 1);
    int lastRow = (height & 1);
    int i, j;

    for (i = 0; i < heightDiv2; i++)
    {
        for (j = 0; j <= width - 16; j += 16)
        {
            img16x8_1 = vld1q_s16(img1 + j);
            img16x8_2 = vld1q_s16(img2 + j);
            img16x8_3 = vld1q_s16(img1 + j + 8);
            img16x8_4 = vld1q_s16(img2 + j + 8);
            img16x8_5 = vld1q_s16(img1 + j + img1_stride);
            img16x8_6 = vld1q_s16(img2 + j + img2_stride);
            img16x8_7 = vld1q_s16(img1 + j + img1_stride + 8);
            img16x8_8 = vld1q_s16(img2 + j + img2_stride + 8);

            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_1), vget_low_s16(img16x8_2)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_3), vget_low_s16(img16x8_4)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_1, img16x8_2));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_3, img16x8_4));
            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_5), vget_low_s16(img16x8_6)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_7), vget_low_s16(img16x8_8)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_5, img16x8_6));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_7, img16x8_8));
        }
        for (; j < width; j++)
        {
            dwt2_dtype img1px = img1[j];
            dwt2_dtype img2px = img2[j];
            dwt2_dtype img3px = img1[j + img1_stride];
            dwt2_dtype img4px = img2[j + img2_stride];
            accumOddWidth += (motion_interaccum_dtype)abs(img1px - img2px) + (motion_interaccum_dtype)abs(img3px - img4px);
        }
        img1 += img1_stride2;
        img2 += img2_stride2;
    }

    if (lastRow)
    {
        for (j = 0; j <= width - 16; j += 16)
        {
            img16x8_1 = vld1q_s16(img1 + j);
            img16x8_2 = vld1q_s16(img2 + j);
            img16x8_3 = vld1q_s16(img1 + j + 8);
            img16x8_4 = vld1q_s16(img2 + j + 8);

            res64x2_1 = vpadalq_s32(res64x2_1, vabdl_s16(vget_low_s16(img16x8_1), vget_low_s16(img16x8_2)));
            res64x2_2 = vpadalq_s32(res64x2_2, vabdl_s16(vget_low_s16(img16x8_3), vget_low_s16(img16x8_4)));
            res64x2_3 = vpadalq_s32(res64x2_3, vabdl_high_s16(img16x8_1, img16x8_2));
            res64x2_4 = vpadalq_s32(res64x2_4, vabdl_high_s16(img16x8_3, img16x8_4));
        }
        for (; j < width; ++j)
        {
            dwt2_dtype img1px = img1[j];
            dwt2_dtype img2px = img2[j];
            accumOddWidth += (motion_interaccum_dtype)abs(img1px - img2px);
        }
    }

    accum = vaddvq_s64(res64x2_1) + vaddvq_s64(res64x2_2) + vaddvq_s64(res64x2_3) + vaddvq_s64(res64x2_4) + accumOddWidth;
    double d_accum = (double)accum / pending_div_factor;
    return (d_accum / (width * height));
#endif

#if MOD_SIMD
    int img1_stride2 = (img1_stride << 1);
    int img2_stride2 = (img2_stride << 1);
    int img1_stride4 = img1_stride2 + img1_stride2;
    int img2_stride4 = img2_stride2 + img2_stride2;

    int16x8_t img16x8_1, img16x8_2, img16x8_3, img16x8_4;
    int16x8_t img16x8_5, img16x8_6, img16x8_7, img16x8_8;
    int32x4_t res32x4_1, res32x4_2, res32x4_3, res32x4_4;
    int32x4_t dupZero = vdupq_n_s32(0);

    motion_accum_dtype accum = 0;
    dwt2_dtype img1px, img2px, img3px, img4px;
    int i, j;

#define LOAD_2_ROWS_MOTION_SCORE(inp0, inp1, out0, out1, out2, out3, out4, \
                                 out5, out6, out7)                         \
    {                                                                      \
        out0 = vld1q_s16(inp0);                                            \
        out1 = vld1q_s16(inp1);                                            \
        out2 = vld1q_s16(inp0 + 8);                                        \
        out3 = vld1q_s16(inp1 + 8);                                        \
        out4 = vld1q_s16(inp0 + img1_stride);                              \
        out5 = vld1q_s16(inp1 + img2_stride);                              \
        out6 = vld1q_s16(inp0 + img1_stride + 8);                          \
        out7 = vld1q_s16(inp1 + img2_stride + 8);                          \
    }

#define CALC_ABS_MOTION_SCORE(load0, load1, load2, load3, load4, load5, load6, load7, \
                              res0, res1, res2, res3)                                 \
    {                                                                                 \
        res0 = vabal_s16(res0, vget_low_s16(load0), vget_low_s16(load1));             \
        res1 = vabal_s16(res1, vget_low_s16(load2), vget_low_s16(load3));             \
        res2 = vabal_high_s16(res2, load0, load1);                                    \
        res3 = vabal_high_s16(res3, load2, load3);                                    \
        res0 = vabal_s16(res0, vget_low_s16(load4), vget_low_s16(load5));             \
        res1 = vabal_s16(res1, vget_low_s16(load6), vget_low_s16(load7));             \
        res2 = vabal_high_s16(res2, load4, load5);                                    \
        res3 = vabal_high_s16(res3, load6, load7);                                    \
    }

    for (i = 0; i <= height - 4; i += 4)
    {
        res32x4_1 = dupZero;
        res32x4_2 = dupZero;
        res32x4_3 = dupZero;
        res32x4_4 = dupZero;
        for (j = 0; j <= width - 16; j += 16)
        {
            LOAD_2_ROWS_MOTION_SCORE(img1 + j, img2 + j, img16x8_1, img16x8_2,
                                     img16x8_3, img16x8_4, img16x8_5, img16x8_6, img16x8_7, img16x8_8)

            CALC_ABS_MOTION_SCORE(img16x8_1, img16x8_2, img16x8_3, img16x8_4, img16x8_5, img16x8_6, img16x8_7, img16x8_8,
                                  res32x4_1, res32x4_2, res32x4_3, res32x4_4)

            LOAD_2_ROWS_MOTION_SCORE(img1 + j + img1_stride2, img2 + j + img2_stride2, img16x8_1, img16x8_2,
                                     img16x8_3, img16x8_4, img16x8_5, img16x8_6, img16x8_7, img16x8_8)

            CALC_ABS_MOTION_SCORE(img16x8_1, img16x8_2, img16x8_3, img16x8_4, img16x8_5, img16x8_6, img16x8_7, img16x8_8,
                                  res32x4_1, res32x4_2, res32x4_3, res32x4_4)
        }
        for (; j < width; j++)
        {
            img1px = img1[j];
            img2px = img2[j];
            img3px = img1[j + img1_stride];
            img4px = img2[j + img2_stride];
            accum += (motion_interaccum_dtype)abs(img1px - img2px) + (motion_interaccum_dtype)abs(img3px - img4px);
            img1px = img1[j + img1_stride2];
            img2px = img2[j + img2_stride2];
            img3px = img1[j + img1_stride2 + img1_stride];
            img4px = img2[j + img2_stride2 + img2_stride];
            accum += (motion_interaccum_dtype)abs(img1px - img2px) + (motion_interaccum_dtype)abs(img3px - img4px);
        }
        accum += (motion_interaccum_dtype)vaddlvq_s32(res32x4_1) + (motion_interaccum_dtype)vaddlvq_s32(res32x4_2) +
                 (motion_interaccum_dtype)vaddlvq_s32(res32x4_3) + (motion_interaccum_dtype)vaddlvq_s32(res32x4_4);

        img1 += img1_stride4;
        img2 += img2_stride4;
    }

    for (i = 0; i < height; i++)
    {
        res32x4_1 = dupZero;
        res32x4_2 = dupZero;
        res32x4_3 = dupZero;
        res32x4_4 = dupZero;
        for (j = 0; j <= width - 16; j += 16)
        {
            img16x8_1 = vld1q_s16(img1 + j);
            img16x8_2 = vld1q_s16(img2 + j);
            img16x8_3 = vld1q_s16(img1 + j + 8);
            img16x8_4 = vld1q_s16(img2 + j + 8);

            res32x4_1 = vabal_s16(res32x4_1, vget_low_s16(img16x8_1), vget_low_s16(img16x8_2));
            res32x4_2 = vabal_s16(res32x4_2, vget_low_s16(img16x8_3), vget_low_s16(img16x8_4));
            res32x4_3 = vabal_high_s16(res32x4_3, img16x8_1, img16x8_2);
            res32x4_4 = vabal_high_s16(res32x4_4, img16x8_3, img16x8_4);
        }
        for (; j < width; ++j)
        {
            dwt2_dtype img1px = img1[j];
            dwt2_dtype img2px = img2[j];
            accum += (motion_interaccum_dtype)abs(img1px - img2px);
        }
        accum += (motion_interaccum_dtype)vaddlvq_s32(res32x4_1) + (motion_interaccum_dtype)vaddlvq_s32(res32x4_2) +
                 (motion_interaccum_dtype)vaddlvq_s32(res32x4_3) + (motion_interaccum_dtype)vaddlvq_s32(res32x4_4);

        img1 += img1_stride;
        img2 += img2_stride;
    }

    // accum = vaddvq_s64(res64x2_1) + vaddvq_s64(res64x2_2) + vaddvq_s64(res64x2_3) + vaddvq_s64(res64x2_4) + accumOddWidth;
    double d_accum = (double)accum / pending_div_factor;
    return (d_accum / (width * height));
#endif
}