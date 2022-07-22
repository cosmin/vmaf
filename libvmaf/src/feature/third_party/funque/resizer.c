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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DYNAMIC_COEFF 0

const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
static const int MAX_ESIZE = 16;

#define CLIP3(X,MIN,MAX) ((X < MIN) ? MIN : (X > MAX) ? MAX : X)
#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
#define MIN(LEFT, RIGHT) (LEFT < RIGHT ? LEFT : RIGHT)

static void interpolateCubic(float x, float* coeffs)
{
    const float A = -0.75f;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

#if DYNAMIC_COEFF
void hresize(const unsigned char** src, int** dst, int count,
    const int* xofs, const short* alpha,
    int swidth, int dwidth, int cn, int xmin, int xmax) {
#else
void hresize(const unsigned char** src, int** dst, int count,
    const int* xofs, int swidth, int dwidth, int cn, int xmin, int xmax) {
    const short alpha[] = {-192, 1216, 1216, -192};
#endif
    for (int k = 0; k < count; k++)
    {
        const unsigned char* S = src[k];
        int* D = dst[k];
        int dx = 0, limit = xmin;
        for (;;)
        {
#if DYNAMIC_COEFF
            for (; dx < limit; dx++, alpha += 4)
#else
            for (; dx < limit; dx++)
#endif
            {
                int j, sx = xofs[dx] - cn;
                int v = 0;
                for (j = 0; j < 4; j++)
                {
                    int sxj = sx + j * cn;
                    if ((unsigned)sxj >= (unsigned)swidth)
                    {
                        while (sxj < 0)
                            sxj += cn;
                        while (sxj >= swidth)
                            sxj -= cn;
                    }
                    v += S[sxj] * alpha[j];
                }
                D[dx] = v;
            }
            if (limit == dwidth)
                break;
#if DYNAMIC_COEFF
            for (; dx < xmax; dx++, alpha += 4)
#else
            for (; dx < xmax; dx++)
#endif
            {
                int sx = xofs[dx];
                D[dx] = S[sx - cn] * alpha[0] + S[sx] * alpha[1] + S[sx + cn] * alpha[2] + S[sx + cn * 2] * alpha[3];
            }
            limit = dwidth;
        }
#if DYNAMIC_COEFF
        alpha -= dwidth * 4;
#endif
    }
}

unsigned char castOp(int val)
{
    int bits = 22;
    int SHIFT = bits;
    int DELTA = (1 << (bits - 1));
    return CLIP3((val + DELTA) >> SHIFT, 0, 255);
}

#if DYNAMIC_COEFF
void vresize(const int** src, unsigned char* dst, const short* beta, int width)
{
    int b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
#else
void vresize(const int** src, unsigned char* dst, int width)
{
    const short beta[] = {-192, 1216, 1216, -192};
    int b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
#endif

    const int* S0 = src[0], * S1 = src[1], * S2 = src[2], * S3 = src[3];

    for (int x = 0; x < width; x++)
        dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
}

static int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

#if DYNAMIC_COEFF
void step(const unsigned char* _src, unsigned char* _dst, const int* xofs, const int* yofs, const short* _alpha, const short* _beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
#else
void step(const unsigned char* _src, unsigned char* _dst, const int* xofs, const int* yofs, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
#endif
{
    int dy, cn = channels;

    int bufstep = (int)((dwidth + 16 - 1) & -16);
    int* _buffer = (int*)malloc(bufstep * ksize * sizeof(int));
    if (_buffer == NULL)
    {
        printf("malloc fails\n");
    }
    const unsigned char* srows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int* rows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int prev_sy[MAX_ESIZE];

    for (int k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = _buffer + bufstep * k;
    }

#if DYNAMIC_COEFF
    const short* beta = _beta + ksize * start;
    for (dy = start; dy < end; dy++, beta += ksize)
#else
    for (dy = start; dy < end; dy++)
#endif
    {
        int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (int k = 0; k < ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, iheight);
            for (k1 = MAX(k1, k); k1 < ksize; k1++)
            {
                if (k1 < MAX_ESIZE && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = MIN(k0, k); // remember the first row that needs to be computed
            srows[k] = _src + (sy * iwidth);
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
#if DYNAMIC_COEFF
            hresize((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                iwidth, dwidth, cn, xmin, xmax);
        vresize((const int**)rows, (_dst + dwidth * dy), beta, dwidth);
#else
            hresize((srows + k0), (rows + k0), ksize - k0, xofs,
                iwidth, dwidth, cn, xmin, xmax);
        vresize((const int**)rows, (_dst + dwidth * dy), dwidth);
#endif
    }
    free(_buffer);
}

void resize(const unsigned char* _src, unsigned char* _dst, int iwidth, int iheight, int dwidth, int dheight)
{
    double  inv_scale_x = (double)dwidth / iwidth;
    double  inv_scale_y = (double)dheight / iheight;

    int  depth = 0, cn = 1;

    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;

    int iscale_x = (int)scale_x;
    int iscale_y = (int)scale_y;

    int k, sx, sy, dx, dy;

    int xmin = 0, xmax = dwidth, width = dwidth * cn;

    // int fixpt = 1;
    float fx, fy;
    int ksize = 4, ksize2;
    ksize2 = ksize / 2;

    unsigned char* _buffer = (unsigned char*)malloc((width + dheight) * (sizeof(int) + sizeof(float) * ksize));

    int* xofs = (int*)_buffer;
    int* yofs = xofs + width;
#if DYNAMIC_COEFF
    float* alpha = (float*)(yofs + dheight);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width * ksize;
    short* ibeta = ialpha + width * ksize;
#endif
    float cbuf[4] = { 0 };

    for (dx = 0; dx < dwidth; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = (int)floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1)
        {
            xmin = dx + 1;
        }

        if (sx + ksize2 >= iwidth)
        {
            xmax = MIN(xmax, dx);
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;

#if DYNAMIC_COEFF
        interpolateCubic(fx, cbuf);
        for (k = 0; k < ksize; k++)
            ialpha[dx * cn * ksize + k] = (short)(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for (; k < cn * ksize; k++)
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
#endif

    }

    for (dy = 0; dy < dheight; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = (int)floor(fy);
        fy -= sy;

        yofs[dy] = sy;

#if DYNAMIC_COEFF
        interpolateCubic(fy, cbuf);
        for (k = 0; k < ksize; k++)
            ibeta[dy * ksize + k] = (short)(cbuf[k] * INTER_RESIZE_COEF_SCALE);
#endif

    }

#if DYNAMIC_COEFF
    step(_src, _dst, xofs, yofs, ialpha, ibeta, iwidth, iheight, dwidth, dheight, cn, ksize, 0, dheight, xmin, xmax);
#else
    step(_src, _dst, xofs, yofs, iwidth, iheight, dwidth, dheight, cn, ksize, 0, dheight, xmin, xmax);
#endif

}