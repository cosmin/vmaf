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

#include <math.h>
#include <mem.h>

void reflect_pad(const float* src, size_t width, size_t height, int reflect, float* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (unsigned int i = reflect; i != (out_height - reflect); i++) {

        for (unsigned int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(float) * width);

        for (unsigned int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (unsigned int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(float) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i+1) * out_width], sizeof(float) * out_width);
    }
}

void integral_image_2(const float* src1, const float* src2, size_t width, size_t height, float* sum)
{
    for (int i = 0; i < (height + 1); ++i)
    {
        for (int j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            float val = src1[(i - 1) * width + (j - 1)] * src2[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void integral_image(const float* src,  size_t width, size_t height, float* sum)
{
    for (int i = 0; i < (height + 1); ++i)
    {
        for (int j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;
 
            float val = src[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i  * (width+1) + j] = val;
        }
    }
}

void compute_metrics(const float* int_1_x, const float* int_1_y, const float* int_2_x, const float* int_2_y, const float* int_xy, size_t width, size_t height, size_t kh, size_t kw, float kNorm, float* var_x, float* var_y, float* cov_xy)
{
    float mx, my, vx, vy, cxy;

    for (unsigned int i = 0; i < (height - kh); i++)
    {
        for (unsigned int j = 0; j < (width - kw); j++)
        {
            mx = (int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw]) / kNorm;
            my = (int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw]) / kNorm;

            vx = ((int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) / kNorm) - (mx * mx);
            vy = ((int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) / kNorm) - (my * my);

            cxy = ((int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) / kNorm) - (mx * my);

            var_x[i * (width - kw) + j] = vx < 0 ? 0 : vx;
            var_y[i * (width - kw) + j] = vy < 0 ? 0 : vy;
            cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : cxy;
        }
    }
}

int compute_vif_funque(const float* x, const float* y, size_t width, size_t height, double *score, double *score_num, double *score_den, int k, int stride, float sigma_nsq)
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = k * k;

    float *x_pad, *y_pad;

    int x_reflect = (int)((kh - stride) / 2);
    int y_reflect = (int)((kw - stride) / 2);

    x_pad = (float*)malloc(sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (float*)malloc(sizeof(float) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

    reflect_pad(x, width, height, x_reflect, x_pad);
    reflect_pad(y, width, height, y_reflect, y_pad);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    float *int_1_x, *int_1_y, *int_2_x, *int_2_y, *int_xy;
    float *mu_x, *mu_y, *var_x, *var_y, *cov_xy;
  
    int_1_x = (float*)calloc((r_width + 1) * (r_height + 1), sizeof(float));
    int_1_y = (float*)calloc((r_width + 1) * (r_height + 1), sizeof(float));
    int_2_x = (float*)calloc((r_width + 1) * (r_height + 1), sizeof(float));
    int_2_y = (float*)calloc((r_width + 1) * (r_height + 1), sizeof(float));
    int_xy = (float*)calloc((r_width + 1) * (r_height + 1), sizeof(float));

    integral_image(x_pad, r_width, r_height, int_1_x);
    integral_image(y_pad, r_width, r_height, int_1_y);
    integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
    integral_image_2(y_pad, y_pad, r_width, r_height, int_2_y);
    integral_image_2(x_pad, y_pad, r_width, r_height, int_xy);

    var_x = (float*)malloc(sizeof(float) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y = (float*)malloc(sizeof(float) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy = (float*)malloc(sizeof(float) * (r_width + 1 - kw) * (r_height + 1 - kh));

    compute_metrics(int_1_x, int_1_y, int_2_x, int_2_y, int_xy, r_width + 1, r_height + 1, kh, kw, k_norm, var_x, var_y, cov_xy);

    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;

    float* g = (float*)malloc(sizeof(float) * s_width * s_height);
    float* sv_sq = (float*)malloc(sizeof(float) * s_width * s_height);
    float exp = 1e-10;
    int index;

    *score = 0;
    *score_num = 0;
    *score_den = 0;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;

            g[index] =  cov_xy[index] / (var_x[index] + exp);
            sv_sq[index] = var_y[index] - g[index] * cov_xy[index];

            if (var_x[index] < exp)
            {
                g[index] = 0;
                sv_sq[index] = var_y[index];
                var_x[index] = 0;
            }

            if (var_y[index] < exp)
            {
                g[index] = 0;
                sv_sq[index] = 0;
            }

            if (g[index] < 0)
            {
                sv_sq[index] = var_x[index];
                g[index] = 0;
            }

            if (sv_sq[index] < exp)
                sv_sq[index] = exp;

            *score_num += log(1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)) + 1e-4;
            *score_den += log(1 + var_x[index] / sigma_nsq) + 1e-4;
        }
    }

    *score += ((*score_num) / (*score_den));

    free(x_pad);
    free(y_pad);
    free(int_1_x);
    free(int_1_y);
    free(int_2_x);
    free(int_2_y);
    free(int_xy);
    free(var_x);
    free(var_y);
    free(cov_xy);
    free(g);
    free(sv_sq);
    ret = 0;

    return ret;
}