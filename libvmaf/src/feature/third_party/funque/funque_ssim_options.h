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

#ifndef FUNQUE_SSIM_OPTIONS_H_
#define FUNQUE_SSIM_OPTIONS_H_

#define ENABLE_MINK3POOL 0
#define ENABLE_PADDING 0

#define DEFAULT_SSIM_LEVELS 4
#define DEFAULT_MS_SSIM_LEVELS 4

typedef struct SsimScore {
    double mean;
    double mink3;
} SsimScore;

typedef struct MsSsimScore {
    double ssim_mean;
    double l_mean;
    double cs_mean;
    double ssim_mink3;
    double l_mink3;
    double cs_mink3;
    
    double ms_ssim_mean;
    double ms_ssim_mink3;

    float **var_x_cum;
    float **var_y_cum;
    float **cov_xy_cum;
} MsSsimScore;

static inline double ssim_clip(double value, double low, double high)
{
    return value < low ? low : (value > high ? high : value);
}

#endif //FUNQUE_SSIM_OPTIONS_H_
