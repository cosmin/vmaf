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

int compute_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, double *score, int max_val, float K1, float K2);
int compute_ms_ssim_funque(dwt2buffers *ref, dwt2buffers *dist, MsSsimScore *score, int max_val,
                           float K1, float K2, int n_levels);
int compute_ms_ssim_mean_scales(MsSsimScore *score, int n_levels);