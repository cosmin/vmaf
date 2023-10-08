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

#include "funque_strred_options.h"

int compute_strred_funque(const struct dwt2buffers* ref, const struct dwt2buffers* dist, struct dwt2buffers* prev_ref, struct dwt2buffers* prev_dist,
                        size_t width, size_t height, double* srred_vals, double* trred_vals, double* strred_vals,
                        double* srred_approx_vals, double* trred_approx_vals, double* strred_approx_vals,
                        double* spat_vals, double* temp_vals, double* spat_temp_vals,
                        int k, int stride, double sigma_nsq_arg, int index, int level);