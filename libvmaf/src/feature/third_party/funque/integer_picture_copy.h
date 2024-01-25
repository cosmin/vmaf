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
#include <stddef.h>
#include "integer_funque_filters.h"

void integer_funque_picture_copy(void *src, spat_fil_output_dtype *dst, int dst_stride, int width,
                                 int height, int bitdepth);

int integer_copy_frame_funque(const struct i_dwt2buffers* ref, const struct i_dwt2buffers* dist,
                      struct i_dwt2buffers* shared_ref, struct i_dwt2buffers* shared_dist,
                      size_t width, size_t height);