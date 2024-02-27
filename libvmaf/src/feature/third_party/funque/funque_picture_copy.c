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

#include <stdint.h>
#include <string.h>
#include <libvmaf/picture.h>

#include "funque_filters.h"
#include "funque_global_options.h"

void funque_picture_copy_hbd(float *dst, ptrdiff_t dst_stride,
                      VmafPicture *src, int offset, int width, int height)
{
    float *float_data = dst;
    uint16_t *data = src->data[0];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float_data[j] = (float) data[j] + offset;
        }
        float_data += dst_stride / sizeof(float);
        data += src->stride[0] / 2;
    }
    return;
}

void funque_picture_copy(float *dst, ptrdiff_t dst_stride,
                  VmafPicture *src, int offset, unsigned bpc, int width, int height)
{
    if (bpc > 8)
        return funque_picture_copy_hbd(dst, dst_stride, src, offset, width, height);

    float *float_data = dst;
    uint8_t *data = src->data[0];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float_data[j] = (float) data[j] + offset;
        }
        float_data += dst_stride / sizeof(float);
        data += src->stride[0];
    }

    return;
}

int copy_frame_funque(const struct dwt2buffers *ref, const struct dwt2buffers *dist,
                      struct dwt2buffers *shared_ref, struct dwt2buffers *shared_dist, size_t width,
                      size_t height)
{
    int subband;
    int total_subbands = DEFAULT_BANDS;

    for(subband = 0; subband < total_subbands; subband++) {
        memcpy(shared_ref->bands[subband], ref->bands[subband], width * height * sizeof(float));
        memcpy(shared_dist->bands[subband], dist->bands[subband], width * height * sizeof(float));
    }
    shared_ref->width = ref->width;
    shared_ref->height = ref->height;
    shared_ref->stride = ref->stride;

    shared_dist->width = dist->width;
    shared_dist->height = dist->height;
    shared_dist->stride = dist->stride;

    return 0;
}
