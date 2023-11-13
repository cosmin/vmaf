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

#include <libvmaf/picture.h>

#include "integer_funque_filters.h"

void integer_funque_picture_copy(void *src, spat_fil_output_dtype *dst, int dst_stride, int width, int height, int bitdepth)
{
	uint8_t *src_8b = NULL;
	uint16_t *src_hbd = NULL;

	if(bitdepth == 8)
	{
		src_8b = (uint8_t*)src;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst[i*width+j] = (spat_fil_output_dtype) src_8b[i*width+j];
            }
            //dst += dst_stride / sizeof(spat_fil_output_dtype);
        }
    }
    else
    {
		src_hbd = (uint16_t*)src;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst[j] = (spat_fil_output_dtype) src_hbd[j];
            }
            dst += dst_stride / sizeof(spat_fil_output_dtype);
        }
    }

    return;
}
