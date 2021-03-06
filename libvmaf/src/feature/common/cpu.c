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

#include "cpudetect.h"
#include "feature/common/cpu.h"


enum vmaf_cpu cpu_autodetect()
{
    X86Capabilities caps = query_x86_capabilities();

    if (caps.avx)
        return VMAF_CPU_AVX;
    else if (caps.sse2)
        return VMAF_CPU_SSE2;
    else
        return VMAF_CPU_NONE;
}
