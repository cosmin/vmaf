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

#pragma once

#ifndef ADM_OPTIONS_H_
#define ADM_OPTIONS_H_

/* Percentage of frame to discard on all 4 sides */
#define ADM_BORDER_FACTOR (0.2)

/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to save intermediate results to files. */
/* #define ADM_OPT_DEBUG_DUMP */

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

/* Enhancement gain imposed on adm, must be >= 1.0, where 1.0 means the gain is completely disabled */
#define DEFAULT_ADM_ENHN_GAIN_LIMIT (100.0)

#define DEFAULT_ADM_LEVELS 4
#define MAX_ADM_LEVELS 4
#define MIN_ADM_LEVELS 1

#endif /* ADM_OPTIONS_H_ */
