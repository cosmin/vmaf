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


const int INTER_RESIZE_COEF_BITS = 11;
// const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
const int INTER_RESIZE_COEF_SCALE = 2048;
static const int MAX_ESIZE = 16;

#define CLIP3(X, MIN, MAX) ((X < MIN) ? MIN : (X > MAX) ? MAX \
                                                        : X)
#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
#define MIN(LEFT, RIGHT) (LEFT < RIGHT ? LEFT : RIGHT)

// enabled by default for funque since resize factor is always 0.5, disabled otherwise
#define OPTIMISED_COEFF 1

void resize(const unsigned char* _src, unsigned char* _dst, int iwidth, int iheight, int dwidth, int dheight);