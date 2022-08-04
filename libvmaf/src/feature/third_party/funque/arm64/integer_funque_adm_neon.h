#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

void integer_integral_image_adm_sums_neon(i_dwt2buffers pyr_1, adm_u16_dtype *x, int k, int stride, i_adm_buffers masked_pyr, int width, int height, int band_index);