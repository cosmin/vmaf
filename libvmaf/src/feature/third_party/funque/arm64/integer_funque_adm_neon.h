#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

void integer_integral_image_adm_sums_neon(i16_adm_buffers pyr_1, uint16_t *x, int k, int stride, i32_adm_buffers masked_pyr, int width, int height, int band_index);