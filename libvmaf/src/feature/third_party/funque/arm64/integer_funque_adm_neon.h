#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

void integer_integral_image_adm_sums_neon(i_dwt2buffers pyr_1, adm_u16_dtype *x, int k, int stride, i_adm_buffers masked_pyr, int width, int height, int band_index);

void integer_dlm_decouple_neon(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, u_adm_buffers i_dlm_add, int32_t *adm_div_lookup)