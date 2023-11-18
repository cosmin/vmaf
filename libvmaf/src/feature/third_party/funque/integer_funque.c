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

#include <errno.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

// #include "config.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "funque_global_options.h"
#include "funque_vif_options.h"
#include "integer_funque_filters.h"
#include "common/macros.h"
#include "integer_funque_vif.h"
#include "integer_funque_adm.h"
#include "funque_adm_options.h"
#include "integer_funque_motion.h"
#include "integer_funque_ssim.h"
#include "funque_ssim_options.h"
#include "resizer.h"
#include "integer_picture_copy.h"
#include "integer_funque_strred.h"
#include "funque_strred_options.h"

#if ARCH_AARCH64
#include "arm64/integer_funque_filters_neon.h"
#include "arm64/integer_funque_ssim_neon.h"
#include "arm64/integer_funque_motion_neon.h"
#include "arm64/integer_funque_adm_neon.h"
#include "arm64/resizer_neon.h"
#include "arm64/integer_funque_vif_neon.h"
#elif ARCH_ARM
#include "arm32/integer_funque_filters_armv7.h"
#include "arm32/integer_funque_ssim_armv7.h"
#include "arm32/integer_funque_adm_armv7.h"
#endif

#if ARCH_X86
#include "x86/integer_funque_filters_avx2.h"
#include "x86/integer_funque_vif_avx2.h"
#include "x86/integer_funque_ssim_avx2.h"
#include "x86/integer_funque_adm_avx2.h"
#include "x86/integer_funque_motion_avx2.h"
#include "x86/resizer_avx2.h"

#if HAVE_AVX512
#include "x86/integer_funque_filters_avx512.h"
#include "x86/resizer_avx512.h"
#include "x86/integer_funque_ssim_avx512.h"
#include "x86/integer_funque_adm_avx512.h"
#include "x86/integer_funque_vif_avx512.h"
#endif
#endif

#include "cpu.h"

#include <time.h>
#include <sys/time.h>

typedef struct IntFunqueState
{
    size_t width_aligned_stride;
    bool debug;

    VmafPicture res_ref_pic;
    VmafPicture res_dist_pic;

    const char *wavelet_csfs;
    spat_fil_coeff_dtype csf_factors[4][4];
    uint16_t csf_interim_rnd[4][4];
    uint8_t csf_interim_shift[4][4];

    size_t resizer_out_stride;
    spat_fil_inter_dtype *spat_tmp_buf;
    spat_fil_output_dtype *filter_buffer;
    size_t filter_buffer_stride;
    i_dwt2buffers i_ref_dwt2out[4];
    i_dwt2buffers i_dist_dwt2out[4];
    i_dwt2buffers i_prev_ref[4];
    i_dwt2buffers i_prev_dist[4];

    // funque configurable parameters
    bool enable_resize;
    bool enable_spatial_csf;
    int num_taps;
    int vif_levels;
    int adm_levels;
    int needed_dwt_levels;
    int needed_full_dwt_levels;
    int ssim_levels;
    int strred_levels;
    double norm_view_dist;
    int ref_display_height;

    // VIF extra variables
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    uint32_t log_18[262144];
    uint32_t log_16[65536];

    // ADM extra variables
    double adm_enhn_gain_limit;
    int adm_csf_mode;
	int32_t adm_div_lookup[65537];

    VmafDictionary *feature_name_dict;

    ModuleFunqueState modules;
    ResizerState resize_module;
    strred_results strred_scores[4];

} IntFunqueState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(IntFunqueState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_resize",
        .alias = "rsz",
        .help = "Enable resize for funque",
        .offset = offsetof(IntFunqueState, enable_resize),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "enable_spatial_csf",
        .alias = "gcsf",
        .help = "enable the global CSF based on spatial filter",
        .offset = offsetof(IntFunqueState, enable_spatial_csf),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "num_taps",
        .alias = "ntaps",
        .help = "Select number of taps to be used for spatial filter",
        .offset = offsetof(IntFunqueState, num_taps),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.b = NADENAU_SPAT_5_TAP_FILTER,
        .min = NADENAU_SPAT_5_TAP_FILTER,
        .max = NGAN_21_TAP_FILTER,
    },
{
        .name = "norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(IntFunqueState, norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
{
        .name = "ref_display_height",
        .alias = "rdf",
        .help = "reference display height in pixels",
        .offset = offsetof(IntFunqueState, ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_levels",
        .alias = "vifl",
        .help = "Number of levels in VIF",
        .offset = offsetof(IntFunqueState, vif_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_VIF_LEVELS,
        .min = MIN_LEVELS,
        .max = MAX_LEVELS,
    },
    {
        .name = "ssim_levels",
        .alias = "ssiml",
        .help = "Number of DWT levels for SSIM",
        .offset = offsetof(IntFunqueState, ssim_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_SSIM_LEVELS,
        .min = MIN_LEVELS,
        .max = MAX_LEVELS,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(IntFunqueState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(IntFunqueState, vif_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_levels",
        .alias = "adml",
        .help = "Number of levels in ADM",
        .offset = offsetof(IntFunqueState, adm_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_LEVELS,
        .min = MIN_ADM_LEVELS,
        .max = MAX_ADM_LEVELS,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(IntFunqueState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(IntFunqueState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "strred_levels",
        .alias = "strredl",
        .help = "Number of levels in STRRED",
        .offset = offsetof(IntFunqueState, strred_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_STRRED_LEVELS,
        .min = MIN_LEVELS,
        .max = MAX_LEVELS,
    },

    {0}};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    IntFunqueState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        goto fail;

    if (s->enable_resize)
    {
        w = (w + 1) >> 1;
        h = (h + 1) >> 1;
    }

    s->needed_dwt_levels = MAX(MAX(s->vif_levels, s->adm_levels), s->ssim_levels);
    s->needed_full_dwt_levels = MAX(s->adm_levels, s->ssim_levels);

    s->width_aligned_stride = ALIGN_CEIL(w * sizeof(float));
    s->resizer_out_stride = (s->width_aligned_stride + 3) / 4;

    if (s->enable_resize)
    {
        int bitdepth_factor = (bpc == 8 ? 1 : 2);
        s->res_ref_pic.data[0] = aligned_malloc(s->resizer_out_stride * h * bitdepth_factor, 32);
        if (!s->res_ref_pic.data[0])
            goto fail;
        s->res_dist_pic.data[0] = aligned_malloc(s->resizer_out_stride * h * bitdepth_factor, 32);

        if (!s->res_dist_pic.data[0])
            goto fail;
    }

    /* This buffer is common along spatial and wavelet buffers*/
    s->filter_buffer = aligned_malloc(ALIGN_CEIL(w * sizeof(spat_fil_output_dtype)) * h, 32);
    if (!s->filter_buffer)
        goto fail;
    s->filter_buffer_stride = w * sizeof(spat_fil_output_dtype);

    /*currently hardcoded to nadeanu_weight To be made configurable via model file*/
    s->wavelet_csfs = "nadenau_weight";

    if (s->enable_spatial_csf) {
        s->spat_tmp_buf = aligned_malloc(ALIGN_CEIL(w * sizeof(spat_fil_inter_dtype)), 32);
        if (!s->spat_tmp_buf) 
            goto fail;
        //memset(s->spat_tmp_buf, 0, ALIGN_CEIL(w * sizeof(spat_fil_inter_dtype)));
    } else {
        if(strcmp(s->wavelet_csfs, "nadenau_weight") == 0) {
            for(int level = 0; level < 4; level++) {
                s->csf_factors[level][0] = i_nadenau_weight_coeffs[level][0];
                s->csf_factors[level][1] = i_nadenau_weight_coeffs[level][1];
                s->csf_factors[level][2] = i_nadenau_weight_coeffs[level][2];
                s->csf_factors[level][3] = i_nadenau_weight_coeffs[level][3];

                s->csf_interim_shift[level][0] = i_nadenau_weight_interim_shift[level][0];
                s->csf_interim_shift[level][1] = i_nadenau_weight_interim_shift[level][1];
                s->csf_interim_shift[level][2] = i_nadenau_weight_interim_shift[level][2];
                s->csf_interim_shift[level][3] = i_nadenau_weight_interim_shift[level][3];

                s->csf_interim_rnd[level][0] = 1 << (i_nadenau_weight_interim_shift[level][0] - 1);
                s->csf_interim_rnd[level][1] = 1 << (i_nadenau_weight_interim_shift[level][1] - 1);
                s->csf_interim_rnd[level][2] = 1 << (i_nadenau_weight_interim_shift[level][2] - 1);
                s->csf_interim_rnd[level][3] = 1 << (i_nadenau_weight_interim_shift[level][3] - 1);

            }
        }
    }
    
    int last_w, last_h;
    last_w = w;
    last_h = h;

    for (int level = 0; level < s->needed_dwt_levels; level++) {
        // dwt output dimensions
        s->i_ref_dwt2out[level].width = (int)(last_w + 1) / 2;
        s->i_ref_dwt2out[level].height = (int)(last_h + 1) / 2;
        s->i_ref_dwt2out[level].stride = (int)ALIGN_CEIL(s->i_ref_dwt2out[level].width * sizeof(dwt2_dtype));

        s->i_dist_dwt2out[level].width = (int)(last_w + 1) / 2;
        s->i_dist_dwt2out[level].height = (int)(last_h + 1) / 2;
        s->i_dist_dwt2out[level].stride = (int)ALIGN_CEIL(s->i_dist_dwt2out[level].width * sizeof(dwt2_dtype));

        s->i_prev_ref[level].width = (int)(last_w + 1) / 2;
        s->i_prev_ref[level].height = (int)(last_h + 1) / 2;
        s->i_prev_ref[level].stride = (int)ALIGN_CEIL(s->i_prev_ref[level].width * sizeof(dwt2_dtype));

        s->i_prev_dist[level].width = (int)(last_w + 1) / 2;
        s->i_prev_dist[level].height = (int)(last_h + 1) / 2;
        s->i_prev_dist[level].stride = (int)ALIGN_CEIL(s->i_prev_dist[level].width * sizeof(dwt2_dtype));

        s->i_prev_ref[level].bands[0] = NULL;
        s->i_prev_dist[level].bands[0] = NULL;

        // Memory allocation for dwt output bands
        for (unsigned i = 0; i < 4; i++)
        {
            s->i_ref_dwt2out[level].bands[i] = aligned_malloc(s->i_ref_dwt2out[level].stride * s->i_ref_dwt2out[level].height, 32);
            if (!s->i_ref_dwt2out[level].bands[i])
                goto fail;

            s->i_dist_dwt2out[level].bands[i] = aligned_malloc(s->i_dist_dwt2out[level].stride * s->i_dist_dwt2out[level].height, 32);
            if (!s->i_dist_dwt2out[level].bands[i])
                goto fail;

            s->i_prev_ref[level].bands[i] = aligned_malloc(s->i_prev_ref[level].stride * s->i_prev_ref[level].height, 32);
            if (!s->i_prev_ref[level].bands[i])
                goto fail;

            s->i_prev_dist[level].bands[i] = aligned_malloc(s->i_prev_dist[level].stride * s->i_prev_dist[level].height, 32);
            if (!s->i_prev_dist[level].bands[i])
                goto fail;
        }

        /* Last width and height is half of the current layer */
        last_w = s->i_ref_dwt2out[level].width;
        last_h = s->i_ref_dwt2out[level].height;

    }

    s->modules.integer_funque_picture_copy = integer_funque_picture_copy;
    s->modules.integer_spatial_filter = integer_spatial_filter;
    s->modules.integer_funque_dwt2 = integer_funque_dwt2;
    //s->modules.integer_funque_dwt2_wavelet = integer_funque_dwt2_wavelet;
    s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque;
    s->modules.integer_compute_ms_ssim_funque = integer_compute_ssim_funque; // TODO:@niranjankumar-ittiam Assign your function call
    s->modules.integer_funque_image_mad = integer_funque_image_mad_c;
    s->modules.integer_funque_adm_decouple = integer_adm_decouple_c;
    s->modules.integer_adm_integralimg_numscore = integer_adm_integralimg_numscore_c;
    s->modules.integer_compute_vif_funque = integer_compute_vif_funque_c;
    s->resize_module.resizer_step = step;
    s->resize_module.hbd_resizer_step = hbd_step;
    s->modules.integer_funque_vifdwt2_band0 = integer_funque_vifdwt2_band0;

    s->modules.integer_compute_strred_funque = integer_compute_strred_funque_c;
    s->modules.integer_copy_prev_frame_strred_funque = integer_copy_prev_frame_strred_funque_c;

#if ARCH_AARCH64
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_ARM_CPU_FLAG_NEON) {
        if (bpc == 8)
        {
            s->modules.integer_spatial_filter = integer_spatial_filter_neon;
        }
        s->modules.integer_funque_dwt2 = integer_funque_dwt2_neon;
        s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_neon;
        s->modules.integer_funque_adm_decouple = integer_adm_decouple_neon;
        s->modules.integer_compute_vif_funque = integer_compute_vif_funque_neon;
        //Commenting this since C was performing better
        // s->resize_module.resizer_step = step_neon;
        // s->modules.integer_funque_image_mad = integer_funque_image_mad_neon;
        // s->modules.integer_adm_integralimg_numscore = integer_adm_integralimg_numscore_neon;
    }
#elif ARCH_ARM
    if (bpc == 8)
    {
        s->modules.integer_spatial_filter = integer_spatial_filter_armv7;
    }
    s->modules.integer_funque_dwt2 = integer_funque_dwt2_armv7;
    s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_armv7;
    s->modules.integer_funque_adm_decouple = integer_dlm_decouple_armv7;
#elif ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        s->modules.integer_spatial_filter = integer_spatial_filter;
        s->modules.integer_funque_dwt2 = integer_funque_dwt2;
        s->modules.integer_funque_vifdwt2_band0 = integer_funque_vifdwt2_band0_avx2;
        s->modules.integer_compute_vif_funque = integer_compute_vif_funque_avx2;
        s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_avx2;
        s->modules.integer_funque_adm_decouple = integer_adm_decouple_c;
        s->modules.integer_funque_image_mad = integer_funque_image_mad_avx2;
        s->resize_module.resizer_step = step_avx2;
        s->resize_module.hbd_resizer_step = hbd_step_avx2;

        s->modules.integer_compute_strred_funque = integer_compute_strred_funque_c;
        s->modules.integer_copy_prev_frame_strred_funque = integer_copy_prev_frame_strred_funque_c;

    }

#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        s->modules.integer_spatial_filter = integer_spatial_filter_avx512;
        s->resize_module.resizer_step = step_avx512;
        s->resize_module.hbd_resizer_step = hbd_step_avx512;
        s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_avx512;
        s->modules.integer_funque_adm_decouple = integer_adm_decouple_avx512;
        s->modules.integer_funque_dwt2 = integer_funque_dwt2_avx512;
        s->modules.integer_funque_vifdwt2_band0 = integer_funque_vifdwt2_band0_avx512;
        s->modules.integer_compute_vif_funque = integer_compute_vif_funque_avx512;
    }
#endif
#endif

    //funque_log_generate(s->log_18);
	div_lookup_generator(s->adm_div_lookup);
#if USE_LOG_18
    strred_funque_log_generate(s->log_18);
#else
    strred_log_generate(s->log_16);
#endif

    return 0;

fail:
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
    if (s->filter_buffer)
        aligned_free(s->filter_buffer);
    if (s->spat_tmp_buf) 
        aligned_free(s->spat_tmp_buf);

    for (int level = 0; level < s->needed_dwt_levels; level++) {
        for (unsigned i = 0; i < 4; i++) {
            if (s->i_ref_dwt2out[level].bands[i])
                aligned_free(s->i_ref_dwt2out[level].bands[i]);
            if (s->i_dist_dwt2out[level].bands[i])
                aligned_free(s->i_dist_dwt2out[level].bands[i]);
            if (s->i_prev_ref[level].bands[i])
                aligned_free(s->i_prev_ref[level].bands[i]);
            if (s->i_prev_dist[level].bands[i])
                aligned_free(s->i_prev_dist[level].bands[i]);
        }
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}


static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    IntFunqueState *s = fex->priv;
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    VmafPicture *res_ref_pic = &s->res_ref_pic;
    VmafPicture *res_dist_pic = &s->res_dist_pic;
    if (s->enable_resize)
    {
        res_ref_pic->bpc = ref_pic->bpc;
        res_ref_pic->h[0] = ref_pic->h[0] / 2;
        res_ref_pic->w[0] = ref_pic->w[0] / 2;
        res_ref_pic->stride[0] = ref_pic->stride[0] / 2;
        res_ref_pic->pix_fmt = ref_pic->pix_fmt;
        res_ref_pic->ref = ref_pic->ref;

        res_dist_pic->bpc = dist_pic->bpc;
        res_dist_pic->h[0] = dist_pic->h[0] / 2;
        res_dist_pic->w[0] = dist_pic->w[0] / 2;
        res_dist_pic->stride[0] = dist_pic->stride[0] / 2;
        res_dist_pic->pix_fmt = dist_pic->pix_fmt;
        res_dist_pic->ref = dist_pic->ref;

        if (ref_pic->bpc == 8)
            resize(s->resize_module ,ref_pic->data[0], res_ref_pic->data[0], ref_pic->w[0], ref_pic->h[0], res_ref_pic->w[0], res_ref_pic->h[0]);
        else
            hbd_resize(s->resize_module ,(unsigned short *)ref_pic->data[0], (unsigned short *)res_ref_pic->data[0], ref_pic->w[0], ref_pic->h[0], res_ref_pic->w[0], res_ref_pic->h[0], ref_pic->bpc);
        
        if (dist_pic->bpc == 8)
            resize(s->resize_module ,dist_pic->data[0], res_dist_pic->data[0], dist_pic->w[0], dist_pic->h[0], res_dist_pic->w[0], res_dist_pic->h[0]);
        else
            hbd_resize(s->resize_module ,(unsigned short *)dist_pic->data[0], (unsigned short *)res_dist_pic->data[0], dist_pic->w[0], dist_pic->h[0], res_dist_pic->w[0], res_dist_pic->h[0], ref_pic->bpc);
    }
    else
    {
        res_ref_pic = ref_pic;
        res_dist_pic = dist_pic;
    }

    int bitdepth_pow2 = (1 << res_ref_pic->bpc) - 1;

    if (s->enable_spatial_csf) {
        s->modules.integer_spatial_filter(res_ref_pic->data[0], s->filter_buffer, s->filter_buffer_stride, res_ref_pic->w[0], res_ref_pic->h[0], (int) res_ref_pic->bpc, s->spat_tmp_buf, s->num_taps);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_ref_dwt2out[0].stride, res_ref_pic->w[0], res_ref_pic->h[0], s->enable_spatial_csf, -1);
        s->modules.integer_spatial_filter(res_dist_pic->data[0], s->filter_buffer, s->filter_buffer_stride, res_dist_pic->w[0], res_dist_pic->h[0], (int) res_dist_pic->bpc, s->spat_tmp_buf, s->num_taps);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride, &s->i_dist_dwt2out[0], s->i_dist_dwt2out[0].stride, res_dist_pic->w[0], res_dist_pic->h[0], s->enable_spatial_csf, -1);
    } else {
        // TODO: Add a function to convert 8-bit buffer to 16-bit buuffer picture
        s->modules.integer_funque_picture_copy(res_ref_pic->data[0], s->filter_buffer, s->filter_buffer_stride, res_ref_pic->w[0], res_ref_pic->h[0], (int) res_ref_pic->bpc);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_ref_dwt2out[0].stride, res_ref_pic->w[0], res_ref_pic->h[0], s->enable_spatial_csf, 0);

        s->modules.integer_funque_picture_copy(res_dist_pic->data[0], s->filter_buffer, s->filter_buffer_stride, res_dist_pic->w[0], res_dist_pic->h[0], (int) res_dist_pic->bpc);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride, &s->i_dist_dwt2out[0], s->i_dist_dwt2out[0].stride, res_dist_pic->w[0], res_dist_pic->h[0], s->enable_spatial_csf, 0);
    }

    double ssim_score[MAX_LEVELS];
    double adm_score[MAX_LEVELS], adm_score_num[MAX_LEVELS], adm_score_den[MAX_LEVELS];
    double vif_score[MAX_LEVELS], vif_score_num[MAX_LEVELS], vif_score_den[MAX_LEVELS];

    double adm_den = 0.0;
    double adm_num = 0.0;

    double vif_den = 0.0;
    double vif_num = 0.0;

    int16_t spatfilter_shifts = 2 * SPAT_FILTER_COEFF_SHIFT - SPAT_FILTER_INTER_SHIFT - SPAT_FILTER_OUT_SHIFT - (res_ref_pic->bpc - 8);
    int16_t dwt_shifts = 2 * DWT2_COEFF_UPSHIFT - DWT2_INTER_SHIFT - DWT2_OUT_SHIFT;
    float pending_div_factor = (1 << ( spatfilter_shifts + dwt_shifts)) * bitdepth_pow2;
    int16_t strred_pending_div = spatfilter_shifts + dwt_shifts;

    for(int level = 0; level < s->needed_dwt_levels; level++) // For ST-RRED Debugging level set to 0
    {
        if (level+1 < s->needed_dwt_levels) {
            if (level+1 > s->needed_full_dwt_levels - 1) {
                // from here on out we only need approx band for VIF
                integer_funque_vifdwt2_band0(s->i_ref_dwt2out[level].bands[0], s->i_ref_dwt2out[level + 1].bands[0],  ((s->i_ref_dwt2out[level + 1].stride + 1) / 2), s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height);
            } else {
                        // compute full DWT if either SSIM or ADM need it for this level
                        integer_funque_dwt2(s->i_ref_dwt2out[level].bands[0], s->i_ref_dwt2out[level].stride, &s->i_ref_dwt2out[level + 1], s->i_ref_dwt2out[level + 1].stride,
                                    s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height, s->enable_spatial_csf, level);
                        integer_funque_dwt2(s->i_dist_dwt2out[level].bands[0], s->i_dist_dwt2out[level].stride, &s->i_dist_dwt2out[level + 1],s->i_dist_dwt2out[level + 1].stride, 
                                    s->i_dist_dwt2out[level].width, s->i_dist_dwt2out[level].height, s->enable_spatial_csf, level);
            }
        }

        // TODO:@Priyanka-885 - Filters - Wavaelet CSF function call goes here
        if (!s->enable_spatial_csf) {
            if (level < s->adm_levels || level < s->ssim_levels) {
                // we need full CSF on all bands
                integer_funque_dwt2_inplace_csf(&s->i_ref_dwt2out[level], s->csf_factors[level], 0, 3, s->csf_interim_rnd[level], s->csf_interim_shift[level], level);
                integer_funque_dwt2_inplace_csf(&s->i_dist_dwt2out[level], s->csf_factors[level], 0, 3, s->csf_interim_rnd[level], s->csf_interim_shift[level], level);
            } else {
                // we only need CSF on approx band
                integer_funque_dwt2_inplace_csf(&s->i_ref_dwt2out[level], s->csf_factors[level], 0, 0, s->csf_interim_rnd[level], s->csf_interim_shift[level], level);
                integer_funque_dwt2_inplace_csf(&s->i_dist_dwt2out[level], s->csf_factors[level], 0, 0, s->csf_interim_rnd[level], s->csf_interim_shift[level], level);
            }
        }

        err = integer_compute_adm_funque(s->modules, s->i_ref_dwt2out[level], s->i_dist_dwt2out[level], &adm_score[level], &adm_score_num[level], &adm_score_den[level], s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height, 0.2, s->adm_div_lookup);

        if (err)
            return err;

        err = s->modules.integer_compute_ms_ssim_funque(&s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &ssim_score[level], 1, 0.01, 0.03, pending_div_factor, s->adm_div_lookup);

        if (err)
            return err;

#if 0 // VIF and ssim is not used
        err = s->modules.integer_compute_ssim_funque(&s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &ssim_score[level], 1, 0.01, 0.03, pending_div_factor, s->adm_div_lookup);

        if (err)
            return err;

        if (level == 0)
        {
#if USE_DYNAMIC_SIGMA_NSQ
            err = s->modules.integer_compute_vif_funque(s->i_ref_dwt2out[level].bands[0], s->i_dist_dwt2out[level].bands[0], s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height,
                    &vif_score[0], &vif_score_num[0], &vif_score_den[0], 9, 1, (double)5.0, (int16_t) pending_div_factor, s->log_18, 0);
#else
            err = s->modules.integer_compute_vif_funque(s->i_ref_dwt2out[level].bands[0], s->i_dist_dwt2out[level].bands[0], s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height,
                    &vif_score[0], &vif_score_num[0], &vif_score_den[0], 9, 1, (double)5.0, (int16_t) pending_div_factor, s->log_18);
#endif

            if (err) return err;
        }
        else
        {
            int vifdwt_stride = (s->i_dwt2_stride + 1)/2;
            int vifdwt_width  = s->i_ref_dwt2out[level].width;
            int vifdwt_height = s->i_ref_dwt2out[level].height;
            //The VIF function reuses the band1, band2, band3 of s->ref_dwt2out &   s->dist_dwt2out
            //Hence VIF is called in the end
            //If the individual modules(VIF,ADM,motion,ssim) are moved to different     files,
            // separate memory allocation for higher level VIF buffers might be needed

            int16_t vif_pending_div = (1 << ( spatfilter_shifts + (dwt_shifts <<    level))) * bitdepth_pow2;
            s->modules.integer_funque_vifdwt2_band0(s->i_ref_dwt2out[level].bands[level-1],    s->i_ref_dwt2out[level].bands[level], vifdwt_stride, vifdwt_width, vifdwt_height);
            s->modules.integer_funque_vifdwt2_band0(s->i_dist_dwt2out[level].bands[level-1],   s->i_dist_dwt2out[level].bands[level], vifdwt_stride, vifdwt_width, vifdwt_height);
            vifdwt_stride = (vifdwt_stride + 1)/2;
            vifdwt_width = (vifdwt_width + 1)/2;
            vifdwt_height = (vifdwt_height + 1)/2;

            // TODO: The below code looks phishy // Why is level used in bands i.e., bands[level]
#if USE_DYNAMIC_SIGMA_NSQ
            err = s->modules.integer_compute_vif_funque(s->i_ref_dwt2out[level].bands[level], s->i_dist_dwt2out[level].bands[level], vifdwt_width, vifdwt_height,
                                    &vif_score[level], &vif_score_num[level], &vif_score_den[level], 9, 1, (double)5.0, (int16_t) vif_pending_div, s->log_18, level);
#else
            err = s->modules.integer_compute_vif_funque(s->i_ref_dwt2out[level].bands[level], s->i_dist_dwt2out[level].bands[level], vifdwt_width, vifdwt_height,
                                    &vif_score[level], &vif_score_num[level], &vif_score_den[level], 9, 1, (double)5.0, (int16_t) vif_pending_div, s->log_18);
#endif

            if (err) return err;
        }
#endif

        if(level <= s->strred_levels - 1) {

            if(index == 0) {
                err |= s->modules.integer_copy_prev_frame_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height);
            }
            else {
#if USE_LOG_18
                err |= s->modules.integer_compute_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height,
                    &s->strred_scores[level], BLOCK_SIZE, level, s->log_18, strred_pending_div, 1, s->enable_spatial_csf);
#else
                err |= s->modules.integer_compute_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height,
                    &s->strred_scores[level], BLOCK_SIZE, level, s->log_16, strred_pending_div, 1);
#endif
                err |= s->modules.integer_copy_prev_frame_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height);
            }
            if (err) return err;
        }



    }

    double vif = vif_den > 0 ? vif_num / vif_den : 1.0;
    double adm = adm_den > 0 ? adm_num / adm_den : 1.0;

#if 0
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_integer_feature_vif_score",
                                                   vif, index);
#endif
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_integer_feature_vif_scale0_score",
                                                   vif_score[0], index);

    // if (s->vif_levels > 1) {
    //     err |= vmaf_feature_collector_append_with_dict(feature_collector,
    //                                                    s->feature_name_dict, "FUNQUE_integer_feature_vif_scale1_score",
    //                                                    vif_score[1], index);

    //     if (s->vif_levels > 2) {
    //         err |= vmaf_feature_collector_append_with_dict(feature_collector,
    //                                                        s->feature_name_dict, "FUNQUE_integer_feature_vif_scale2_score",
    //                                                        vif_score[2], index);

    //         if (s->vif_levels > 3) {
    //             err |= vmaf_feature_collector_append_with_dict(feature_collector,
    //                                                            s->feature_name_dict, "FUNQUE_integer_feature_vif_scale3_score",
    //                                                            vif_score[3], index);
    //         }
    //     }
    // }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_integer_feature_adm_score",
                                                   adm, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_integer_feature_adm_scale0_score",
                                                   adm_score[0], index);
//    if (s->adm_levels > 1) {
//
//        err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                       s->feature_name_dict, "FUNQUE_integer_feature_adm_scale1_score",
//                                                       adm_score[1], index);
//
//        if (s->adm_levels > 2) {
//            err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                           s->feature_name_dict, "FUNQUE_integer_feature_adm_scale2_score",
//                                                           adm_score[2], index);
//
//            if (s->adm_levels > 3) {
//                err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                               s->feature_name_dict, "FUNQUE_integer_feature_adm_scale3_score",
//                                                               adm_score[3], index);
//            }
//        }
//    }

    err |= vmaf_feature_collector_append(feature_collector, "FUNQUE_integer_feature_ssim_scale0_score",
                                         ssim_score[0], index);

//    if (s->ssim_levels > 1) {
//        err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                       s->feature_name_dict, "FUNQUE_integer_feature_ssim_scale1_score",
//                                                       ssim_score[1], index);
//
//        if (s->ssim_levels > 2) {
//            err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                           s->feature_name_dict, "FUNQUE_integer_feature_ssim_scale2_score",
//                                                           ssim_score[2], index);
//
//            if (s->ssim_levels > 3) {
//                err |= vmaf_feature_collector_append_with_dict(feature_collector,
//                                                               s->feature_name_dict, "FUNQUE_integer_feature_ssim_scale3_score",
//                                                               ssim_score[3], index);
//            }
//        }
//    }

    err |= vmaf_feature_collector_append(feature_collector, "FUNQUE_integer_feature_strred_scale0_score",
                                         s->strred_scores[0].srred_vals[0], index);
    if (s->strred_levels > 1) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_integer_feature_strred_scale1_score",
                                                       s->strred_scores[1].srred_vals[1], index);

        if (s->strred_levels > 2) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                           s->feature_name_dict, "FUNQUE_integer_feature_strred_scale2_score",
                                                           s->strred_scores[2].srred_vals[2], index);

            if (s->strred_levels > 3) {
                err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                               s->feature_name_dict, "FUNQUE_integer_feature_strred_scale3_score",
                                                               s->strred_scores[3].srred_vals[3], index);
            }
        }
    }

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    IntFunqueState *s = fex->priv;
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
    if (s->filter_buffer)
        aligned_free(s->filter_buffer);
    if (s->spat_tmp_buf) 
        aligned_free(s->spat_tmp_buf);
    

    for(int level = 0; level < 4; level++) {
        for (unsigned i = 0; i < 4; i++)
        {
            if (s->i_ref_dwt2out[level].bands[i])
                aligned_free(s->i_ref_dwt2out[level].bands[i]);
            if (s->i_dist_dwt2out[level].bands[i])
                aligned_free(s->i_dist_dwt2out[level].bands[i]);
            if (s->i_prev_ref[level].bands[i])
                aligned_free(s->i_prev_ref[level].bands[i]);
            if (s->i_prev_dist[level].bands[i])
                aligned_free(s->i_prev_dist[level].bands[i]);
        }
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {

    "FUNQUE_integer_feature_vif_scale0_score",

    "FUNQUE_integer_feature_adm_score", "FUNQUE_integer_feature_adm_scale0_score",

    "FUNQUE_integer_feature_ssim_scale0_score",

    "FUNQUE_integer_feature_strred_scale0_score",
    "FUNQUE_integer_feature_strred_scale1_score",
    "FUNQUE_integer_feature_strred_scale2_score",
    "FUNQUE_integer_feature_strred_scale3_score",

    NULL};

VmafFeatureExtractor vmaf_fex_integer_funque = {
    .name = "integer_funque",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(IntFunqueState),
    .provided_features = provided_features,
};