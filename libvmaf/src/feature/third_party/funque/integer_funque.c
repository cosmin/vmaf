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
#include "arm64/integer_funque_strred_neon.h"
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
#include "x86/integer_funque_strred_avx2.h"
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

    void *pad_ref;
    void *pad_dist;

    const char *wavelet_csfs;
    int spatial_csf_filter;
    int wavelet_csf_filter;
    char *spatial_csf_filter_type;
    char *wavelet_csf_filter_type;
    spat_fil_coeff_dtype csf_factors[4][4];
    uint16_t csf_interim_rnd[4][4];
    uint8_t csf_interim_shift[4][4];

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
    int vif_levels;
    int adm_levels;
    int needed_dwt_levels;
    int needed_full_dwt_levels;
    int ssim_levels;
    int ms_ssim_levels;
    int strred_levels;
    double norm_view_dist;
    int ref_display_height;
    int i_process_ref_width;
    int i_process_ref_height;
    int i_process_dist_width;
    int i_process_dist_height;

    // VIF extra variables
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    uint32_t log_18[262144];
    uint32_t log_22[4194304];

    // ADM extra variables
    double adm_enhn_gain_limit;
    int adm_csf_mode;
	int32_t adm_div_lookup[65537];

    VmafDictionary *feature_name_dict;

    ModuleFunqueState modules;
    ResizerState resize_module;
    strred_results strred_scores;
    MsSsimScore_int *score;

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
        .default_val.b = true,
    },
    {
        .name = "spatial_csf_filter",
        .alias = "spatial_csf_filter",
        .help = "Select number of taps to be used for spatial filter",
        .offset = offsetof(IntFunqueState, spatial_csf_filter),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = NADENAU_SPAT_5_TAP_FILTER,
        .min = NADENAU_SPAT_5_TAP_FILTER,
        .max = NGAN_21_TAP_FILTER,
    },
    {
        .name = "wavelet_csf_filter",
        .alias = "wave_filter",
        .help = "Select wavelet filter",
        .offset = offsetof(IntFunqueState, wavelet_csf_filter),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = NADENAU_WEIGHT_FILTER,
        .min = NADENAU_WEIGHT_FILTER,
        .max = LI_FILTER,
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
        .name = "ms_ssim_levels",
        .alias = "ms_ssiml",
        .help = "Number of DWT levels for MS_SSIM",
        .offset = offsetof(IntFunqueState, ms_ssim_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_MS_SSIM_LEVELS,
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

static int integer_alloc_dwt2buffers(i_dwt2buffers *dwt2out, int w, int h)
{
    dwt2out->width = (int) w;
    dwt2out->height = (int) h;
    dwt2out->stride = dwt2out->width * sizeof(dwt2_dtype);

    for(unsigned i = 0; i < 4; i++) {
        dwt2out->bands[i] = aligned_malloc(dwt2out->stride * dwt2out->height, 32);
        if(!dwt2out->bands[i])
            goto fail;
        memset(dwt2out->bands[i], 0, dwt2out->stride * dwt2out->height);
    }
    return 0;

fail:
    for(unsigned i = 0; i < 4; i++) {
        if(dwt2out->bands[i])
            aligned_free(dwt2out->bands[i]);
        dwt2out->bands[i] = NULL;
    }
    return -ENOMEM;
}
void integer_select_filter_type(IntFunqueState *s)
{
    if(s->enable_spatial_csf == 1) {
        if(s->spatial_csf_filter == 5)
            s->spatial_csf_filter_type = "nadenau_spat";
        else if(s->spatial_csf_filter == 21)
            s->spatial_csf_filter_type = "ngan_spat";
    } else {
        switch(s->wavelet_csf_filter) {
            case 1:
                s->wavelet_csf_filter_type = "nadenau_weight";
                break;

            case 2:
                s->wavelet_csf_filter_type = "watson";
                break;

            case 3:
                s->wavelet_csf_filter_type = "li";
                break;

            default:
                s->wavelet_csf_filter_type = "nadenau_weight";
                break;
        }
    }
}

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

    s->needed_dwt_levels =
        MAX5(s->vif_levels, s->adm_levels, s->ssim_levels, s->ms_ssim_levels, s->strred_levels);
    s->needed_full_dwt_levels = MAX(s->adm_levels, s->ssim_levels);

    int ref_process_width, ref_process_height, dist_process_width, dist_process_height,
        process_wh_div_factor;

    int last_w = w;
    int last_h = h;

    if(s->ms_ssim_levels != 0) {
#if ENABLE_PADDING
        int two_pow_level_m1 = pow(2, (s->needed_dwt_levels - 1));
        ref_process_width =
            (int) (((last_w + two_pow_level_m1) >> s->needed_dwt_levels) << s->needed_dwt_levels);
        ref_process_height =
            (int) (((last_h + two_pow_level_m1) >> s->needed_dwt_levels) << s->needed_dwt_levels);
        dist_process_width =
            (int) (((last_w + two_pow_level_m1) >> s->needed_dwt_levels) << s->needed_dwt_levels);
        dist_process_height =
            (int) (((last_h + two_pow_level_m1) >> s->needed_dwt_levels) << s->needed_dwt_levels);

#else  // Cropped width and height
        ref_process_width = (int) ((last_w >> s->needed_dwt_levels) << s->needed_dwt_levels);
        ref_process_height = (int) ((last_h >> s->needed_dwt_levels) << s->needed_dwt_levels);
        dist_process_width = (int) ((last_w >> s->needed_dwt_levels) << s->needed_dwt_levels);
        dist_process_height = (int) ((last_h >> s->needed_dwt_levels) << s->needed_dwt_levels);
#endif

        last_w = ref_process_width;
        last_h = ref_process_height;
    } else {
        ref_process_width = last_w;
        ref_process_height = last_h;
        dist_process_width = last_w;
        dist_process_height = last_h;
    }

    s->width_aligned_stride = ALIGN_CEIL(ref_process_width * sizeof(float));

    int bitdepth_factor = (bpc == 8 ? 1 : 2);
    if (s->enable_resize)
    {
        s->res_ref_pic.data[0] =
            aligned_malloc(s->width_aligned_stride * ref_process_height * bitdepth_factor, 32);
        if (!s->res_ref_pic.data[0])
            goto fail;
        s->res_dist_pic.data[0] =
            aligned_malloc(s->width_aligned_stride * dist_process_height * bitdepth_factor, 32);

        if (!s->res_dist_pic.data[0])
            goto fail;
    }

    /* This buffer is common along spatial and wavelet buffers*/
    s->filter_buffer = aligned_malloc(
        ALIGN_CEIL(ref_process_width * sizeof(spat_fil_output_dtype)) * ref_process_height, 32);
    if(!s->filter_buffer)
        goto fail;
    s->filter_buffer_stride = ref_process_width * sizeof(spat_fil_output_dtype);

#if ENABLE_PADDING
    s->pad_ref = aligned_malloc(s->width_aligned_stride * ref_process_height * bitdepth_factor, 32);
    if(!s->pad_ref)
        goto fail;
    memset(s->pad_ref, 0, s->width_aligned_stride * ref_process_height * bitdepth_factor);

    s->pad_dist =
        aligned_malloc(s->width_aligned_stride * dist_process_height * bitdepth_factor, 32);
    if(!s->pad_dist)
        goto fail;
    memset(s->pad_dist, 0, s->width_aligned_stride * dist_process_height * bitdepth_factor);
#endif

    integer_select_filter_type(s);

    if (s->enable_spatial_csf) {
        s->spat_tmp_buf =
            aligned_malloc(ALIGN_CEIL(ref_process_width * sizeof(spat_fil_inter_dtype)), 32);
        if(!s->spat_tmp_buf)
            goto fail;
        // memset(s->spat_tmp_buf, 0, ALIGN_CEIL(w * sizeof(spat_fil_inter_dtype)));
    } else {
        if(strcmp(s->wavelet_csf_filter_type, "nadenau_weight") == 0) {
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
        } else if(strcmp(s->wavelet_csf_filter_type, "watson") == 0) {
            for(int level = 0; level < 4; level++) {
                s->csf_factors[level][0] = i_watson_coeffs[level][0];
                s->csf_factors[level][1] = i_watson_coeffs[level][1];
                s->csf_factors[level][2] = i_watson_coeffs[level][2];
                s->csf_factors[level][3] = i_watson_coeffs[level][3];

                s->csf_interim_shift[level][0] = i_nadenau_weight_interim_shift[level][0];
                s->csf_interim_shift[level][1] = i_nadenau_weight_interim_shift[level][1];
                s->csf_interim_shift[level][2] = i_nadenau_weight_interim_shift[level][2];
                s->csf_interim_shift[level][3] = i_nadenau_weight_interim_shift[level][3];

                s->csf_interim_rnd[level][0] = 1 << (i_nadenau_weight_interim_shift[level][0] - 1);
                s->csf_interim_rnd[level][1] = 1 << (i_nadenau_weight_interim_shift[level][1] - 1);
                s->csf_interim_rnd[level][2] = 1 << (i_nadenau_weight_interim_shift[level][2] - 1);
                s->csf_interim_rnd[level][3] = 1 << (i_nadenau_weight_interim_shift[level][3] - 1);
            }
        } else if(strcmp(s->wavelet_csf_filter_type, "li") == 0) {
            for(int level = 0; level < 4; level++) {
                s->csf_factors[level][0] = i_li_coeffs[level][0];
                s->csf_factors[level][1] = i_li_coeffs[level][1];
                s->csf_factors[level][2] = i_li_coeffs[level][2];
                s->csf_factors[level][3] = i_li_coeffs[level][3];

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

    int err = 0;
    int tref_width, tref_height, tdist_width, tdist_height;

    for(int level = 0; level < s->needed_dwt_levels; level++) {
        // dwt output dimensions
        process_wh_div_factor = pow(2, (level + 1));
        tref_width = (ref_process_width + (process_wh_div_factor * 3 / 4)) / process_wh_div_factor;
        tref_height =
            (ref_process_height + (process_wh_div_factor * 3 / 4)) / process_wh_div_factor;
        tdist_width =
            (dist_process_width + (process_wh_div_factor * 3 / 4)) / process_wh_div_factor;
        tdist_height =
            (dist_process_height + (process_wh_div_factor * 3 / 4)) / process_wh_div_factor;

        err |= integer_alloc_dwt2buffers(&s->i_ref_dwt2out[level], tref_width, tref_height);
        err |= integer_alloc_dwt2buffers(&s->i_dist_dwt2out[level], tdist_width, tdist_height);

        s->i_prev_ref[level].bands[0] = NULL;
        s->i_prev_dist[level].bands[0] = NULL;

        for(int subband = 1; subband < 4; subband++) {
            s->i_prev_ref[level].bands[subband] =
                calloc(tref_width * tref_height, sizeof(dwt2_dtype));
            s->i_prev_dist[level].bands[subband] =
                calloc(tref_width * tref_height, sizeof(dwt2_dtype));
        }

        /* Last width and height is half of the current layer */
        last_w = (int) (last_w + 1) / 2;
        last_h = (int) (last_h + 1) / 2;
    }

    if(err)
        goto fail;

    s->modules.integer_funque_picture_copy = integer_funque_picture_copy;
    s->modules.integer_spatial_filter = integer_spatial_filter;
    s->modules.integer_funque_dwt2 = integer_funque_dwt2;
    // s->modules.integer_funque_dwt2_wavelet = integer_funque_dwt2_wavelet;
    s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque;
    s->modules.integer_compute_ms_ssim_funque = integer_compute_ms_ssim_funque_c;
    s->modules.integer_mean_2x2_ms_ssim_funque = integer_mean_2x2_ms_ssim_funque_c;
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
            if(s->spatial_csf_filter == 21)
                s->modules.integer_spatial_filter = integer_spatial_filter_neon;
            else
                s->modules.integer_spatial_filter = integer_spatial_5tap_filter_neon;
        }
        s->modules.integer_funque_dwt2_inplace_csf = integer_funque_dwt2_inplace_csf_neon;
        s->modules.integer_funque_dwt2 = integer_funque_dwt2_neon;
        s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_neon;
        s->modules.integer_compute_ms_ssim_funque = integer_compute_ms_ssim_funque_neon;
        s->modules.integer_mean_2x2_ms_ssim_funque = integer_mean_2x2_ms_ssim_funque_neon;
        s->modules.integer_funque_adm_decouple = integer_adm_decouple_neon;
        s->modules.integer_compute_vif_funque = integer_compute_vif_funque_neon;
        //Commenting this since C was performing better
        // s->resize_module.resizer_step = step_neon;
        // s->modules.integer_funque_image_mad = integer_funque_image_mad_neon;
        // s->modules.integer_adm_integralimg_numscore = integer_adm_integralimg_numscore_neon;

        s->modules.integer_compute_strred_funque = integer_compute_strred_funque_neon;
        s->modules.integer_copy_prev_frame_strred_funque = integer_copy_prev_frame_strred_funque_c;
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
        if (bpc == 8)
        {
            if(s->spatial_csf_filter == 21)
                s->modules.integer_spatial_filter = integer_spatial_filter_avx2;
            else
                s->modules.integer_spatial_filter = integer_spatial_5tap_filter_avx2;
        }
        s->modules.integer_funque_dwt2_inplace_csf = integer_funque_dwt2_inplace_csf_avx2;

        s->modules.integer_funque_dwt2 = integer_funque_dwt2_avx2;
        s->modules.integer_funque_vifdwt2_band0 = integer_funque_vifdwt2_band0_avx2;
        s->modules.integer_compute_vif_funque = integer_compute_vif_funque_avx2;
        s->modules.integer_compute_ssim_funque = integer_compute_ssim_funque_avx2;
        s->modules.integer_compute_ms_ssim_funque = integer_compute_ms_ssim_funque_avx2;
        s->modules.integer_mean_2x2_ms_ssim_funque = integer_mean_2x2_ms_ssim_funque_avx2;
        s->modules.integer_funque_adm_decouple = integer_adm_decouple_c;
        s->modules.integer_funque_image_mad = integer_funque_image_mad_avx2;
        s->resize_module.resizer_step = step_avx2;
        s->resize_module.hbd_resizer_step = hbd_step_avx2;

        s->modules.integer_compute_strred_funque = integer_compute_strred_funque_avx2;
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

    // funque_log_generate(s->log_18);
    div_lookup_generator(s->adm_div_lookup);
    strred_funque_log_generate(s->log_18);
    strred_funque_generate_log22(s->log_22);

    return 0;

fail:
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
#if ENABLE_PADDING
    if(s->pad_ref)
        aligned_free(s->pad_ref);
    if(s->pad_dist)
        aligned_free(s->pad_dist);
#endif
    if(s->filter_buffer)
        aligned_free(s->filter_buffer);
    if(s->spat_tmp_buf)
        aligned_free(s->spat_tmp_buf);

    for(int level = 0; level < s->needed_dwt_levels; level++) {
        for(unsigned i = 0; i < 4; i++) {
            if(s->i_ref_dwt2out[level].bands[i])
                aligned_free(s->i_ref_dwt2out[level].bands[i]);
            if(s->i_dist_dwt2out[level].bands[i])
                aligned_free(s->i_dist_dwt2out[level].bands[i]);
            if(s->i_prev_ref[level].bands[i])
                aligned_free(s->i_prev_ref[level].bands[i]);
            if(s->i_prev_dist[level].bands[i])
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

    if(s->ms_ssim_levels != 0) {
#if ENABLE_PADDING
        int two_pow_level_m1 = pow(2, (s->needed_dwt_levels - 1));
        s->i_process_ref_width = ((res_ref_pic->w[0] + two_pow_level_m1) >> s->needed_dwt_levels)
                                 << s->needed_dwt_levels;
        s->i_process_ref_height = ((res_ref_pic->h[0] + two_pow_level_m1) >> s->needed_dwt_levels)
                                  << s->needed_dwt_levels;
        s->i_process_dist_width = ((res_dist_pic->w[0] + two_pow_level_m1) >> s->needed_dwt_levels)
                                  << s->needed_dwt_levels;
        s->i_process_dist_height = ((res_dist_pic->h[0] + two_pow_level_m1) >> s->needed_dwt_levels)
                                   << s->needed_dwt_levels;
#else
        s->i_process_ref_width = (res_ref_pic->w[0] >> s->needed_dwt_levels)
                                 << s->needed_dwt_levels;
        s->i_process_ref_height = (res_ref_pic->h[0] >> s->needed_dwt_levels)
                                  << s->needed_dwt_levels;
        s->i_process_dist_width = (res_dist_pic->w[0] >> s->needed_dwt_levels)
                                  << s->needed_dwt_levels;
        s->i_process_dist_height = (res_dist_pic->h[0] >> s->needed_dwt_levels)
                                   << s->needed_dwt_levels;
#endif
    } else {
        s->i_process_ref_width = res_ref_pic->w[0];
        s->i_process_ref_height = res_ref_pic->h[0];
        s->i_process_dist_width = res_dist_pic->w[0];
        s->i_process_dist_height = res_dist_pic->h[0];
    }

#if ENABLE_PADDING
    int bitdepth_pow2 = (1 << res_ref_pic->bpc) - 1;

    int reflect_width, reflect_height;
    reflect_width = (s->i_process_ref_width - res_ref_pic->w[0]) / 2;
    reflect_height = (s->i_process_ref_height - res_ref_pic->h[0]) / 2;
    integer_reflect_pad_for_input(res_ref_pic->data[0], s->pad_ref, res_ref_pic->w[0],
                                  res_ref_pic->h[0], reflect_width, reflect_height,
                                  res_ref_pic->bpc);

    reflect_width = (s->i_process_dist_width - res_dist_pic->w[0]) / 2;
    reflect_height = (s->i_process_dist_height - res_dist_pic->h[0]) / 2;
    integer_reflect_pad_for_input(res_dist_pic->data[0], s->pad_dist, res_dist_pic->w[0],
                                  res_dist_pic->h[0], reflect_width, reflect_height,
                                  res_ref_pic->bpc);

    if(s->enable_spatial_csf) {
        s->modules.integer_spatial_filter(s->pad_ref, s->filter_buffer, s->filter_buffer_stride,
                                          s->i_process_ref_width, s->i_process_ref_height,
                                          (int) res_ref_pic->bpc, s->spat_tmp_buf,
                                          s->spatial_csf_filter_type);
        s->modules.integer_funque_dwt2(
            s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_process_ref_width,
            s->i_process_ref_width, s->i_process_ref_height, s->enable_spatial_csf, -1);
        s->modules.integer_spatial_filter(s->pad_dist, s->filter_buffer, s->filter_buffer_stride,
                                          s->i_process_dist_width, s->i_process_dist_height,
                                          (int) res_dist_pic->bpc, s->spat_tmp_buf,
                                          s->spatial_csf_filter_type);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride,
                                       &s->i_dist_dwt2out[0], s->i_process_dist_width,
                                       s->i_process_dist_width, s->i_process_dist_height,
                                       s->enable_spatial_csf, -1);
    } else {
        s->modules.integer_funque_picture_copy(s->pad_ref, s->filter_buffer,
                                               s->filter_buffer_stride, s->i_process_ref_width,
                                               s->i_process_ref_height, (int) res_ref_pic->bpc);
        s->modules.integer_funque_dwt2(
            s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_process_ref_width,
            s->i_process_ref_width, s->i_process_ref_height, s->enable_spatial_csf, 0);

        s->modules.integer_funque_picture_copy(s->pad_dist, s->filter_buffer,
                                               s->filter_buffer_stride, s->i_process_dist_width,
                                               s->i_process_dist_height, (int) res_dist_pic->bpc);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride,
                                       &s->i_dist_dwt2out[0], s->i_process_dist_width,
                                       s->i_process_dist_width, s->i_process_dist_height,
                                       s->enable_spatial_csf, 0);
    }
#else
    int bitdepth_pow2 = (1 << res_ref_pic->bpc) - 1;

    if (s->enable_spatial_csf) {
        s->modules.integer_spatial_filter(res_ref_pic->data[0], s->filter_buffer,
                                          s->filter_buffer_stride, s->i_process_ref_width,
                                          s->i_process_ref_height, (int) res_ref_pic->bpc,
                                          s->spat_tmp_buf, s->spatial_csf_filter_type);
        s->modules.integer_funque_dwt2(
            s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_process_ref_width,
            s->i_process_ref_width, s->i_process_ref_height, s->enable_spatial_csf, -1);
        s->modules.integer_spatial_filter(res_dist_pic->data[0], s->filter_buffer,
                                          s->filter_buffer_stride, s->i_process_dist_width,
                                          s->i_process_dist_height, (int) res_dist_pic->bpc,
                                          s->spat_tmp_buf, s->spatial_csf_filter_type);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride,
                                       &s->i_dist_dwt2out[0], s->i_process_dist_width,
                                       s->i_process_dist_width, s->i_process_dist_height,
                                       s->enable_spatial_csf, -1);
    } else {
        s->modules.integer_funque_picture_copy(res_ref_pic->data[0], s->filter_buffer,
                                               s->filter_buffer_stride, s->i_process_ref_width,
                                               s->i_process_ref_height, (int) res_ref_pic->bpc);
        s->modules.integer_funque_dwt2(
            s->filter_buffer, s->filter_buffer_stride, &s->i_ref_dwt2out[0], s->i_process_ref_width,
            s->i_process_ref_width, s->i_process_ref_height, s->enable_spatial_csf, 0);

        s->modules.integer_funque_picture_copy(res_dist_pic->data[0], s->filter_buffer,
                                               s->filter_buffer_stride, s->i_process_dist_width,
                                               s->i_process_dist_height, (int) res_dist_pic->bpc);
        s->modules.integer_funque_dwt2(s->filter_buffer, s->filter_buffer_stride,
                                       &s->i_dist_dwt2out[0], s->i_process_dist_width,
                                       s->i_process_dist_width, s->i_process_dist_height,
                                       s->enable_spatial_csf, 0);
    }
#endif

    double ssim_score[MAX_LEVELS];
    MsSsimScore_int ms_ssim_score[MAX_LEVELS];
    // s->score = &ms_ssim_score;
    s->score = ms_ssim_score;
    double adm_score[MAX_LEVELS], adm_score_num[MAX_LEVELS], adm_score_den[MAX_LEVELS];
    double vif_score[MAX_LEVELS], vif_score_num[MAX_LEVELS], vif_score_den[MAX_LEVELS];

    int32_t *var_x_cum = (int32_t *) calloc(res_ref_pic->w[0] * res_ref_pic->h[0], sizeof(int32_t));
    int32_t *var_y_cum = (int32_t *) calloc(res_ref_pic->w[0] * res_ref_pic->h[0], sizeof(int32_t));
    int32_t *cov_xy_cum =
        (int32_t *) calloc(res_ref_pic->w[0] * res_ref_pic->h[0], sizeof(int32_t));

    ms_ssim_score[0].var_x_cum = &var_x_cum;
    ms_ssim_score[0].var_y_cum = &var_y_cum;
    ms_ssim_score[0].cov_xy_cum = &cov_xy_cum;

    double adm_den = 0.0;
    double adm_num = 0.0;

    double vif_den = 0.0;
    double vif_num = 0.0;

    int16_t spatfilter_shifts = 2 * SPAT_FILTER_COEFF_SHIFT - SPAT_FILTER_INTER_SHIFT - SPAT_FILTER_OUT_SHIFT - (res_ref_pic->bpc - 8);
    int16_t dwt_shifts = 2 * DWT2_COEFF_UPSHIFT - DWT2_INTER_SHIFT - DWT2_OUT_SHIFT;
    float pending_div_factor = (1 << ( spatfilter_shifts + dwt_shifts)) * bitdepth_pow2;

    s->strred_scores.spat_vals_cumsum = 0;
    s->strred_scores.temp_vals_cumsum = 0;
    s->strred_scores.spat_temp_vals_cumsum = 0;

    for(int level = 0; level < s->needed_dwt_levels;
        level++)  // For ST-RRED Debugging level set to 0
    {
        if(level + 1 < s->needed_dwt_levels) {
            if(level + 1 > s->needed_full_dwt_levels - 1) {
                // from here on out we only need approx band for VIF
                integer_funque_vifdwt2_band0(
                    s->i_ref_dwt2out[level].bands[0], s->i_ref_dwt2out[level + 1].bands[0],
                    ((s->i_ref_dwt2out[level + 1].stride + 1) / 2), s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height);
            } else {
                // compute full DWT if either SSIM or ADM need it for this level
                s->modules.integer_funque_dwt2(s->i_ref_dwt2out[level].bands[0],
                                    s->i_ref_dwt2out[level].width * sizeof(dwt2_dtype),
                                    &s->i_ref_dwt2out[level + 1],
                                    s->i_ref_dwt2out[level + 1].width * sizeof(dwt2_dtype),
                                    s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height,
                                    s->enable_spatial_csf, level + 1);
                s->modules.integer_funque_dwt2(s->i_dist_dwt2out[level].bands[0],
                                    s->i_dist_dwt2out[level].width * sizeof(dwt2_dtype),
                                    &s->i_dist_dwt2out[level + 1],
                                    s->i_dist_dwt2out[level + 1].width * sizeof(dwt2_dtype),
                                    s->i_dist_dwt2out[level].width, s->i_dist_dwt2out[level].height,
                                    s->enable_spatial_csf, level + 1);
            }
        }

        if(!s->enable_spatial_csf) {
            if(level < s->adm_levels || level < s->ssim_levels) {
                // we need full CSF on all bands
                s->modules.integer_funque_dwt2_inplace_csf(&s->i_ref_dwt2out[level], s->csf_factors[level], 0,
                                                3, s->csf_interim_rnd[level],
                                                s->csf_interim_shift[level], level);
                s->modules.integer_funque_dwt2_inplace_csf(&s->i_dist_dwt2out[level], s->csf_factors[level], 0,
                                                3, s->csf_interim_rnd[level],
                                                s->csf_interim_shift[level], level);
            } else {
                // we only need CSF on approx band
                s->modules.integer_funque_dwt2_inplace_csf(&s->i_ref_dwt2out[level], s->csf_factors[level], 0,
                                                0, s->csf_interim_rnd[level],
                                                s->csf_interim_shift[level], level);
                s->modules.integer_funque_dwt2_inplace_csf(&s->i_dist_dwt2out[level], s->csf_factors[level], 0,
                                                0, s->csf_interim_rnd[level],
                                                s->csf_interim_shift[level], level);
            }
        }

        // TODO: Need to modify for crop width and height
        if((s->adm_levels != 0) && (level <= s->adm_levels - 1)) {
            err = integer_compute_adm_funque(
                s->modules, s->i_ref_dwt2out[level], s->i_dist_dwt2out[level], &adm_score[level],
                &adm_score_num[level], &adm_score_den[level], s->i_ref_dwt2out[level].width,
                s->i_ref_dwt2out[level].height, ADM_BORDER_FACTOR, s->adm_div_lookup);

            float adm_pending_div = pending_div_factor;
            if(!s->enable_spatial_csf)
                adm_pending_div = (1 << (i_nadenau_pending_div_factors[level][1])) * bitdepth_pow2;

            adm_num += adm_score_num[level] / adm_pending_div;
            adm_den += adm_score_den[level] / adm_pending_div;

            if(err)
                return err;
        }

        if((s->ms_ssim_levels != 0) && (level < s->ms_ssim_levels)) {
            err = s->modules.integer_compute_ms_ssim_funque(
                &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &ms_ssim_score[level], 1, 0.01,
                0.03, pending_div_factor, s->adm_div_lookup, (level + 1),
                (int) (s->enable_spatial_csf == false));

            err = s->modules.integer_mean_2x2_ms_ssim_funque(var_x_cum, var_y_cum, cov_xy_cum,
                                                             s->i_ref_dwt2out[level].width,
                                                             s->i_ref_dwt2out[level].height, level);

            if(level != s->ms_ssim_levels - 1) {
                ms_ssim_score[level + 1].var_x_cum = ms_ssim_score[level].var_x_cum;
                ms_ssim_score[level + 1].var_y_cum = ms_ssim_score[level].var_y_cum;
                ms_ssim_score[level + 1].cov_xy_cum = ms_ssim_score[level].cov_xy_cum;
            }
        }
        if((s->ssim_levels != 0) && (level < s->ssim_levels)) {
            int16_t ssim_pending_div = 0;
            float k1 = 0.01;
            float k2 = 0.03;
            if(s->enable_spatial_csf) {
                ssim_pending_div =
                    ((1 << (spatfilter_shifts + dwt_shifts)) * bitdepth_pow2) >> level;
            } else {
                ssim_pending_div = (1 << i_nadenau_pending_div_factors[level][0]) * 255;
                k2 = k2 * (1 << (i_nadenau_pending_div_factors[level][1] -
                                 i_nadenau_pending_div_factors[level][0]));
            }
            err = s->modules.integer_compute_ssim_funque(
                &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &ssim_score[level], 1, k1, k2,
                ssim_pending_div, s->adm_div_lookup);
        }

        if(err)
            return err;

        if((s->vif_levels != 0) && (level <= s->vif_levels - 1)) {
            int16_t vif_pending_div = 0;
            if(s->enable_spatial_csf) {
                vif_pending_div = (1 << (spatfilter_shifts + dwt_shifts - level)) * bitdepth_pow2;
            } else {
                vif_pending_div = (1 << (i_nadenau_pending_div_factors[level][0])) * bitdepth_pow2;
            }
#if USE_DYNAMIC_SIGMA_NSQ
            err = s->modules.integer_compute_vif_funque(
                s->i_ref_dwt2out[level].bands[0], s->i_dist_dwt2out[level].bands[0],
                s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height, &vif_score[level],
                &vif_score_num[level], &vif_score_den[level], 9, 1, (double) 5.0, vif_pending_div,
                s->log_18, 0);
#else
            err = s->modules.integer_compute_vif_funque(
                s->i_ref_dwt2out[level].bands[0], s->i_dist_dwt2out[level].bands[0],
                s->i_ref_dwt2out[level].width, s->i_ref_dwt2out[level].height, &vif_score[level],
                &vif_score_num[level], &vif_score_den[level], 9, 1, (double) 5.0, vif_pending_div,
                s->log_18);
#endif
            vif_num += vif_score_num[level];
            vif_den += vif_score_den[level];

            if(err)
                return err;
        }

        if((s->strred_levels != 0) && (level <= s->strred_levels - 1)) {
            int32_t strred_pending_div = spatfilter_shifts + dwt_shifts - level;

            if(index == 0) {
                err |= s->modules.integer_copy_prev_frame_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height);
            } else {
                err |= s->modules.integer_compute_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height, &s->strred_scores, BLOCK_SIZE, level, s->log_18,
                    s->log_22, strred_pending_div, (double) 0.1, s->enable_spatial_csf);

                err |= s->modules.integer_copy_prev_frame_strred_funque(
                    &s->i_ref_dwt2out[level], &s->i_dist_dwt2out[level], &s->i_prev_ref[level],
                    &s->i_prev_dist[level], s->i_ref_dwt2out[level].width,
                    s->i_ref_dwt2out[level].height);
            }
            if(err)
                return err;
        }
    }

    if(s->ms_ssim_levels != 0) {
        err |= integer_compute_ms_ssim_mean_scales(ms_ssim_score, s->ssim_levels);
    }

    if(s->vif_levels > 0) {
        double vif = vif_den > 0 ? vif_num / vif_den : 1.0;

        err |=
            vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                    "FUNQUE_integer_feature_vif_score", vif, index);

        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "FUNQUE_integer_feature_vif_scale0_score",
                                                       vif_score[0], index);

        if(s->vif_levels > 1) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "FUNQUE_integer_feature_vif_scale1_score",
                vif_score[1], index);

            if(s->vif_levels > 2) {
                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_vif_scale2_score", vif_score[2], index);

                if(s->vif_levels > 3) {
                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_vif_scale3_score", vif_score[3], index);
                }
            }
        }
    }

    if(s->adm_levels > 0) {
        double adm = adm_den > 0 ? adm_num / adm_den : 1.0;
        err |=
            vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                    "FUNQUE_integer_feature_adm_score", adm, index);

        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "FUNQUE_integer_feature_adm_scale0_score",
                                                       adm_score[0], index);
        if(s->adm_levels > 1) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "FUNQUE_integer_feature_adm_scale1_score",
                adm_score[1], index);

            if(s->adm_levels > 2) {
                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_adm_scale2_score", adm_score[2], index);

                if(s->adm_levels > 3) {
                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_adm_scale3_score", adm_score[3], index);
                }
            }
        }
    }

    if(s->ssim_levels > 0) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "FUNQUE_integer_feature_ssim_scale0_score",
                                                       ssim_score[0], index);

        if(s->ssim_levels > 1) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "FUNQUE_integer_feature_ssim_scale1_score",
                ssim_score[1], index);

            if(s->ssim_levels > 2) {
                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_ssim_scale2_score", ssim_score[2], index);

                if(s->ssim_levels > 3) {
                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_ssim_scale3_score", ssim_score[3], index);
                }
            }
        }
    }

    if(s->strred_levels > 0) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "FUNQUE_integer_feature_strred_scale0_score",
                                                       s->strred_scores.strred_vals[0], index);
        if(s->strred_levels > 1) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "FUNQUE_integer_feature_strred_scale1_score", s->strred_scores.strred_vals[1],
                index);

            if(s->strred_levels > 2) {
                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_strred_scale2_score", s->strred_scores.strred_vals[2],
                    index);

                if(s->strred_levels > 3) {
                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_strred_scale3_score",
                        s->strred_scores.strred_vals[3], index);
                }
            }
        }
    }

    if(s->ms_ssim_levels > 0) {
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "FUNQUE_integer_feature_ms_ssim_mean_scale0_score", s->score[0].ms_ssim_mean, index);

        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "FUNQUE_integer_feature_ms_ssim_cov_scale0_score", s->score[0].ms_ssim_cov, index);

        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "FUNQUE_integer_feature_ms_ssim_mink3_scale0_score", s->score[0].ms_ssim_mink3, index);

        if(s->ms_ssim_levels > 1) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "FUNQUE_integer_feature_ms_ssim_mean_scale1_score", s->score[1].ms_ssim_mean,
                index);

            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "FUNQUE_integer_feature_ms_ssim_cov_scale1_score", s->score[1].ms_ssim_cov, index);

            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "FUNQUE_integer_feature_ms_ssim_mink3_scale1_score", s->score[1].ms_ssim_mink3,
                index);

            if(s->ms_ssim_levels > 2) {
                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_ms_ssim_mean_scale2_score", s->score[2].ms_ssim_mean,
                    index);

                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_ms_ssim_cov_scale2_score", s->score[2].ms_ssim_cov,
                    index);

                err |= vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict,
                    "FUNQUE_integer_feature_ms_ssim_mink3_scale2_score", s->score[2].ms_ssim_mink3,
                    index);

                if(s->ms_ssim_levels > 3) {
                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_ms_ssim_mean_scale3_score",
                        s->score[3].ms_ssim_mean, index);

                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_ms_ssim_cov_scale3_score", s->score[3].ms_ssim_cov,
                        index);

                    err |= vmaf_feature_collector_append_with_dict(
                        feature_collector, s->feature_name_dict,
                        "FUNQUE_integer_feature_ms_ssim_mink3_scale3_score",
                        s->score[3].ms_ssim_mink3, index);
                }
            }
        }
    }

    free(var_x_cum);
    free(var_y_cum);
    free(cov_xy_cum);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    IntFunqueState *s = fex->priv;
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
    if(s->filter_buffer)
        aligned_free(s->filter_buffer);
    if(s->spat_tmp_buf)
        aligned_free(s->spat_tmp_buf);

    for(int level = 0; level < 4; level++) {
        for(unsigned i = 0; i < 4; i++) {
            if(s->i_ref_dwt2out[level].bands[i])
                aligned_free(s->i_ref_dwt2out[level].bands[i]);
            if(s->i_dist_dwt2out[level].bands[i])
                aligned_free(s->i_dist_dwt2out[level].bands[i]);
            if(s->i_prev_ref[level].bands[i])
                aligned_free(s->i_prev_ref[level].bands[i]);
            if(s->i_prev_dist[level].bands[i])
                aligned_free(s->i_prev_dist[level].bands[i]);
        }
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"FUNQUE_integer_feature_vif_score",
                                          "FUNQUE_integer_feature_vif_scale0_score",
                                          "FUNQUE_integer_feature_vif_scale1_score",
                                          "FUNQUE_integer_feature_vif_scale2_score",
                                          "FUNQUE_integer_feature_vif_scale3_score",

                                          "FUNQUE_integer_feature_adm_score",
                                          "FUNQUE_integer_feature_adm_scale0_score",
                                          "FUNQUE_integer_feature_adm_scale1_score",
                                          "FUNQUE_integer_feature_adm_scale2_score",
                                          "FUNQUE_integer_feature_adm_scale3_score",

                                          "FUNQUE_integer_feature_ssim_scale0_score",
                                          "FUNQUE_integer_feature_ssim_scale1_score",
                                          "FUNQUE_integer_feature_ssim_scale2_score",
                                          "FUNQUE_integer_feature_ssim_scale3_score",

                                          "FUNQUE_integer_feature_strred_scale0_score",
                                          "FUNQUE_integer_feature_strred_scale1_score",
                                          "FUNQUE_integer_feature_strred_scale2_score",
                                          "FUNQUE_integer_feature_strred_scale3_score",

                                          "FUNQUE_integer_feature_ms_ssim_mean_scale0_score",
                                          "FUNQUE_integer_feature_ms_ssim_mean_scale1_score",
                                          "FUNQUE_integer_feature_ms_ssim_mean_scale2_score",
                                          "FUNQUE_integer_feature_ms_ssim_mean_scale3_score",
                                          "FUNQUE_integer_feature_ms_ssim_cov_scale0_score",
                                          "FUNQUE_integer_feature_ms_ssim_cov_scale1_score",
                                          "FUNQUE_integer_feature_ms_ssim_cov_scale2_score",
                                          "FUNQUE_integer_feature_ms_ssim_cov_scale3_score",
                                          "FUNQUE_integer_feature_ms_ssim_mink3_scale0_score",
                                          "FUNQUE_integer_feature_ms_ssim_mink3_scale1_score",
                                          "FUNQUE_integer_feature_ms_ssim_mink3_scale2_score",
                                          "FUNQUE_integer_feature_ms_ssim_mink3_scale3_score",

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