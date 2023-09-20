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

// #include "config.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "funque_global_options.h"
#include "funque_filters.h"
#include "funque_vif.h"
#include "funque_vif_options.h"
#include "funque_adm.h"
#include "funque_adm_options.h"
#include "funque_ssim_options.h"
//#include "funque_motion.h"
#include "funque_picture_copy.h"
#include "funque_ssim.h"
#include "resizer.h"

typedef struct FunqueState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    
    VmafPicture res_ref_pic;
    VmafPicture res_dist_pic;

    size_t float_dwt2_stride;
    float *spat_filter;
    float csf_factors[4][4];
    dwt2buffers ref_dwt2out[4];
    dwt2buffers dist_dwt2out[4];

    // funque configurable parameters
    bool enable_resize;
    bool enable_spatial_csf;
    int vif_levels;
    int adm_levels;
    int needed_dwt_levels;
    int needed_full_dwt_levels;
    int ssim_levels;
    double norm_view_dist;
    int ref_display_height;

    // VIF extra variables
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    
    //ADM extra variables
    double adm_enhn_gain_limit;
    int adm_csf_mode;

    VmafDictionary *feature_name_dict;
    ResizerState resize_module;
} FunqueState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FunqueState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_resize",
        .alias = "rsz",
        .help = "Enable resize for funque",
        .offset = offsetof(FunqueState, enable_resize),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "enable_spatial_csf",
        .alias = "gcsf",
        .help = "enable the global CSF based on spatial filter",
        .offset = offsetof(FunqueState, enable_spatial_csf),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
{
        .name = "norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(FunqueState, norm_view_dist),
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
        .offset = offsetof(FunqueState, ref_display_height),
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
        .offset = offsetof(FunqueState, vif_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_VIF_LEVELS,
        .min = MIN_LEVELS,
        .max = MAX_LEVELS,
    },
    {
        .name = "ssim_levels",
        .alias = "ssiml",
        .help = "Number of DWT levels for SSIM",
        .offset = offsetof(FunqueState, ssim_levels),
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
        .offset = offsetof(FunqueState, vif_enhn_gain_limit),
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
        .offset = offsetof(FunqueState, vif_kernelscale),
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
        .offset = offsetof(FunqueState, adm_levels),
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
        .offset = offsetof(FunqueState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },

    { 0 }
};

static int alloc_dwt2buffers(dwt2buffers *dwt2out, int w, int h) {
    dwt2out->width = (int) (w+1)/2;
    dwt2out->height = (int) (h+1)/2;
    dwt2out->stride = ALIGN_CEIL(dwt2out->width * sizeof(float));

    for(unsigned i=0; i<4; i++)
    {
        dwt2out->bands[i] = aligned_malloc(dwt2out->stride * dwt2out->height, 32);
        if (!dwt2out->bands[i]) goto fail;

        dwt2out->bands[i] = aligned_malloc(dwt2out->stride * dwt2out->height, 32);
        if (!dwt2out->bands[i]) goto fail;
    }
    return 0;

    fail:
    for(unsigned i=0; i<4; i++)
    {
        if (dwt2out->bands[i]) aligned_free(dwt2out->bands[i]);
    }
    return -ENOMEM;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;

    FunqueState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    if (s->enable_resize)
    {
        w = (w+1)>>1;
        h = (h+1)>>1;
    }

    s->needed_dwt_levels = MAX(MAX(s->vif_levels, s->adm_levels), s->ssim_levels);
    s->needed_full_dwt_levels = MAX(s->adm_levels, s->ssim_levels);

    s->float_stride = ALIGN_CEIL(w * sizeof(float));

    if(s->enable_resize)
    {
        s->res_ref_pic.data[0] = aligned_malloc(s->float_stride * h, 32);
        if (!s->res_ref_pic.data[0])
            goto fail;
        s->res_dist_pic.data[0] = aligned_malloc(s->float_stride * h, 32);
        if (!s->res_dist_pic.data[0])
            goto fail;
    }

    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto fail;

    if (s->enable_spatial_csf) {
        s->spat_filter = aligned_malloc(s->float_stride * h, 32);
        if (!s->spat_filter)
            goto fail;
    } else {
        for(int level = 0; level < 4; level++) {
            s->csf_factors[level][0] = 1.0f / funque_dwt_quant_step(&funque_dwt_7_9_YCbCr_threshold[0], level, 0, s->norm_view_dist, s->ref_display_height);
            s->csf_factors[level][1] = 1.0f / funque_dwt_quant_step(&funque_dwt_7_9_YCbCr_threshold[0], level, 1, s->norm_view_dist, s->ref_display_height);
            s->csf_factors[level][2] = 1.0f / funque_dwt_quant_step(&funque_dwt_7_9_YCbCr_threshold[0], level, 2, s->norm_view_dist, s->ref_display_height);
            s->csf_factors[level][3] = s->csf_factors[level][1]; // same as horizontal
        }
    }

    int err = 0;

    int last_w = w;
    int last_h = h;

    for (int level = 0; level < s->needed_dwt_levels; level++) {
        err |= alloc_dwt2buffers(&s->ref_dwt2out[level], last_w, last_h);
        err |= alloc_dwt2buffers(&s->dist_dwt2out[level], last_w, last_h);
        last_w = s->ref_dwt2out[level].width;
        last_h = s->ref_dwt2out[level].height;
    }

    if (err) goto fail;

    s->resize_module.resizer_step = step;

    return 0;

fail:
    if (s->res_ref_pic.data[0]) aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0]) aligned_free(s->res_dist_pic.data[0]);
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    if (s->spat_filter) aligned_free(s->spat_filter);
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

// #define MIN(x, y) (((x) < (y)) ? (x) : (y))

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    FunqueState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    VmafPicture *res_ref_pic = &s->res_ref_pic;
    VmafPicture *res_dist_pic = &s->res_dist_pic;

    if(s->enable_resize)
    {
        res_ref_pic->bpc = ref_pic->bpc;
        res_ref_pic->h[0]   = ref_pic->h[0] / 2;
        res_ref_pic->w[0]   = ref_pic->w[0] / 2;
        res_ref_pic->stride[0] = ref_pic->stride[0] / 2;
        res_ref_pic->pix_fmt = ref_pic->pix_fmt;
        res_ref_pic->ref = ref_pic->ref;

        res_dist_pic->bpc = dist_pic->bpc;
        res_dist_pic->h[0]   = dist_pic->h[0] / 2;
        res_dist_pic->w[0]   = dist_pic->w[0] / 2;
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
            hbd_resize(s->resize_module ,(unsigned short *)dist_pic->data[0], (unsigned short *)res_dist_pic->data[0], dist_pic->w[0], dist_pic->h[0], res_dist_pic->w[0], res_dist_pic->h[0], dist_pic->bpc);
    }
    else
    {
        res_ref_pic = ref_pic;
        res_dist_pic = dist_pic;
    }
    
    funque_picture_copy(s->ref, s->float_stride, res_ref_pic, 0, ref_pic->bpc);
    funque_picture_copy(s->dist, s->float_stride, res_dist_pic, 0, dist_pic->bpc);

    int bitdepth_pow2 = (1 << res_ref_pic->bpc) - 1;

    normalize_bitdepth(s->ref, s->ref, bitdepth_pow2, s->float_stride, res_ref_pic->w[0], res_ref_pic->h[0]);
    normalize_bitdepth(s->dist, s->dist, bitdepth_pow2, s->float_stride, res_dist_pic->w[0], res_dist_pic->h[0]);

    if (s->enable_spatial_csf) {
        spatial_filter(s->ref, s->spat_filter, res_ref_pic->w[0], res_ref_pic->h[0]);
        funque_dwt2(s->spat_filter, &s->ref_dwt2out[0], res_ref_pic->w[0], res_ref_pic->h[0]);
        spatial_filter(s->dist, s->spat_filter, res_dist_pic->w[0], res_dist_pic->h[0]);
        funque_dwt2(s->spat_filter, &s->dist_dwt2out[0], res_dist_pic->w[0], res_dist_pic->h[0]);
    } else {
        funque_dwt2(s->ref, &s->ref_dwt2out[0], res_ref_pic->w[0], res_ref_pic->h[0]);
        funque_dwt2(s->dist, &s->dist_dwt2out[0], res_dist_pic->w[0], res_dist_pic->h[0]);
    }

    double ssim_score[MAX_LEVELS];
    double adm_score[MAX_LEVELS], adm_score_num[MAX_LEVELS], adm_score_den[MAX_LEVELS];
    double vif_score[MAX_LEVELS], vif_score_num[MAX_LEVELS], vif_score_den[MAX_LEVELS];

    double adm_den = 0.0;
    double adm_num = 0.0;

    double vif_den = 0.0;
    double vif_num = 0.0;

    for (int level = 0; level < s->needed_dwt_levels; level++) {
        // pre-compute the next level of DWT
        if (level+1 < s->needed_dwt_levels) {
            if (level+1 > s->needed_full_dwt_levels - 1) {
                // from here on out we only need approx band for VIF
                funque_vifdwt2_band0(s->ref_dwt2out[level].bands[0],  s->ref_dwt2out[level + 1].bands[0],  s->ref_dwt2out[level].stride, s->ref_dwt2out[level].width, s->ref_dwt2out[level].height);
            } else {
                // compute full DWT if either SSIM or ADM need it for this level
                funque_dwt2(s->ref_dwt2out[level].bands[0], &s->ref_dwt2out[level + 1], s->ref_dwt2out[level].width,
                            s->ref_dwt2out[level].height);
                funque_dwt2(s->dist_dwt2out[level].bands[0], &s->dist_dwt2out[level + 1],
                            s->dist_dwt2out[level].width, s->dist_dwt2out[level].height);
            }
        }

        if (!s->enable_spatial_csf) {
            if (level < s->adm_levels || level < s->ssim_levels) {
                // we need full CSF on all bands
                funque_dwt2_inplace_csf(&s->ref_dwt2out[level], s->csf_factors[level], 0, 3);
                funque_dwt2_inplace_csf(&s->dist_dwt2out[level], s->csf_factors[level], 0, 3);
            } else {
                // we only need CSF on approx band
                funque_dwt2_inplace_csf(&s->ref_dwt2out[level], s->csf_factors[level], 0, 0);
                funque_dwt2_inplace_csf(&s->dist_dwt2out[level], s->csf_factors[level], 0, 0);
            }
        }

        if (level <= s->adm_levels - 1) {
            err |= compute_adm_funque(s->ref_dwt2out[level], s->dist_dwt2out[level], &adm_score[level], &adm_score_num[level], &adm_score_den[level], ADM_BORDER_FACTOR);
            adm_num += adm_score_num[level];
            adm_den += adm_score_den[level];
        }

        if (level <= s->ssim_levels - 1) {
            err |= compute_ssim_funque(&s->ref_dwt2out[level], &s->dist_dwt2out[level], &ssim_score[level], 1, (float)0.01, (float)0.03);
        }

        if (level <= s->vif_levels - 1) {
            #if USE_DYNAMIC_SIGMA_NSQ
            err |= compute_vif_funque(s->ref_dwt2out[level].bands[0], s->dist_dwt2out[level].bands[0], s->ref_dwt2out[level].width, s->ref_dwt2out[level].height,
                                        &vif_score[level], &vif_score_num[level], &vif_score_den[level], VIF_WINDOW_SIZE, 1, (double)VIF_SIGMA_NSQ, level);
            #else
            err |= compute_vif_funque(s->ref_dwt2out[level].bands[0], s->dist_dwt2out[level].bands[0], s->ref_dwt2out[level].width, s->ref_dwt2out[level].height,
                                 &vif_score[level], &vif_score_num[level], &vif_score_den[level], VIF_WINDOW_SIZE, 1, (double)VIF_SIGMA_NSQ);
            #endif
            vif_num += vif_score_num[level];
            vif_den += vif_score_den[level];
        }

        if (err) return err;
    }

    double vif = vif_den > 0 ? vif_num / vif_den : 1.0;
    double adm = adm_den > 0 ? adm_num / adm_den : 1.0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_feature_vif_score",
                                                   vif, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_feature_vif_scale0_score",
                                                   vif_score[0], index);

    if (s->vif_levels > 1) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_feature_vif_scale1_score",
                                                       vif_score[1], index);

        if (s->vif_levels > 2) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                           s->feature_name_dict, "FUNQUE_feature_vif_scale2_score",
                                                           vif_score[2], index);

            if (s->vif_levels > 3) {
                err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                               s->feature_name_dict, "FUNQUE_feature_vif_scale3_score",
                                                               vif_score[3], index);
            }
        }
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_feature_adm_score",
                                                   adm, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict, "FUNQUE_feature_adm_scale0_score",
                                                   adm_score[0], index);
    if (s->adm_levels > 1) {

        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_feature_adm_scale1_score",
                                                       adm_score[1], index);

        if (s->adm_levels > 2) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                           s->feature_name_dict, "FUNQUE_feature_adm_scale2_score",
                                                           adm_score[2], index);

            if (s->adm_levels > 3) {
                err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                               s->feature_name_dict, "FUNQUE_feature_adm_scale3_score",
                                                               adm_score[3], index);
            }
        }
    }

    err |= vmaf_feature_collector_append(feature_collector, "FUNQUE_feature_ssim_scale0_score",
                                         ssim_score[0], index);

    if (s->ssim_levels > 1) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_feature_ssim_scale1_score",
                                                       ssim_score[1], index);

        if (s->ssim_levels > 2) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                           s->feature_name_dict, "FUNQUE_feature_ssim_scale2_score",
                                                           ssim_score[2], index);

            if (s->ssim_levels > 3) {
                err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                               s->feature_name_dict, "FUNQUE_feature_ssim_scale3_score",
                                                               ssim_score[3], index);
            }
        }
    }

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    FunqueState *s = fex->priv;
    if (s->res_ref_pic.data[0]) aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0]) aligned_free(s->res_dist_pic.data[0]);
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    if (s->spat_filter) aligned_free(s->spat_filter);

    for(int level = 0; level < s->needed_dwt_levels; level += 1) {
        for(unsigned i=0; i<4; i++)
        {
            if (s->ref_dwt2out[level].bands[i]) aligned_free(s->ref_dwt2out[level].bands[i]);
            if (s->dist_dwt2out[level].bands[i]) aligned_free(s->dist_dwt2out[level].bands[i]);
        }
    }

    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "FUNQUE_feature_vif_score",
    "FUNQUE_feature_vif_scale0_score", "FUNQUE_feature_vif_scale1_score",
    "FUNQUE_feature_vif_scale2_score", "FUNQUE_feature_vif_scale3_score",

    "FUNQUE_feature_adm_score",
    "FUNQUE_feature_adm_scale0_score","FUNQUE_feature_adm_scale1_score",
    "FUNQUE_feature_adm_scale2_score","FUNQUE_feature_adm_scale3_score",

    "FUNQUE_feature_ssim_scale0_score", "FUNQUE_feature_ssim_scale1_score",
    "FUNQUE_feature_ssim_scale2_score", "FUNQUE_feature_ssim_scale3_score",

    NULL
};

VmafFeatureExtractor vmaf_fex_float_funque = {
    .name = "float_funque",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(FunqueState),
    .provided_features = provided_features,
};