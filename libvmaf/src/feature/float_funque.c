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

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "funque_vif.h"
#include "funque_vif_options.h"
#include "funque_adm.h"
#include "funque_adm_options.h"
#include "funque_motion.h"
#include "funque_motion_tools.h"
#include "picture_copy.h"
#include "funque_filters.h"


typedef struct FunqueState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    
    float *filter_ref;
    float *filter_dist;
    dwt2buffers ref_dwt2out;
    dwt2buffers dist_dwt2out;

    //VIF extra variables
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    
    //ADM extra variables
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode;
    
    //motion score extra variables
    float *tmp;
    float *blur[3];
    unsigned index;
    double score;
    bool motion_force_zero;

    //SSIM extra variables
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    double max_db;

    VmafDictionary *feature_name_dict;
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
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FunqueState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
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
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(FunqueState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdf",
        .help = "reference display height in pixels",
        .offset = offsetof(FunqueState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(FunqueState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },

    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FunqueState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(FunqueState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },

    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(FunqueState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(FunqueState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(FunqueState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },

    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    FunqueState *s = fex->priv;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto fail;

    s->filter_ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->filter_ref) goto fail;
    s->filter_dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->filter_dist) goto fail;
    for(unsigned i=0; i<4; i++)
    {
        s->ref_dwt2out.bands[i] = aligned_malloc(s->float_stride * h/4, 32);
        if (!s->ref_dwt2out.bands[i]) goto fail;
        s->ref_dwt2out.width[i] = (int) w/2;
        s->ref_dwt2out.height[i] = (int) h/2;
        s->dist_dwt2out.bands[i] = aligned_malloc(s->float_stride * h/4, 32);
        if (!s->dist_dwt2out.bands[i]) goto fail;
        s->dist_dwt2out.width[i] = (int) w/2;
        s->dist_dwt2out.height[i] = (int) h/2;
    }

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    if (s->filter_ref) aligned_free(s->filter_ref);
    if (s->filter_dist) aligned_free(s->filter_dist);
    for(unsigned i=0; i<4; i++)
    {
        if (s->ref_dwt2out.bands[i]) aligned_free(s->ref_dwt2out.bands[i]);
        if (s->dist_dwt2out.bands[i]) aligned_free(s->dist_dwt2out.bands[i]);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    FunqueState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc);
    float *tmp_filter = aligned_malloc(ALIGN_CEIL(ref_pic->w[0] * ref_pic->h[0] * sizeof(float)), MAX_ALIGN);
    spatial_filter(s->ref, tmp_filter, s->float_stride, -128, ref_pic->w[0], ref_pic->h[0]);
    funque_dwt2(tmp_filter, &s->ref_dwt2out, s->float_stride, ref_pic->w[0], ref_pic->h[0]);
    spatial_filter(s->dist, tmp_filter, s->float_stride, -128, ref_pic->w[0], ref_pic->h[0]);
    funque_dwt2(tmp_filter, &s->dist_dwt2out, s->float_stride, ref_pic->w[0], ref_pic->h[0]);
    double score, score_num, score_den;
    double scores[8];
    // TODO: update to funque VIF
    err = compute_vif(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride,
                      &score, &score_num, &score_den, scores,
                      s->vif_enhn_gain_limit,
                      s->vif_kernelscale);
    if (err) return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_feature_vif_scale0_score",
            scores[0] / scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_feature_vif_scale1_score",
            scores[2] / scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_feature_vif_scale2_score",
            scores[4] / scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_feature_vif_scale3_score",
            scores[6] / scores[7], index);
    //add adm and it's scores
    //add motion score and it's score
    //add ssim and it's scores

    if (!s->debug) return err;
    //Update the below for VIF
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale0", scores[0], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale0", scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale1", scores[2], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale1", scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale2", scores[4], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale2", scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale3", scores[6], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale3", scores[7], index);
    
    //add adm and it's scores
    //add motion score and it's score
    //add ssim and it's scores
    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    FunqueState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "FUNQUE_feature_vif_scale0_score", "FUNQUE_feature_vif_scale1_score",
    "FUNQUE_feature_vif_scale2_score", "FUNQUE_feature_vif_scale3_score",
    "FUNQUE_vif", "FUNQUE_vif_num", "FUNQUE_vif_den", "FUNQUE_vif_num_scale0", "FUNQUE_vif_den_scale0",
    "FUNQUE_vif_num_scale1", "FUNQUE_vif_den_scale1", "FUNQUE_vif_num_scale2", "FUNQUE_vif_den_scale2",
    "FUNQUE_vif_num_scale3", "FUNQUE_vif_den_scale3",
    
    "FUNQUE_feature_adm2_score", "FUNQUE_feature_adm_scale0_score",
    "FUNQUE_feature_adm_scale1_score", "FUNQUE_feature_adm_scale2_score",
    "FUNQUE_feature_adm_scale3_score", "FUNQUE_adm_num", "FUNQUE_adm_den", "FUNQUE_adm_scale0",
    "FUNQUE_adm_num_scale0", "FUNQUE_adm_den_scale0", "FUNQUE_adm_num_scale1", "FUNQUE_adm_den_scale1",
    "FUNQUE_adm_num_scale2", "FUNQUE_adm_den_scale2", "FUNQUE_adm_num_scale3", "FUNQUE_adm_den_scale3",
    
    "FUNQUE_feature_motion_score", "FUNQUE_feature_motion2_score",
    "FUNQUE_feature_motion2_score",

    "FUNQUE_float_ssim",

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
