//
// Created by Cosmin Stejerean on 1/3/23.
//

#ifndef VMAF_FUNQUE_GLOBAL_OPTIONS_H
#define VMAF_FUNQUE_GLOBAL_OPTIONS_H

/* normalized viewing distance = viewing distance / ref display's physical height */
#define DEFAULT_NORM_VIEW_DIST (3.0)

/* reference display height in pixels */
#define DEFAULT_REF_DISPLAY_HEIGHT (1080)

#define MAX_LEVELS 4
#define MIN_LEVELS 0
#define DEFAULT_BANDS 4

typedef struct FrameBufLen {
    int buf_size[4][4];
    int total_buf_size;
} FrameBufLen;

#endif //VMAF_FUNQUE_GLOBAL_OPTIONS_H
