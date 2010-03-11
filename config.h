/*
 *  config.h
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */

#define MAC_APP                 1
#define TEST_MANY_SETTINGS      1
#define SORT_AND_SHOW           0
#define HARDWIRE_HAAR_SETTINGS  1
#define TEST_NO_CROP            0
#define ADAPTIVE_FACE_SEARCH    1
#define ADAPTIVE_RECURSIVE      0
#define ADAPTIVE_NUM_STEPS      61      /* 21 default */
#define HAAR_SCALE_FACTOR       1.1    /* 1.1 default */
#define DRAW_FACES              0
#define DRAW_WAIT               2000
#define SHOW_ALL_RECTANGLES     1
#define VERBOSE                 1

#ifdef NOT_MAC_APP
 #undef MAC_APP
 #define MAC_APP 0
 #undef TEST_MANY_SETTINGS
 #define TEST_MANY_SETTINGS 0
 #undef DRAW_FACES 
 #define DRAW_FACES 0
#endif
