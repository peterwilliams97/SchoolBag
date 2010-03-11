#ifndef FACE_COMMON_H
#define FACE_COMMON_H
/*
 *  face_common.h
 *  FaceTracker
 *
 *  Created by peter on 4/03/10.
 */

#ifdef NOT_MAC_APP
 #include "cv.h"
 #include "highgui.h"
#else
 #include <OpenCV/OpenCV.h>
#endif


struct PwPoint {
    int x, y;
    PwPoint(): x(0), y(0) {}
    PwPoint(int x_, int y_): x(x_), y(y_) {}
};

struct PwRect {
    int x, y, width, height;
    PwRect(): x(0), y(0), width(0), height(0) {}
    PwRect(int x_, int y_, int w_, int h_): x(x_), y(y_), width(w_), height(h_) {}
};

#endif // #ifndef FACE_COMMON_H