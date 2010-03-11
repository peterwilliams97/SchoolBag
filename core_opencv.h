/*
 *  core_opencv.h
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */
 
#include "face_common.h"
  
PwRect CvRectToPwRect(CvRect rect);

IplImage*  rotateImage(const IplImage* image, double angle, PwPoint centerIn);
IplImage*  resizeImage(const IplImage* image, int x_pels, int y_pels);
IplImage*  cropImage(const IplImage* image, PwRect rect);
IplImage*  scaleImageWH(IplImage* image, int max_width, int max_height);

/*
 * Return point that a rotation of 'angle' around 'centerIn' would move to 'pt'
 */
CvPoint getUnrotatedPoint(PwPoint centerIn, double angle, PwPoint pt);
