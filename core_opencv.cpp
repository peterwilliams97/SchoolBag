/*
 *  core_opencv.cpp
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */
#include <iostream>
#include "core_opencv.h"

using namespace std;;

PwRect CvRectToPwRect(CvRect rect) {
    return PwRect(rect.x, rect.y, rect.width, rect.height);
}

CvRect  PwRectToCvRect(PwRect rect) {
    return cvRect(rect.x, rect.y, rect.width, rect.height);
}

CvPoint PwPointToCvPoint(PwPoint point) {
    return cvPoint(point.x, point.y);
}

static void showMatrix(const CvMat* rot_mat) {
#if VERBOSE_HISTOGRAM
    cout << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 0, 0) << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 0, 1) << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 0, 2) << " | " << endl;
    cout << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 1, 0) << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 1, 1) << " | " << setprecision(4) << setw(7) << cvmGet(rot_mat, 1, 2) << " | " << endl;
#endif
}

IplImage*  rotateImage(const IplImage* image, double angle, PwPoint centerIn)   {
#if 0
    CvPoint px1 = cvPoint(0, image->height/2);
    CvPoint px2 = cvPoint(image->width, image->height/2);
    CvPoint py1 = cvPoint(image->width/2, 0);
    CvPoint py2 = cvPoint(image->width/2, image->height);
    cvLine(image, px1, px2, CV_RGB(255,255,255), 3, 8, 0);
    cvLine(image, py1, py2, CV_RGB(255,255,255), 3, 8, 0);
#endif    
    IplImage* dest_image = cvCloneImage(image);
    if (angle != 0.0) {
        cvZero(dest_image);
        dest_image->origin = image->origin;

        CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
        double scale = 1.0;

        // Compute rotation matrix
        //CvPoint2D32f center = cvPoint2D32f( image->width/2, image->height/2 );
        CvPoint2D32f center = cvPoint2D32f(centerIn.x, centerIn.y);
        cv2DRotationMatrix(center, angle, scale, rot_mat);
#if VERBOSE_HISTOGRAM        
        cout << "rotate by " << angle << " degrees around (" << center.x << ", " << center.y << ")" << endl;
#endif
        showMatrix(rot_mat);
        
        // Do the transformation
        cvWarpAffine(image, dest_image, rot_mat, CV_WARP_FILL_OUTLIERS, CV_RGB(0,0,0));
      //  cvReleaseImage(&image);
        cvReleaseMat(&rot_mat);
    }
    return dest_image;
}


IplImage*  resizeImage(const IplImage* image, int x_pels, int y_pels)   {
#if 0
    CvPoint px1 = cvPoint(0, image->height/2);
    CvPoint px2 = cvPoint(image->width, image->height/2);
    CvPoint py1 = cvPoint(image->width/2, 0);
    CvPoint py2 = cvPoint(image->width/2, image->height);
    CvPoint p1 = cvPoint(0,0);
    CvPoint p2 = cvPoint(image->width, image->height);
    cvLine(image, px1, px2, CV_RGB(255,255,0), 3, 8, 0);
    cvLine(image, py1, py2, CV_RGB(255,255,0), 3, 8, 0);
 //   cvRectangle(image, p1, p2, CV_RGB(255,255,0), 3, 8, 0);
#endif  
  
    IplImage* dest_image = cvCreateImage(cvSize (image->width + 2*x_pels, image->height + 2*y_pels), IPL_DEPTH_8U, 3);
    if (x_pels == 0 && y_pels == 0) {
        cvCopy(image, dest_image);
    }
    else {
        cvZero(dest_image);

        CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
        cvmSet(rot_mat, 0, 0, 1.0);
        cvmSet(rot_mat, 0, 1, 0.0);
        cvmSet(rot_mat, 1, 0, 0.0);
        cvmSet(rot_mat, 1, 1, 1.0);
        cvmSet(rot_mat, 0, 2, (double)x_pels);
        cvmSet(rot_mat, 1, 2, (double)x_pels);
        
        cout << "pad by " << x_pels << ", " << y_pels << endl;
        showMatrix(rot_mat);

        // Do the transformation
        cvWarpAffine(image, dest_image, rot_mat, CV_WARP_FILL_OUTLIERS, CV_RGB(0,0,0));
      //  cvReleaseImage(&image);
        cvReleaseMat(&rot_mat);
    }
    return dest_image;
}

#if 1
IplImage*  cropImage(const IplImage* image, PwRect rect)   {
#if 0
    CvPoint px1 = cvPoint(0, image->height/2);
    CvPoint px2 = cvPoint(image->width, image->height/2);
    CvPoint py1 = cvPoint(image->width/2, 0);
    CvPoint py2 = cvPoint(image->width/2, image->height);
    CvPoint p1 = cvPoint(0,0);
    CvPoint p2 = cvPoint(image->width, image->height);
    cvLine(image, px1, px2, CV_RGB(255,255,0), 3, 8, 0);
    cvLine(image, py1, py2, CV_RGB(255,255,0), 3, 8, 0);
 //   cvRectangle(image, p1, p2, CV_RGB(255,255,0), 3, 8, 0);
#endif  
    IplImage* dest_image = 0;
    bool same_size = (rect.x == 0 && rect.y == 0 && rect.width == image->width && rect.height == image->height);
    if (rect.width == 0 || rect.height == 0 || same_size) {
        dest_image = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 3);
        cvCopy(image, dest_image);
    }
    else {
        dest_image = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 3);
        cvZero(dest_image);

        CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
        cvmSet(rot_mat, 0, 0, 1.0);
        cvmSet(rot_mat, 0, 1, 0.0);
        cvmSet(rot_mat, 1, 0, 0.0);
        cvmSet(rot_mat, 1, 1, 1.0);
        cvmSet(rot_mat, 0, 2, -(double)rect.x);
        cvmSet(rot_mat, 1, 2, -(double)rect.y);
        
     //   cout << "crop to " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
     //   showMatrix(rot_mat);

        // Do the transformation
        cvWarpAffine(image, dest_image, rot_mat, CV_WARP_FILL_OUTLIERS, CV_RGB(0,0,0));
      //  cvReleaseImage(&image);
        cvReleaseMat(&rot_mat);
    }
    return dest_image;
}
#else
IplImage* cropImage(const IplImage* image, PwRect rect) {
    IplImage* dest_image  = cvCreateImage (cvSize(rect.width, rect.height), IPL_DEPTH_8U, 3);
    cvSetImageROI(image, rect); 
    cvResize (image, dest_image, CV_INTER_NN);
    cvReleaseImage(&image);
    return dest_image;
}
#endif

IplImage* scaleImageWH(IplImage* image, int max_width, int max_height) {
    double scale_x = (double)max_width/(double)image->width;
    double scale_y = (double)max_height/(double)image->height;
    IplImage* dest_image;
    if (scale_x >= 1.0 && scale_y >= 1.0) {
        dest_image = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 3);
        cvCopy(image, dest_image);
    }
    else {
        double scale = min(scale_x, scale_y);
        int width = cvRound(scale * (double)image->width);
        int height = cvRound(scale * (double)image->height);
        dest_image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
        cvZero(dest_image);

        CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
        cvmSet(rot_mat, 0, 0, scale);
        cvmSet(rot_mat, 0, 1, 0.0);
        cvmSet(rot_mat, 1, 0, 0.0);
        cvmSet(rot_mat, 1, 1, scale);
        cvmSet(rot_mat, 0, 2, 0.0/* -scale*(image->width - dest_image->width)/2 */);
        cvmSet(rot_mat, 1, 2, 0.0 /* -scale*(image->height - dest_image->height)/2 */);
        
     // Do the transformation
        cvWarpAffine(image, dest_image, rot_mat, CV_WARP_FILL_OUTLIERS, CV_RGB(0,0,0));
      //  cvReleaseImage(&image);
        cvReleaseMat(&rot_mat);
    }
    return dest_image;
}


CvPoint transform(const CvMat* mat, CvPoint p) {
    double A00 = cvmGet(mat, 0, 0);
    double A01 = cvmGet(mat, 0, 1);
    double A10 = cvmGet(mat, 1, 0);
    double A11 = cvmGet(mat, 1, 1);
    double a0  = cvmGet(mat, 0, 2);
    double a1  = cvmGet(mat, 1, 2);
    CvPoint p1;
    p1.x = cvRound((double)p.x*A00 + (double)p.y*A01 + (double)a0);
    p1.y = cvRound((double)p.x*A10 + (double)p.y*A11 + (double)a1);
    return p1;
}

// !@#$% wrong
CvMat* invertMat(const CvMat* mat) {
    double A00 = cvmGet(mat, 0, 0);
    double A01 = cvmGet(mat, 0, 1);
    double A10 = cvmGet(mat, 1, 0);
    double A11 = cvmGet(mat, 1, 1);
    double a0  = cvmGet(mat, 0, 2);
    double a1  = cvmGet(mat, 1, 2);
    double d = (A00*A11 - A01*A10);
    double B00 =  A11/d;
    double B01 = -A01/d;
    double B10 = -A10/d;
    double B11 =  A00/d;
    double b0 = (a0*B00 + a1*B01);
    double b1 = (a0*B10 + a1*B11);
    CvMat* inv_mat = cvCreateMat(2,3,CV_32FC1);
    cvmSet(inv_mat, 0, 0, B00);
    cvmSet(inv_mat, 0, 1, B01);
    cvmSet(inv_mat, 1, 0, B10);
    cvmSet(inv_mat, 1, 1, B11);
    cvmSet(inv_mat, 0, 2, b0);
    cvmSet(inv_mat, 1, 2, b1);
    return inv_mat;
}

/*
 * Return point that a rotation of 'angle' around 'centerIn' would move to 'pt'
 !@#$ buggy
 */
CvPoint getUnrotatedPoint(CvPoint centerIn, double angle, CvPoint pt) {
    CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
    CvPoint2D32f center = cvPoint2D32f(centerIn.x, centerIn.y);
    cv2DRotationMatrix(center, angle, 1.0, rot_mat);
    CvMat* inv_mat = invertMat(rot_mat);
    CvPoint inv_pt = transform(inv_mat, pt);
    CvPoint pt2 = transform(rot_mat, pt);
    cout << "----------------------------------------- original " << endl;
    showMatrix(rot_mat);
    cout << "----------------------------------------- inverse " << endl;
    showMatrix(inv_mat);
    
    cvReleaseMat(&rot_mat);
    cvReleaseMat(&inv_mat);
    return inv_pt;
}

static double target_delta = 1.0e-6;
static double invertMatTestOne(const CvMat* mat) {
    CvMat* inv_mat = invertMat(mat);
    CvMat* inv2_mat = invertMat(inv_mat);
    
    cout << "----------------------------------------- original " << endl;
    showMatrix(mat);
    cout << "----------------------------------------- inverted inverse " << endl;
    showMatrix(inv2_mat);
    cout << "----------------------------------------- inverse " << endl;
    showMatrix(inv_mat);

     double delta[6];
     double max_delta = 0.0;
     for (int row = 0; row < 2; row++) {
        for (int col= 0; col < 3; col++) {
            int k = 3*row + col;
            delta[k] = cvmGet(inv2_mat, row, col) - cvmGet(mat, row, col);
            if (max_delta < fabs(delta[k]))
                max_delta = fabs(delta[k]);
        }
    }
    cvReleaseMat(&inv_mat);
    cvReleaseMat(&inv2_mat);
    if (max_delta > target_delta) {
        cerr << max_delta << " > " << target_delta << " target delta exceeded in invertMatTest()" << endl;
        abort();
    }
    cout << "----------------------------------------- max_delta = " << max_delta << endl;
    return max_delta;
}

void invertMatTest() {
    CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
    double scale = 1.0;
    double angle = 0.0;
    int x = 0, y = 0;
    CvPoint2D32f center;
    
    center = cvPoint2D32f(0, 0);
    cv2DRotationMatrix(center, 45.0, 1.0, rot_mat);
    invertMatTestOne(rot_mat);
    for (scale = 0.1; scale <= 10.0; scale *= 1.01) {
        for (angle = 0.0; angle <= 360.0; angle += 1.0) {
            for (x = -10; x <= 10; x++) {
                for (y = -10; y <= 10; y++) {
                    center = cvPoint2D32f(x, y);
                    cv2DRotationMatrix(center, angle, scale, rot_mat);
                    invertMatTestOne(rot_mat);
                }
            }
        }
    }
    cvReleaseMat(&rot_mat);
}


CvMat* concatMat(const CvMat* mat1, const CvMat* mat2) {
    double A00 = cvmGet(mat2, 0, 0);
    double A01 = cvmGet(mat1, 0, 1);
    double A10 = cvmGet(mat1, 1, 0);
    double A11 = cvmGet(mat1, 1, 1);
    double a0  = cvmGet(mat1, 0, 2);
    double a1  = cvmGet(mat2, 1, 2);
    
    double B00 = cvmGet(mat2, 0, 0);
    double B01 = cvmGet(mat2, 0, 1);
    double B10 = cvmGet(mat2, 1, 0);
    double B11 = cvmGet(mat2, 1, 1);
    double b0  = cvmGet(mat2, 0, 2);
    double b1  = cvmGet(mat2, 1, 2);
    
    double C00 = A00*B00 + A10*B01;
    double C01 = A01*B00 + A11*B01;
    double C10 = A00*B10 + A10*B11;
    double C11 = A01*B10 + A11*B11;
    double c0  = a0*B00 + a1*B01 + b0;
    double c1  = a0*B10 + a1*B11 + b1;
    
    CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
    cvmSet(rot_mat, 0, 0, C00);
    cvmSet(rot_mat, 0, 1, C01);
    cvmSet(rot_mat, 1, 0, C10);
    cvmSet(rot_mat, 1, 1, C11);
    cvmSet(rot_mat, 0, 2, c0);
    cvmSet(rot_mat, 1, 2, c1);
    return rot_mat;
}

