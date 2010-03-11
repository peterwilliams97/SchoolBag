/*
 *  core_common.cpp
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */

#if 0

#include "core_common.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <list>
#include "config.h"
#include "face_util.h"
#include "face_common.h"
#include "face_io.h"
#include "face_draw.h"
#include "face_calc.h"
#include "face_results.h"
#include "cropped_frames.h"

#ifdef NOT_MAC_APP
#include "cdef/OD3FaceFinder.h"
#include "cdef/ImageFileReader.h"

using namespace img;
#endif

using namespace std;


#if DRAW_FACES 
 static const char * WINDOW_NAME  = "Face Tracker with Sub-Frames";
#endif

#if ADAPTIVE_FACE_SEARCH
const string test_type_name = "adaptive";
#else
const string test_type_name = "histogram";
#endif


static const int small_image_scale = 2;

// This is where data files are read from and written to
//static string  test_file_dir;

// (Diameter of area seached)/(face diameter detected by AgeRage)
static const double FACE_CROP_RATIO = 2.5; //  1.7; // 3.0; // = 1.5;

// Target minimum crop rectangle width
static const int MIN_CROP_WIDTH = 70;

/* 
 *  All the members of DetectorState are needed for a cvHaarDetectObjects() call
 */
struct DetectorState {
    IplImage*       _current_frame; 
    CvHaarClassifierCascade* _cascade;  
    CvMemStorage*   _storage;
  // Params  
    double          _face_crop_ratio; // // (Diameter of area seached)/(face diameter detected by AgeRage)
    double          _scale_factor;  // =1.1, 
    int             _min_neighbors; // =3, 
    FileEntry       _entry;
    string          _cascade_name;
};

    

static bool SortFacesByArea(const CvRect& r1, const CvRect& r2) {
    return r1.width*r1.height > r2.width*r2.height;
}


/*
 *  Detects faces in dp._current_frame cropped to rect
 *  Detects faces in whole image if rect == 0
 *  Returns list of face rectangles sorted by size
 */
vector<CvRect> detectFacesCrop(const DetectorState& dp, const CvRect* rect)    {
    CvSeq* faces = 0;
#if TEST_NO_CROP
    *((CvRect*) rect) = EMPTY_RECT;
#endif
    IplImage* cropped_image = dp._current_frame;
    if (rect) {
       cropped_image = cropImage(dp._current_frame, *rect);
       assert(containsRect(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height), *rect));
    }
    IplImage* gray_image  = cvCreateImage(cvSize(cropped_image->width, cropped_image->height), IPL_DEPTH_8U, 1);
    IplImage* small_image = cvCreateImage(cvSize(cropped_image->width/small_image_scale, cropped_image->height/small_image_scale), IPL_DEPTH_8U, 1);

    // convert to gray and downsize
    cvCvtColor (cropped_image, gray_image, CV_BGR2GRAY);
    cvResize (gray_image, small_image, CV_INTER_LINEAR);
        
        // detect faces
    faces = cvHaarDetectObjects (small_image, dp._cascade, dp._storage,
#if !HARDWIRE_HAAR_SETTINGS
                                    dp._scale_factor, dp._min_neighbors, 
#else                                    
                                        HAAR_SCALE_FACTOR, 2, 
#endif                                        
                                        CV_HAAR_DO_CANNY_PRUNING, cvSize (30, 30));
         
    vector <CvRect> face_list(faces != 0 ? faces->total : 0);
    for (int j = 0; j < (int)face_list.size(); j++) {
        face_list[j] = *((CvRect*) cvGetSeqElem (faces, j));
        assert(containsRect(cvRect(0, 0, small_image->width, small_image->height), face_list[j]));
       // Scale up to original image size
        face_list[j].x = cvRound((double)face_list[j].x*(double)cropped_image->width /(double)small_image->width);
        face_list[j].y = cvRound((double)face_list[j].y*(double)cropped_image->height/(double)small_image->height);
        face_list[j].width  = cvRound((double)face_list[j].width* (double)cropped_image->width /(double)small_image->width);
        face_list[j].height = cvRound((double)face_list[j].height*(double)cropped_image->height/(double)small_image->height);
        CvRect r = face_list[j];
        assert(containsRect(cvRect(0, 0, cropped_image->width, cropped_image->height), face_list[j]));
       
        // Correct for offset of cropped image in original
        if (rect) {
            face_list[j].x += rect->x;
            face_list[j].y += rect->y;
            assert(containsRect(*rect, face_list[j]));
        }
        assert(containsRect(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height), face_list[j]));
    }

    if (face_list.size() > 1) 
        sort(face_list.begin(), face_list.end(), SortFacesByArea);
 
    // Free images afer last call to cvGetSeqElem() !
    if (rect) 
        cvReleaseImage(&cropped_image); 
    cvReleaseImage(&gray_image);
    cvReleaseImage(&small_image);   
       
    return face_list;
}

vector<CvRect> detectFaces(const DetectorState& dp)    {
    return detectFacesCrop(dp, 0);
}

/*
 * Detect faces within a set of frames (ROI rectangles)
 *   croppedFrameList contains the frames at input and recieves the lists of faces for
 *   each rect at output
 */
void detectFacesMultiFrame(const DetectorState& dp, CroppedFrameList* croppedFrameList) {	
    for (int i = 0; i < (int)croppedFrameList->_frames.size(); i++) {
        CvRect rect =  croppedFrameList->_frames[i]._rect;
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        croppedFrameList->_frames[i]._faces = faces;
    }
}


CroppedFrameList_Histogram createMultiFrameList_ConcentricImage(const DetectorState& dp) {
    int    num_frames = 40;
    CroppedFrameList_Histogram cropped_frame_list;
    cropped_frame_list._frames.resize(num_frames);
    int image_width  = dp._current_frame->width;
    int image_height = dp._current_frame->height;
    CvRect rect;
    for (int i = 0; i < num_frames; i++) { 
        rect.x = i/2;
        rect.y = i/2; 
        rect.width  = image_width  - i; 
        rect.height = image_height - i; 
        cropped_frame_list._frames[i]._rect = rect;
    }
    cropped_frame_list._primary = cropped_frame_list._frames[0]._rect;
    return cropped_frame_list;
}


 
/*  
 *   Detect faces in an image and a set of frames (ROI rects) within that image
 */ 
CroppedFrameList_Histogram detectFaces_Histogram(const DetectorState& dp)    {
    //CroppedFrameList frame_list = createMultiFrameList_Cross(mp, face);
    CroppedFrameList_Histogram frame_list = createMultiFrameList_ConcentricImage(dp);
    detectFacesMultiFrame(dp, &frame_list);
    return frame_list;
}


/*****************************************************************************************
 * Mehran's adaptive method
 *****************************************************************************************/

static bool hasValidFace(const vector<CvRect> faces, int min_allowed_width, int min_allowed_height) {
    return (faces.size() > 0 && faces[0].width >= min_allowed_width && faces[0].height >= min_allowed_height);
}

static bool hasValidFaceTolerance(const vector<CvRect> faces, CvPoint face_center, int tolerance) {
    bool within_tolerance = false;
    if (faces.size() > 0) {
        CvPoint center = getCenter(faces[0]);
        double distance = hypot((double)(center.x - face_center.x), (double)(center.y - face_center.y));
        within_tolerance = (distance <= tolerance);
    }
    return within_tolerance;
}

static bool hasValidFaceBoth(const vector<CvRect> faces, int min_allowed_width, int min_allowed_height, CvPoint face_center, int tolerance) {
    return hasValidFace(faces, min_allowed_width, min_allowed_height) && hasValidFaceTolerance(faces, face_center, tolerance);
}

static CvRect findSmallestFaceRectangle(const DetectorState& dp, CvRect outer_rect, int min_allowed_width, int min_allowed_height) {
    double min_delta = 0.01;
    double delta = 0.1;
    double good_scale_factor =  1.0 + delta;
    CvRect good_rect = outer_rect;
    
    while (delta >=  min_delta) {
        double scale_factor = good_scale_factor;
        while (true) {
            scale_factor /= 1.0 + delta;
            CvRect rect = scaleRectConcentric(outer_rect, scale_factor); 
            vector<CvRect> faces = detectFacesCrop(dp, &rect);
            if (!hasValidFace(faces, min_allowed_width, min_allowed_height)) 
                break;
            good_rect = rect;
            good_scale_factor = scale_factor;
        }
        delta /= 2.0;
    }
    return good_rect;
}

/*
enlarge the frame and detect face
- repeat while face center falls within some tolerance of the original face center till failure
- then reduce the frame from *-
- use the mid point of these as the stable face radius

    Test in range 0.5x to 2.0x original face size = facter of 4;
*/
static CvRect findFaceSize(const DetectorState& dp, /* CvRect outer_frame, */ CvRect start_frame, CvRect start_face,
                                        int min_allowed_width, int min_allowed_height, double tolerance_ratio) {
    double min_ratio = 0.5;     // Min frame size / start_frame_size
    double max_ratio = 2.0;     // Max frame size / start_frame_size 
    int num_steps = 21;         // Number of frame sizes to check
    
    double ratio_range = max_ratio/min_ratio;
    double ratio_step = log(ratio_range)/(double)num_steps;
    int tolerance = cvRound(hypot(start_face.width, start_face.height)*tolerance_ratio);
    
    int min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpan(num_steps);
    CvPoint face_center = getCenter(start_face);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = scaleRectConcentric(start_frame, exp(ratio_step*(double)(i- num_steps/2))) ;
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFaceBoth(faces, min_allowed_width, min_allowed_height, face_center, tolerance))
            break;
        CroppedFrame frame(rect, faces);
        frameSpan[i] = frame;
         min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2+1; i < num_steps; i++) {
        assert(ratio_step*(double)(i- num_steps/2) <= 1.0);
        CvRect rect = scaleRectConcentric(start_frame, exp(ratio_step*(double)(i- num_steps/2)));
   //     assert(containsRect(outer_frame, rect));
  //      assert(containsRect(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height), outer_frame));
        assert(containsRect(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height), rect));

        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFaceBoth(faces, min_allowed_width, min_allowed_height, face_center, tolerance))
            break;
        CroppedFrame frame(rect, faces);
        frameSpan[i] = frame;
        max_i = i;
        if (max_i < 0)
            min_i = i;
    }
    
    CvRect best_face;
    if (min_i >= 0 && max_i >= 0) {
        int mid_i = (min_i + max_i)/2; 
        best_face = frameSpan[mid_i]._faces[0];
    }
    else {
        cerr << "findFaceSize could not find suitable face" << endl;
        best_face = start_face;
    }
    return best_face;
}

#if ADAPTIVE_RECURSIVE
static CroppedFrameList_Adaptive findFaceCenter(const DetectorState& dp, CvRect base_rect, int min_allowed_width, int min_allowed_height) {
    int    num_steps = ADAPTIVE_NUM_STEPS;    // Max number of steps to search in x and y direction
    
    int    image_width  = dp._current_frame->width;
    int    image_height = dp._current_frame->height;
    CvRect rect = base_rect;
    
        
    int dx = (image_width - rect.width)/num_steps;
    int dy = (image_height - rect.height)/num_steps;
    
    CroppedFrameList_Adaptive frame_list;
    int mid_ix = -1, mid_iy = -1;
    int min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanX(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x + (i- num_steps/2)*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        frame_list._frames.push_back(frame);
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
       CvRect rect = cvRect(base_rect.x + (i- num_steps/2)*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        frame_list._frames.push_back(frame);
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_ix = (min_i + max_i)/2; 
    
    min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanY(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + (i- num_steps/2)*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        frame_list._frames.push_back(frame);
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + (i- num_steps/2)*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        frame_list._frames.push_back(frame);
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_iy = (min_i + max_i)/2; 
   
    if (mid_ix > 0 && mid_iy > 0) {
        CvPoint center;
        center.x = getCenter(frameSpanX[mid_ix]._rect).x;
        center.y = getCenter(frameSpanY[mid_iy]._rect).y;
        frame_list._position_face  = averageRects(frameSpanX[mid_ix]._faces[0], frameSpanY[mid_iy]._faces[0]);
        frame_list._position_frame = cvRect(center.x - base_rect.width/2, center.y - base_rect.height/2, base_rect.width, base_rect.height);
        assert(containsRect(frame_list._position_frame, frame_list._position_face));
    }
    return frame_list;
}

CroppedFrameList_Adaptive detectFacesCenter_Adaptive(const DetectorState& dp)    {
   
   
  //  double frame_to_original = 1.1; // 1.3 kind of works; // 1.6 works;
  //  double frame_growth = 1.1;
      int    image_width  = dp._current_frame->width;
    int    image_height = dp._current_frame->height;
    
    // False positive detection. 
    //      tolerance_ratio = (max dist between face centers)/(frame diameter) = 0.1
    //      min_scale = (min face diameter)/(frame diameter) = 0.8 
    double  tolerance_ratio = cvRound(0.1/dp._face_crop_ratio);
    double  min_face_factor = cvRound(0.8/dp._face_crop_ratio);
    // Min face sizes - used to eliminate false positive face detections. min_face_factor * original detected face size
    int min_allowed_width  = cvRound((double)image_width*min_face_factor);
    int min_allowed_height = cvRound((double)image_height*min_face_factor);


      
    // Guess at right size for rectangle
   // CvRect base_rect = scaleRectConcentric(cvRect(0,0,image_width,image_height), frame_to_original/dp._face_crop_ratio); 
    // Find smallest rectangle that contains a face
    CvRect base_rect = scaleRectConcentric(cvRect(0,0,image_width,image_height), 1.0/1.2);
    base_rect = findSmallestFaceRectangle(dp, base_rect, min_allowed_width, min_allowed_height);
    base_rect = scaleRectConcentric(base_rect, 1.1);
    
   CroppedFrameList_Adaptive frame_list = findFaceCenter(dp, base_rect, min_allowed_width, min_allowed_height);
    if (!isEmptyRect(frame_list._position_face)) {
        CvRect outer_frame = cvRect(0, 0, image_width, image_height);
        frame_list._final_face = findFaceSize(dp, outer_frame, /* position_frame, */ frame_list._position_face,
                                        min_allowed_width, min_allowed_height, tolerance_ratio); 
    }
    return frame_list;
}
#else
CroppedFrameList_Adaptive detectFacesCenter_Adaptive(const DetectorState& dp)    {
   
    CroppedFrameList_Adaptive frame_list;
  //  double frame_to_original = 1.1; // 1.3 kind of works; // 1.6 works;
  //  double frame_growth = 1.1;
    int    num_steps = ADAPTIVE_NUM_STEPS;    // Max number of steps to search in x and y direction
    int    image_width  = dp._current_frame->width;
    int    image_height = dp._current_frame->height;
    
    // False positive detection. 
    //      tolerance_ratio = (max dist between face centers)/(frame diameter) = 0.1
    //      min_scale = (min face diameter)/(frame diameter) = 0.8 
    double  tolerance_ratio = cvRound(0.1/dp._face_crop_ratio);
    double  min_face_factor = cvRound(0.8/dp._face_crop_ratio);
    // Min face sizes - used to eliminate false positive face detections. min_face_factor * original detected face size
    int min_allowed_width  = cvRound((double)image_width*min_face_factor);
    int min_allowed_height = cvRound((double)image_height*min_face_factor);
    
    // Guess at right size for rectangle
   // CvRect base_rect = scaleRectConcentric(cvRect(0,0,image_width,image_height), frame_to_original/dp._face_crop_ratio); 
    // Find smallest rectangle that contains a face
    CvRect base_rect = scaleRectConcentric(cvRect(0,0,image_width,image_height), 1.0/1.2);
    base_rect = findSmallestFaceRectangle(dp, base_rect, min_allowed_width, min_allowed_height);
    base_rect = scaleRectConcentric(base_rect, 1.1);
    CvRect rect = base_rect;

    int dx = (image_width - rect.width)/num_steps;
    int dy = (image_height - rect.height)/num_steps;
    int mid_ix = -1, mid_iy = -1;
    int min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanX(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x + (i- num_steps/2)*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        frame_list._frames.push_back(frame);
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
       CvRect rect = cvRect(base_rect.x + (i- num_steps/2)*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        frame_list._frames.push_back(frame);
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_ix = (min_i + max_i)/2; 
    
    min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanY(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + (i- num_steps/2)*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        frame_list._frames.push_back(frame);
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + (i- num_steps/2)*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (!hasValidFace(faces, min_allowed_width, min_allowed_height))
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        frame_list._frames.push_back(frame);
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_iy = (min_i + max_i)/2; 
   
    if (mid_ix > 0 && mid_iy > 0) {
        CvPoint center;
        center.x = getCenter(frameSpanX[mid_ix]._rect).x;
        center.y = getCenter(frameSpanY[mid_iy]._rect).y;
        CvRect position_face  = averageRects(frameSpanX[mid_ix]._faces[0], frameSpanY[mid_iy]._faces[0]);
        CvRect position_frame = cvRect(center.x - base_rect.width/2, center.y - base_rect.height/2, base_rect.width, base_rect.height);
        assert(containsRect(position_frame, position_face));
        frame_list._position_frame = position_frame;
        CvRect outer_frame = cvRect(0, 0, image_width, image_height);
        frame_list._final_face = findFaceSize(dp, /*outer_frame, */position_frame, position_face,
                                        min_allowed_width, min_allowed_height, tolerance_ratio); 
    }
    return frame_list;
}
#endif


#if DRAW_FACES && SHOW_ALL_RECTANGLES
static void drawCroppedFrame(const void* ptr, const CroppedFrame& frame) {
    const DrawParams* wp = (const DrawParams*)ptr;
    CvScalar frame_color = (frame._faces.size() > 0) ? CV_RGB(0,0,255) : CV_RGB(125,125,125);
    drawRect(wp, frame._rect, frame_color, false);
#if VERBOSE    
    cout << rectAsString(frame._rect) << " : ";
#endif
    if (frame._faces.size() > 0) {
        drawRect(wp, frame._faces[0], CV_RGB(255,0,0), false);
        drawLine(wp, cvPoint(frame._rect.x, frame._rect.y), cvPoint(frame._faces[0].x, frame._faces[0].y), CV_RGB(255,0,0), false);
#if VERBOSE       
        cout << rectAsString(frame._faces[0]);
        if (!containsRect(frame._rect, frame._faces[0]))
            cout << " ***";
#endif            
    }
#if VERBOSE    
    cout << endl;
#endif    
}
#endif


/*
 *  Process (detect faces) in set of frames within an image with a particular
 *  combination of settings (in dp)
 *  Returns a list of results, one for each frame
 */
CroppedFrameList_Histogram  
    processOneImage_Histogram(const DetectorState& dp) {
   
    CroppedFrameList_Histogram frame_list = detectFaces_Histogram(dp);  
 
#if DRAW_FACES       
    // draw faces
    DrawParams wp;
    wp._draw_image = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
   
    cvFlip (dp._current_frame, wp._draw_image, 1);
    
    CvRect face_rect = dp._entry.getFaceRect(dp._face_crop_ratio);
    CvPoint center = cvPoint(dp._entry._face_center.x - face_rect.x, dp._entry._face_center.y - face_rect.y);

#if SHOW_ALL_RECTANGLES   
    frame_list.iterateFrameList((void*) &wp, drawCroppedFrame);
#endif 
   
    drawCircle(&wp, center, dp._entry._face_radius, CV_RGB(255,0,0), true);
   // drawRect(&wp, frame_list.middleConsecutiveWithFaces(), CV_RGB(0,0,255), false);
    drawRect(&wp, frame_list.getBestFace(), CV_RGB(255,255,0), false);
 
    cvShowImage (WINDOW_NAME, wp._draw_image); 
    cvWaitKey(DRAW_WAIT);
    cvReleaseImage(&wp._draw_image);
#endif  // #if DRAW_FACES 
  
    return frame_list;
}

/*
 *  Process (detect faces) in set of frames within an image with a particular
 *  combination of settings (in dp)
 *  Returns a list of results, one for each frame
 */
CroppedFrameList_Adaptive  
    processOneImage_Adaptive(const DetectorState& dp) {
   
    CroppedFrameList_Adaptive frame_list = detectFacesCenter_Adaptive(dp);  
    CvRect best_face = frame_list.getBestFace();
    CvRect position_frame = frame_list._position_frame;
#if DRAW_FACES     
    // draw faces
    DrawParams wp;
    wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
   
    cvFlip (dp._current_frame, wp._draw_image, 1);
   
    CvRect face_rect = dp._entry.getFaceRect(dp._face_crop_ratio);
    CvPoint center = cvPoint(dp._entry._face_center.x - face_rect.x, dp._entry._face_center.y - face_rect.y);
    CvRect face_rect_adjusted = cvRect(0, 0, face_rect.width, face_rect.height);

    cout << rectAsString(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height)) << " : " << rectAsString(face_rect) << " boundary : face first guess" << endl;
 
#if SHOW_ALL_RECTANGLES   
    frame_list.iterateFrameList((void*) &wp, drawCroppedFrame);
#endif 
   
    cout << "         outer  frame = " << rectAsString(cvRect(0, 0, dp._current_frame->width, dp._current_frame->height)) << endl;
    cout << "* best position frame = " << rectAsString(offsetRectByRect(position_frame, face_rect)) << endl;
    cout << "***         best face = " << rectAsString(offsetRectByRect(best_face, face_rect)) << endl;
    drawRect(&wp, face_rect_adjusted, CV_RGB(0,0,255), true);
    drawCircle(&wp, center, dp._entry._face_radius, CV_RGB(255,0,0), true);
    drawRect(&wp, position_frame, CV_RGB(255,0,255), false);
    drawRect(&wp, best_face, CV_RGB(255,255,255), false);
    
    cvShowImage (WINDOW_NAME, wp._draw_image); 
    cvWaitKey(DRAW_WAIT);
    cvReleaseImage(&wp._draw_image);
#endif // #if DRAW_FACES     
    return frame_list;
}

static IplImage* scaleImage640x480(IplImage* image) {
    if (image->width > image->height)
        return scaleImageWH(image, 640, 480);
    else
        return scaleImageWH(image, 480, 640);
}



/*
 * Draw results in original image
 */
#if DRAW_FACES
static void drawResultImage(const FaceDetectResult& result) {
    FileEntry  entry = result._entry;
    IplImage*  image  = cvLoadImage(entry._image_name.c_str());
    if (!image) {
        cerr << "Could not find '" << entry._image_name << "'" << endl;
        abort();
    }
    IplImage*  scaled_image = scaleImage640x480(image);
    cvReleaseImage(&image);
    DrawParams wp;
    wp._draw_image    = cvCreateImage(cvSize (scaled_image->width, scaled_image->height), IPL_DEPTH_8U, 3);
    
    cvFlip (scaled_image, wp._draw_image, 1);
    CvRect face_rect = result._face_rect; 
    CvRect orig_rect = entry.getFaceRect(1.0);
    
    drawRect(&wp, face_rect, CV_RGB(255,0,0), false);
    drawRect(&wp, orig_rect, CV_RGB(0,0,255), false);

    string marked_image_name = entry._image_name + "." + test_type_name + ".marked.jpg";
    cvShowImage (WINDOW_NAME, wp._draw_image); 
    cvWaitKey(DRAW_WAIT);
    cvSaveImage(marked_image_name.c_str(), wp._draw_image);

    cvReleaseImage(&wp._draw_image);
    cvReleaseImage(&scaled_image);
}
#define DRAW_RESULT_IMAGE(r) drowResultImage(r)
#else
#define DRAW_RESULT_IMAGE(r)
#endif                   

/*
 *  Ranges of inputs to the program
 */
struct  ParamRanges {
    int     _min_neighbors_min, _min_neighbors_max, _min_neighbors_delta;
    double  _scale_factor_min,  _scale_factor_max,  _scale_factor_delta;
    vector<FileEntry>   _file_entries;
    vector<string>      _cascades;
    
    // !@#$ does not belong here
    mutable ofstream _output_file;
    mutable long     _last_flush_time;
    long     _flush_dt;
    
    ParamRanges() {
        _last_flush_time = 0L;
        _flush_dt = 5L;
    }
    void flushIfNecessary() const {
        long t = time(0);
        if (t > _last_flush_time + _flush_dt) {
            _output_file.flush();
            _last_flush_time = t;
        }
    }
    
};

/*
 * Calculate a good crop ratio
 */
double calcCropRatio(const IplImage* image, CvRect face_rect, int min_width, double init_ratio) {
    CvPoint center = getCenter(face_rect);
    cout << "face_rect = " << rectAsString(face_rect) << endl;
    cout << "image w x h = " << image->width << " x " << image->height << endl;
    cout << "min_width = " << min_width << ", init_ratio = " << init_ratio <<  endl;
    assert(0 <= center.x && center.x < image->width);
    assert(0 <= center.y && center.y < image->height);
    
    int furthest_edge = max(center.x, center.y);
    furthest_edge = max(furthest_edge, image->width - center.x);
    furthest_edge = max(furthest_edge, image->height - center.y);  
    
    int radius0 = getRadius(face_rect);
   
    int target_radius = min(min_width/2, furthest_edge);
    double target_ratio = (double)target_radius/(double)radius0;
    double ratio = max(init_ratio, target_ratio);
    cout << "face crop ratio = " << ratio << endl;
    return ratio;
}

/*
 * Load cascade, either from the OS X app resouce bundle or from a known location
 */ 
#if MAC_APP
static const int CASCADE_NAME_LEN = 2048;
static char   CASCADE_NAME[CASCADE_NAME_LEN] = "~/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
#endif
static const string getCascadePath(const string cascade_name)     {
    string cascade_path;
#if MAC_APP
    CFBundleRef mainBundle  = CFBundleGetMainBundle();
    assert (mainBundle);
    
    CFURLRef  cascade_url = CFBundleCopyResourceURL (mainBundle, 
                     CFStringCreateWithCString(NULL, cascade_name.c_str(), kCFStringEncodingASCII),
                    CFSTR("xml"), NULL);
    assert (cascade_url);
    Boolean     got_it      = CFURLGetFileSystemRepresentation (cascade_url, true, 
                                                                reinterpret_cast<UInt8 *>(CASCADE_NAME), CASCADE_NAME_LEN);
    if (! got_it)
        abort ();
    cascade_path = CASCADE_NAME;
#else
    cascade_path = cascade_name + ".xml";
#endif    
    return cascade_path;
}

#endif
