
#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <list>
#include "face_io.h"
#include "face_draw.h"
#include "face_calc.h"
#include "cropped_frames.h"

using namespace std;

#define SORT_AND_SHOW 0
#define DO_NOT_SET_ROI 1
#define TEST_NO_CROP 0
#define ADAPTIVE_FACE_SEARCH 1

static const char  * WINDOW_NAME  = "Face Tracker with Sub-Frames";
static const CFIndex CASCADE_NAME_LEN = 2048;
static char    CASCADE_NAME[CASCADE_NAME_LEN] = "~/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
static const int small_image_scale = 2;


/* 
 *  All the members of DetectorState are needed for a cvHaarDetectObjects() call
 */
struct DetectorState {
    IplImage*       _current_frame; 
    CvHaarClassifierCascade* _cascade;  
    CvMemStorage*   _storage;
  // Params  
    double          _scale_factor;  // =1.1, 
    int             _min_neighbors; // =3, 
    FileEntry       _entry;
    string          _cascade_name;
};

/*
 * Rectangles returned by CvHaar need to be expanded by this amount for face to be detected in the expanded rectangle
 */
double haar_face_pad_width = 1.4;
double haar_face_pad_height = 1.6;
/*
    Frames are rectangles that contain faces
    Outer rect is face rect returned from cvHaar scaled up by _outer_pad_width x _outer_pad_height
    Inner rect is outer rect scaled down by _frac_width x _frac_height
    Number of inner frames checked is based on _num_inc_x and _num_inc_y
 
    !@#$ Need to make inner rect big enough to detect face in

 */
struct MultiFrameParams {
    double _outer_pad_width;
    double _outer_pad_height;
    double _frac_width;   
    double _frac_height;
    int    _num_inc_x;
    int    _num_inc_y;
    MultiFrameParams() {
        _frac_width = _frac_height = 0.8;
        _outer_pad_width = haar_face_pad_width / _frac_width;
        _outer_pad_height = haar_face_pad_height / _frac_height;
        _num_inc_x = _num_inc_y = 5;
    }
};



    
string rectAsString(CvRect r) {
    ostringstream s;
    s << "[" << setw(3) << r.x << "," << setw(3) << r.y << "," << setw(3) << r.width << "," << setw(3)  << r.height << "]";
    return s.str();
}

/*
 * Draw a set of crop frames onto an image
 */
void drawCropFrames(const DrawParams* wp, const MultiFrameParams* mp, const CroppedFrameList* frameList) {
    CvScalar onColor  = CV_RGB(255,255,255);
    CvScalar offColor = CV_RGB(255,0,0);
    CvScalar mainColor = CV_RGB(0,0,255);
    CvScalar color;
    
    drawRect(wp, frameList->_primary, mainColor, false);
    
    for (int i = 0; i < frameList->_frames.size(); i++) {
        CvRect r = frameList->_frames[i]._rect;
        color = frameList->_frames[i].hasFaces() ? onColor : offColor;
        drawRect(wp, r, color, false);
    }
}





bool SortFacesByArea(const CvRect& r1, const CvRect& r2) {
    return r1.width*r1.height > r2.width*r2.height;
}


/*
 *  Detects faces in dp->_current_frame cropped to rect
 *  Detects faces in whole image if rect == 0
 *  Returns list of face rectangles sorted by size
 */
vector<CvRect> detectFacesCrop(const DetectorState* dp, const CvRect* rect)    {
    CvSeq* faces = 0;
#if TEST_NO_CROP
   // rect = 0;
   *((CvRect*) rect) = EMPTY_RECT;
   // cerr << "****************** TEST NO CROP ***************" << endl;
#endif
#if !DO_NOT_SET_ROI    
        if (rect) {
            cvSetImageROI(dp->_current_frame, *rect);
            cvSetImageROI(dp->_gray_image, *rect);
        }
#endif
    IplImage* cropped_image = dp->_current_frame;
    if (rect) 
       cropped_image = cropImage(dp->_current_frame, *rect);
   // IplImage* cropped_image =  cropImage(dp->_current_frame, *rect);
    IplImage* gray_image  = cvCreateImage(cvSize(cropped_image->width, cropped_image->height), IPL_DEPTH_8U, 1);
    IplImage* small_image = cvCreateImage(cvSize(cropped_image->width/small_image_scale, cropped_image->height/small_image_scale), IPL_DEPTH_8U, 1);

    // convert to gray and downsize
    cvCvtColor (cropped_image, gray_image, CV_BGR2GRAY);
    cvResize (gray_image, small_image, CV_INTER_LINEAR);
        
        // detect faces
    faces = cvHaarDetectObjects (small_image, dp->_cascade, dp->_storage,
#if !DO_NOT_SET_ROI
                                    dp->_scale_factor, dp->_min_neighbors, 
#else                                    
                                        1.1, 2, 
#endif                                        
                                                CV_HAAR_DO_CANNY_PRUNING,
                                            cvSize (30, 30));
        
#if !DO_NOT_SET_ROI       
        if (rect) {
            cvResetImageROI(dp->_current_frame);
            cvResetImageROI(dp->_gray_image);
        }
#endif   
         
    vector <CvRect> face_list(faces != 0 ? faces->total : 0);
    for (int j = 0; j < face_list.size(); j++) {
        face_list[j] = *((CvRect*) cvGetSeqElem (faces, j));
        // Scale up to original image size
        face_list[j].x = cvRound((double)face_list[j].x*(double)dp->_current_frame->width /(double)small_image->width);
        face_list[j].y = cvRound((double)face_list[j].y*(double)dp->_current_frame->height/(double)small_image->height);
        face_list[j].width  = cvRound((double)face_list[j].width* (double)dp->_current_frame->width /(double)small_image->width);
        face_list[j].height = cvRound((double)face_list[j].height*(double)dp->_current_frame->height/(double)small_image->height);
        // Correct for offset of cropped image in original
        if (rect) {
            face_list[j].x += rect->x;
            face_list[j].y += rect->y;
        }
    }
    //cout << "face_list.size() = " << face_list.size() << endl;
    if (face_list.size() > 1) 
        sort(face_list.begin(), face_list.end(), SortFacesByArea);
 
    // Free images afer last call to cvGetSeqElem() !
    if (rect) 
        cvReleaseImage(&cropped_image); 
    cvReleaseImage(&gray_image);
    cvReleaseImage(&small_image);   
       
    return face_list;
}


/*
 * Pad out a face rectangle to create a search rectangle
 */
CvRect getPaddedFace(CvRect face, double xpad, double ypad) {
    CvRect paddedFace;
    paddedFace.width = cvRound((double)face.width * xpad);
    paddedFace.height = cvRound((double)face.height * ypad);
    paddedFace.x = max(0, face.x - (paddedFace.width - face.width)/2);
    paddedFace.y = max(0, face.y - (paddedFace.height - face.height)/2);
    // !@#$ need to handle cases where padding is outside image boundary
    return paddedFace;
}

/*
 * Create a multi-rect list in a cross shape with x and y arms
 */
 CroppedFrameList createMultiFrameList_Cross(const MultiFrameParams* mp, CvRect faceIn) {
    // Rectangle that covers whole face;
    CvRect face = getPaddedFace(faceIn, mp->_outer_pad_width, mp->_outer_pad_height);
    // Rectangle to search within
    CvRect paddedFace = getPaddedFace(face, 1.0/(double)mp->_frac_width, 1.0/(double)mp->_frac_height);	

    // Size of frames to test. For now they are same as original face rectangle
    // Need to be large enought to contain entire face !@#$
    int cropWidth  = face.width;
    int cropHeight = face.height;
    
    double dx = (double)(paddedFace.width-cropWidth)/(double)(mp->_num_inc_x-1);
    double dy = (double)(paddedFace.height-cropHeight)/(double)(mp->_num_inc_y-1);
    int cx = (paddedFace.width-cropWidth)/2;
    int cy = (paddedFace.height-cropHeight)/2;
    
    CroppedFrameList croppedFrameList;	
    croppedFrameList._primary = face;
    croppedFrameList._frames.resize(mp->_num_inc_x + mp->_num_inc_y);
    CvRect rect;
    rect.width = cropWidth;
    rect.height = cropHeight;
    for (int i = 0; i < mp->_num_inc_x; i++) {
        rect.x = paddedFace.x + cvRound((double)i * dx);
        rect.y = cy;
        croppedFrameList._frames[i]._rect = rect;
    }
    for (int i = 0; i < mp->_num_inc_y; i++) {
        rect.x = cx;
        rect.y = paddedFace.y + cvRound((double)i * dy);
        croppedFrameList._frames[mp->_num_inc_x+i]._rect = rect;
    }
    return croppedFrameList;
}

/*
 *   Create a list of concentric rectangles starting from image border
 */
/*
 CroppedFrameList createMultiFrameList_ConcentricImage(const DetectorState* dp) {
   // double innerFrameFrac = 0.9;
    int    numFrames = 40;
    CroppedFrameList croppedFrameList;
    croppedFrameList._frames.resize(numFrames);
    int imageWidth  = dp->_small_image->width; //dp->_current_frame->width;
    int imageHeight = dp->_small_image->height;//dp->_current_frame->height;
    CvRect rect;
    for (int i = 0; i < numFrames; i++) { 
        rect.x = i/2; // cvRound((1.0 - innerFrameFrac)*imageWidth *(double)i/(double)(numFrames - 1));
        rect.y = i/2; // cvRound((1.0 - innerFrameFrac)*imageHeight*(double)i/(double)(numFrames - 1));
        rect.width  = imageWidth  - i; // 2 * rect.x;
        rect.height = imageHeight - i; // 2 * rect.y;
        croppedFrameList._frames[i]._rect = rect;
  //      cout << "rect[" << i << "] = " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
    }
    croppedFrameList._primary = croppedFrameList._frames[0]._rect;
    return croppedFrameList;
 }
*/ 


 CroppedFrameList createMultiFrameList_ConcentricImage2(const DetectorState* dp) {
    int    num_frames = 40;
    CroppedFrameList cropped_frame_list;
    cropped_frame_list._frames.resize(num_frames);
    int image_width  = dp->_current_frame->width;
    int image_height = dp->_current_frame->height;
    CvRect rect;
    for (int i = 0; i < num_frames; i++) { 
        rect.x = i/2;
        rect.y = i/2; 
        rect.width  = image_width  - i; 
        rect.height = image_height - i; 
        cropped_frame_list._frames[i]._rect = rect;
  //      cout << "rect[" << i << "] = " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
    }
    cropped_frame_list._primary = cropped_frame_list._frames[0]._rect;
    return cropped_frame_list;
 }


/*
 * Detect faces within a set of frames (ROI rectangles)
 *   croppedFrameList contains the frames at input and recieves the lists of faces for
 *   each rect at output
 */
 void detectFacesMultiFrame(const DetectorState* dp, CroppedFrameList* croppedFrameList) {	
    for (int i = 0; i < croppedFrameList->_frames.size(); i++) {
        CvRect rect =  croppedFrameList->_frames[i]._rect;
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        croppedFrameList->_frames[i]._faces = faces;
//        cout << "Test " << i << ": " <<   croppedFrameList->_frames[i].hasFaces();
    }
}


vector<CvRect> detectFaces(DetectorState* dp)    {
    return detectFacesCrop(dp, 0);
}

 
/*  
 *   Detect faces in an image and a set of frames (ROI rects) within that image
 */ 
CroppedFrameList detectFaces2(const DetectorState* dp)    {
    //CroppedFrameList frameList = createMultiFrameList_Cross(mp, face);
    CroppedFrameList frameList = createMultiFrameList_ConcentricImage2(dp);
    detectFacesMultiFrame(dp, &frameList);
    return frameList;
}



/*  
 *   Detect faces in an image and a set of frame starting from middle
 - start from the origianal frame given to you
- enlarge the frame by some factor, say 1.1x the radius
- detect face
- slide to the left till you don't find any more faces
- slide to the right till you don't find any more faces
- do the same thing horizontally
- find the mid point of the above ranges
- consider this point your median face center (now we need to find a good, stable radius)
*- enlarge the frame and detect face
- repeat while face center falls within some tolerance of the original face center till failure
- then reduce the frame from *-
- use the mid point of these as the stable face radius
 */ 
  
CvRect detectFacesCenterAdaptive(const DetectorState* dp)    {
    //CroppedFrameList frameList = createMultiFrameList_Cross(mp, face);
    CroppedFrameList frameList;
    double frame_to_original = 1.6;
    double frame_growth = 1.1;
    int    num_steps = 11;
    int image_width  = dp->_current_frame->width;
    int image_height = dp->_current_frame->height;
    CvRect base_rect = scaleRectConcentric(cvRect(0,0,image_width,image_height), frame_to_original/FACE_CROP_SCALE); 
    CvRect rect = base_rect;
    CvRect final_rect = EMPTY_RECT;
    
    int dx = (image_width - rect.width)/num_steps;
    int dy = (image_height - rect.height)/num_steps;
    int mid_ix = -1, mid_iy = -1;
    int min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanX(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x + i*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (faces.size() == 0)
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
       CvRect rect = cvRect(base_rect.x + i*dx, base_rect.y, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (faces.size() == 0)
            break;
        CroppedFrame frame(rect, faces);
        frameSpanX[i] = frame;
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_ix = (min_i + max_i)/2; 
    
    min_i = -1, max_i = -1;
    vector<CroppedFrame> frameSpanY(num_steps);
    for (int i = num_steps/2; i >= 0; i--) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + i*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (faces.size() == 0)
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        min_i = i;
        if (max_i < 0)
            max_i = i;
    }
    for (int i = num_steps/2 + 1; i < num_steps; i++) {
        CvRect rect = cvRect(base_rect.x, base_rect.y + i*dy, base_rect.width, base_rect.height);
        vector<CvRect> faces = detectFacesCrop(dp, &rect);
        if (faces.size() == 0)
            break;
        CroppedFrame frame(rect, faces);
        frameSpanY[i] = frame;
        max_i = i;
        if (min_i < 0)
            min_i = i;
    }
    if (min_i >= 0 && max_i >= 0)
        mid_iy = (min_i + max_i)/2; 
   
    CvPoint center;
    
    if (mid_ix > 0 && mid_iy > 0) {
        center.x = getCenter(frameSpanX[mid_ix]._rect).x;
        center.y = getCenter(frameSpanY[mid_iy]._rect).y;
        final_rect = cvRect(center.x - rect.width/2, center.y - rect.height/2, rect.width, rect.height);
    }
        
 
  
    return final_rect;
}




static void drawCroppedFrame(const void* ptr, const CroppedFrame& frame) {
    const DrawParams* wp = (const DrawParams*)ptr;
    drawRect(wp, frame._rect, CV_RGB(0,255,255), false);
    if (frame._faces.size() > 0) {
        drawRect(wp, frame._faces[0], CV_RGB(255,255,255), false);
    }
}

#define SHOW_ALL_RECTANGLEsS 1
/*
 *  Process (detect faces) in set of frames within an image with a particular
 *  combination of settings (in dp)
 *  Returns a list of results, one for each frame
 */
CroppedFrameList  
    processOneImage2(const DetectorState& dp /*,
                     const DrawParams& wp */) {
   
    CroppedFrameList frameList = detectFaces2(&dp);  
    
    // draw faces
    DrawParams wp;
    wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
   
    cvFlip (dp._current_frame, wp._draw_image, 1);
    
    CvRect face_rect = dp._entry.getFaceRect(FACE_CROP_SCALE);
    CvPoint center = cvPoint(dp._entry._face_center.x - face_rect.x, dp._entry._face_center.y - face_rect.y);

#if SHOW_ALL_RECTANGLEsS   
    frameList.iterateFrameList((void*) &wp, drawCroppedFrame);
#endif 
   
    drawCircle(&wp, center, dp._entry._face_radius, CV_RGB(255,0,0), true);
    drawRect(&wp, frameList.middleConsecutiveWithFaces(), CV_RGB(0,0,255), false);
    drawRect(&wp, frameList.getBestFace(), CV_RGB(255,255,0), false);
 
    cvShowImage (WINDOW_NAME, wp._draw_image); 
    cvWaitKey(1000);
    
    cvReleaseImage(&wp._draw_image);
    return frameList;
}

/*
 *  Process (detect faces) in set of frames within an image with a particular
 *  combination of settings (in dp)
 *  Returns a list of results, one for each frame
 */
void  
    processOneImage3(const DetectorState& dp /*,
                     const DrawParams& wp */) {
   
    CvRect new_face_rect = detectFacesCenterAdaptive(&dp);  
    
    // draw faces
    DrawParams wp;
    wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
   
    cvFlip (dp._current_frame, wp._draw_image, 1);
    
   
    CvRect face_rect = dp._entry.getFaceRect(FACE_CROP_SCALE);
    CvPoint center = cvPoint(dp._entry._face_center.x - face_rect.x, dp._entry._face_center.y - face_rect.y);

#if SHOW_ALL_RECTANGLEsS   
  //  frameList.iterateFrameList((void*) &wp, drawCroppedFrame);
#endif 
   
    drawCircle(&wp, center, dp._entry._face_radius, CV_RGB(255,0,0), true);
    drawRect(&wp, face_rect, CV_RGB(0,0,255), false);
    drawRect(&wp, new_face_rect, CV_RGB(255,255,255), false);

 //   drawRect(&wp, frameList.getBestFace(), CV_RGB(255,255,0), false);
 
    cvShowImage (WINDOW_NAME, wp._draw_image); 
    cvWaitKey(1000);
    
    cvReleaseImage(&wp._draw_image);
   
}

struct FaceDetectResult {
    int     _num_frames_total;
    int     _num_frames_faces;
    int     _max_consecutive_faces;
    CvRect  _middle_frame_consecutive_faces;
    int     _ordinal;
    double  _scale_factor;
    int     _min_neighbors;
    FileEntry  _entry;
    string  _cascade_name;
    CvRect  _face_rect;
    int     _num_false_positives;
    int     _dbg_index;
    
    FaceDetectResult() {
 //       cout << "FaceDetectResult::FaceDetectResult()" << endl;
        _entry._image_name = _cascade_name = "";
        _dbg_index  = -1;
    }
    FaceDetectResult(int num_faces_total, int num_frames_faces, int max_consecutive_faces, CvRect middle_frame_consecutive_faces,
                    double scale_factor, int min_neighbors, 
                       const FileEntry& entry, string cascade_name, 
                        CvRect face_rect, int num_false_positives) {
        _num_frames_total = num_faces_total;
        _num_frames_faces = num_frames_faces;
        _max_consecutive_faces = max_consecutive_faces;
        _middle_frame_consecutive_faces = middle_frame_consecutive_faces;
        _scale_factor = scale_factor;
        _min_neighbors = min_neighbors;
        _entry = entry;
        _cascade_name = cascade_name;
        _ordinal = -1;
        _face_rect = face_rect;
        _num_false_positives = num_false_positives;
        _dbg_index  = -2;
    }
    FaceDetectResult& assign(const FaceDetectResult& r)  {
  //      if (r._dbg_index == 0) {
  //          cerr << "Bogus FaceDetectResult record" << endl;
  //      }
        _num_frames_total = r._num_frames_total;
        _num_frames_faces = r._num_frames_faces;
        _max_consecutive_faces = r._max_consecutive_faces;
        _middle_frame_consecutive_faces = r._middle_frame_consecutive_faces;
        _ordinal          = r._ordinal;
        _scale_factor     = r._scale_factor;
        _min_neighbors    = r._min_neighbors;
        _entry            = r._entry;
        _cascade_name     = r._cascade_name;
        _face_rect        = r._face_rect;
        _num_false_positives = r._num_false_positives;
        _dbg_index          = r._dbg_index;
        return *this;
    }

    FaceDetectResult& operator=(const FaceDetectResult& r)  {
//        cout << "operator=() from: " << r._dbg_index << " to:" << _dbg_index << endl;
        return assign(r);
    }
    FaceDetectResult(const FaceDetectResult& r) {
    // cout << "copy ctor " << &r <<  endl;
    //   cout << "copy ctor " << r._dbg_index <<  endl;
        assign(r);
    }
};


void checkResults( vector<FaceDetectResult> results, const string title) {
    vector<FaceDetectResult>::iterator rit;
    int i = 0;
    cout << "---------------- checkResults() !@#$ " << title << ": " << results.size() << " --------------" << endl;
    for (rit = results.begin(); rit != results.end(); rit++) {
        ++i;
        (*rit)._dbg_index = i;
         cout << (*rit)._dbg_index << ": " << (*rit)._entry._image_name << ", ";
    }
    cout << endl;
}
#define CHECK_RESULTS(r, t) checkResults(r, t)

bool resultsSortFuncImageName(const FaceDetectResult& r1, const FaceDetectResult& r2) {
    return (r1._entry._image_name.compare(r2._entry._image_name) > 0);
}
bool resultsSortFuncOrder(const FaceDetectResult& r1, const FaceDetectResult& r2) {
    return r1._num_frames_faces > r2._num_frames_faces;
}
//   bool resultsSortFunc(const FaceDetectResult& r1, const FaceDetectResult& r2) {
bool resultsSortFunc( FaceDetectResult r1,  FaceDetectResult r2) {
  //  cout << "resultsSortFunc in" << endl;
    int d = 0;
    if (d==0)
        d = (r1._num_frames_faces != 0 ? 1 : 0) - (r2._num_frames_faces != 0 ? 1 : 0);
     if (d==0)
        d = r2._num_false_positives - r1._num_false_positives;
  //  if (d==0)
  //      d = r2._ordinal - r1._ordinal;
    if (d==0)
        d = r1._num_frames_faces - r2._num_frames_faces;
    if (d==0)
        d = r1._entry._image_name.compare(r2._entry._image_name);
    if (d==0)
        d = r1._cascade_name.compare(r2._cascade_name);
    if (d==0)
        d = r1._min_neighbors - r2._min_neighbors;
    if (d==0)
        d = cvRound(r1._scale_factor - r2._scale_factor);

    //cout << "resultsSortFunc out" << endl;
    return (d >= 0);
}
/* !@#$ screws up vector order leading to a subsequent crash in sort()
void computeResultOrder(vector<FaceDetectResult>& results,
                        vector<FaceDetectResult>::iterator it0,
                        vector<FaceDetectResult>::iterator it1) {
    sort(it0, it1, resultsSortFuncOrder);
    int ordinal = -1;
    int num_frames_faces = -1;
    vector<FaceDetectResult>::iterator it;
    for (it = it0; it != it1; it++) {
        if ((*it)._num_frames_faces != num_frames_faces) {
            ++ordinal;
            num_frames_faces = (*it)._num_frames_faces;
        }
        (*it)._ordinal = ordinal;
    }
}
*/

void  showOneResultFile(const FaceDetectResult& r, ostream& of) { 
    const string sep = ", ";
   
    of << setw(4) << r._num_frames_total
        <<  sep << setw(4) << r._num_frames_faces
        <<  sep << setw(4) << r._max_consecutive_faces
        <<  sep << setw(4) << r._middle_frame_consecutive_faces.x - r._entry._pad
        <<  sep << setw(4) << r._middle_frame_consecutive_faces.y - r._entry._pad
        <<  sep << setw(4) << r._middle_frame_consecutive_faces.width
        <<  sep << setw(4) << r._middle_frame_consecutive_faces.height
        <<  sep << setw(4) << r._num_false_positives
        <<  sep << setw(4) << r._face_rect.x - r._entry._pad
        <<  sep << setw(4) << r._face_rect.y - r._entry._pad
        <<  sep << setw(4) << r._face_rect.width
        <<  sep << setw(4) << r._face_rect.height
        <<  sep << setw(6) << setprecision(4) << r._scale_factor 
        <<  sep << setw(4) << r._min_neighbors
        <<  sep << setw(4) << r._entry._pad
        <<  sep << setw(4) << r._entry._face_center.x
        <<  sep << setw(4) << r._entry._face_center.y
        <<  sep << setw(4) << r._entry._face_radius
        <<  sep << setw(4) << r._entry._face_angle
        <<  sep << setw(20) << r._entry._image_name
        <<  sep << setw(32) << r._cascade_name
        << endl;
}
void  showHeaderFile(ostream& of) { 
    const string sep = ", ";
   
    of << setw(4) << "frames"
        <<  sep << setw(4) << "with faces"
        <<  sep << setw(4) << "consecutive faces"
        <<  sep << setw(4) << "best x"
        <<  sep << setw(4) << "best y"
        <<  sep << setw(4) << "best w"
        <<  sep << setw(4) << "best h"
        <<  sep << setw(4) << "false +ves"
        <<  sep << setw(4) << "face x"
        <<  sep << setw(4) << "face y"
        <<  sep << setw(4) << "face w"
        <<  sep << setw(4) << "face h"
        <<  sep << setw(5) << "scale factor" 
        <<  sep << setw(4) << "min neighbors"
        <<  sep << setw(4) << "pad"
        <<  sep << setw(4) << KEY_FACE_CENTER_X
        <<  sep << setw(4) << KEY_FACE_CENTER_Y
        <<  sep << setw(4) << KEY_FACE_RADIUS
        <<  sep << setw(4) << KEY_FACE_ANGLE
        <<  sep << setw(20) << KEY_IMAGE_NAME
        
        <<  sep << setw(32) << "cascade name"
        << endl;
}

void showResultsRange(vector<FaceDetectResult>& results,
                        vector<FaceDetectResult>::iterator it0,
                        vector<FaceDetectResult>::iterator it1) {
    vector<FaceDetectResult>::iterator it;
    for (it = it0; it != it1; it++) {
        showOneResultFile(*it, cout);
    }
}
                        
#if SORT_AND_SHOW
#define SHOw_RESULTS(r) showOneResult(r)
void showResults(vector<FaceDetectResult> results) {
    cout << "======================== showResults " << results.size() << " =================================" << endl;
    if (results.size() > 0 ) {

    // Compute ordinals for each image
        sort(results.begin(), results.end(), resultsSortFuncImageName);
        vector<FaceDetectResult>::iterator it, it0 = results.begin();
    /*
        for (it = results.begin(); it != results.end(); it++) {
            if ((*it)._image_name.compare((*it0)._image_name) != 0) {
           //     computeResultOrder(results, it0, it);
                cout << "++++++" << (*it0)._image_name << endl;
                showResultsRange(results, it0, it);
                it0 = it;
                cout << "++++++"  << endl;
            }
        }
        if (it != it0) {
       //     computeResultOrder(results, it0, it);
            showResultsRange(results, it0, it);
        }
        */
    
        CHECK_RESULTS(results, "showResults");
        cout << " sort(results.begin(), results.end(), resultsSortFunc)" << endl;
        int begin_adr = reinterpret_cast<unsigned int>(&(*results.begin()));
        int end_adr = (unsigned)(&(*results.end()));
        cout << " results.begin() = " << hex << begin_adr << ", results.end() = " << hex << end_adr << endl;
        cout << "sizeof(vector<FaceDetectResult>::iterator) = " << sizeof(vector<FaceDetectResult>::iterator) << endl;
        cout << " ((int)results.end() - (int)results.begin())/sizeof(vector<FaceDetectResult>::iterator) = " << (end_adr - begin_adr)/sizeof(vector<FaceDetectResult>::iterator)
            << " results.size() = " << results.size()  << endl;
        sort(results.begin(), results.end(), resultsSortFunc);
        showResultsRange(results, results.begin(), results.end());          
     }
}
#else      // #if SORT_AND_SHOW
#define SHOW_RESULTS(r) 
#endif     // #if SORT_AND_SHOW


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

    
vector<FaceDetectResult>  
    processOneImage(      DetectorState& dp,
                   // const DrawParams& wp,
                    const ParamRanges& pr) {
    double scale_factor = 1.1;
    int    min_neighbors = 2;
    dp._scale_factor = scale_factor;
    dp._min_neighbors = min_neighbors;
    CroppedFrameList frameList; // = processOneImage2(dp, wp);
    vector<FaceDetectResult> results;
  
    for (min_neighbors = pr._min_neighbors_min; min_neighbors <= pr._min_neighbors_max; min_neighbors += pr._min_neighbors_delta) {
        for (scale_factor = pr._scale_factor_min; scale_factor <= pr._scale_factor_max; scale_factor += pr._scale_factor_delta) {
            dp._scale_factor = scale_factor;
            dp._min_neighbors = min_neighbors;
#if ADAPTIVE_FACE_SEARCH
            processOneImage3(dp);
#else            
            frameList = processOneImage2(dp);
#endif            
            int num_false_positives = frameList.numFalsePositives(); // !@#$ This will be true for the test set of images
            FaceDetectResult r(frameList._frames.size(),  frameList.numWithFaces(), frameList.maxConsecutiveWithFaces(), frameList.middleConsecutiveWithFaces(),
                    scale_factor, min_neighbors, 
                    dp._entry, dp._cascade_name, 
                    frameList.getBestFace(), num_false_positives );
            results.push_back(r);
          
            showOneResultFile(r, cout);
            showOneResultFile(r, pr._output_file);
           
        }
    }
   
    cout << "=========================================================" << endl;
//    computeResultOrder(results, results.begin(), results.end());
    
    SHOW_RESULTS(results);
    return results;
}


vector<FaceDetectResult> 
    detectInOneImage(DetectorState& dp,
               const ParamRanges& pr,
               const FileEntry& entry) {
  //  const int scale = 2;
  //  DrawParams wp;
      
    dp._entry = entry;
   
    IplImage*  image  = cvLoadImage(dp._entry._image_name.c_str());
    if (!image) {
        cerr << "Could not find " << dp._entry._image_name << endl;
        abort();
    }
    IplImage*  image2 = rotateImage(image, entry.getStraighteningAngle(), entry._face_center); 
    IplImage*  image3 = cropImage(image2,  entry.getFaceRect(FACE_CROP_SCALE));  
    dp._current_frame = resizeImage(image3, entry._pad, entry._pad);
  //  dp._gray_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 1);
  //  dp._small_image   = cvCreateImage(cvSize (dp._current_frame->width / scale, dp._current_frame->height / scale), IPL_DEPTH_8U, 1);
    assert (dp._current_frame );

    vector<FaceDetectResult>  results = processOneImage(dp, /* wp,*/ pr) ;
  
    cvReleaseImage(&dp._current_frame); 
    cvReleaseImage(&image2); 
    cvReleaseImage(&image3); 
   // cvReleaseImage(&dp._gray_image);
  //  cvReleaseImage(&dp._small_image);
   // cvReleaseImage(&wp._draw_image);
    
    return results;
}


vector<FaceDetectResult>  main_stuff (const ParamRanges& pr, const string cascadeName)     {
   
    // locate haar cascade from inside application bundle
    // (this is the mac way to package application resources)
    CFBundleRef mainBundle  = CFBundleGetMainBundle ();
    assert (mainBundle);
    
    CFURLRef    cascade_url = CFBundleCopyResourceURL (mainBundle, 
                     CFStringCreateWithCString(NULL, cascadeName.c_str(), kCFStringEncodingASCII),
              //   CFSTR("haarcascade_frontalface_alt2"),  
                  CFSTR("xml"), NULL);
    assert (cascade_url);
    Boolean     got_it      = CFURLGetFileSystemRepresentation (cascade_url, true, 
                                                                reinterpret_cast<UInt8 *>(CASCADE_NAME), CASCADE_NAME_LEN);
    if (! got_it)
        abort ();
    
    DetectorState dp;
   
    // create all necessary instances
    cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
    dp._cascade_name = cascadeName;
    dp._cascade = (CvHaarClassifierCascade*) cvLoad (CASCADE_NAME, 0, 0, 0);
    dp._storage = cvCreateMemStorage(0);
    assert (dp._storage);
    
    // did we load the cascade?!?
    if (!dp._cascade)
        abort ();

    vector<FaceDetectResult>  all_results;
    for (vector<FileEntry>::const_iterator it = pr._file_entries.begin(); it != pr._file_entries.end(); it++) {
        FileEntry e = *it;
        e._image_name = test_file_dir + e._image_name;
        cout << "--------------------- " << e._image_name << " -----------------" << endl;
        vector<FaceDetectResult>  results = detectInOneImage(dp, pr, e) ;   
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    cvReleaseMemStorage(&dp._storage);
    cvFree(&dp._cascade);
    
    return all_results;
}




int main (int argc, char * const argv[]) {
    startup();
    
//    invertMatTest();
    
//    string response;
    
//    cin >> response;
 //   cout << "========= " << response << " ===========" << endl;
    string files_list_name = "files_list_verbose.csv";
    string output_file_name = "results8.csv";
    
    ParamRanges pr;
    pr._min_neighbors_min = 3; // 2;
    pr._min_neighbors_max = 3;
    pr._min_neighbors_delta = 1;
    pr._scale_factor_min = 1.2; // 1.18;  // False positives start below here
    pr._scale_factor_max = 1.23;
    pr._scale_factor_delta = 0.01;
     pr._cascades.push_back("haarcascade_frontalface_alt2");
//    pr._cascades.push_back("haarcascade_frontalface_alt");
//    pr._cascades.push_back("haarcascade_frontalface_alt_tree");
 //   pr._cascades.push_back("haarcascade_frontalface_default");

    vector<FileEntry> file_entries;
#if 0
    FileEntry e[] = {
        {"brad-profile-1.jpg",  0.0},
        {"brad-profile-2.jpg",  0.0},
    //    {"john_in_bed.jpg",     0.0},
   //     {"madeline_shades.jpg", 0.0},
    //    {"madeline_silly.jpg",  0.0},
    } ;
    for (int i = 0; i < sizeof(e)/sizeof(e[0]); i++) {
        file_entries.push_back(e[i]);
    }
#else
     file_entries = readFileListVerbose(test_file_dir + files_list_name) ;
#endif
#if 1
    pr._file_entries = file_entries;
#else
    int num_pads = 3;
    pr._file_entries =  vector<FileEntry> (file_entries.size() * num_pads * 2);
    for (int i = 0; i < file_entries.size(); i++) {
        cout <<  file_entries[i]._image_name << " , " << file_entries[i]._face_angle << endl;
        for (int j = 0; j < num_pads; j++) {
            int k = num_pads*i + j;
            pr._file_entries[2*k] = file_entries[i];
            pr._file_entries[2*k]._pad = j;
            pr._file_entries[2*k+1] = file_entries[i];
            pr._file_entries[2*k+1]._pad = j;
            pr._file_entries[2*k+1]._face_angle = 0.0;
          
           // FileEntry e =  pr._file_entries[num_pads*i + j];
           // FileEntry f =  file_entries[i];
          /// j = j;
        }
    }
#endif
    vector<FaceDetectResult> results, all_results;
    
    pr._output_file.open((test_file_dir + output_file_name).c_str());
    showHeaderFile(cout);
    showHeaderFile(pr._output_file);
    
    for (vector<string>::const_iterator it = pr._cascades.begin(); it != pr._cascades.end(); it++) {
        cout << "--------------------- " << *it << " -----------------" << endl;
        results = main_stuff(pr, *it);
   //    checkResults(results, "results = main_stuff(pr, *it)");
        all_results.insert(all_results.end(), results.begin(), results.end());
  //      checkResults(all_results, "all_results.insert(all_results.end(), results.begin(), results.end())");
        cout << "---------------- all_results --------------" << endl;
        SHOW_RESULTS(all_results);
    }
    
    pr._output_file.close();
    cout << "================ all_results ==============" << endl;
    SHOW_RESULTS(all_results);
    return 0;
}



