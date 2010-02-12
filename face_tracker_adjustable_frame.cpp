
#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace std;

namespace  {
    
const char  * WINDOW_NAME  = "Face Tracker with Sub-Frames";
const CFIndex CASCADE_NAME_LEN = 2048;
      char    CASCADE_NAME[CASCADE_NAME_LEN] = "~/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
      string  test_file_dir = "/Users/user/Desktop/percipo_pics/";

 
    /* 
     *  All the members of DetectorState are needed for a cvHaarDetectObjects() call
     */
    struct DetectorState {
	IplImage*       _current_frame; 
	IplImage*       _gray_image; 
	IplImage*       _small_image;
	CvHaarClassifierCascade* _cascade;  
	CvMemStorage*   _storage;
      // Params  
        double          _scale_factor;  // =1.1, 
        int             _min_neighbors; // =3, 
        string          _image_name;
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
    
    struct DrawParams {
	IplImage* _draw_image;
	double	  _scale;
	int	  _small_image_width;
    };
    
    CvRect EMPTY_RECT = { 0, 0, 0, 0 };
    /*
     *   A cropped rect and list of faces detected in that rect
     */
    struct CroppedFrame {
	CvRect          _rect;
	vector<CvRect>  _faces;
        CroppedFrame() {   _rect  = EMPTY_RECT;   }
	bool hasFaces() const { return _faces.size() > 0; }
        int  numFalsePositives() const { return max(0, (int)_faces.size() - 1); } // !@#$ Based on set of test images with 1 face each !!!
    };
    
    /*
        A rectangle within which a face was found and 
        a list of CroppedFrames whose search rects may be based on the
        face rectangle in some way
    */
    struct CroppedFrameList {
	CvRect _primary;		// Starting rectangle main face
	std::vector<CroppedFrame> _frames;
        int  numWithFaces() const {
            int n = 0;
            for (int i = 0; i < _frames.size(); i++) {
                n += _frames[i].hasFaces() ? 1 : 0;
            }
            return n;
        }
        int maxConsecutiveWithFaces() const {
            int max_consecutive = 0;
            int n = 0;
            for (int i = 0; i < _frames.size(); i++) {
                n = _frames[i].hasFaces() ? n+1 : 0;
                max_consecutive = max(max_consecutive, n);
            }
            return max_consecutive;
        }
        int numFalsePositives() const {
            int n = 0;
            for (int i = 0; i < _frames.size(); i++) {
                n += _frames[i].numFalsePositives();
            }
            return n; 
        }
        // !@#$
        CvRect getBestFace() {
            CvRect r = EMPTY_RECT;
            int x = 0, y = 0, w = 0, h = 0;
            int n = 0;
            for (int i = 0; i < _frames.size(); i++) {
                if (_frames[i].hasFaces()) {
                    r = _frames[i]._faces[0];
                    x += r.x + r.width/2;
                    y += r.y + r.height/2;
                    w += r.width;
                    h += r.height;
                    ++n;
                }
            }
            if (n > 0) {
                x /= n;
                y /= n;
                w /= n;
                h /= n;
                r.x = x - w/2;
                r.y = y - h/2;
                r.width = w;
                r.height = h;
            }
            return r;
        }
    };
        
          
    string rectAsString(CvRect r) {
        ostringstream s;
        s << "[" << setw(3) << r.x << "," << setw(3) << r.y << "," << setw(3) << r.width << "," << setw(3)  << r.height << "]";
        return s.str();
    }

    
    int scaleX(const DrawParams* wp, int x) {
	return cvRound((double)(wp->_small_image_width - x) * wp->_scale);
    }
    
    int scaleY(const DrawParams* wp, int y) {
	return cvRound((double)(y) * wp->_scale);
    }
    
    void drawRect(const DrawParams* wp, CvRect r, CvScalar color) {
	CvPoint p1, p2;
	p1.x = scaleX(wp, r.x);
	p1.y = scaleY(wp, r.y);
	p2.x = scaleX(wp, r.x + r.width);
	p2.y = scaleY(wp, r.y + r.height);
	cvRectangle(wp->_draw_image, p1, p2, color);
    }
    
    void drawLine(const DrawParams* wp, CvPoint p1in, CvPoint p2in, CvScalar color) {
	CvPoint p1, p2;
	p1.x = scaleX(wp, p1in.x);
	p1.y = scaleY(wp, p1in.y);
	p2.x = scaleX(wp, p2in.x);
	p2.y = scaleY(wp, p2in.y);
	cvLine(wp->_draw_image, p1, p2, color, 3, 8, 0);
    }
      
    void drawMarker(const DrawParams* wp, CvPoint center, int diameter, CvScalar color) {
	CvRect r;
        r.x = center.x - diameter/2;
        r.y = center.y - diameter/2;
        r.width = diameter;
        r.height = diameter;
        drawRect(wp , r, color);
    }

    
    CvPoint getCenter(CvRect r) {
	CvPoint c;
	c.x = r.x + r.width/2;
	c.y = r.y + r.height/2;
	return c;
    }
    
    /*
     * Draw a set of crop frames onto an image
     */
    void drawCropFrames(const DrawParams* wp, const MultiFrameParams* mp, const CroppedFrameList* frameList) {
	CvScalar onColor  = CV_RGB(255,255,255);
	CvScalar offColor = CV_RGB(255,0,0);
	CvScalar mainColor = CV_RGB(0,0,255);
	CvScalar color;
//	int      diameter = max(5, (frameList->_face.width + frameList->_face.height)/50);	
	
	drawRect(wp, frameList->_primary, mainColor);
       // drawMarker(wp, getCenter(frameList->_face), diameter, mainColor);
	
	for (int i = 0; i < frameList->_frames.size(); i++) {
            CvRect r = frameList->_frames[i]._rect;
	    CvPoint c = getCenter(r);
            color = frameList->_frames[i].hasFaces() ? onColor : offColor;
            drawRect(wp, r, color);
           // drawMarker(wp, getCenter(r), diameter, color);  
	}
   }
    
    IplImage* cropImage(IplImage* image, CvRect* rect) {
	IplImage * cropped  = cvCreateImage (cvSize(rect->width, rect->height), IPL_DEPTH_8U, 3);
	cvSetImageROI(image, *rect); 
	cvResize (image, cropped, CV_INTER_NN);
	return cropped;
    }
    
    
    bool SortFacesByArea(const CvRect& r1, const CvRect& r2) {
        return r1.width*r1.height > r2.width*r2.height;
    }
    
    /*
     *  Detects faces in dp->_current_frame cropped to rect
     *  Detects faces in whole image if rect == 0
     *  Returns list of face rectangles sorted by size
     */
     vector<CvRect> detectFacesCrop(const DetectorState* dp, CvRect* rect)    {
	CvSeq* faces = 0;
	//try { 
	    if (rect) {
		cvSetImageROI(dp->_current_frame, *rect);
		cvSetImageROI(dp->_gray_image, *rect);
	    }
	    // convert to gray and downsize
	    cvCvtColor (dp->_current_frame, dp->_gray_image, CV_BGR2GRAY);
	    cvResize (dp->_gray_image,dp->_small_image, CV_INTER_LINEAR);
	    
	    // detect faces
	    faces = cvHaarDetectObjects (dp->_small_image, dp->_cascade, dp->_storage,
                                        dp->_scale_factor, dp->_min_neighbors, 
                                        //    1.1, 2, 
                                                    CV_HAAR_DO_CANNY_PRUNING,
						cvSize (30, 30));
	    if (rect) {
		cvResetImageROI(dp->_current_frame);
		cvResetImageROI(dp->_gray_image);
	    }
	//}
	//catch (exception& e) {
	//    cerr << "OpenCV error: " << e.what() << endl;
	//}
        
	vector <CvRect> face_list(faces != 0 ? faces->total : 0);
        for (int j = 0; j < face_list.size(); j++) {
            face_list[j] = *((CvRect*) cvGetSeqElem (faces, j));
        }
        sort(face_list.begin(), face_list.end(), SortFacesByArea);
	return face_list;
    }
   
    
    /*
	Pad out a face rectangle to create a search rectangle
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
     CroppedFrameList createMultiFrameList_ConcentricImage(const DetectorState* dp) {
        double innerFrameFrac = 0.9;
        int    numFrames = 40;
        CroppedFrameList croppedFrameList;
        croppedFrameList._frames.resize(numFrames);
        int imageWidth  = dp->_small_image->width; //dp->_current_frame->width;
        int imageHeight = dp->_small_image->height;//dp->_current_frame->height;
        CvRect rect;
        for (int i = 0; i < numFrames; i++) { 
            rect.x = cvRound((1.0 - innerFrameFrac)*imageWidth *(double)i/(double)(numFrames - 1));
            rect.y = cvRound((1.0 - innerFrameFrac)*imageHeight*(double)i/(double)(numFrames - 1));
            rect.width  = imageWidth  - 2 * rect.x;
            rect.height = imageHeight - 2 * rect.y;
            croppedFrameList._frames[i]._rect = rect;
      //      cout << "rect[" << i << "] = " << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
        }
        croppedFrameList._primary = croppedFrameList._frames[0]._rect;
        return croppedFrameList;
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
        CroppedFrameList frameList = createMultiFrameList_ConcentricImage(dp);
        detectFacesMultiFrame(dp, &frameList);
	return frameList;
    }

 
    /*
     *  Process (detect faces) in set of frames within an image with a particular
     *  combination of settings (in dp)
     *  Returns a list of results, one for each frame
     */
       
    CroppedFrameList  
        processOneImage2(const DetectorState& dp,
			 const DrawParams& wp) {

	CroppedFrameList frameList = detectFaces2(&dp);  
	
	// draw faces
	cvFlip (dp._current_frame, wp._draw_image, 1);
        cvShowImage (WINDOW_NAME, wp._draw_image); 
        
        return frameList;
    }
    
    struct FaceDetectResult {
        int     _num_frames_total;
        int     _num_frames_faces;
        int     _max_consecutive_faces;
        int     _ordinal;
        double  _scale_factor;
        int     _min_neighbors;
        string  _image_name;
        string  _cascade_name;
        CvRect  _face_rect;
        int     _num_false_positives;
  //      mutable int     _dbg_index;
        
        FaceDetectResult() {
     //       cout << "FaceDetectResult::FaceDetectResult()" << endl;
            _image_name = _cascade_name = "";
   //         _dbg_index  = -1;
        }
        FaceDetectResult(int num_faces_total, int num_frames_faces, int max_consecutive_faces, 
                        double scale_factor, int min_neighbors, 
                            string image_name, string cascade_name, 
                            CvRect face_rect, int num_false_positives) {
            _num_frames_total = num_faces_total;
            _num_frames_faces = num_frames_faces;
            _max_consecutive_faces = max_consecutive_faces;
            _scale_factor = scale_factor;
            _min_neighbors = min_neighbors;
            _image_name = image_name;
            _cascade_name = cascade_name;
            _ordinal = -1;
            _face_rect = face_rect;
            _num_false_positives = num_false_positives;
  //           _dbg_index  = -2;
        }
        FaceDetectResult& assign(const FaceDetectResult& r)  {
      //      if (r._dbg_index == 0) {
      //          cerr << "Bogus FaceDetectResult record" << endl;
      //      }
            _num_frames_total = r._num_frames_total;
            _num_frames_faces = r._num_frames_faces;
            _max_consecutive_faces = r._max_consecutive_faces;
            _ordinal          = r._ordinal;
            _scale_factor     = r._scale_factor;
            _min_neighbors    = r._min_neighbors;
            _image_name       = r._image_name;
            _cascade_name     = r._cascade_name;
            _face_rect        = r._face_rect;
            _num_false_positives = r._num_false_positives;
      //      _dbg_index          = r._dbg_index;
            return *this;
        }

        FaceDetectResult& operator=(const FaceDetectResult& r)  {
    //        cout << "operator=() from: " << r._dbg_index << " to:" << _dbg_index << endl;
            return assign(r);
        }
        FaceDetectResult(const FaceDetectResult& r) {
    //        cout << "copy ctor " << r._dbg_index << endl;
            assign(r);
        }
    };
    
    /*
    void checkResults(const vector<FaceDetectResult> results, const string title) {
        vector<FaceDetectResult>::const_iterator rit;
        int i = 0;
        cout << "---------------- checkResults() !@#$ " << title << ": " << results.size() << " --------------" << endl;
        for (rit = results.begin(); rit != results.end(); rit++) {
            ++i;
            cout << i << ": " << (*rit)._image_name << ", ";
            (*rit)._dbg_index = i;
        }
        cout << endl;
    }
    */
    
    bool resultsSortFuncImageName(const FaceDetectResult& r1, const FaceDetectResult& r2) {
        return (r1._image_name.compare(r2._image_name) > 0);
    }
    bool resultsSortFuncOrder(const FaceDetectResult& r1, const FaceDetectResult& r2) {
        return r1._num_frames_faces > r2._num_frames_faces;
    }
    bool resultsSortFunc(const FaceDetectResult& r1, const FaceDetectResult& r2) {
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
            d = r1._image_name.compare(r2._image_name);
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
    void  showOneResult(const FaceDetectResult& r) { 
        cout << setw(4) << r._num_frames_total
            << setw(4) << r._ordinal
            << setw(4) << r._num_frames_faces
            
             << "  "  << setw(4) << r._num_false_positives
            << "  "  << setw(10) << rectAsString(r._face_rect)

            << setw(5) << setprecision(3) << r._scale_factor 
            << setw(4) << r._min_neighbors
            << " "   << setw(20) << r._image_name
            << " "   << setw(32) << r._cascade_name
           
             << endl;
    }
    void  showOneResultFile(const FaceDetectResult& r, ofstream& of) { 
        const string sep = ", ";
       
        of << setw(4) << r._num_frames_total
            <<  sep << setw(4) << r._num_frames_faces
            <<  sep << setw(4) << r._max_consecutive_faces
            <<  sep << setw(4) << r._num_false_positives
            <<  sep << setw(4) << r._face_rect.x
            <<  sep << setw(4) << r._face_rect.y
            <<  sep << setw(4) << r._face_rect.width
            <<  sep << setw(4) << r._face_rect.height
            <<  sep << setw(5) << setprecision(3) << r._scale_factor 
            <<  sep << setw(4) << r._min_neighbors
            <<  sep << setw(20) << r._image_name
            <<  sep << setw(32) << r._cascade_name
            << endl;
    }
    void  showHeaderFile(ofstream& of) { 
        const string sep = ", ";
       
        of << setw(4) << "frames"
            <<  sep << setw(4) << "with faces"
            <<  sep << setw(4) << "consecutive faces"
            <<  sep << setw(4) << "false +ves"
            <<  sep << setw(4) << "x"
            <<  sep << setw(4) << "y"
            <<  sep << setw(4) << "width"
            <<  sep << setw(4) << "height"
            <<  sep << setw(5) << "scale_factor" 
            <<  sep << setw(4) << "min_neighbors"
            <<  sep << setw(20) << "image_name"
            <<  sep << setw(32) << "cascade_name"
            << endl;
    }

    void showResultsRange(vector<FaceDetectResult>& results,
                            vector<FaceDetectResult>::iterator it0,
                            vector<FaceDetectResult>::iterator it1) {
        vector<FaceDetectResult>::iterator it;
        for (it = it0; it != it1; it++) {
            showOneResult(*it);
        }
    }
                            

    void showResults(vector<FaceDetectResult> results) {
        cout << "======================== showResults " << results.size() << " =================================" << endl;
        if (results.size() > 0 ) {

        // Compute ordinals for each image
            sort(results.begin(), results.end(), resultsSortFuncImageName);
            vector<FaceDetectResult>::iterator it, it0 = results.begin();
        
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
            
         //   checkResults(results, "showResults");
            cout << " sort(results.begin(), results.end(), resultsSortFunc" << endl;
            sort(results.begin(), results.end(), resultsSortFunc);
            showResultsRange(results, results.begin(), results.end());          
         }
    }
    
    /*
     * File to be processed
     */
    struct FileEntry {
        string _image_name;
        double _rotation;
    };
    
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
                        const DrawParams& wp,
                        const ParamRanges& pr) {
        double scale_factor = 1.1;
        int    min_neighbors = 2;
        dp._scale_factor = scale_factor;
        dp._min_neighbors = min_neighbors;
        CroppedFrameList frameList; // = processOneImage2(dp, wp);
        vector<FaceDetectResult> results;
        int i = 0;
 
        for (min_neighbors = pr._min_neighbors_min; min_neighbors <= pr._min_neighbors_max; min_neighbors += pr._min_neighbors_delta) {
            for (scale_factor = pr._scale_factor_min; scale_factor <= pr._scale_factor_max; scale_factor += pr._scale_factor_delta) {
            //  cout << "******************* Run " << i+1 << " ******************" << endl;
                dp._scale_factor = scale_factor;
                dp._min_neighbors = min_neighbors;
                frameList = processOneImage2(dp, wp);
                int num_false_positives = frameList.numFalsePositives(); // !@#$ This will be true for the test set of images
                FaceDetectResult r(frameList._frames.size(),  frameList.numWithFaces(), frameList.maxConsecutiveWithFaces(),
                        scale_factor, min_neighbors, 
                        dp._image_name, dp._cascade_name, frameList.getBestFace(), num_false_positives );
                results.push_back(r);
                showOneResultFile(r, pr._output_file);
                pr.flushIfNecessary();
       
               //     cout << "Num frames = " << setw(4)  << frameList._frames.size() << endl;
                cout << " " << setw(4) << i 
                        << setw(5) << setprecision(3) << scale_factor 
                        << setw(4) << min_neighbors
                        << setw(4) << frameList.numWithFaces()
                        << setw(4)  << frameList._frames.size()
                        << endl;
                }
            //   cout << "$$$$$$$$$$$$$$$$$$$ Run " << i+1 << " $$$$$$$$$$$$$$$$$$" << endl;
                ++i;
            }
       
        cout << "=========================================================" << endl;
    //    computeResultOrder(results, results.begin(), results.end());
        
        showResults(results);
        return results;
    }
    
    vector<FaceDetectResult> 
        detectInOneImage(DetectorState& dp,
                   const ParamRanges& pr,
                   const FileEntry& entry) {
        const int scale = 2;
         
        DrawParams wp;
      
        dp._image_name = entry._image_name;
        dp._current_frame = cvLoadImage(dp._image_name.c_str());
        if (!dp._current_frame) {
            cerr << "Could not find " << dp._image_name << endl;
            abort();
        }
        dp._gray_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 1);
        dp._small_image   = cvCreateImage(cvSize (dp._current_frame->width / scale, dp._current_frame->height / scale), IPL_DEPTH_8U, 1);
        wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
        assert (dp._current_frame && dp._gray_image && wp._draw_image);
        
        wp._scale = scale;
        wp._small_image_width = dp._small_image->width;
        
        vector<FaceDetectResult>  results = processOneImage(dp, wp, pr) ;
            // wait a tenth of a second for keypress and window drawing
        cvWaitKey (10);
       
        cvReleaseImage(&dp._current_frame); 
        cvReleaseImage(&dp._gray_image);
        cvReleaseImage(&dp._small_image);
        cvReleaseImage(&wp._draw_image);
        
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
            cout << "--------------------- " << (*it)._image_name << " -----------------" << endl;
            vector<FaceDetectResult>  results = detectInOneImage(dp, pr, *it) ;   
            all_results.insert(all_results.end(), results.begin(), results.end());
        }
        
        cvReleaseMemStorage(&dp._storage);
        cvFree(&dp._cascade);
        
        return all_results;
    }
    
    void startup() {
        char cwd[200];
        getcwd(cwd, 200);
        cout << "cwd is " << cwd << endl;
    }

    string trim(const string& in) {
        string out = "";
        string whitespace = " \t\r\n";
        int p2 = in.find_last_not_of(whitespace);
        if (p2 != string::npos) {
            int p1 = in.find_first_not_of(whitespace);
            if (p1 == string::npos) p1 = 0;
            out = in.substr(p1, (p2-p1)+1);
        }
        return out;
    }
    
    /*
     *  Read a list of files and settings. Comma separated. One file per line.
     *
     */
     vector<FileEntry> readFileList(const string conf_file_name) {
        ifstream input_file;
        string file_path = test_file_dir + conf_file_name;
        input_file.open(file_path.c_str(), fstream::in);
        if (!input_file.is_open()) {
            cerr << "Could not open " << conf_file_name << endl;
            exit(1);
        }
        vector<FileEntry> file_entries;
        int n = 1;
        string line, delimiters = ",";
        while (getline(input_file, line)) {
            FileEntry entry;
            line = trim(line);
            if (line.size() == 0)
                break;
            string::size_type last_pos = 0;
            string::size_type pos = line.find_first_of(delimiters, last_pos);
            if (pos == string::npos || last_pos == string::npos) {
                cerr << "Bad line " << n << " in " << conf_file_name << endl;
                exit(2);
            }
            entry._image_name = test_file_dir + trim(line.substr(last_pos, pos - last_pos));
            pos = line.find_first_of(delimiters, last_pos);
            if (pos == string::npos || last_pos == string::npos) {
                cerr << "Bad line " << n << " in " << conf_file_name << endl;
                exit(3);
            }
            entry._rotation = atof(trim(line.substr(last_pos, pos - last_pos)).c_str());
            file_entries.push_back(entry);
        }
        input_file.close();
        return file_entries;
    }
    
    
}



int main (int argc, char * const argv[]) {
    startup();
    
//    string response;
    
//    cin >> response;
 //   cout << "========= " << response << " ===========" << endl;
    ParamRanges pr;
    pr._min_neighbors_min = 0;
    pr._min_neighbors_max = 2;
    pr._min_neighbors_delta = 1;
    pr._scale_factor_min = 1.05;
    pr._scale_factor_max = 1.3;
    pr._scale_factor_delta = 0.05;
    pr._cascades.push_back("haarcascade_frontalface_alt");
    pr._cascades.push_back("haarcascade_frontalface_alt2");
    pr._cascades.push_back("haarcascade_frontalface_alt_tree");
    pr._cascades.push_back("haarcascade_frontalface_default");

#if 1
    FileEntry e[] = {
        {"brad-profile-1.jpg",  0.0},
        {"brad-profile-2.jpg",  0.0},
    //    {"john_in_bed.jpg",     0.0},
   //     {"madeline_shades.jpg", 0.0},
    //    {"madeline_silly.jpg",  0.0},
    } ;
    for (int i = 0; i < sizeof(e)/sizeof(e[0]); i++) {
        pr._file_entries.push_back(e[i]);
    }
#else
     pr._file_entries = readFileList("files_list.csv") ;
#endif

    vector<FaceDetectResult> results, all_results;
    
    pr._output_file.open("results.csv");
    showHeaderFile(pr._output_file);
    
    for (vector<string>::const_iterator it = pr._cascades.begin(); it != pr._cascades.end(); it++) {
        cout << "--------------------- " << *it << " -----------------" << endl;
        results = main_stuff(pr, *it);
   //    checkResults(results, "results = main_stuff(pr, *it)");
        all_results.insert(all_results.end(), results.begin(), results.end());
  //      checkResults(all_results, "all_results.insert(all_results.end(), results.begin(), results.end())");
        cout << "---------------- all_results --------------" << endl;
        showResults(all_results);
    }
    
    pr._output_file.close();
    cout << "================ all_results ==============" << endl;
    showResults(all_results);
    return 0;
}



