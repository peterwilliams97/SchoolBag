
#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>
#include <iomanip>

using namespace std;

namespace  {
    
const char  * WINDOW_NAME  = "Face Tracker with Sub-Frames";
const CFIndex CASCADE_NAME_LEN = 2048;
      char    CASCADE_NAME[CASCADE_NAME_LEN] = "~/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
 
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
    bool doing_fake_crops = false;
    /*
        A cropped rect and list of faces detected in that rect
    */
    struct CroppedFrame {
	CvRect          _rect;
	vector<CvRect>  _faces;
	bool            _fakeFaces;
        
        CroppedFrame() {
            _rect  = EMPTY_RECT;
        }
	
	bool hasFaces() const {
	    if (doing_fake_crops)
		return _fakeFaces;
	    else
		return _faces.size() > 0;
	}
    };
    
    /*
        A rectangle within which a face was found and 
        a list of CroppedFrames whose search rects may be based on the
        face rectangle in some way
    */
    struct CroppedFrameList {
	CvRect _face;		// main face
	std::vector<CroppedFrame> _frames;
        int  numWithFaces() {
            int n = 0;
            for (int i = 0; i < _frames.size(); i++) {
                n += _frames[i].hasFaces() ? 1 : 0;
            }
            return n;
        }
    };
    
      
    
    
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
    
    void drawCropFrames(const DrawParams* wp, const MultiFrameParams* mp, const CroppedFrameList* frameList) {
	CvScalar onColor  = CV_RGB(255,255,255);
	CvScalar offColor = CV_RGB(255,0,0);
	CvScalar mainColor = CV_RGB(0,0,255);
	CvScalar color;
	int      diameter = max(5, (frameList->_face.width + frameList->_face.height)/50);	
	
	drawRect(wp, frameList->_face, mainColor);
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
    
    /*
	Detect faces in dp->_current_frame cropped to rect
	Detect faces in whole image if rect == 0
     */
     CvSeq* detectFacesCrop(const DetectorState* dp, CvRect* rect)    {
	CvSeq* faces = 0;
	try { 
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
	}
	catch (exception& e) {
	    cerr << "OpenCV error: " << e.what() << endl;
	}
	
	return faces;
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
        Create a multi-rect list in a cross shape with x and y arms
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
	croppedFrameList._face = face;
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
        Create a multi-rect list concentric from image rect
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
        croppedFrameList._face = croppedFrameList._frames[0]._rect;
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
	    CvSeq* faces = detectFacesCrop(dp, &rect);
            vector <CvRect> vr(faces != 0 ? faces->total : 0);
            for (int j = 0; j < vr.size(); j++) {
                 vr[j] = *((CvRect*) cvGetSeqElem (faces, j));
            }
            croppedFrameList->_frames[i]._faces = vr;;
    //        cout << "Test " << i << ": " <<   croppedFrameList->_frames[i].hasFaces();
	}
    }
    
    
    CvSeq* detectFaces(DetectorState* dp)    {
	return detectFacesCrop(dp, 0);
    }
    
    bool do_frames = true;

    bool SortFrameByArea(const CroppedFrameList& f1, const CroppedFrameList& f2) {
        return f1._face.width*f1._face.height > f2._face.width*f2._face.height;
    }
    
    /*  
        Detect faces in an image and a set of frames (ROI rects) within that image
    */
    std::vector<CroppedFrameList> detectFaces2(const DetectorState* dp, const MultiFrameParams* mp, int maxFaces)    {
	CvSeq* faces = detectFacesCrop(dp, 0);
	std::vector<CroppedFrameList> faceList(faces ? faces->total : 0);
	for (int i = 0; i < faceList.size(); i++) {
            faceList[i]._face = *((CvRect*) cvGetSeqElem (faces, i));
        }
        std::sort(faceList.begin(), faceList.end(), SortFrameByArea);
        if (maxFaces >= 0 && faceList.size() > maxFaces) 
            faceList.resize(maxFaces);
            
        for (int i = 0; i < faceList.size(); i++) {
	    CvRect face = faceList[i]._face;
            //CroppedFrameList frameList = createMultiFrameList_Cross(mp, face);
            CroppedFrameList frameList = createMultiFrameList_ConcentricImage(dp);
            detectFacesMultiFrame(dp, &frameList);
	    faceList[i] = frameList;
	}
	return faceList;
    }
    
       
    std::vector<CroppedFrameList>  
        processOneImage2(DetectorState& dp,
			 MultiFrameParams& mp,
			 DrawParams& wp) {
#if 0
        CvSeq* faces = detectFaces(&dp);
	
        // draw faces
        cvFlip (dp._current_frame, draw_image, 1);
        for (int i = 0; i < (faces ? faces->total : 0); i++)
        {
            CvRect* r = (CvRect*) cvGetSeqElem (faces, i);
            CvPoint center;
            int radius;
            center.x = cvRound((dp._small_image->width - r->width*0.5 - r->x) *scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            cvCircle (draw_image, center, radius, CV_RGB(0,255,0), 3, 8, 0 );
	}
        // just show the image
	cvShowImage (WINDOW_NAME, wp._draw_image); 
#else
	std::vector<CroppedFrameList> faceList = detectFaces2(&dp, &mp, 1);  
	
	// draw faces
	cvFlip (dp._current_frame, wp._draw_image, 1);
        /*
	for (int i = 0; i < faceList.size(); i++)
	{
	    CvRect* r = &(faceList[i]._face);;
	    CvPoint center;
	    int radius;
	    center.x = cvRound((dp._small_image->width - r->width*0.5 - r->x) * wp._scale);
	    center.y = cvRound((r->y + r->height*0.5) * wp._scale);
	    radius = cvRound((r->width + r->height)*0.25* wp._scale);
	    //cvCircle (draw_image, center, radius, CV_RGB(0,255,0), 3, 8, 0 );
	    
	   CvPoint p1, p2;
	    p1.x = cvRound((double)(dp._small_image->width - r->x) * wp._scale);
	    p1.y = cvRound((double)(r->y) * wp._scale);
	    p2.x = cvRound((double)(dp._small_image->width - (r->x + r->width)) * wp._scale);
	    p2.y = cvRound((double)(r->y + r->height)*wp._scale);
	    cvRectangle(wp._draw_image, p1, p2, CV_RGB(0,0,255), (i==0 ? 3:1));
            drawCropFrames(&wp, &mp, &faceList[i]) ;
        }	
        */
        cvShowImage (WINDOW_NAME, wp._draw_image); 
        return faceList;
        // just show the image
	
#endif
   	
    }
    
    struct FaceDetectResult {
        int     _num_frames_total;
        int     _num_frames_faces;
        int     _ordinal;
        double  _scale_factor;
        int     _min_neighbors;
       string  _image_name;
        FaceDetectResult(int num_faces_total, int num_frames_faces, double scale_factor, int min_neighbors, string image_name) {
            _num_frames_total = num_faces_total;
            _num_frames_faces = num_frames_faces;
            _scale_factor = scale_factor;
            _min_neighbors = min_neighbors;
            _image_name = image_name;
            _ordinal = -1;
        }
        FaceDetectResult& operator=(const FaceDetectResult& r) {
            _num_frames_total = r._num_frames_total;
            _num_frames_faces = r._num_frames_faces;
            _ordinal          = r._ordinal;
            _scale_factor     = r._scale_factor;
            _min_neighbors    = r._min_neighbors;
            _image_name       = r._image_name;
            return *this;
        }
    };
    
    bool resultsOrderSortFunc(const FaceDetectResult r1, const FaceDetectResult& r2) {
        return r1._num_frames_faces > r2._num_frames_faces;
    }
    bool resultsSortFunc(const FaceDetectResult r1, const FaceDetectResult& r2) {
      //  cout << "resultsSortFunc in" << endl;
        int d = 0;
        if (d==0)
            d = r2._ordinal - r1._ordinal;
        if (d==0)
            d = r1._min_neighbors - r2._min_neighbors;
        if (d==0)
            d = (int)(r1._scale_factor - r2._scale_factor);
       if (d==0)
            d = (int)(r1._image_name.compare(r2._image_name) > 0);
        //cout << "resultsSortFunc out" << endl;
        return (d >= 0);
    }
    
    void computeResultOrder(vector<FaceDetectResult>& results) {
        sort(results.begin(), results.end(), resultsOrderSortFunc);
        int ordinal = -1;
        int numFramesFaces = -1;
        for (int i = 0; i < results.size(); i++) {
            if (results[i]._num_frames_faces != numFramesFaces) {
                ++ordinal;
                numFramesFaces = results[i]._num_frames_faces;
            }
            results[i]._ordinal = ordinal;
        }
    }
     
     vector<FaceDetectResult>  processOneImage(DetectorState& dp,
			 MultiFrameParams& mp,
			 DrawParams& wp) {
        double scale_factor = 1.1;
        int    min_neighbors = 2;
        dp._scale_factor = scale_factor;
        dp._min_neighbors = min_neighbors;
        vector<CroppedFrameList> faceList = processOneImage2(dp, mp, wp);
        vector<FaceDetectResult> results;
        int i = 0;
    //   for (min_neighbors = 2; min_neighbors <= 7; min_neighbors++) {
   //         for (scale_factor = 1.01; scale_factor <= 1.5; scale_factor += .005) {
       for (min_neighbors = 2; min_neighbors <= 3; min_neighbors++) {
            for (scale_factor = 1.01; scale_factor <= 1.5; scale_factor += .1) {
            //  cout << "******************* Run " << i+1 << " ******************" << endl;
                dp._scale_factor = scale_factor;
                dp._min_neighbors = min_neighbors;
                faceList = processOneImage2(dp, mp, wp);
              //  FaceDetectResult r(45,  25, scale_factor, min_neighbors, dp._image_name);
             //   results.push_back(r);

                if (faceList.size() > 0) {
                    CroppedFrameList frameList = faceList[0];
                   
                    FaceDetectResult r(frameList._frames.size(),  frameList.numWithFaces(), scale_factor, min_neighbors, dp._image_name);
                    results.push_back(r);
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
        }
        
        cout << "=========================================================" << endl;
        computeResultOrder(results);
        for (vector<FaceDetectResult>::const_iterator it = results.begin(); it != results.end(); it++)
            resultsSortFunc(*it, *it);
        cout << "=========================================================" << endl;

        cout << " sort(results.begin(), results.end(), resultsSortFunc" << endl;
        sort(results.begin(), results.end(), resultsSortFunc);
        for (int i = 0; i < results.size(); i++) {
            FaceDetectResult& r = results[i];
           cout << " " << setw(4) << i 
                << setw(4) << r._num_frames_total
                << setw(4) << r._num_frames_faces
                << setw(5) << setprecision(3) << r._scale_factor 
                << setw(4) << r._min_neighbors
            //    << "   "   << r._image_name
                 << endl;
        }
        return results;
     }


    vector<FaceDetectResult>  main_stuff ()     {
        
        char cwd[200];
        getcwd(cwd, 200);
        cout << "cwd is " << cwd << endl;
        
        const int scale = 2;
        bool do_file = true;
        
        // locate haar cascade from inside application bundle
        // (this is the mac way to package application resources)
        CFBundleRef mainBundle  = CFBundleGetMainBundle ();
        assert (mainBundle);
        CFURLRef    cascade_url = CFBundleCopyResourceURL (mainBundle, CFSTR("haarcascade_frontalface_alt2"), CFSTR("xml"), NULL);
        assert (cascade_url);
        Boolean     got_it      = CFURLGetFileSystemRepresentation (cascade_url, true, 
                                                                    reinterpret_cast<UInt8 *>(CASCADE_NAME), CASCADE_NAME_LEN);
        if (! got_it)
            abort ();
        
        DetectorState dp;
        MultiFrameParams mp;
        DrawParams wp;
         vector<FaceDetectResult>  results;
        // create all necessary instances
        cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
        CvCapture * camera = 0;
        if (!do_file) {
            camera = cvCreateCameraCapture (CV_CAP_ANY);
            // you do own an iSight, don't you ?!?
            if (!camera)
                abort ();
        }
        dp._cascade = (CvHaarClassifierCascade*) cvLoad (CASCADE_NAME, 0, 0, 0);
        dp._storage = cvCreateMemStorage(0);
        assert (dp._storage);

       
        // did we load the cascade?!?
        if (!dp._cascade)
            abort ();

        if (do_file) {
            const char* brad1 = "brad-profile-1.jpg";
            const char* brad2 = "brad-profile-2.jpg";
            const char* john1 = "john_in_bed.jpg";
            const char* fn = brad2;
            dp._image_name = fn;
            dp._current_frame = cvLoadImage(fn);
            if (!dp._current_frame) {
                cerr << "Could not find " << fn << endl;
                abort();
            }
        }
        else {
            // get an initial rect and duplicate it for later work
            dp._current_frame = cvQueryFrame (camera);
        }
        dp._gray_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 1);
        dp._small_image   = cvCreateImage(cvSize (dp._current_frame->width / scale, dp._current_frame->height / scale), IPL_DEPTH_8U, 1);
        wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
        assert (dp._current_frame && dp._gray_image && wp._draw_image);
        
        wp._scale = scale;
        wp._small_image_width = dp._small_image->width;
        
        if (do_file) {
             results = processOneImage(dp, mp, wp) ;
            // wait a tenth of a second for keypress and window drawing
            cvWaitKey (10);
        }
        else {
            // as long as there are images ...
            while (dp._current_frame = cvQueryFrame (camera))
            {     
                results = processOneImage(dp, mp, wp) ;
                // wait a tenth of a second for keypress and window drawing
                int key = cvWaitKey (100);
                if (key == 'q' || key == 'Q')
                    break;
            }
        }

        cvReleaseImage(&dp._current_frame); 
        cvReleaseImage(&dp._gray_image);
        cvReleaseImage(&dp._small_image);
        cvReleaseImage(&wp._draw_image);
        cvReleaseMemStorage(&dp._storage);
        cvFree(&dp._cascade);
        
        return results;
    }

}

int main (int argc, char * const argv[]) {
    vector<FaceDetectResult> results, all_results;
    for (int i = 0; i < 5; i++) {
        results = main_stuff();
       // all_results += results;
    }
    return 0;
}



