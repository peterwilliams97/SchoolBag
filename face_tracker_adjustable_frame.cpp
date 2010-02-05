
#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>

using namespace std;

namespace  {
    
const char  * WINDOW_NAME  = "Face Tracker";
const CFIndex CASCADE_NAME_LEN = 2048;
      char    CASCADE_NAME[CASCADE_NAME_LEN] = "~/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
 
    struct DetectParams {
	IplImage* _current_frame; 
	IplImage* _gray_image; 
	IplImage* _small_image;
	CvHaarClassifierCascade* _cascade;  
	CvMemStorage* _storage;
    };
    
    /*
     Rectangles returned by CvHaar need to be expanded by this amount for face to be detected in the expanded rectangle
     */
    double haar_face_pad_width = 1.4;
    double haar_face_pad_height = 1.6;
    /*
	Frames are rectangles that contain faces
	Outer frame is face rect returned from cvHaar scaled up by _outer_pad_width x _outer_pad_height
	Inner frame is outer frame scaled down by _frac_width x _frac_height
	Number of inner frames checked is based on _num_inc_x and _num_inc_y
     
	!@#$ Need to make inner frame big enough to detect face in

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
    
    bool doing_fake_crops = false;
    struct CroppedFrame {
	CvRect _rect;
	CvSeq* _faces;
	bool   _fakeFaces;
	
	bool hasFaces() const {
	    if (doing_fake_crops)
		return _fakeFaces;
	    else
		return _faces != 0 && _faces->total > 0;
	}
    };
    
    struct CroppedFrameList {
	CvRect _face;		// main face
	double _dx, _dy;	// Distance between successive sub-frames
	std::vector<CroppedFrame> _x;
	std::vector<CroppedFrame> _y;
    };
    
    CroppedFrameList makeFakeFrameList(const MultiFrameParams* mp, CvRect face) {
        CroppedFrameList frameList;
	frameList._face = face;
	bool on = false, middle;
	frameList._x.resize(mp->_num_inc_x);
	for (int i = 0; i < mp->_num_inc_x; i++) {
	    middle = (i == mp->_num_inc_x/2 || i==  mp->_num_inc_x/2 -1);
	    frameList._x[i]._fakeFaces = (on || middle);
	    on = !on;
	}
	on = false;
	frameList._y.resize(mp->_num_inc_y);
	for (int i = 0; i < mp->_num_inc_y; i++) {
	    middle = (i == mp->_num_inc_y/2 || i==  mp->_num_inc_y/2 -1);
	    frameList._y[i]._fakeFaces = (on || middle);
	    on = !on;
	}
        return frameList;	
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
    
    CvPoint getCenter(CvRect r) {
	CvPoint c;
	c.x = r.x + r.width/2;
	c.y = r.y + r.height/2;
	return c;
    }
    
    void drawCropFrames(const DrawParams* wp, const MultiFrameParams* mp, const CroppedFrameList* frameList) {
	
	CvScalar onColor  = CV_RGB(255,255,0);
	CvScalar offColor = CV_RGB(255,0,0);
	CvScalar mainColor = CV_RGB(0,0,255);
	CvScalar color;
		
	CvPoint center = getCenter(frameList->_face);
	
	drawRect(wp, frameList->_face, mainColor);
	// E-W
	CvPoint p1, p2;
	p1.y = p2.y = center.y;
	for (int i = 0; i < frameList->_x.size(); i++) {
	 //   p1.x = center.x - radius + cvRound((double)i/(double)( mp->_num_inc_x) * 2.0 * (double)radius);
	 //   p2.x = center.x - radius + cvRound((double)(i+1)/(double)( mp->_num_inc_x) * 2.0 * (double)radius);
	    CvPoint c = getCenter(frameList->_x[i]._rect);
	    p1.x = c.x - cvRound(frameList->_dy/2.0);
	    p2.x = c.x + cvRound(frameList->_dy/2.0);
	    color = frameList->_x[i].hasFaces() ? onColor : offColor;
	    drawLine(wp, p1, p2, color);
	    drawRect(wp, frameList->_x[i]._rect, color);
	}
	// N-S
	p1.x = p2.x = center.x;
	for (int i = 0; i < mp->_num_inc_y; i++) {
	 //   p1.y = center.y - radius + cvRound((double)i/(double)( mp->_num_inc_y) * 2.0 * (double)radius);
	 //   p2.y = center.y - radius + cvRound((double)(i+1)/(double)( mp->_num_inc_y) * 2.0 * (double)radius);
	    CvPoint c = getCenter(frameList->_y[i]._rect);
	    p1.y = c.y - cvRound(frameList->_dx/2.0);
	    p2.y = c.y + cvRound(frameList->_dx/2.0);
	    color = frameList->_x[i].hasFaces() ? onColor : offColor;;
	    drawLine(wp, p1, p2, color);
	    drawRect(wp, frameList->_y[i]._rect, color);
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
  
    CvSeq* detectFacesCrop(const DetectParams* dp, CvRect* rect)    {
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
						    1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
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
	Given a face rectangle, detect faces with different ROIs around that rectangle
	mp gives search params
        !@#$ refactor the multiple versions of this into a set of component functions
     */
    CroppedFrameList detectFacesMultiFrame(const DetectParams* dp, const MultiFrameParams* mp, CvRect faceIn) {	
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
	
	CvRect rect;
	rect.width = cropWidth;
	rect.height = cropHeight;
	
	CroppedFrameList croppedFrameList;	
	croppedFrameList._face = face;
	croppedFrameList._dx = dx;
	croppedFrameList._dy = dy;
	croppedFrameList._x.resize(mp->_num_inc_x);
	croppedFrameList._y.resize(mp->_num_inc_y);
	
	for (int i = 0;i < mp->_num_inc_x; i++) {
	    rect.x = paddedFace.x + cvRound((double)i * dx);
	    rect.y = cy;
	    CvSeq* faces = detectFacesCrop(dp, &rect);
	    croppedFrameList._x[i]._rect = rect;
	    croppedFrameList._x[i]._faces = faces;
	}
	for (int i = 0; i < mp->_num_inc_y; i++) {
	    rect.x = cx;
	    rect.y = paddedFace.y + cvRound((double)i * dy);
	    CvSeq* faces = detectFacesCrop(dp, &rect);
	    croppedFrameList._y[i]._rect = rect;
	    croppedFrameList._y[i]._faces = faces;
	}
	
	return croppedFrameList;
    }
    
    
    CvSeq* detectFaces(DetectParams* dp)    {
	return detectFacesCrop(dp, 0);
    }
    
    bool do_frames = true;

    bool SortFrameByArea(const CroppedFrameList& f1, const CroppedFrameList& f2) {
        return f1._face.width*f1._face.height > f2._face.width*f2._face.height;
    }
    
    std::vector<CroppedFrameList> detectFaces2(const DetectParams* dp, const MultiFrameParams* mp, int maxFaces)    {
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
            CroppedFrameList frameList;
	    if (do_frames) {
		frameList = detectFacesMultiFrame(dp, mp, face);
	    }
	    else {
		frameList = makeFakeFrameList(mp, face);
	    }
	    faceList[i] = frameList;
	}
	return faceList;
    }
    
    void processOneImage(DetectParams& dp,
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
#else
	std::vector<CroppedFrameList> faceList = detectFaces2(&dp, &mp, 1);  
	
	// draw faces
	cvFlip (dp._current_frame, wp._draw_image, 1);
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
#endif
	// just show the image
	cvShowImage (WINDOW_NAME, wp._draw_image);    	
    }
}




int main (int argc, char * const argv[]) 
{
    
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
    
    DetectParams dp;
    MultiFrameParams mp;
    DrawParams wp;
    
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
	const char* fn = brad1;
	dp._current_frame = cvLoadImage(fn);
	if (!dp._current_frame) {
	    cerr << "Could not find " << fn << endl;
	    abort();
	}
    }
    else {
	// get an initial frame and duplicate it for later work
	dp._current_frame = cvQueryFrame (camera);
    }
    dp._gray_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 1);
    dp._small_image   = cvCreateImage(cvSize (dp._current_frame->width / scale, dp._current_frame->height / scale), IPL_DEPTH_8U, 1);
    wp._draw_image    = cvCreateImage(cvSize (dp._current_frame->width, dp._current_frame->height), IPL_DEPTH_8U, 3);
    assert (dp._current_frame && dp._gray_image && wp._draw_image);
    
    wp._scale = scale;
    wp._small_image_width = dp._small_image->width;
    
    if (do_file) {
	processOneImage(dp, mp, wp) ;
	// wait a tenth of a second for keypress and window drawing
	cvWaitKey (-1);
    }
    else {
	// as long as there are images ...
	while (dp._current_frame = cvQueryFrame (camera))
	{     
	    processOneImage(dp, mp, wp) ;
	    // wait a tenth of a second for keypress and window drawing
	    int key = cvWaitKey (100);
	    if (key == 'q' || key == 'Q')
		break;
	}
    }
    
    // be nice and return no error
    return 0;
}
