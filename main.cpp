/*
 *  main.cpp
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */
#if 0
----------------
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <list>
#include <vector>
#include "config.h"
#include "core_common.h"
#include "face_util.h"
#include "face_io.h"
#include "face_draw.h"
#include "face_calc.h"
#include "face_results.h"
#include "cropped_frames.h"

using namespace std;

#if TEST_MANY_SETTINGS
    
vector<FaceDetectResult>  
    processOneImage(DetectorState& dp,
                    const ParamRanges& pr) {
    double scale_factor = 1.1;
    int    min_neighbors = 2;
    dp._scale_factor = scale_factor;
    dp._min_neighbors = min_neighbors;
 
#if ADAPTIVE_FACE_SEARCH
    CroppedFrameList_Adaptive frame_list;
#else
    CroppedFrameList_Histogram frame_list;
#endif
    vector<FaceDetectResult> results;
  
    for (min_neighbors = pr._min_neighbors_min; min_neighbors <= pr._min_neighbors_max; min_neighbors += pr._min_neighbors_delta) {
        for (scale_factor = pr._scale_factor_min; scale_factor <= pr._scale_factor_max; scale_factor += pr._scale_factor_delta) {
            dp._scale_factor = scale_factor;
            dp._min_neighbors = min_neighbors;
#if ADAPTIVE_FACE_SEARCH
            frame_list = processOneImage_Adaptive(dp);
#else            
            frame_list = processOneImage_Histogram(dp);
#endif  
            CvRect best_face_orig_coords = offsetRectByRect(frame_list.getBestFace(), dp._entry.getFaceRect(dp._face_crop_ratio));
           
#if RESULTS_VERSION == 1
            int num_false_positives = frame_list.numFalsePositives(); // !@#$ This will be true for the test set of images           
            FaceDetectResult r(frame_list._frames.size(),  frame_list.numWithFaces(), frame_list.maxConsecutiveWithFaces(), frame_list.getBestFace(),
                    scale_factor, min_neighbors, 
                    dp._entry, dp._cascade_name, 
                    best_face_orig_coords, num_false_positives );
#elif RESULTS_VERSION == 2
            FaceDetectResult r(dp._entry, dp._cascade_name, best_face_orig_coords);
#endif
            results.push_back(r);
          
            showOneResultFile(r, cout);
            showOneResultFile(r, pr._output_file);
#if DRAW_FACES            
            drawResultImage(r);
#endif            
        }
    }
#if VERBOSE   
    cout << "=========================================================" << endl;
#endif
    SHOW_RESULTS(results);
    return results;
}




vector<FaceDetectResult> 
    detectInOneImage(DetectorState& dp,
               const ParamRanges& pr,
               const FileEntry& entry) {
    dp._entry = entry;
   
    IplImage*  image  = cvLoadImage(dp._entry._image_name.c_str());
    if (!image) {
        cerr << "Could not find '" << dp._entry._image_name << "'" << endl;
        abort();
    }
    IplImage*  scaled_image = scaleImage640x480(image);
    IplImage*  image2 = rotateImage(scaled_image, entry.getStraighteningAngle(), entry._face_center); 
    CvRect face_rect =  entry.getFaceRect(1.0);
    dp._face_crop_ratio = calcCropRatio(image, face_rect, MIN_CROP_WIDTH, FACE_CROP_RATIO);
    CvRect crop_rect =  entry.getFaceRect(dp._face_crop_ratio);
    cout << "crop_rect = " << rectAsString(crop_rect)<< endl;
    dp._current_frame = cropImage(image2,  crop_rect);  
    assert (dp._current_frame );

    vector<FaceDetectResult>  results = processOneImage(dp, pr) ;
  
    cvReleaseImage(&dp._current_frame); 
    cvReleaseImage(&scaled_image);
    cvReleaseImage(&image2);    
    return results;
}



vector<FaceDetectResult>  main_stuff (const ParamRanges& pr, const string cascade_name)     {
/* 
#if MAC_APP
    CFBundleRef mainBundle  = CFBundleGetMainBundle ();
    assert (mainBundle);
    
    CFURLRef  cascade_url = CFBundleCopyResourceURL (mainBundle, 
                     CFStringCreateWithCString(NULL, cascade_name.c_str(), kCFStringEncodingASCII),
                    CFSTR("xml"), NULL);
    assert (cascade_url);
    Boolean     got_it      = CFURLGetFileSystemRepresentation (cascade_url, true, 
                                                                reinterpret_cast<UInt8 *>(CASCADE_NAME), CASCADE_NAME_LEN);
    if (! got_it)
        abort ();
    string cascade_path = CASCADE_NAME;
#else
    string cascade_path = cascade_name + ".xml";
#endif    
*/
    
    DetectorState dp;
 
    dp._cascade_name = cascade_name;
    const string cascade_path = getCascadePath(cascade_name);
    dp._cascade = (CvHaarClassifierCascade*) cvLoad (cascade_path.c_str(), 0, 0, 0);
    if (!dp._cascade) {
        cerr << "Could not load cascade '" << cascade_path << "'" << endl;
        abort(); 
    }
    dp._storage = cvCreateMemStorage(0);
    dp._face_crop_ratio = FACE_CROP_RATIO;
    assert (dp._storage);
   
#if DRAW_FACES   
    // create all necessary instances
    cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
#endif 
    
    

    vector<FaceDetectResult>  all_results;
    for (vector<FileEntry>::const_iterator it = pr._file_entries.begin(); it != pr._file_entries.end(); it++) {
        FileEntry e = *it;
    
#if VERBOSE        
        cout << "--------------------- " << e._image_name << " -----------------" << endl;
#endif        
        vector<FaceDetectResult>  results = detectInOneImage(dp, pr, e) ;   
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    cvReleaseMemStorage(&dp._storage);
    cvFree(&dp._cascade);
    
    return all_results;
}



int main (int argc, char * const argv[]) {
    startup();

    string test_file_dir = "/Users/user/Desktop/percipo_pics/";
    string files_list_name = "files_list_verbose.csv";
    string output_file_name = "results_" + test_type_name + "_" + intToStr(ADAPTIVE_NUM_STEPS) + "_" + doubleToStr(HAAR_SCALE_FACTOR) + ".csv";    
    
    ParamRanges pr;
    pr._min_neighbors_min = 3; // 2;
    pr._min_neighbors_max = 3;
    pr._min_neighbors_delta = 1;
    pr._scale_factor_min = 1.2; // 1.18;  // False positives start below here
    pr._scale_factor_max = 1.2;
    pr._scale_factor_delta = .1;
     pr._cascades.push_back("haarcascade_frontalface_alt2");
//    pr._cascades.push_back("haarcascade_frontalface_alt");
//    pr._cascades.push_back("haarcascade_frontalface_alt_tree");
 //   pr._cascades.push_back("haarcascade_frontalface_default");

 
    vector<FileEntry> file_entries = readFileListVerbose(test_file_dir + files_list_name, test_file_dir) ;
    pr._file_entries = file_entries;
    vector<FaceDetectResult> results, all_results;
    
    pr._output_file.open((test_file_dir + output_file_name).c_str());
    showHeaderFile(cout);
    showHeaderFile(pr._output_file);
    
    for (vector<string>::const_iterator it = pr._cascades.begin(); it != pr._cascades.end(); it++) {
        cout << "--------------------- " << *it << " -----------------" << endl;
        results = main_stuff(pr, *it);
        all_results.insert(all_results.end(), results.begin(), results.end());
        cout << "---------------- all_results --------------" << endl;
        SHOW_RESULTS(all_results);
    }
    
    pr._output_file.close();
    cout << "================ all_results ==============" << endl;
    SHOW_RESULTS(all_results);
    return 0;
}

#else // #if TEST_MANY_SETTINGS

FaceDetectResult processOneImage(DetectorState& dp)  {
#if ADAPTIVE_FACE_SEARCH
    CroppedFrameList_Adaptive  frame_list = processOneImage_Adaptive(dp);
#else
    CroppedFrameList_Histogram frame_list = processOneImage_Histogram(dp);
#endif
    CvRect best_face_orig_coords = offsetRectByRect(frame_list.getBestFace(), dp._entry.getFaceRect(dp._face_crop_ratio));
    FaceDetectResult r(dp._entry, dp._cascade_name, best_face_orig_coords);
    showOneResultFile(r, cout);
   // showOneResultFile(r, pr._output_file);
    DRAW_RESULT_IMAGE(r);
    return r;
}

FaceDetectResult detectInOneImage(DetectorState& dp,
                               //   const ParamRanges& pr,
                                   FileEntry& entry) {
    dp._entry = entry;
   
    IplImage*  image  = cvLoadImage(dp._entry._image_name.c_str());
    if (!image) {
        cerr << "Could not find '" << dp._entry._image_name << "'" << endl;
        abort();
    }
    if (entry._face_radius == 0) {
        CvRect face = cvRect(0, 0, image->width, image->height);
        entry._face_radius = getRadius(face);
        entry._face_center = getCenter(face);
    }

    IplImage*  scaled_image = scaleImage640x480(image);
    IplImage*  image2 = rotateImage(scaled_image, entry.getStraighteningAngle(), entry._face_center); 
    CvRect face_rect =  entry.getFaceRect(1.0);
    dp._face_crop_ratio = calcCropRatio(image, face_rect, MIN_CROP_WIDTH, FACE_CROP_RATIO);
    CvRect crop_rect =  entry.getFaceRect(dp._face_crop_ratio);
    cout << "crop_rect = " << rectAsString(crop_rect)<< endl;
    dp._current_frame = cropImage(image2,  crop_rect);  
    assert (dp._current_frame );

    FaceDetectResult  result = processOneImage(dp) ;
  
    cvReleaseImage(&dp._current_frame); 
    cvReleaseImage(&scaled_image);
    cvReleaseImage(&image2);    
    return result;
}

FaceDetectResult peterFramingFilter(FileEntry& entry)     {
    const string cascade_name = "haarcascade_frontalface_alt2";
   /* 
    CFBundleRef mainBundle  = CFBundleGetMainBundle ();
    assert (mainBundle);
    
    CFURLRef cascade_url = CFBundleCopyResourceURL (mainBundle, 
                     CFStringCreateWithCString(NULL, cascade_name.c_str(), kCFStringEncodingASCII),
                    CFSTR("xml"), NULL);
    assert (cascade_url);
    Boolean  got_it  = CFURLGetFileSystemRepresentation (cascade_url, true, 
                                                                reinterpret_cast<UInt8 *>(CASCADE_NAME), CASCADE_NAME_LEN);
    if (! got_it)
        abort ();
   */ 
    
      
#if DRAW_FACES   
    // create all necessary instances
    cvNamedWindow (WINDOW_NAME, CV_WINDOW_AUTOSIZE);
#endif    

    DetectorState dp;
    dp._cascade_name = cascade_name;
    const string cascade_path = getCascadePath(cascade_name);
    dp._cascade = (CvHaarClassifierCascade*) cvLoad (cascade_path.c_str(), 0, 0, 0);
    if (!dp._cascade) {
        cerr << "Could not load cascade '" << cascade_path << "'" << endl;
        abort(); 
    }
    dp._storage = cvCreateMemStorage(0);
    assert (dp._storage);
    
    dp._face_crop_ratio = FACE_CROP_RATIO;
    FaceDetectResult result = detectInOneImage(dp, entry) ;   
    
    cvReleaseMemStorage(&dp._storage);
    cvFree(&dp._cascade);
    return result;
}

/*
I'm not sure it makes sense to pass the face detection parameters as
inputs to your framing code -- because this detection will be only
based on one detection (not very accurate).  The input image will be
of a single face, roughly centered, and already uprighted (rotated).

So, I imagine that the input would be a single image of a single face,
uprighted, with some generous padding around the face.

The output could be a new image, or the face center and radius,
either way is fine with me.

If you're using opencv, the image type is "IplImage *"
It will probably be better to use greyscale images, so that we
dont have to convert back and forth :
 gray = cvCreateImage( cvSize(width,height), 8, 1 ) ;

For the first go, let's have your framing code be a standalone
app that takes a jpeg filename on the command line, and generates
a new file, like this :

int main(int argc, char **argv)
{
 IplImage *image = NULL ;
 IplImage *new_image = NULL ;
 char new_filename[8192] ;

 image = cvLoadImage(argv[1], 1) ;

 new_image=peter_framing_filter(image) ;

 sprintf(new_filename,"%s.framed.jpg",argv[1]) ;
 cvSaveImage(new_filename, new_image, NULL) ;

 return(0) ;
}

*/


int main(int argc, char* argv[]) {
    if (argc <= 1) {
        cerr << "Usage: peter_framing_filter <filename>" << endl;
        return 1;
    }
     FileEntry entry;
    entry._image_name = argv[1];
    FaceDetectResult result = peterFramingFilter(entry) ;
    
    IplImage*  image  = cvLoadImage(entry._image_name.c_str());
    if (!image) {
        cerr << "Could not find '" << entry._image_name << "'" << endl;
        abort();
    }
    IplImage*  scaled_image = scaleImage640x480(image);
    IplImage*  cropped_image = cropImage(scaled_image, result._face_rect);  
    string cropped_image_name = entry._image_name + ".framed.jpg";
    cvSaveImage(cropped_image_name.c_str(), cropped_image);
  
    cvReleaseImage(&scaled_image);
    cvReleaseImage(&cropped_image); 
    cvReleaseImage(&image);    
    return 0;
}

#endif  // #if TEST_MANY_SETTINGS


#endif
