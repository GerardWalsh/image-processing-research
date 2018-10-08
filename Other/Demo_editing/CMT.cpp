#include "CMT.h"
#include "Matcher.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdlib>
#include <iostream>
#include <ctime>

namespace cmt {

void CMT::initialize(const Mat im_gray, const Rect rect)
{
    FILE_LOG(logDEBUG) << "CMT::initialize() call";
    //Remember initial size
    size_initial = rect.size();

    //Remember initial image
    im_prev = im_gray;

    //Compute center of rect
    Point2f center = Point2f(rect.x + rect.width/2.0, rect.y + rect.height/2.0);

    //Initialize rotated bounding box
    bb_rot = RotatedRect(center, size_initial, 0.0);

    //Initialize detector and descriptor
#if CV_MAJOR_VERSION > 2
    detector = cv::ORB::create();
    descriptor = cv::ORB::create();
#else
    detector = FeatureDetector::create(str_detector);
    descriptor = DescriptorExtractor::create(str_descriptor);
#endif

    gpuOrb = cv::cuda::ORB::create(5000, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);   
    FastFeaturesDetect = cv::cuda::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16, 5000);


    //Get initial keypoints in whole image and compute their descriptors
    vector<KeyPoint> keypoints, keypointsTest;
    detector->detect(im_gray, keypoints);
	
     //Added
     cv::cuda::GpuMat imGrayScale, keypointsTestGpu, mask1, keys1, keys2, desc1;
     imGrayScale.upload(im_gray);

     //FastFeaturesDetect->detectAsync(imGrayScale, keys1, cv::noArray(), stream1); //Fast features should produce higher accuracy
     gpuOrb->detectAsync(imGrayScale, keys1, cv::noArray(), stream1);
     gpuOrb->computeAsync(imGrayScale, keys1, databaseAsync, stream1);
     stream1.waitForCompletion();
     //desc1.copyTo(databaseAsync);	
     //End added

    //Divide keypoints into foreground and background keypoints according to selection
    vector<KeyPoint> keypoints_fg;
    vector<KeyPoint> keypoints_bg;

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint k = keypoints[i];
        Point2f pt = k.pt;

        if (pt.x > rect.x && pt.y > rect.y && pt.x < rect.br().x && pt.y < rect.br().y)
        {
            keypoints_fg.push_back(k);
        }

        else
        {
            keypoints_bg.push_back(k);
        }

    }

    //Create foreground classes
    vector<int> classes_fg;
    classes_fg.reserve(keypoints_fg.size());
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        classes_fg.push_back(i);
    }

    //Compute foreground/background features
    Mat descs_fg;
    Mat descs_bg;
    descriptor->compute(im_gray, keypoints_fg, descs_fg);
    descriptor->compute(im_gray, keypoints_bg, descs_bg);
    std::cout << "FG KEYPOINTS " << keypoints_fg.size() << std::endl;

    //Only now is the right time to convert keypoints to points, as compute() might remove some keypoints
    vector<Point2f> points_fg;
    vector<Point2f> points_bg;

    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_fg.push_back(keypoints_fg[i].pt);
    }

    FILE_LOG(logDEBUG) << points_fg.size() << " foreground points.";

    for (size_t i = 0; i < keypoints_bg.size(); i++)
    {
        points_bg.push_back(keypoints_bg[i].pt);
    }

    //Create normalized points
    vector<Point2f> points_normalized;
    for (size_t i = 0; i < points_fg.size(); i++)
    {
        points_normalized.push_back(points_fg[i] - center);
    }

    //Initialize matcher
    matcher.initialize(points_normalized, descs_fg, classes_fg, descs_bg, center);

    //Initialize consensus
    consensus.initialize(points_normalized);

    //Create initial set of active keypoints
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_active.push_back(keypoints_fg[i].pt);
        classes_active = classes_fg;
    }

    FILE_LOG(logDEBUG) << "CMT::initialize() return";
	totalPipe = 0;
	totalKernelLaunch = 0;
	totalProcess = 0;
}

void CMT::processFrame(Mat im_gray, int frame, Mat im_gray_next, int pipeline) {
	float start = cv::getTickCount();
    FILE_LOG(logDEBUG) << "CMT::processFrame() call";

    vector<KeyPoint> keypoints;
    Mat descriptors;

    //Initialize variables in device memory
    cv::cuda::GpuMat imGrayScaleP, keypointsTestGpuP, mask1P, keys1P, desc1P, matchesGpuP;

    //vector<Point2f> points_matched_global;
    //vector<int> classes_matched_global;

	    //Check and prime pipeline
    if(pipeline && frame == 1){
	    //Check if pipeline needs to be primed

		imCurrent = im_gray;
		imNext = im_gray_next;
		imPrevious = im_prev;
		
		//Variables for holding in device memory
		cv::cuda::GpuMat imGrayScale, keypointsTestGpu, mask1, keys1, desc1;

		//Upload image into device memory
		imGrayScale.upload(imCurrent);

		//Detect keypoints and compute descriptors - detectAndCompute() uneccessary eliminates memory transfers
		gpuOrb->detectAndComputeAsync(imGrayScale, cv::noArray(), keys1, desc1, false, stream1);

		//Match keypoints globally
		matcher.matchGlobalAsync(keys1, keypointsLocal, desc1, descriptorsLocal, points_matched_global, classes_matched_global, stream1, true);
		//std::cout << "pipe" << std::endl;
		
			}
	   
    if(pipeline){

		

		//Grab hold of next image in pipeline
		imNext = im_gray;

		//Upload NEXT image in pipeline into device memory
		imGrayScaleP.upload(imNext);

		//Detect keypoints and compute descriptors for next image in pipeline
		//clock_t t2 = clock();
		float pipelineStart = cv::getTickCount();
		gpuOrb->detectAndComputeAsync(imGrayScaleP, cv::noArray(), keys1P, desc1P, false, stream1); //The boolean flag is a problem
		//std::cout << "Async kernel launch (ms) :" << 1000*(clock() - t2)/((double)CLOCKS_PER_SEC) << std::endl;
		//imshow("Image being processed by GPU", imNext);	
		//cv::waitKey(0);

		//Perform knn (2) matching per keypoint for next image in pipeline
		matcher.matchGlobalNoSync(desc1P, matchesGpuP, stream1);

		float pipelineEnd = cv::getTickCount() - pipelineStart;
		totalKernelLaunch += (pipelineEnd)/cv::getTickFrequency();
    		//std::cout << "time for " <<(pipelineEnd)/cv::getTickFrequency() <<"Kernel launch"<< "total kernel launch" << totalKernelLaunch << std::endl;
			}
	
    else{		
		float T1 = cv::getTickCount();

		//Upload CURRENT image in pipeline into device memory
		imGrayScaleP.upload(im_gray);

		//Detect keypoints and compute descriptors, for current image
		//clock_t t3 = clock();
		
		gpuOrb->detectAndComputeAsync(imGrayScaleP, cv::noArray(), keys1P, desc1P, false, stream1);
		//std::cout << "sync kernel completion (ms) :" << 1000*(clock() - t3)/((double)CLOCKS_PER_SEC) << std::endl;

		//Perform knn (2) matching per keypoint current image
		matcher.matchGlobalNoSync(desc1P, matchesGpuP, stream1);

		stream1.waitForCompletion();
		
		float T2 = cv::getTickCount();

		totalPipe += (T2-T1)/cv::getTickFrequency();
		//std::cout << "time" <<(T2-T1)/cv::getTickFrequency() <<"Pipelined"<< frame<< "total" << totalPipe << std::endl;

		matcher.processMatches(matchesGpuP, keys1P, points_matched_global, classes_matched_global, desc1P, keypoints, descriptors);

		//matcher.matchGlobalAsync(keys1P, keypoints, desc1P, descriptors, points_matched_global, classes_matched_global, stream1, true);
		//std::cout << "size of matches sync" << points_matched_global.size() << std::endl;
		
		}

    FILE_LOG(logDEBUG) << points_matched_global.size() << " points matched globally.";
		   
    //Track keypoints
    vector<Point2f> points_tracked;
    vector<unsigned char> status;

    if(pipeline) tracker.track(imPrevious, imCurrent, points_active, points_tracked, status);
    else tracker.track(im_prev, im_gray, points_active, points_tracked, status);
		
    FILE_LOG(logDEBUG) << points_tracked.size() << " tracked points.";

    //keep only successful classes
    vector<int> classes_tracked;
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        if (status[i])
        {
            classes_tracked.push_back(classes_active[i]);
        }

    }

    //Fuse tracked and globally matched points
    vector<Point2f> points_fused;
    vector<int> classes_fused;

	//std::cout << "Points matched globally before fusion:" << points_matched_global .size() <<std::endl;

    fusion.preferFirst(points_tracked, classes_tracked, points_matched_global, classes_matched_global,
            points_fused, classes_fused);

    points_matched_global.clear();
    classes_matched_global.clear();

    FILE_LOG(logDEBUG) << points_fused.size() << " points fused.";

    //Estimate scale and rotation from the fused points
    float scale;
    float rotation;
    consensus.estimateScaleRotation(points_fused, classes_fused, scale, rotation);
	//points_fused.clear();
	//classes_fused.clear();
    FILE_LOG(logDEBUG) << "scale " << scale << ", " << "rotation " << rotation;

    //Find inliers and the center of their votes
    Point2f center;
    vector<Point2f> points_inlier;
    vector<int> classes_inlier;

    consensus.findConsensus(points_fused, classes_fused, scale, rotation,
            center, points_inlier, classes_inlier);

    FILE_LOG(logDEBUG) << points_inlier.size() << " inlier points.";
    FILE_LOG(logDEBUG) << "center " << center;

    //Match keypoints locally
    vector<Point2f> points_matched_local;
    vector<int> classes_matched_local;

    if (pipeline) matcher.matchLocal(keypointsLocal, descriptorsLocal, center, scale, rotation, points_matched_local, classes_matched_local);
    else matcher.matchLocal(keypoints, descriptors, center, scale, rotation, points_matched_local, classes_matched_local);

    FILE_LOG(logDEBUG) << points_matched_local.size() << " points matched locally.";

    //Clear active points
    points_active.clear();
    classes_active.clear();


    //Fuse locally matched points and inliers
    fusion.preferFirst(points_matched_local, classes_matched_local, points_inlier, classes_inlier, points_active, classes_active);
 
    FILE_LOG(logDEBUG) << points_active.size() << " final fused points.";

    //TODO: Use theta to suppress result
    bb_rot = RotatedRect(center,  size_initial * scale, rotation/CV_PI * 180);
	
	
    if (pipeline){

	//float T3 = cv::getTickCount();
	    //Synchronise device with host
	    stream1.waitForCompletion();
            //std::cout << "size of matches async" << matchesGpuP.size() << std::endl;	

	    //Device must be synchronised prior to calling processMatches()
	    matcher.processMatches(matchesGpuP, keys1P, points_matched_global, classes_matched_global, desc1P, keypointsLocal, descriptorsLocal);
	    //std::cout << "size of points matched global after sync " << points_matched_global.size() << std::endl;

	    //Update images to be processed	
	    imPrevious = imCurrent;
	    imCurrent = imNext;

	//float T4 = cv::getTickCount();
	//totalProcess += (T4-T3)/cv::getTickFrequency();
	//std::cout << "time" << (T4-T3)/cv::getTickFrequency() <<"Process matches"<< frame<< "total" << totalProcess << std::endl;
	//std::cout << "pipe" << std::endl;
		}
	

    else im_prev = im_gray;
	

    FILE_LOG(logDEBUG) << "CMT::processFrame() return";
	float res = (cv::getTickCount()-start)/cv::getTickFrequency();
	totalProcess += res;
	//std::cout << " total Process time " << totalProcess <<"Loop time"<< std::endl;
	//cv::waitKey(0); //for debugging
}

} /* namespace CMT */

