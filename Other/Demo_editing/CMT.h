#ifndef CMT_H

#define CMT_H

#include "common.h"
#include "Consensus.h"
#include "Fusion.h"
#include "Matcher.h"
#include "Tracker.h"

#include <opencv2/features2d/features2d.hpp>

	//Added
		#include "opencv2/core/cuda.hpp" 	
		#include "opencv2/cudafeatures2d.hpp"
		#include <opencv2/cudaimgproc.hpp>
		#include <opencv2/cudafilters.hpp>
	//End added

using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::Ptr;
using cv::RotatedRect;
using cv::Size2f;

	//Added	
		using cv::DMatch;
	//End added

namespace cmt
{

class CMT
{
public:
    CMT() : str_detector("FAST"), str_descriptor("BRISK") {};
    void initialize(const Mat im_gray, const Rect rect);
    void processFrame(const Mat im_gray, int frame, Mat im_gray_next, int pipeline);
    void processFrameNP(const Mat im_gray);

    Fusion fusion;
    Matcher matcher;
    Tracker tracker;
    Consensus consensus;

    string str_detector;
    string str_descriptor;

    vector<Point2f> points_active; //public for visualization purposes
    RotatedRect bb_rot;
	//Added
		cv::cuda::GpuMat databaseAsync;
		vector<Point2f> points_matched_global;
		vector<int> classes_matched_global;
	//End added

private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;

	//Added
		Ptr<cv::cuda::Feature2DAsync> FastFeaturesDetect;
		Ptr< cv::cuda::DescriptorMatcher > bfmatcherAsyncPipe;
		Ptr<cv::cuda::ORB> gpuOrb;
		cv::cuda::Stream stream1;
			
	//End added

    Size2f size_initial;

    vector<int> classes_active;

    float theta;

    Mat im_prev, im_new, im_gpu, im_opticalFlow, im_cpu;
    Mat imPrevious, imCurrent, imNext;
    Mat descriptorsLocal;
    vector<KeyPoint> keypointsLocal;

    float totalPipe;
    float totalProcess;

    float totalKernelLaunch;

};

} /* namespace CMT */

#endif /* end of include guard: CMT_H */
