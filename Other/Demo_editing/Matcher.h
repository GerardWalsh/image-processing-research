#ifndef MATCHER_H

#define MATCHER_H

#include "common.h"

//#include "opencv2/features2d/features2d.hpp"

//Added 
	#include "opencv2/opencv_modules.hpp"
	#include "opencv2/core.hpp"
	#include "opencv2/core/cuda.hpp" // GPU structures and methods
	#include "opencv2/cudafeatures2d.hpp"
	#include <iostream>
	#include <cstdlib>
	//#include "CMT.h"
//End added

using cv::KeyPoint;
using cv::Ptr;
using cv::DescriptorMatcher;

namespace cmt {

class Matcher
{
public:
    Matcher() : thr_dist(0.25), thr_ratio(0.8), thr_cutoff(20) {};
    void initialize(const vector<Point2f> & pts_fg_norm, const Mat desc_fg, const vector<int> & classes_fg,
            const Mat desc_bg, const Point2f center);
    void matchGlobal(const vector<KeyPoint> & keypoints, const Mat descriptors,
            vector<Point2f> & points_matched, vector<int> & classes_matched);
    void matchLocal(const vector<KeyPoint> & keypoints, const Mat descriptors,
            const Point2f center, const float scale, const float rotation,
            vector<Point2f> & points_matched, vector<int> & classes_matched);
	//Added
		void matchGlobalAsync(cv::cuda::GpuMat & keypoints, vector<KeyPoint> & keypointsCpu, cv::cuda::GpuMat & descriptors, Mat & descriptorsCpu, vector<Point2f> & points_matched, vector<int> & classes_matched, cv::cuda::Stream & stream1, bool flag);
		void matchGlobalNoSync(cv::cuda::GpuMat & descriptors, cv::cuda::GpuMat & matchesGpu, cv::cuda::Stream & stream1);
		void processMatches(cv::cuda::GpuMat & matches, cv::cuda::GpuMat & keypoints, vector<Point2f> & points_matched, vector<int> & classes_matched, cv::cuda::GpuMat & descriptors, vector<KeyPoint> & keypointsCpu, Mat & descriptorsCpu);
	//End added

private:
    vector<Point2f> pts_fg_norm;
    Mat database;
    vector<int> classes;
    int desc_length;
    int num_bg_points;
    Ptr<DescriptorMatcher> bfmatcher;
	//Added
		Ptr< cv::cuda::DescriptorMatcher > bfmatcherAsync;
		Ptr<cv::cuda::ORB> gpuOrbConvert;
		cv::cuda::GpuMat data;
	//End added
    float thr_dist;
    float thr_ratio;
    float thr_cutoff;
};

} /* namespace CMT */

#endif /* end of include guard: MATCHER_H */
