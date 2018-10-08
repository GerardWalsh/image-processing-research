#ifndef COMMON_H

#define COMMON_H

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
	//Added
		#include "opencv2/core/cuda.hpp" 
	//End added

#include "logging/log.h"

using cv::Mat;
using cv::Point2f;
using cv::Rect;
using cv::Size2f;
using std::numeric_limits;
using std::string;
using std::vector;

namespace cmt
{
    float median(vector<float> & A);
    Point2f rotate(const Point2f v, const float angle);
    template<class T>
    int sgn(T x)
    {
        if (x >=0) return 1;
        else return -1;
    }
	//Added
		//cv::cuda::Stream stream1;
	//End added

} /* namespace cmt */

#endif /* end of include guard: COMMON_H */
