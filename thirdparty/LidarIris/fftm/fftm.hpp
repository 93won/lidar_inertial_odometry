#ifndef _OPENCV_fftm_HPP_
#define _OPENCV_fftm_HPP_
#ifdef __cplusplus

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
cv::RotatedRect FFTMatch(const cv::Mat& im0, const cv::Mat& im1);
cv::RotatedRect LogPolarFFTTemplateMatch(cv::Mat& im0, cv::Mat& im1/*, double canny_threshold1=200, double canny_threshold2=100*/);
#endif
#endif

// 2019.09.17:
// remove canny