#ifndef _LIDAR_IRIS_H_
#define _LIDAR_IRIS_H_

#include <vector>
#include <opencv2/opencv.hpp>

// Simple 3D point structure to replace PCL dependency
struct SimplePoint3D {
    float x, y, z;
    SimplePoint3D() : x(0), y(0), z(0) {}
    SimplePoint3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

typedef std::vector<SimplePoint3D> SimplePointCloud;


class LidarIris
{
public:
    struct FeatureDesc
    {
        cv::Mat1b img;
        cv::Mat1b T;
        cv::Mat1b M;
    };

    LidarIris(int nscale, int minWaveLength, float mult, float sigmaOnf, int matchNum) : _nscale(nscale),
                                                                                         _minWaveLength(minWaveLength),
                                                                                         _mult(mult),
                                                                                         _sigmaOnf(sigmaOnf),
                                                                                         _matchNum(matchNum)
    {
    }
    LidarIris(const LidarIris &) = delete;
    LidarIris &operator=(const LidarIris &) = delete;

    static cv::Mat1b GetIris(const SimplePointCloud &cloud);
    float Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias = nullptr);

    FeatureDesc GetFeature(const cv::Mat1b &src);
    std::vector<cv::Mat2f> LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf);
    void GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias);
    
    static inline cv::Mat circRowShift(const cv::Mat &src, int shift_m_rows);
    static inline cv::Mat circColShift(const cv::Mat &src, int shift_n_cols);
    static cv::Mat circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols);

private:
    void LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M);

    int _nscale;
    int _minWaveLength;
    float _mult;
    float _sigmaOnf;
    int _matchNum;
};

#endif
