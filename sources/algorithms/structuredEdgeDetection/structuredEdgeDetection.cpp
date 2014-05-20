#include "structuredEdgeDetection.h"

void StructuredEdgeDetection::__imresize(const cv::Mat &src, cv::Mat &dst,
                                         const cv::Size dstSize)
{
    if ( dstSize != src.size() ) {
        int resizeType = dstSize.height < src.size().height ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(src, dst, dstSize, 0.0f, 0.0f, resizeType);
    }
    else
        dst = src; // no data copying
}

void StructuredEdgeDetection::__imresize(const cv::Mat &src, cv::Mat &dst,
                                         const float resizeFactor)
{
    if (std::abs(resizeFactor - 1.0f) > 1e-2) {
        int resizeType = resizeFactor < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(src, dst, cv::Size(0.0, 0.0), resizeFactor, resizeFactor, resizeType);
    }
    else
        dst = src; // no data copying
}

void StructuredEdgeDetection::__imresize(const cv::Mat &img, 
                                         const cv::Size dstSize)
{
    if ( dstSize != img.size() ) {
        int resizeType = dstSize.height < src.size().height ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(img, img, dstSize, 0.0f, 0.0f, resizeType);
    }
}
void StructuredEdgeDetection::__imresize(const cv::Mat &img, 
                                         const float resizeFactor)
{
    if (std::abs(resizeFactor - 1.0f) > 1e-2) {
        int resizeType = resizeFactor < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(img, img, cv::Size(0.0, 0.0), resizeFactor, resizeFactor, resizeType);
    }
}

void StructuredEdgeDetection::__imsmooth(const cv::Mat &src, 
                                         const cv::Mat &dst,
                                         const int radius)
{
    if (radius < 1) 
        return;

    cv::Size fSize(radius+!(radius&1), radius+!(radius&1));
    cv::boxFilter(src, dst, -1, fSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(dst, dst, -1, fSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
}

void StructuredEdgeDetection::__imsmooth(const cv::Mat &img,
                                         const int radius)
{
    if (radius < 1) 
        return;

    cv::Size fSize(radius+!(radius&1), radius+!(radius&1));
    cv::boxFilter(src, dst, -1, fSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(dst, dst, -1, fSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
}

void StructuredEdgeDetection::__getFeatures(const cv::Mat &img, 
                         std::vector<cv::Mat> &regularFeatures, 
                         std::vector<cv::Mat> &additionalFeatures)
{
    cv::Mat luvImg;
    cv::cvtColor(img, luvImg, CV_RGB2Luv);

    cv::Mat shrinked;
    __imresize(luvImg, shrinked, 1/__shrink_number__);
    cv::split(shrinked, regularFeatures);

    std::vector <float> scales = {1.0, 0.5};
    for (size_t i = 0; i < scales.size(); ++i) {

        cv::Mat I;
        if (abs(__shrink_number__ - scales[i]) < 1e-2)
            I = shrinked; // no data copying
        else
            __imresize(luvImg, I, scales[i]);

        cv::Mat Dx, Dy, magnitude, angle;
        cv::Sobel(I, Dx, cv::DataType<float>::type, 1, 0, 1, 
            1.0, 0.0, cv::BORDER_REFLECT);
        cv::Sobel(I, Dy, cv::DataType<float>::type, 1, 0, 1, 
            1.0, 0.0, cv::BORDER_REFLECT);

        cv::reduce(Dx.reshape(1, 3), Dx, 1, CV_REDUCE_MAX, -1);
        Dx.reshape(1, I.rows);

        cv::reduce(Dy.reshape(1, 3), Dy, 1, CV_REDUCE_MAX, -1);
        Dy.reshape(1, I.rows);

        cv::phase(Dx, Dy, angle);
        cv::magnitude(Dx, Dy, magnitude);

        cv::Mat smoothed;
        __imsmooth(magnitude, smoothed, __gradient_normalization__);
        magnitude /= smoothed + 0.1;

        std::vector <cv::Mat> hist(__gradient_orientations__, cv::Mat(...));
        for (int i = 0; i < __gradient_orientations__; ++i)
            ...

        __imresize( magnitude, src.size() );
        regularFeatures.push_back(magnitude);
        for (size_t j = 0; j < hist.size(); ++j)
            __imresize( hist[j], src.size() );
        std::copy(hist.begin(), hist.end(), std::back_inserter(regularFeatures));
    }

    for (size_t i = 0; i < regularFeatures.size(); ++i) {

        additionalFeatures.push_back(regularFeatures[i]);
        __imsmooth( additionalFeatures[i], additionalFeatures[i],
            cvRound(__nonreg_features_smoothing__ 
                    / float(__shrink_number__)) );

        __imsmooth( regularFeatures[i], regularFeatures[i], 
            cvRound(__reg_features_smoothing__
                    / float(__shrink_number__)) );
    }
}

void StructuredEdgeDetection::detectSingleScale(cv::InputArray _src,
                                                cv::OutputArrayOfArrays _dst)
{
    cv::Mat src = _src.getMat().clone(); 
    CV_Assert( src.type() == cv::DataType<float>::type
           &&  src.channels() == 3);

    // Extraction
    std::vector <cv::Mat> regularFeatures;
    std::vector <cv::Mat> additionalFeatures;
    __getFeatures(src, regularFeatures, 
                  additionalFeatures);

    // Detection
    ...

    // Ending
    std::vector <cv::Mat> dst = _dst.getMat();
    dst.resize(__edge_orientations__);
    for (size_t i = 0; i < result.size(); ++i)
        result[i].copyTo(dst[i]);
}

void StructuredEdgeDetection::detectMultipleScales(cv::InputArray _src,
                                                   cv::OutputArrayOfArrays _dst)
{
    cv::Mat src = _src.getMat(); 
    CV_Assert( src.type() == cv::DataType<float>::type
           &&  src.channels() == 3);

    std::vector <float> scales = {0.5f, 1.0f, 2.0f};
    
    cv::vector <cv::Mat> result(__edge_orientations__, 
        cv::Mat( src.rows, src.cols, src.type(), cv::Scalar::all(0) ));

    for (size_t i = 0; i < scales.size(); ++i) {
        
        cv::Mat scaled;
        __imresize(src, scaled, scales[i]);
       
        std::vector <cv::Mat> currentResult(__edge_orientations__, 
            cv::Mat( src.rows, src.cols, src.type(), cv::Scalar::all(0) ));
        detectSingleScale(scaled, currentResult);
        
        for (size_t j = 0; j < currentResult.size(); ++j) {
            __imresize( currentResult[j], result[j].size() );
            result[j] += currentResult[j];
        }
    }

    for (size_t i = 0; i < result.size(); ++i)
        result[i] /= scales.size();

    std::vector <cv::Mat> dst = _dst.getMat();
    dst.resize(__edge_orientations__);
    
    for (size_t i = 0; i < result.size(); ++i)
        result[i].copyTo(dst[i]);
}

void StructuredEdgeDetection::train()
{
    //__classifier.train(...);
}

void StructuredEdgeDetection::load(std::string filename) 
{
    __classifier.load( filename.c_str() );
}

void StructuredEdgeDetection::save(std::string filename) 
{
    __classifier.save( filename.c_str() );    
}

StructuredEdgeDetection::StructuredEdgeDetection(){}

StructuredEdgeDetection::StructuredEdgeDetection(const std::string filename)
{
    __classifier.load( filename.c_str() );
}

StructuredEdgeDetection::~StructuredEdgeDetection(){}
