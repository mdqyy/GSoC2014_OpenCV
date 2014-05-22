#include "structuredEdgeDetection.h"

cv::Mat StructuredEdgeDetection::__imresize
    (const cv::Mat &src, const cv::Size &sizeDst)
{
    int resizeType = sizeDst.height <= src.size().height 
                   ? cv::INTER_AREA 
                   : cv::INTER_LINEAR;

    cv::Mat dst; 
    if ( sizeDst != src.size() ) 
        cv::resize(src, dst, sizeDst, 0.0f, 0.0f, resizeType);
    else
        dst = src;

    return dst;
}

cv::Mat StructuredEdgeDetection::__imsmooth
    (const cv::Mat &src, const int rad)
{
    if (rad < 1) 
        return src;

    cv::Mat dst;

    cv::Size sizeFltr(rad + !(rad&1), rad + !(rad&1));
    cv::boxFilter(src, dst, -1, sizeFltr, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(dst, dst, -1, sizeFltr, cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    return dst; 
}

void StructuredEdgeDetection::__getFeatures
    (const cv::Mat &img, Mat3D &regularFeatures, Mat3D &additionalFeatures)
{
    cv::Mat luvImg;
    cv::cvtColor(img, luvImg, CV_RGB2Luv);

    std::vector <cv::Mat> features;

    cv::Size sizeSc(img.cols/__shrink_number__, img.rows/__shrink_number__);
    cv::Mat shrinked = __imresize(luvImg, sizeSc);
    cv::split(shrinked, features);

    float scalesData[] = {1.0, 0.5};
    std::vector <float> scales(std::begin(scalesData), std::end(scalesData));
    
    for (size_t k = 0; k < scales.size(); ++k) 
    {
        cv::Size sizeSc( int(scales[k]*img.cols), int(scales[k]*img.rows) );
        cv::Mat I = abs(__shrink_number__ - scales[k]) < 1e-2 
                  ? shrinked 
                  : __imresize(luvImg, sizeSc);

        cv::Mat Dx, Dy, magnitude, phase;
        cv::Sobel(I, Dx, cv::DataType<float>::type, 1, 0, 1, 
                  1.0, 0.0, cv::BORDER_REFLECT);
        cv::Sobel(I, Dy, cv::DataType<float>::type, 1, 0, 1, 
                  1.0, 0.0, cv::BORDER_REFLECT);

        cv::reduce(Dx.reshape(1, I.rows*I.cols), Dx, 1, CV_REDUCE_MAX, -1);
        cv::reduce(Dy.reshape(1, I.rows*I.cols), Dy, 1, CV_REDUCE_MAX, -1);

        Dx = Dx.reshape(1, I.rows);
        Dy = Dy.reshape(1, I.rows);

        cv::phase(Dx, Dy, phase);
        cv::magnitude(Dx, Dy, magnitude);

        cv::Mat smoothed = __imsmooth(magnitude, __gradient_normalization__);
        magnitude /= smoothed + 0.1;

        int binSize = std::max(1, int(__shrink_number__/scales[k]) );

        int histHeight = int( std::floor(I.rows / binSize) );
        int histWidth  = int( std::floor(I.cols / binSize) );

        int histType = CV_MAKETYPE(cv::DataType<float>::type, __gradient_orientations__);
        cv::Mat hist( histHeight, histWidth, histType, cv::Scalar::all(0) );

        for (size_t i = 0; i < phase.rows; ++i)
        {
            float *anglePtr  = phase.ptr<float>(i);
            float *lengthPtr = magnitude.ptr<float>(i);

            for (size_t j = 0; j < phase.cols; ++j)
            {
                float angle = anglePtr[j] * __gradient_orientations__;
                float *data = (float *) hist.data;
                        
                int index = int( (i/binSize)*histHeight 
                          + (j/binSize)*__gradient_orientations__ 
                          + std::floor(angle / (2*CV_PI)) );
                data[index] += lengthPtr[j];
            }
        }

        magnitude = __imresize( magnitude, img.size() );
        features.push_back(magnitude);
        features.push_back(/**/ __imresize( hist, img.size() ) /**/);
    }

    // Mixing and smoothing

    int resType = CV_MAKETYPE(cv::DataType<float>::type, __edge_orientations__);
    regularFeatures.create(img.size(), resType);
    additionalFeatures.create(img.size(), resType);

    std::vector <int> fromTo(2*__edge_orientations__, 0);
    for (int i = 0; i < 2*__edge_orientations__; ++i)
        fromTo.push_back(i/2);
    cv::mixChannels(features, regularFeatures, fromTo);

    int rad1 = cvRound(__nonreg_features_smoothing__ / float(__shrink_number__));
    additionalFeatures = __imsmooth(regularFeatures, rad1);

    int rad2 = cvRound(__reg_features_smoothing__ / float(__shrink_number__));
    regularFeatures = __imsmooth(regularFeatures, rad2);
}

void StructuredEdgeDetection::detectSingleScale
    (cv::InputArray _src, cv::OutputArrayOfArrays _dst)
{
    cv::Mat src = _src.getMat();
    CV_Assert( src.type() == CV_MAKETYPE(cv::DataType<float>::type, 3) );

    // Extraction
    Mat3D regularFeatures, additionalFeatures;
    __getFeatures(src, regularFeatures, additionalFeatures);

    // Detection
    //...

    // Ending
    //result.copyTo(_dst.getMat());
}

void StructuredEdgeDetection::detectMultipleScales
    (cv::InputArray _src, cv::OutputArrayOfArrays _dst)
{
    cv::Mat src = _src.getMat(); 
    CV_Assert( src.type() == CV_MAKETYPE(cv::DataType<float>::type, 3));
    
    int resType = CV_MAKETYPE(cv::DataType<float>::type, __edge_orientations__);
    Mat3D result( src.size(), resType, cv::Scalar::all(0) );

    float scalesData[] = {0.5f, 1.0f, 2.0f};
    std::vector <float> scales(std::begin(scalesData), std::end(scalesData));

    for (size_t i = 0; i < scales.size(); ++i) 
    {        
        cv::Size sizeSc( int(scales[i]*src.cols), int(scales[i]*src.rows) );
        cv::Mat scaledSrc = __imresize(src, sizeSc);
       
        Mat3D scaledResult( scaledSrc.size(), resType, cv::Scalar::all(0) );
        detectSingleScale(scaledSrc, scaledResult);
        
        result += __imresize( scaledResult, result.size() );
    }
    result /= float( scales.size() );

    result.copyTo(_dst.getMat());
}

void StructuredEdgeDetection::train() {}

void StructuredEdgeDetection::load(const std::string &filename) {}

void StructuredEdgeDetection::save(const std::string &filename) {}

StructuredEdgeDetection::StructuredEdgeDetection()
{  
    __non_maximum_supression__ = false; 
    __stride_width__ = 2;            
    __shrink_number__ = 2;           
    __patch_width__ = 32;             
    __gradient_orientations__ = 4;   
    __gradient_smoothing__ = 0;      
    __reg_features_smoothing__ = 2;  
    __nonreg_features_smoothing__ = 8;   
    __gradient_normalization__ = 4; 
    __selfsimilarity_cells__ = 5;   
    __number_of_trees__ = 8;             
    __number_of_trees_to_evaluate__ = 4; 
    __edge_orientations__ = 13;           
}

StructuredEdgeDetection::StructuredEdgeDetection(const std::string &filename){}