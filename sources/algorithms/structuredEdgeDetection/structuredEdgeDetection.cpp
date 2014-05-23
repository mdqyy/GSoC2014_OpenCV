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

    cv::boxFilter(src, dst, -1, cv::Size(2*rad/3, 2*rad/3),
        cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(dst, dst, -1, cv::Size(2*rad/3, 2*rad/3),
        cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    return dst; 
}

void StructuredEdgeDetection::__getFeatures
    (const cv::Mat &img, NChannelsMat &features)
{
    cv::Mat luvImg;
    cv::cvtColor(img, luvImg, CV_RGB2Luv);

    int shrink  = __rf.options.shrinkNumber;
    int outNum  = __rf.options.numberOfOutputChannels;
    int gradNum = __rf.options.numberOfGradientOrientations;
    int gnormRad = __rf.options.gradientNormalizationRadius;

    std::vector <cv::Mat> featureArray;

    cv::Size sizeSc(img.cols/shrink, img.rows/shrink);
    cv::Mat shrinked = __imresize(luvImg, sizeSc);
    cv::split(shrinked, features);

    float scalesData[] = {1.0, 0.5};
    std::vector <float> scales(std::begin(scalesData), std::end(scalesData));
    
    for (size_t k = 0; k < scales.size(); ++k) 
    {
        cv::Size sizeSc( int(scales[k]*img.cols), int(scales[k]*img.rows) );
        cv::Mat I = abs(shrink - scales[k]) < 1e-2 
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

        cv::Mat smoothed = __imsmooth(magnitude, gnormRad);
        magnitude /= smoothed + 0.1;

        int binSize = std::max(1, int(shrink/scales[k]) );

        int histHeight = int( std::floor(I.rows / binSize) );
        int histWidth  = int( std::floor(I.cols / binSize) );

        int histType = CV_MAKETYPE(cv::DataType<float>::type, gradNum);
        cv::Mat hist( histHeight, histWidth, histType, cv::Scalar::all(0) );

        for (size_t i = 0; i < phase.rows; ++i)
        {
            float *anglePtr  = phase.ptr<float>(i);
            float *lengthPtr = magnitude.ptr<float>(i);

            for (size_t j = 0; j < phase.cols; ++j)
            {
                float angle = anglePtr[j] * gradNum;
                float *data = (float *) hist.data;
                        
                int index = int( (/**/ (i/binSize)*histHeight + (j/binSize) 
                          + std::floor(angle / (2*CV_PI)) /**/)*gradNum);
                data[index] += lengthPtr[j];
            }
        }

        magnitude = __imresize( magnitude, shrinked.size() );
        featureArray.push_back(magnitude);
        featureArray.push_back(/**/ __imresize( hist, shrinked.size() ) /**/);
    }

    // Mixing and smoothing

    int resType = CV_MAKETYPE(cv::DataType<float>::type, outNum);
    features.create(img.size(), resType);

    std::vector <int> fromTo(2*outNum, 0);
    for (int i = 0; i < 2*outNum; ++i)
        fromTo.push_back(i/2);
    cv::mixChannels(featureArray, features, fromTo);
}

void StructuredEdgeDetection::__detectEdges
    (const NChannelsMat &features, cv::Mat &dst)
{
    NChannelsMat regFeatures;
    NChannelsMat ssFeatures;

    int radReg = cvRound(__rf.options.regFeatureSmoothingRadius 
               / float(  __rf.options.shrinkNumber));
    regFeatures = __imsmooth(features, radReg);

    int radSS = cvRound(__rf.options.ssFeatureSmoothingRadius 
              / float(  __rf.options.shrinkNumber));
    ssFeatures = __imsmooth(features, radSS);


    int nTreesEval = __rf.options.numberOfTreesToEvaluate;
    int nTrees = __rf.options.numberOfTrees;
    int nTreesNodes = __rf.numberOfTreeNodes;

    int nFeatures = features.total()*features.channels();

    int shrink = __rf.options.shrinkNumber;
    int stride = __rf.options.stride;
    int pSize  = __rf.options.patchSize;
    int gridSize = __rf.options.selfsimilarityGridSize;

    const int height = (int) std::ceil( double(features.rows*shrink - pSize) / stride ); 
    const int width  = (int) std::ceil( double(features.cols*shrink - pSize) / stride ); 
    const int channels = features.channels();
    // image size in patches

    int indType = CV_MAKETYPE(cv::DataType<unsigned int>::type, nTreesEval);
    NChannelsMat indexes(height, width, indType);

    std::vector <int> offsetI(pSize/shrink)*(pSize/shrink)*channels, 0); 
    for (int i = 0; i < CV_SQR(pSize/shrink)*channels; ++i)
    {
        int x = i/channels%(pSize/shrink);
        int y = i/channels/(pSize/shrink);
        
        offsetI[i] = y*(features.cols/shrink)*channels + x*channels + (i%channels);
    }
    // lookup table for mapping linear index to offsets

    int *offsetX = new int[CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)*channels]; 
    int *offsetY = new int[CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)*channels]; 
    for (int i = 0, n = 0; i < CV_SQR(gridSize)*channels; ++i)
        for (int j = (i + 1)/channels; j < CV_SQR(gridSize); ++j, ++n)
        {
            int x1 = i/channels%gridSize, y1 = i/channels/gridSize;
            int x2 = j%gridSize, y2 = j/gridSize;

            ...offsetX[n] = y1*(features.cols/shrink)*channels + x1*channels + (i%channels);
            ...offsetY[n] = y2*(features.cols/shrink)*channels + x2*channels + (i%channels);
        }
    // lookup tables for mapping linear index to offset pairs

    for (int i = 0; i < height; ++i)
    {
       float *regFeaturesPtr = regFeatures.ptr<float>(i*stride/shrink);
       float  *ssFeaturesPtr = ssFeatures.ptr<float>(i*stride/shrink);

       for (int j = 0, k = 0; j < width; ++k, j += !(k %= nTreesEval))
       // for j,k in [0;width)x[0;nTreesEval)
       {
           int currentNode = ( ((i + j)%(2*nTreesEval) + k)%nTrees )*nTreesNodes;
           // select root node of the tree to evaluate

           int offset = j*stride/shrink;
           while (__rf.childs[currentNode] != -1)
           {
               unsigned int currentId = __rf.featureIds[currentNode];
               float currentFeature = (currentId < nFeatures)
                                    ? regFeaturesPtr[offset + offsetI[currentId]]
                                    :   ssFeaturesPtr[offset + offsetX[currentId - nFeatures]]
                                      - ssFeaturesPtr[offset + offsetY[currentId - nFeatures]];
                
               // compare feature to threshold and move left or right accordingly
               if (currentFeature < __rf.thresholds[currentNode])
                   currentNode = __rf.childs[currentNode] - 1;
               else
                   currentNode = __rf.childs[currentNode];
           }

           indexes.data[i*height*channels + j*channels + k] = currentNode;
       }
    }

    delete [] offsetI;
    delete [] offsetX;
    delete [] offsetY;

    ...
}

void StructuredEdgeDetection::detectSingleScale
    (cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    CV_Assert( src.type() == CV_MAKETYPE(cv::DataType<float>::type, 3) );

    NChannelsMat features;
    __getFeatures(src, features);
    __detectEdges(features, _dst.getMat());
}

void StructuredEdgeDetection::detectMultipleScales
    (cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat(); 
    CV_Assert( src.type() == CV_MAKETYPE(cv::DataType<float>::type, 3));
    
    int resType = CV_MAKETYPE(cv::DataType<float>::type, __rf.options.numberOfOutputChannels);
    NChannelsMat result( src.size(), resType, cv::Scalar::all(0) );

    float scalesData[] = {0.5f, 1.0f, 2.0f};
    std::vector <float> scales(std::begin(scalesData), std::end(scalesData));

    for (size_t i = 0; i < scales.size(); ++i) 
    {        
        cv::Size sizeSc( int(scales[i]*src.cols), int(scales[i]*src.rows) );
        cv::Mat scaledSrc = __imresize(src, sizeSc);
       
        NChannelsMat scaledResult( scaledSrc.size(), resType, cv::Scalar::all(0) );
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
                          __rf.options.stride = 2;            
                    __rf.options.shrinkNumber = 2;           
                       __rf.options.patchSize = 32;             
                  __rf.options.patchInnerSize = 16;             
    __rf.options.numberOfGradientOrientations = 4;   
         __rf.options.gradientSmoothingRadius = 0;      
       __rf.options.regFeatureSmoothingRadius = 2;  
        __rf.options.ssFeatureSmoothingRadius = 8;   
     __rf.options.gradientNormalizationRadius = 4; 
          __rf.options.selfsimilarityGridSize = 5;   
                   __rf.options.numberOfTrees = 8;             
         __rf.options.numberOfTreesToEvaluate = 4; 
          __rf.options.numberOfOutputChannels = 13;           

    
}

StructuredEdgeDetection::StructuredEdgeDetection(const std::string &filename){}