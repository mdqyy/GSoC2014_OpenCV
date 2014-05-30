#include "structuredEdgeDetection.h"

#include "../../opencv_size.h"

cv::Mat StructuredEdgeDetection::__imresize
    (const cv::Mat &img, const cv::Size &sizeDst)
{
    int resizeType = sizeDst <= img.size()
        ? cv::INTER_AREA
        : cv::INTER_LINEAR;

    cv::Mat res;

    if (sizeDst != img.size())
    {
        cv::resize(img, res, sizeDst, 0.0f, 0.0f, resizeType);
        return res.clone();
    }
    else
        return img.clone();
}

cv::Mat StructuredEdgeDetection::__imsmooth
    (const cv::Mat &img, const int rad)
{
    cv::Mat dst;

    cv::Size crad(CV_INC_IF_EVEN(2*rad/3), CV_INC_IF_EVEN(2*rad/3));
    if (crad < 3)
        return img.clone();

    cv::boxFilter(img, dst, -1, crad, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(dst, dst, -1, crad, cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    return dst.clone();
}

void StructuredEdgeDetection::__imhog
    (const cv::Mat &img, cv::Mat &magnitude, cv::Mat &histogram,
    const int numberOfBins, const int sizeOfPatch, const int gnrmRad)
{
    cv::Mat phase, Dx, Dy;

    cv::Sobel(img, Dx, cv::DataType<float>::type,
        1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT);
    cv::Sobel(img, Dy, cv::DataType<float>::type,
        0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT);

    cv::reduce(Dx.reshape(1, img.rows*img.cols), Dx, 1, CV_REDUCE_MAX, -1);
    cv::reduce(Dy.reshape(1, img.rows*img.cols), Dy, 1, CV_REDUCE_MAX, -1);

    cv::phase(Dx.reshape(1, img.rows), Dy.reshape(1, img.rows), phase);
    cv::magnitude(Dx.reshape(1, img.rows), Dy.reshape(1, img.rows), magnitude);

    magnitude /= __imsmooth(magnitude, gnrmRad) + 0.1;

    int histType = CV_MAKETYPE(cv::DataType<float>::type, numberOfBins);
    histogram.create( img.size()/float(sizeOfPatch), histType );

    histogram.setTo(0);
    for (int i = 0; i < phase.rows; ++i)
    {
        float *histPtr = histogram.ptr<float>(i/sizeOfPatch);
        const float *anglePtr  = phase.ptr<float>(i);
        const float *lengthPtr = magnitude.ptr<float>(i);

        for (int j = 0; j < phase.cols; ++j)
        {
            float angle = anglePtr[j]*numberOfBins;

            int index = cvFloor(/**/ ((j/sizeOfPatch)
                + angle/(2*CV_PI))*numberOfBins /**/);
            histPtr[index] += lengthPtr[j];
        }
    }
}

void StructuredEdgeDetection::__getFeatures
    (const cv::Mat &img, NChannelsMat &features)
{
    cv::Mat labImg = img;

    labImg.convertTo(labImg, cv::DataType<uchar>::type, 255.0);
    cv::cvtColor(labImg, labImg, CV_RGB2Lab);
    labImg.convertTo(labImg, cv::DataType<float>::type, 1/255.0);

    int shrink  = __rf.options.shrinkNumber;
    int outNum  = __rf.options.numberOfOutputChannels;
    int gradNum = __rf.options.numberOfGradientOrientations;
    int gnrmRad = __rf.options.gradientNormalizationRadius;

    std::vector <cv::Mat> featureArray;

    cv::Size nSize = img.size() / float(shrink);
    cv::split(__imresize(labImg, nSize), featureArray);

    CV_INIT_VECTOR(float, scales, {1.0, 0.5});

    for (size_t k = 0; k < scales.size(); ++k)
    {
        int sizeOfPatch = std::max( 1, int(shrink*scales[k]) );

        cv::Mat magnitude, histogram;
        __imhog(/**/ __imresize(labImg, scales[k]*img.size()),
            magnitude, histogram, gradNum, sizeOfPatch, gnrmRad /**/);

        featureArray.push_back(/**/ __imresize( magnitude, nSize ).clone() /**/);
        featureArray.push_back(/**/ __imresize( histogram, nSize ).clone() /**/);
    }

    // Mixing and smoothing

    int resType = CV_MAKETYPE(cv::DataType<float>::type, outNum);
    features.create(nSize, resType);

    std::vector <int> fromTo;
    for (int i = 0; i < 2*outNum; ++i)
        fromTo.push_back(i/2);
    cv::mixChannels(featureArray, features, fromTo);
}

void StructuredEdgeDetection::__detectEdges
    (const NChannelsMat &features, cv::Mat &dst)
{
    int shrink = __rf.options.shrinkNumber;
    int rfs = __rf.options.regFeatureSmoothingRadius;
    int sfs = __rf.options.ssFeatureSmoothingRadius;

    int nTreesEval = __rf.options.numberOfTreesToEvaluate;
    int nTrees = __rf.options.numberOfTrees;
    int nTreesNodes = __rf.numberOfTreeNodes;

    const int channels = features.channels();
    int pSize  = __rf.options.patchSize;

    int nFeatures = pSize*pSize*channels/shrink/shrink;
    int outNum = __rf.options.numberOfOutputChannels;

    int stride = __rf.options.stride;
    int ipSize = __rf.options.patchInnerSize;
    int gridSize = __rf.options.selfsimilarityGridSize;

    const int height = cvCeil( double(features.rows*shrink - pSize) / stride );
    const int width  = cvCeil( double(features.cols*shrink - pSize) / stride );
    // image size in patches

    //-------------------------------------------------------------------------

    NChannelsMat regFeatures = __imsmooth(features, cvRound(rfs / float(shrink)));
    NChannelsMat  ssFeatures = __imsmooth(features, cvRound(sfs / float(shrink)));

    NChannelsMat indexes(height, width, CV_MAKETYPE(cv::DataType<int>::type, nTreesEval));

    std::vector <int> offsetI(/**/ CV_SQR(pSize/shrink)*channels, 0);
    for (int i = 0; i < CV_SQR(pSize/shrink)*channels; ++i)
    {
        int x = i/channels%(pSize/shrink);
        int y = i/channels/(pSize/shrink);

        offsetI[i] = y*(features.cols/shrink)*channels + x*channels + (i%channels);
    }
    // lookup table for mapping linear index to offsets

    std::vector <int> offsetX( CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)*channels, 0);
    std::vector <int> offsetY( CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)*channels, 0);
    for (int i = 0, n = 0; i < CV_SQR(gridSize)*channels; ++i)
        for (int j = (i + 1)/channels; j < CV_SQR(gridSize); ++j, ++n)
        {
            float hc  = (pSize/shrink) / (2.0f*gridSize);
            // half of cell

            int x1 = cvRound(/**/ 2*( (i/channels%gridSize) + 0.5 )*hc /**/);
            int y1 = cvRound(/**/ 2*( (i/channels/gridSize) + 0.5 )*hc /**/);
            // "+ 0.5" means cell center

            int x2 = cvRound(/**/ 2*( (j%gridSize) + 0.5 )*hc /**/);
            int y2 = cvRound(/**/ 2*( (j/gridSize) + 0.5 )*hc /**/);
            // "+ 0.5" means cell center

            offsetX[n] = y1*(features.cols/shrink)*channels + x1*channels + (i%channels);
            offsetY[n] = y2*(features.cols/shrink)*channels + x2*channels + (i%channels);
        }
        // lookup tables for mapping linear index to offset pairs

        for (int i = 0; i < height; ++i)
        {
            float *regFeaturesPtr = regFeatures.ptr<float>(i*stride/shrink);
            float  *ssFeaturesPtr = ssFeatures.ptr<float>(i*stride/shrink);

            int *indexPtr = indexes.ptr<int>(i);

            for (int j = 0, k = 0; j < width; ++k, j += !(k %= nTreesEval))
                // for j,k in [0;width)x[0;nTreesEval)
            {
                int currentNode = ( ((i + j)%(2*nTreesEval) + k)%nTrees )*nTreesNodes;
                // select root node of the tree to evaluate

                int offset = (j*stride/shrink) * channels;
                while (__rf.childs[currentNode] != 0)
                {
                    int currentId = __rf.featureIds[currentNode];
                    float currentFeature;

                    if (currentId >= nFeatures)
                    {
                        int xIndex = offsetX[currentId - nFeatures];
                        float A = ssFeaturesPtr[offset + xIndex];

                        int yIndex = offsetY[currentId - nFeatures];
                        float B = ssFeaturesPtr[offset + yIndex];

                        currentFeature = A - B;
                    }
                    else
                        currentFeature = regFeaturesPtr[offset + offsetI[currentId]];

                    // compare feature to threshold and move left or right accordingly
                    if (currentFeature < __rf.thresholds[currentNode])
                        currentNode = __rf.childs[currentNode] - 1;
                    else
                        currentNode = __rf.childs[currentNode];
                }

                indexPtr[j*nTreesEval + k] = currentNode;
            }
        }

        dst.create(features.size(), CV_MAKETYPE(cv::DataType<float>::type, outNum));

        for (int i = 0; i < height; ++i)
        {
            int *indexPtr = indexes.ptr<int>(i);

            for (int j = 0, k = 0; j < width; ++k, j += !(k %= nTreesEval))
            {// for j,k in [0;width)x[0;nTreesEval)

                int currentNode = indexPtr[j*channels + k];
                float *E1 = E + (r*stride) + (c*stride)*h2;
                int b0=eBnds[k], b1=eBnds[k+1]; if(b0==b1) continue;
                for( int b=b0; b<b1; b++ ) E1[eids[eBins[b]]]++;
            }
        }

        dst *= 2.0f * CV_SQR(stride) / CV_SQR(ipSize) / nTreesEval;
}

void StructuredEdgeDetection::detectSingleScale
    (cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    CV_Assert( src.type() == CV_32FC3 );

    NChannelsMat features;
    __getFeatures(src, features);
    __detectEdges(features, _dst.getMat());
}

void StructuredEdgeDetection::detectMultipleScales
    (cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    CV_Assert( src.type() == CV_32FC3 );

    int resType = CV_MAKETYPE(cv::DataType<float>::type, __rf.options.numberOfOutputChannels);
    NChannelsMat result(src.size(), resType, 0);

    CV_INIT_VECTOR(float, scales, {0.5f, 1.0f, 2.0f});
    for (size_t i = 0; i < scales.size(); ++i)
    {
        cv::Mat cSource = __imresize(src, scales[i]*src.size());

        NChannelsMat cResult(cSource.size(), resType, 0);
        detectSingleScale(cSource, cResult);

        result += __imresize(cResult, result.size());
    }
    result /= float(scales.size());

    result.copyTo(_dst.getMat());
}

StructuredEdgeDetection::StructuredEdgeDetection(const std::string &filename)
{
    cv::FileStorage modelFile(filename, cv::FileStorage::READ);

    __rf.options.stride = modelFile["options"]["stride"];
    __rf.options.shrinkNumber = modelFile["options"]["shrinkNumber"];
    __rf.options.patchSize = modelFile["options"]["patchSize"];
    __rf.options.patchInnerSize = modelFile["options"]["patchInnerSize"];

    __rf.options.numberOfGradientOrientations = modelFile["options"]["numberOfGradientOrientations"];
    __rf.options.gradientSmoothingRadius = modelFile["options"]["gradientSmoothingRadius"];
    __rf.options.regFeatureSmoothingRadius = modelFile["options"]["regFeatureSmoothingRadius"];
    __rf.options.ssFeatureSmoothingRadius = modelFile["options"]["ssFeatureSmoothingRadius"];
    __rf.options.gradientNormalizationRadius = modelFile["options"]["gradientNormalizationRadius"];

    __rf.options.selfsimilarityGridSize = modelFile["options"]["selfsimilarityGridSize"];

    __rf.options.numberOfTrees = modelFile["options"]["numberOfTrees"];
    __rf.options.numberOfTreesToEvaluate = modelFile["options"]["numberOfTreesToEvaluate"];

    __rf.options.numberOfOutputChannels =
        2*(__rf.options.numberOfGradientOrientations + 1) + 3;
    //--------------------------------------------

    cv::FileNode childs = modelFile["childs"];
    cv::FileNode featureIds = modelFile["featureIds"];

    std::vector <int> currentTree;

    for(cv::FileNodeIterator it = childs.begin();
        it != childs.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf.childs));
    }

    for(cv::FileNodeIterator it = featureIds.begin();
        it != featureIds.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf.featureIds));
    }

    cv::FileNode thresholds = modelFile["thresholds"];
    std::vector <float> fcurrentTree;

    for(cv::FileNodeIterator it = thresholds.begin();
        it != thresholds.end(); ++it)
    {
        (*it) >> fcurrentTree;
        std::copy(fcurrentTree.begin(), fcurrentTree.end(),
            std::back_inserter(__rf.thresholds));
    }

    cv::FileNode edgeBoundaries = modelFile["edgeBoundaries"];
    cv::FileNode edgeBins = modelFile["edgeBins"];

    for(cv::FileNodeIterator it = edgeBoundaries.begin();
        it != edgeBoundaries.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf.edgeBoundaries));
    }

    for(cv::FileNodeIterator it = edgeBins.begin();
        it != edgeBins.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf.edgeBins));
    }

    __rf.numberOfTreeNodes = int( __rf.childs.size() ) / __rf.options.numberOfTrees;
}