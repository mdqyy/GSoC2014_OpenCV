/**
*  \file structuredEdgeDetection.h
*  \brief implementation of structured forests for fast edge detection, for details look in original paper
*/

#ifndef structuredEdgeDetection_H
#define structuredEdgeDetection_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#ifndef CV_SQR
#  define CV_SQR(x)  ((x)*(x))
#endif

#define CV_INIT_VECTOR(type, name, ...) \
    static const type name##_a[] = __VA_ARGS__; \
    std::vector <type> name(name##_a, \
    name##_a + sizeof(name##_a) / sizeof(*name##_a))

#define CV_INC_IF_EVEN(x) ((x) + !((x)&1))

typedef cv::Mat NChannelsMat;

struct RandomForestOptions
{
    //----------------------------------------------------------
    // model params

    int numberOfOutputChannels; // number of edge orientation bins for output

    int patchSize;      // width of image patches
    int patchInnerSize; // width of patch predicted part
    //----------------------------------------------------------

    // feature params

    int regFeatureSmoothingRadius;    // radius for smoothing of regular features
    // (using convolution with triangle filter

    int ssFeatureSmoothingRadius;     // radius for smoothing of additional features
    // (using convolution with triangle filter)

    int shrinkNumber;                 // amount to shrink channels

    int numberOfGradientOrientations; // number of orientations per gradient scale

    int gradientSmoothingRadius;      // radius for smoothing of gradients
    // (using convolution with triangle filter)

    int gradientNormalizationRadius;  // gradient normalization radius
    int selfsimilarityGridSize;       // number of self similarity cells

    //----------------------------------------------------------
    // detection params

    int numberOfTrees;            // number of trees in forest to train
    int numberOfTreesToEvaluate;  // number of trees to evaluate per location

    int stride;                   // stride at which to compute edges
};

struct RandomForest
{
    RandomForestOptions options;

    int numberOfTreeNodes;

    std::vector <int> featureIds;     // feature coordinate thresholded at k-th node
    std::vector <float> thresholds;   // threshold applied to featureIds[k] at k-th node
    std::vector <int> childs;         // k --> child[k] - 1, child[k]

    std::vector <int> edgeBoundaries; // ...
    std::vector <int> edgeBins;       // ...
};

class StructuredEdgeDetection
{
public:
    RandomForest __rf; // random forest trained to detect edges

    cv::Mat __imresize(const cv::Mat &img, const cv::Size &sizeDst);
    cv::Mat __imsmooth(const cv::Mat &img, const int rad);
    // image smoothing, authors used triangle convolution

    void __imhog(const cv::Mat &img, cv::Mat &magnitude, cv::Mat &histogram,
        const int numberOfBins, const int sizeOfPatch,
        const int gradientNormalizationRadius);
    // gradient magnitude, histogram of gradient orientations

    void __getFeatures(const cv::Mat &img, NChannelsMat &features);
    // extracting features for __rf from img

    void __detectEdges(const NChannelsMat &features, cv::Mat &dst);
    // edge detection

    //----------------------------------------------------------

    void detectSingleScale(cv::InputArray src, cv::OutputArray dst);
    // detect edges in src, dst  is vector of matrices
    // with edges probabilities for each edge orientation

    void detectMultipleScales(cv::InputArray src, cv::OutputArray dst);
    // detect edges in {0.5, 1, and 2}-times scaled source image, then average

    StructuredEdgeDetection(const std::string &filename);
    // load options and forest from filename

    virtual ~StructuredEdgeDetection() {};
};

#endif
