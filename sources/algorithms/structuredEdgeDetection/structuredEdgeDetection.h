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


typedef cv::Mat NChannelsMat;

struct RandomForestOptions
{
    //----------------------------------------------------------
    // model params

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
                                      
    int numberOfOutputChannels;       // number of edge orientation bins for output

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

    std::vector <unsigned int> featureIds;
    std::vector <unsigned int> thresholds;
    std::vector <unsigned int> childs;

    ...
};

class StructuredEdgeDetection
{
public:
    RandomForest __rf; // random forest trained to detect edges

    cv::Mat __imresize(const cv::Mat &src, const cv::Size &sizeDst);
    cv::Mat __imsmooth(const cv::Mat &img, const int rad); 
    // image smoothing, authors used triangle convolution

    void __getFeatures(const cv::Mat &img, NChannelsMat &features);
    // extracting features for __classifier from img

    void __detectEdges(const NChannelsMat &features, cv::Mat &dst);
    // edge detection

    //----------------------------------------------------------
    
    void detectSingleScale(cv::InputArray src, cv::OutputArray dst);     
    // detect edges in src, dst  is vector of matrices 
    // with edges probabilities for each edge orientation

    void detectMultipleScales(cv::InputArray src, cv::OutputArray dst);  
    // detect edges in {0.5, 1, and 2}-times scaled source image, then average

    void train(); // ...

    void save(const std::string &filename); 
    // serialize options and __classifier into filename

    void load(const std::string &filename); 
    // load options and __classifier from filename

    StructuredEdgeDetection(); 
    StructuredEdgeDetection(const std::string &filename); 
    // load options and __classifier from filename 

    virtual ~StructuredEdgeDetection() {};
};

#endif
