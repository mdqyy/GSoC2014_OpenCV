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

class StructuredEdgeDetection
{
public:
    CvRTrees __classifier; // random forest trained to detect edges

    void __imresize(const cv::Mat &src, cv::Mat &dst, const cv::Size dstSize);
    void __imresize(const cv::Mat &src, cv::Mat &dst, const float resizeFactor);

    void __imresize(const cv::Mat &img, const cv::Size dstSize);
    void __imresize(const cv::Mat &img, const float resizeFactor);

    void __imsmooth(const cv::Mat &img, const int radius); 
    void __imsmooth(const cv::Mat &src, const cv::Mat &dst, const int radius); 
    // image smoothing, authors used triangle convolution

    void __getFeatures(const cv::Mat &img, std::vector <cv::Mat> &regularFeatures, 
                       std::vector <cv::Mat> &additionalFeatures);
    // extracting features for __classifier from img

    //----------------------------------------------------------

    bool __non_maximum_supression__; // if true apply non-maximum suppression to edges    
    
    int __stride_width__;            // stride at which to compute edges    
    int __shrink_number__;           // amount to shrink channels
    int __patch_width__;             // width of image patches
    int __gradient_orientations__;   // number of orientations per gradient scale
                                     
    int __gradient_smoothing__;      // radius for smoothing of gradients
                                     // (using convolution with triangle filter)
    
    int __reg_features_smoothing__;  // radius for smoothing of regular features
                                     // (using convolution with triangle filter
    
    int __nonreg_features_smoothing__;   // radius for smoothing of additional features
                                         // (using convolution with triangle filter)
    
    int __gradient_normalization__; // gradient normalization radius
    int __selfsimilarity_cells__;   // number of self similarity cells
    
    int __number_of_trees__;              // number of trees in forest to train
    int __number_of_trees_to_evaluate__;  // number of trees to evaluate per location
    
    int __edge_orientations__; // number of edge orientation bins for output

    //----------------------------------------------------------

    StructuredEdgeDetection(); 
    
    StructuredEdgeDetection(const std::string filename); 
    // load options and __classifier from filename 
    
    ~StructuredEdgeDetection();
    
    void save(const std::string filename); 
    // serialize options and __classifier into filename
    
    void load(const std::string filename); 
    // load options and __classifier from filename

    void train(); // ...

    void detectSingleScale(cv::InputArray src,   
                           cv::OutputArrayOfArrays dst);     
    // detect edges in src, dst  is vector of matrices 
    // with edges probabilities for each edge orientation

    void detectMultipleScales(cv::InputArray src,   
                              cv::OutputArrayOfArrays dst);  
    // detect edges in {0.5, 1, and 2}-times scaled source image, then average
};

#endif
