#include <string>
#include <stack>
#include <vector>
#include <fstream>
#include <cmath>
#include <ctime>

#include <cv.h>
#include <highgui.h>

#include <mat.h>
#include <mex.h>

#include "thirdparty/MxArray.h"

#include <structuredEdgeDetection.h>

#define INPUT_ARGS(N, M) \
    if (nrhs < (N)) { \
    mexErrMsgTxt("INPUT_ARGS(_N_, M)"); \
    return; \
    } \
    \
    if (nrhs > (M)) { \
    mexErrMsgTxt("INPUT_ARGS(N, _M_)"); \
    return; \
    }

#define OUTPUT_ARGS(N, M) \
    if (nlhs < (N)) { \
    mexErrMsgTxt("OUTPUT_ARGS(_N_, M)"); \
    return; \
    } \
    \
    if (nlhs > (M)) { \
    mexErrMsgTxt("OUTPUT_ARGS(N, _M_)"); \
    return; \
    }

MEXFUNCTION_LINKAGE void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs != 1) mexErrMsgTxt("nlhs != 1");
    if (nrhs != 2) mexErrMsgTxt("nrhs != 2");

    cv::Mat src = MxArray(prhs[0]).toMat();
    src.convertTo(src, cv::DataType<float>::type);
    cv::cvtColor(src, src, CV_BGR2RGB);

    std::string modelFile = MxArray(prhs[1]).toString();
    StructuredEdgeDetection img2edges(modelFile);

    cv::Mat edges;
    img2edges.detectSingleScale(src, edges);

    edges.convertTo(edges, cv::DataType<double>::type);

    plhs[0] = MxArray(edges);
}   