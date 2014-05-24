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


//vector <structuredEdgeDetection *> objects;
//int creations = 0;
//int deletions = 0;
//
//enum COMMAND
//{
//    CREATE   = 1,
//    DELETE   = 2,
//    CLEAR    = 3
//};
//
//void cmdCreate(int nlhs, mxArray *plhs[],
//               int nrhs, const mxArray *prhs[])
//{
//    INPUT_ARGS(3, 3)
//    OUTPUT_ARGS(1, 1)
//
//    ...
//    creations++;
//}
//
//void deleteo(int nrhs, const mxArray *prhs[])
//{
//    INPUT_ARGS(2, 2)
//
//    ...
//    deletions++;
//}
//
//void cmdClear()
//{
//    for(int i = 0; i < objects.size(); ++i)
//        delete [] objects[i];
//    creations = 0;
//    deletions = 0;
//    objects.resize(0);
//}
//
//void mexFunction(int nlhs, mxArray *plhs[],
//                 int nrhs, const mxArray *prhs[])
//{
//    if (nrhs == 0) {
//        mexErrMsgTxt("nrhs == 0");
//        return;
//    }
//    COMMAND cmd = (COMMAND)(int) *( (double *) mxGetPr(prhs[0]) );
//
//    switch (cmd) {
//        case CREATE:
//            cmdCreate(nlhs,plhs,nrhs,prhs);
//            break;
//
//        case DELETE:
//            cmdDelete(nrhs,prhs);
//            break;
//
//        case CLEAR:
//            cmdClear();
//            break;
//
//        default:
//            mexErrMsgTxt("switch (cmd): default");
//            break;
//    }
//}

MEXFUNCTION_LINKAGE void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs != 1) mexErrMsgTxt("nlhs != 1");
    if (nrhs != 1) mexErrMsgTxt("nrhs != 1");

    if (mxIsComplex(prhs[0]) || mxIsSparse(prhs[0]) || !mxIsClass(prhs[0], "double"))
        mexErrMsgTxt("mxIsComplex(prhs[0]) || mxIsSparse(prhs[0]) || !mxIsClass(prhs[0], \"double\")");

    const size_t sizeOfSizes = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *sizes = mxGetDimensions(prhs[0]);

    if (sizeOfSizes < 2 || sizeOfSizes > 3 || (sizeOfSizes == 3 && sizes[2] != 3))
        mexErrMsgTxt("sizeOfSizes < 2 || sizeOfSizes > 3 || (sizeOfSizes == 3 && sizes[2] != 3)");

    cv::Mat src = MxArray(prhs[0]).toMat();
    src.convertTo(src, cv::DataType<float>::type);
    cv::cvtColor(src, src, CV_BGR2RGB);

    StructuredEdgeDetection img2edges;

    cv::Mat edges;
    img2edges.detectSingleScale(src, edges);

    edges.convertTo(edges, cv::DataType<double>::type);
    plhs[0] = MxArray(edges);
}
