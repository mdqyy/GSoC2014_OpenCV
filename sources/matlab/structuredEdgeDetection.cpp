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
		mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", \
                          "INPUT_ARGS(_N_, M)"); \
		return; \
	} \
    \
	if (nrhs > (M)) { \
		mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", \
                          "INPUT_ARGS(N, _M_)"); \
		return; \
	}

#define OUTPUT_ARGS(N, M) \
	if (nlhs < (N)) { \
		mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", \
                          "OUTPUT_ARGS(_N_, M)"); \
		return; \
	} \
    \
	if (nlhs > (M)) { \
		mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", \
                          "OUTPUT_ARGS(N, _M_)"); \
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
//        mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", 
//                          "nrhs == 0");
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
//            mexErrMsgIdAndTxt("structuredEdgeDetection: wrong_call", 
//                              "switch (cmd): default");
//            break;
//    }
//}

MEXFUNCTION_LINKAGE void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    StructuredEdgeDetection img2edges;
    img2edges.__convertFromMatlab("model.matlab.xml", "model.opencv.xml");
}
