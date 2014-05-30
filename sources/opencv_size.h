#ifndef OPENCV_SIZE_H
#define OPENCV_SIZE_H

cv::Size operator * (const float x, const cv::Size &arg)
{
    return cv::Size(cvRound(x*arg.width), cvRound(x*arg.height));
}

cv::Size operator * (const cv::Size &arg, const float x)
{
    return cv::Size(cvRound(x*arg.width), cvRound(x*arg.height));
}

cv::Size operator / (const cv::Size &arg, const float x)
{
    return cv::Size(cvRound(arg.width/x), cvRound(arg.height/x));
}

bool operator <= (const cv::Size &arg1, const cv::Size &arg2)
{
    return arg1.width <= arg2.width && arg1.height <= arg2.height;
}

bool operator >= (const cv::Size &arg1, const cv::Size &arg2)
{
    return arg1.width >= arg2.width && arg1.height >= arg2.height;
}

bool operator < (const cv::Size &arg1, const cv::Size &arg2)
{
    return arg1.width < arg2.width && arg1.height < arg2.height;
}

bool operator > (const cv::Size &arg1, const cv::Size &arg2)
{
    return arg1.width > arg2.width && arg1.height > arg2.height;
}

bool operator <= (const cv::Size &arg1, const float x)
{
    return arg1.width <= x && arg1.height <= x;
}

bool operator >= (const cv::Size &arg1, const float x)
{
    return arg1.width >= x && arg1.height >= x;
}

bool operator < (const cv::Size &arg1, const float x)
{
    return arg1.width < x && arg1.height < x;
}

bool operator > (const cv::Size &arg1, const float x)
{
    return arg1.width > x && arg1.height > x;
}

#endif