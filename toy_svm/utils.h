#ifndef __UTILS_H
#define __UTILS_H

#include "iostream"
#include "vector"
#include <algorithm>

using namespace std;


void info(char* chars);

inline size_t argmin(vector<double> data)
{
    return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
}

inline size_t argmax(vector<double> data)
{
    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

#endif