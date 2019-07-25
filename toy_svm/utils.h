#ifndef __UTILS_H
#define __UTILS_H

#include "iostream"
#include "vector"
#include "string"
#include "sstream"
#include <algorithm>

using namespace std;

void info(char* chars);
vector<string> split(string s,char token);

inline size_t argmin(vector<double> data)
{
    return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
}

inline size_t argmax(vector<double> data)
{
    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}


#endif