#ifndef __TOYSVM_H
#define __TOYSVM_H

#include "utils.h"
#include "vector"

using namespace std;

extern int toy_svm_version;

struct dataItem {
    int index;
    double value;
};

struct svm_problem {
    vector<int> labels;
    vector<vector<dataItem> > trains;
};

class Kernel {
public:
    Kernel() {};

    static double k_function(const vector<dataItem> data1, const vector<dataItem> data2);
};

double Kernel::k_function(const vector<dataItem> data1, const vector<dataItem> data2) {
    // Linear Kernel Function
    double sum = 0;

    auto iter1 = data1.begin();
    auto iter2 = data2.begin();
    for (; iter1 != data1.end() && iter2 != data2.end(); ) {
        if (iter1->index == iter2->index) {
            sum += iter1->value * iter2->value;
            ++iter1, ++iter2;
        } else {
            if (iter1->index > iter2->index) ++iter2;
            else ++iter1;
        }
    }
    return sum;
}

#endif