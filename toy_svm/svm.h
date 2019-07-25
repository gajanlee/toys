#ifndef __TOYSVM_H
#define __TOYSVM_H

#include "iostream"
#include "vector"

#include "utils.h"

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



class Solver {
public:
    Solver() {};

    void Solve(const svm_problem &problem);
    void initialize_arguments();
    bool select_working_set_free(int &first, int &second);
    bool select_working_set_entire(int &first, int &second);
    bool update_alpha_pair(const int first, const int second);
    void save_model();

    double caculate_error(int index);
    double caculate_gx(int index);

    double predict(vector<dataItem> item);

private:
    double bias;    // rho
    double C;
    vector<double> alphas;
    vector<double> error_cache;

    svm_problem problem;

    bool is_free_alpha(int index) { return (alphas[index] > 0 && alphas[index] < C); }
    double get_alpha(int index) { return alphas[index]; }
    vector<dataItem> get_data(int index) { return problem.trains[index]; }
    int get_label(int index) { return problem.labels[index]; }
    double get_error(int index) { return error_cache[index]; }
    size_t train_size() { return problem.labels.size(); }
    

    double clipAlpha(double new_unc, double L, double H);

    void update_alpha(int index, double alpha_new);
    void update_error(int index);
    void update_bias(double bias_new);

    bool satisfiedKKT(int index);
};

#endif