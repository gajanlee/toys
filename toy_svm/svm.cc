#include "svm.h"


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


void Solver::Solve(const svm_problem &problem) {

    this->problem = problem;
    initialize_arguments();
    int MAX_ITER = 100;

    for(int iter = 0; iter < MAX_ITER; ++iter) {
        int first, second;

        // 不在select里面判断了，select只负责找first
        // 在update里面判断
        if (select_working_set_free(first, second) == false) {
            if (select_working_set_entire(first, second) == false) {
                // 整个训练集都符合KKT条件
                break;
            }
        }
        
        update_alpha_pair(first, second);
    }

    save_model();
}

void Solver::initialize_arguments() {
    bias = 0, C = 1.0;
    alphas.resize(train_size());
    error_cache.resize(train_size());
    for (size_t i = 0; i < train_size(); ++i) {
        alphas[i] = 1;
    }
    for (size_t i = 0; i < train_size(); ++i) {
        error_cache[i] = caculate_error(i);
    }
}

//
// select_working_set_free
// Find the alpha which 0 < alpha < C
//
bool Solver::select_working_set_free(int &first, int &second) {
    first = -1;
    for (size_t i = 0; i < alphas.size(); ++i) {
        if (is_free_alpha(i) && !satisfiedKKT(i)) {
            first = i; break;
        }
    }
    if (first == -1) { return false; }
    
    double error1 = error_cache[first];
    second = (error1 >= 0) ? argmin(error_cache) : argmax(error_cache);
    return true;
}

bool Solver::select_working_set_entire(int &first, int &second) {
    first = -1;
    for (size_t i = 0; i < alphas.size(); ++i) {
        if (!satisfiedKKT(i)) {
            first = i; break;
        }
    }
    if (first == -1) { return false; }
    
    double error1 = error_cache[first];
    second = (error1 >= 0) ? argmin(error_cache) : argmax(error_cache);
    return true;
}

// TODO: message
bool Solver::update_alpha_pair(const int first, const int second) {
    double alpha1_old = get_alpha(first), alpha2_old = get_alpha(second);
    int y_1 = get_label(first), y_2 = get_label(second);

    double L = ( (y_1 != y_2) ? max(0., alpha2_old-alpha1_old) : max(0., alpha2_old+alpha1_old-C) );
    double H = ( (y_1 != y_2) ? min(C, C+alpha2_old-alpha1_old) : min(C, alpha2_old+alpha1_old) );

    if (L == H) { return false; }

    // K11 + K22 - 2*K12
    double eta = Kernel::k_function(get_data(first), get_data(first)) + \
                Kernel::k_function(get_data(second), get_data(second)) - \
                Kernel::k_function(get_data(first), get_data(second)) * 2;
    
    if (eta <= 0) { return false; }

    // error_old
    double error1 = get_error(first), error2 = get_error(second);

    double alpha2_new_unc = alpha2_old + (y_2 * (error1-error2) / eta);
    double alpha2_new = clipAlpha(alpha2_new_unc, L, H);
    update_alpha(second, alpha2_new);
    update_error(second);

    if (abs(alpha2_new - alpha2_old) < 1e-5) { return false; }
    
    double alpha1_new = alpha1_old + (y_1 * y_2 * (alpha2_old-alpha1_old));
    update_alpha(first, alpha1_new);
    update_error(first);

    double bias_1_new = -error1 - y_1*Kernel::k_function(get_data(first), get_data(first))*(alpha1_new-alpha1_old) - \ 
                                y_2*Kernel::k_function(get_data(second), get_data(first))*(alpha2_new-alpha2_old);
    double bias_2_new = -error2 - y_1*Kernel::k_function(get_data(first), get_data(second))*(alpha1_new-alpha1_old) - \
                                y_2*Kernel::k_function(get_data(second), get_data(second))*(alpha2_new-alpha2_old);

    if ((0 < alpha1_new) && (alpha1_new < C)) {
        update_bias(bias_1_new);
    } else if ((0 < alpha2_new) && (alpha2_new < C)) {
        update_bias(bias_2_new);
    } else {
        update_bias((bias_1_new + bias_2_new) / 2.);
    }
}

void Solver::update_alpha(int index, double alpha_new) {
    alphas[index] = alpha_new;
}

void Solver::update_error(int index) {
    error_cache[index] = caculate_error(index);
}

void Solver::update_bias(double bias_new) {
    bias = bias_new;
}

double Solver::caculate_error(int index) {
    double error = caculate_gx(index) - get_label(index);
    return error;
}

double Solver::caculate_gx(int index) {
    double gx = 0;
    for (size_t i = 0; i < alphas.size(); ++i) {
        gx += alphas[i] * get_label(i) * Kernel::k_function(get_data(i), get_data(index));
    }
    return gx;
}

double Solver::clipAlpha(double new_unc, double L, double H) {
    if (new_unc > H) {
        return H;
    } else if(new_unc < L) {
        return L;
    }
    return new_unc;
}

bool Solver::satisfiedKKT(int index) {
    double yi_gx = get_label(index) * caculate_gx(index);
    if (alphas[index] == 0) {
        return yi_gx >= 1;
    } else if (alphas[index] > 0 && alphas[index] < C) {
        return yi_gx == 1;
    } else if (alphas[index] == C) {
        return yi_gx <= 1;
    }
}

void Solver::save_model() {
    
}