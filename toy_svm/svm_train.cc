#include "fstream"
#include "iostream"
#include "string"
#include "vector"

#include "svm.h"
#include "utils.h"

using namespace std;

svm_problem load_data_items(string filename) {
    string line;
    ifstream file (filename);

    svm_problem problem;
    vector<vector<dataItem>> trains;
    vector<int> labels;

    if (file.is_open()) {
        while (getline(file, line)) {
            vector<dataItem> items;

            vector<string> data_label = split(line, '\t');
            vector<string> datas = split(data_label[0], ' ');
            int label = atoi(data_label[1].c_str());
            
            for (auto data: datas) {
                vector<string> index_value = split(data, ':');
                auto item = dataItem{atoi(index_value[0].c_str()), 
                                atof(index_value[1].c_str())};
                //auto item = dataItem{1, 1.5};
                items.push_back(item);
            }

            trains.push_back(items);
            labels.push_back(label);
        }
    } else 
        cout << "Unable to open file " << filename << endl; 

    /*for (auto items: trains) {
        for (auto item: items) {
            cout << item.index << " " << item.value << " ";
        }
        cout << endl;
    }
    for (auto label: labels) {
        cout << label << endl;
    }*/
    problem.trains = trains;
    problem.labels = labels;

    return problem;
}

int main() {
    auto problem = load_data_items("data.svm");

    auto solver = Solver();
    solver.Solve(problem);

    auto positive_items = vector<dataItem>{
        dataItem{0, 1},
        dataItem{1, 4},
    };
    auto negative_items = vector<dataItem>{
        dataItem{0, 1},
        dataItem{1, 1.5},
    };

    cout << "positive sample predict result is " << solver.predict(positive_items) << endl;
    cout << "negative sample predict result is " << solver.predict(negative_items) << endl;
    return 0;
}