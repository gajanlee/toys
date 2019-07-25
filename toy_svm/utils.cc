#include "utils.h"

void info(char* chars) {
    cout << chars << endl;
}

vector<string> split(string s,char token) {
    istringstream iss(s);
    string word;
    vector<string> vs;
    while(getline(iss, word, token)) {
        vs.push_back(word);
    }
    return vs;
}