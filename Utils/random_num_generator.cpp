#pragma once

#include <iostream>
#include <random>

using namespace std;

double generateRandomNumber(double a, double b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(a, b);
    return distr(gen);
}
vector<vector<vector<double>>> dataset_generator(int set_size = 100) {
    vector<vector<vector<double>>> set(2);
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    for (int i = 0; i < set_size; i++) {
        double a1 = generateRandomNumber(0, 0.7);
        double a2 = generateRandomNumber(0, 0.7);
        double o = (a1 * a1) + (a2 * a2);
        inputs.push_back({ a1, a2 });
        outputs.push_back({ o });
    }
    set[0] = inputs;
    set[1] = outputs;
    return set;
}