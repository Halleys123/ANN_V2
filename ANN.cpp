#include "MLP.cpp"
#include "vector_utils.h"

#include <iostream>

using namespace std;

int main()
{
    try
    {
        int size_of_one_eepoch = 1;
        int total_layers = 3;

        vector<int> nodes_in_each_layer = { 1, 3, 1 };
        vector<vector<double>> biases = {
            {0},
            {0.5399346351623535, 8.194790840148926, 0.16368073225021362},
            {4.503326416015625}
        };
        vector<vector<vector<double>>> weights = {
            {{1}},
            {{-5.293692588806152}, {-7.686620712280273}, {2.4524362087249756}},
            {{-5.90725040435791, -8.005425453186035, 3.714153289794922}}
        };

        vector<vector<double>> desired_outputs = { {0.8} };
        vector<vector<double>> inputs = { { 1 } };

        MLP mlp(total_layers, nodes_in_each_layer, weights, biases);
        cout << mlp.compute({0.8}) << endl;

        mlp.train(size_of_one_eepoch, inputs, desired_outputs);
        return 0;

    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}