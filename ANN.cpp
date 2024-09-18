#include "MLP.cpp"
#include "vector_utils.h"
#include "random_num_generator.h"

#include <iostream>

using namespace std;

int main()
{
    try
    {
        int size_of_one_eepoch = 500;

        vector<int> nodes_in_each_layer = { 2, 3, 1 };

        vector<vector<vector<double>>> set = dataset_generator(size_of_one_eepoch);

        vector<vector<double>> inputs = set[0];
        vector<vector<double>> desired_outputs = set[1];

        MLP mlp(nodes_in_each_layer.size(), nodes_in_each_layer);
        mlp.train(size_of_one_eepoch, inputs, desired_outputs, 2, 0.0001, 500, 1e-5, 1e-8);

        cout << mlp.compute({0.5, 0.6}) << endl;
        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}