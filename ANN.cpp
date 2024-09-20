#include "MLP.cpp"
#include "vector_utils.h"
#include "random_num_generator.h"

#include <iostream>

using namespace std;

int main()
{
    try
    {
        TrainingParameters params;

        params.presentations_per_eepoch = 10;
        params.initial_learning_rate = 0.5;
        params.patience_limit = 150;
        params.early_stop_threshold = 200;
        params.target_error = 0.001;
        //params.min_learning_rate = 1e-3;
        params.min_learning_rate = 0.5;
        params.min_error_improvement = 1e-5;
        params.stable_error_threshold = 1e-3;
        //params.show_error = false;

        vector<int> nodes_in_each_layer = { 2, 3, 3, 1 };

        vector<vector<vector<double>>> set = dataset_generator(params.presentations_per_eepoch);

        vector<vector<double>> inputs = set[0];
        vector<vector<double>> desired_outputs = set[1];

        //cout << inputs << endl;
        //cout << desired_outputs << endl;

        MLP mlp(nodes_in_each_layer.size(), nodes_in_each_layer);

        mlp.train(params, inputs, desired_outputs);

        cout << mlp.compute({0.4, 0.3}) << endl;
        while (true) {
            double a, b;
            cin >> a;
            cin >> b;
            vector<double> c = mlp.compute({ a, b });
            printf("%f^2 + %f^2 = %f\n", a, b, c[0]);
        }
        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}