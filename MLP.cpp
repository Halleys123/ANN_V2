#include "Layer.cpp"
#include "vector_utils.h"
#include <string>
#include <vector>

class MLP
{
private:
	int total_layers = 0;
	vector<int> neuron_in_each_layer;
	vector<Layer> layers;

public:
	MLP(int total_layers, vector<int> neuron_in_each_layer, vector<vector<vector<double>>> weights, vector<vector<double>> biases) : neuron_in_each_layer(neuron_in_each_layer), total_layers(total_layers)
	{
		if (neuron_in_each_layer.size() < 3)
			throw invalid_argument("The number of layers must be greater than 2");
		if (biases.size() != total_layers)
			throw invalid_argument("Size of bias vector and total layers don't match");
		if (weights.size() != total_layers)
			throw invalid_argument("Size of weight vector and total layers don't match");
		for (int i = 0; i < total_layers; i++)
		{
			if (biases[i].size() != neuron_in_each_layer[i])
				throw invalid_argument("Invalid bias vector\nLayer number " + to_string(i) + " should have a bias vector of size " + to_string(neuron_in_each_layer[i]) + " but is " + to_string(biases[i].size()));
			if (weights[i].size() != neuron_in_each_layer[i])
				throw invalid_argument("Invalid weight vector\nLayer number " + to_string(i) + " should have a weight vector of size " + to_string(neuron_in_each_layer[i]) + " but is " + to_string(weights[i].size()));
			if (i == 0)
				continue;
			for (int neuron = 0; neuron < neuron_in_each_layer[i]; neuron++)
			{
				if (weights[i][neuron].size() != weights[i - 1].size())
					throw invalid_argument("Invalid weight vector\nLayer Number " + to_string(i + 1) + ", node number " + to_string(neuron + 1) + " should be of size " + to_string(weights[i - 1].size()) + " but is " + to_string(weights[i][neuron].size()));
			}
		}

		for (int i = 0; i < total_layers; i++)
			layers.push_back(Layer(neuron_in_each_layer[i], biases[i], weights[i], i == 0));
	}
	vector<double> forward_propogation(vector<double> input)
	{
		if (input.size() != neuron_in_each_layer[0])
			throw invalid_argument("Input vector should be of size " + to_string(neuron_in_each_layer[0]) + " but is of size " + to_string(input.size()));
		vector<double> input_to_ith_layer = input;
		for (int layer_no = 0; layer_no < total_layers; layer_no++)
		{
			input_to_ith_layer = layers[layer_no].compute(input_to_ith_layer);
		}
		return input_to_ith_layer;
	}
	void backward_propogation(vector<double> e_l, bool is_output_layer = false, vector<double>& output_from_last_layer, vector<double>& desired_output)
	{		
		for (int i = total_layers - 1; i > 0; i--) {

		}
	}
	void train(int size_of_eepoch, vector<vector<double>> inputs, vector<vector<double>> desired_outputs)
	{
		double this_eepoch_error = INT_MAX;
		double permissible_error = 0.01;
		int j = 0;
		while (j < 1)
		{
			this_eepoch_error = 0;
			for (int i = 0; i < size_of_eepoch; i++)
			{
				vector<double> output_from_last_layer = forward_propogation(inputs[i]);
				backward_propogation({}, i == total_layers - 1, output_from_last_layer, desired_outputs[i]);
			}
		}
	}
	vector<double> compute(vector<double> input)
	{
		return forward_propogation(input);
	}
};