#include <vector>
#include "Neuron.cpp"
#include "vector_utils.h"

class Layer
{
private:
	vector<Neuron> neurons;
	int total_neurons = 0;
	bool is_input_layer = false;

public:
	Layer(int total_neurons, vector<double> biases, vector<vector<double>> weights, bool is_input_layer = false) : total_neurons(total_neurons), is_input_layer(is_input_layer)
	{
		if (total_neurons != biases.size())
			throw invalid_argument("The number of biases must be equal to the number of neurons");
		if (total_neurons != weights.size())
			throw invalid_argument("The number of weights must be equal to the number of neurons");
		if (is_input_layer)
		{
			for (int i = 0; i < weights.size(); i++)
			{
				if (weights[i].size() != 1)
					throw invalid_argument("Neurons in input layer can only have weight vector of size 1");
			}
		}
		for (int i = 0; i < total_neurons; i++)
			neurons.push_back(Neuron(weights[i], biases[i], is_input_layer));
	}
	int get_neuron_count() {
		return total_neurons;
	}
	vector<double> compute(vector<double> inputs)
	{
		vector<double> output_vector(total_neurons, 0);
		if (!is_input_layer)
		{
			if (inputs.size() != neurons[0].get_weights().size())
				throw invalid_argument("The number of inputs must be equal to the number of neurons");
			for (int i = 0; i < total_neurons; i++)
			{
				output_vector[i] = neurons[i].compute(inputs);
			}
		}
		else
		{
			if (inputs.size() != total_neurons)
				throw invalid_argument("The number of inputs must be equal to the number of neurons");
			for (int i = 0; i < total_neurons; i++)
			{
				output_vector[i] = neurons[i].compute({inputs[i]});
			}
		}
		return output_vector;
	}
	vector<double> backward_propogation(bool is_output = false) {

	}
	void print_layer_info()
	{
		cout << "Layer info" << endl;
		cout << "Total neurons: " << total_neurons << endl;
		cout << "Is input layer: " << is_input_layer << endl;

		for (int i = 0; i < total_neurons; i++)
		{
			cout << "Neuron " << i << endl;
			cout << "Weights: " << neurons[i].get_weights() << endl;
			cout << "Bias: " << neurons[i].get_bias() << endl;
		}
	}
};