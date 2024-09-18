#include "Layer.cpp"
#include "vector_utils.h"
#include <string>
#include <vector>

class MLP
{
private:
	int total_layers = 0;
	double learning_rate = 0.5;
	double permissible_error = 0.01;

	bool show_error = false;

	vector<vector<double>> output_from_each_layer;
	vector<int> neuron_in_each_layer;
	vector<Layer> layers;

public:
	MLP(int total_layers, vector<int> neuron_in_each_layer, vector<vector<vector<double>>> weights = {}, vector<vector<double>> biases = {}) : neuron_in_each_layer(neuron_in_each_layer), total_layers(total_layers)
	{
		if (neuron_in_each_layer.size() < 3)
			throw invalid_argument("The number of layers must be greater than 2");

		if (weights.empty() && biases.empty()) {
			for (int i = 0; i < total_layers; i++) {
				vector<vector<double>> w_l;
				vector<double> b_l;
				if (i == 0) {
					for (int j = 0; j < neuron_in_each_layer[i]; j++) {
						b_l.push_back(0);
					}
					for (int j = 0; j < neuron_in_each_layer[i]; j++) {
						vector<double> w(1, 1);
						w_l.push_back(w);
					}
				}
				else {
					for (int j = 0; j < neuron_in_each_layer[i]; j++) {
						b_l.push_back(0);
					}
					for (int j = 0; j < neuron_in_each_layer[i]; j++) {
						vector<double> w(neuron_in_each_layer[i - 1]);
						for (int k = 0; k < neuron_in_each_layer[i - 1]; k++) {
							w[k] = 0.2;
						}
						w_l.push_back(w);
					}
				}
				weights.push_back(w_l);
				biases.push_back(b_l);
			}
		}
		else {
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
		}

		for (int i = 0; i < total_layers; i++)
			layers.push_back(Layer(neuron_in_each_layer[i], biases[i], weights[i], i == 0));
	}
	vector<double> forward_propogation(vector<double> input)
	{
		vector<vector<double>> outputs(total_layers);
		if (input.size() != neuron_in_each_layer[0])
			throw invalid_argument("Input vector should be of size " + to_string(neuron_in_each_layer[0]) + " but is of size " + to_string(input.size()));
		vector<double> input_to_ith_layer = input;
		for (int layer_no = 0; layer_no < total_layers; layer_no++)
		{
			input_to_ith_layer = layers[layer_no].compute(input_to_ith_layer);
			outputs[layer_no] = input_to_ith_layer;
		}
		this->output_from_each_layer = outputs;
		return input_to_ith_layer;
	}
	void backward_propagation(const vector<double>& desired_output) {

		vector<vector<double>> deltas(total_layers, vector<double>());

		// Calculate delta for output layer
		deltas[total_layers - 1].resize(layers[total_layers - 1].get_neuron_count());

		for (int i = 0; i < layers[total_layers - 1].get_neuron_count(); ++i) {
			double output = output_from_each_layer[total_layers - 1][i];
			deltas[total_layers - 1][i] = (desired_output[i] - output) * unipolar_sigmoid_derivative(output);
		}

		// Calculate delta for hidden layers
		for (int l = total_layers - 2; l > 0; --l) {
			deltas[l].resize(layers[l].get_neuron_count());
			for (int i = 0; i < layers[l].get_neuron_count(); ++i) {
				double error = 0.0;
				for (int j = 0; j < layers[l + 1].get_neuron_count(); ++j) {
					error += deltas[l + 1][j] * layers[l + 1].get_weights(j)[i];
				}
				deltas[l][i] = error * unipolar_sigmoid_derivative(output_from_each_layer[l][i]);
			}
		}

		// Update weights and biases
		for (int l = 1; l < total_layers; ++l) {
			for (int i = 0; i < layers[l].get_neuron_count(); ++i) {
				for (int j = 0; j < layers[l - 1].get_neuron_count(); ++j) {
					double delta_weight = learning_rate * deltas[l][i] * output_from_each_layer[l - 1][j];
					//printf("Layers: %d\nNeuron %d from last layer and %d from current\nUpdate: %f\nOutput from neuron %d = %f\nDelta %d %d: %f\n----------------\n", l, j, i, delta_weight, j, output_from_each_layer[l - 1][j], l, i, deltas[l][i]);
					layers[l].update_weight(i, j, layers[l].get_weights(i)[j] + delta_weight);
				}
				double delta_bias = learning_rate * deltas[l][i];
				layers[l].set_bias(i, layers[l].get_bias(i) + delta_bias);
			}
		}
	}
	void toggle_show_error(bool val) {
		this->show_error = val;
	}
	void train(int size_of_eepoch, vector<vector<double>> presentations, vector<vector<double>> desired_outputs,
		double training_rate = 0.5, double permissible_error = 0.01,
		double early_stopping_threshold = 500, double min_error_delta = 1e-5,
		double min_learning_rate = 1e-5, double tolerance_threshold = 1e-3,
		int tolerance_count = 100, double dynamic_reduction_factor = 0.05)
	{
		this->learning_rate = training_rate;
		this->permissible_error = permissible_error;

		double this_eepoch_error = INT_MAX;
		double previous_eepoch_error = INT_MAX;
		// Times ANN got similar error value
		int stable_error_count = 0;
		// Counts how many times did ANN get similar value in order to stop learning based on early_stopping_threshold
		int early_stopping_counter = 0;
		double initial_learning_rate = training_rate;

		while (this_eepoch_error > permissible_error && early_stopping_counter < early_stopping_threshold)
		{
			this_eepoch_error = 0;

			for (int num = 0; num < size_of_eepoch; num++)
			{
				vector<double> output = forward_propogation(presentations[num]);
				double sample_error = 0.0;
				for (int i = 0; i < output.size(); i++) {
					sample_error += (desired_outputs[num][i] - output[i]) * (desired_outputs[num][i] - output[i]);
				}

				this_eepoch_error += sample_error;
				backward_propagation(desired_outputs[num]);
			}
			this_eepoch_error /= size_of_eepoch;
			if(show_error)
				cout << "Error: " << this_eepoch_error << endl;

			// min error delta means how much the error should at least fall so that early stopping counter don't increase.
			if (fabs(previous_eepoch_error - this_eepoch_error) < min_error_delta) {
				early_stopping_counter++;
			}
			else {
				early_stopping_counter = 0; 
			}

			// tolerance threshold means how much should be the error difference in order for stable error count to increase
			// which in turn will decrease the learning rate;
			if (fabs(previous_eepoch_error - this_eepoch_error) < tolerance_threshold) {
				stable_error_count++;
				if (stable_error_count >= tolerance_count) {
					double reduction = 1.0 - (dynamic_reduction_factor * stable_error_count);
					this->learning_rate = max(this->learning_rate * reduction, min_learning_rate);
					cout << "Learning rate reduced to: " << this->learning_rate << endl;
					stable_error_count = 0;
				}
			}
			else {
				stable_error_count = 0;
			}
			previous_eepoch_error = this_eepoch_error;
		}

		if (early_stopping_counter >= early_stopping_threshold) {
			cout << "Early stopping triggered after " << early_stopping_counter << " epochs of no significant improvement." << endl;
		}
	}
	vector<double> compute(vector<double> input)
	{
		return forward_propogation(input);
	}
	void print_mlp() {
		for (int i = 0; i < total_layers; i++) {
			layers[i].print_layer_info();
			cout << "-----------------------------------" << endl;
		}
	}
};