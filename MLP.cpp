#include "Layer.cpp"
#include "vector_utils.h"
#include "random_num_generator.h"
#include <string>
#include <conio.h>
#include <vector>

struct TrainingParameters {
	int presentations_per_eepoch = 1000;                          // Number of training epochs (reasonable default: 1000)
	double initial_learning_rate = 0.01;            // Learning rate starts low (0.001 to 0.01 for most cases)
	double target_error = 0.001;                    // Desired error before stopping (0.001 to 0.01 for good convergence)
	double early_stop_threshold = 100;              // Max epochs without improvement before stopping (50-100 epochs)
	double min_error_improvement = 1e-6;            // Minimum change in error to reset early stopping (very small, 1e-6)
	double min_learning_rate = 1e-6;                // Smallest allowable learning rate (1e-6 to prevent collapse)
	double stable_error_threshold = 1e-4;           // Error difference considered stable (1e-4 for small changes)
	int patience_limit = 50;                        // Number of epochs to wait before reducing learning rate (30-50 epochs)
	double dynamic_lr_reduction_factor = 0.1;       // Factor to reduce learning rate (reduce by 10% each time)
	bool show_error = true;                         // Flag to display error during training (usually true for tracking)
};

enum NORMALIZE_TYPE {
	NORMALIZE,
	DENORMALIZE
};

class MLP
{
private:
	int total_layers = 0;
	double learning_rate = 0.5;
	double permissible_error = 0.01;

	bool show_error = true;

	vector<double> input_min;							// For normalization
	vector<double> input_max;
	vector<double> output_max;
	vector<double> output_min;

	vector<vector<double>> output_from_each_layer;
	vector<int> neuron_in_each_layer;
	vector<Layer> layers;

public:
	MLP(int total_layers, vector<int>& neuron_in_each_layer, vector<vector<vector<double>>> weights = {}, vector<vector<double>> biases = {}) : neuron_in_each_layer(neuron_in_each_layer), total_layers(total_layers)
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
						//b_l.push_back(generateRandomNumber(0, 1));
						b_l.push_back(0);
					}
					for (int j = 0; j < neuron_in_each_layer[i]; j++) {
						vector<double> w(neuron_in_each_layer[i - 1]);
						for (int k = 0; k < neuron_in_each_layer[i - 1]; k++) {
							w[k] = 0.2;
							//w[k] = generateRandomNumber(-1, 1);
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
	vector<double> forward_propogation(vector<double>& input)
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
				//cout << layers[l].get_weights(i) << endl;
				double delta_bias = learning_rate * deltas[l][i];
				layers[l].set_bias(i, layers[l].get_bias(i) + delta_bias);
			}
		}
	}
	void toggle_show_error(bool val) {
		this->show_error = val;
	}
	void get_current_weight_and_bias(vector<vector<vector<double>>>& weights, vector<vector<double>>& bias) {
		weights.clear();
		bias.clear();
		for(int i = 0;i < total_layers;i++) {
			vector<vector<double>> cur_layer_weights;
			vector<double> cur_layer_bias;
			for (int j = 0; j < neuron_in_each_layer[i]; j++) {
				cur_layer_weights.push_back(layers[i].get_weights(j));
				cur_layer_bias.push_back(layers[i].get_bias(j));
			}
			weights.push_back(cur_layer_weights);
			bias.push_back(cur_layer_bias);
		}
	}
	void train(const TrainingParameters& params, vector<vector<double>>& presentations, vector<vector<double>>& desired_outputs)
	{
		//normalize_training_data(presentations, this->input_min, this->input_max);
		//normalize_training_data(desired_outputs, this->output_min, this->output_max);

		cout << presentations << endl << desired_outputs << endl;

		this->learning_rate = params.initial_learning_rate;
		this->permissible_error = params.target_error;
		this->show_error = params.show_error;

		double this_epoch_error = INT_MAX;
		double previous_epoch_error = INT_MAX;
		double minimum_error = INT_MAX;

		vector<vector<vector<double>>> best_weights;  // Store the best weights
		vector<vector<double>> best_bias;  // Store the best weights
	
		int stable_error_count = 0;
		int early_stopping_counter = 0;

		//double x = 0;

		get_current_weight_and_bias(best_weights, best_bias);
		
		while (this_epoch_error > params.target_error)
		{
			if (_kbhit()) {  // If a key is pressed
				char ch = _getch();  // Get the pressed key
				if (ch == 's' || ch == 'S') {
					cout << "Training stopped by user input." << endl;
					break;  // Exit the loop and stop training
				}
			}
			this_epoch_error = 0;

			for (int num = 0; num < params.presentations_per_eepoch; num++)
			{
				vector<double> output = forward_propogation(presentations[num]);
				//cout << output << " ";
				double sample_error = 0.0;
				for (int i = 0; i < output.size(); i++) {
					sample_error += abs(desired_outputs[num][i] - output[i]);
				}

				this_epoch_error += sample_error;
				backward_propagation(desired_outputs[num]);
			}

			this_epoch_error /= params.presentations_per_eepoch;

			if (params.show_error) {
				//printf(",(%f, %f)", x, this_epoch_error);
				//x += 0.01;
				cout <<  this_epoch_error << endl;
			}
			
			if (this_epoch_error < minimum_error) {
				minimum_error = this_epoch_error;
				get_current_weight_and_bias(best_weights, best_bias);  // Save weights when error is minimized
				//cout << "New minimum error: " << minimum_error << ", saving current weights." << endl;
			}


			if (fabs(previous_epoch_error - this_epoch_error) < params.min_error_improvement) {
				early_stopping_counter++;
			}
			else {
				early_stopping_counter = 0;
			}

			if (fabs(previous_epoch_error - this_epoch_error) < params.stable_error_threshold) {
				stable_error_count++;
				if (stable_error_count >= params.patience_limit) {
					this->learning_rate = max(this->learning_rate * 0.1, params.min_learning_rate);
					//cout << "Learning rate reduced to: " << this->learning_rate << endl;
					stable_error_count = 0;
				}
			}
			else {
				stable_error_count = 0;
			}

			previous_epoch_error = this_epoch_error;
		}

		if (early_stopping_counter >= params.early_stop_threshold) {
			cout << "Early stopping triggered after " << early_stopping_counter << " epochs of no significant improvement." << endl;
		}
		/*if (this_epoch_error > minimum_error) {
			cout << "Reverting to the best weights with minimum error: " << minimum_error << endl;
			for (int layer_no = 0; layer_no < total_layers; layer_no++) {
				for (int neuron_no = 0; neuron_no < neuron_in_each_layer[layer_no]; neuron_no++) {
					layers[layer_no].set_bias(neuron_no, best_bias[layer_no][neuron_no]);
					layers[layer_no].update_weights(neuron_no, best_weights[layer_no][neuron_no]);
				}
			}
		}*/
	}
	vector<double> compute(vector<double> input)
	{
		//normalize_prediction_data(input, NORMALIZE);
		//cout << input << endl;
		auto result = forward_propogation(input);
		//cout << result << endl;
		//normalize_prediction_data(result, DENORMALIZE);
		return result;
	}
	void normalize_prediction_data(vector<double>& data, NORMALIZE_TYPE type) {
		switch (type)
		{
		case NORMALIZE:
			for (int i = 0; i < data.size(); i++) {
				data[i] = (data[i] - input_min[i]) / (input_max[i] - input_min[i]);
			}
			break;
		case DENORMALIZE:
			for (int i = 0; i < data.size(); i++) {
				data[i] = (data[i] * (output_max[i] - output_min[i])) + output_min[i];
			}
			break;
		default:
			break;
		}
	}
	void normalize_training_data(vector<vector<double>>& data, vector<double>& min_val, vector<double>& max_val) {
		vector<double> min(data[0].size(), INT_MAX);
		vector<double> max(data[0].size(), INT_MIN);

		for (auto v : data) {
			for (int i = 0; i < v.size(); i++) {
				if (min[i] > v[i]) min[i] = v[i];
				if (max[i] < v[i]) max[i] = v[i];
			}
		}

		min_val = min;
		max_val = max;

		for (int j = 0; j < data.size(); j++) {
			for (int i = 0; i < data[j].size(); i++) {
				data[j][i] = ((data[j][i] - min[i]) / (max[i] - min[i]));
			}
		}
	}
	void print_mlp() {
		for (int i = 0; i < total_layers; i++) {
			layers[i].print_layer_info();
			cout << "-----------------------------------" << endl;
		}
	}
};