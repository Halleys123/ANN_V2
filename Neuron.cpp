#pragma once

#include "ACT_FUNC_ENUM.cpp"
#include "ActivationFunctions.h"
#include "vector_utils.h"

#include <vector>
#include <stdexcept>

using namespace std;

class Neuron
{
private:
    double net_i = 0.0;
    double f_net_i = 0.0;

    bool is_input_layer_neuron = false;
    double bias = NULL;
    vector<double> weights;
    
    bool init = true;

public:
    Neuron() {
        init = false;
    }
    Neuron(vector<double> weights, double bias, bool is_input_layer_neuron = false) : weights(weights), bias(bias), is_input_layer_neuron(is_input_layer_neuron){}
    
    void set_weights(vector<double> weights)
    {
        this->weights = weights;
    }
    void update_weight(int prev_neuron, double weight) {
        this->weights[prev_neuron] = weight;
    }
    void set_bias(double bias) {
        this->bias = bias;
    }

    double get_bias() {
        return this->bias;
    }
    const vector<double>& get_weights() const {
        return this->weights;
    }

    double compute(vector<double> inputs, ACTIVATION_FUNCTIONS func = UNIPOLAR_SIGMOID, int lambda = 1)
    {
        if (weights.size() != inputs.size())
            throw invalid_argument("The number of inputs must be equal to the number of weights");

        net_i = 0;
        f_net_i = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            net_i += inputs[i] * weights[i];
        }
        net_i += bias;
        if (is_input_layer_neuron) return net_i;
        if (func == ACTIVATION_FUNCTIONS::UNIPOLAR_SIGMOID)
        {
            f_net_i = unipolar_sigmoid(net_i, lambda);
        }
        else if (func == ACTIVATION_FUNCTIONS::BIPOLAR_SIGMOID)
        {
            f_net_i = bipolar_sigmoid(net_i, lambda);
        }
        else
        {
            throw invalid_argument("Invalid activation function");
        }
        return f_net_i;
    }
};