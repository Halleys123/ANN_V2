// actiavtion functions
#include <cmath>

double unipolar_sigmoid(double x, double alpha = 1)
{
    return 1 / (1 + exp(-alpha * x));
}

double bipolar_sigmoid(double x, double alpha = 1)
{
    return (2 / (1 + exp(-alpha * x))) - 1;
}

double unipolar_sigmoid_derivative(double x, double alpha = 1)
{
    return alpha * unipolar_sigmoid(x, alpha) * (1 - unipolar_sigmoid(x, alpha));
}

double bipolar_sigmoid_derivative(double x, double alpha = 1)
{
    return (alpha / 2) * (1 + bipolar_sigmoid(x, alpha)) * (1 - bipolar_sigmoid(x, alpha));
}

double linear(double x)
{
    return x;
}

double linear_derivative(double x)
{
    return 1;
}