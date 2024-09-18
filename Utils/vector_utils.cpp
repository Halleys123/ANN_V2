#include <vector>
#include <iostream>

using namespace std;

std::ostream &operator<<(std::ostream &os, const vector<double> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const vector<vector<double>> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const vector<vector<vector<double>>> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}
vector<double> operator+(const vector<double> &v1, const vector<double> &v2)
{
    if (v1.size() != v2.size())
        throw invalid_argument("Vectors must be of same size");
    vector<double> result(v1.size());
    for (int i = 0; i < v1.size(); i++)
    {
        result[i] = v1[i] + v2[i];
    }
    return result;
}
vector<double> operator-(const vector<double> &v1, const vector<double> &v2)
{
    if (v1.size() != v2.size())
        throw invalid_argument("Vectors must be of same size");
    vector<double> result(v1.size());
    for (int i = 0; i < v1.size(); i++)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

double vector_sum(const vector<double> &input)
{
    double sum = 0.0;
    for (double i : input)
    {
        sum += input[i];
    }
    return sum;
}
// vector * scaler
vector<double> operator*(const vector<double> &v1, double scaler)
{
    vector<double> result(v1.size());
    for (int i = 0; i < v1.size(); i++)
    {
        result[i] = v1[i] * scaler;
    }
    return result;
}
// vector * vector
vector<double> operator*(const vector<double>& v1, const vector<double>& v2)
{
    if (v1.size() != v2.size())
        throw invalid_argument("Vectors must be of same size");
    vector<double> result(v1.size());
    for (int i = 0; i < v1.size(); i++)
    {
        result[i] = v1[i] * v2[i];
    }
    return result;

}