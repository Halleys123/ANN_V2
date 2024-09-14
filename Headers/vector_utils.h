#pragma once

#include <vector>
#include <iostream>

using namespace std;

// print purposes
std::ostream& operator<<(std::ostream& os, const vector<double>& v);
std::ostream& operator<<(std::ostream& os, const vector<vector<double>>& v);
std::ostream& operator<<(std::ostream& os, const vector<vector<vector<double>>>& v);

// operator overloads
vector<double> operator+(const vector<double>& v1, const vector<double>& v2);
vector<double> operator-(const vector<double>& v1, const vector<double>& v2);
vector<double> operator*(const vector<double>& v1, double scaler); // scaler * vector

// sum of all elements of a vector
double vector_sum(const vector<double>& input);
