package com.jazz.activations;

public interface ActivationFunction {
    double activate(double z);
    double activate_derivative(double z);
}
