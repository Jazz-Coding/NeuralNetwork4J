package com.jazz.activations;

public class SigmoidActivation implements ActivationFunction{
    @Override
    public double activate(double z) {
        return 1D / (1D + Math.exp(-z));
    }

    @Override
    public double activate_derivative(double z) {
        return activate(z) * (1D - activate(z));
    }
}
