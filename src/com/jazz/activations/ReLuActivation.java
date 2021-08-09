package com.jazz.activations;

public class ReLuActivation implements ActivationFunction{
    @Override
    public double activate(double z) {
        if(z > 0){
            return z;
        } else{
            return 0;
        }
    }

    @Override
    public double activate_derivative(double z) {
        if(z > 0){
            return 1;
        } else {
            return 0;
        }
    }
}
