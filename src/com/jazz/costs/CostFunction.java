package com.jazz.costs;

import com.jazz.activations.ActivationFunction;
import com.jazz.matrices.Matrix;

public interface CostFunction {
    /**
     * Computes the cost.
     */
    double cost(Matrix output, Matrix expectedOutput);

    /**
     * Computes the error in the final layer.
     * @param zVector - Neuron outputs prior to application of the activation function.
     * @param output - Neuron outputs after application of the activation function.
     * @param expectedOutput - Training labels.
     * @param activationFunction - The activation function used.
     */
    Matrix error(Matrix zVector, Matrix output, Matrix expectedOutput, ActivationFunction activationFunction);
}
