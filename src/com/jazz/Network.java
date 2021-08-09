package com.jazz;

import com.jazz.activations.ActivationFunction;
import com.jazz.activations.SigmoidActivation;
import com.jazz.matrices.Matrix;
import com.jazz.matrices.MatrixMultiplier;

import java.util.ArrayList;
import java.util.List;

public class Network {
    // Math class.
    MatrixMultiplier mp = new MatrixMultiplier();

    // Network parameters.
    private List<Matrix> weights = new ArrayList<>();
    private List<Matrix> biases = new ArrayList<>();

    // Activation function.
    private ActivationFunction activationFunction;

    // The number of neurons in each layer.
    private int[] dimensions;

    public Network(int[] dimensions) {
        this.dimensions = dimensions;

        // e.g. input [784,30,10] means we have 784 input neurons, 30 hidden neurons, and 10 output neurons
        // so there will be two weight matrices with dimensions 30x784 and 10x30 respectively.
        for (int i = 0; i < dimensions.length-1; i++) {
            int columns = dimensions[i];
            int rows = dimensions[i+1];

            Matrix randomWeightMatrix = mp.randomMatrix(rows,columns);
            weights.add(randomWeightMatrix);
        }

        // There will also be two bias vectors with dimensions 30 and 10 respectively.
        for (int i = 1; i < dimensions.length; i++) {
            int rows = dimensions[i];

            Matrix randomBiases = mp.randomMatrix(rows,1);
            biases.add(randomBiases);
        }

        // For now, we will just use sigmoid.
        this.activationFunction = new SigmoidActivation();
    }

    /**
     * Computes the output for one pass of the network.
     * The output from one layer is the activation function applied to the weighted sum + bias
     */
    public Matrix feedForward(Matrix inputs){
        // Create a copy to avoid modifying the actual inputs.
        Matrix previousLayerOutput = new Matrix(inputs.getData());

        for (int i = 1; i < dimensions.length; i++) {
            int parameterIndex = i-1;
            Matrix weightMatrix = this.weights.get(parameterIndex);
            Matrix biasMatrix = this.biases.get(parameterIndex);

            Matrix weightedSum = mp.multiply(weightMatrix,previousLayerOutput);
            Matrix biasedSum = mp.add(weightedSum,biasMatrix);

            Matrix activations = mp.activate(biasedSum,this.activationFunction);
            previousLayerOutput = activations;
        }

        return previousLayerOutput;
    }

    /**
     * Train using stochastic gradient descent.
     */
    public void SGD(double learningRate, int miniBatchSize){
        // Partition the data into mini-batches and apply the rules of gradient descent.

    }

    public static void main(String[] args) {
        // Testing.
        int[] testNetworkSizes = new int[]{784,30,10};

        Network net = new Network(testNetworkSizes);

        MatrixMultiplier mp = new MatrixMultiplier();

        Matrix testInput = mp.randomMatrix(784,1);
        System.out.println("Feeding forward...");

        long start = System.currentTimeMillis();
        Matrix output = net.feedForward(testInput);
        long end = System.currentTimeMillis();
        long duration = end-start;

        System.out.println("Output: ");
        System.out.println(output);
        System.out.println("Feedforward execution time: " + duration + " ms");
    }
}
