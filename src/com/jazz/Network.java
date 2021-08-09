package com.jazz;

import com.jazz.activations.ActivationFunction;
import com.jazz.activations.SigmoidActivation;
import com.jazz.costs.CostFunction;
import com.jazz.costs.QuadraticCost;
import com.jazz.data.MnistLoader;
import com.jazz.matrices.Matrix;
import com.jazz.matrices.MatrixMultiplier;
import com.jazz.utils.Pair;

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

    // Cost function.
    private CostFunction costFunction;

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
        this.costFunction = new QuadraticCost();
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

    public int prediction(Matrix inputs){
        Matrix rawOutput = feedForward(inputs);
        Matrix vector = mp.transpose(rawOutput);
        double[] arrayOutput = vector.getData()[0];
        int prediction = argmax(arrayOutput);

        return prediction;
    }

    /**
     * Converts training examples into formats immediately usable by the network for training.
     */
    private List<Pair<Matrix,Integer>> preprocess(List<Pair<Matrix,Integer>> trainingExamples){
        List<Pair<Matrix,Integer>> processed = new ArrayList<>();
        for (Pair<Matrix, Integer> trainingExample : trainingExamples) {
            Matrix image = trainingExample.getA();

            Matrix flattened = mp.flatten(image);
            Matrix transposed = mp.transpose(flattened);

            processed.add(new Pair<>(transposed,trainingExample.getB()));
        }

        return processed;
    }

    /**
     * Train using stochastic gradient descent.
     */
    public void SGD(double learningRate, double lambda, int miniBatchSize, int epochs){
        // Get the data.
        MnistLoader loader = new MnistLoader();

        List<Pair<Matrix,Integer>> trainingExamples = loader.getTrainingExamples();
        List<Pair<Matrix,Integer>> tests = loader.getTestExamples();

        System.out.println("Preprocessing...");
        trainingExamples = preprocess(trainingExamples);
        tests = preprocess(trainingExamples);

        // Partition the data into mini-batches.
        List<List<Pair<Matrix,Integer>>> miniBatches = new ArrayList<>();
        for (int i = 0; i < trainingExamples.size(); i+=miniBatchSize) {
            List<Pair<Matrix,Integer>> miniBatch = trainingExamples.subList(i,Math.min(i+miniBatchSize, trainingExamples.size()));
            miniBatches.add(miniBatch);
        }

        // Perform gradient descent on each mini-batch.
        for (int i = 0; i < epochs; i++) {
            System.out.println("EPOCH: " + i);
            for (List<Pair<Matrix, Integer>> miniBatch : miniBatches) {
                updateMiniBatch(miniBatch,learningRate,lambda,trainingExamples.size());
            }

            // Evaluation.
            double performance = evaluate(tests);
            System.out.println("Accuracy: " + performance + "%");
        }
    }

    private int argmax(double[] outputs){
        int maxArg = -1;
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < outputs.length; i++) {
            if(outputs[i] > max){
                max = outputs[i];
                maxArg = i;
            }
        }

        return maxArg;
    }

    private double evaluate(List<Pair<Matrix,Integer>> testData){
        int correct = 0;
        int total = testData.size();

        for (Pair<Matrix, Integer> testDatum : testData) {
            Matrix testInput = testDatum.getA();
            int expectedOutput = testDatum.getB();

            int prediction = prediction(testInput);

            if(prediction == expectedOutput){
                correct++;
            }
        }

        return (((double) correct) / total*100);
    }

    private void updateMiniBatch(List<Pair<Matrix,Integer>> miniBatch, double learningRate, double lambda, int trainingSize){
        // Gradients with respect to network parameters.
        List<Matrix> gradientBiases = new ArrayList<>();
        List<Matrix> gradientWeights = new ArrayList<>();

        // Initialize, since we are doing in-place addition here.
        for (Matrix correspondingWeightMatrix : weights) {
            int rows = correspondingWeightMatrix.getRows();
            int columns = correspondingWeightMatrix.getColumns();

            gradientWeights.add(new Matrix(rows, columns));
        }

        for (Matrix correspondingBiasMatrix : biases) {
            int rows = correspondingBiasMatrix.getRows();
            int columns = correspondingBiasMatrix.getColumns();

            gradientBiases.add(new Matrix(rows, columns));
        }

        for (Pair<Matrix, Integer> trainingExample : miniBatch) {
            Matrix image = trainingExample.getA();

            int label = trainingExample.getB();

            // We will convert this label to a "perfect" output we imagine the network should give.
            double[] expectedOutput = new double[dimensions[dimensions.length-1]];
            expectedOutput[label] = 1.0;

            Matrix expectedOutputMatrix = mp.fromVector(expectedOutput);

            // Use the equations of backpropagation to determine which changes need to be made to the weights and biases.
            Pair<List<Matrix>, List<Matrix>> deltaGradients = backpropagate(image, expectedOutputMatrix);

            // Sum the individual gradient components over many examples.
            List<Matrix> deltaGradientWeights = deltaGradients.getA();
            for (int i = 0; i < deltaGradientWeights.size(); i++) {
                Matrix correspondingGradientWeights = gradientWeights.get(i);
                Matrix correspondingDeltaGradientWeight = deltaGradientWeights.get(i);

                correspondingGradientWeights.addInPlace(correspondingDeltaGradientWeight);
            }

            List<Matrix> deltaGradientBiases = deltaGradients.getB();
            for (int i = 0; i < deltaGradientBiases.size(); i++) {
                gradientBiases.get(i).addInPlace(deltaGradientBiases.get(i));
            }
        }

        // Apply the gradients that have been collated to update the weights and biases.
        double regularizationConstant = 1-learningRate * (lambda / trainingSize);
        double learningRateConstant = -(learningRate / miniBatch.size());

        // Update weights.
        for (int i = 0; i < gradientWeights.size(); i++) {
            Matrix weightMat = weights.get(i);
            // Apply L2 regularization. This encourages the network to use smaller weights in the hopes of better generalization.
            weightMat.multiplyInPlace(regularizationConstant);

            Matrix weightGradientMat = gradientWeights.get(i);
            weightGradientMat.multiplyInPlace(learningRateConstant);

            weightMat.addInPlace(weightGradientMat);
        }

        // Update biases.
        for (int i = 0; i < gradientBiases.size(); i++) {
            Matrix biasMat = biases.get(i);

            Matrix biasGradientMat = gradientBiases.get(i);
            biasGradientMat.multiplyInPlace(learningRateConstant);

            biasMat.addInPlace(biasGradientMat);
        }
    }

    private Pair<List<Matrix>,List<Matrix>> backpropagate(Matrix image, Matrix expectedOutput){
        List<Matrix> gradientBiases = new ArrayList<>();
        List<Matrix> gradientWeights = new ArrayList<>();

        Matrix layerOutput = new Matrix(image.getData());

        List<Matrix> layerActivations = new ArrayList<>();

        // Sadly we have to create duplicates each time here to avoid modifying the original.
        layerActivations.add(new Matrix(layerOutput.getData()));

        // Layer outputs BEFORE the activation function is applied.
        List<Matrix> zVectors = new ArrayList<>();

        // Re-calculate layer outputs BEFORE applying the activation function.
        for (int i = 1; i < dimensions.length; i++) {
            int parameterIndex = i-1;
            Matrix weightMatrix = this.weights.get(parameterIndex);
            Matrix biasMatrix = this.biases.get(parameterIndex);

            Matrix weightedSum = mp.multiply(weightMatrix,layerOutput);
            Matrix zVector = mp.add(weightedSum,biasMatrix);

            zVectors.add(zVector);

            layerOutput = mp.activate(zVector,this.activationFunction);
            layerActivations.add(new Matrix(layerOutput.getData()));
        }

        // Backward pass.

        // Error in the output layer.
        Matrix finalZVector = zVectors.get(zVectors.size() - 1);
        Matrix finalActivation = layerActivations.get(layerActivations.size() - 1);
        Matrix error = this.costFunction.error(finalZVector, finalActivation, expectedOutput,this.activationFunction);

        // Now relate the error in the output layer to the weights and biases of that layer, using the equations of backpropagation.
        gradientBiases.add(new Matrix(error.getData()));

        Matrix weightGradient = mp.multiply(error,mp.transpose(layerActivations.get(layerActivations.size()-2)));
        gradientWeights.add(weightGradient);

        // Now that we have the error in the output layer, we have a means to work out the error in previous layers.
        int numLayers = dimensions.length;
        for (int layerIndex = 2; layerIndex < numLayers; layerIndex++) {
            Matrix zVector = zVectors.get(zVectors.size() - layerIndex);
            Matrix activationDerivative = mp.activateDerivative(zVector,this.activationFunction);

            // Backpropagate the error.
            Matrix correspondingWeights = this.weights.get((this.weights.size()-layerIndex)+1);
            Matrix correspondingWeightsTransposed = mp.transpose(correspondingWeights);

            Matrix weightErrorProduct = mp.multiply(correspondingWeightsTransposed,error);

            error = mp.hadamardProduct(weightErrorProduct,activationDerivative);
            gradientBiases.add(0,new Matrix(error.getData()));

            Matrix prevActivations = layerActivations.get((layerActivations.size()-layerIndex)-1);
            Matrix prevActivationsTransposed = mp.transpose(prevActivations);

            Matrix gradientWeight = mp.multiply(error,prevActivationsTransposed);

            gradientWeights.add(0,gradientWeight);
        }

        return new Pair<>(gradientWeights,gradientBiases);
    }

    public static void main(String[] args) {
        // Testing.
        int[] testNetworkSizes = new int[]{784,30,10};

        // Hyper-parameters.
        double learningRate = 0.3;
        double l2RegularizationRate = 0.005;
        int miniBatchSize = 10;
        int epochs = 30;


        Network net = new Network(testNetworkSizes);
        net.SGD(learningRate,l2RegularizationRate,miniBatchSize,epochs);
    }
}
