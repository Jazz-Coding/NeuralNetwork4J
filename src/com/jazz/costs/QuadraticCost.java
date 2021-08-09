package com.jazz.costs;

import com.jazz.activations.ActivationFunction;
import com.jazz.matrices.Matrix;
import com.jazz.matrices.MatrixMultiplier;

/**
 * Mean squared error.
 */
public class QuadraticCost implements CostFunction{
    private MatrixMultiplier mp = new MatrixMultiplier();

    @Override
    public double cost(Matrix output, Matrix expectedOutput) {
        Matrix diff = mp.subtract(output,expectedOutput);
        Matrix flat = mp.transpose(diff);

        double magnitude = 0;

        double[] diffVector = flat.getData()[0];
        for (double v : diffVector) {
            magnitude += Math.pow(v,2);
        }

        magnitude = Math.sqrt(magnitude);


        return 0.5 * Math.pow(magnitude,2);
    }

    @Override
    public Matrix error(Matrix zVector, Matrix output, Matrix expectedOutput, ActivationFunction activationFunction) {
        Matrix diff = mp.subtract(output,expectedOutput);
        Matrix costDerivative = mp.activateDerivative(zVector,activationFunction);

        Matrix product = mp.hadamardProduct(diff,costDerivative);

        return product;
    }
}
