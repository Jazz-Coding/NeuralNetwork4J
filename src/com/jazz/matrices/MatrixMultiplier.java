package com.jazz.matrices;

import com.jazz.activations.ActivationFunction;

import java.util.Random;

/**
 * Multiplies matrices, among other things.
 */
public class MatrixMultiplier {

    private Random random;

    public MatrixMultiplier() {
        this.random = new Random();
    }

    public MatrixMultiplier(long seed){
        this.random = new Random(seed);
    }

    /**
     * Iterative multiplication.
     */
    public Matrix multiply(Matrix matA, Matrix matB){
        if(matA.getColumns() != matB.getRows()){
            throw new ArithmeticException("Matrices have incompatible dimensions!");
        }

        double[][] dataA = matA.getData();
        double[][] dataB = matB.getData();

        int n = matA.getRows();
        int p = matB.getColumns();
        int m = matA.getColumns();

        double[][] dataC = new double[n][p];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += dataA[i][k] * dataB[k][j];
                }
                dataC[i][j] = sum;
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    /**
     * Elementwise addition and subtraction operations.
     */
    public Matrix add(Matrix matA, Matrix matB){
        if((matA.getRows() != matB.getRows()) || (matA.getColumns() != matB.getColumns())){
            throw new ArithmeticException("Matrices have incompatible dimensions!");
        }

        double[][] dataA = matA.getData();
        double[][] dataB = matB.getData();

        int n = matA.getRows();
        int m = matA.getColumns();

        double[][] dataC = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                dataC[i][j] = dataA[i][j] + dataB[i][j];
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    public Matrix subtract(Matrix matA, Matrix matB){
        if((matA.getRows() != matB.getRows()) || (matA.getColumns() != matB.getColumns())){
            throw new ArithmeticException("Matrices have incompatible dimensions!");
        }

        double[][] dataA = matA.getData();
        double[][] dataB = matB.getData();

        int n = matA.getRows();
        int m = matA.getColumns();

        double[][] dataC = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                dataC[i][j] = dataA[i][j] - dataB[i][j];
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    /**
     * Reverses the dimensions of a Matrix.
     */
    public Matrix transpose(Matrix mat){
        int rows = mat.getRows();
        int columns = mat.getColumns();

        double[][] data = mat.getData();
        double[][] dataC = new double[columns][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                dataC[j][i] = data[i][j];
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    /**
     * Random (Gaussian) matrix.
     */
    public Matrix randomMatrix(int rows, int columns){
        if(rows < 0 || columns < 0){
            throw new IllegalArgumentException("Can't generate a matrix with negative dimensions!");
        }

        double[][] dataC = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                dataC[i][j] = random.nextGaussian();
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    /**
     * Applies an activation function elementwise.
     */
    public Matrix activate(Matrix mat, ActivationFunction activationFunction){
        int rows = mat.getRows();
        int columns = mat.getColumns();

        double[][] data = mat.getData();
        double[][] dataC = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                dataC[i][j] = activationFunction.activate(data[i][j]);
            }
        }

        Matrix matC = new Matrix(dataC);
        return matC;
    }

    public static void main(String[] args) {
        // Testing.
        MatrixMultiplier mp = new MatrixMultiplier();

        Matrix inputs = mp.randomMatrix(10,1);
        Matrix weights = mp.randomMatrix(10,10);
        Matrix biases = mp.randomMatrix(10, 1);
        System.out.println("Inputs: ");
        System.out.println(inputs);

        System.out.println("Weights: ");
        System.out.println(weights);

        System.out.println("Biases: ");
        System.out.println(biases);

        System.out.println("Weights X Inputs + Biases");
        System.out.println(mp.add(mp.multiply(weights,inputs),biases));
    }
}
