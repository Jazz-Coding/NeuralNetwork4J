package com.jazz.matrices;

import java.util.Arrays;

public class Matrix {
    private double[][] data;
    private int rows;
    private int columns;

    // Zero-initialization.
    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;

        this.data = new double[rows][columns];
    }

    // Data-initialization.
    public Matrix(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.columns = data[0].length;
    }

    public double[][] getData() {
        return data;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public void incrementItem(int i, int j, double increment){
        this.data[i][j] += increment;
    }

    public void addInPlace(Matrix matB){
        if((getRows() != matB.getRows()) || (getColumns() != matB.getColumns())){
            throw new ArithmeticException("Matrices have incompatible dimensions!");
        }

        double[][] dataB = matB.getData();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.data[i][j] += dataB[i][j];
            }
        }
    }

    public void multiplyInPlace(double scalar){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                data[i][j] *= scalar;
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int index = 0;
        for (double[] datum : data) {
            sb.append(Arrays.toString(datum));
            if(index < data.length-1) {
                sb.append("\n");
            }
            index++;
        }
        sb.append("]");
        return sb.toString();
    }
}
