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
