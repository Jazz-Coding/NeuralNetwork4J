package com.jazz.data;

import com.jazz.matrices.Matrix;
import com.jazz.utils.Pair;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class MnistLoader {

    /**
     * Loads the training images and associated labels.
     * All credit for this section to RayDeeA : https://stackoverflow.com/a/20383900
     */
    public List<Pair<Matrix,Integer>> loadData(String type){
        String prefix = "";
        if(type.equalsIgnoreCase("train")){
            prefix = "train";
        } else if(type.equalsIgnoreCase("test")){
            prefix = "t10k";
        }
        String trainImagesPath = prefix + "-images.idx3-ubyte";
        String trainLabelsPath = prefix + "-labels.idx1-ubyte";

        List<Pair<Matrix,Integer>> trainingExamples = new ArrayList<>();

        try {
            FileInputStream isImages = new FileInputStream(trainImagesPath);
            FileInputStream isLabels = new FileInputStream(trainLabelsPath);

            int magicNumberImages = (isImages.read() << 24) | (isImages.read() << 16) | (isImages.read() << 8) | (isImages.read());
            int numberOfImages = (isImages.read() << 24) | (isImages.read() << 16) | (isImages.read() << 8) | (isImages.read());
            int numberOfRows  = (isImages.read() << 24) | (isImages.read() << 16) | (isImages.read() << 8) | (isImages.read());
            int numberOfColumns = (isImages.read() << 24) | (isImages.read() << 16) | (isImages.read() << 8) | (isImages.read());

            int magicNumberLabels = (isLabels.read() << 24) | (isLabels.read() << 16) | (isLabels.read() << 8) | (isLabels.read());
            int numberOfLabels = (isLabels.read() << 24) | (isLabels.read() << 16) | (isLabels.read() << 8) | (isLabels.read());

            // Image dimensions.
            int nPixels = numberOfRows * numberOfColumns;
            for (int i = 0; i < numberOfImages; i++) {
                if(i % 100 == 0) {System.out.println("Number of images extracted: " + i);}

                double[][] pixels = new double[numberOfRows][numberOfColumns];
                for (int j = 0; j < nPixels; j++) {
                    int gray = isImages.read();
                    double scaled = gray / 255D;

                    int rowIndex = j / numberOfColumns;
                    int colIndex = j % numberOfColumns;

                    pixels[rowIndex][colIndex] = scaled;
                }

                Matrix pixelMatrix = new Matrix(pixels);
                int label = isLabels.read();
                Pair<Matrix,Integer> trainingExample = new Pair<>(pixelMatrix,label);
                trainingExamples.add(trainingExample);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return trainingExamples;
    }

    public List<Pair<Matrix,Integer>> getTrainingExamples(){
        return loadData("train");
    }

    public List<Pair<Matrix,Integer>> getTestExamples(){
        return loadData("test");
    }
}
