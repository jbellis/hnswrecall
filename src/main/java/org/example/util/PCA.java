package org.example.util;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class PCA {
    public static float[][] computePCARotation(List<float[]> centeredVectors) {
        // Convert the list of float[] vectors to a RealMatrix
        double[][] data = new double[centeredVectors.size()][];
        for (int i = 0; i < centeredVectors.size(); i++) {
            float[] currentVector = centeredVectors.get(i);
            data[i] = new double[currentVector.length];
            for (int j = 0; j < currentVector.length; j++) {
                data[i][j] = currentVector[j];
            }
        }
        RealMatrix matrix = MatrixUtils.createRealMatrix(data);

        // Compute the covariance matrix and eigendecomposition
        RealMatrix covarianceMatrix = new Covariance(matrix).getCovarianceMatrix();
        EigenDecomposition eigenDecomposition = new EigenDecomposition(covarianceMatrix);

        // Get eigenvalues and sort eigenvectors based on them
        double[] eigenvalues = eigenDecomposition.getRealEigenvalues();
        Integer[] indices = IntStream.range(0, eigenvalues.length).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, (i1, i2) -> Double.compare(eigenvalues[i2], eigenvalues[i1]));

        double[][] sortedRotation = new double[covarianceMatrix.getRowDimension()][covarianceMatrix.getColumnDimension()];
        for (int i = 0; i < covarianceMatrix.getColumnDimension(); i++) {
            double[] eigenvector = eigenDecomposition.getEigenvector(indices[i]).toArray();
            for (int j = 0; j < eigenvector.length; j++) {
                sortedRotation[j][i] = eigenvector[j];
            }
        }

        // Convert sortedRotation from double[][] to float[][]
        float[][] rotationFloat = new float[sortedRotation.length][sortedRotation[0].length];
        for (int i = 0; i < sortedRotation.length; i++) {
            for (int j = 0; j < sortedRotation[i].length; j++) {
                rotationFloat[i][j] = (float) sortedRotation[i][j];
            }
        }

        return rotationFloat;
    }

    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
}

