// #include <eigen3/Eigen/Sparse>
#include <Eigen/Sparse>
#include <lis.h>
#include <iostream>
// #include <eigen3/unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION

#include "library/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "library/stb_image_write.h"

using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";
const double EPS = 0.00001;

// Some useful alias
using spMatrix = Eigen::SparseMatrix<double, RowMajor>;
using spVector = Eigen::VectorXd;

void save_img(MatrixXd m, int width, int height, std::string outputName)
{
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img(height, width);
    img = m.unaryExpr([](double val) -> unsigned char
                      { return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); });

    // Save the image using stb_image_write
    if (stbi_write_png(outputName.c_str(), width, height, 1, img.data(), width) == 0)
    {
        std::cerr << "\nError: Could not save " + outputName + " image" << std::endl;
        return;
    }

    std::cout << "\nImage saved to " << outputName << std::endl;
}

// assume convolution matrix to be 3x3
std::tuple<SparseMatrix<double>, VectorXd>
applyConvolution(VectorXd img, MatrixXd convolutionMatrix, int width, int height)
{
    SparseMatrix<double> filterMatrix(img.size(), img.size());
    std::vector<Triplet<double>> filter_data;
    for (int idx = 0; idx < img.size(); idx++)
    {
        int row = idx / width;
        int column = idx % width;

        for (int i = -1; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                if (column + i < 0 || column + i >= width || row + j < 0 || row + j >= height)
                    continue;

                filter_data.emplace_back(idx, column + i + (j + row) * width, convolutionMatrix(i + 1, j + 1));
            }
        }
    }
    filterMatrix.setFromTriplets(filter_data.begin(), filter_data.end());

    return {filterMatrix, filterMatrix * img};
}

int main(int argc, char *argv[])
{
    srand(777); // repetible randoms
    MatrixXd H_av2{{1, 1, 1},
                   {1, 1, 1},
                   {1, 1, 1}};
    H_av2 *= 1.0 / 9;

    MatrixXd H_sh2{{0, -3, 0},
                   {-1, 9, -3},
                   {0, -1, 0}};

    MatrixXd H_lap{{0, -1, 0},
                   {-1, 4, -1},
                   {0, -1, 0}};

    // Load the image using stb_image
    int width, height, channels;
    unsigned char *image_data = stbi_load(IMAGE_NAME.c_str(), &width, &height, &channels, 1); // Force 1 channel

    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << IMAGE_NAME << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << " x " << height << " with " << channels << " channels." << std::endl;

    MatrixXd gray(height, width);

    // Fill the matrix with image data
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            gray(i, j) = static_cast<double>(image_data[(i * width + j)]);
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    MatrixXd noisy = gray.unaryExpr([](double val) -> double
                                    { return std::min(255.0, std::max(0.0, static_cast<double>(val + rand() % 101 - 50))); });

    // task 2
    save_img(noisy, width, height, "output/noisy_base_image.png");

    auto gray_vector = gray.transpose().reshaped();   // v
    auto noisy_vector = noisy.transpose().reshaped(); // w

    std::cout << gray_vector.dot(gray_vector) << std::endl;
    double gray_vector_norm = std::sqrt(gray_vector.dot(gray_vector));
    double noisy_vector_norm = std::sqrt(noisy_vector.dot(noisy_vector));

    // task 3
    std::cout << "\nSize of original vector is: " << gray_vector.size() << "\n";
    std::cout << "Size of noisy vector is: " << noisy_vector.size() << "\n";

    std::cout << "\nNorm of original vector is: " << gray_vector_norm << "\n";
    std::cout << "Norm of noisy vector is: " << noisy_vector_norm << "\n";

    // task 4
    auto [A1, smooth_noisy_vector] = applyConvolution(noisy_vector, H_av2, width, height);
    std::cout << "\nNumber of non-zero entries in matrix A1: " << A1.nonZeros() << std::endl;

    // task 5
    save_img(smooth_noisy_vector, width, height, "output/smoothing_to_noisy_vector.png");

    // task 6
    auto [A2, sharpened_original] = applyConvolution(gray_vector, H_sh2, width, height);
    std::cout << "\nNumber of non-zero entries in Matrix A2: " << A2.nonZeros() << ", is symmetrical? "
              << (A2.isApprox(A2.transpose()) ? "true" : "false") << std::endl;

    // task 7
    save_img(sharpened_original, width, height, "output/sharpened_base_image.png");

    // task 8
    if (saveMarket(A2, "A2.mtx"))
    {
        std::cout << "\nA2.mtx successfully saved.\n";
    }
    else
    {
        std::cerr << "Error: A2 couldn't be saved.\n";
    }

    if (saveMarketVector(noisy_vector, "noisy_image.mtx"))
    {
        std::cout << "noisy_image.mtx successfully saved.\n";
    }
    else
    {
        std::cerr << "Error: noisy_image couldn't be saved.\n";
    }

    // Load matrix
    spMatrix matrixA2;
    Eigen::loadMarket(matrixA2, "A2.mtx");

    std::cout << "\nSize of matrix A2 is " << matrixA2.rows() << " X " << matrixA2.cols() << std::endl;
    std::cout << "Non-zero entries in A2 is: " << matrixA2.nonZeros() << std::endl;

    // Load vector w from the noisy_image.mtx
    spVector w;
    Eigen::loadMarketVector(w, "noisy_image.mtx");
    std::cout << "The size of vector w is " << w.size() << std::endl;

    // Set vector x (Same size as w, initially empty)
    spVector x(w.size());

    int maxIter = 1000;
    double tol = 1.0e-9;

    // task9
    MatrixXd x;
    if (loadMarket(x, "x.mtx"))
    {
        std::cout << "\nx.mtx successfully loaded.\n";
    }
    else
    {
        std::cerr << "Error: x couldn't be loaded.\n";
        return -1; // Exit if loading fails
    }

    MatrixXd solutionX = Eigen::Map<Eigen::MatrixXd>(x.data(), height, width).transpose();
    std::cout << "The size of this solutionX is: " << solutionX.rows() << " x " << solutionX.cols() << std::endl;
    save_img(solutionX, width, height, "output/matrix_X_solution.png");

    // Task 10
    auto [A3, laplation_edge] = applyConvolution(x, H_lap, width, height);
    std::cout << "\nIs Matrix A3 symmetrical? " << (A3.isApprox(A3.transpose()) ? "true" : "false") << std::endl;

    // Task 11
    save_img(laplation_edge, width, height, "output/edeg_detection_of_matrix_X.png");

    // Task 12
    if (saveMarket(A3, "A3.mtx"))
    {
        std::cout << "\nA3.mtx successfully saved.\n";
    }
    else
    {
        std::cerr << "Error: A3 couldn't be saved.\n";
    }

    // Load matrix
    spMatrix matrixA3;
    loadMarket(matrixA3, "A3.mtx");

    // Create an identity matrix of the same size as matrixA3
    spMatrix I(matrixA3.rows(), matrixA3.cols());
    I.setIdentity(); // Sets I as the identity matrix

    // Set vector y (Same size as w, initially empty)
    spVector y(w.size());

    // Add the identity matrix to matrixA3
    spMatrix newMatrixA3 = I + matrixA3;
    std::cout << "\nSize of matrix the I+A3 is " << newMatrixA3.rows() << " X " << newMatrixA3.cols() << std::endl;
    std::cout << "Non-zero entries in the I+A3 is: " << newMatrixA3.nonZeros() << std::endl;

    // Check if newMatrixA3 is symmetric
    bool isSymmetric = newMatrixA3.isApprox(newMatrixA3.transpose());
    std::cout << (isSymmetric ? "newMatrixA3 is symmetric." : "newMatrixA3 is not symmetric.") << std::endl;

    // Execute Conjugate Gradient solver only if the matrix is symmetric
    if (isSymmetric)
    {

        // Set parameters for solver
        double tol2 = 1.e-10;     // Tolerance for the solver
        int maxIteration2 = 1000; // Max iterations

        // Set up Conjugate Gradient solver from Eigen-->CG solver for symmetric and positive-definite matrix A
        Eigen::ConjugateGradient<spMatrix, Eigen::Lower | Eigen::Upper> solverCG;

        // Set solver parameters
        solverCG.setMaxIterations(maxIteration2);
        solverCG.setTolerance(tol2);

        // Compute the decomposition of newMatrixA3 (prepares the matrix for solving)
        solverCG.compute(newMatrixA3); // Factor the matrix newMatrixA3

        // Solve the system: newMatrixA3 * y = w
        y = solverCG.solve(w);

        std::cout << "\nSolver CG results: " << std::endl;
        std::cout << "Iteration number is: " << solverCG.iterations() << std::endl;
        std::cout << "Final relative residual (error) is: " << solverCG.error() << std::endl;
    }
    else
    {
        std::cerr << "Matrix is not symmetric, skipping CG solver." << std::endl;
    }

    // Task 13
    if (saveMarket(y, "y.mtx"))
    {
        std::cout << "\ny.mtx successfully saved.\n";
    }
    else
    {
        std::cerr << "Error: y couldn't be saved.\n";
    }

    Eigen::MatrixXd solutionY = Eigen::Map<Eigen::MatrixXd>(y.data(), height, width).transpose();
    std::cout << "The size of this solutionY is: " << solutionY.rows() << " x " << solutionY.cols() << std::endl;
    save_img(solutionY, width, height, "output/matrix_Y_image.png");

    return 0;
}
