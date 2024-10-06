// #include <eigen3/Eigen/Sparse>
#include <Eigen/Sparse>
#include <iostream>
// #include <eigen3/unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";
const double EPS = 0.00001;

// Some useful alias
using spMatrix = Eigen::SparseMatrix<double, RowMajor>;  //This sparse matrix type is stored row by row
using spVector = Eigen::VectorXd;  // Dynamically sized vector of type double


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
    std::cout << "\nimage saved to " << outputName << std::endl;
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


// Use solver to solve linear system
void solveLinearSystem(const spMatrix &A, const spVector &w, spVector &result, const std::string &resultFileName,
                       double tolerance, int maxIterations, const std::string &imageFileName, int width, int height, bool isSymmetric = true)
{
    if (A.rows() != A.cols())
    {
        std::cerr << "Error: The matrix A is not square. Linear systems require square matrices." << std::endl;
        return;
    }

    if (w.size() != A.rows())
    {
        std::cerr << "Error: Size of vector w does not match the number of rows in matrix A." << std::endl;
        return;
    }

    if (w.size() != width * height)
    {
        std::cerr << "Error: Size of the result vector does not match the image dimensions." << std::endl;
        return;
    }

    if (isSymmetric)
    {
        // Use Conjugate Gradient for symmetric matrices
        Eigen::ConjugateGradient<spMatrix, Eigen::Lower | Eigen::Upper> solverCG;
        solverCG.setMaxIterations(maxIterations);
        solverCG.setTolerance(tolerance);
        solverCG.compute(A);
        result = solverCG.solve(w);

        if (solverCG.info() != Eigen::Success)
        {
            std::cerr << "Conjugate Gradient solver did not converge!" << std::endl;
        }
        else
        {
            std::cout << "\nSolver CG results: " << std::endl;
            std::cout << "Iteration number is: " << solverCG.iterations() << std::endl;
            std::cout << "Final relative residual (error) is: " << solverCG.error() << std::endl;
        }
    }
    else
    {
        // Use BiCGSTAB for non-symmetric matrices
        Eigen::BiCGSTAB<spMatrix, Eigen::IncompleteLUT<double>> solver;
        solver.setMaxIterations(maxIterations);
        solver.setTolerance(tolerance);
        solver.compute(A);
        result = solver.solve(w);

        if (solver.info() != Eigen::Success)
        {
            std::cerr << "BiCGSTAB solver did not converge!" << std::endl;
        }
        else
        {
            std::cout << "\nSolver BiCGSTAB results: " << std::endl;
            std::cout << "Iteration number is: " << solver.iterations() << std::endl;
            std::cout << "Final relative residual (error) is: " << solver.error() << std::endl;
        }
    }

    // Save result to file
    if (saveMarket(result, resultFileName))
    {
        std::cout << resultFileName << " successfully saved." << std::endl;
    }
    else
    {
        std::cerr << "Error: " << resultFileName << " couldn't be saved." << std::endl;
    }

    // Convert result to a matrix for image saving
    Eigen::MatrixXd solution = Eigen::Map<Eigen::MatrixXd>(result.data(), height, width).transpose();

    save_img(solution, width, height, imageFileName);
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

    // task 1

    std::cout << "Original image loaded: " << width << "x" << height << " with " << channels << " Channels." << std::endl;

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

    save_img(noisy, width, height, "noisy.png");

    auto gray_vector = gray.transpose().reshaped();   // v
    auto noisy_vector = noisy.transpose().reshaped(); // w

    std::cout << (gray_vector.dot(gray_vector)) << std::endl;
    double gray_vector_norm = std::sqrt(gray_vector.dot(gray_vector));
    double noisy_vector_norm = std::sqrt(noisy_vector.dot(noisy_vector));

    // task 3

    std::cout << "\nsize of original vector is: " << gray_vector.size() << "\n";
    std::cout << "size of noisy vector is: " << noisy_vector.size() << "\n";

    std::cout << "\nnorm of original vector is: " << gray_vector_norm << "\n";
    std::cout << "norm of noisy vector is: " << noisy_vector_norm << "\n";

    // task 4

    auto [A1, smooth_noisy_vector] = applyConvolution(noisy_vector, H_av2, width, height);
    printf("\nnumber of non-zero entries in matrix A1 is: %i\n", A1.nonZeros());

    // task 5

    save_img(smooth_noisy_vector, width, height, "smooth_noisy_vector.png");

    // task 6

    auto [A2, sharpened_original] = applyConvolution(gray_vector, H_sh2, width, height);
    printf("\nnumber of non-zero entries in Matrix A2 is: %i. Is symmetrical? %s\n", A2.nonZeros(),
           A2.isApprox(A2.transpose()) ? "true" : "false");

    // task 7

    save_img(sharpened_original, width, height, "sharpened_original.png");

    // task 8

    if (saveMarket(A2, "A2.mtx"))
    {
        printf("\nA2.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: A2 couldn't be saved. \n");
    }

    if (saveMarketVector(noisy_vector, "noisy_image.mtx"))
    {
        printf("nosiy_image.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: nosiy_image couldn't be saved. \n");
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

    // Set vector x(Same size as w, initially empty)
    spVector x(w.size());

    // Task 8, solve A2x = w

    solveLinearSystem(matrixA2, w, x, "x.mtx", 1.e-9, 1000, "solutionX_image.png", width, height, false);

    // Task 10
    auto [A3, laplation_edge] = applyConvolution(x, H_lap, width, height);
    printf("\nIs Matrix A3 symmetrical? %s\n",
           A3.isApprox(A3.transpose()) ? "true" : "false");

    // Task 11
    save_img(laplation_edge, width, height, "laplationEdge.png");

    // Task 12 && Task 13
    if (saveMarket(A3, "A3.mtx"))
    {
        printf("\nA3.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: A3 couldn't be saved. \n");
    }

    // Load matrix
    spMatrix matrixA3;
    Eigen::loadMarket(matrixA3, "A3.mtx");

    // Create an identity matrix of the same size as matrixA3
    spMatrix I(matrixA3.rows(), matrixA3.cols());
    I.setIdentity(); // Sets I as the identity matrix

    // Add the identity matrix to matrixA3
    spMatrix newMatrixA3 = I + matrixA3;
    std::cout << "\nSize of matrix the I+A3 is " << newMatrixA3.rows() << " X " << newMatrixA3.cols() << std::endl;
    std::cout << "Non-zero entries in the I+A3 is: " << newMatrixA3.nonZeros() << std::endl;

    // Check if newMatrixA3 is symmetric(It is!! So use CG)
    if (newMatrixA3.isApprox(newMatrixA3.transpose()))
    {
        std::cout << "newMatrixA3 is symmetricial." << std::endl;

        spVector y(w.size());

        solveLinearSystem(newMatrixA3, w, y, "y.mtx", 1.e-10, 1000, "solutionY_image.png", width, height, true);
    }
    else
    {
        std::cout << "newMatrixA3 is not symmetricial." << std::endl;
    }

    return 0;
}