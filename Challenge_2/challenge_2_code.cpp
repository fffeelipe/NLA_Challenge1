#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "Library/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Library/stb_image_write.h"

using namespace Eigen;
using namespace std;

// Some useful alias
using spMatrix = Eigen::SparseMatrix<double, RowMajor>;
using spVector = Eigen::VectorXd;

const std::string IMAGE_PATH = "Input/Albert_Einstein_Head.jpg";

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
    std::list<Triplet<double>> filter_data;
    for (int idx = 0; idx < img.size(); idx++)
    {
        int row = idx / width;
        int column = idx % width;

        for (int i = -1; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                if (column + i < 0 || column + i >= width || row + j < 0 || row + j >= height || convolutionMatrix(i + 1, j + 1) == 0.)
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
    // TASK 1: Load the image as Eigen matrix A and compute A^T * A
    // Load the image into an Eigen matrix. Each pixel value is represented as an entry in matrix A.
    // Then compute the matrix product A^T * A to analyze the correlation between image columns.
    // Finally, compute and print the Euclidean norm of A^T * A.

int n, m, channels;
    unsigned char *image_data = stbi_load(IMAGE_PATH.c_str(), &n, &m, &channels, 1); // Force 1 channel
    //m = height, n = width

    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << IMAGE_PATH << std::endl;
        return 1;
    }
    MatrixXd A(m, n);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = static_cast<double>(image_data[(i * n + j)]);
        }
    }
    stbi_image_free(image_data);

    auto At_A  = A.transpose() * A;

    auto norm_At_A = At_A.norm();

    printf("the norm of A^T * A is %d\n", norm_At_A);


    // TASK 2: Solve the eigenvalue problem A^T A x = λ x
    // Use Eigen's solver to compute the eigenvalues of A^T A. Report the two largest singular values of matrix A.

    SelfAdjointEigenSolver<MatrixXd> saeigensolver(At_A);
  if (saeigensolver.info() != Eigen::Success) {
    printf("couldn't calculate the eigenpairs, sorry\n");
    abort();
}
    printf("The two largest eigenvalues are: %d and %d\n", saeigensolver.eigenvalues()(n-1), saeigensolver.eigenvalues()(n-2));
  

    // TASK 3: Export matrix A^T A in Matrix Market format
    // Save A^T A in the Matrix Market format (.mtx) to be used with external solvers such as LIS.

    if (saveMarket(At_A, "output/At_A.mtx"))
    {
        printf("At_A.mtx succesfully saved.\n");
    }
    else
    {
        printf("Error: At_A couldn't be saved.\n");
        return 1;
    }

    // TASK 4: Find a shift μ to accelerate the eigenvalue solver
    // Identify an optimal shift μ that can improve the convergence of the iterative eigensolver.
    // Report μ and the number of iterations required for convergence.

    // TASK 5: Perform SVD on matrix A
    // Perform Singular Value Decomposition (SVD) on matrix A using Eigen's SVD module.
    // Report the Euclidean norm of the singular values (Σ) matrix.

    // TASK 6: Compute matrices C and D for k = 40 and k = 80
    // Compute C and D matrices using k = 40 and k = 80 based on the SVD results.
    // Report the number of non-zero entries in C and D.

    // TASK 7: Compress image using C * D^T for k = 40 and k = 80
    // Reconstruct the compressed images from the matrices C and D.
    // Export the images as .png files and save them.

    // TASK 8: Create a black-and-white checkerboard image
    // Generate a checkerboard pattern image of 200x200 pixels. Each pixel alternates between black and white.
    // Report the Euclidean norm of the resulting matrix representing the image.
    MatrixXd BW_Matrix(200, 200);
    // Fill the matrix with color data
    for (int i = 0; i < 200; ++i)
    {
        for (int j = 0; j < 200; ++j)
        {
            if (((i / 20) + (j / 20)) % 2 == 0) 
            {
                BW_Matrix(i, j) = 255;  // White
            } else {
                BW_Matrix(i, j) = 0;    // Black
            }
        }
        
        
    }

    save_img(BW_Matrix, 200, 200, "Assets/BW_Matrix.png");
    auto BW_Matrix_Vector = BW_Matrix.transpose().reshaped();

    double BW_Matrix_norm = std::sqrt(BW_Matrix_Vector.dot(BW_Matrix_Vector));
    printf("\nnorm of BW_Matrix is: %f\n", BW_Matrix_norm );


    // TASK 9: Introduce noise into the checkerboard image
    // Add random noise to the checkerboard image by altering pixel values within the range [-50, 50].
    // Save and export the noisy image as a .png file.

    MatrixXd BW_noisy = BW_Matrix; // Start with the original checkerboard

    // Random number generator for noise in the range [-50, 50]
    srand(777); // Seed rand() for reproducible random numbers

    // Add noise to each pixel of BW_noisy
    BW_noisy = BW_noisy.unaryExpr([&](double val) -> double {
        int noise = rand() % 101 - 50;  // Generate noise in the range [-50, 50]
        double noisy_val = val + noise;  // Add noise to the original pixel value
        return std::clamp(noisy_val, 0.0, 255.0);  // Clamp result to range [0, 255]
    });

    save_img(BW_noisy, 200, 200, "Assets/BW_Noisy.png");


    // TASK 10: Perform SVD on the noisy image
    // Use SVD to analyze the noisy image and report the two largest singular values.

    // TASK 11: Compute C and D for k = 5 and k = 10
    // Based on the SVD of the noisy image, compute C and D matrices for k = 5 and k = 10.
    // Report the size of the C and D matrices.

    // TASK 12: Compress the noisy image using C * D^T for k = 5 and k = 10
    // Reconstruct the compressed versions of the noisy image using k = 5 and k = 10.
    // Save the compressed images as .png files.

    // TASK 13: Compare compressed images with the original and noisy images
    // Compare the quality of the compressed images with both the original and noisy versions.
    // Report the results and observations.

    return 0;
}