// #include <eigen3/Eigen/Sparse>
#include <Eigen/Sparse>

#include <iostream>
// #include <eigen3/unsupported/Eigen/SparseExtra>  
#include <unsupported/Eigen/SparseExtra>  


#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";
const double EPS = 0.00001;

// Some useful alias
using spMatrix = Eigen::SparseMatrix<double, RowMajor>;
using spVector = Eigen::VectorXd;


void save_img(MatrixXd m, int width, int height, std::string outputName) {
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img(height, width);
    img = m.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val)));
    });

    // Save the image using stb_image_write
    if (stbi_write_png(outputName.c_str(), width, height, 1,
                       img.data(), width) == 0) {
        std::cerr << "\nError: Could not save " + outputName + " image" << std::endl;

        return;
    }

    std::cout << "\nimage saved to " << outputName << std::endl;
}

// assume convolution matrix to be 3x3
std::tuple<SparseMatrix<double>, VectorXd>
applyConvolution(VectorXd img, MatrixXd convolutionMatrix, int width, int height) {
    SparseMatrix<double> filterMatrix(img.size(), img.size());
    std::vector<Triplet<double> > filter_data;
    for (int idx = 0; idx < img.size(); idx++) {
        int row = idx / width;
        int column = idx % width;

        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (column + i < 0 || column + i >= width || row + j < 0 || row + j >= height) continue;

                filter_data.emplace_back(idx, column + i + (j + row) * width, convolutionMatrix(i + 1, j + 1));
            }

        }

    }

    filterMatrix.setFromTriplets(filter_data.begin(), filter_data.end());

    return {filterMatrix, filterMatrix * img};
}


int main(int argc, char *argv[]) {
    srand(777); // repetible randoms
    MatrixXd H_av2{{1, 1, 1},
                   {1, 1, 1},
                   {1, 1, 1}};
    H_av2 *= 1.0 / 9;

    MatrixXd H_sh2{{0,  -3, 0},
                   {-1, 9,  -3},
                   {0,  -1, 0}};

    // Load the image using stb_image
    int width, height, channels;
    unsigned char *image_data = stbi_load(IMAGE_NAME.c_str(), &width, &height, &channels, 1); // Force 1 channel

    if (!image_data) {
        std::cerr << "Error: Could not load image " << IMAGE_NAME << std::endl;
        return 1;
    }

    // task 1
    // std::cout << "***The name of the input image is: " << IMAGE_NAME << std::endl;

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

    MatrixXd gray(height, width);

    // Fill the matrix with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            gray(i, j) = static_cast<double>(image_data[(i * width + j)]);
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    MatrixXd noisy = gray.unaryExpr([](double val) -> double {
        return std::min(255.0, std::max(0.0, static_cast<double>(val + rand() % 101 - 50)));
    });

    // task 2
    save_img(noisy, width, height, "noisy.png");

    auto gray_vector = gray.transpose().reshaped();    // v
    auto noisy_vector = noisy.transpose().reshaped();  // w

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
    printf("\nnumber of non-zero entries in matrix A1: %i\n", A1.nonZeros());

    // task 5
    save_img(smooth_noisy_vector, width, height, "smooth_noisy_vector.png");
    
    // task 6
    auto [A2, sharpened_original] = applyConvolution(gray_vector, H_sh2, width, height);
    printf("\nnumber of non-zero entries in Matrix A2: %i, is symetrical? %s\n", A2.nonZeros(),
           A2.isApprox(A2.transpose()) ? "true" : "false");

    // task 7
    save_img(sharpened_original, width, height, "sharpened_original.png");


    std::cout << "\n!!!Task8 start-----\n" << std::endl;
    //task 8 tbf
    if (saveMarket(A2, "A2.mtx")) {
        printf("\nA2.mtx succesfully saved. \n");
    } else {
        printf("Error: A2 couldn't be saved. \n");
    }

<<<<<<< HEAD
    
    if (saveMarketVector(noisy_vector, "noisy_image.mtx")) {
        printf("nosiy_image.mtx succesfully saved. \n");
    } else {
        printf("Error: nosiy_image couldn't be saved. \n");
=======
    
>>>>>>> 717c0769cca804ccdeda6ee76aa7c9ccfac71ec9
    }

    // Load matrix
    spMatrix matrixA2;
    Eigen::loadMarket(matrixA2, "A2.mtx");
    
    std::cout << "\nSize of matrix A2 is " << matrixA2.rows() << " X " << matrixA2.cols() << std::endl;
    std::cout << "Non-zero entries in A2 is: " << matrixA2.nonZeros() << std::endl;
    spMatrix checkMatrix = spMatrix(matrixA2.transpose()) - matrixA2;  // check the symmetry
    std::cout << "Norm of skew-symmetric part: " << checkMatrix.norm() << std::endl;  


    // Load vector w from the noisy_image.mtx
    spVector w;
    Eigen::loadMarketVector(w, "noisy_image.mtx");
    std::cout << "The size of vector w is " << w.size() << std::endl;

    //Set vector x(Same size as w, initially empty)
    spVector x(w.size());

    // Set parameters for solver
    double tol = 1.e-9;  // Tolerance for the solver
    int maxIteration = 1000;  // Max iterations

    // Set up Conjugate Gradient solver from Eigen-->CG solver for symmetric and postive-define matrix A
    // Eigen::ConjugateGradient<SpMatrix, Eigen::Lower|Eigen::Upper> solver;

    // Solver1: Set up the BiCGSTAB solver with preconditioner
    // BiCGSTAB --> A general-purpose solver for non-symmetric matrix
    // Eigen::BiCGSTAB<spMatrix, Eigen::DiagonalPreconditioner<double>> solver; //diagonal(Jacobi) preconditioner
    Eigen::BiCGSTAB<spMatrix, Eigen::IncompleteLUT<double>> solver;  // Incomplete LU Factorization(ILU) preconditioner
    
    // Set solver parameters
    solver.setMaxIterations(maxIteration);
    solver.setTolerance(tol);

    // Compute the decomposition of matrix A2 (prepares the matrix for solving)
    solver.compute(matrixA2);  //Factor the matrix A2
    x = solver.solve(w); // Solve the system A2 * x = w

    std::cout << "\nSolver results: " << std::endl;
    std::cout << "Iteration number is: " << solver.iterations() << std::endl;
    std::cout << "Final relative residual(error) is: " << solver.error() << std::endl;

    return 0;
}