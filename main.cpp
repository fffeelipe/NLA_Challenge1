// #include <eigen3/Eigen/Sparse>
#include <Eigen/Sparse>

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
    // std::cout << "***The name of the input image is: " << IMAGE_NAME << std::endl;

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
    save_img(noisy, width, height, "output/noisy.png");

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
    printf("\nnumber of non-zero entries in matrix A1: %i\n", A1.nonZeros());

    // task 5
    save_img(smooth_noisy_vector, width, height, "output/smooth_noisy_vector.png");

    // task 6
    auto [A2, sharpened_original] = applyConvolution(gray_vector, H_sh2, width, height);
    printf("\nnumber of non-zero entries in Matrix A2: %i, is symmetrical? %s\n", A2.nonZeros(),
           A2.isApprox(A2.transpose()) ? "true" : "false");

    // task 7
    save_img(sharpened_original, width, height, "output/sharpened_original.png");

    // task 8 tbf
    // First preconditioner --> jacobi
    // mery@mery-IdeaPad-3-15ADA05:~/shared-docker/lis-2.0.34/test$ ./test1 A2.mtx w.mtx nla_x.mtx histnla.txt -i bicg -p jacobi -tol 1e-9

    // Another preconditioner is more efficient --> ilu
    // root@36d8968386ff test # mpirun -n 1 ./test1 /shared-folder/NLA_Challenge1/output/A2.mtx /shared-folder/NLA_Challenge1/output/w.mtx /shared-folder/NLA_Challenge1/output/x.mtx hist.txt -i bicg -p ilu -tol 1e-9 

    //  auto gray_vector = gray.transpose().reshaped();    // v
    // auto noisy_vector = noisy.transpose().reshaped();  // w
    if (saveMarket(A2, "output/A2.mtx"))
    {
        printf("\nA2.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: A2 couldn't be saved. \n");
    }

    // Export vector in .mtx format
    int n = noisy_vector.size();
    // Eigen::saveMarketVector(b, "./rhs.mtx");
    FILE *out = fopen("output/w.mtx", "w");
    fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out, "%d\n", n);
    for (int i = 0; i < n; i++)
    {
        fprintf(out, "%d %f\n", i, noisy_vector(i));
    }
    fclose(out);

    /* jacobi preconditioner
    number of processes = 1
    matrix size = 87296 x 87296 (782086 nonzero entries)

    initial vector x      : all components set to 0
    precision             : double
    linear solver         : BiCG
    preconditioner        : Jacobi
    convergence condition : ||b-Ax||_2 <= 1.0e-09 * ||b-Ax_0||_2
    matrix storage format : CSR
    linear solver status  : normal end

    BiCG: number of iterations = 100
    BiCG:   double             = 100
    BiCG:   quad               = 0
    BiCG: elapsed time         = 4.169440e-01 sec.
    BiCG:   preconditioner     = 2.255654e-02 sec.
    BiCG:     matrix creation  = 2.145767e-06 sec.
    BiCG:   linear solver      = 3.943875e-01 sec.
    BiCG: relative residual    = 8.774665e-10*/


    /* ilu preconditioner
    number of processes = 1
    matrix size = 87296 x 87296 (782086 nonzero entries)

    initial vector x      : all components set to 0
    precision             : double
    linear solver         : BiCG
    preconditioner        : ILU(0)
    convergence condition : ||b-Ax||_2 <= 1.0e-09 * ||b-Ax_0||_2
    matrix storage format : CSR
    linear solver status  : normal end

    BiCG: number of iterations = 16
    BiCG:   double             = 16
    BiCG:   quad               = 0
    BiCG: elapsed time         = 5.555391e-02 sec.
    BiCG:   preconditioner     = 3.671193e-02 sec.
    BiCG:     matrix creation  = 9.536743e-07 sec.
    BiCG:   linear solver      = 1.884198e-02 sec.
    BiCG: relative residual    = 4.028669e-10
    */

    // task9
    spVector x(noisy_vector.size());
    int temp;
    char line[100];
    FILE *file = fopen("output/x.mtx", "r");
    fgets(line, 100, file);
    fgets(line, 100, file);

    int size;
    sscanf(line, "%d\n", &size);

    double v;
    for (int i = 0; i < size; i++)
    {
        fgets(line, 100, file);
        sscanf(line, "%d %lf\n", &temp, &v);
        x(i) = v;
    }

    fclose(file);

    std::cout << "The size of vector x is " << x.size() << std::endl;

    Eigen::MatrixXd solutionX = Eigen::Map<Eigen::MatrixXd>(x.data(), height, width).transpose();
    std::cout << "The size of this solutionX is: " << solutionX.rows() << " x " << solutionX.cols() << std::endl;
    save_img(solutionX, width, height, "output/solutionX_image.png");

    // Task 10
    auto [A3, laplation_edge] = applyConvolution(gray_vector, H_lap, width, height);
    printf("\nIs Matrix A3 symmetrical? %s\n",
           A3.isApprox(A3.transpose()) ? "true" : "false");

    // Task 11
    save_img(laplation_edge, width, height, "output/laplationEdge.png");

    // Task 12
    if (saveMarket(A3, "output/A3.mtx"))
    {
        printf("\nA3.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: A3 couldn't be saved. \n");
    }

    // Load matrix
    spMatrix matrixA3;
    Eigen::loadMarket(matrixA3, "output/A3.mtx");

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
        std::cout << "newMatrixA3 is symmetric." << std::endl;
    }
    else
    {
        std::cout << "newMatrixA3 is not symmetric." << std::endl;
    }

    // Set vector y(Same size as w, initially empty)
    spVector y(noisy_vector.size());

    // Set parameters for solver
    double tol2 = 1.e-10; // Tolerance for the solver

    // Set up Conjugate Gradient solver from Eigen-->CG solver for symmetric and postive-define matrix A
    Eigen::ConjugateGradient<spMatrix, Eigen::Lower | Eigen::Upper> solverCG;

    // Set solver parameters
    solverCG.setTolerance(tol2);

    // Compute the decomposition of matrix A3 (prepares the matrix for solving)
    solverCG.compute(newMatrixA3);    // Factor the matrix A2
    y = solverCG.solve(noisy_vector); // Solve the system () * x = w

    std::cout << "\nSolver CG results: " << std::endl;
    std::cout << "Iteration number is: " << solverCG.iterations() << std::endl;
    std::cout << "Final relative residual(error) is: " << solverCG.error() << std::endl;

    // Task 13
    if (saveMarket(y, "output/y.mtx"))
    {
        printf("\ny.mtx succesfully saved. \n");
    }
    else
    {
        printf("Error: y couldn't be saved. \n");
    }

    Eigen::MatrixXd solutionY = Eigen::Map<Eigen::MatrixXd>(y.data(), height, width).transpose();
    std::cout << "The size of this solutionY is: " << solutionY.rows() << " x " << solutionY.cols() << std::endl;
    save_img(solutionY, width, height, "output/solutionY_image.png");

    return 0;
}