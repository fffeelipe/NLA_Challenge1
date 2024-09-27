#include <Eigen/Dense>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";


void save_img(MatrixXd m, int width, int height, std::string outputName){
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img(height, width);
            img = m.unaryExpr([](double val) -> unsigned char {
                return static_cast<unsigned char>(val);
            });

            // Save the image using stb_image_write
            if (stbi_write_png(outputName.c_str(), width, height, 1,
                               img.data(), width) == 0) {
                std::cerr << "Error: Could not save " + outputName + " image" << std::endl;

                return ;
            }

            std::cout << "image saved to " << outputName << std::endl;

}

int main(int argc, char* argv[]) {
            srand(777); //repetible randoms

            // Load the image using stb_image
            int width, height, channels;
                unsigned char* image_data = stbi_load(IMAGE_NAME.c_str(), &width, &height, &channels, 1);  // Force 1 channel

            if (!image_data) {
                std::cerr << "Error: Could not load image " << IMAGE_NAME << std::endl;
                return 1;
            }

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

            save_img(gray, width, height, "initial.png");
            
            MatrixXd noisy = gray.unaryExpr([](double val) -> double {
                return std::min(255.0, std::max(0.0, static_cast<double>(val + rand() % 101 - 50)));
            });

            save_img(noisy, width, height, "noisy.png");

            auto gray_vector = gray.reshaped();
            auto noisy_vector = noisy.reshaped();

std::cout<<(gray_vector.dot(gray_vector))<<std::endl;
        double gray_vector_norm = std::sqrt(gray_vector.dot(gray_vector));
        double noisy_vector_norm = std::sqrt(noisy_vector.dot(noisy_vector));

        std::cout<<"norm of original vector is: "<<gray_vector_norm<<"\n";
        std::cout<<"norm of noisy vector is: "<<noisy_vector_norm<<"\n";



        return 0;           
        }