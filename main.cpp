#include <Eigen/Dense>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";


int main(int argc, char* argv[]) {
    const char* input_image_path = IMAGE_NAME.c_str();

    // Load the image using stb_image
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Force load with 1 channel

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

    // Prepare Eigen matrices for each RGB channel
    MatrixXd img(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * 3;  // 3 channels (RGB)
            img(i, j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    grayscale_image = img.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });

    // Save the grayscale image using stb_image_write
    const std::string output_image_path = "output_grayscale.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                        grayscale_image.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;

        return 1;
    }

    std::cout << "Grayscale image saved to " << output_image_path << std::endl;

    return 0;
}