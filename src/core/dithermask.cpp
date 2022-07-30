// core/dithermask.cpp
#include "dithermask.h"

namespace pbrt {

DitherMask::DitherMask(const std::string& filename) {
    // Read mask information
    std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
    if (!file) {
        // Check that we have opened the file
        Error("Impossible to read mask file %s.", filename.c_str());
        assert(false); // Crash
    }

    // Flag
    char type[5];
    file.read(type, 5*sizeof(char)); // MASK\n

    // Reading information
    unsigned int width(0), height(0), dim(0);
    file >> width >> height >> dim;

    // escape char
    char endOfLine;
    file.read(&endOfLine, sizeof(char));

    // Byte order (not used)
    char byteOrder[5]; // -1.0\n
    file.read(byteOrder, 5*sizeof(char));

    // Set internal:
    this->maskDimension = dim;
    this->maskSize.x = width;
    this->maskSize.y = height;
    this->nPixels = width * height;

    // Resize the mask
    this->values.resize(width * height * dim);
    file.read((char*) &this->values[0], (width * height * dim)*sizeof(float));
    float v = this->values[0];
    if (v > 1.0 || v < 0.0) {
        Error("Mask values are corrupted %f.", v);
        assert(false); // Crash
    }

    // Default offset into mask
    maskOffset = Point2i(0, 0);
}

void DitherMask::SetOffset(const Point2f& u) {
    maskOffset = Point2i(u.x * maskSize.x, u.y * maskSize.y);
}

Float DitherMask::Value(Point2i pixel, int dim) {
    // Get corresponding mask x, y from pixel
    int x((pixel.x + maskOffset.x) % maskSize.x);
    int y((pixel.y + maskOffset.y) % maskSize.y);

    return values[(dim * nPixels) + (y * maskSize.x) + (x)];
}

}   // namespace pbrt
