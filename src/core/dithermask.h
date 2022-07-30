#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_DITHERMASK_H
#define PBRT_CORE_DITHERMASK_H

// core/dithermask.h
#include <fstream>
#include "pbrt.h"
#include "geometry.h"

namespace pbrt {

class DitherMask {
  public:
    // DitherMask Public Methods
    DitherMask(const std::string& filename);
    void SetOffset(const Point2f& u);
    Float Value(Point2i pixel, int dim = 0);

    // Mask information
    Point2i maskSize;
    Point2i maskOffset;
    int nPixels;
    int maskDimension;
    std::vector<float> values;
};

}   // namespace pbrt

#endif  // PBRT_CORE_DITHERMASK_H
