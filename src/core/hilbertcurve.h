#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_HILBERTCURVE_H
#define PBRT_CORE_HILBERTCURVE_H

// core/hilbertcurve.h*
#include "pbrt.h"
#include "geometry.h"
#include <stdint.h>

namespace pbrt {

class HilbertCurve {
  public:
    // Hilbert Curve Public Methods
    HilbertCurve(uint64_t order, uint64_t dim) :
        order(order), dim(dim), pow_2_order(std::pow(2, order)),
        pow_2_dim(std::pow(2, dim)),
        dmax((uint64_t)(std::pow(2, order * dim)) - 1) {}
    // N-D coordinate -> Hilbert Curve distance
    uint64_t pt2d(const std::vector<uint64_t> &pt);
    // distance -> N-D coordinate
    void d2pt(uint64_t d, std::vector<uint64_t> &pt);
    // Returns unit sample coordinates on the curve at u * dmax distance
    void sample(Float u, std::vector<Float> &pt);

    uint64_t order;         // Order of the Hilbert Curve
    uint64_t dim;           // Dimension of the Hilbert curve
    uint64_t pow_2_order;   // 2^{order}
    uint64_t pow_2_dim;     // 2^{dim}
    uint64_t dmax;          // Maximum distance value along the curve

  private:
    // Hilbert Curve Private Methods
    uint64_t calc_J(const uint64_t P);
    uint64_t calc_T(const uint64_t P);
    uint64_t calc_tS_tT(const uint64_t xJ, const uint64_t val);
    uint64_t gray_calc(const uint64_t B);
    uint64_t gray_inv_calc(const uint64_t G);
    // Returns byte oriented representation of point in pt
    void H_decode(const std::vector<uint64_t> &h,
                    std::vector<uint64_t> &pt);
    // Returns byte oriented representation of distance in h
    void H_encode(const std::vector<uint64_t> &pt,
                    std::vector<uint64_t> &h);
};

// 2D Hilbert Curve for performance
class HilbertCurve2D {
  public:
    // Hilbert Curve Public Methods
    HilbertCurve2D(uint64_t m) :
        m(m), n(std::pow(2, m)),
        dmax((uint64_t)(std::pow(2, m * 2)) - 1) {}
    // 2D coordinate -> Hilbert Curve distance
    uint64_t xy2d(uint64_t x, uint64_t y);
    // distance -> 2D coordinate
    void d2xy(uint64_t d, uint64_t *x, uint64_t *y);
    // rotate/flip a quadrant appropriately
    void rot(uint64_t t, uint64_t *x, uint64_t *y,
                uint64_t rx, uint64_t ry);
    // Returns unit sample coordinates on the curve at u * dmax distance
    Point2f sample(Float u);

    uint64_t m;     // Order of the Hilbert Curve
    uint64_t n;     // Square divided into n x n cells (n = 2^m)
    uint64_t dmax;  // Maximum distance value along the curve
};

}   // namespace pbrt

#endif  // PBRT_CORE_HILBERT_H

