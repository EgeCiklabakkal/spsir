// core/hilbertcurve.cpp
#include "hilbertcurve.h"

namespace pbrt {

uint64_t HilbertCurve::calc_J(const uint64_t P) {
    uint64_t i = 1;
    while (i < dim) {
        if (((P >> i) & 1) == (P & 1))
            i++;
        else
            break;
    }

    uint64_t result = dim;
    if (i != dim)
        result -= i;

    return result;
}

uint64_t HilbertCurve::calc_T(const uint64_t P) {
    uint64_t result;
    if (P <= 2)
        return 0;
    result = P - 2 + (P & 1);
    result = result ^ (result >> 1);

    return result;
}

uint64_t HilbertCurve::calc_tS_tT(const uint64_t xJ, const uint64_t val) {
    uint64_t temp = xJ % dim;
    uint64_t result = (val >> temp) | (val << (dim - temp));
    result = result & ((1 << dim) - 1);

    return result;
}

uint64_t HilbertCurve::gray_calc(const uint64_t B) {
    return B ^ (B >> 1);
}

uint64_t HilbertCurve::gray_inv_calc(const uint64_t G) {
    uint64_t a = G;
    uint64_t result = G;

    while(a != 0) {
        a = a >> 1;
        result = result ^ a;
    }

    return result;
}

void HilbertCurve::H_decode(const std::vector<uint64_t> &h,
                                std::vector<uint64_t> &pt) {
    uint64_t A, W, P, xJ;
    int i, k, n, m;

    n = dim;
    m = order;
    xJ = 0;
    W = 0;
    for (k = 0; k < n; k++)
        pt[k] = 0;

    for (i = 1; i <= m; i++) {
        P = h[i - 1];
        A = W ^ calc_tS_tT(xJ, gray_calc(P));

        for (k = 1; k <= n; k++) {
            pt[k - 1] = pt[k - 1] | ((A >> (n - k)) & 1) << (m - i);
        }

        W = W ^ calc_tS_tT(xJ, calc_T(P));
        xJ += calc_J(P) - 1;
    }
}

void HilbertCurve::H_encode(const std::vector<uint64_t> &pt,
                                std::vector<uint64_t> &h) {
    uint64_t A, W, S, P, xJ, xJx;
    int i, k, n, m;

    n = dim;
    m = order;
    xJ = 0;
    W = 0;
    for (k = 0; k < m; k++)
        h[k] = 0;

    for (i = 1; i <= m; i++) {
        A = 0;
        for (k = 1; k <= n; k++) {
            A = A | (((pt[k - 1] >> (m - i)) & 1) << (n - k));
        }

        xJx = dim - xJ % dim;
        S = calc_tS_tT(xJx, A ^ W);
        P = gray_inv_calc(S);
        h[i - 1] = P;
        W = W ^ calc_tS_tT(xJx, calc_T(P));
        xJ += calc_J(P) - 1;
    }
}

uint64_t HilbertCurve::pt2d(const std::vector<uint64_t> &pt) {
    uint64_t d(0);
    std::vector<uint64_t> h(order, 0);

    H_encode(pt, h);
    for (int k = order - 1; k >= 0; k--) {
        d += h[k] * pow(2, dim * ((order - 1) - k));
    }

    return d;
}

void HilbertCurve::d2pt(uint64_t d, std::vector<uint64_t> &pt) {
    std::vector<uint64_t> h(order, 0);

    // Byte oriented representation
    for (int k = order - 1; k >= 0; k--) {
        h[k] = d % pow_2_dim;
        d = d >> dim;
    }
    H_decode(h, pt);
}

void HilbertCurve::sample(Float u, std::vector<Float> &pt) {
    std::vector<uint64_t> pt_byte(dim, 0);
    uint64_t dist;

    dist = u * dmax;
    d2pt(dist, pt_byte);

    for (int i = 0; i < dim; i++) {
        pt[i] = pt_byte[i] / Float(pow_2_order);
    }
}

uint64_t HilbertCurve2D::xy2d(uint64_t x, uint64_t y) {
    uint64_t rx, ry, s, d = 0;
    for (s = n/2; s > 0; s /= 2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &x, &y, rx, ry);
    }

    return d;
}

void HilbertCurve2D::d2xy(uint64_t d, uint64_t *x, uint64_t *y) {
    uint64_t rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void HilbertCurve2D::rot(uint64_t t, uint64_t *x, uint64_t *y,
                        uint64_t rx, uint64_t ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = t-1 - *x;
            *y = t-1 - *y;
        }

        // Swap x and y
        uint64_t tmp = *x;
        *x = *y;
        *y = tmp;
    }
}

Point2f HilbertCurve2D::sample(Float u) {
    uint64_t x, y, dist;

    dist = u * dmax;
    d2xy(dist, &x, &y);
    return Point2f(x, y) / Float(n);
}

}   // namespace pbrt
