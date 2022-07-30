#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_RESERVOIR_H
#define PBRT_CORE_RESERVOIR_H

// core/reservoir.h*
#include "pbrt.h"
#include "interaction.h"
#include "light.h"

namespace pbrt {

template <typename T>
class Reservoir {
  public:
    // Reservoir Public Methods
    Reservoir(int N); // Reservoir of size N
    Reservoir(std::vector<T> y, Float wsum, int M);
    Float update(T xi, Float wi, Float u, int m = 1);
    void update(T xi, Float wi, std::vector<Float>& u, int m = 1);

    // Reservoir Public Data
    std::vector<T> y;   // The output samples
    Float wsum;         // The sum of weights
    int M;              // The number of samples seen so far
    int N;              // The number of final samples
};

template <typename T>
Reservoir<T>::Reservoir(int N)
    : y(N), wsum(0.f), M(0), N(N) {}

template <typename T>
Reservoir<T>::Reservoir(std::vector<T> y, Float wsum, int M)
    : y(y), wsum(wsum), M(M), N(y.size()) {}

// Update by rescaling a single u
template <typename T>
Float Reservoir<T>::update(T xi, Float wi, Float u, int m) {
    wsum += wi;
    M += m;

    if (wi == 0) {
        return u;
    }

    Float p = wi / wsum;
    for (int i = 0; i < N; i++) {
        if (u < p) {
            y[i] = xi;
            u = u / p;
        } else {
            u = (u - p) / (Float(1) - p);
        }
    }

    return u;
}

// Update by rescaling individual u's for each sample
template <typename T>
void Reservoir<T>::update(T xi, Float wi, std::vector<Float>& u, int m) {
    wsum += wi;
    M += m;

    if (wi > 0) {
        Float p = wi / wsum;
        for (int i = 0; i < N; i++) {
            if (u[i] < p) {
                y[i] = xi;
                u[i] = u[i] / p;
            } else {
                u[i] = (u[i] - p) / (Float(1) - p);   
            }
        }
    }
}

// Primary sample space light sample
class PSSLightSample {
  public:
    PSSLightSample () {}
    PSSLightSample(const Point2f &uLight,
                    Float uLightNum,
                    std::shared_ptr<Light> light,
                    Float targetPdf)
    : uLight(uLight), uLightNum(uLightNum),
      light(light), targetPdf(targetPdf) {}

    Point2f uLight;     // 2D sample on the light
    Float uLightNum;    // Random to select the light
    Float targetPdf;    // Target function value
    std::shared_ptr<Light> light;
};

class DistDirSample {
    // Volume sample containing
    // distance along the ray
    // direction to light sample
    public:
        DistDirSample() {}
        DistDirSample(Float t, const Vector3f &wi,
                    Float targetPdf,
                    const VisibilityTester &visibility,
                    const Spectrum &f,
                    const Spectrum &Li,
                    const Spectrum &Tr,
                    Float G)
        : t(t), wi(wi), targetPdf(targetPdf),
          visibility(visibility), f(f), Li(Li),
          Tr(Tr), G(G) {}

    Float t;     // distance
    Vector3f wi; // direction
    Float targetPdf, G;
    VisibilityTester visibility;
    Spectrum f, Li, Tr;
};

}   // namespace pbrt

#endif  // PBRT_CORE_RESERVOIR_H
