
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MEDIA_HOMOGENEOUSBLOCKING_H
#define PBRT_MEDIA_HOMOGENEOUSBLOCKING_H

// media/homogeneousblocking.h*
#include "medium.h"
#include "reservoir.h"
#include "light.h"
#include "hilbertcurve.h"
#include "lowdiscrepancy.h"

namespace pbrt {

// Homogeneous media that always samples the medium
// i.e. blocks the shading of the object behind the medium

// HomogeneousBlockingMedium Declarations
class HomogeneousBlockingMedium : public Medium {
  public:
    // HomogeneousBlockingMedium Public Methods
    HomogeneousBlockingMedium(const Spectrum &sigma_a,
                        const Spectrum &sigma_s, Float g, int M, int N,
                        const VolSampleStrategy sampleStrategy =
                            VolSampleStrategy::Transmittance)
        : sigma_a(sigma_a),
          sigma_s(sigma_s),
          sigma_t(sigma_s + sigma_a),
          g(g),
          M(M),
          N(N),
          sampleStrategy(sampleStrategy) {}
    Spectrum Tr(const Ray &ray, Sampler &sampler) const;
    Spectrum Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    MediumInteraction *mi) const;

    // Given a light, sample distance along the ray
    Spectrum SampleLightDriven(const Scene &scene, const Ray &ray,
                        Sampler &sampler, MemoryArena &arena,
                        const Shape *shape, const Light &light,
                        Point2f &uLight,
                        const std::shared_ptr<DitherMask> &ditherMask) const;

    // Sample 3D candidates (distance, direction)
    Spectrum SampleDistDir(const Scene &scene, const Ray &ray,
                    Sampler &sampler, MemoryArena &arena,
                    const Distribution1D *lightDistrib, const Shape *shape,
                    const std::shared_ptr<DitherMask> &ditherMask) const;

  private:
    // Sample distance with prob. proportional to transmittance (pbrt default)
    Spectrum SampleTransmittance(const Ray &ray, Sampler &sampler,
                    MemoryArena &arena,
                    MediumInteraction *mi) const;

    // RIS with reservoir sampling + stratified candidates + blue-noise
    Spectrum SampleRISReservoirLightDriven(const Scene &scene, const Ray &ray,
                    Sampler &sampler, MemoryArena &arena, const Shape *shape,
                    const Light &light, Point2f &uLight,
                    const std::shared_ptr<DitherMask> &ditherMask) const;

    // RIS with inverse cdf sampling + stratified candidates + blue-noise
    Spectrum SampleRISiCDFLightDriven(const Scene &scene, const Ray &ray,
                    Sampler &sampler, MemoryArena &arena, const Shape *shape,
                    const Light &light, Point2f &uLight,
                    const std::shared_ptr<DitherMask> &ditherMask) const;

    // RIS with bidir. cdf sampling + stratified candidates + blue-noise
    Spectrum SampleRISBidirectionalLightDriven(const Scene &scene, 
                    const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    const Shape *shape, const Light &light,
                    Point2f &uLight, Float offset, Float u,
                    const std::shared_ptr<DitherMask> &ditherMask,
                    int n = 0) const;

    // (3D candidates) RIS with reservoir sampling + QMC + blue-noise
    Spectrum SampleRISReservoirDistDir(const Scene &scene, const Ray &ray,
                    Sampler &sampler, MemoryArena &arena,
                    const Distribution1D *lightDistrib,
                    const Shape *shape,
                    const std::shared_ptr<DitherMask> &ditherMask) const;

    // (3D candidates) RIS with bidir. cdf sampling +
    //  Hilbert Curve candidates + blue-noise
    Spectrum SampleRISBidirectionalDistDir(const Scene &scene, const Ray &ray,
                    Sampler &sampler, MemoryArena &arena,
                    const Distribution1D *lightDistrib,
                    const Shape *shape,
                    Float offset, Float u,
                    const std::shared_ptr<DitherMask> &ditherMask,
                    int n = 0) const;

    // Sample distance and direction given 3D PSS candidate
    DistDirSample GetDistDirSample(const std::vector<Float> &sample3d,
                                  Float *w, const Scene &scene,
                                  const Ray &ray, MemoryArena &arena,
                                  const Distribution1D *lightDistrib,
                                  Float dMax, int nLights) const;

    // HomogeneousBlockingMedium Private Data
    const VolSampleStrategy sampleStrategy;
    const Spectrum sigma_a, sigma_s, sigma_t;
    const Float g;
    int M, N;
};

}  // namespace pbrt

#endif  // PBRT_MEDIA_HOMOGENEOUSBLOCKING_H
