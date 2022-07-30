
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

#ifndef PBRT_CORE_INTEGRATOR_H
#define PBRT_CORE_INTEGRATOR_H

// core/integrator.h*
#include "pbrt.h"
#include "primitive.h"
#include "spectrum.h"
#include "light.h"
#include "reflection.h"
#include "sampler.h"
#include "material.h"
#include "reservoir.h"
#include "dithermask.h"
#include "lowdiscrepancy.h"

namespace pbrt {

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();
    virtual void Render(const Scene &scene) = 0;
};

Spectrum UniformSampleAllLights(const Interaction &it, const Scene &scene,
                                MemoryArena &arena, Sampler &sampler,
                                const std::vector<int> &nLightSamples,
                                bool handleMedia = false);
Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia = false,
                               const Distribution1D *lightDistrib = nullptr);

// RIS with reservoir sampling + QMC + blue-noise
Spectrum ReservoirLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            int N = 1, bool handleMedia = false,
                            const Distribution1D *lightDistrib = nullptr);

// RIS with inverse cdf sampling + Hilbert curve candidates + blue-noise
Spectrum InverseCDFLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            int N = 1, bool handleMedia = false,
                            const Distribution1D *lightDistrib = nullptr);

// RIS with bidir. cdf sampling + Hilbert curve candidates + blue-noise
Spectrum BidirectionalCDFLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            Float offset, Float u, int N = 1, int n = 0,
                            bool handleMedia = false,
                            const Distribution1D *lightDistrib = nullptr);

// RIS + MIS with reservoir sampling + QMC + blue-noise
Spectrum ReservoirBSDFEnvMIS(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler, int M,
                               const std::shared_ptr<DitherMask> &ditherMask,
                               int N = 1, bool handleMedia = false,
                               const Distribution1D *lightDistrib = nullptr);

// RIS + MIS with inverse cdf sampling + Hilbert curve candidates + blue-noise
Spectrum InverseCDFBSDFEnvMIS(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler, int M,
                               const std::shared_ptr<DitherMask> &ditherMask,
                               int N = 1, bool handleMedia = false,
                               const Distribution1D *lightDistrib = nullptr);

// RIS + MIS with bidir. cdf sampling + Hilbert curve candidates + blue-noise
Spectrum BidirectionalBSDFEnvMIS(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler, int M,
                               const std::shared_ptr<DitherMask> &ditherMask,
                               Float offset, Float u, int N = 1, int i = 0,
                               bool handleMedia = false,
                               const Distribution1D *lightDistrib = nullptr);

// Select a light with random 'u', then rescale it
int ChooseLight(Float u, Float &pdf, int nLights,
                const Distribution1D *lightDistrib = nullptr,
                Float *uRemapped = nullptr);

// Invert CDF using bisection method
int BisectInvert(std::vector<Float> cdf, Float u);

// Calculate unshadowed contribution and candidate weight
Float ComputeCandidateWeight(const Interaction &it, const Light &light,
                                const Point2f &uLight, Float lightChoicePdf,
                                Float *targetPdf, Float *candidatePdf,
                                bool specular = false);

// Sample distance distributed according to transmittance
Float DistanceSampling(Float u, const Ray &ray,
                       const Spectrum &sigma_t, Float dMax);

// Pdf of distance sampling
Float DistancePdf(const Spectrum &Tr, const Spectrum &sigma_t, Float dMax);

// Calculate unshadowed contribution inside volume
// Both functions essentially compute the same value
//  they have small differences in variables being used
//  ideally they should be unified
Float TargetFunctionVolume(const Ray &ray, Float t, const MediumInteraction &mi,
                            const Spectrum &sigma_t, const Light &light,
                            const Point2f &uLight, Sampler &sampler,
                            const Shape *shape);
Float UnshadowedContributionVolume(Spectrum *f, Float *G,
                                   const MediumInteraction &mi,
                                   const Vector3f &wi,
                                   const Spectrum &sigma_t,
                                   const std::shared_ptr<Light> &light,
                                   Spectrum *Tr, const Interaction &pLight,
                                   Float sourcePdf, const Spectrum &Li);

// Sample a point on a light with pdf in area measure
Spectrum AreaSampleOneLight(Point2f uLight, const Scene &scene,
                            int nLights, std::shared_ptr<Light> &light,
                            const MediumInteraction &mi,
                            Float *lightPdf, Float &lightChoicePdf,
                            Vector3f *wi, Interaction *pLight,
                            VisibilityTester *visibility,
                            const Distribution1D *lightDistrib);

// Calculate unshadowed contribution for wi sampled by BSDF in env. map
Float ComputeCandidateWeightBSDFMIS(const Interaction &it,
                        const Point2f &uShading, const Light &light,
                        const Scene &scene, Sampler &sampler,
                        Float *targetPdf, Float *candidatePdf,
                        Vector3f *sampledwi, int M, int mi,
                        MemoryArena &arena, bool handleMedia = false,
                        bool specular = false);

// Calculate unshadowed contribution for wi sampled by env. map
Float ComputeCandidateWeightLightMIS(const Interaction &it,
                        const Point2f &uLight, const Light &light,
                        const Scene &scene, Sampler &sampler,
                        Float *targetPdf, Float *candidatePdf,
                        Vector3f *sampledwi, int M, int mi,
                        MemoryArena &arena, bool handleMedia = false,
                        bool specular = false);

Spectrum EstimateDirect(const Interaction &it, const Point2f &uShading,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia = false,
                        bool specular = false);

// pbrt's EstimateDirect with light sampling only (no BSDF MIS)
Spectrum EstimateDirectLightOnly(const Interaction &it, const Light &light,
                        const Point2f &uLight, const Scene &scene,
                        Sampler &sampler,
                        bool handleMedia = false, bool specular = false);

// pbrt's EstimateDirect with a given direction 'wi'
Spectrum EstimateDirectWi(const Interaction &it, const Vector3f &wi,
                        const Light &light, const Scene &scene,
                        Sampler &sampler, MemoryArena &arena,
                        bool handleMedia = false,
                        bool specular = false);

std::unique_ptr<Distribution1D> ComputeLightPowerDistribution(
    const Scene &scene);

// SamplerIntegrator Declarations
class SamplerIntegrator : public Integrator {
  public:
    // SamplerIntegrator Public Methods
    SamplerIntegrator(std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
        : camera(camera), sampler(sampler), pixelBounds(pixelBounds) {}
    virtual void Preprocess(const Scene &scene, Sampler &sampler) {}
    void Render(const Scene &scene);
    virtual Spectrum Li(const RayDifferential &ray, const Scene &scene,
                        Sampler &sampler, MemoryArena &arena,
                        int depth = 0) const = 0;
    Spectrum SpecularReflect(const RayDifferential &ray,
                             const SurfaceInteraction &isect,
                             const Scene &scene, Sampler &sampler,
                             MemoryArena &arena, int depth) const;
    Spectrum SpecularTransmit(const RayDifferential &ray,
                              const SurfaceInteraction &isect,
                              const Scene &scene, Sampler &sampler,
                              MemoryArena &arena, int depth) const;
    Bounds2i GetPixelBounds() const { return pixelBounds; }
    std::shared_ptr<Sampler> GetSampler() const { return sampler; }

  protected:
    // SamplerIntegrator Protected Data
    std::shared_ptr<const Camera> camera;

  private:
    // SamplerIntegrator Private Data
    std::shared_ptr<Sampler> sampler;
    const Bounds2i pixelBounds;
};

}  // namespace pbrt

#endif  // PBRT_CORE_INTEGRATOR_H
