
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

#ifndef PBRT_INTEGRATORS_VOLPATH_H
#define PBRT_INTEGRATORS_VOLPATH_H

// integrators/volpath.h*
#include "pbrt.h"
#include "integrator.h"
#include "scene.h"
#include "lightdistrib.h"
#include "dithermask.h"

namespace pbrt {

// Strategy used in radiance calculation
enum class VolPathLiStrategy { Default, LightDriven, DistDir };

// VolPathIntegrator Declarations
class VolPathIntegrator : public SamplerIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds, Float rrThreshold = 1,
                      const std::string &lightSampleStrategy = "spatial",
                      const VolPathLiStrategy &radianceStrategy =
                        VolPathLiStrategy::Default,
                      const std::shared_ptr<DitherMask> &ditherMask = nullptr)
        : SamplerIntegrator(camera, sampler, pixelBounds),
          maxDepth(maxDepth),
          rrThreshold(rrThreshold),
          lightSampleStrategy(lightSampleStrategy),
          radianceStrategy(radianceStrategy),
          ditherMask(ditherMask) { }
    void Preprocess(const Scene &scene, Sampler &sampler);
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

  private:
    Spectrum LiDefault(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
    Spectrum LiLightDriven(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
    Spectrum LiDistDir(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

    // VolPathIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    const std::string lightSampleStrategy;
    const VolPathLiStrategy radianceStrategy;
    std::shared_ptr<LightDistribution> lightDistribution;
    const std::shared_ptr<DitherMask> ditherMask;
};

VolPathIntegrator *CreateVolPathIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_VOLPATH_H
