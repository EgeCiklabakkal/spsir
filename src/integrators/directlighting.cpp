
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


// integrators/directlighting.cpp*
#include "integrators/directlighting.h"
#include "interaction.h"
#include "paramset.h"
#include "camera.h"
#include "film.h"
#include "stats.h"
#include "progressreporter.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// DirectLightingIntegrator Method Definitions
void DirectLightingIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    if (strategy == LightStrategy::UniformSampleAll) {
        // Compute number of samples to use for each light
        for (const auto &light : scene.lights)
            nLightSamples.push_back(sampler.RoundCount(light->nSamples));

        // Request samples for sampling all lights
        for (int i = 0; i < maxDepth; ++i) {
            for (size_t j = 0; j < scene.lights.size(); ++j) {
                sampler.Request2DArray(nLightSamples[j]);
                sampler.Request2DArray(nLightSamples[j]);
            }
        }
    } 

    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

Spectrum DirectLightingIntegrator::Li(const RayDifferential &ray,
                                      const Scene &scene, Sampler &sampler,
                                      MemoryArena &arena, int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L += light->Le(ray);
        return L;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf)
        return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth);
    Vector3f wo = isect.wo;
    // Compute emitted light if ray hit an area light source
    L += isect.Le(wo);
    if (scene.lights.size() > 0) {
        // Compute direct lighting for _DirectLightingIntegrator_ integrator
        const Distribution1D *lightDistrib = 
            lightDistribution->Lookup(isect.p);
        if (strategy == LightStrategy::UniformSampleAll) {
            L += UniformSampleAllLights(isect, scene, arena, sampler,
                                        nLightSamples);
        } else if (strategy == LightStrategy::LightOnly) {
            // Direct lighting with light sampling only
            if (risStrategy == RISStrategy::Reservoir) {
                L += ReservoirLightOnly(isect, scene, arena, sampler, M,
                                        ditherMask, N, false, lightDistrib);
            } else if (risStrategy == RISStrategy::InverseCDF) {
                L += InverseCDFLightOnly(isect, scene, arena, sampler, M,
                                         ditherMask, N, false, lightDistrib);
            } else if (risStrategy == RISStrategy::BidirectionalCDF) {
                Spectrum LNRIS(0.f);
                Point2i currentPixel = sampler.GetCurrentPixel();

                // Blue-noise offset for candidates (on the Hilbert Curve)
                Float offset = ditherMask->Value(currentPixel, 0);

                // Blue-noise offset for canonical input randoms
                Float u = ditherMask->Value(currentPixel, 1);

                for (int i = 0; i < N; ++i) {
                    // Sample 1 out of M / N
                    LNRIS += BidirectionalCDFLightOnly(
                                isect, scene, arena, sampler,
                                M, ditherMask, offset,
                                std::fmod(RadicalInverse(0, i) + u, 1.0),
                                N, i, false, lightDistrib);
                }
                L += LNRIS / Float(N);
            } 
        } else if (strategy == LightStrategy::BSDFEnvMIS) {
            // Direct lighting with MIS inside Env. map
            if (risStrategy == RISStrategy::Reservoir) {
                L += ReservoirBSDFEnvMIS(isect, scene, arena, sampler, M,
                                        ditherMask, N, false, lightDistrib);
            } else if (risStrategy == RISStrategy::InverseCDF) {
                L += InverseCDFBSDFEnvMIS(isect, scene, arena, sampler, M,
                                        ditherMask, N, false, lightDistrib);
            } else if (risStrategy == RISStrategy::BidirectionalCDF) {
                // Sample 1 out of M / N
                Spectrum LNRIS(0.f);
                Point2i currentPixel = sampler.GetCurrentPixel();

                // Blue-noise offset for candidates (on the Hilbert Curve)
                Float offset = ditherMask->Value(currentPixel, 0);

                // Blue-noise offset for canonical input randoms
                Float u = ditherMask->Value(currentPixel, 1);
                for (int i = 0; i < N; ++i) {
                    // Sample 1 out of M / N
                    LNRIS += BidirectionalBSDFEnvMIS(
                                isect, scene, arena, sampler,
                                M, ditherMask, offset,
                                std::fmod(RadicalInverse(0, i) + u, 1.0),
                                N, i, false, lightDistrib);
                }
                L += LNRIS / Float(N);
            } 
        } else
            L += UniformSampleOneLight(isect, scene,
                    arena, sampler, false, lightDistrib);
    }
    if (depth + 1 < maxDepth) {
        // Trace rays for specular reflection and refraction
        L += SpecularReflect(ray, isect, scene, sampler, arena, depth);
        L += SpecularTransmit(ray, isect, scene, sampler, arena, depth);
    }
    return L;
}

DirectLightingIntegrator *CreateDirectLightingIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int M = params.FindOneInt("M", 32);
    int N = params.FindOneInt("N", 1);
    std::string maskpath = params.FindOneString("mask", "mask.mask");
    int maskOffsetSeed = params.FindOneInt("maskoffsetseed", 0);
    std::string st = params.FindOneString("strategy", "all");
    std::string risst = params.FindOneString("risstrategy", "reservoirsampling");
    std::string lst = params.FindOneString("lightsamplestrategy", "power");

    LightStrategy strategy;
    if (st == "one")
        strategy = LightStrategy::UniformSampleOne;
    else if (st == "all")
        strategy = LightStrategy::UniformSampleAll;
    else if (st == "lightonly")
        strategy = LightStrategy::LightOnly;
    else if (st == "bsdfenvmis")
        strategy = LightStrategy::BSDFEnvMIS;
    else {
        Warning(
            "Strategy \"%s\" for direct lighting unknown. "
            "Using \"all\".",
            st.c_str());
        strategy = LightStrategy::UniformSampleAll;
    }

    RISStrategy risStrategy;
    if (risst == "reservoir")
        risStrategy = RISStrategy::Reservoir;
    else if (risst == "inversecdf")
        risStrategy = RISStrategy::InverseCDF;
    else if (risst == "bidirectional")
        risStrategy = RISStrategy::BidirectionalCDF;
    else {
        Warning(
            "Strategy \"%s\" for RIS unknown. "
            "Using \"reservoir\".",
            risst.c_str());
        risStrategy = RISStrategy::Reservoir;
    }

    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }

    // Read dither mask, default is file "mask.mask"
    std::shared_ptr<DitherMask> ditherMask =
        std::make_shared<DitherMask>(AbsolutePath(ResolveFilename(maskpath)));

    // mask offset
    Point2f maskOffsetU = R2Sample(maskOffsetSeed);
    ditherMask->SetOffset(maskOffsetU);

    return new DirectLightingIntegrator(strategy, maxDepth, camera, sampler,
                                        pixelBounds, M, N, risStrategy,
                                        ditherMask, lst);
}

}  // namespace pbrt
