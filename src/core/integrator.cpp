
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

// core/integrator.cpp*
#include "integrator.h"
#include "scene.h"
#include "interaction.h"
#include "sampling.h"
#include "parallel.h"
#include "film.h"
#include "sampler.h"
#include "integrator.h"
#include "progressreporter.h"
#include "camera.h"
#include "stats.h"
#include "reservoir.h"
#include "hilbertcurve.h"
#include "lights/diffuse.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// Integrator Method Definitions
Integrator::~Integrator() {}

// Integrator Utility Functions
Spectrum UniformSampleAllLights(const Interaction &it, const Scene &scene,
                                MemoryArena &arena, Sampler &sampler,
                                const std::vector<int> &nLightSamples,
                                bool handleMedia) {
    ProfilePhase p(Prof::DirectLighting);
    Spectrum L(0.f);
    for (size_t j = 0; j < scene.lights.size(); ++j) {
        // Accumulate contribution of _j_th light to _L_
        const std::shared_ptr<Light> &light = scene.lights[j];
        int nSamples = nLightSamples[j];
        const Point2f *uLightArray = sampler.Get2DArray(nSamples);
        const Point2f *uScatteringArray = sampler.Get2DArray(nSamples);
        if (!uLightArray || !uScatteringArray) {
            // Use a single sample for illumination from _light_
            Point2f uLight = sampler.Get2D();
            Point2f uScattering = sampler.Get2D();
            L += EstimateDirect(it, uScattering, *light, uLight, scene, sampler,
                                arena, handleMedia);
        } else {
            // Estimate direct lighting using sample arrays
            Spectrum Ld(0.f);
            for (int k = 0; k < nSamples; ++k)
                Ld += EstimateDirect(it, uScatteringArray[k], *light,
                                     uLightArray[k], scene, sampler, arena,
                                     handleMedia);
            L += Ld / nSamples;
        }
    }
    return L;
}

Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia, const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    int nLights = int(scene.lights.size());
    if (nLights == 0) return Spectrum(0.f);
    int lightNum;
    Float lightPdf;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sampler.Get1D(), &lightPdf);
        if (lightPdf == 0) return Spectrum(0.f);
    } else {
        lightNum = std::min((int)(sampler.Get1D() * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }
    const std::shared_ptr<Light> &light = scene.lights[lightNum];
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    return EstimateDirect(it, uScattering, *light, uLight,
                          scene, sampler, arena, handleMedia) / lightPdf;
}

Spectrum ReservoirLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            int N, bool handleMedia,
                            const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());

    Reservoir<PSSLightSample> r(N); // Reservoir of size N
    Point2f uLight;                 // 2D random to sample light
    Float uLightNum, uLightXRemapped, lightChoicePdf;

    // Blue-noise offset for candidates
    Float offsetx = ditherMask->Value(currentPixel, 0);
    Float offsety = ditherMask->Value(currentPixel, 1);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 2);
    std::vector<Float> u(N); // Canonical random numbers used in sampling
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    // Generate candidates
    for (int i = 0; i < M; ++i) {
        // Halton candidates offset by dither mask
        uLight.x = std::fmod(RadicalInverse(0, i) + offsetx, 1.0);
        uLight.y = std::fmod(RadicalInverse(1, i) + offsety, 1.0);

        // Select a light using the first dimension of the 2D candidate
        uLightNum = uLight.x;
        int lightNum = ChooseLight(uLightNum, lightChoicePdf,
                                   nLights, lightDistrib, &uLightXRemapped);
        uLight.x = uLightXRemapped; // Rescale first dimension
        const std::shared_ptr<Light> &light = scene.lights[lightNum];

        // Compute the weight of the candidate
        Float targetPdf, candidatePdf;
        Float w = ComputeCandidateWeight(it, *light, uLight, lightChoicePdf,
                                    &targetPdf, &candidatePdf, false);
        PSSLightSample candidate(uLight, uLightNum, light, targetPdf);

        // Update reservoir
        r.update(candidate, w, u);
    }
    if (!r.wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    for (int i = 0; i < N; ++i) {
        // Approximate RIS pdf^-1 W
        // same as r.W from
        // Spatiotemporal reservoir resampling for real-time ray tracing 
        //  with dynamic direct lighting, Bitterli et. al., 2020
        Float W = (r.y[i].targetPdf) ? (Float(1) / r.y[i].targetPdf) *
                                            (r.wsum / Float(r.M)) : 0.f;

        // Evaluate the integrand for the sample inside the reservoir
        //  (visibility also evaluated)
        L += EstimateDirectLightOnly(it, *(r.y[i].light), r.y[i].uLight,
                                        scene, sampler, false, false) * W;
    }

    // Average N samples
    return L / Float(N);
}

Spectrum InverseCDFLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            int N, bool handleMedia,
                            const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());

    Point2f uLight; // 2D random to sample light
    Float uLightNum, uLightXRemapped, lightChoicePdf;

    // Blue-noise offset for candidates
    Float offset = ditherMask->Value(currentPixel, 0);

    // CDF variables
    std::vector<PSSLightSample> candidates(M);
    std::vector<Float> cdf(M);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 1);
    std::vector<Float> u(N); // Canonical random numbers used in sampling
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    HilbertCurve2D hc(32); // 2D Hilbert Curve of order 32
    // Generate candidates
    for (int i = 0; i < M; ++i) {
        // Hilbert Curve candidates offset by dither mask
        uLight = hc.sample((Float(i) / M) + (offset / M));

        // Select a light using the first dimension of the 2D candidate
        uLightNum = uLight.x;
        int lightNum = ChooseLight(uLightNum, lightChoicePdf,
                                   nLights, lightDistrib, &uLightXRemapped);
        uLight.x = uLightXRemapped; // Rescale first dimension
        const std::shared_ptr<Light> &light = scene.lights[lightNum];

        // Compute the weight of the candidate
        Float targetPdf, candidatePdf;
        Float w = ComputeCandidateWeight(it, *light, uLight, lightChoicePdf,
                                    &targetPdf, &candidatePdf, false);
        PSSLightSample candidate(uLight, uLightNum, light, targetPdf);

        // Store the candidates and build the cdf
        candidates[i] = candidate;
        cdf[i] = (i > 0) ? cdf[i - 1] + w : w;
    }
    Float wsum = cdf[M - 1];
    if (!wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    // Apply bisection method to invert the cdf
    for (int i = 0; i < N; ++i) {
        int sampleIndex(0);
        if (u[i] == 0) {
            int j = 0;
            while (cdf[j] == 0)
                j++;
            sampleIndex = j;
        } else {
            sampleIndex = BisectInvert(cdf, u[i]);
        }

        // Approximate RIS pdf^-1 W
        Float W = (candidates[sampleIndex].targetPdf) ?
                    (Float(1) / candidates[sampleIndex].targetPdf) *
                        (wsum / Float(M)) : 0.f;

        // Evaluate the integrand for the final samples
        L += EstimateDirectLightOnly(it, *(candidates[sampleIndex].light), 
                                        candidates[sampleIndex].uLight, scene,
                                        sampler, false, false) * W;
    }

    // Average N samples
    return L / Float(N);
}

Spectrum BidirectionalCDFLightOnly(const Interaction &it, const Scene &scene,
                            MemoryArena &arena, Sampler &sampler, int M,
                            const std::shared_ptr<DitherMask> &ditherMask,
                            Float offset, Float u, int N, int n,
                            bool handleMedia,
                            const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());

    PSSLightSample selectedSample;  // Sample selected by bidir. cdf
    Point2f uLight;                 // 2D random to sample light
    Float uLightNum, uLightXRemapped, lightChoicePdf;

    // Bidirectional CDF sampling variables
    int indexFront(0);
    int indexBack(M / N - 1);
    // Calculate weights of front and back
    Float wsumFront(0.f);
    Float wsumBack(0.f);

    HilbertCurve2D hc(32); // 2D Hilbert Curve of order 32
    {   // Front
        // Hilbert Curve candidates offset by dither mask
        // candidate order is interleaved:
        // Select from candidates #0, N, 2N, ... , (M / N - 1)N
        // Select from candidates #1, N+1, 2N+1, ..., (M / N - 1)N + 1
        // ...
        uLight = hc.sample((Float(indexFront * N + n) / M) + (offset / M));

        // Select a light using the first dimension of the 2D candidate
        uLightNum = uLight.x;
        int lightNum = ChooseLight(uLightNum, lightChoicePdf,
                                   nLights, lightDistrib, &uLightXRemapped);
        uLight.x = uLightXRemapped; // Rescale first dimension
        const std::shared_ptr<Light> &light = scene.lights[lightNum];

        // Compute the weight of the candidate
        Float targetPdf, candidatePdf;
        Float w = ComputeCandidateWeight(it, *light, uLight, lightChoicePdf,
                                    &targetPdf, &candidatePdf, false);
        selectedSample = PSSLightSample(uLight, uLightNum, light, targetPdf);

        // Update front weight
        wsumFront += w;
    } { // Back
        // Hilbert curve candidates offset by dither mask
        uLight = hc.sample((Float(indexBack * N + n) / M) + (offset / M));

        // Select a light using the first dimension of the 2D candidate
        uLightNum = uLight.x;
        int lightNum = ChooseLight(uLightNum, lightChoicePdf,
                                   nLights, lightDistrib, &uLightXRemapped);
        uLight.x = uLightXRemapped; // Rescale first dimension
        const std::shared_ptr<Light> &light = scene.lights[lightNum];
        
        // Compute the weight of the candidate
        Float targetPdf, candidatePdf;
        Float w = ComputeCandidateWeight(it, *light, uLight, lightChoicePdf,
                                    &targetPdf, &candidatePdf, false);

        // Follow selectedSample with front only

        // Update back weight, avoid adding the same weight twice
        wsumBack += (indexFront != indexBack) ? w : 0.f;
    }
    while (indexFront != indexBack) {
        if (wsumFront <= u * (wsumFront + wsumBack)) {
            // Advance front index
            indexFront++;

            // Hilbert curve candidates offset by dither mask
            uLight = hc.sample((Float(indexFront * N + n) / M) + (offset / M));

            // Select a light using the first dimension of the 2D candidate
            uLightNum = uLight.x;
            int lightNum = ChooseLight(uLightNum, lightChoicePdf, nLights,
                                        lightDistrib, &uLightXRemapped);
            uLight.x = uLightXRemapped; // Rescale first dimension
            const std::shared_ptr<Light> &light = scene.lights[lightNum];

            // Compute the weight of the candidate
            Float targetPdf, candidatePdf;
            Float w = ComputeCandidateWeight(it, *light, uLight,
                                                lightChoicePdf, &targetPdf,
                                                &candidatePdf, false);
            selectedSample = PSSLightSample(uLight, uLightNum,
                                            light, targetPdf);

            // Update front weight, avoid adding the same weight twice
            wsumFront += (indexFront != indexBack) ? w : 0.f;
        } else {
            // Decrease back index
            indexBack--;

            // Hilbert curve candidates offset by dither mask
            uLight = hc.sample((Float(indexBack * N + n) / M) + (offset / M));
            
            // Select a light using the first dimension of the 2D candidate
            uLightNum = uLight.x;
            int lightNum = ChooseLight(uLightNum, lightChoicePdf, nLights,
                                        lightDistrib, &uLightXRemapped);
            uLight.x = uLightXRemapped; // Rescale first dimension
            const std::shared_ptr<Light> &light = scene.lights[lightNum];

            // Compute the weight of the candidate
            Float targetPdf, candidatePdf;
            Float w = ComputeCandidateWeight(it, *light, uLight,
                                                lightChoicePdf, &targetPdf,
                                                &candidatePdf, false);
        
            // Follow selectedSample with front only
 
            // Update back weight, avoid adding the same weight twice
            wsumBack += (indexFront != indexBack) ? w : 0.f;
        }
    }
    Float wsum = wsumFront + wsumBack;
    if (!wsum) // No contribution
        return Spectrum(0);

    // Approximate RIS pdf^-1 W
    Float W = (selectedSample.targetPdf) ?
                (Float(1) / selectedSample.targetPdf) *
                    (wsum / Float(M / N)) : 0.f;

    // Evaluate the integrand for the selected sample
    return EstimateDirectLightOnly(it, *(selectedSample.light),
                                    selectedSample.uLight, scene,
                                    sampler, false, false) * W;
}

Spectrum ReservoirBSDFEnvMIS(const Interaction &it, const Scene &scene,
                        MemoryArena &arena, Sampler &sampler, int M,
                        const std::shared_ptr<DitherMask> &ditherMask,
                        int N,
                        bool handleMedia, const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());
    Float targetPdf, candidatePdf;
    Vector3f wi;

    // Reservoir to store candidate (direction) and it's target pdf
    Reservoir<std::pair<Vector3f, Float>> r(N);

    // Blue-noise offset for candidates
    Float offsetx = ditherMask->Value(currentPixel, 0);
    Float offsety = ditherMask->Value(currentPixel, 1);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 2);
    std::vector<Float> u(N); // Canonical random numbers used in sampling
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    const std::shared_ptr<Light> &light = scene.lights[0]; // env. map
    int mi = M / 2;     // # of BSDF candidates
    int mj = M - mi;    // # of Light candidates
    for (int i = 0; i < mi; ++i) { // BSDF candidates
        // Halton candidates offset by dither mask
        Point2f uScattering;
        uScattering.x = std::fmod(RadicalInverse(0, i) + offsetx, 1.0);
        uScattering.y = std::fmod(RadicalInverse(1, i) + offsety, 1.0);

        // Compute the weight of the candidate and get sampled wi
        Float w = ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                *light, scene, sampler,
                                                &targetPdf, &candidatePdf,
                                                &wi, M, mi,
                                                arena, handleMedia);

        // Update reservoir
        r.update(std::pair<Vector3f, Float>(wi, targetPdf), w, u);
    }
    for (int i = 0; i < mj; ++i) { // Light candidates
        // Halton candidates offset by dither mask
        Point2f uLight;
        uLight.x = std::fmod(RadicalInverse(0, i) + offsetx, 1.0);
        uLight.y = std::fmod(RadicalInverse(1, i) + offsety, 1.0);

        // Compute the weight of the candidate and get sampled wi
        Float w = ComputeCandidateWeightLightMIS(it, uLight,
                                                *light, scene, sampler,
                                                &targetPdf, &candidatePdf,
                                                &wi, M, mj,
                                                arena, handleMedia);

        // Update reservoir
        r.update(std::pair<Vector3f, Float>(wi, targetPdf), w, u);
    }
    if (!r.wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    for (int i = 0; i < N; ++i) {
        // Approximate RIS pdf^-1 W
        Float W = (r.y[i].second) ? (Float(1) / r.y[i].second) * r.wsum : 0.f;

        // Evaluate the integrand for the sample inside the reservoir
        L += EstimateDirectWi(it, r.y[i].first, *light, scene, sampler,
                                arena, handleMedia) * W;
    }

    // Average N samples
    return L / Float(N);
}

Spectrum InverseCDFBSDFEnvMIS(const Interaction &it, const Scene &scene,
                        MemoryArena &arena, Sampler &sampler, int M,
                        const std::shared_ptr<DitherMask> &ditherMask,
                        int N,
                        bool handleMedia, const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());
    Float targetPdf, candidatePdf;
    Vector3f wi;
 
    // CDF variables
    std::vector<std::pair<Vector3f, Float>> candidates(M);
    std::vector<Float> cdf(M);

    // Blue-noise offset for candidates
    Float offset = ditherMask->Value(currentPixel, 0);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 1);
    std::vector<Float> u(N); // Canonical random numbers used in sampling
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    const std::shared_ptr<Light> &light = scene.lights[0]; // env. map
    int mi = M / 2;     // # of BSDF candidates
    int mj = M - mi;    // # of Light candidates
    HilbertCurve2D hc(32);
    for (int i = 0; i < mi; ++i) { // BSDF candidates
        // Hilbert Curve candidates offset by dither mask
        Point2f uScattering = hc.sample(Float(i) / mi + (offset / mi));

        // Compute the weight of the candidate and get sampled wi
        Float w = ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                *light, scene, sampler,
                                                &targetPdf, &candidatePdf,
                                                &wi, M, mi,
                                                arena, handleMedia);
        
        // Store the candidates and build the cdf
        candidates[i] = std::pair<Vector3f, Float>(wi, targetPdf);
        cdf[i] = (i > 0) ? cdf[i - 1] + w : w;
    }
    for (int i = 0; i < mj; ++i) { // Light candidates
        // Hilbert Curve candidates offset by dither mask
        Point2f uLight = hc.sample(Float(i) / mj + (offset / mj));

        // Compute the weight of the candidate and get sampled wi
        Float w = ComputeCandidateWeightLightMIS(it, uLight,
                                                *light, scene, sampler,
                                                &targetPdf, &candidatePdf,
                                                &wi, M, mj,
                                                arena, handleMedia);

        // Store the candidates and build the cdf
        candidates[mi + i] = std::pair<Vector3f, Float>(wi, targetPdf);
        cdf[mi + i] = cdf[mi + i - 1] + w;
    }
    Float wsum = cdf[M - 1];
    if (!wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    // Apply bisection method to invert the cdf
    for (int i = 0; i < N; ++i) {
        int sampleIndex(0);
        if (u[i] == 0) {
            int j = 0;
            while (cdf[j] == 0)
                j++;
            sampleIndex = j;
        } else {
            sampleIndex = BisectInvert(cdf, u[i]);
        }

        // Approximate RIS pdf^-1 W
        Float W = (candidates[sampleIndex].second) ? 
                    (Float(1) / candidates[sampleIndex].second) * wsum : 0.f;

        // Evaluate the integrand for the final samples
        L += EstimateDirectWi(it, candidates[sampleIndex].first, *light,
                                scene, sampler, arena, handleMedia) * W;
    }

    // Average N samples
    return L / Float(N);
}

Spectrum BidirectionalBSDFEnvMIS(const Interaction &it, const Scene &scene,
                        MemoryArena &arena, Sampler &sampler, int M,
                        const std::shared_ptr<DitherMask> &ditherMask,
                        Float offset, Float u,
                        int N, int n,
                        bool handleMedia, const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    Point2i currentPixel = sampler.GetCurrentPixel();
    int nLights = int(scene.lights.size());
    Float targetPdf, candidatePdf, selectedTargetPdf, w;
    Vector3f wi, selectedwi;

    const std::shared_ptr<Light> &light = scene.lights[0]; // env. map
    int mi = M / 2;     // # of BSDF candidates
    int mj = M - mi;    // # of Light candidates

    // Bidirectional CDF sampling variables
    int indexFront(0);
    int indexBack(M/N - 1);
    // Calculate weights of front and back
    Float wsumFront(0.f);
    Float wsumBack(0.f);

    HilbertCurve2D hc(32);
    {   // Front
        if (indexFront * N + n < mi) { // BSDF candidates
            // Hilbert Curve candidates offset by dither mask
            Point2f uScattering = hc.sample(Float(indexFront * N + n) / mi +
                                                (offset / mi));

            // Compute the weight of the candidate and get sampled wi
            wsumFront += ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mi/N,
                                                    arena, handleMedia);
        } else { // Light candidates
            // Hilbert Curve candidates offset by dither mask
            Point2f uLight = hc.sample(Float(indexFront * N + n - mi) / mj +
                                                (offset / mj));
            
            // Compute the weight of the candidate and get sampled wi
            wsumFront += ComputeCandidateWeightLightMIS(it, uLight,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mj/N,
                                                    arena, handleMedia);
        }

        // Update selected sample
        selectedwi = wi;
        selectedTargetPdf = targetPdf;
    } { // Back
        if (indexBack * N + n < mi) { // BSDF candidates
            // Hilbert Curve candidates offset by dither mask
            Point2f uScattering = hc.sample(Float(indexBack * N + n) / mi +
                                                (offset / mi));

            // Compute the weight of the candidate and get sampled wi
            w = ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mi/N,
                                                    arena, handleMedia);
        } else { // Light candidates
            // Hilbert Curve candidates offset by dither mask
            Point2f uLight = hc.sample(Float(indexBack * N + n - mi) / mj +
                                                (offset / mj));

            // Compute the weight of the candidate and get sampled wi
            w = ComputeCandidateWeightLightMIS(it, uLight,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mj/N,
                                                    arena, handleMedia);
        }

        // Follow selected sample with front only

        // Update back weight, avoid adding the same weight twice
        wsumBack += (indexFront != indexBack) ? w : 0.f;
    }
    // Bidirectional CDF
    while (indexFront != indexBack) {
        if (wsumFront < u * (wsumFront + wsumBack)) {
            // Advance front index
            indexFront++;
            if (indexFront * N + n < mi) { // BSDF candidates
                // Hilbert Curve candidates offset by dither mask
                Point2f uScattering = hc.sample(Float(indexFront * N + n) / mi
                                                    + (offset / mi));

                // Compute the weight of the candidate and get sampled wi
                w = ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mi/N,
                                                    arena, handleMedia);
            } else { // Light candidates
                    // Hilbert Curve candidates offset by dither mask
                Point2f uLight = hc.sample(Float(indexFront * N + n - mi) / mj
                                                    + (offset / mj));

                // Compute the weight of the candidate and get sampled wi
                w = ComputeCandidateWeightLightMIS(it, uLight,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mj/N,
                                                    arena, handleMedia);
            }

            // Update selected sample
            selectedwi = wi;
            selectedTargetPdf = targetPdf;

            // Update front weight, avoid adding the same weight twice
            wsumFront += (indexFront != indexBack) ? w : 0.f;
        } else {
            // Decrease back index
            indexBack--;
            if (indexBack * N + n < mi) { // BSDF candidates
                // Hilbert Curve candidates offset by dither mask
                Point2f uScattering = hc.sample(Float(indexBack * N + n) / mi +
                                        (offset / mi));

                // Compute the weight of the candidate and get sampled wi
                w = ComputeCandidateWeightBSDFMIS(it, uScattering,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mi/N,
                                                    arena, handleMedia);
            } else { // Light candidates
                // Hilbert Curve candidates offset by dither mask
                Point2f uLight = hc.sample(Float(indexBack * N + n - mi) / mj +
                                        (offset / mj));

                // Compute the weight of the candidate and get sampled wi
                w = ComputeCandidateWeightLightMIS(it, uLight,
                                                    *light, scene, sampler,
                                                    &targetPdf, &candidatePdf,
                                                    &wi, M/N, mj/N,
                                                    arena, handleMedia);
            }

            // Follow selected sample with front only

            // Update back weight, avoid adding the same weight twice
            wsumBack += (indexFront != indexBack) ? w : 0.f;
        }
    }

    Float wsum = wsumFront + wsumBack;
    if (!wsum) // No contribution
        return Spectrum(0);

    // Approximate RIS pdf^-1 W
    Float W = (selectedTargetPdf) ? (Float(1) / selectedTargetPdf) * wsum 
                                        : 0.f;

    // Evaluate the integrand for the selected sample
    return EstimateDirectWi(it, selectedwi, *light, scene, sampler,
                                arena, handleMedia) * W;
}

int ChooseLight(Float u, Float &pdf, int nLights,
                const Distribution1D *lightDistrib,
                Float *uRemapped) {
    int lightNum;
    if (lightDistrib) { // Discrete pdf
        lightNum = lightDistrib->SampleDiscrete(u, &pdf, uRemapped);
    } else { // Uniform distribution
        lightNum = std::min((int)(u * nLights), nLights - 1);
        pdf = Float(1) / nLights;
        if (uRemapped) // Rescale u
            *uRemapped = (u - (lightNum * pdf)) / (pdf);
    }

    return lightNum;
}

int BisectInvert(std::vector<Float> cdf, Float u) {
    int L = 0;
    int R = cdf.size() - 1;
    Float p = u * cdf[R];
    while (L < R - 1) {
        int k = (L + R) / Float(2.0);
        if (p > cdf[k])
            L = k;
        else
            R = k;
    }

    return R;
}

Float ComputeCandidateWeight(const Interaction &it, const Light &light,
                                const Point2f &uLight, Float lightChoicePdf,
                                Float *targetPdf, Float *candidatePdf,
                                bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
            << wi << ", pdf: " << lightPdf;
    if (lightChoicePdf > 0 && lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f << ", scatteringPdf: " << scatteringPdf;
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
            VLOG(2) << "  medium p: " << p;
        }
        *targetPdf = RGBSpectrum(f * Li).y(); // f * G * Li
        *candidatePdf = lightPdf * lightChoicePdf;

        return *targetPdf / *candidatePdf;
    }

    *targetPdf = 0.f;
    *candidatePdf = lightPdf * lightChoicePdf;

    return 0.f;
}

Float DistanceSampling(Float u, const Ray & ray,
                        const Spectrum &sigma_t, Float dMax) {
    // Sample distance
    int channel = 0; // Assume equal channels, no random decision
    Float dist = -std::log(1 - u * (1 - std::exp(-sigma_t[channel] * dMax)))
                            / sigma_t[channel];
    return  dist / ray.d.Length();
}

Float DistancePdf(const Spectrum &Tr, const Spectrum &sigma_t, Float dMax) {
    // Distance sampling pdf
    Spectrum density = (sigma_t * Tr) / (Spectrum(1) - Exp(-sigma_t * dMax));
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
        pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }

    return pdf;
}

Float TargetFunctionVolume(const Ray &ray, Float t,
                            const MediumInteraction &mi,
                            const Spectrum &sigma_t, const Light &light,
                            const Point2f &uLight,
                            Sampler &sampler, const Shape *shape) {
    Vector3f wi;
    Float lightPdf;
    VisibilityTester visibility;
    Interaction pLight;
    // Sample a point on the light, return area measure pdf and the point
    Spectrum Li = light.Sample_Li_Area(mi, uLight, &wi, &lightPdf,
                                        &visibility, &pLight);
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute phase function's value for light sample
        // Evaluate phase function for light sampling strategy
        Float p = mi.phase->p(mi.wo, wi);
        Spectrum f = Spectrum(p);
        // Geometry term
        Float G;
        if (IsDeltaLight(light.flags)) {
            G = 1.0 / DistanceSquared(mi.p, pLight.p);
        } else {
            G = AbsDot(pLight.n, -wi) / DistanceSquared(mi.p, pLight.p);
        }

        if (!f.IsBlack()) {
            // Compute transmittance along the initial ray
            Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) *
                                ray.d.Length());

            // Add the transmittance from the intersection to the light
            if (!shape) { // Ray outside, connect to light 
                          // (assume volume throughout scene)
                Tr *= Exp(-sigma_t * (mi.p - pLight.p).Length());
            } else {
                // If the medium is a shape trace a ray to find intersection
                // In our experiments, the medium always exists throughout,
                //  thus this part may be unnecessary
                Ray rayToLight(mi.SpawnRayTo(pLight));
                SurfaceInteraction isect;
                Float tToLight;
                shape->Intersect(rayToLight, &tToLight, &isect);
                if (ray.medium == isect.mediumInterface.inside) { // Ray inside
                    Tr *= Exp(-sigma_t * std::min(tToLight, MaxFloat) *
                                                    rayToLight.d.Length());
                } else { // Ray outside
                    Tr *= Exp(-sigma_t * (mi.p - pLight.p).Length());
                }
            }

            // Avg. Spectrum channels
            Float Trf(0);
            for (int i = 0; i < Spectrum::nSamples; ++i) Trf += Tr[i];
            Trf *= 1 / (Float)Spectrum::nSamples;

            // Unshadowed contribution
            return RGBSpectrum(f * Li * Trf * G).y();
        }
    }

    return 0;
}

Float UnshadowedContributionVolume(Spectrum *f, Float *G,
                                   const MediumInteraction &mi,
                                   const Vector3f &wi,
                                   const Spectrum &sigma_t,
                                   const std::shared_ptr<Light> &light,
                                   Spectrum *Tr, const Interaction &pLight,
                                   Float sourcePdf, const Spectrum &Li) {
    Float targetPdf = 0;
    if (sourcePdf > 0 && !Li.IsBlack()) {
        // Compute phase function's value for light sample
        // Evaluate phase function for light sampling strategy
        Float p = mi.phase->p(mi.wo, wi);
        *f = Spectrum(p);
        // Geometry term
        if (IsDeltaLight(light->flags)) {
            *G = 1.0 / DistanceSquared(mi.p, pLight.p);
        } else {
            *G = AbsDot(pLight.n, -wi) / DistanceSquared(mi.p, pLight.p);
        }

        if (!(*f).IsBlack()) {
            // Compute transmittance from medium to light
            // Tr already contains up to medium interaction
            *Tr *= Exp(-sigma_t * (mi.p - pLight.p).Length());

            // Avg. Spectrum channels
            Float Trf(0);
            for (int i = 0; i < Spectrum::nSamples; ++i) Trf += (*Tr)[i];
            Trf *= 1 / (Float)Spectrum::nSamples;

            // Unshadowed contribution
            targetPdf = RGBSpectrum((*f) * Li * Trf * (*G)).y();
        }
    }

    return targetPdf;
}

Spectrum AreaSampleOneLight(Point2f uLight, const Scene &scene,
                            int nLights, std::shared_ptr<Light> &light,
                            const MediumInteraction &mi,
                            Float *lightPdf, Float &lightChoicePdf,
                            Vector3f *wi, Interaction *pLight,
                            VisibilityTester *visibility,
                            const Distribution1D *lightDistrib) {
    Float uLightXRemapped;
    Float uLightNum = uLight.x;
    int lightNum = ChooseLight(uLightNum, lightChoicePdf, nLights,
                               lightDistrib, &uLightXRemapped);
    uLight.x = uLightXRemapped;
    light = scene.lights[lightNum];
    return light->Sample_Li_Area(mi, uLight, wi, lightPdf, visibility, pLight);
}

Float ComputeCandidateWeightBSDFMIS(const Interaction &it,
                        const Point2f &uScattering, const Light &light,
                        const Scene &scene, Sampler &sampler,
                        Float *targetPdf, Float *candidatePdf,
                        Vector3f *sampledwi, int M, int mi,
                        MemoryArena &arena, bool handleMedia,
                        bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Vector3f wi;
    Float scatteringPdf = 0;
    Spectrum f;
    bool sampledSpecular = false;
    if (it.IsSurfaceInteraction()) {
        // Sample scattered direction for surface interactions
        BxDFType sampledType;
        const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
        f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                 bsdfFlags, &sampledType);
        f *= AbsDot(wi, isect.shading.n);
        sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
    }
    VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
        scatteringPdf;
    if (!f.IsBlack() && scatteringPdf > 0) {
        // Unshadowed contribution in direction wi sampled by BSDF
        Ray ray = it.SpawnRay(wi);
        *sampledwi = wi;
        *targetPdf = RGBSpectrum(f * light.Le(ray)).y();
        *candidatePdf = scatteringPdf;

        Float denomMIS = mi * scatteringPdf + (M - mi) * light.Pdf_Li(it, wi);

        return *targetPdf / denomMIS;
    }
    *sampledwi = wi;
    *targetPdf = 0.f;
    *candidatePdf = scatteringPdf;

    return 0.f;
} 

Float ComputeCandidateWeightLightMIS(const Interaction &it,
                        const Point2f &uLight, const Light &light,
                        const Scene &scene, Sampler &sampler,
                        Float *targetPdf, Float *candidatePdf,
                        Vector3f *sampledwi, int M, int mi,
                        MemoryArena &arena, bool handleMedia,
                        bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li <<
            ", wi: " << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        Float G = 0.f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f <<
                ", scatteringPdf: " << scatteringPdf;
        } 
        // Unshadowed contribution in direction wi sampled by Light
        *sampledwi = wi;
        *targetPdf = RGBSpectrum(f * Li).y();
        *candidatePdf = lightPdf;

        Float denomMIS = mi * lightPdf + (M - mi) * scatteringPdf;

        return *targetPdf / denomMIS;
    }
    *sampledwi = wi;
    *targetPdf = 0.f;
    *candidatePdf = lightPdf;

    return 0.f;
}

Spectrum EstimateDirect(const Interaction &it, const Point2f &uScattering,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia, bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: " 
            << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f << ", scatteringPdf: " << scatteringPdf;
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
            VLOG(2) << "  medium p: " << p;
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
            if (handleMedia) {
                Li *= visibility.Tr(scene, sampler);
                VLOG(2) << "  after Tr, Li: " << Li;
            } else {
              if (!visibility.Unoccluded(scene)) {
                VLOG(2) << "  shadow ray blocked";
                Li = Spectrum(0.f);
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                if (IsDeltaLight(light.flags))
                    Ld += f * Li / lightPdf;
                else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
        Spectrum f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        } else {
            // Sample scattered direction for medium interactions
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
            scatteringPdf;
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }

            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            bool foundSurfaceInteraction =
                handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                            : scene.Intersect(ray, &lightIsect);

            // Add light contribution from material sampling
            Spectrum Li(0.f);
            if (foundSurfaceInteraction) {
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);
            if (!Li.IsBlack()) Ld += f * Li * Tr * weight / scatteringPdf;
        }
    }
    return Ld;
}

Spectrum EstimateDirectLightOnly(const Interaction &it, const Light &light,
                        const Point2f &uLight, const Scene &scene,
                        Sampler &sampler,
                        bool handleMedia, bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li <<
            ", wi: " << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            VLOG(2) << "  medium p: " << p;
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
            if (handleMedia) {
                Li *= visibility.Tr(scene, sampler);
                VLOG(2) << "  after Tr, Li: " << Li;
            } else {
              if (!visibility.Unoccluded(scene)) {
                VLOG(2) << "  shadow ray blocked";
                Li = Spectrum(0.f);
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                Ld += f * Li;
            }
        }
    }

    return Ld;
}

Spectrum EstimateDirectWi(const Interaction &it, const Vector3f &wi,
                          const Light &light, const Scene &scene,
                          Sampler &sampler, MemoryArena &arena,
                          bool handleMedia, bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    Spectrum f;
    bool sampledSpecular = false;
    if (it.IsSurfaceInteraction()) {
        // Evaluate BSDF for light sampling strategy
        const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
        f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
    }
    if (!f.IsBlack()) {
        // Account for light contributions along sampled direction _wi_
        // Find intersection and compute transmittance
        SurfaceInteraction lightIsect;
        Ray ray = it.SpawnRay(wi);
        Spectrum Tr(1.f);
        bool foundSurfaceInteraction =
            handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                        : scene.Intersect(ray, &lightIsect);

        // Add light contribution from material sampling
        Spectrum Li(0.f);
        if (foundSurfaceInteraction) {
            if (lightIsect.primitive->GetAreaLight() == &light)
                Li = lightIsect.Le(-wi);
        } else
            Li = light.Le(ray);
        if (!Li.IsBlack()) Ld += f * Li * Tr;
    }

    return Ld;
}

std::unique_ptr<Distribution1D> ComputeLightPowerDistribution(
    const Scene &scene) {
    if (scene.lights.empty()) return nullptr;
    std::vector<Float> lightPower;
    for (const auto &light : scene.lights)
        lightPower.push_back(light->Power().y());
    return std::unique_ptr<Distribution1D>(
        new Distribution1D(&lightPower[0], lightPower.size()));
}

// SamplerIntegrator Method Definitions
void SamplerIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        ParallelFor2D([&](Point2i tile) {
            // Render section of image corresponding to _tile_

            // Allocate _MemoryArena_ for tile
            MemoryArena arena;

            // Get sampler instance for tile
            int seed = tile.y * nTiles.x + tile.x;
            std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

            // Compute sample bounds for tile
            int x0 = sampleBounds.pMin.x + tile.x * tileSize;
            int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
            int y0 = sampleBounds.pMin.y + tile.y * tileSize;
            int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
            LOG(INFO) << "Starting image tile " << tileBounds;

            // Get _FilmTile_ for tile
            std::unique_ptr<FilmTile> filmTile =
                camera->film->GetFilmTile(tileBounds);

            // Loop over pixels in tile to render them
            for (Point2i pixel : tileBounds) {
                {
                    ProfilePhase pp(Prof::StartPixel);
                    tileSampler->StartPixel(pixel);
                }

                // Do this check after the StartPixel() call; this keeps
                // the usage of RNG values from (most) Samplers that use
                // RNGs consistent, which improves reproducability /
                // debugging.
                if (!InsideExclusive(pixel, pixelBounds))
                    continue;

                do {
                    // Initialize _CameraSample_ for current sample
                    CameraSample cameraSample =
                        // Don't consume random samples for camera sample
                        tileSampler->GetCameraSampleSimple(pixel, true);
                        //tileSampler->GetCameraSample(pixel);

                    // Generate camera ray for current sample
                    RayDifferential ray;
                    Float rayWeight =
                        camera->GenerateRayDifferential(cameraSample, &ray);
                    ray.ScaleDifferentials(
                        1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                    ++nCameraRays;

                    // Evaluate radiance along camera ray
                    Spectrum L(0.f);
                    if (rayWeight > 0) L = Li(ray, scene, *tileSampler, arena);

                    // Issue warning if unexpected radiance value returned
                    if (L.HasNaNs()) {
                        LOG(ERROR) << StringPrintf(
                            "Not-a-number radiance value returned "
                            "for pixel (%d, %d), sample %d. Setting to black.",
                            pixel.x, pixel.y,
                            (int)tileSampler->CurrentSampleNumber());
                        L = Spectrum(0.f);
                    } else if (L.y() < -1e-5) {
                        LOG(ERROR) << StringPrintf(
                            "Negative luminance value, %f, returned "
                            "for pixel (%d, %d), sample %d. Setting to black.",
                            L.y(), pixel.x, pixel.y,
                            (int)tileSampler->CurrentSampleNumber());
                        L = Spectrum(0.f);
                    } else if (std::isinf(L.y())) {
                          LOG(ERROR) << StringPrintf(
                            "Infinite luminance value returned "
                            "for pixel (%d, %d), sample %d. Setting to black.",
                            pixel.x, pixel.y,
                            (int)tileSampler->CurrentSampleNumber());
                        L = Spectrum(0.f);
                    }
                    VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " <<
                        ray << " -> L = " << L;

                    // Add camera ray's contribution to image
                    filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

                    // Free _MemoryArena_ memory from computing image sample
                    // value
                    arena.Reset();
                } while (tileSampler->StartNextSample());
            }
            LOG(INFO) << "Finished image tile " << tileBounds;

            // Merge image tile into _Film_
            camera->film->MergeFilmTile(std::move(filmTile));
            reporter.Update();
        }, nTiles);
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
}

Spectrum SamplerIntegrator::SpecularReflect(
    const RayDifferential &ray, const SurfaceInteraction &isect,
    const Scene &scene, Sampler &sampler, MemoryArena &arena, int depth) const {
    // Compute specular reflection direction _wi_ and BSDF value
    Vector3f wo = isect.wo, wi;
    Float pdf;
    BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_SPECULAR);
    Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, type);

    // Return contribution of specular reflection
    const Normal3f &ns = isect.shading.n;
    if (pdf > 0.f && !f.IsBlack() && AbsDot(wi, ns) != 0.f) {
        // Compute ray differential _rd_ for specular reflection
        RayDifferential rd = isect.SpawnRay(wi);
        if (ray.hasDifferentials) {
            rd.hasDifferentials = true;
            rd.rxOrigin = isect.p + isect.dpdx;
            rd.ryOrigin = isect.p + isect.dpdy;
            // Compute differential reflected directions
            Normal3f dndx = isect.shading.dndu * isect.dudx +
                            isect.shading.dndv * isect.dvdx;
            Normal3f dndy = isect.shading.dndu * isect.dudy +
                            isect.shading.dndv * isect.dvdy;
            Vector3f dwodx = -ray.rxDirection - wo,
                     dwody = -ray.ryDirection - wo;
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);
            rd.rxDirection =
                wi - dwodx + 2.f * Vector3f(Dot(wo, ns) * dndx + dDNdx * ns);
            rd.ryDirection =
                wi - dwody + 2.f * Vector3f(Dot(wo, ns) * dndy + dDNdy * ns);
        }
        return f * Li(rd, scene, sampler, arena, depth + 1) * AbsDot(wi, ns) /
               pdf;
    } else
        return Spectrum(0.f);
}

Spectrum SamplerIntegrator::SpecularTransmit(
    const RayDifferential &ray, const SurfaceInteraction &isect,
    const Scene &scene, Sampler &sampler, MemoryArena &arena, int depth) const {
    Vector3f wo = isect.wo, wi;
    Float pdf;
    const Point3f &p = isect.p;
    const BSDF &bsdf = *isect.bsdf;
    Spectrum f = bsdf.Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                               BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR));
    Spectrum L = Spectrum(0.f);
    Normal3f ns = isect.shading.n;
    if (pdf > 0.f && !f.IsBlack() && AbsDot(wi, ns) != 0.f) {
        // Compute ray differential _rd_ for specular transmission
        RayDifferential rd = isect.SpawnRay(wi);
        if (ray.hasDifferentials) {
            rd.hasDifferentials = true;
            rd.rxOrigin = p + isect.dpdx;
            rd.ryOrigin = p + isect.dpdy;

            Normal3f dndx = isect.shading.dndu * isect.dudx +
                            isect.shading.dndv * isect.dvdx;
            Normal3f dndy = isect.shading.dndu * isect.dudy +
                            isect.shading.dndv * isect.dvdy;

            // The BSDF stores the IOR of the interior of the object being
            // intersected.  Compute the relative IOR by first out by
            // assuming that the ray is entering the object.
            Float eta = 1 / bsdf.eta;
            if (Dot(wo, ns) < 0) {
                // If the ray isn't entering, then we need to invert the
                // relative IOR and negate the normal and its derivatives.
                eta = 1 / eta;
                ns = -ns;
                dndx = -dndx;
                dndy = -dndy;
            }

            /*
              Notes on the derivation:
              - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [ \eta (\wo \cdot \N) - \cos \theta_t ] \N
                It flips the normal to lie in the same hemisphere as \wo, and then \eta is the relative IOR from
                \wo's medium to \wi's medium.
              - If we denote the term in brackets by \mu, then we have: \wi = -\eta \omega_o + \mu \N
              - Now let's take the partial derivative. (We'll use "d" for \partial in the following for brevity.)
                We get: -\eta d\omega_o / dx + \mu dN/dx + d\mu/dx N.
              - We have the values of all of these except for d\mu/dx (using bits from the derivation of specularly
                reflected ray deifferentials).
              - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We already have d(\wo \cdot N)/dx.
              - The second term takes a little more work. We have:
                 \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
                 Starting from (\wo \cdot N)^2 and reading outward, we have \cos^2 \theta_o, then \sin^2 \theta_o,
                 then \sin^2 \theta_i (via Snell's law), then \cos^2 \theta_i and then \cos \theta_i.
              - Let's take the partial derivative of the sqrt expression. We get:
                1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot N)^2)).
              - That partial derivatve is equal to:
                d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo \cdot N).
              - Plugging it in, we have d\mu/dx =
                \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot N))/(-\wi \cdot N).
             */
            Vector3f dwodx = -ray.rxDirection - wo,
                     dwody = -ray.ryDirection - wo;
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);

            Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
            Float dmudx =
                (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdx;
            Float dmudy =
                (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdy;

            rd.rxDirection =
                wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
            rd.ryDirection =
                wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
        }
        L = f * Li(rd, scene, sampler, arena, depth + 1) * AbsDot(wi, ns) / pdf;
    }
    return L;
}

}  // namespace pbrt
