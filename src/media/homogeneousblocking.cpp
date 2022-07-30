
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


// media/homogeneousblocking.cpp*
#include "media/homogeneousblocking.h"
#include "sampler.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"
#include "integrator.h"
#include "lightdistrib.h"
#include "scene.h"

namespace pbrt {

// HomogeneousBlockingMedium Method Definitions
Spectrum HomogeneousBlockingMedium::Tr(const Ray &ray, Sampler &sampler) const {
    ProfilePhase _(Prof::MediumTr);
    return Exp(-sigma_t * std::min(ray.tMax * ray.d.Length(), MaxFloat));
}

Spectrum HomogeneousBlockingMedium::Sample(const Ray &ray, Sampler &sampler,
                                   MemoryArena &arena,
                                   MediumInteraction *mi) const {
    return SampleTransmittance(ray, sampler, arena, mi);
}

Spectrum HomogeneousBlockingMedium::SampleLightDriven(const Scene &scene,
                    const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    const Shape *shape, const Light &light, Point2f &uLight,
                    const std::shared_ptr<DitherMask> &ditherMask) const {
    if (sampleStrategy == VolSampleStrategy::RISReservoir) {
        return SampleRISReservoirLightDriven(scene, ray, sampler, arena, shape,
                                                light, uLight, ditherMask);
    } else if (sampleStrategy == VolSampleStrategy::RISBidirectionalCDF) {
        Spectrum LNRIS(0.f);
        Point2i currentPixel = sampler.GetCurrentPixel(); 

        // Blue-noise offset for candidates
        Float offset = ditherMask->Value(currentPixel, 0);

        // Blue-noise offset for canonical input randoms
        Float u = ditherMask->Value(currentPixel, 1);
        for (int i = 0; i < N; ++i) {
            // Sample 1 out of M / N
            LNRIS += SampleRISBidirectionalLightDriven(
                                    scene, ray, sampler, arena, shape,
                                    light, uLight, offset,
                                    std::fmod(RadicalInverse(0, i) + u, 1.0),
                                    ditherMask, i);
        }
        return LNRIS / Float(N);
    } else if (sampleStrategy == VolSampleStrategy::RISInverseCDF) {
        return SampleRISiCDFLightDriven(scene, ray, sampler, arena, shape,
                                            light, uLight, ditherMask);
    } else { // Default
        return SampleRISReservoirLightDriven(scene, ray, sampler, arena, shape,
                                            light, uLight, ditherMask);
    }
}

Spectrum HomogeneousBlockingMedium::SampleDistDir(const Scene &scene,
                    const Ray &ray, Sampler &sampler, MemoryArena &arena,
                    const Distribution1D *lightDistrib,
                    const Shape *shape,
                    const std::shared_ptr<DitherMask> &ditherMask) const {
    if (sampleStrategy == VolSampleStrategy::RISReservoir) {
        return SampleRISReservoirDistDir(scene, ray, sampler, arena,
                                        lightDistrib, shape, ditherMask);
    } else {
        Spectrum LNRIS(0.f);
        Point2i currentPixel = sampler.GetCurrentPixel(); 

        // Blue-noise offset for candidates
        Float offset = ditherMask->Value(currentPixel, 0);
        
        // Blue-noise offset for canonical input randoms
        Float u = ditherMask->Value(currentPixel, 1);
        for (int i = 0; i < N; ++i) {
            // Sample 1 out of M / N
            LNRIS += SampleRISBidirectionalDistDir(
                                    scene, ray, sampler, arena,
                                    lightDistrib, shape, offset,
                                    std::fmod(RadicalInverse(0, i) + u, 1.0),
                                    ditherMask, i);
        }
        return LNRIS / Float(N);
    }
}

Spectrum HomogeneousBlockingMedium::SampleTransmittance(const Ray &ray, 
                                        Sampler &sampler, MemoryArena &arena,
                                        MediumInteraction *mi) const {
    ProfilePhase _(Prof::MediumSample);
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();
    // Sample a channel and distance along the ray
    int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);
    Float dist = -std::log(1 - sampler.Get1D() * (1 - std::exp(-sigma_t[channel] * dMax)))
                            / sigma_t[channel];
    Float t = dist / ray.d.Length();
    *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                            ARENA_ALLOC(arena, HenyeyGreenstein)(g));

    // Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = (sigma_t * Tr) / (Spectrum(1) - Exp(-sigma_t * dMax));
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return Tr * sigma_s / pdf;
}

Spectrum HomogeneousBlockingMedium::SampleRISReservoirLightDriven(
                        const Scene &scene, const Ray &ray, Sampler &sampler,
                        MemoryArena &arena, const Shape *shape,
                        const Light &light, Point2f &uLight,
                        const std::shared_ptr<DitherMask> &ditherMask) const {
    ProfilePhase _(Prof::MediumSample);
    Point2i currentPixel = sampler.GetCurrentPixel(); 
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();

    // We focus on sampling the distance only 
    //  and use a single point light in our experiment
    //  else, we need stratification over both distance and lights
    //  the following will not provide that
    if(!IsDeltaLight(light.flags))
        uLight = sampler.Get2D();
    else
        uLight = Point2f(); // No random decision

    // Reservoir to store candidate (distance) and it's target pdf
    Reservoir<std::pair<Float, Float>> r(N);

    // Blue-noise offset for candidates
    Float offset = ditherMask->Value(currentPixel, 0);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 1);
    std::vector<Float> u(N);
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    // Generate candidates
    for (int i = 0; i < M; ++i) {
        int channel = 0; // Assume equal channels, no random decision

        // Stratified candidates offset by dither mask
        Float distSample = Float(i) / M + offset / M;

        // Sample distance distributed according to transmittance
        distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
        Float dist = -std::log(1 - distSample *
                                (1 - std::exp(-sigma_t[channel] * dMax)))
                                / sigma_t[channel];
        Float t = dist / ray.d.Length();
        MediumInteraction mi(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));
        // Target pdf
        Float targetPdf = TargetFunctionVolume(ray, t, mi, sigma_t,
                                                 light, uLight, sampler, shape);
        // Candidate pdf
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

        // Return weighting factor for scattering from homogeneous medium
        Spectrum density = (sigma_t * Tr) /
                                (Spectrum(1) - Exp(-sigma_t * dMax));

        // Source pdf
        Float pdf = 0;
        for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
            pdf *= 1 / (Float)Spectrum::nSamples;
        if (pdf == 0) {
            CHECK(Tr.IsBlack());
            pdf = 1;
        }

        // Update reservoir
        r.update(std::pair<Float, Float>(t, targetPdf), targetPdf / pdf, u);
    }
    if (!r.wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    for (int i = 0; i < N; ++i) {
        Float t = r.y[i].first;
        MediumInteraction mi(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());
        Float pdfMedium = r.y[i].second * (M / r.wsum);

        // Evaluate integrand for the sample inside the reservoir
        L += (Tr * sigma_s / pdfMedium) * EstimateDirect(mi, Point2f(), light,
                                                        uLight, scene, sampler,
                                                        arena, true, false);
    }

    // Average N samples
    return L / Float(N);
}

Spectrum HomogeneousBlockingMedium::SampleRISiCDFLightDriven(
                        const Scene &scene, const Ray &ray, Sampler &sampler,
                        MemoryArena &arena, const Shape *shape,
                        const Light &light, Point2f &uLight,
                        const std::shared_ptr<DitherMask> &ditherMask) const {
    ProfilePhase _(Prof::MediumSample);
    Point2i currentPixel = sampler.GetCurrentPixel(); 
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();

    // CDF variables
    std::vector<std::pair<Float, Float>> candidates(M);
    std::vector<Float> cdf(M);

    if(!IsDeltaLight(light.flags))
        uLight = sampler.Get2D();
    else
        uLight = Point2f(); // No random decision

    // Blue-noise offset for candidates
    Float offset = ditherMask->Value(currentPixel, 0);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 1);
    std::vector<Float> u(N);
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    // Generate candidates
    for (int i = 0; i < M; ++i) {
        int channel = 0; // Assume equal channels, no random decision
        
        // Stratified candidates offset by dither mask
        Float distSample = Float(i) / M + offset / M;
 
        // Sample distance distributed according to transmittance
        distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
        Float dist = -std::log(1 - distSample *
                                (1 - std::exp(-sigma_t[channel] * dMax)))
                                / sigma_t[channel];
        Float t = dist / ray.d.Length();
        MediumInteraction mi(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));

        // Target pdf
        Float targetPdf = TargetFunctionVolume(ray, t, mi, sigma_t,
                                                light, uLight, sampler, shape);

        // Candidate pdf
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

        // Return weighting factor for scattering from homogeneous medium
        Spectrum density = (sigma_t * Tr) / (Spectrum(1) - Exp(-sigma_t * dMax));

        // Source pdf
        Float pdf = 0;
        for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
            pdf *= 1 / (Float)Spectrum::nSamples;
        if (pdf == 0) {
            CHECK(Tr.IsBlack());
            pdf = 1;
        }

        // Store the candidates and build the cdf
        candidates[i] = std::pair<Float, Float>(t, targetPdf);
        cdf[i] = (i > 0) ? cdf[i - 1] + targetPdf / pdf : targetPdf / pdf;
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

        Float t = candidates[sampleIndex].first;
        MediumInteraction mi(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());
        Float pdfMedium = candidates[sampleIndex].second * (M / wsum);

        // Evaluate the integrand for the final samples
        L += (Tr * sigma_s / pdfMedium) * EstimateDirect(mi, Point2f(), light,
                                                        uLight, scene, sampler,
                                                        arena, true, false);
    }

    // Average N samples
    return L / Float(N);
}

Spectrum HomogeneousBlockingMedium::SampleRISBidirectionalLightDriven(
                                const Scene &scene, const Ray &ray,
                                Sampler &sampler, MemoryArena &arena,
                                const Shape *shape, const Light &light,
                                Point2f &uLight, Float offset, Float u,
                                const std::shared_ptr<DitherMask> &ditherMask,
                                int n) const {
    ProfilePhase _(Prof::MediumSample);
    Point2i currentPixel = sampler.GetCurrentPixel(); 
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();
    MediumInteraction mi;
    Float selectedTargetPdf;
    Spectrum selectedTr;
    int channel = 0; // Assume equal channels, no random decision

    if(!IsDeltaLight(light.flags))
        uLight = sampler.Get2D();
    else
        uLight = Point2f(); // No random decision

    // Bidirectional CDF sampling variables
    int indexFront(0);
    int indexBack(M / N - 1);
    // Calculate weights of front and back
    Float wsumFront(0.f);
    Float wsumBack(0.f);

    {   // Front
        // Stratified candidates offset by dither mask
        // Candidate order is interleaved
        Float distSample = Float(indexFront * N + n) / M + offset / M;

        // Sample distance distributed according to transmittance
        distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
        Float dist = -std::log(1 - distSample *
                                (1 - std::exp(-sigma_t[channel] * dMax)))
                                / sigma_t[channel];
        Float t = dist / ray.d.Length();
        MediumInteraction candidatemi(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));
        // Target pdf
        Float targetPdf = TargetFunctionVolume(ray, t, candidatemi, sigma_t,
                                                light, uLight, sampler, shape);
        // Candidate pdf
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

        // Return weighting factor for scattering from homogeneous medium
        Spectrum density = (sigma_t * Tr) /
                                (Spectrum(1) - Exp(-sigma_t * dMax));

        // Source pdf
        Float pdf = 0;
        for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
            pdf *= 1 / (Float)Spectrum::nSamples;
        if (pdf == 0) {
            CHECK(Tr.IsBlack());
            pdf = 1;
        }   

        // Update front weight
        wsumFront += targetPdf / pdf;

        selectedTargetPdf = targetPdf;
        selectedTr = Tr;
        mi = candidatemi;
    } { // Back
        // Stratified candidates offset by dither mask
        Float distSample = Float(indexBack * N + n) / M + offset / M;

        // Sample distance distributed according to transmittance
        distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
        Float dist = -std::log(1 - distSample *
                                (1 - std::exp(-sigma_t[channel] * dMax)))
                                / sigma_t[channel];
        Float t = dist / ray.d.Length();
        MediumInteraction candidatemi(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));
        // Target pdf
        Float targetPdf = TargetFunctionVolume(ray, t, candidatemi, sigma_t,
                                                light, uLight, sampler, shape);
        // Candidate pdf
        // Compute the transmittance and sampling density
        Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

        // Return weighting factor for scattering from homogeneous medium
        Spectrum density = (sigma_t * Tr) /
                                (Spectrum(1) - Exp(-sigma_t * dMax));

        // Source pdf
        Float pdf = 0;
        for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
            pdf *= 1 / (Float)Spectrum::nSamples;
        if (pdf == 0) {
            CHECK(Tr.IsBlack());
            pdf = 1;
        }   

        // Update back weight, avoid adding the same weight twice
        wsumBack += (indexFront != indexBack) ? targetPdf / pdf : 0.f;
    }
    // Bidirectional CDF
    while (indexFront != indexBack) {
        if (wsumFront < u * (wsumFront + wsumBack)) {
            // Advance front index
            indexFront++;

            // Stratified candidates offset by dither mask
            Float distSample = Float(indexFront * N + n) / M + offset / M;

            // Sample distance distributed according to transmittance
            distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
            Float dist = -std::log(1 - distSample *
                                    (1 - std::exp(-sigma_t[channel] * dMax)))
                                    / sigma_t[channel];
            Float t = dist / ray.d.Length();
            MediumInteraction candidatemi(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));

            // Target pdf
            Float targetPdf = TargetFunctionVolume(ray, t, candidatemi, sigma_t,
                                                light, uLight, sampler, shape);
            // Candidate pdf
            // Compute the transmittance and sampling density
            Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) *
                            ray.d.Length());

            // Return weighting factor for scattering from homogeneous medium
            Spectrum density = (sigma_t * Tr) /
                                (Spectrum(1) - Exp(-sigma_t * dMax));

            // Source pdf
            Float pdf = 0;
            for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
                pdf *= 1 / (Float)Spectrum::nSamples;
            if (pdf == 0) {
                CHECK(Tr.IsBlack());
                pdf = 1;
            }   

            // Update front weight, avoid adding the same weight twice
            wsumFront += (indexFront != indexBack) ? targetPdf / pdf : 0.f;

            selectedTargetPdf = targetPdf;
            selectedTr = Tr;
            mi = candidatemi;
        } else {
            // Decrease back index
            indexBack--;

            // Stratified candidates offset by dither mask
            Float distSample = Float(indexBack * N + n) / M + offset / M;
            
            // Sample distance distributed according to transmittance
            distSample = (distSample != 1.0) ? distSample : (1.0 - 1e-6);
            Float dist = -std::log(1 - distSample *
                                    (1 - std::exp(-sigma_t[channel] * dMax)))
                                    / sigma_t[channel];
            Float t = dist / ray.d.Length();
            MediumInteraction candidatemi(ray(t), -ray.d, ray.time, this,
                                    ARENA_ALLOC(arena, HenyeyGreenstein)(g));

            // Target pdf
            Float targetPdf = TargetFunctionVolume(ray, t, candidatemi, sigma_t,
                                                light, uLight, sampler, shape);

            // Candidate pdf
            // Compute the transmittance and sampling density
            Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) *
                            ray.d.Length());

            // Return weighting factor for scattering from homogeneous medium
            Spectrum density = (sigma_t * Tr) /
                                (Spectrum(1) - Exp(-sigma_t * dMax));

            // Source pdf
            Float pdf = 0;
            for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
                pdf *= 1 / (Float)Spectrum::nSamples;
            if (pdf == 0) {
                CHECK(Tr.IsBlack());
                pdf = 1;
            }   

            // Update back weight, avoid adding the same weight twice
            wsumBack += (indexFront != indexBack) ? targetPdf / pdf : 0.f;
        }
    }
    Float wsum = wsumFront + wsumBack;
    if (!wsum) // No contribution
        return Spectrum(0);

    Float pdfMedium = selectedTargetPdf * (M / (N * wsum));

    // Evaluate the integrand for the selected sample
    return (selectedTr * sigma_s / pdfMedium) *
                EstimateDirect(mi, Point2f(), light, uLight, scene,
                                sampler, arena, true, false);
}

Spectrum HomogeneousBlockingMedium::SampleRISReservoirDistDir(
                    const Scene &scene, const Ray &ray, Sampler &sampler,
                    MemoryArena &arena, const Distribution1D *lightDistrib,
                    const Shape *shape,
                    const std::shared_ptr<DitherMask> &ditherMask) const {
    ProfilePhase _(Prof::MediumSample);
    int nLights = int(scene.lights.size());
    Point2i currentPixel = sampler.GetCurrentPixel();
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();

    // Reservoir to store candidates (distance, direction)
    Reservoir<DistDirSample> r(N);

    // Blue-noise offset for candidates
    Float offsetx = ditherMask->Value(currentPixel, 0);
    Float offsety = ditherMask->Value(currentPixel, 1);
    Float offsetz = ditherMask->Value(currentPixel, 2);

    // Stratified blue-noise offsetting for regular i / N samples
    Float dlprime = ditherMask->Value(currentPixel, 3);
    std::vector<Float> u(N);
    for (int i = 0; i < N; ++i) {
        u[i] = Float(i) / N + dlprime / N;
    }

    // Generate candidates
    for (int i = 0; i < M; ++i) {
        // Halton candidates offset by dither mask 
        std::vector<Float> sample3d(3); // 3D candidate
        sample3d[0] = std::fmod(RadicalInverse(0, i) + offsetx, 1.0);
        sample3d[1] = std::fmod(RadicalInverse(1, i) + offsety, 1.0);
        sample3d[2] = std::fmod(RadicalInverse(2, i) + offsetz, 1.0);

        // Sample the distance and direction
        Float w;
        DistDirSample candidate = GetDistDirSample(sample3d, &w, scene, ray,
                                                  arena, lightDistrib,
                                                  dMax, nLights);

        // Reservoir update
        r.update(candidate, w, u);
    }
    if (!r.wsum) // No contribution
        return Spectrum(0);

    Spectrum L(0.f);
    for (int i = 0; i < N; ++i) {
        // Evaluate visibility for the selected sample
        if(r.y[i].visibility.Unoccluded(scene)) {
            Float pdf = r.y[i].targetPdf * (Float(r.M) / r.wsum);
            // Evaluate the integrand for the sample inside the reservoir
            L += (pdf > 0) ?
                (sigma_s * r.y[i].f * r.y[i].Li * r.y[i].Tr * r.y[i].G) / pdf
                : Spectrum(0);
        }
    } 

    // Average N samples
    return L / Float(N);
}

Spectrum HomogeneousBlockingMedium::SampleRISBidirectionalDistDir(
                    const Scene &scene, const Ray &ray, Sampler &sampler,
                    MemoryArena &arena, const Distribution1D *lightDistrib,
                    const Shape *shape, Float offset, Float u,
                    const std::shared_ptr<DitherMask> &ditherMask,
                    int n) const {
    ProfilePhase _(Prof::MediumSample);
    int nLights = int(scene.lights.size());
    Point2i currentPixel = sampler.GetCurrentPixel();
    int currentSampleIndex = sampler.CurrentSampleNumber();
    Float tMax = std::min(ray.tMax, MaxFloat);
    Float dMax = tMax * ray.d.Length();
    DistDirSample selectedSample;

    // Bidirectional CDF sampling variables
    int indexFront(0);
    int indexBack(M / N - 1);
    // Calculate weights of front and back
    Float wsumFront(0.f);
    Float wsumBack(0.f);

    HilbertCurve hc(21, 3); // Ideally should be 32 x 3
    {   // Front
        // Hilbert Curve candidates offset by dither mask
        std::vector<Float> sample3d(3);
        Float sample1d = Float(indexFront * N + n) / M + offset / M;
        sample1d = (sample1d != 1.0) ? sample1d : (1.0 - 1e-6);
        // Warp through Hilbert Curve to obtain 3D candidates
        hc.sample(sample1d, sample3d);

        // Sample the distance and direction
        Float w;
        selectedSample = GetDistDirSample(sample3d, &w, scene, ray,
                                                  arena, lightDistrib,
                                                  dMax, nLights);

        // Update front weight
        wsumFront += w;
    } { // Back
        // Hilbert Curve candidates offset by dither mask
        std::vector<Float> sample3d(3);
        Float sample1d = Float(indexBack * N + n) / M + offset / M;
        sample1d = (sample1d != 1.0) ? sample1d : (1.0 - 1e-6);
        // Warp through Hilbert Curve to obtain 3D candidates
        hc.sample(sample1d, sample3d);

        // Sample the distance and direction
        Float w;
        GetDistDirSample(sample3d, &w, scene, ray, arena, lightDistrib,
                              dMax, nLights);
        
        // Update back weight, avoid adding the same weight twice
        wsumBack += (indexFront != indexBack) ? w : 0.f;
    }
    // Bidirectional CDF
    while (indexFront != indexBack) {
        if (wsumFront <= u * (wsumFront + wsumBack)) {
            // Advance front index
            indexFront++;
            // Hilbert Curve candidates offset by dither mask
            std::vector<Float> sample3d(3);
            Float sample1d = Float(indexFront * N + n) / M + offset / M;
            sample1d = (sample1d != 1.0) ? sample1d : (1.0 - 1e-6);
            // Warp through Hilbert Curve to obtain 3D candidates
            hc.sample(sample1d, sample3d);

            // Sample the distance and direction
            Float w;
            selectedSample = GetDistDirSample(sample3d, &w, scene, ray,
                                                      arena, lightDistrib,
                                                      dMax, nLights);

            // Update front weight, avoid adding the same weight twice
            wsumFront += (indexFront != indexBack) ? w : 0.f;
        } else {
            // Decrease back index
            indexBack--;
            // Hilbert Curve candidates offset by dither mask
            std::vector<Float> sample3d(3);
            Float sample1d = Float(indexBack * N + n) / M + offset / M;
            sample1d = (sample1d != 1.0) ? sample1d : (1.0 - 1e-6);
            // Warp through Hilbert Curve to obtain 3D candidates
            hc.sample(sample1d, sample3d);

            // Sample the distance and direction
            Float w;
            GetDistDirSample(sample3d, &w, scene, ray, arena, lightDistrib,
                                  dMax, nLights);

            // Update back weight, avoid adding the same weight twice
            wsumBack += (indexFront != indexBack) ? w : 0.f;
        }
    }
    Float wsum = wsumFront + wsumBack;
    if (!wsum) // No contribution
        return Spectrum(0);

    // Evaluate visibility for the selected sample
    if(selectedSample.visibility.Unoccluded(scene)) {
        // Evaluate the integrand for the selected sample
        Float pdf = selectedSample.targetPdf * ((M/N) / wsum);
        return (pdf > 0) ? (sigma_s * selectedSample.f
                                * selectedSample.Li
                                * selectedSample.Tr
                                * selectedSample.G) / pdf
                            : Spectrum(0);
    }

    return Spectrum(0);
}

DistDirSample HomogeneousBlockingMedium::GetDistDirSample(
                                    const std::vector<Float> &sample3d,
                                    Float *w, const Scene &scene,
                                    const Ray &ray, MemoryArena &arena,
                                    const Distribution1D *lightDistrib,
                                    Float dMax, int nLights) const {
    Float lightPdf, lightChoicePdf, targetPdf, sourcePdf;
    // Sample distance
    Float t = DistanceSampling(sample3d[0], ray, sigma_t, dMax);
    MediumInteraction mi(ray(t), -ray.d, ray.time, this,
                            ARENA_ALLOC(arena, HenyeyGreenstein)(g));
    // Distance Candidate pdf
    // Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());
    sourcePdf = DistancePdf(Tr, sigma_t, dMax);

    // Sample light
    Vector3f wi;
    VisibilityTester visibility;
    Interaction pLight;
    std::shared_ptr<Light> light;
    Spectrum Li = AreaSampleOneLight(Point2f(sample3d[1], sample3d[2]), scene,
                                     nLights, light, mi, &lightPdf,
                                     lightChoicePdf, &wi, &pLight,
                                     &visibility, lightDistrib);

    // Source Pdf = distance pdf * light pdf
    sourcePdf *= lightPdf * lightChoicePdf;

    // Target Pdf = unshadowed contribution
    Spectrum f;
    Float G;
    targetPdf = UnshadowedContributionVolume(&f, &G, mi, wi, sigma_t, light,
                                             &Tr, pLight, sourcePdf, Li);

    // w = targetPdf / sourcePdf
    *w = (sourcePdf > 0) ? targetPdf / sourcePdf : 0.0;

    // Candidate
    return DistDirSample(t, wi, targetPdf, visibility, f, Li, Tr, G);
}

}  // namespace pbrt
