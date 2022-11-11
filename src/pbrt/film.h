// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_FILM_H
#define PBRT_FILM_H

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/bsdf.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

namespace pbrt {

// PixelSensor Definition
class PixelSensor {
  public:
    // PixelSensor Public Methods
    static PixelSensor *Create(const ParameterDictionary &parameters,
                               const RGBColorSpace *colorSpace, Float exposureTime,
                               const FileLoc *loc, Allocator alloc);

    static PixelSensor *CreateDefault(Allocator alloc = {});

    PixelSensor(Spectrum r, Spectrum g, Spectrum b, const RGBColorSpace *outputColorSpace,
                Spectrum sensorIllum, Float imagingRatio, Allocator alloc)
        : r_bar(r, alloc), g_bar(g, alloc), b_bar(b, alloc), imagingRatio(imagingRatio) {
        // Compute XYZ from camera RGB matrix
        // Compute _rgbCamera_ values for training swatches
        Float rgbCamera[nSwatchReflectances][3];
        for (int i = 0; i < nSwatchReflectances; ++i) {
            RGB rgb = ProjectReflectance<RGB>(swatchReflectances[i], sensorIllum, &r_bar,
                                              &g_bar, &b_bar);
            for (int c = 0; c < 3; ++c)
                rgbCamera[i][c] = rgb[c];
        }

        // Compute _xyzOutput_ values for training swatches
        Float xyzOutput[24][3];
        Float sensorWhiteG = InnerProduct(sensorIllum, &g_bar);
        Float sensorWhiteY = InnerProduct(sensorIllum, &Spectra::Y());
        for (size_t i = 0; i < nSwatchReflectances; ++i) {
            Spectrum s = swatchReflectances[i];
            XYZ xyz =
                ProjectReflectance<XYZ>(s, &outputColorSpace->illuminant, &Spectra::X(),
                                        &Spectra::Y(), &Spectra::Z()) *
                (sensorWhiteY / sensorWhiteG);
            for (int c = 0; c < 3; ++c)
                xyzOutput[i][c] = xyz[c];
        }

        // Initialize _XYZFromSensorRGB_ using linear least squares
        pstd::optional<SquareMatrix<3>> m =
            LinearLeastSquares(rgbCamera, xyzOutput, nSwatchReflectances);
        if (!m)
            ErrorExit("Sensor XYZ from RGB matrix could not be solved.");
        XYZFromSensorRGB = *m;
    }

    PixelSensor(const RGBColorSpace *outputColorSpace, Spectrum sensorIllum,
                Float imagingRatio, Allocator alloc)
        : r_bar(&Spectra::X(), alloc),
          g_bar(&Spectra::Y(), alloc),
          b_bar(&Spectra::Z(), alloc),
          imagingRatio(imagingRatio) {
        // Compute white balancing matrix for XYZ _PixelSensor_
        if (sensorIllum) {
            Point2f sourceWhite = SpectrumToXYZ(sensorIllum).xy();
            Point2f targetWhite = outputColorSpace->w;
            XYZFromSensorRGB = WhiteBalance(sourceWhite, targetWhite);
        }
    }

    PBRT_CPU_GPU
    RGB ToSensorRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
        L = SafeDiv(L, lambda.PDF());
        return imagingRatio * RGB((r_bar.Sample(lambda) * L).Average(),
                                  (g_bar.Sample(lambda) * L).Average(),
                                  (b_bar.Sample(lambda) * L).Average());
    }

    // PixelSensor Public Members
    SquareMatrix<3> XYZFromSensorRGB;

  private:
    // PixelSensor Private Methods
    template <typename Triplet>
    static Triplet ProjectReflectance(Spectrum r, Spectrum illum, Spectrum b1,
                                      Spectrum b2, Spectrum b3);

    // PixelSensor Private Members
    DenselySampledSpectrum r_bar, g_bar, b_bar;
    Float imagingRatio;
    static constexpr int nSwatchReflectances = 24;
    static Spectrum swatchReflectances[nSwatchReflectances];
};

// PixelSensor Inline Methods
template <typename Triplet>
inline Triplet PixelSensor::ProjectReflectance(Spectrum refl, Spectrum illum, Spectrum b1,
                                               Spectrum b2, Spectrum b3) {
    Triplet result;
    Float g_integral = 0;
    for (Float lambda = Lambda_min; lambda <= Lambda_max; ++lambda) {
        g_integral += b2(lambda) * illum(lambda);
        result[0] += b1(lambda) * refl(lambda) * illum(lambda);
        result[1] += b2(lambda) * refl(lambda) * illum(lambda);
        result[2] += b3(lambda) * refl(lambda) * illum(lambda);
    }
    return result / g_integral;
}

// VisibleSurface Definition
class VisibleSurface {
  public:
    // VisibleSurface Public Methods
    PBRT_CPU_GPU
    VisibleSurface(const SurfaceInteraction &si, SampledSpectrum albedo,
                   const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    operator bool() const { return set; }

    VisibleSurface() = default;

    std::string ToString() const;

    // VisibleSurface Public Members
    Point3f p;
    Normal3f n, ns;
    Point2f uv;
    Float time = 0;
    Vector3f dpdx, dpdy;
    SampledSpectrum albedo;
    bool set = false;
};

// FilmBaseParameters Definition
struct FilmBaseParameters {
    FilmBaseParameters(const ParameterDictionary &parameters, Filter filter,
                       const PixelSensor *sensor, const FileLoc *loc);
    FilmBaseParameters(Point2i fullResolution, Bounds2i pixelBounds, Filter filter,
                       Float diagonal, const PixelSensor *sensor, std::string filename)
        : fullResolution(fullResolution),
          pixelBounds(pixelBounds),
          filter(filter),
          diagonal(diagonal),
          sensor(sensor),
          filename(filename) {}

    Point2i fullResolution;
    Bounds2i pixelBounds;
    Filter filter;
    Float diagonal;
    const PixelSensor *sensor;
    std::string filename;
};

// FilmBase Definition
class FilmBase {
  public:
    // FilmBase Public Methods
    FilmBase(FilmBaseParameters p)
        : fullResolution(p.fullResolution),
          pixelBounds(p.pixelBounds),
          filter(p.filter),
          diagonal(p.diagonal * .001f),
          sensor(p.sensor),
          filename(p.filename) {
        CHECK(!pixelBounds.IsEmpty());
        CHECK_GE(pixelBounds.pMin.x, 0);
        CHECK_LE(pixelBounds.pMax.x, fullResolution.x);
        CHECK_GE(pixelBounds.pMin.y, 0);
        CHECK_LE(pixelBounds.pMax.y, fullResolution.y);
        LOG_VERBOSE("Created film with full resolution %s, pixelBounds %s",
                    fullResolution, pixelBounds);
    }

    PBRT_CPU_GPU
    Point2i FullResolution() const { return fullResolution; }
    PBRT_CPU_GPU
    Bounds2i PixelBounds() const { return pixelBounds; }
    PBRT_CPU_GPU
    Float Diagonal() const { return diagonal; }
    PBRT_CPU_GPU
    Filter GetFilter() const { return filter; }
    PBRT_CPU_GPU
    const PixelSensor *GetPixelSensor() const { return sensor; }
    std::string GetFilename() const { return filename; }

    PBRT_CPU_GPU
    SampledWavelengths SampleWavelengths(Float u) const {
        return SampledWavelengths::SampleVisible(u);
    }

    // TODO [MIS]: returns \alpha_k
    PBRT_CPU_GPU
    Float GetMISAlpha(const Point2i p) const;

    // TODO [MIS]: first version with 2 sampling method
    PBRT_CPU_GPU
    void UpdateProbsMIS(const Point2i p, SampledSpectrum L, const SampledWavelengths &lambda, Float fpdf, Float gpdf);

    PBRT_CPU_GPU
    void UpdateNSamplesMIS(Point2i p);
     
    PBRT_CPU_GPU
    void ComputeUpdatedAlpha(Point2i p);

    PBRT_CPU_GPU
    Bounds2f SampleBounds() const;

    std::string BaseToString() const;

  protected:
    // FilmBase Protected Members
    Point2i fullResolution;
    Bounds2i pixelBounds;
    Filter filter;
    Float diagonal;
    const PixelSensor *sensor;
    std::string filename;
};

// RGBFilm Definition
class RGBFilm : public FilmBase {
  public:
    // RGBFilm Public Methods
    PBRT_CPU_GPU
    bool UsesVisibleSurface() const { return false; }

    PBRT_CPU_GPU
    void AddSample(Point2i pFilm, SampledSpectrum L, const SampledWavelengths &lambda,
                   const VisibleSurface *, Float weight) {
        // Convert sample radiance to _PixelSensor_ RGB
        RGB rgb = sensor->ToSensorRGB(L, lambda);

        // Optionally clamp sensor RGB value
        Float m = std::max({rgb.r, rgb.g, rgb.b});
        if (m > maxComponentValue)
            rgb *= maxComponentValue / m;

        DCHECK(InsideExclusive(pFilm, pixelBounds));
        // Update pixel values with filtered sample contribution
        Pixel &pixel = pixels[pFilm];
        for (int c = 0; c < 3; ++c)
            pixel.rgbSum[c] += weight * rgb[c];
        pixel.weightSum += weight;
    }

    PBRT_CPU_GPU
    RGB GetPixelRGB(Point2i p, Float splatScale = 1) const {
        const Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);
        // Normalize _rgb_ with weight sum
        Float weightSum = pixel.weightSum;
        if (weightSum != 0)
            rgb /= weightSum;

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.rgbSplat[c] / filterIntegral;

        // Convert _rgb_ to output RGB color space
        rgb = outputRGBFromSensorRGB * rgb;

        return rgb;
    }

    // [MIS Divergence]: returns \alpha_k
    PBRT_CPU_GPU
    Float GetMISAlpha(const Point2i p) const {

        if (Options->alphaFixed)
            return Options->alphaMIS;
 
        const Pixel &pixel = pixels[p];
       
        // [MIS] 5 samples for first method / 5 samples for second method
        if (pixel.nsamples < (int)(Pixel::samplesBatch / 2)) {
            return 0.99;
        } else if (pixel.nsamples < Pixel::samplesBatch) {
            return 0.01;
        }

        // after we always get MIS alpha
        return pixel.alphaMIS;
    }

    PBRT_CPU_GPU
    void UpdateNSamplesMIS(Point2i p) {

        if (Options->alphaFixed)
            return;
        
        Pixel &pixel = pixels[p];

        // unused now
        pixel.nsamples += 1;
    }

    // [MIS Divergence]: first version with 2 sampling method
    PBRT_CPU_GPU
    void UpdateProbsMIS(const Point2i p, SampledSpectrum L, const SampledWavelengths &lambda, Float fpdf, Float gpdf) {
        
        Pixel &pixel = pixels[p];

        // if (p.x == 100 && p.y == 100)
        //     std::cout << "Data for pixel " << p << " at samples " << pixel.nsamples << std::endl;
        // else
        //     return;

        // std::cout << "bsdfPDF: " << fpdf << std::endl;
        // std::cout << "lightPDF: " << gpdf << std::endl;

        RGB rgb = sensor->ToSensorRGB(L, lambda);
        Float luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        Float luminanceTsallis = std::pow(luminance, pixel.gammaTsallis);

        // std::cout << "Luminance: " <<  luminance << std::endl;
        // std::cout << "LuminanceT: " <<  luminanceTsallis << std::endl;

        // [MIS]: update xi and xi' with respect to equations 30-31
        Float alphaProbs = pixel.alphaMIS * fpdf + (1 - pixel.alphaMIS) * gpdf;

        // std::cout << "p(\alpha, X_i): " << alphaProbs << std::endl;

        // check not null value
        if (alphaProbs <= 0)
            alphaProbs += std::numeric_limits<Float>::epsilon();

        pixel.xiSum += (luminanceTsallis / std::pow(alphaProbs, pixel.gammaTsallis + 1)) 
                        * (fpdf - gpdf);

        // std::cout << "xi term: " << (luminanceTsallis / std::pow(alphaProbs, pixel.gammaTsallis + 1)) 
                        // * (fpdf - gpdf) << std::endl;
        pixel.xiPrimeSum += (luminanceTsallis / std::pow(alphaProbs, pixel.gammaTsallis + 2)) 
                        * (std::pow(fpdf - gpdf, 2));

        // std::cout << "xi' term: " << (luminanceTsallis / std::pow(alphaProbs, pixel.gammaTsallis + 2)) 
                        // * ((fpdf - gpdf) * (fpdf - gpdf)) << std::endl;
        // }

        // std::cout << "[" << pixel.nsamples << "] xi: " << pixel.xiSum << ", xi': " << pixel.xiPrimeSum << std::endl;
    }

    PBRT_CPU_GPU
    void ComputeUpdatedAlpha(Point2i p) {
        
        if (Options->alphaFixed)
            return;
        
        Pixel &pixel = pixels[p];

        if ((pixel.nsamples % Pixel::samplesBatch) != 0)
            return;

        // Need to update UpdateProbsMIS, in order to take into account 
        // luminance without balance heuristic and then update internal data 
        // in order to compute xi_{\alpha}

        Float xiAlpha = pixel.xiSum / pixel.nsamples;

        Float xiPrimeAlpha = pixel.xiPrimeSum * (-pixel.gammaTsallis / pixel.nsamples);

        if (xiPrimeAlpha <= 0)
            xiPrimeAlpha += std::numeric_limits<Float>::epsilon();

        // std::cout << " -- xiAlpha: " << xiAlpha << std::endl;
        // std::cout << " -- xiPrimeAlpha: " << xiPrimeAlpha << std::endl;
        // std::cout << " -- Gradient step: " << (xiAlpha / xiPrimeAlpha) << std::endl;
        // std::cout << " -- new Alpha MIS: " << pixel.alphaMIS - (xiAlpha / xiPrimeAlpha) << std::endl;

        // std::cout << "-------------------------" << std::endl;

        // [MIS]: update xi and xi' with respect to equation 32
        pixel.alphaMIS = pixel.alphaMIS - (xiAlpha / xiPrimeAlpha);
        // std::cout << p << ":: computed alpha: " << pixel.alphaMIS << std::endl;

        // reset for next Pixel::batchSamples (number of generated paths)
        pixel.reset();

        if (pixel.alphaMIS <= 0) 
            pixel.alphaMIS = std::numeric_limits<Float>::epsilon();

        if (pixel.alphaMIS >= 1) 
            pixel.alphaMIS = 1. - std::numeric_limits<Float>::epsilon();
    }

    RGBFilm(FilmBaseParameters p, const RGBColorSpace *colorSpace,
            Float maxComponentValue = Infinity, bool writeFP16 = true,
            Allocator alloc = {});

    static RGBFilm *Create(const ParameterDictionary &parameters, Float exposureTime,
                           Filter filter, const RGBColorSpace *colorSpace,
                           const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    void AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda);

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

    PBRT_CPU_GPU
    RGB ToOutputRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
        RGB sensorRGB = sensor->ToSensorRGB(L, lambda);
        return outputRGBFromSensorRGB * sensorRGB;
    }

    PBRT_CPU_GPU void ResetPixel(Point2i p) { std::memset(&pixels[p], 0, sizeof(Pixel)); }

  private:
    // RGBFilm::Pixel Definition
    struct Pixel {
        static const int nPDFs = 2;
        static const int samplesBatch = 10;

        Pixel() {};

        void reset() {

            xiSum = 0.;
            xiPrimeSum = 0.;
        }

        // By default static number of PDFs
        double alphaMIS = Options->alphaMIS;
        double gammaTsallis = Options->tsallisMIS;
        double xiSum = 0.;
        double xiPrimeSum = 0.;
        int nsamples = 0;

        double rgbSum[3] = {0., 0., 0.};
        double weightSum = 0.;
        AtomicDouble rgbSplat[3];
    };

    // RGBFilm Private Members
    const RGBColorSpace *colorSpace;
    Float maxComponentValue;
    bool writeFP16;
    Float filterIntegral;
    SquareMatrix<3> outputRGBFromSensorRGB;
    Array2D<Pixel> pixels;
};

// GBufferFilm Definition
class GBufferFilm : public FilmBase {
  public:
    // GBufferFilm Public Methods
    GBufferFilm(FilmBaseParameters p, const AnimatedTransform &outputFromRender,
                bool applyInverse, const RGBColorSpace *colorSpace,
                Float maxComponentValue = Infinity, bool writeFP16 = true,
                Allocator alloc = {});

    static GBufferFilm *Create(const ParameterDictionary &parameters, Float exposureTime,
                               const CameraTransform &cameraTransform, Filter filter,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc);

    PBRT_CPU_GPU
    void AddSample(Point2i pFilm, SampledSpectrum L, const SampledWavelengths &lambda,
                   const VisibleSurface *visibleSurface, Float weight);

    PBRT_CPU_GPU
    void AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    RGB ToOutputRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
        RGB cameraRGB = sensor->ToSensorRGB(L, lambda);
        return outputRGBFromSensorRGB * cameraRGB;
    }

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const { return true; }

    PBRT_CPU_GPU
    RGB GetPixelRGB(Point2i p, Float splatScale = 1) const {
        const Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        if (weightSum != 0)
            rgb /= weightSum;

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.rgbSplat[c] / filterIntegral;

        rgb = outputRGBFromSensorRGB * rgb;

        return rgb;
    }

    PBRT_CPU_GPU
    Float GetMISAlpha(const Point2i p) const {
        return 0.5;
    }

    PBRT_CPU_GPU
    void UpdateProbsMIS(const Point2i p, SampledSpectrum L, const SampledWavelengths &lambda, Float fpdf, Float gpdf) {}

    PBRT_CPU_GPU
    void UpdateNSamplesMIS(Point2i p) {}

    PBRT_CPU_GPU
    void ComputeUpdatedAlpha(Point2i p) {}


    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

    PBRT_CPU_GPU void ResetPixel(Point2i p) { std::memset(&pixels[p], 0, sizeof(Pixel)); }

  private:
    // GBufferFilm::Pixel Definition
    struct Pixel {
        Pixel() = default;
        double rgbSum[3] = {0., 0., 0.};
        double weightSum = 0., gBufferWeightSum = 0.;
        AtomicDouble rgbSplat[3];
        Point3f pSum;
        Float dzdxSum = 0, dzdySum = 0;
        Normal3f nSum, nsSum;
        Point2f uvSum;
        double rgbAlbedoSum[3] = {0., 0., 0.};
        VarianceEstimator<Float> rgbVariance[3];
    };

    // GBufferFilm Private Members
    AnimatedTransform outputFromRender;
    bool applyInverse;
    Array2D<Pixel> pixels;
    const RGBColorSpace *colorSpace;
    Float maxComponentValue;
    bool writeFP16;
    Float filterIntegral;
    SquareMatrix<3> outputRGBFromSensorRGB;
};

// SpectralFilm Definition
class SpectralFilm : public FilmBase {
  public:
    // SpectralFilm Public Methods
    PBRT_CPU_GPU
    bool UsesVisibleSurface() const { return false; }

    PBRT_CPU_GPU
    SampledWavelengths SampleWavelengths(Float u) const {
        return SampledWavelengths::SampleUniform(u, lambdaMin, lambdaMax);
    }

    PBRT_CPU_GPU
    void AddSample(Point2i pFilm, SampledSpectrum L, const SampledWavelengths &lambda,
                   const VisibleSurface *, Float weight) {
        // Start by doing more or less what RGBFilm::AddSample() does so
        // that we can maintain accurate RGB values.

        // Convert sample radiance to _PixelSensor_ RGB
        RGB rgb = sensor->ToSensorRGB(L, lambda);

        // Optionally clamp sensor RGB value
        Float m = std::max({rgb.r, rgb.g, rgb.b});
        if (m > maxComponentValue)
            rgb *= maxComponentValue / m;

        DCHECK(InsideExclusive(pFilm, pixelBounds));
        // Update RGB fields in Pixel structure.
        Pixel &pixel = pixels[pFilm];
        for (int c = 0; c < 3; ++c)
            pixel.rgbSum[c] += weight * rgb[c];
        pixel.rgbWeightSum += weight;

        // Spectral processing starts here.
        // Optionally clamp spectral value. (TODO: for spectral should we
        // just clamp channels individually?)
        Float lm = L.MaxComponentValue();
        if (lm > maxComponentValue)
            L *= maxComponentValue / lm;

        // The CIE_Y_integral factor effectively cancels out the effect of
        // the conversion of light sources to use photometric units for
        // specification.  We then do *not* divide by the PDF in |lambda|
        // but take advantage of the fact that we know that it is uniform
        // in SampleWavelengths(), the fact that the buckets all have the
        // same extend, and can then just average radiance in buckets
        // below.
        L *= weight * CIE_Y_integral;

        // Accumulate contributions in spectral buckets.
        for (int i = 0; i < NSpectrumSamples; ++i) {
            int b = LambdaToBucket(lambda[i]);
            pixel.bucketSums[b] += L[i];
            pixel.weightSums[b] += weight;
        }
    }

    PBRT_CPU_GPU
    RGB GetPixelRGB(Point2i p, Float splatScale = 1) const;

    PBRT_CPU_GPU
    Float GetMISAlpha(const Point2i p) const {
        return 0.5;
    }

    PBRT_CPU_GPU
    void UpdateNSamplesMIS(Point2i p) {}
   

    PBRT_CPU_GPU
    void UpdateProbsMIS(const Point2i p, SampledSpectrum L, const SampledWavelengths &lambda, Float fpdf, Float gpdf) {}

    PBRT_CPU_GPU
    void ComputeUpdatedAlpha(Point2i p) {}

    SpectralFilm(FilmBaseParameters p, Float lambdaMin, Float lambdaMax, int nBuckets,
                 const RGBColorSpace *colorSpace, Float maxComponentValue = Infinity,
                 bool writeFP16 = true, Allocator alloc = {});

    static SpectralFilm *Create(const ParameterDictionary &parameters, Float exposureTime,
                                Filter filter, const RGBColorSpace *colorSpace,
                                const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    void AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda);

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);

    // Returns an image with both RGB and spectral components, following
    // the layout proposed in "An OpenEXR Layout for Sepctral Images" by
    // Fichet et al., https://jcgt.org/published/0010/03/01/.
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

    PBRT_CPU_GPU
    RGB ToOutputRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
        LOG_FATAL("ToOutputRGB() is unimplemented. But that's ok since it's only used "
                  "in the SPPM integrator, which is inherently very much based on "
                  "RGB output.");
        return {};
    }

    PBRT_CPU_GPU void ResetPixel(Point2i p) {
        Pixel &pix = pixels[p];
        pix.rgbSum[0] = pix.rgbSum[1] = pix.rgbSum[2] = 0.;
        pix.rgbWeightSum = 0.;
        pix.rgbSplat[0] = pix.rgbSplat[1] = pix.rgbSplat[2] = 0.;
        std::memset(pix.bucketSums, 0, nBuckets * sizeof(double));
        std::memset(pix.weightSums, 0, nBuckets * sizeof(double));
        std::memset(pix.bucketSplats, 0, nBuckets * sizeof(AtomicDouble));
    }

  private:
    PBRT_CPU_GPU
    int LambdaToBucket(Float lambda) const {
        DCHECK_RARE(1e6f, lambda < lambdaMin || lambda > lambdaMax);
        int bucket = nBuckets * (lambda - lambdaMin) / (lambdaMax - lambdaMin);
        return Clamp(bucket, 0, nBuckets - 1);
    }

    // SpectralFilm::Pixel Definition
    struct Pixel {
        Pixel() = default;
        // Continue to store RGB, both to include in the final image as
        // well as for previews during rendering.
        double rgbSum[3] = {0., 0., 0.};
        double rgbWeightSum = 0.;
        AtomicDouble rgbSplat[3];
        // The following will all have nBuckets entries.
        double *bucketSums, *weightSums;
        AtomicDouble *bucketSplats;
    };

    // SpectralFilm Private Members
    const RGBColorSpace *colorSpace;
    Float lambdaMin, lambdaMax;
    int nBuckets;
    Float maxComponentValue;
    bool writeFP16;
    Float filterIntegral;
    Array2D<Pixel> pixels;
    SquareMatrix<3> outputRGBFromSensorRGB;
};

PBRT_CPU_GPU
inline SampledWavelengths Film::SampleWavelengths(Float u) const {
    auto sample = [&](auto ptr) { return ptr->SampleWavelengths(u); };
    return Dispatch(sample);
}

PBRT_CPU_GPU
inline Bounds2f Film::SampleBounds() const {
    auto sb = [&](auto ptr) { return ptr->SampleBounds(); };
    return Dispatch(sb);
}

PBRT_CPU_GPU
inline Bounds2i Film::PixelBounds() const {
    auto pb = [&](auto ptr) { return ptr->PixelBounds(); };
    return Dispatch(pb);
}

PBRT_CPU_GPU
inline Point2i Film::FullResolution() const {
    auto fr = [&](auto ptr) { return ptr->FullResolution(); };
    return Dispatch(fr);
}

PBRT_CPU_GPU
inline Float Film::Diagonal() const {
    auto diag = [&](auto ptr) { return ptr->Diagonal(); };
    return Dispatch(diag);
}

PBRT_CPU_GPU
inline Filter Film::GetFilter() const {
    auto filter = [&](auto ptr) { return ptr->GetFilter(); };
    return Dispatch(filter);
}

PBRT_CPU_GPU
inline bool Film::UsesVisibleSurface() const {
    auto uses = [&](auto ptr) { return ptr->UsesVisibleSurface(); };
    return Dispatch(uses);
}

PBRT_CPU_GPU
inline RGB Film::GetPixelRGB(Point2i p, Float splatScale) const {
    auto get = [&](auto ptr) { return ptr->GetPixelRGB(p, splatScale); };
    return Dispatch(get);
}

PBRT_CPU_GPU
inline RGB Film::ToOutputRGB(SampledSpectrum L, const SampledWavelengths &lambda) const {
    auto out = [&](auto ptr) { return ptr->ToOutputRGB(L, lambda); };
    return Dispatch(out);
}

PBRT_CPU_GPU
inline Float Film::GetMISAlpha(const Point2i p) const {
    auto get = [&](auto ptr) { return ptr->GetMISAlpha(p); };
    return Dispatch(get);
}

PBRT_CPU_GPU
inline void Film::UpdateNSamplesMIS(const Point2i p) {
    auto upd = [&](auto ptr) { return ptr->UpdateNSamplesMIS(p); };
    return Dispatch(upd);
}

PBRT_CPU_GPU
inline void Film::UpdateProbsMIS(const Point2i p, SampledSpectrum L, const SampledWavelengths &lambda, Float fpdf, Float gpdf) {
    auto upd = [&](auto ptr) { return ptr->UpdateProbsMIS(p, L, lambda, fpdf, gpdf); };
    return Dispatch(upd);
}

PBRT_CPU_GPU
inline void Film::ComputeUpdatedAlpha(const Point2i p) {
    auto upd = [&](auto ptr) { return ptr->ComputeUpdatedAlpha(p); };
    return Dispatch(upd);
}

PBRT_CPU_GPU
inline void Film::AddSample(Point2i pFilm, SampledSpectrum L,
                            const SampledWavelengths &lambda,
                            const VisibleSurface *visibleSurface, Float weight) {
    auto add = [&](auto ptr) {
        return ptr->AddSample(pFilm, L, lambda, visibleSurface, weight);
    };
    return Dispatch(add);
}

PBRT_CPU_GPU
inline const PixelSensor *Film::GetPixelSensor() const {
    auto filter = [&](auto ptr) { return ptr->GetPixelSensor(); };
    return Dispatch(filter);
}

PBRT_CPU_GPU
inline void Film::ResetPixel(Point2i p) {
    auto rp = [&](auto ptr) { ptr->ResetPixel(p); };
    return Dispatch(rp);
}

}  // namespace pbrt

#endif  // PBRT_FILM_H
