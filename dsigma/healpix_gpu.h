#ifndef DSIGMA_HEALPIX_GPU_H
#define DSIGMA_HEALPIX_GPU_H

#include <cuda_runtime.h>
#include <vector_types.h> // For float3
#include <cmath> // For fabsf

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 // M_PI / 2
#endif

#ifndef M_2PI
#define M_2PI 6.28318530717958647692 // 2 * M_PI
#endif

// Helper constants for HEALPix
static const double PI = M_PI;
static const double TWO_PI = M_2PI;
static const double HALF_PI = M_PI_2;
static const double INV_PI = 1.0 / M_PI;
static const double INV_TWO_PI = 1.0 / M_2PI;

// Forward declarations of __device__ functions

/**
 * @brief Converts spherical coordinates (theta, phi) to a Cartesian vector.
 * @param theta Polar angle (colatitude) in radians.
 * @param phi Azimuthal angle in radians.
 * @param vec Output Cartesian vector (float3).
 */
__device__ void ang2vec_gpu(double theta, double phi, float3& vec);

/**
 * @brief Converts angular coordinates (theta, phi) to a HEALPix pixel index
 *        in the RING scheme.
 * @param nside HEALPix resolution parameter.
 * @param theta Polar angle (colatitude) in radians.
 * @param phi Azimuthal angle in radians.
 * @return HEALPix pixel index.
 */
__device__ long ang2pix_ring_gpu(long nside, double theta, double phi);

/**
 * @brief Converts a HEALPix pixel index in the RING scheme to a Cartesian vector
 *        pointing to the center of the pixel.
 * @param nside HEALPix resolution parameter.
 * @param pix HEALPix pixel index.
 * @param vec Output Cartesian vector (float3).
 */
__device__ void pix2vec_ring_gpu(long nside, long pix, float3& vec);

/**
 * @brief Calculates an estimate of the maximum pixel radius for a given nside.
 *        This is an approximation.
 * @param nside HEALPix resolution parameter.
 * @return Approximate maximum pixel radius in radians.
 */
__device__ __forceinline__ double get_max_pixrad_gpu(long nside) {
    // This is an approximation. A common one is related to pixel resolution.
    // The average pixel spacing is roughly related to sqrt(Area_sphere / Npix).
    // Npix = 12 * nside^2. Area_sphere = 4*PI.
    // So, spacing ~ sqrt(4*PI / (12*nside^2)) = sqrt(PI / (3*nside^2)) = (1/nside) * sqrt(PI/3).
    // Max pixel radius is a bit larger.
    // Another common approximation for pixel radius is ~2/nside radians.
    // Healpix_Base::max_pixrad() is more complex.
    // For now, using a simpler formula, e.g. M_PI / (2.0 * nside) as used in some contexts,
    // or the one from problem statement: sqrt(M_PI / (3.0 * nside * nside)) * (2.0 / M_PI) * 2.0
    // Let's use a slightly more standard approximation if possible, related to ~resolution.
    // The solid angle of a pixel is approx. 4*PI / (12*nside^2) = PI / (3*nside^2).
    // Radius of a disk with this area is r_eff = sqrt( (PI / (3*nside^2)) / PI) = 1 / (sqrt(3)*nside).
    // Max radius can be larger. A common figure is around 2 times the mean spacing.
    // Let's use M_PI / (2.0 * nside) for now, it's a known upper bound in some cases.
    // Or simply: return 2.0 / nside; (approx pixel diameter)
    // A value often quoted for pixel angular radius is ~ 1 / nside.
    // The problem description suggested M_PI / (2.0 * nside). Let's use that.
    if (nside <= 0) return PI; // Or some other appropriate error/default
    return PI / (2.0 * static_cast<double>(nside));
}

/**
 * @brief Converts spherical coordinates from (sin_ra, cos_ra, sin_dec, cos_dec)
 *        to a Cartesian vector.
 * @param sin_ra Sine of Right Ascension.
 * @param cos_ra Cosine of Right Ascension.
 * @param sin_dec Sine of Declination.
 * @param cos_dec Cosine of Declination.
 * @param vec Output Cartesian vector (float3).
 */
__device__ __forceinline__ void spherical_to_cartesian_gpu(double sin_ra, double cos_ra, double sin_dec, double cos_dec, float3& vec) {
    // Standard conversion:
    // x = cos(dec) * cos(ra)
    // y = cos(dec) * sin(ra)
    // z = sin(dec)
    // Here, ra is phi-like, dec is (pi/2 - theta)-like.
    // So, x = sin(theta) * cos(phi)
    //     y = sin(theta) * sin(phi)
    //     z = cos(theta)
    // With dec = pi/2 - theta: sin(dec) = cos(theta), cos(dec) = sin(theta)
    // With ra = phi
    vec.x = static_cast<float>(cos_dec * cos_ra); // cos_dec * cos_ra
    vec.y = static_cast<float>(cos_dec * sin_ra); // cos_dec * sin_ra
    vec.z = static_cast<float>(sin_dec);          // sin_dec
}

/**
 * @brief Computes the dot product of two float3 vectors.
 * @param a First vector.
 * @param b Second vector.
 * @return Dot product.
 */
__device__ double dot_gpu(const float3& a, const float3& b);

/**
 * @brief Computes the cross product of two float3 vectors.
 * @param a First vector.
 * @param b Second vector.
 * @param result Output cross product vector.
 */
__device__ void cross_gpu(const float3& a, const float3& b, float3& result);

/**
 * @brief Computes acos(val) ensuring val is within [-1, 1].
 * @param val Input value.
 * @return acos(val) clamped to the valid domain.
 */
__device__ double acos_safe_gpu(double val);

/**
 * @brief Converts a HEALPix pixel index in the RING scheme to angular coordinates.
 * @param nside HEALPix resolution parameter.
 * @param pix HEALPix pixel index.
 * @param theta Output polar angle (colatitude) in radians.
 * @param phi Output azimuthal angle in radians.
 */
__device__ void pix2ang_ring_gpu(long nside, long pix, double& theta, double& phi);

/**
 * @brief Converts a HEALPix pixel index in the RING scheme to longitude/latitude coordinates.
 * @param nside HEALPix resolution parameter.
 * @param pix HEALPix pixel index.
 * @param lon Output longitude in degrees (0 to 360).
 * @param lat Output latitude in degrees (-90 to 90).
 */
__device__ void pix2lonlat_ring_gpu(long nside, long pix, double& lon, double& lat);

/**
 * @brief Gets the pixel resolution in degrees for a given nside.
 * @param nside HEALPix resolution parameter.
 * @return Pixel resolution in degrees.
 */
__device__ __forceinline__ double get_pixel_resolution_degrees_gpu(long nside) {
    if (nside <= 0) return 180.0; // Error case
    // HEALPix pixel solid angle is 4*PI / (12*nside^2) steradians
    // Convert to degrees: sqrt(solid_angle) * (180/PI)
    // Approximation: pixel angular size ~ sqrt(4*PI / (12*nside^2)) * 180/PI
    //                                  = sqrt(PI/3) / nside * 180/PI
    //                                  ≈ 60 / nside degrees
    const double PI_OVER_180 = PI / 180.0;
    const double solid_angle = 4.0 * PI / (12.0 * static_cast<double>(nside) * nside);
    return sqrt(solid_angle) * 180.0 / PI;
}

/**
 * @brief Performs a cone search to find all HEALPix pixels within a given radius
 *        of a longitude/latitude position using the RING scheme.
 * @param nside HEALPix resolution parameter.
 * @param lon_deg Longitude in degrees.
 * @param lat_deg Latitude in degrees.
 * @param radius_deg Search radius in degrees.
 * @param result_pixels Output array to store found pixel indices.
 * @return Number of pixels found (size of result_pixels filled).
 */
__device__ int cone_search_ring_gpu(long nside, double lon_deg, double lat_deg, double radius_deg, long* result_pixels);

/**
 * @brief Gets the pixel area in steradians for a given nside.
 * @param nside HEALPix resolution parameter.
 * @return Pixel area in steradians.
 */
__device__ double get_pixel_area_steradians_gpu(long nside);

#endif // DSIGMA_HEALPIX_GPU_H
