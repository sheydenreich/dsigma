#ifndef DSIGMA_HEALPIX_GPU_H
#define DSIGMA_HEALPIX_GPU_H

#include <cuda_runtime.h>
#include <vector_types.h> // For float3

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
__device__ double get_max_pixrad_gpu(long nside);

/**
 * @brief Converts spherical coordinates from (sin_ra, cos_ra, sin_dec, cos_dec)
 *        to a Cartesian vector.
 * @param sin_ra Sine of Right Ascension.
 * @param cos_ra Cosine of Right Ascension.
 * @param sin_dec Sine of Declination.
 * @param cos_dec Cosine of Declination.
 * @param vec Output Cartesian vector (float3).
 */
__device__ void spherical_to_cartesian_gpu(double sin_ra, double cos_ra, double sin_dec, double cos_dec, float3& vec);

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

#endif // DSIGMA_HEALPIX_GPU_H
