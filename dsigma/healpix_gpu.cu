#include "healpix_gpu.h"
#include <cmath> // For M_PI, sqrt, fabs, atan2, acos, fmod, sin, cos

// Define M_PI if not already defined (it's common in cmath but good to be safe)


__device__ double dot_gpu(const float3& a, const float3& b) {
    return static_cast<double>(a.x) * b.x + static_cast<double>(a.y) * b.y + static_cast<double>(a.z) * b.z;
}

__device__ void cross_gpu(const float3& a, const float3& b, float3& result) {
    result.x = static_cast<float>(static_cast<double>(a.y) * b.z - static_cast<double>(a.z) * b.y);
    result.y = static_cast<float>(static_cast<double>(a.z) * b.x - static_cast<double>(a.x) * b.z);
    result.z = static_cast<float>(static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x);
}

__device__ double acos_safe_gpu(double val) {
    if (val <= -1.0) {
        return PI;
    } else if (val >= 1.0) {
        return 0.0;
    } else {
        return acos(val);
    }
}

__device__ void ang2vec_gpu(double theta, double phi, float3& vec) {
    const double sin_theta = sin(theta);
    vec.x = static_cast<float>(sin_theta * cos(phi));
    vec.y = static_cast<float>(sin_theta * sin(phi));
    vec.z = static_cast<float>(cos(theta));
}




__device__ long ang2pix_ring_gpu(long nside, double theta, double phi) {
    if (nside <= 0) return -1; // Invalid nside

    const double z = cos(theta);
    const double za = fabs(z);
    const long ncap = 2 * nside * (nside - 1); // number of pixels in the north polar cap

    // Normalize phi to [0, 2*PI)
    double phi_norm = phi;
    while (phi_norm < 0.0) phi_norm += TWO_PI;
    while (phi_norm >= TWO_PI) phi_norm -= TWO_PI;

    // Equatorial region
    if (za <= 2.0 / 3.0) {
        long jr = static_cast<long>(nside * (2.0 - 1.5 * z)); // ring index from pole for z > 0 (nside to 2*nside)
        if (z < 0.0) jr = 4 * nside - jr; // Global ring index (nside to 3*nside)

        long jp = static_cast<long>((phi_norm * INV_PI * 0.5) * (4*nside) + (((jr - nside) % 2 == 1) ? 0.0 : 0.5) + 1.0);
        if (jp > 4*nside) jp = 4*nside;
        if (jp < 1) jp = 1;

        long pix_idx = ncap + (jr - nside) * 4 * nside + jp - 1; // 0-indexed
        return pix_idx;
    }
    // Polar caps
    else {
        if (z > 0.0) { // North polar cap
            long jr = static_cast<long>(nside * sqrt(3.0 * (1.0 - z)) + 1e-9) + 1;
            if (jr >= nside) jr = nside - 1;
            if (jr < 1) jr = 1;

            long jp = static_cast<long>((phi_norm * INV_PI * 0.5) * (4*jr) + 1.0);
            if (jp > 4 * jr) jp = 4 * jr;
            if (jp < 1) jp = 1;

            return 2 * jr * (jr - 1) + jp - 1; // 0-indexed
        } else { // South polar cap
            long jr = static_cast<long>(nside * sqrt(3.0 * (1.0 + z)) + 1e-9) + 1;
            if (jr >= nside) jr = nside - 1;
            if (jr < 1) jr = 1;

            long jp = static_cast<long>((phi_norm * INV_PI * 0.5) * (4*jr) + 1.0);
            if (jp > 4 * jr) jp = 4 * jr;
            if (jp < 1) jp = 1;

            return 12 * nside * nside - 2 * jr * (jr + 1) + jp - 1;
        }
    }
}


__device__ void pix2vec_ring_gpu(long nside, long pix, float3& vec) {
    if (nside <= 0 || pix < 0 || pix >= 12 * nside * nside) {
        vec.x = vec.y = vec.z = 0.0f; // Invalid input
        return;
    }

    const long npix = 12 * nside * nside;
    const long ncap = 2 * nside * (nside - 1); // number of pixels in one polar cap

    double z, phi;

    if (pix < ncap) { // North polar cap
        long iring = static_cast<long>(0.5 * (1.0 + sqrt(1.0 + 2.0 * static_cast<double>(pix))));
        if (iring == 0) iring = 1; // Should not happen for pix > 0. For pix=0, iring=1.
        long ipix_in_ring = pix - 2 * iring * (iring - 1); // pixel index within the ring [0, 4*iring-1]

        z = 1.0 - static_cast<double>(iring * iring) / (3.0 * nside * nside);
        phi = (static_cast<double>(ipix_in_ring) + 0.5) * PI / (2.0 * iring);
    } else if (pix < npix - ncap) { // Equatorial belt
        long ipix_belt = pix - ncap;
        long iring_global = (ipix_belt / (4 * nside)) + nside; // global ring index [nside, 3*nside]
        long ipix_in_ring = ipix_belt % (4 * nside); // pixel index within the ring [0, 4*nside-1]

        // CRITICAL FIX: Use the correct HEALPix formula for equatorial region z-coordinate
        // The formula is: z = (4*nside - 2*iring) / (3*nside)
        z = (4.0 * static_cast<double>(nside) - 2.0 * static_cast<double>(iring_global)) / (3.0 * static_cast<double>(nside));

        // Determine phi with proper shifting
        long s = (iring_global - nside) % 2; // s=0 for first ring in belt, s=1 for second, etc.
        phi = (static_cast<double>(ipix_in_ring) + 0.5 + s * 0.5) * PI / (2.0 * nside);

    } else { // South polar cap
        long pix_from_south_start = pix - (npix - ncap); // offset from start of south cap
        long iring_s = static_cast<long>(floor(0.5 * (1.0 + sqrt(1.0 + 2.0 * pix_from_south_start))));
        if (iring_s == 0 && pix_from_south_start > 0) iring_s = 1;

        long ipix_in_ring_s = pix_from_south_start - 2 * iring_s * (iring_s - 1);

        z = -(1.0 - static_cast<double>(iring_s * iring_s) / (3.0 * nside * nside));
        phi = (static_cast<double>(ipix_in_ring_s) + 0.5) * PI / (2.0 * iring_s);
    }

    // Clamp z to avoid domain errors for acos due to precision
    if (z > 1.0) z = 1.0;
    else if (z < -1.0) z = -1.0;

    double theta = acos(z);
    ang2vec_gpu(theta, phi, vec);
}


