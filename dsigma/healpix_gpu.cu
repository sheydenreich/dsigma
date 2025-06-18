#include <cmath>
#include <cstdio>
#include "healpix_gpu.h"

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TWO_PI
#define TWO_PI (2.0 * M_PI)
#endif

#ifndef INV_PI
#define INV_PI (1.0 / M_PI)
#endif

// Constants for HEALPix - EXACT values from specification
#define TWOTHIRD (2.0 / 3.0)

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
        return M_PI;
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

// Corrected ang2pix implementation based on exact Java zPhi2Pix specification
__device__ long ang2pix_ring_gpu(long nside, double theta, double phi) {
    if (nside <= 0) return -1;

    const double z = cos(theta);
    const double za = abs(z);
    const long nl2 = 2 * nside;
    const long nl4 = 4 * nside;
    const long ncap = nl2 * (nside - 1); // number of pixels in the north polar cap
    const long npix = 12 * nside * nside;

    // Normalize phi to [0, 2*PI)
    double phi_norm = phi;
    while (phi_norm < 0.0) phi_norm += TWO_PI;
    while (phi_norm >= TWO_PI) phi_norm -= TWO_PI;

    long ipix1;
    double tt = phi_norm / (M_PI / 2.0); // in [0,4] - equivalent to HALFPI

    if (za < TWOTHIRD) { // equatorial region
        long jp = (long) (nside * (0.5 + tt - 0.75 * z)); // index of ascending edge line
        long jm = (long) (nside * (0.5 + tt + 0.75 * z)); // index of descending edge line

        long ir = nside + 1 + jp - jm; // in [1,2n+1]
        long kshift = 0;
        if ((ir % 2) == 0)
            kshift = 1; // 1 if ir even, 0 otherwise
        long ip = ((jp + jm - nside + kshift + 1) / 2) + 1; // in [1,4n]
        ipix1 = ncap + nl4 * (ir - 1) + ip;
    } else { // North and South polar caps
        double tp = tt - (long) tt;
        double tmp = sqrt(3.0 * (1.0 - za));
        long jp = (long) (nside * tp * tmp); // increasing edge line index
        long jm = (long) (nside * (1.0 - tp) * tmp); // decreasing edge index

        long ir = jp + jm + 1; // ring number counted from closest pole
        long ip = (long) (tt * ir) + 1; // in [1,4*ir]
        if (ip > 4 * ir)
            ip = ip - 4 * ir;

        ipix1 = 2 * ir * (ir - 1) + ip;
        if (z <= 0.0)
            ipix1 = npix - 2 * ir * (ir + 1) + ip;
    }
    return ipix1 - 1; // in [0, npix-1]
}

__device__ void pix2ang_ring_gpu(long nside, long pix, double& theta, double& phi) {
    if (nside <= 0 || pix < 0 || pix >= 12 * nside * nside) {
        theta = phi = 0.0;
        return;
    }

    const long npix = 12 * nside * nside;
    const long ncap = 2 * nside * (nside - 1);
    const long nsidesq = nside * nside;
    const long nl2 = 2 * nside;
    const long nl4 = 4 * nside;
    double z;

    long ipix1 = pix + 1; // Convert to 1-based indexing as in Java

    if (ipix1 <= ncap) { // North polar cap
        // Java algorithm for north cap
        double hip = ipix1 / 2.0;
        double fihip = (long) hip; // get integer part of hip
        long iring = (long) (sqrt(hip - sqrt(fihip))) + 1; // counted from north pole
        long iphi = ipix1 - 2 * iring * (iring - 1);
        
        theta = acos_safe_gpu(1.0 - iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
        
    } else if (ipix1 <= nl2 * (5 * nside + 1)) { // Equatorial region
        // Java algorithm for equatorial region
        long ip = ipix1 - ncap - 1;
        long iring = (ip / nl4) + nside; // counted from North pole
        long iphi = ip % nl4 + 1;
        
        // Critical fix: Use exact Java formula for fodd
        // fodd = 0.5 * (1. + BitManipulation.MODULO(iring + nside, 2))
        // BitManipulation.MODULO(a, 2) is equivalent to (a % 2)
        double fodd = 0.5 * (1.0 + ((iring + nside) % 2)); // 1 if iring+nside is odd, 1/2 otherwise
        
        theta = acos_safe_gpu((nl2 - iring) / (1.5 * nside));
        phi = ((double)iphi - fodd) * M_PI / (2.0 * nside);

    } else { // South polar cap
        // Java algorithm for south cap
        long ip = npix - ipix1 + 1;
        double hip = ip / 2.0;
        double fihip = (long) hip;
        long iring = (long) (sqrt(hip - sqrt(fihip))) + 1; // counted from South pole
        long iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        
        theta = acos_safe_gpu(-1.0 + iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
    }
    
    // Normalize phi to [0, 2*PI)
    while (phi < 0.0) phi += TWO_PI;
    while (phi >= TWO_PI) phi -= TWO_PI;
}

__device__ void pix2vec_ring_gpu(long nside, long pix, float3& vec) {
    double theta, phi;
    pix2ang_ring_gpu(nside, pix, theta, phi);
    ang2vec_gpu(theta, phi, vec);
}

__device__ void pix2lonlat_ring_gpu(long nside, long pix, double& lon, double& lat) {
    double theta, phi;
    pix2ang_ring_gpu(nside, pix, theta, phi);
    
    lon = phi * 180.0 / M_PI;
    lat = 90.0 - theta * 180.0 / M_PI;
    
    // Ensure longitude is in [0, 360) - FIXED: Changed from [-180, 180] to [0, 360)
    while (lon < 0.0) lon += 360.0;
    while (lon >= 360.0) lon -= 360.0;
}

__device__ double get_pixel_area_steradians_gpu(long nside) {
    return (4.0 * M_PI) / (12.0 * nside * nside);
}




