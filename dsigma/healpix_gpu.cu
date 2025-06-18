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

// Maximum number of pixels that can be returned by cone search
#define MAX_CONE_SEARCH_RESULTS 1024

// Simple queue implementation for frontier-based search
struct PixelQueue {
    long pixels[MAX_CONE_SEARCH_RESULTS];
    int head;
    int tail;
    int size;
    
    __device__ void init() {
        head = tail = size = 0;
    }
    
    __device__ bool push(long pixel) {
        if (size >= MAX_CONE_SEARCH_RESULTS) return false;
        pixels[tail] = pixel;
        tail = (tail + 1) % MAX_CONE_SEARCH_RESULTS;
        size++;
        return true;
    }
    
    __device__ bool pop(long& pixel) {
        if (size <= 0) return false;
        pixel = pixels[head];
        head = (head + 1) % MAX_CONE_SEARCH_RESULTS;
        size--;
        return true;
    }
    
    __device__ bool empty() const {
        return size == 0;
    }
    
    __device__ bool contains(long pixel) const {
        for (int i = 0; i < size; i++) {
            int idx = (head + i) % MAX_CONE_SEARCH_RESULTS;
            if (pixels[idx] == pixel) return true;
        }
        return false;
    }
};

// Simple set implementation for tracking visited pixels
struct PixelSet {
    long pixels[MAX_CONE_SEARCH_RESULTS];
    int count;
    
    __device__ void init() {
        count = 0;
    }
    
    __device__ bool add(long pixel) {
        // Check if already present
        for (int i = 0; i < count; i++) {
            if (pixels[i] == pixel) return false;
        }
        
        if (count >= MAX_CONE_SEARCH_RESULTS) return false;
        pixels[count++] = pixel;
        return true;
    }
    
    __device__ bool contains(long pixel) const {
        for (int i = 0; i < count; i++) {
            if (pixels[i] == pixel) return true;
        }
        return false;
    }
};

__device__ void get_neighbors_ring_gpu(long nside, long pix, long* neighbors, int& num_neighbors) {
    // Improved neighbor finding for ring scheme
    // This implements a simplified version based on HEALPix geometry
    num_neighbors = 0;
    
    if (pix < 0 || pix >= 12 * nside * nside) return;
    
    const long npix = 12 * nside * nside;
    const long ncap = 2 * nside * (nside - 1);
    
    // Convert to 1-based indexing for calculations
    long ipix1 = pix + 1;
    
    // Find which region the pixel is in
    if (ipix1 <= ncap) {
        // North polar cap
        double hip = ipix1 / 2.0;
        double fihip = (double)((long)hip);
        long iring = (long)(sqrt(hip - sqrt(fihip))) + 1;
        long iphi = ipix1 - 2 * iring * (iring - 1);
        
        // Add neighbors within the same ring and adjacent rings
        for (long ring_offset = -1; ring_offset <= 1; ring_offset++) {
            long test_ring = iring + ring_offset;
            if (test_ring >= 1 && test_ring <= nside - 1) {
                for (long phi_offset = -1; phi_offset <= 1; phi_offset++) {
                    if (ring_offset == 0 && phi_offset == 0) continue; // Skip self
                    
                    long test_iphi = iphi + phi_offset;
                    if (test_iphi <= 0) test_iphi += 4 * test_ring;
                    if (test_iphi > 4 * test_ring) test_iphi -= 4 * test_ring;
                    
                    long neighbor_pix = 2 * test_ring * (test_ring - 1) + test_iphi - 1;
                    if (neighbor_pix >= 0 && neighbor_pix < npix && num_neighbors < 8) {
                        neighbors[num_neighbors++] = neighbor_pix;
                    }
                }
            }
        }
    } else if (ipix1 <= 2 * nside * (5 * nside + 1)) {
        // Equatorial region
        long ip = ipix1 - ncap - 1;
        long iring = (ip / (4 * nside)) + nside;
        long iphi = (ip % (4 * nside)) + 1;
        
        // Add neighbors in same and adjacent rings
        for (long ring_offset = -1; ring_offset <= 1; ring_offset++) {
            long test_ring = iring + ring_offset;
            if (test_ring >= nside && test_ring <= 3 * nside) {
                for (long phi_offset = -1; phi_offset <= 1; phi_offset++) {
                    if (ring_offset == 0 && phi_offset == 0) continue; // Skip self
                    
                    long test_iphi = iphi + phi_offset;
                    if (test_iphi <= 0) test_iphi += 4 * nside;
                    if (test_iphi > 4 * nside) test_iphi -= 4 * nside;
                    
                    long neighbor_pix;
                    if (test_ring >= nside && test_ring <= 3 * nside) {
                        neighbor_pix = ncap + (test_ring - nside) * 4 * nside + test_iphi - 1;
                    } else {
                        continue;
                    }
                    
                    if (neighbor_pix >= 0 && neighbor_pix < npix && num_neighbors < 8) {
                        neighbors[num_neighbors++] = neighbor_pix;
                    }
                }
            }
        }
    } else {
        // South polar cap
        long ip = npix - ipix1 + 1;
        double hip = ip / 2.0;
        double fihip = (double)((long)hip);
        long iring = (long)(sqrt(hip - sqrt(fihip))) + 1;
        long iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        
        // Add neighbors within the same ring and adjacent rings
        for (long ring_offset = -1; ring_offset <= 1; ring_offset++) {
            long test_ring = iring + ring_offset;
            if (test_ring >= 1 && test_ring <= nside - 1) {
                for (long phi_offset = -1; phi_offset <= 1; phi_offset++) {
                    if (ring_offset == 0 && phi_offset == 0) continue; // Skip self
                    
                    long test_iphi = iphi + phi_offset;
                    if (test_iphi <= 0) test_iphi += 4 * test_ring;
                    if (test_iphi > 4 * test_ring) test_iphi -= 4 * test_ring;
                    
                    long neighbor_pix = npix - 2 * test_ring * (test_ring + 1) + test_iphi - 1;
                    if (neighbor_pix >= 0 && neighbor_pix < npix && num_neighbors < 8) {
                        neighbors[num_neighbors++] = neighbor_pix;
                    }
                }
            }
        }
    }
    
    // Fallback: if we couldn't find enough neighbors using the geometric approach,
    // add some adjacent pixel indices as backup
    if (num_neighbors < 4) {
        long test_pixels[] = {pix - 1, pix + 1, pix - 4 * nside, pix + 4 * nside};
        for (int i = 0; i < 4 && num_neighbors < 8; i++) {
            long test_pix = test_pixels[i];
            if (test_pix >= 0 && test_pix < npix) {
                // Check if already added
                bool already_added = false;
                for (int j = 0; j < num_neighbors; j++) {
                    if (neighbors[j] == test_pix) {
                        already_added = true;
                        break;
                    }
                }
                if (!already_added) {
                    neighbors[num_neighbors++] = test_pix;
                }
            }
        }
    }
}

__device__ double angular_distance_gpu(const float3& vec1, const float3& vec2) {
    // Calculate angular distance between two unit vectors on sphere
    double dot_product = dot_gpu(vec1, vec2);
    // Clamp to valid range for acos
    dot_product = fmax(-1.0, fmin(1.0, dot_product));
    return acos_safe_gpu(dot_product);
}

__device__ bool pixel_within_radius_gpu(long nside, long pixel, const float3& center_vec, double radius) {
    // Check if a pixel is within the search radius
    float3 pixel_vec;
    pix2vec_ring_gpu(nside, pixel, pixel_vec);
    
    double distance = angular_distance_gpu(center_vec, pixel_vec);
    
    // Use sqrt of pixel area as buffer (more reasonable than π/(2*nside))
    double pixel_radius = sqrt(get_pixel_area_steradians_gpu(nside));
    
    return distance <= (radius + pixel_radius);
}

__device__ int cone_search_ring_gpu(long nside, double lon_deg, double lat_deg, double radius_deg, long* result_pixels) {
    // Convert degrees to radians
    double lon_rad = lon_deg * M_PI / 180.0;
    double lat_rad = lat_deg * M_PI / 180.0;
    double radius_rad = radius_deg * M_PI / 180.0;
    
    // Convert to theta, phi (colatitude, azimuth)
    double theta = M_PI / 2.0 - lat_rad;  // theta = pi/2 - latitude
    double phi = lon_rad;
    
    // Convert center position to Cartesian vector
    float3 center_vec;
    ang2vec_gpu(theta, phi, center_vec);
    
    // Find the central pixel
    long center_pixel = ang2pix_ring_gpu(nside, theta, phi);
    if (center_pixel < 0) return 0;
    
    // Initialize data structures for frontier search
    PixelQueue frontier;
    PixelSet visited;
    PixelSet results;
    
    frontier.init();
    visited.init();
    results.init();
    
    // Start with center pixel
    if (pixel_within_radius_gpu(nside, center_pixel, center_vec, radius_rad)) {
        frontier.push(center_pixel);
        visited.add(center_pixel);
        results.add(center_pixel);
    }
    
    // Frontier-based search
    while (!frontier.empty()) {
        long current_pixel;
        if (!frontier.pop(current_pixel)) break;
        
        // Get neighbors
        long neighbors[8];
        int num_neighbors;
        get_neighbors_ring_gpu(nside, current_pixel, neighbors, num_neighbors);
        
        for (int i = 0; i < num_neighbors; i++) {
            long neighbor = neighbors[i];
            
            // Skip if already visited
            if (visited.contains(neighbor)) continue;
            
            visited.add(neighbor);
            
            // Check if neighbor is within radius
            if (pixel_within_radius_gpu(nside, neighbor, center_vec, radius_rad)) {
                results.add(neighbor);
                frontier.push(neighbor);
            }
        }
    }
    
    // Copy results to output array
    int result_count = results.count;
    for (int i = 0; i < result_count; i++) {
        result_pixels[i] = results.pixels[i];
    }
    
    return result_count;
}




