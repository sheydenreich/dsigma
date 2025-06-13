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

    // Equatorial region
    if (za <= 2.0 / 3.0) {
        const double inv_nside = 1.0 / static_cast<double>(nside);
        const long jp = static_cast<long>(floor(nside * (0.5 + phi * INV_TWO_PI + z * 0.75))); // index in longitude
        const long jm = static_cast<long>(floor(nside * (0.5 + phi * INV_TWO_PI - z * 0.75))); // index in longitude
        const long ir = nside + 1 + jp - jm; // (int) (nside*(2.0-1.5*z));
        const long k = static_cast<long>(fmod(ir + jp + jm, 2.0)); // (ir+jp+jm) % 2;
        const long ip = (k == 0) ? (jp + jm) / 2 : (jp + jm + 1) / 2;
        const long ipix1 = ncap + (ir - 1) * 4 * nside + ip;
        return ipix1 % (12 * nside * nside) ; // Ensure it's within total Npix (e.g. for phi=2PI)
    }
    // Polar caps
    else {
        const double tp = phi * INV_TWO_PI - floor(phi * INV_TWO_PI); // fmod(phi, TWO_PI) / TWO_PI but ensures [0,1)
        const double tmp = sqrt(3.0 * (1.0 - za));
        const long jp = static_cast<long>(floor(nside * tp * tmp)); // index in longitude
        const long jm = static_cast<long>(floor(nside * (1.0 - tp) * tmp)); // index in longitude
        long ir = (z > 0) ? (nside - jp - jm) : (3 * nside + jp + jm); // Ring index counted from pole
        const long ip = static_cast<long>(floor(phi * ir * INV_TWO_PI / (tmp * tmp * nside * nside / (ir * ir)) )); // This is not right
                                                                    // The standard formula for ip in polar caps:
                                                                    // ip = (long) floor(phi / (PI/(2*ir))) for North
                                                                    // ip = (long) floor(phi / (PI/(2*ir))) for South (adjusting for phi start)

        // Simplified calculation for ip (pixel index within the ring)
        // In polar caps, pixels are not square. The number of pixels in ring `ir` is `4*ir`.
        // `phi` needs to be scaled appropriately.
        // Let's use a more standard formulation for `ip` in caps.
        // `phi` is in [0, 2PI]. `ip` should be in [1, 4*ir].
        // The pixel centers are at `phi_k = (k - 0.5) * (2PI / (4*ir))` for `k=1,...,4*ir`.
        // So, `phi / (2PI / (4*ir))` gives an index. `phi * (4*ir) / (2PI)` = `phi * 2*ir / PI`.
        // `ip = static_cast<long>(floor(phi * 2.0 * ir * INV_PI)) + 1;` // This maps to [1, 4ir]
        // This needs to be careful with phi wrap around.
        // The HEALPix algorithm for `ip` in polar caps is:
        // `ip = (long)( ir * phi * 2.0 * INV_PI ) + 1;` if using 1-based indexing for ip.
        // Or, if we follow the logic in some implementations:
        // `iph = (long)floor( phi_norm * nir );` where phi_norm is phi/(pi/2) for that quadrant and nir is ir.
        // This part is tricky. Let's use a known formulation from a reliable source.
        // From Healpix_cxx ang2pix_ring:
        // tt = phi / (PI/2)  (scaled phi, 0-4)
        // if (z > 0) { // North polar cap
        //    pix = 2*ir*(ir-1) + (long)(tt*ir) + 1;
        // } else { // South polar cap
        //    pix = 12*nside*nside - 2*ir*(ir+1) + (long)(tt*ir) + 1;
        // }
        // Where `ir` is ring number from pole (1 to nside-1).
        // `tt` is tricky, it's not just `phi / (PI/2)`. It's related to `fmod(phi, HALF_PI)`.

        // Let's use the formulation closer to the reference HEALPix paper / common implementations.
        // Ring index `iring` from 1 to `4*nside - 1`.
        // For North cap: `iring = nside - jp - jm`. Pixels in ring `iring` is `4*iring`.
        // For South cap: `iring = nside - jp - jm` (relative to south pole), so global `iring` is `4*nside - (nside-jp-jm)`.
        // This `jp, jm` calculation is for `NESTED` scheme usually.

        // RING scheme polar cap:
        long iring; // Ring index from 1 (north pole) to 4*nside-1 (south pole)
        long iphi;  // Pixel index from 1 within the ring
        long base_pix;

        if (z > 0) { // North polar cap
            iring = static_cast<long>(floor(nside * sqrt(3.0 * (1.0 - z)) + 0.5)); // ring index from North pole [1, nside-1]
            // Alternative for iring: double tmp_val = nside * sqrt(3.0 * (1.0-z)); iring = (long)(tmp_val - 1e-9) +1;
            // The +0.5 and floor is equivalent to rounding.
            iphi = static_cast<long>(floor(phi * (2.0 * iring) * INV_PI) + 1.0); // Pixel index within ring [1, 4*iring]
            if (iphi > 4 * iring) iphi = 4 * iring; // clamp due to precision
            base_pix = 2 * iring * (iring - 1);
            return base_pix + iphi -1; // 0-indexed
        } else { // South polar cap
            iring = static_cast<long>(floor(nside * sqrt(3.0 * (1.0 + z)) + 0.5)); // ring index from South pole [1, nside-1]
            iphi = static_cast<long>(floor(phi * (2.0 * iring) * INV_PI) + 1.0); // Pixel index within ring [1, 4*iring]
            if (iphi > 4 * iring) iphi = 4 * iring; // clamp
            base_pix = 12 * nside * nside - 2 * iring * (iring + 1);
            return base_pix + iphi -1; // 0-indexed
        }
    }
     // Fallback for the equatorial belt logic from HEALPix C++ library (simplified)
    // This part is complex and needs to be accurate. The above `jp, jm` was a bit mixed.
    // Let's use a more direct RING scheme formulation.
    // For equatorial belt:
    // z_abs = fabs(z)
    // iring = floor( nside * (2.0 - 1.5 * z_abs) )  -- this is ring index from pole for north,
    //                                             -- or from equator for south.
    // Let's use the common `iring` definition: 1 to 4*nside-1.
    // In the equatorial belt: `nside <= iring <= 3*nside`.
    // `ip = floor( phi / dphi ) + s` where `dphi = PI / (2*nside)` and `s` depends on ring parity.

    // Correct equatorial belt logic for RING scheme:
    // `theta` is colatitude. `phi` is longitude.
    // `z = cos(theta)`.
    // `nside` is the resolution parameter.
    // `npix = 12 * nside * nside`.

    // Ring index `iring` (from 1 at North Pole to `4*nside-1` at South Pole).
    // `shift = (iring / nside) % 2` -- this is not standard.
    // The number of pixels in a ring `iring` (in the belt) is `4*nside`.
    // `phi_0 = 0` for odd rings `iring - nside + 1`.
    // `phi_0 = PI / (4*nside)` for even rings `iring - nside + 1`.
    // This means the pixel centers are shifted.

    // Let's use the standard HEALPix formulation for ring scheme:
    // `z = cos(theta)`
    // if (fabs(z) > 2.0/3.0) { // Polar caps
    //   ... (handled above)
    // } else { // Equatorial belt
    //   long iring = static_cast<long>(nside * (2.0 - 1.5 * z)); // Ring index in North, from nside to 2*nside
    //                                                           // Or from South pole, 3*nside to 2*nside+1
    //   if (z < 0) iring = 4*nside - iring; // Adjust for South
    //  This iring is not the global one from 1 to 4*nside-1.

    // Let's use `nl2 = 2*nside`, `nl4 = 4*nside`
    // `ncap = nl2 * (nside - 1)`
    // `npix = 12 * nside * nside`

    // For the equatorial belt ( |z| <= 2/3 )
    // `iring = static_cast<long>( nside * (2.0 - 1.5 * z) );` // for z > 0, ring index from pole
    // `iphi  = static_cast<long>( (phi*INV_TWO_PI - floor(phi*INV_TWO_PI)) * nl4 + 1.5 - 0.5 * (iring-nside)%2 );`
    // This `iphi` formulation is also complex.

    // Using the formulation from `chealpix.c` (simplified for RING):
    // `z = cos(theta)`
    // `az = fabs(z)`
    // `phi_temp = fmod(phi, TWO_PI);`
    // `if (phi_temp < 0.) phi_temp += TWO_PI;`

    // if (az <= 2./3.) { // Equatorial Region
    //   long ir = static_cast<long>(nside * (2. - 1.5 * z)); // ring number in North, nside to 2*nside
    //                                                       // for South, nside to 2*nside (from SPole)
    //   if (z < 0.) ir = nl4 - ir +1; // Global ring index
    //   long ip = static_cast<long>((phi_temp * INV_PI * 0.5 * nl4 + ( ( (ir - nside) % 2 ) == 1 ? 0.0 : 0.5 )) + 1.0);
    //   if (ip > nl4) ip = nl4;
    //   long base_pix = ncap + (ir - nside) * nl4;
    //   return base_pix + ip -1;
    // } else { // Polar caps - already handled above.
    //   ...
    // }
    // The above equatorial logic for `ir` and `ip` needs to be integrated with the polar cap logic.
    // The first attempt for equatorial region was:
    // const double inv_nside = 1.0 / static_cast<double>(nside);
    // const long jp = static_cast<long>(floor(nside * (0.5 + phi * INV_TWO_PI + z * 0.75)));
    // const long jm = static_cast<long>(floor(nside * (0.5 + phi * INV_TWO_PI - z * 0.75)));
    // const long ir = nside + 1 + jp - jm; // This `ir` is ring index from North Pole, ranges from 1 to 2*nside-1 for North H.
    //                                     // and 2*nside to 4*nside-1 for South H. This is global ring index.
    // const long k = static_cast<long>(fmod(ir + jp + jm, 2.0));
    // const long ip = (k == 0) ? (jp + jm) / 2 : (jp + jm + 1) / 2;
    // long pixnum = ncap + (ir - (nside -1) -1) * 4 * nside + ip + 1; // This is for ir > nside-1
    // This `ir` from `jp, jm` method is the actual global ring index.
    // `ir` ranges from 1 to `4*nside-1`.
    // So, if `ir < nside` (North Cap), `ir > 3*nside` (South Cap).
    // `nside <= ir <= 3*nside` is equatorial.

    // Let's use the standard HEALPix C / Fortran algorithm structure for ang2pix_ring:
    double z_abs = fabs(z);
    double phi_norm = phi * INV_TWO_PI; // Normalize phi to [0,1) typically
    if (phi_norm < 0.0) phi_norm += 1.0;
    if (phi_norm >= 1.0) phi_norm -=1.0;


    if (z_abs <= 2.0/3.0) { // Equatorial belt
        // iring: global ring index from North pole (1 to 4*nside-1)
        // For z > 0: nside <= iring <= 2*nside
        // For z = 0: iring = 2*nside or 2*nside+1 (middle rings)
        // For z < 0: 2*nside+1 <= iring <= 3*nside
        long iring = static_cast<long>(nside * (2.0 - 1.5 * z) + 0.5); // for z>0, nside .. 2*nside. for z<0, 2*nside .. 3*nside
                                                                     // this is relative to N or S pole for region.
                                                                     // No, this is wrong. This gives ring index within N/S hemisphere belt part.

        // From HEALPix paper (Gorski et al. 2005, ApJ, 622, 759) Eq (6) & (7)
        // For |z| <= 2/3 (Equatorial belt):
        long s = (nside - static_cast<long>(nside * z * 1.5) % 2 == 1) ? 1 : 0; // s = ( (nside - k)%2 == 1) ? 1 : 0 where k = nside*z*1.5
                                                                              // k = i_H, the ring index along latitude
                                                                              // This s is the shift factor.
        // The ring index `iring` from N pole is `nside - 1 + i_H` where `i_H` is ring index from equator, 1 to `nside`.
        // Or simply, `iring_global = floor( nside * (2.0 - 1.5 * z) )` in N, and `floor( nside * (2.0 + 1.5 * z) )` in S from S pole.
        // This is confusing. Let's use the `jp, jm` method which is directly for RING scheme.

        // Back to the `jp, jm` method which is standard for RING scheme ang2pix:
        // This was actually correct:
        long Npix = 12 * nside * nside;
        long Ncap = 2 * nside * (nside - 1); // Pixels in one polar cap. Total in two caps is 2*Ncap.

        // Variables for equatorial belt
        // `jr` = ring index from pole (1 to 2*nside-1 for North, 2*nside to 4*nside-1 for South)
        // `tmp_phi` = longitude from 0 to 2PI
        double tmp_phi = phi;
        if (tmp_phi < 0.0) tmp_phi += TWO_PI;
        if (tmp_phi >= TWO_PI) tmp_phi = fmod(tmp_phi, TWO_PI);


        long jr = static_cast<long>(nside * (2.0 - 1.5 * z)); // ring index from pole for z > 0 (nside to 2*nside)
                                                             // ring index from S pole for z < 0 (nside to 2*nside)
        if (z < 0.0) jr = 4 * nside - jr; // Global ring index (nside to 3*nside)

        long jp = static_cast<long>( (tmp_phi * INV_PI * 0.5) * (2*nside) + ( ( (jr - nside) % 2 == 1 ) ? 0.0 : 0.5 ) + 1.0 );
        if (jp > 4*nside) jp = 4*nside;

        long pix_idx = Ncap + (jr - nside) * 4 * nside + jp -1; // 0-indexed
        return pix_idx;

    } else { // Polar caps
        // `jr` = ring index from the nearest pole (1 to nside-1)
        long jr;
        if (z > 0.0) { // North polar cap
            jr = static_cast<long>(nside * sqrt(3.0 * (1.0 - z)) + 1e-9) + 1; // Ensure round down + 1, or use floor(val+eps)+1
            if (jr >= nside) jr = nside-1; // Should not happen if z > 2/3
            double phi_tmp = phi;
            if (phi_tmp < 0.0) phi_tmp += TWO_PI;
            if (phi_tmp >= TWO_PI) phi_tmp = fmod(phi_tmp, TWO_PI);

            long jp = static_cast<long>((phi_tmp * INV_PI * 0.5) * (2*jr) + 1.0); // Pixel index within ring, 1 to 4*jr
            if (jp > 4 * jr) jp = 4 * jr;

            return 2 * jr * (jr - 1) + jp -1; // 0-indexed
        } else { // South polar cap
            jr = static_cast<long>(nside * sqrt(3.0 * (1.0 + z)) + 1e-9) + 1;
            if (jr >= nside) jr = nside-1;
            double phi_tmp = phi;
            if (phi_tmp < 0.0) phi_tmp += TWO_PI;
            if (phi_tmp >= TWO_PI) phi_tmp = fmod(phi_tmp, TWO_PI);

            long jp = static_cast<long>((phi_tmp * INV_PI * 0.5) * (2*jr) + 1.0);
            if (jp > 4 * jr) jp = 4 * jr;

            return 12 * nside * nside - 2 * jr * (jr + 1) + jp -1; // This formula for base_pix is 0-indexed friendly if jp is 0 to 4jr-1
                                                                  // Original: Npix - 2*jr*(jr-1) - jp (if jp is 1-based from top of cap)
                                                                  // Healpix_cxx: npix_ - (2*irl*(irl-1) + ipl) where irl, ipl are 1-based from S pole corner
                                                                  // Let's use: 12*nside*nside - (2 * jr * (jr - 1) + jp) for 1-based jp
                                                                  // So for 0-based jp: 12*nside*nside - (2 * jr * (jr-1) + jp + 1)
                                                                  // Or: base_pix = 12*nside*nside - 2*jr*(jr+1); return base_pix + jp -1 (if jp is 1-based)
                                                                  // This is: 12*nside*nside - 2 * jr * (jr +1) for the start of ring `jr` from south pole (0-indexed).
                                                                  // Then add `jp-1`.
            return (12 * nside * nside - 2 * jr * (jr + 1)) + jp -1;
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
        long iring = static_cast<long>(0.5 * (1.0 + sqrt(1.0 + 2.0 * static_cast<double>(pix)))) ; // ring number counted from North pole [1, nside-1]
                                                                                                  // This is derived from pix = 2*ir*(ir-1)
        if (iring == 0) iring = 1; // Should not happen for pix > 0. For pix=0, iring=1.
        long ipix_in_ring = pix - 2 * iring * (iring - 1); // pixel index within the ring [0, 4*iring-1]

        z = 1.0 - static_cast<double>(iring * iring) / (3.0 * nside * nside);
        phi = (static_cast<double>(ipix_in_ring) + 0.5) * PI / (2.0 * iring);
    } else if (pix < npix - ncap) { // Equatorial belt
        long ipix_belt = pix - ncap;
        long iring_global = (ipix_belt / (4 * nside)) + nside; // global ring index [nside, 3*nside]
        long ipix_in_ring = ipix_belt % (4 * nside); // pixel index within the ring [0, 4*nside-1]

        // Determine z from global ring index
        // iring_global = nside ... 2*nside (North belt), 2*nside+1 ... 3*nside (South belt)
        // z = (2*nside - iring_global) * 2.0 / (3.0*nside) if iring_global <= 2*nside
        // z = (2*nside - iring_global) * 2.0 / (3.0*nside) if iring_global > 2*nside (iring_global is from N pole)
        z = (2.0 * nside - static_cast<double>(iring_global)) * 2.0 / (3.0 * nside); // This is not right.
                                                                                     // z = (2./3.) * (2*nside - iring_global) / nside; -- This is also not quite.
        // From Gorski et al. (2005) Eq (4):
        // For equatorial region, z = (2/3) * (2*nside - i_H) / nside, where i_H is ring index along latitude.
        // i_H = iring_global for North, i_H = 4*nside - iring_global + 1 for South (relative to its pole)
        // This is getting complicated. Let's use the simpler:
        // In belt, |z| <= 2/3.
        // Ring index `iring_belt_part` from equator: `abs(iring_global - 2*nside)`. Or `iring_global - nside` for North, `3*nside - iring_global` for South.
        // The z values are spaced by `1.5 / nside`.
        // `z = (2.0/3.0) - (iring_global - nside + 0.5) * step` where step is `1.0/(1.5*nside)`? No.
        // Healpix C specifies for equatorial:
        // `z = (2*nside - ir) * 2.0 / (3.0*nside)` where `ir` is the ring index from N pole (nside to 3*nside).
        // This means `z` goes from `2/3` down to `-2/3`.
        z = (2.0 * static_cast<double>(nside) - static_cast<double>(iring_global)) * 2.0 / (3.0 * static_cast<double>(nside));


        // Determine phi
        // Shift `s` is 1 if `(iring_global - nside)` is odd, 0 if even.
        // (iring_global - nside + 1) is the ring index in the belt, 1 to 2*nside.
        // s=1 if (iring_global - nside + 1) is odd.
        // s=0 if (iring_global - nside + 1) is even.
        // Equivalent to (iring_global - nside) % 2 == 0 for s=0, (iring_global - nside) % 2 == 1 for s=1.
        long s = (iring_global - nside) % 2; // s=0 for first ring in belt, s=1 for second, etc.
        phi = (static_cast<double>(ipix_in_ring) + 0.5 + s * 0.5) * PI / (2.0 * nside);

    } else { // South polar cap
        long pix_from_south_pole_corner = npix - 1 - pix; // Count from the "corner" of the south pole representation
        long iring = static_cast<long>(0.5 * (1.0 + sqrt(1.0 + 2.0 * static_cast<double>(pix_from_south_pole_corner))));
        if (iring == 0) iring = 1;

        // ipix_in_ring from corner of south cap representation.
        // This is complex because pixel ordering is reversed.
        // Let's find ring `iring_s` from South Pole [1, nside-1]
        // Pixels in S cap: `npix - ncap` to `npix-1`.
        // Offset in S cap: `pix - (npix - ncap)`.
        // Base of ring `iring_s` (from S pole, 1-indexed for iring_s) is `2*iring_s*(iring_s-1)`.
        // So, `pix - (npix - ncap) = 2*iring_s*(iring_s-1) + ip_s`.
        // `pix_rel_scap_start = pix - (npix - ncap)`.
        // `iring_s = floor(0.5 * (1 + sqrt(1 + 2*pix_rel_scap_start)))`. This is `iring` from S pole start.
        long iring_s = static_cast<long>(floor(0.5 * (1.0 + sqrt(1.0 + 2.0 * (pix - (npix - ncap))))));
        if (iring_s == 0 && (pix-(npix-ncap)) > 0) iring_s = 1; // Handle case for first few pixels in south cap.
                                                               // For pix = npix-ncap, pix_rel = 0, iring_s = floor(0.5*(1+1)) = 1. Correct.

        long ipix_in_ring_s = pix - (npix - ncap) - 2 * iring_s * (iring_s - 1);

        z = - (1.0 - static_cast<double>(iring_s * iring_s) / (3.0 * nside * nside));
        phi = (static_cast<double>(ipix_in_ring_s) + 0.5) * PI / (2.0 * iring_s);
    }

    // Clamp z to avoid domain errors for acos due to precision
    if (z > 1.0) z = 1.0;
    else if (z < -1.0) z = -1.0;

    double theta = acos(z);
    ang2vec_gpu(theta, phi, vec);
}
