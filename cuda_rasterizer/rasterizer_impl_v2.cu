#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace py = pybind11;
using namespace pybind11::literals;

#define EPSILON 1e-6f
#define PI 3.14159265358979323846f
#define SH_DEGREE 3
#define SH_COEFFS_SIZE 16

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f
#define SH_C2 1.0925484305920792f
#define SH_C20 0.31539156525252005f
#define SH_C22 0.5462742152960396f
#define SH_C3 0.5900435899266435f
#define SH_C30 0.4570459564643759f
#define SH_C31 2.890611442640554f
#define SH_C32 0.3731763325901154f
#define SH_C33 1.4453057113204849f

__device__ float compute_sh_scattering(float theta, float phi, const float* sh_coeffs) {
    float cos_theta = __cosf(theta);
    float sin_theta = __sinf(theta);
    float cos_phi = __cosf(phi);
    float sin_phi = __sinf(phi);

    float sin_theta_sq = sin_theta * sin_theta;
    float cos_theta_sq = cos_theta * cos_theta;
    float sin_cos_theta = sin_theta * cos_theta;
    float sin_cos_phi = sin_theta * cos_phi;

    float sh_basis[16];

    sh_basis[0] = SH_C0;

    sh_basis[1] = SH_C1 * sin_theta * sin_phi;
    sh_basis[2] = SH_C1 * sin_theta * cos_phi;
    sh_basis[3] = SH_C1 * cos_theta;

    sh_basis[4] = SH_C2 * sin_cos_phi * sin_phi;
    sh_basis[5] = SH_C2 * sin_cos_theta * cos_phi;
    sh_basis[6] = SH_C20 * (3.0f * cos_theta_sq - 1.0f);
    sh_basis[7] = -SH_C2 * sin_cos_theta * sin_phi;
    sh_basis[8] = -SH_C2 * sin_cos_theta * cos_phi;

    float three_cos_theta_sq_minus_one = 3.0f * cos_theta_sq - 1.0f;
    float five_cos_theta_cu_minus_three = (5.0f * cos_theta_sq - 3.0f) * cos_theta;

    sh_basis[9] = SH_C22 * sin_theta * sin_phi * three_cos_theta_sq_minus_one;
    sh_basis[10] = SH_C22 * sin_cos_phi * three_cos_theta_sq_minus_one;
    sh_basis[11] = SH_C22 * cos_theta * three_cos_theta_sq_minus_one;
    sh_basis[12] = SH_C3 * sin_theta * sin_phi * five_cos_theta_cu_minus_three;
    sh_basis[13] = SH_C3 * sin_cos_phi * five_cos_theta_cu_minus_three;

    float fifteen_cos_theta_sq_sq_minus_six_cos_theta_sq_minus_one =
        (15.0f * cos_theta_sq - 6.0f) * cos_theta_sq - 1.0f;
    sh_basis[14] = SH_C30 * fifteen_cos_theta_sq_sq_minus_six_cos_theta_sq_minus_one;
    sh_basis[15] = -SH_C3 * sin_theta * sin_phi * five_cos_theta_cu_minus_three;

    float scattering = sh_coeffs[0] * sh_basis[0];
    #pragma unroll
    for (int i = 1; i < SH_COEFFS_SIZE; i++) {
        scattering = fmaf(sh_coeffs[i], sh_basis[i], scattering);
    }
    return scattering;
}

__device__ void compute_view_direction(float beta, float alpha, float& vx, float& vy, float& vz) {
    vx = sinf(beta) * cosf(alpha);
    vy = sinf(beta) * sinf(alpha);
    vz = -cosf(beta);
}

struct SARRenderParams {
    int n_gaussians;
    float* means_world;
    float* cov_world;
    float* sh_coeffs;
    float* transmittance;
    float radar_x, radar_y, radar_z;
    float track_angle;
    float incidence_angle;
    float azimuth_angle;
    float range_resolution;
    float azimuth_resolution;
    int range_samples;
    int azimuth_samples;
    bool prefiltered;
    bool debug;
};

__device__ float3 world_to_radar(float3 world_point, float3 radar_pos, float alpha, float beta) {
    float dx = world_point.x - radar_pos.x;
    float dy = world_point.y - radar_pos.y;
    float dz = world_point.z - radar_pos.z;

    float cos_a = cosf(alpha);
    float sin_a = sinf(alpha);
    float cos_b = cosf(beta);
    float sin_b = sinf(beta);

    float xr = cos_a * dx + sin_a * dy;
    float yr = cos_b * sin_a * dx - cos_b * cos_a * dy - sin_b * dz;
    float zr = -sin_b * sin_a * dx + sin_b * cos_a * dy - cos_b * dz;

    return make_float3(xr, yr, zr);
}

__device__ void transform_covariance_3x3(
    const float* cov_world,
    float alpha, float beta,
    float& cov_XrXr, float& cov_YrYr, float& cov_ZrZr,
    float& cov_XrYr, float& cov_XrZr, float& cov_YrZr
) {
    float cov_xx = cov_world[0];
    float cov_yy = cov_world[1];
    float cov_zz = cov_world[2];
    float cov_xy = cov_world[3];
    float cov_xz = cov_world[4];
    float cov_yz = cov_world[5];

    float cos_a = cosf(alpha);
    float sin_a = sinf(alpha);
    float cos_b = cosf(beta);
    float sin_b = sinf(beta);

    float R00 = cos_a;
    float R01 = sin_a;
    float R02 = 0.0f;

    float R10 = cos_b * sin_a;
    float R11 = -cos_b * cos_a;
    float R12 = -sin_b;

    float R20 = -sin_b * sin_a;
    float R21 = sin_b * cos_a;
    float R22 = -cos_b;

    cov_XrXr = R00*R00*cov_xx + R01*R01*cov_yy + R02*R02*cov_zz
             + 2.0f*R00*R01*cov_xy + 2.0f*R00*R02*cov_xz + 2.0f*R01*R02*cov_yz;
    cov_YrYr = R10*R10*cov_xx + R11*R11*cov_yy + R12*R12*cov_zz
             + 2.0f*R10*R11*cov_xy + 2.0f*R10*R12*cov_xz + 2.0f*R11*R12*cov_yz;
    cov_ZrZr = R20*R20*cov_xx + R21*R21*cov_yy + R22*R22*cov_zz
             + 2.0f*R20*R21*cov_xy + 2.0f*R20*R22*cov_xz + 2.0f*R21*R22*cov_yz;

    cov_XrYr = R00*R10*cov_xx + R01*R11*cov_yy + R02*R12*cov_zz
             + (R00*R11+R01*R10)*cov_xy + (R00*R12+R02*R10)*cov_xz + (R01*R12+R02*R11)*cov_yz;
    cov_XrZr = R00*R20*cov_xx + R01*R21*cov_yy + R02*R22*cov_zz
             + (R00*R21+R01*R20)*cov_xy + (R00*R22+R02*R20)*cov_xz + (R01*R22+R02*R21)*cov_yz;
    cov_YrZr = R10*R20*cov_xx + R11*R21*cov_yy + R12*R22*cov_zz
             + (R10*R21+R11*R20)*cov_xy + (R10*R22+R12*R20)*cov_xz + (R11*R22+R12*R21)*cov_yz;
}

__device__ void project_to_ipp(
    float xr, float yr, float zr,
    float Rc, float rho_r, float rho_a,
    int Nr, int Na,
    float& r_coord, float& c_coord,
    float& Rmin, float& dydr, float& dzdr
) {
    Rmin = sqrtf(yr * yr + zr * zr + EPSILON * EPSILON);

    r_coord = Rmin / rho_r + (float)Nr / 2.0f - Rc / rho_r;
    c_coord = xr / rho_a + (float)Na / 2.0f;

    dydr = yr / (rho_r * Rmin);
    dzdr = zr / (rho_r * Rmin);
}

__device__ void compute_jacobian_ipp(
    float yr, float zr, float Rmin, float rho_r, float rho_a,
    float& J00, float& J01, float& J02,
    float& J10, float& J11, float& J12
) {
    J00 = 0.0f;
    J01 = yr / (rho_r * Rmin);
    J02 = zr / (rho_r * Rmin);

    J10 = 1.0f / rho_a;
    J11 = 0.0f;
    J12 = 0.0f;
}

__device__ void project_covariance_ipp(
    float cov_XrXr, float cov_YrYr, float cov_ZrZr,
    float cov_XrYr, float cov_XrZr, float cov_YrZr,
    float J00, float J01, float J02,
    float J10, float J11, float J12,
    float& cov_rr, float& cov_cc, float& cov_rc
) {
    cov_rr = J00*J00*cov_XrXr + J01*J01*cov_YrYr + J02*J02*cov_ZrZr
           + 2.0f*J00*J01*cov_XrYr + 2.0f*J00*J02*cov_XrZr + 2.0f*J01*J02*cov_YrZr;
    cov_cc = J10*J10*cov_XrXr + J11*J11*cov_YrYr + J12*J12*cov_ZrZr
           + 2.0f*J10*J11*cov_XrYr + 2.0f*J10*J12*cov_XrZr + 2.0f*J11*J12*cov_YrZr;
    cov_rc = J00*J10*cov_XrXr + J01*J11*cov_YrYr + J02*J12*cov_ZrZr
           + (J00*J11+J01*J10)*cov_XrYr + (J00*J12+J02*J10)*cov_XrZr + (J01*J12+J02*J11)*cov_YrZr;
}

__device__ void project_covariance_shadow(
    float cov_XrXr, float cov_YrYr, float cov_ZrZr,
    float cov_XrYr, float cov_XrZr, float cov_YrZr,
    float& cov_xx, float& cov_yy, float& cov_xy
) {
    cov_xx = cov_XrXr;
    cov_yy = cov_YrYr;
    cov_xy = cov_XrYr;
}

__device__ float compute_gaussian_density_2d(
    float px, float py,
    float mux, float muy,
    float cov_xx, float cov_xy, float cov_yy
) {
    float dx = px - mux;
    float dy = py - muy;

    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (fabsf(det) < EPSILON) return 0.0f;

    float inv_det = 1.0f / det;
    float exponent = -0.5f * (cov_yy * dx * dx - 2.0f * cov_xy * dx * dy + cov_xx * dy * dy) * inv_det;

    if (exponent < -20.0f) return 0.0f;

    float result = expf(exponent) / (2.0f * PI * sqrtf(det));

    return fmaxf(result, 0.0f);
}

__global__ void preprocess_kernel(
    const float* __restrict__ means_world,
    const float* __restrict__ cov_world,
    const float* __restrict__ transmittance,
    const float* __restrict__ sh_coeffs,
    float radar_x, float radar_y, float radar_z,
    float alpha, float beta, float Rc,
    float rho_r, float rho_a,
    int Nr, int Na,
    float* __restrict__ ipp_r,
    float* __restrict__ ipp_c,
    float* __restrict__ cov_ipp_rr,
    float* __restrict__ cov_ipp_cc,
    float* __restrict__ cov_ipp_rc,
    float* __restrict__ shadow_x,
    float* __restrict__ shadow_y,
    float* __restrict__ cov_shadow_xx,
    float* __restrict__ cov_shadow_yy,
    float* __restrict__ cov_shadow_xy,
    float* __restrict__ gaussian_yr,
    float* __restrict__ gaussian_z,
    float* __restrict__ scattering,
    bool* __restrict__ valid_mask,
    int n_gaussians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gaussians) return;

    if (__ldg(&transmittance[idx]) < 0.01f) {
        valid_mask[idx] = false;
        return;
    }

    float3 radar_pos = make_float3(radar_x, radar_y, radar_z);
    float3 point_world = make_float3(
        __ldg(&means_world[idx * 3 + 0]),
        __ldg(&means_world[idx * 3 + 1]),
        __ldg(&means_world[idx * 3 + 2])
    );

    float3 point_radar = world_to_radar(point_world, radar_pos, alpha, beta);
    float xr = point_radar.x;
    float yr = point_radar.y;
    float zr = point_radar.z;

    gaussian_yr[idx] = yr;
    gaussian_z[idx] = zr;

    float r_coord, c_coord, Rmin, dydr, dzdr;
    project_to_ipp(xr, yr, zr, Rc, rho_r, rho_a, Nr, Na, r_coord, c_coord, Rmin, dydr, dzdr);

    ipp_r[idx] = r_coord;
    ipp_c[idx] = c_coord;

    shadow_x[idx] = xr;
    shadow_y[idx] = yr;

    float cov_XrXr, cov_YrYr, cov_ZrZr, cov_XrYr, cov_XrZr, cov_YrZr;
    transform_covariance_3x3(&cov_world[idx * 6], alpha, beta,
                            cov_XrXr, cov_YrYr, cov_ZrZr,
                            cov_XrYr, cov_XrZr, cov_YrZr);

    float J00, J01, J02, J10, J11, J12;
    compute_jacobian_ipp(yr, zr, Rmin, rho_r, rho_a, J00, J01, J02, J10, J11, J12);

    float cov_rr, cov_cc, cov_rc;
    project_covariance_ipp(cov_XrXr, cov_YrYr, cov_ZrZr, cov_XrYr, cov_XrZr, cov_YrZr,
                          J00, J01, J02, J10, J11, J12,
                          cov_rr, cov_cc, cov_rc);

    cov_ipp_rr[idx] = cov_rr;
    cov_ipp_cc[idx] = cov_cc;
    cov_ipp_rc[idx] = cov_rc;

    float shadow_cov_xx, shadow_cov_yy, shadow_cov_xy;
    project_covariance_shadow(cov_XrXr, cov_YrYr, cov_ZrZr, cov_XrYr, cov_XrZr, cov_YrZr,
                             shadow_cov_xx, shadow_cov_yy, shadow_cov_xy);

    cov_shadow_xx[idx] = shadow_cov_xx;
    cov_shadow_yy[idx] = shadow_cov_yy;
    cov_shadow_xy[idx] = shadow_cov_xy;

    bool valid = (r_coord >= 0 && r_coord < (float)Nr &&
                  c_coord >= 0 && c_coord < (float)Na);
    valid_mask[idx] = valid;

    float theta = beta;
    float phi = alpha;
    if (phi < 0.0f) phi += 2.0f * PI;

    const float* sh_coeff_i = &sh_coeffs[idx * SH_COEFFS_SIZE];
    scattering[idx] = compute_sh_scattering(theta, phi, sh_coeff_i);
}

__global__ void compute_ranges_kernel(
    const float* __restrict__ ipp_r,
    const float* __restrict__ ipp_c,
    const float* __restrict__ cov_ipp_rr,
    const float* __restrict__ cov_ipp_cc,
    const float* __restrict__ cov_ipp_rc,
    const bool* __restrict__ valid_mask,
    int Nr, int Na,
    int* __restrict__ range_min,
    int* __restrict__ range_max,
    int* __restrict__ azi_min,
    int* __restrict__ azi_max,
    int n_gaussians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gaussians) return;

    bool valid = static_cast<bool>(__ldg(reinterpret_cast<const unsigned char*>(&valid_mask[idx])));
    if (!valid) {
        range_min[idx] = 0;
        range_max[idx] = 0;
        azi_min[idx] = 0;
        azi_max[idx] = 0;
        return;
    }

    float r_center = __ldg(&ipp_r[idx]);
    float c_center = __ldg(&ipp_c[idx]);

    float cov_rr = __ldg(&cov_ipp_rr[idx]);
    float cov_cc = __ldg(&cov_ipp_cc[idx]);
    float cov_rc = __ldg(&cov_ipp_rc[idx]);

    float det = cov_rr * cov_cc - cov_rc * cov_rc;
    if (det < EPSILON) det = EPSILON;
    float inv_det = 1.0f / det;

    float std_r = sqrtf(fmaxf(cov_rr * cov_cc * inv_det, EPSILON));
    float std_c = sqrtf(fmaxf(cov_cc * cov_rr * inv_det, EPSILON));

    float range_half = 3.0f * std_r;
    float azi_half = 3.0f * std_c;

    int r_min = (int)floorf(r_center - range_half);
    int r_max = (int)ceilf(r_center + range_half);
    int c_min = (int)floorf(c_center - azi_half);
    int c_max = (int)ceilf(c_center + azi_half);

    r_min = max(0, r_min);
    r_max = min(Nr - 1, r_max);
    c_min = max(0, c_min);
    c_max = min(Na - 1, c_max);

    range_min[idx] = r_min;
    range_max[idx] = r_max;
    azi_min[idx] = c_min;
    azi_max[idx] = c_max;
}

#define BLOCK_X 16
#define BLOCK_Y 16
#define MAX_GAUSSIANS_PER_TILE 64

struct SARRasterizationSettings {
    int n_gaussians;
    float radar_x, radar_y, radar_z;
    float track_angle;
    float incidence_angle;
    float azimuth_angle;
    float range_resolution;
    float azimuth_resolution;
    int range_samples;
    int azimuth_samples;
    bool prefiltered;
    bool debug;
};

__global__ void build_tile_gaussian_list_kernel(
    const float* __restrict__ ipp_r,
    const float* __restrict__ ipp_c,
    const int* __restrict__ range_min,
    const int* __restrict__ range_max,
    const int* __restrict__ azi_min,
    const int* __restrict__ azi_max,
    const float* __restrict__ gaussian_z,
    const bool* __restrict__ valid_mask,
    int tiles_x, int tiles_y,
    int Nr, int Na,
    int* __restrict__ tile_gaussian_count,
    int* __restrict__ tile_gaussian_list,
    int n_gaussians
) {
    int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    if (tile_id >= tiles_x * tiles_y) return;

    int tile_c = tile_id % tiles_x;
    int tile_r = tile_id / tiles_x;

    int pix_min_c = tile_c * BLOCK_X;
    int pix_min_r = tile_r * BLOCK_Y;
    int pix_max_c = min(pix_min_c + BLOCK_X, Na);
    int pix_max_r = min(pix_min_r + BLOCK_Y, Nr);

    int count = 0;
    int list_offset = tile_id * MAX_GAUSSIANS_PER_TILE;

    for (int i = 0; i < n_gaussians; i++) {
        bool is_valid = static_cast<bool>(__ldg(reinterpret_cast<const unsigned char*>(&valid_mask[i])));
        if (!is_valid) continue;

        int gauss_r_min = __ldg(&range_min[i]);
        int gauss_r_max = __ldg(&range_max[i]);
        int gauss_c_min = __ldg(&azi_min[i]);
        int gauss_c_max = __ldg(&azi_max[i]);

        bool intersects = !(gauss_r_max < pix_min_r || gauss_r_min > pix_max_r ||
                            gauss_c_max < pix_min_c || gauss_c_min > pix_max_c);

        if (intersects && count < MAX_GAUSSIANS_PER_TILE) {
            tile_gaussian_list[list_offset + count] = i;
            count++;
        }
    }

    tile_gaussian_count[tile_id] = count;
}

__global__ void build_shadow_tile_list_kernel(
    const float* __restrict__ shadow_x,
    const float* __restrict__ shadow_y,
    const float* __restrict__ cov_shadow_xx,
    const float* __restrict__ cov_shadow_yy,
    const float* __restrict__ cov_shadow_xy,
    const float* __restrict__ gaussian_z,
    const bool* __restrict__ valid_mask,
    int tiles_x, int tiles_y,
    int Nr, int Na,
    int* __restrict__ shadow_tile_gaussian_count,
    int* __restrict__ shadow_tile_gaussian_list,
    int n_gaussians
) {
    int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    if (tile_id >= tiles_x * tiles_y) return;

    int tile_c = tile_id % tiles_x;
    int tile_r = tile_id / tiles_x;

    int pix_min_c = tile_c * BLOCK_X;
    int pix_min_r = tile_r * BLOCK_Y;
    int pix_max_c = min(pix_min_c + BLOCK_X, Na);
    int pix_max_r = min(pix_min_r + BLOCK_Y, Nr);

    int count = 0;
    int list_offset = tile_id * MAX_GAUSSIANS_PER_TILE;

    for (int i = 0; i < n_gaussians; i++) {
        bool is_valid = static_cast<bool>(__ldg(reinterpret_cast<const unsigned char*>(&valid_mask[i])));
        if (!is_valid) continue;

        float mux = __ldg(&shadow_x[i]);
        float muy = __ldg(&shadow_y[i]);
        float cov_xx = __ldg(&cov_shadow_xx[i]);
        float cov_yy = __ldg(&cov_shadow_yy[i]);
        float cov_xy = __ldg(&cov_shadow_xy[i]);

        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (fabsf(det) < EPSILON) continue;

        float inv_det = 1.0f / det;

        float std_x = sqrtf(fmaxf(cov_xx * cov_yy * inv_det, EPSILON));
        float std_y = sqrtf(fmaxf(cov_xx * cov_yy * inv_det, EPSILON));

        float x_half = 3.0f * std_x;
        float y_half = 3.0f * std_y;

        int gauss_c_min = (int)floorf(mux - x_half);
        int gauss_c_max = (int)ceilf(mux + x_half);
        int gauss_r_min = (int)floorf(muy - y_half);
        int gauss_r_max = (int)ceilf(muy + y_half);

        gauss_c_min = max(0, gauss_c_min);
        gauss_c_max = min(Na - 1, gauss_c_max);
        gauss_r_min = max(0, gauss_r_min);
        gauss_r_max = min(Nr - 1, gauss_r_max);

        bool intersects = !(gauss_r_max < pix_min_r || gauss_r_min > pix_max_r ||
                            gauss_c_max < pix_min_c || gauss_c_min > pix_max_c);

        if (intersects && count < MAX_GAUSSIANS_PER_TILE) {
            shadow_tile_gaussian_list[list_offset + count] = i;
            count++;
        }
    }

    shadow_tile_gaussian_count[tile_id] = count;
}

__global__ void compute_omega_kernel(
    const float* __restrict__ shadow_x,
    const float* __restrict__ shadow_y,
    const float* __restrict__ cov_shadow_xx,
    const float* __restrict__ cov_shadow_yy,
    const float* __restrict__ cov_shadow_xy,
    const float* __restrict__ gaussian_z,
    const float* __restrict__ scattering,
    const float* __restrict__ transmittance,
    const int* __restrict__ shadow_tile_gaussian_count,
    const int* __restrict__ shadow_tile_gaussian_list,
    int tiles_x, int tiles_y,
    int Nr, int Na,
    float* __restrict__ omega_sum,
    int* __restrict__ omega_count,
    int n_gaussians
) {
    __shared__ float shared_shadow_x[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_shadow_y[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_xx[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_yy[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_xy[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_gaussian_z[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_scattering[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_transmittance[MAX_GAUSSIANS_PER_TILE];
    __shared__ int shared_tile_count;
    __shared__ int shared_indices[MAX_GAUSSIANS_PER_TILE];

    int pix_c = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_r = blockIdx.y * blockDim.y + threadIdx.y;

    if (pix_c >= Na || pix_r >= Nr) return;

    int tile_c = pix_c / BLOCK_X;
    int tile_r = pix_r / BLOCK_Y;
    int tile_id = tile_r * tiles_x + tile_c;

    int local_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (local_thread_idx == 0) {
        shared_tile_count = __ldg(&shadow_tile_gaussian_count[tile_id]);
    }
    __syncthreads();

    int gaussian_count = shared_tile_count;
    int list_offset = tile_id * MAX_GAUSSIANS_PER_TILE;

    if (gaussian_count == 0) return;

    if (local_thread_idx < gaussian_count) {
        int idx = __ldg(&shadow_tile_gaussian_list[list_offset + local_thread_idx]);
        shared_indices[local_thread_idx] = idx;
        shared_shadow_x[local_thread_idx] = __ldg(&shadow_x[idx]);
        shared_shadow_y[local_thread_idx] = __ldg(&shadow_y[idx]);
        shared_cov_xx[local_thread_idx] = __ldg(&cov_shadow_xx[idx]);
        shared_cov_yy[local_thread_idx] = __ldg(&cov_shadow_yy[idx]);
        shared_cov_xy[local_thread_idx] = __ldg(&cov_shadow_xy[idx]);
        shared_gaussian_z[local_thread_idx] = __ldg(&gaussian_z[idx]);
        shared_scattering[local_thread_idx] = __ldg(&scattering[idx]);
        shared_transmittance[local_thread_idx] = __ldg(&transmittance[idx]);
    }
    __syncthreads();

    int local_indices[MAX_GAUSSIANS_PER_TILE];
    float z_values[MAX_GAUSSIANS_PER_TILE];

    for (int i = 0; i < gaussian_count; i++) {
        local_indices[i] = shared_indices[i];
        z_values[i] = shared_gaussian_z[i];
    }

    for (int i = 1; i < MAX_GAUSSIANS_PER_TILE; i++) {
        if (i >= gaussian_count) break;
        int temp_idx = local_indices[i];
        float temp_z = z_values[i];
        int j = i - 1;
        while (j >= 0 && z_values[j] > temp_z) {
            local_indices[j + 1] = local_indices[j];
            z_values[j + 1] = z_values[j];
            j--;
        }
        local_indices[j + 1] = temp_idx;
        z_values[j + 1] = temp_z;
    }

    float omega = 1.0f;

    for (int k = 0; k < gaussian_count; k++) {
        int i = local_indices[k];

        float gamma_shadow = compute_gaussian_density_2d(
            (float)pix_c, (float)pix_r,
            shared_shadow_x[i], shared_shadow_y[i],
            shared_cov_xx[i], shared_cov_xy[i], shared_cov_yy[i]
        );

        float sigma_i = shared_transmittance[i];

        atomicAdd(&omega_sum[i], omega);
        atomicAdd(&omega_count[i], 1);

        omega *= (1.0f - gamma_shadow * sigma_i);

        if (omega < 1e-8f) {
            omega = 0.0f;
            break;
        }
    }
}

__global__ void render_kernel(
    const float* __restrict__ ipp_r,
    const float* __restrict__ ipp_c,
    const float* __restrict__ cov_ipp_rr,
    const float* __restrict__ cov_ipp_cc,
    const float* __restrict__ cov_ipp_rc,
    const float* __restrict__ gaussian_z,
    const float* __restrict__ scattering,
    const float* __restrict__ transmittance,
    const float* __restrict__ omega,
    const bool* __restrict__ valid_mask,
    const int* __restrict__ tile_gaussian_count,
    const int* __restrict__ tile_gaussian_list,
    int tiles_x, int tiles_y,
    int Nr, int Na,
    float* __restrict__ output_image
) {
    __shared__ float shared_ipp_r[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_ipp_c[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_rr[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_cc[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_cov_rc[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_gaussian_z[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_scattering[MAX_GAUSSIANS_PER_TILE];
    __shared__ float shared_transmittance[MAX_GAUSSIANS_PER_TILE];
    __shared__ int shared_tile_count;
    __shared__ int shared_indices[MAX_GAUSSIANS_PER_TILE];

    int pix_c = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_r = blockIdx.y * blockDim.y + threadIdx.y;

    if (pix_c >= Na || pix_r >= Nr) return;

    int tile_c = pix_c / BLOCK_X;
    int tile_r = pix_r / BLOCK_Y;
    int tile_id = tile_r * tiles_x + tile_c;

    int local_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if (local_thread_idx == 0) {
        shared_tile_count = __ldg(&tile_gaussian_count[tile_id]);
    }
    __syncthreads();

    int gaussian_count = shared_tile_count;
    int list_offset = tile_id * MAX_GAUSSIANS_PER_TILE;

    if (gaussian_count == 0) {
        output_image[pix_c * Nr + pix_r] = 0.0f;
        return;
    }

    if (local_thread_idx < gaussian_count) {
        int idx = __ldg(&tile_gaussian_list[list_offset + local_thread_idx]);
        shared_indices[local_thread_idx] = idx;
        shared_ipp_r[local_thread_idx] = __ldg(&ipp_r[idx]);
        shared_ipp_c[local_thread_idx] = __ldg(&ipp_c[idx]);
        shared_cov_rr[local_thread_idx] = __ldg(&cov_ipp_rr[idx]);
        shared_cov_cc[local_thread_idx] = __ldg(&cov_ipp_cc[idx]);
        shared_cov_rc[local_thread_idx] = __ldg(&cov_ipp_rc[idx]);
        shared_gaussian_z[local_thread_idx] = __ldg(&gaussian_z[idx]);
        shared_scattering[local_thread_idx] = __ldg(&scattering[idx]);
        shared_transmittance[local_thread_idx] = __ldg(&transmittance[idx]);
    }
    __syncthreads();

    int local_indices[MAX_GAUSSIANS_PER_TILE];
    float z_values[MAX_GAUSSIANS_PER_TILE];

    for (int i = 0; i < gaussian_count; i++) {
        local_indices[i] = shared_indices[i];
        z_values[i] = shared_gaussian_z[i];
    }

    for (int i = 1; i < MAX_GAUSSIANS_PER_TILE; i++) {
        if (i >= gaussian_count) break;
        int temp_idx = local_indices[i];
        float temp_z = z_values[i];
        int j = i - 1;
        while (j >= 0 && z_values[j] > temp_z) {
            local_indices[j + 1] = local_indices[j];
            z_values[j + 1] = z_values[j];
            j--;
        }
        local_indices[j + 1] = temp_idx;
        z_values[j + 1] = temp_z;
    }

    float pixel_value = 0.0f;
    float transmittance_acc = 1.0f;

    for (int k = 0; k < gaussian_count; k++) {
        int i = local_indices[k];

        float gamma_ipp = compute_gaussian_density_2d(
            (float)pix_c, (float)pix_r,
            shared_ipp_c[i], shared_ipp_r[i],
            shared_cov_cc[i], shared_cov_rc[i], shared_cov_rr[i]
        );

        if (gamma_ipp < 1e-8f) continue;

        float sigma_i = shared_transmittance[i];
        float S_i = fmaxf(0.0f, shared_scattering[i]);

        float alpha_i = sigma_i * gamma_ipp;

        pixel_value += transmittance_acc * alpha_i * S_i;

        transmittance_acc *= (1.0f - alpha_i);

        if (transmittance_acc < 1e-8f) {
            transmittance_acc = 0.0f;
            break;
        }
    }

    output_image[pix_c * Nr + pix_r] = pixel_value;
}

__global__ void backward_kernel(
    const float* __restrict__ ipp_r,
    const float* __restrict__ ipp_c,
    const float* __restrict__ cov_ipp_rr,
    const float* __restrict__ cov_ipp_cc,
    const float* __restrict__ cov_ipp_rc,
    const float* __restrict__ gaussian_yr,
    const float* __restrict__ gaussian_z,
    const float* __restrict__ scattering,
    const float* __restrict__ transmittance,
    const float* __restrict__ omega,
    const bool* __restrict__ valid_mask,
    const int* __restrict__ tile_gaussian_count,
    const int* __restrict__ tile_gaussian_list,
    const float* __restrict__ means_world,
    const float* __restrict__ grad_output,
    int tiles_x, int tiles_y,
    int Nr, int Na,
    int n_gaussians,
    float* __restrict__ grad_means_world,
    float* __restrict__ grad_cov_world,
    float* __restrict__ grad_sh_coeffs,
    float* __restrict__ grad_transmittance,
    float radar_x, float radar_y, float radar_z,
    float alpha, float beta, float Rc,
    float rho_r, float rho_a
) {
    int pix_c = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_r = blockIdx.y * blockDim.y + threadIdx.y;

    if (pix_c >= Na || pix_r >= Nr) return;

    int tile_c = pix_c / BLOCK_X;
    int tile_r = pix_r / BLOCK_Y;
    int tile_id = tile_r * tiles_x + tile_c;

    int gaussian_count = tile_gaussian_count[tile_id];
    int list_offset = tile_id * MAX_GAUSSIANS_PER_TILE;

    if (gaussian_count == 0) return;

    int local_indices[MAX_GAUSSIANS_PER_TILE];
    float z_values[MAX_GAUSSIANS_PER_TILE];

    for (int i = 0; i < gaussian_count; i++) {
        local_indices[i] = tile_gaussian_list[list_offset + i];
        z_values[i] = gaussian_z[local_indices[i]];
    }

    for (int i = 1; i < MAX_GAUSSIANS_PER_TILE; i++) {
        if (i >= gaussian_count) break;
        int temp_idx = local_indices[i];
        float temp_z = z_values[i];
        int j = i - 1;
        while (j >= 0 && z_values[j] > temp_z) {
            local_indices[j + 1] = local_indices[j];
            z_values[j + 1] = z_values[j];
            j--;
        }
        local_indices[j + 1] = temp_idx;
        z_values[j + 1] = temp_z;
    }

    float pixel_grad = grad_output[pix_c * Nr + pix_r];

    float transmittance_acc = 1.0f;
    float prev_transmittance = 1.0f;

    for (int k = 0; k < gaussian_count; k++) {
        int idx = local_indices[k];

        float gamma_ipp = compute_gaussian_density_2d(
            (float)pix_c, (float)pix_r,
            ipp_c[idx], ipp_r[idx],
            cov_ipp_cc[idx], cov_ipp_rc[idx], cov_ipp_rr[idx]
        );

        if (gamma_ipp < 1e-8f) continue;

        float sigma_i = transmittance[idx];
        float S_i = fmaxf(0.0f, scattering[idx]);

        float alpha_i = sigma_i * gamma_ipp;

        float d_pixel_d_alpha = prev_transmittance * S_i;
        float d_pixel_d_S = prev_transmittance * alpha_i;
        float d_pixel_d_T = alpha_i * S_i;

        float dL_d_pixel = pixel_grad;

        float grad_sigma = dL_d_pixel * d_pixel_d_alpha * gamma_ipp;
        atomicAdd(&grad_transmittance[idx], grad_sigma);

        float cos_theta = __cosf(beta);
        float sin_theta = __sinf(beta);
        float cos_phi = __cosf(alpha);
        float sin_phi = __sinf(alpha);
        if (alpha < 0.0f) {
            cos_phi = __cosf(alpha + 2.0f * PI);
            sin_phi = __sinf(alpha + 2.0f * PI);
        }

        float sin_theta_sq = sin_theta * sin_theta;
        float cos_theta_sq = cos_theta * cos_theta;
        float sin_cos_theta = sin_theta * cos_theta;
        float sin_cos_phi = sin_theta * cos_phi;

        float sh_basis[16];
        sh_basis[0] = SH_C0;
        sh_basis[1] = SH_C1 * sin_theta * sin_phi;
        sh_basis[2] = SH_C1 * sin_theta * cos_phi;
        sh_basis[3] = SH_C1 * cos_theta;

        sh_basis[4] = SH_C2 * sin_cos_phi * sin_phi;
        sh_basis[5] = SH_C2 * sin_cos_theta * cos_phi;
        sh_basis[6] = SH_C20 * (3.0f * cos_theta_sq - 1.0f);
        sh_basis[7] = -SH_C2 * sin_cos_theta * sin_phi;
        sh_basis[8] = -SH_C2 * sin_cos_theta * cos_phi;

        float three_cos_theta_sq_minus_one = 3.0f * cos_theta_sq - 1.0f;
        float five_cos_theta_cu_minus_three = (5.0f * cos_theta_sq - 3.0f) * cos_theta;

        sh_basis[9] = SH_C22 * sin_theta * sin_phi * three_cos_theta_sq_minus_one;
        sh_basis[10] = SH_C22 * sin_cos_phi * three_cos_theta_sq_minus_one;
        sh_basis[11] = SH_C22 * cos_theta * three_cos_theta_sq_minus_one;
        sh_basis[12] = SH_C3 * sin_theta * sin_phi * five_cos_theta_cu_minus_three;
        sh_basis[13] = SH_C3 * sin_cos_phi * five_cos_theta_cu_minus_three;

        float fifteen_cos_theta_sq_sq_minus_six_cos_theta_sq_minus_one =
            (15.0f * cos_theta_sq - 6.0f) * cos_theta_sq - 1.0f;
        sh_basis[14] = SH_C30 * fifteen_cos_theta_sq_sq_minus_six_cos_theta_sq_minus_one;
        sh_basis[15] = -SH_C3 * sin_theta * sin_phi * five_cos_theta_cu_minus_three;

        for (int sh_idx = 0; sh_idx < SH_COEFFS_SIZE; sh_idx++) {
            float grad_sh = dL_d_pixel * d_pixel_d_S * sh_basis[sh_idx];
            atomicAdd(&grad_sh_coeffs[idx * SH_COEFFS_SIZE + sh_idx], grad_sh);
        }

        float dx = (float)pix_c - ipp_c[idx];
        float dy = (float)pix_r - ipp_r[idx];

        float cov_rr = cov_ipp_rr[idx];
        float cov_cc = cov_ipp_cc[idx];
        float cov_rc = cov_ipp_rc[idx];
        float det = cov_rr * cov_cc - cov_rc * cov_rc;

        if (det > EPSILON) {
            float inv_det = 1.0f / det;
            float exponent = -0.5f * (cov_cc * dx * dx - 2.0f * cov_rc * dx * dy + cov_rr * dy * dy) * inv_det;

            if (exponent > -20.0f) {
                float exp_val = expf(exponent);
                float base_val = exp_val / (2.0f * PI * sqrtf(det));

                float d_gamma_d_cov_rr = 0.5f * base_val * inv_det * (dy * dy - cov_cc * inv_det * 0.5f);
                float d_gamma_d_cov_cc = 0.5f * base_val * inv_det * (dx * dx - cov_rr * inv_det * 0.5f);
                float d_gamma_d_cov_rc = -base_val * inv_det * dx * dy;

                float grad_cov_rr = dL_d_pixel * d_pixel_d_alpha * sigma_i * d_gamma_d_cov_rr;
                float grad_cov_cc = dL_d_pixel * d_pixel_d_alpha * sigma_i * d_gamma_d_cov_cc;
                float grad_cov_rc = dL_d_pixel * d_pixel_d_alpha * sigma_i * d_gamma_d_cov_rc;

                atomicAdd(&grad_cov_world[idx * 6 + 0], grad_cov_rr);
                atomicAdd(&grad_cov_world[idx * 6 + 1], grad_cov_cc);
                atomicAdd(&grad_cov_world[idx * 6 + 2], 0.0f);
                atomicAdd(&grad_cov_world[idx * 6 + 3], grad_cov_rc);
                atomicAdd(&grad_cov_world[idx * 6 + 4], 0.0f);
                atomicAdd(&grad_cov_world[idx * 6 + 5], 0.0f);
            }
        }

        float d_center_dr = -(cov_rc * dx - cov_rr * dy) / (det + EPSILON);
        float d_center_dc = -(cov_cc * dx - cov_rc * dy) / (det + EPSILON);

        float grad_ipp_r = dL_d_pixel * d_pixel_d_alpha * sigma_i * d_center_dr;
        float grad_ipp_c = dL_d_pixel * d_pixel_d_alpha * sigma_i * d_center_dc;

        float yr = gaussian_yr[idx];
        float zr = gaussian_z[idx];

        float Rmin = sqrtf(yr * yr + zr * zr + EPSILON);
        float d_r_d_yr = yr / (rho_r * Rmin);
        float d_r_d_zr = zr / (rho_r * Rmin);

        float grad_xr = grad_ipp_c / rho_a;
        float grad_yr = grad_ipp_r * d_r_d_yr / rho_r;
        float grad_zr = grad_ipp_r * d_r_d_zr / rho_r;

        float cos_a = cosf(alpha);
        float sin_a = sinf(alpha);
        float cos_b = cosf(beta);
        float sin_b = sinf(beta);

        float d_xr_d_xw = cos_a;
        float d_xr_d_yw = sin_a;
        float d_xr_d_zw = 0.0f;

        float d_yr_d_xw = cos_b * sin_a;
        float d_yr_d_yw = -cos_b * cos_a;
        float d_yr_d_zw = -sin_b;

        float d_zr_d_xw = -sin_b * sin_a;
        float d_zr_d_yw = sin_b * cos_a;
        float d_zr_d_zw = -cos_b;

        float grad_xw = grad_xr * d_xr_d_xw + grad_yr * d_yr_d_xw + grad_zr * d_zr_d_xw;
        float grad_yw = grad_xr * d_xr_d_yw + grad_yr * d_yr_d_yw + grad_zr * d_zr_d_yw;
        float grad_zw = grad_xr * d_xr_d_zw + grad_yr * d_yr_d_zw + grad_zr * d_zr_d_zw;

        atomicAdd(&grad_means_world[idx * 3 + 0], grad_xw);
        atomicAdd(&grad_means_world[idx * 3 + 1], grad_yw);
        atomicAdd(&grad_means_world[idx * 3 + 2], grad_zw);

        prev_transmittance *= (1.0f - alpha_i);
        if (prev_transmittance < 1e-8f) {
            prev_transmittance = 0.0f;
        }
    }
}

class CUDASARRenderer {
public:
    void render(const SARRenderParams& input, float* output_image, float* output_omega) {
        int n = input.n_gaussians;
        int H = input.range_samples;
        int W = input.azimuth_samples;

        int tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
        int tiles_y = (H + BLOCK_Y - 1) / BLOCK_Y;
        int total_tiles = tiles_x * tiles_y;

        float* d_means_world;
        float* d_cov_world;
        float* d_transmittance;
        float* d_sh_coeffs;
        float* d_ipp_r;
        float* d_ipp_c;
        float* d_cov_ipp_rr;
        float* d_cov_ipp_cc;
        float* d_cov_ipp_rc;
        float* d_shadow_x;
        float* d_shadow_y;
        float* d_cov_shadow_xx;
        float* d_cov_shadow_yy;
        float* d_cov_shadow_xy;
        float* d_gaussian_yr;
        float* d_gaussian_z;
        float* d_scattering;
        bool* d_valid_mask;
        int* d_range_min;
        int* d_range_max;
        int* d_azi_min;
        int* d_azi_max;
        int* d_tile_gaussian_count;
        int* d_tile_gaussian_list;
        int* d_shadow_tile_gaussian_count;
        int* d_shadow_tile_gaussian_list;
        float* d_omega_sum;
        int* d_omega_count;
        float* d_omega;

        cudaMalloc(&d_means_world, n * 3 * sizeof(float));
        cudaMalloc(&d_cov_world, n * 6 * sizeof(float));
        cudaMalloc(&d_transmittance, n * sizeof(float));
        cudaMalloc(&d_sh_coeffs, n * SH_COEFFS_SIZE * sizeof(float));
        cudaMalloc(&d_ipp_r, n * sizeof(float));
        cudaMalloc(&d_ipp_c, n * sizeof(float));
        cudaMalloc(&d_cov_ipp_rr, n * sizeof(float));
        cudaMalloc(&d_cov_ipp_cc, n * sizeof(float));
        cudaMalloc(&d_cov_ipp_rc, n * sizeof(float));
        cudaMalloc(&d_shadow_x, n * sizeof(float));
        cudaMalloc(&d_shadow_y, n * sizeof(float));
        cudaMalloc(&d_cov_shadow_xx, n * sizeof(float));
        cudaMalloc(&d_cov_shadow_yy, n * sizeof(float));
        cudaMalloc(&d_cov_shadow_xy, n * sizeof(float));
        cudaMalloc(&d_gaussian_yr, n * sizeof(float));
        cudaMalloc(&d_gaussian_z, n * sizeof(float));
        cudaMalloc(&d_scattering, n * sizeof(float));
        cudaMalloc(&d_valid_mask, n * sizeof(bool));
        cudaMalloc(&d_range_min, n * sizeof(int));
        cudaMalloc(&d_range_max, n * sizeof(int));
        cudaMalloc(&d_azi_min, n * sizeof(int));
        cudaMalloc(&d_azi_max, n * sizeof(int));
        cudaMalloc(&d_tile_gaussian_count, total_tiles * sizeof(int));
        cudaMalloc(&d_tile_gaussian_list, total_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
        cudaMalloc(&d_shadow_tile_gaussian_count, total_tiles * sizeof(int));
        cudaMalloc(&d_shadow_tile_gaussian_list, total_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
        cudaMalloc(&d_omega_sum, n * sizeof(float));
        cudaMalloc(&d_omega_count, n * sizeof(int));
        cudaMalloc(&d_omega, n * sizeof(float));

        cudaMemcpy(d_means_world, input.means_world, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cov_world, input.cov_world, n * 6 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transmittance, input.transmittance, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sh_coeffs, input.sh_coeffs, n * SH_COEFFS_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        float track_angle_rad = input.track_angle * 0.017453292519943295f;
        float incidence_rad = input.incidence_angle * 0.017453292519943295f;
        float azimuth_rad = input.azimuth_angle * 0.017453292519943295f;
        float sin_beta = sinf(incidence_rad) * cosf(azimuth_rad);
        sin_beta = fmaxf(fminf(sin_beta, 1.0f), -1.0f);
        float beta_rad = asinf(sin_beta);
        float tan_beta = tanf(beta_rad);
        float radar_altitude = input.radar_z;

        float radar_pos_x = radar_altitude * tan_beta * sinf(track_angle_rad);
        float radar_pos_y = -radar_altitude * tan_beta * cosf(track_angle_rad);
        float radar_pos_z = radar_altitude;

        float Rc = radar_altitude / cosf(beta_rad);

        int block_dim = 256;
        int grid_dim_preprocess = (n + block_dim - 1) / block_dim;

        preprocess_kernel<<<grid_dim_preprocess, block_dim>>>(
            d_means_world,
            d_cov_world,
            d_transmittance,
            d_sh_coeffs,
            radar_pos_x, radar_pos_y, radar_pos_z,
            track_angle_rad, beta_rad, Rc,
            input.range_resolution, input.azimuth_resolution,
            input.range_samples, input.azimuth_samples,
            d_ipp_r,
            d_ipp_c,
            d_cov_ipp_rr,
            d_cov_ipp_cc,
            d_cov_ipp_rc,
            d_shadow_x,
            d_shadow_y,
            d_cov_shadow_xx,
            d_cov_shadow_yy,
            d_cov_shadow_xy,
            d_gaussian_yr,
            d_gaussian_z,
            d_scattering,
            d_valid_mask,
            n
        );

        compute_ranges_kernel<<<grid_dim_preprocess, block_dim>>>(
            d_ipp_r,
            d_ipp_c,
            d_cov_ipp_rr,
            d_cov_ipp_cc,
            d_cov_ipp_rc,
            d_valid_mask,
            input.range_samples,
            input.azimuth_samples,
            d_range_min,
            d_range_max,
            d_azi_min,
            d_azi_max,
            n
        );

        int tile_block_dim = 256;
        int tile_grid_dim = (total_tiles + tile_block_dim - 1) / tile_block_dim;

        dim3 build_grid(tiles_x, tiles_y);
        dim3 build_block(1, 1);
        build_tile_gaussian_list_kernel<<<build_grid, build_block>>>(
            d_ipp_r,
            d_ipp_c,
            d_range_min,
            d_range_max,
            d_azi_min,
            d_azi_max,
            d_gaussian_z,
            d_valid_mask,
            tiles_x,
            tiles_y,
            input.range_samples,
            input.azimuth_samples,
            d_tile_gaussian_count,
            d_tile_gaussian_list,
            n
        );

        cudaMalloc(&d_shadow_tile_gaussian_count, total_tiles * sizeof(int));
        cudaMalloc(&d_shadow_tile_gaussian_list, total_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
        cudaMalloc(&d_omega_sum, n * sizeof(float));
        cudaMalloc(&d_omega_count, n * sizeof(int));
        cudaMalloc(&d_omega, n * sizeof(float));

        cudaMemset(d_omega_sum, 0, n * sizeof(float));
        cudaMemset(d_omega_count, 0, n * sizeof(int));

        build_shadow_tile_list_kernel<<<build_grid, build_block>>>(
            d_shadow_x,
            d_shadow_y,
            d_cov_shadow_xx,
            d_cov_shadow_yy,
            d_cov_shadow_xy,
            d_gaussian_z,
            d_valid_mask,
            tiles_x,
            tiles_y,
            input.range_samples,
            input.azimuth_samples,
            d_shadow_tile_gaussian_count,
            d_shadow_tile_gaussian_list,
            n
        );

        dim3 omega_blocks((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
        dim3 omega_threads(BLOCK_X, BLOCK_Y);
        compute_omega_kernel<<<omega_blocks, omega_threads>>>(
            d_shadow_x,
            d_shadow_y,
            d_cov_shadow_xx,
            d_cov_shadow_yy,
            d_cov_shadow_xy,
            d_gaussian_z,
            d_scattering,
            d_transmittance,
            d_shadow_tile_gaussian_count,
            d_shadow_tile_gaussian_list,
            tiles_x,
            tiles_y,
            input.range_samples,
            input.azimuth_samples,
            d_omega_sum,
            d_omega_count,
            n
        );

        float* h_omega_sum = new float[n];
        int* h_omega_count = new int[n];
        float* h_omega = new float[n];
        cudaMemcpy(h_omega_sum, d_omega_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_omega_count, d_omega_count, n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) {
            if (h_omega_count[i] > 0) {
                h_omega[i] = h_omega_sum[i] / h_omega_count[i];
            } else {
                h_omega[i] = 1.0f;
            }
        }
        cudaMemcpy(d_omega, h_omega, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(output_omega, d_omega, n * sizeof(float), cudaMemcpyDeviceToDevice);
        delete[] h_omega_sum;
        delete[] h_omega_count;
        delete[] h_omega;

        float* d_output;
        cudaMalloc(&d_output, H * W * sizeof(float));

        dim3 render_blocks((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
        dim3 render_threads(BLOCK_X, BLOCK_Y);

        render_kernel<<<render_blocks, render_threads>>>(
            d_ipp_r,
            d_ipp_c,
            d_cov_ipp_rr,
            d_cov_ipp_cc,
            d_cov_ipp_rc,
            d_gaussian_z,
            d_scattering,
            d_transmittance,
            d_omega,
            d_valid_mask,
            d_tile_gaussian_count,
            d_tile_gaussian_list,
            tiles_x,
            tiles_y,
            input.range_samples,
            input.azimuth_samples,
            d_output
        );

        cudaMemcpy(output_image, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_means_world);
        cudaFree(d_cov_world);
        cudaFree(d_transmittance);
        cudaFree(d_sh_coeffs);
        cudaFree(d_ipp_r);
        cudaFree(d_ipp_c);
        cudaFree(d_cov_ipp_rr);
        cudaFree(d_cov_ipp_cc);
        cudaFree(d_cov_ipp_rc);
        cudaFree(d_shadow_x);
        cudaFree(d_shadow_y);
        cudaFree(d_cov_shadow_xx);
        cudaFree(d_cov_shadow_yy);
        cudaFree(d_cov_shadow_xy);
        cudaFree(d_gaussian_yr);
        cudaFree(d_gaussian_z);
        cudaFree(d_scattering);
        cudaFree(d_valid_mask);
        cudaFree(d_range_min);
        cudaFree(d_range_max);
        cudaFree(d_azi_min);
        cudaFree(d_azi_max);
        cudaFree(d_tile_gaussian_count);
        cudaFree(d_tile_gaussian_list);
        cudaFree(d_shadow_tile_gaussian_count);
        cudaFree(d_shadow_tile_gaussian_list);
        cudaFree(d_omega_sum);
        cudaFree(d_omega_count);
        cudaFree(d_omega);
        cudaFree(d_output);
    }

    void backward(
        const float* d_ipp_r,
        const float* d_ipp_c,
        const float* d_cov_ipp_rr,
        const float* d_cov_ipp_cc,
        const float* d_cov_ipp_rc,
        const float* d_gaussian_yr,
        const float* d_gaussian_z,
        const float* d_scattering,
        const float* d_transmittance,
        const float* d_omega,
        const bool* d_valid_mask,
        const int* d_tile_gaussian_count,
        const int* d_tile_gaussian_list,
        const float* d_means_world,
        const float* d_grad_output,
        int tiles_x, int tiles_y,
        int Nr, int Na,
        int n,
        float* d_grad_means_world,
        float* d_grad_cov_world,
        float* d_grad_sh_coeffs,
        float* d_grad_transmittance,
        float radar_x, float radar_y, float radar_z,
        float alpha, float beta, float Rc,
        float rho_r, float rho_a
    ) {
        dim3 render_blocks((Na + BLOCK_X - 1) / BLOCK_X, (Nr + BLOCK_Y - 1) / BLOCK_Y);
        dim3 render_threads(BLOCK_X, BLOCK_Y);

        backward_kernel<<<render_blocks, render_threads>>>(
            d_ipp_r,
            d_ipp_c,
            d_cov_ipp_rr,
            d_cov_ipp_cc,
            d_cov_ipp_rc,
            d_gaussian_yr,
            d_gaussian_z,
            d_scattering,
            d_transmittance,
            d_omega,
            d_valid_mask,
            d_tile_gaussian_count,
            d_tile_gaussian_list,
            d_means_world,
            d_grad_output,
            tiles_x, tiles_y,
            Nr, Na,
            n,
            d_grad_means_world,
            d_grad_cov_world,
            d_grad_sh_coeffs,
            d_grad_transmittance,
            radar_x, radar_y, radar_z,
            alpha, beta, Rc,
            rho_r, rho_a
        );
    }
};

std::vector<torch::Tensor> render_sar_backward(
    torch::Tensor means_world,
    torch::Tensor cov_world,
    torch::Tensor transmittance,
    torch::Tensor sh_coeffs,
    torch::Tensor grad_output,
    float radar_x, float radar_y, float radar_z,
    float track_angle,
    float incidence_angle,
    float azimuth_angle,
    float range_resolution,
    float azimuth_resolution,
    int range_samples,
    int azimuth_samples
) {
    int n = means_world.size(0);
    int H = range_samples;
    int W = azimuth_samples;

    int tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
    int tiles_y = (H + BLOCK_Y - 1) / BLOCK_Y;
    int total_tiles = tiles_x * tiles_y;

    float* d_means_world;
    float* d_cov_world;
    float* d_transmittance;
    float* d_sh_coeffs;
    float* d_grad_output;
    float* d_ipp_r;
    float* d_ipp_c;
    float* d_cov_ipp_rr;
    float* d_cov_ipp_cc;
    float* d_cov_ipp_rc;
    float* d_shadow_x;
    float* d_shadow_y;
    float* d_cov_shadow_xx;
    float* d_cov_shadow_yy;
    float* d_cov_shadow_xy;
    float* d_gaussian_yr;
    float* d_gaussian_z;
    float* d_scattering;
    bool* d_valid_mask;
    int* d_range_min;
    int* d_range_max;
    int* d_azi_min;
    int* d_azi_max;
    int* d_tile_gaussian_count;
    int* d_tile_gaussian_list;
    int* d_shadow_tile_gaussian_count;
    int* d_shadow_tile_gaussian_list;
    float* d_omega_sum;
    int* d_omega_count;
    float* d_omega;

    cudaMalloc(&d_means_world, n * 3 * sizeof(float));
    cudaMalloc(&d_cov_world, n * 6 * sizeof(float));
    cudaMalloc(&d_transmittance, n * sizeof(float));
    cudaMalloc(&d_sh_coeffs, n * SH_COEFFS_SIZE * sizeof(float));
    cudaMalloc(&d_grad_output, H * W * sizeof(float));
    cudaMalloc(&d_ipp_r, n * sizeof(float));
    cudaMalloc(&d_ipp_c, n * sizeof(float));
    cudaMalloc(&d_cov_ipp_rr, n * sizeof(float));
    cudaMalloc(&d_cov_ipp_cc, n * sizeof(float));
    cudaMalloc(&d_cov_ipp_rc, n * sizeof(float));
    cudaMalloc(&d_shadow_x, n * sizeof(float));
    cudaMalloc(&d_shadow_y, n * sizeof(float));
    cudaMalloc(&d_cov_shadow_xx, n * sizeof(float));
    cudaMalloc(&d_cov_shadow_yy, n * sizeof(float));
    cudaMalloc(&d_cov_shadow_xy, n * sizeof(float));
    cudaMalloc(&d_gaussian_yr, n * sizeof(float));
    cudaMalloc(&d_gaussian_z, n * sizeof(float));
    cudaMalloc(&d_scattering, n * sizeof(float));
    cudaMalloc(&d_valid_mask, n * sizeof(bool));
    cudaMalloc(&d_range_min, n * sizeof(int));
    cudaMalloc(&d_range_max, n * sizeof(int));
    cudaMalloc(&d_azi_min, n * sizeof(int));
    cudaMalloc(&d_azi_max, n * sizeof(int));
    cudaMalloc(&d_tile_gaussian_count, total_tiles * sizeof(int));
    cudaMalloc(&d_tile_gaussian_list, total_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
    cudaMalloc(&d_shadow_tile_gaussian_count, total_tiles * sizeof(int));
    cudaMalloc(&d_shadow_tile_gaussian_list, total_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
    cudaMalloc(&d_omega_sum, n * sizeof(float));
    cudaMalloc(&d_omega_count, n * sizeof(int));
    cudaMalloc(&d_omega, n * sizeof(float));

    cudaMemcpy(d_means_world, means_world.data_ptr<float>(), n * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cov_world, cov_world.data_ptr<float>(), n * 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transmittance, transmittance.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh_coeffs, sh_coeffs.data_ptr<float>(), n * SH_COEFFS_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output.data_ptr<float>(), H * W * sizeof(float), cudaMemcpyHostToDevice);

    float track_angle_rad = track_angle * 0.017453292519943295f;
    float incidence_rad = incidence_angle * 0.017453292519943295f;
    float azimuth_rad = azimuth_angle * 0.017453292519943295f;
    float sin_beta = sinf(incidence_rad) * cosf(azimuth_rad);
    sin_beta = fmaxf(fminf(sin_beta, 1.0f), -1.0f);
    float beta_rad = asinf(sin_beta);
    float tan_beta = tanf(beta_rad);
    float radar_altitude = radar_z;

    float radar_pos_x = radar_altitude * tan_beta * sinf(track_angle_rad);
    float radar_pos_y = -radar_altitude * tan_beta * cosf(track_angle_rad);
    float radar_pos_z = radar_altitude;

    float Rc = radar_altitude / cosf(beta_rad);

    int block_dim = 256;
    int grid_dim_preprocess = (n + block_dim - 1) / block_dim;

    preprocess_kernel<<<grid_dim_preprocess, block_dim>>>(
        d_means_world, d_cov_world, d_transmittance, d_sh_coeffs,
        radar_pos_x, radar_pos_y, radar_pos_z,
        track_angle_rad, beta_rad, Rc,
        range_resolution, azimuth_resolution,
        range_samples, azimuth_samples,
        d_ipp_r, d_ipp_c, d_cov_ipp_rr, d_cov_ipp_cc, d_cov_ipp_rc,
        d_shadow_x, d_shadow_y, d_cov_shadow_xx, d_cov_shadow_yy, d_cov_shadow_xy,
        d_gaussian_yr, d_gaussian_z, d_scattering, d_valid_mask, n
    );

    compute_ranges_kernel<<<grid_dim_preprocess, block_dim>>>(
        d_ipp_r, d_ipp_c, d_cov_ipp_rr, d_cov_ipp_cc, d_cov_ipp_rc,
        d_valid_mask, range_samples, azimuth_samples,
        d_range_min, d_range_max, d_azi_min, d_azi_max, n
    );

    dim3 build_grid(tiles_x, tiles_y);
    dim3 build_block(1, 1);
    build_tile_gaussian_list_kernel<<<build_grid, build_block>>>(
        d_ipp_r, d_ipp_c, d_range_min, d_range_max, d_azi_min, d_azi_max,
        d_gaussian_z, d_valid_mask, tiles_x, tiles_y,
        range_samples, azimuth_samples,
        d_tile_gaussian_count, d_tile_gaussian_list, n
    );

    cudaMemset(d_omega_sum, 0, n * sizeof(float));
    cudaMemset(d_omega_count, 0, n * sizeof(int));

    build_shadow_tile_list_kernel<<<build_grid, build_block>>>(
        d_shadow_x, d_shadow_y, d_cov_shadow_xx, d_cov_shadow_yy, d_cov_shadow_xy,
        d_gaussian_z, d_valid_mask, tiles_x, tiles_y,
        range_samples, azimuth_samples,
        d_shadow_tile_gaussian_count, d_shadow_tile_gaussian_list, n
    );

    dim3 omega_blocks((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
    dim3 omega_threads(BLOCK_X, BLOCK_Y);
    compute_omega_kernel<<<omega_blocks, omega_threads>>>(
        d_shadow_x, d_shadow_y, d_cov_shadow_xx, d_cov_shadow_yy, d_cov_shadow_xy,
        d_gaussian_z, d_scattering, d_transmittance,
        d_shadow_tile_gaussian_count, d_shadow_tile_gaussian_list,
        tiles_x, tiles_y, range_samples, azimuth_samples,
        d_omega_sum, d_omega_count, n
    );

    float* h_omega_sum = new float[n];
    int* h_omega_count = new int[n];
    float* h_omega = new float[n];
    cudaMemcpy(h_omega_sum, d_omega_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_omega_count, d_omega_count, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (h_omega_count[i] > 0) {
            h_omega[i] = h_omega_sum[i] / h_omega_count[i];
        } else {
            h_omega[i] = 1.0f;
        }
    }
    cudaMemcpy(d_omega, h_omega, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_omega_sum;
    delete[] h_omega_count;
    delete[] h_omega;

    auto grad_means = torch::zeros({n, 3}, means_world.options());
    auto grad_cov = torch::zeros({n, 6}, cov_world.options());
    auto grad_sh = torch::zeros({n, SH_COEFFS_SIZE}, sh_coeffs.options());
    auto grad_transmittance = torch::zeros({n}, transmittance.options());

    CUDASARRenderer renderer;
    renderer.backward(
        d_ipp_r, d_ipp_c, d_cov_ipp_rr, d_cov_ipp_cc, d_cov_ipp_rc,
        d_gaussian_yr, d_gaussian_z, d_scattering, d_transmittance, d_omega,
        d_valid_mask, d_tile_gaussian_count, d_tile_gaussian_list,
        d_means_world, d_grad_output,
        tiles_x, tiles_y, range_samples, azimuth_samples, n,
        grad_means.data_ptr<float>(),
        grad_cov.data_ptr<float>(),
        grad_sh.data_ptr<float>(),
        grad_transmittance.data_ptr<float>(),
        radar_pos_x, radar_pos_y, radar_pos_z,
        track_angle_rad, beta_rad, Rc,
        range_resolution, azimuth_resolution
    );

    cudaFree(d_means_world);
    cudaFree(d_cov_world);
    cudaFree(d_transmittance);
    cudaFree(d_sh_coeffs);
    cudaFree(d_grad_output);
    cudaFree(d_ipp_r);
    cudaFree(d_ipp_c);
    cudaFree(d_cov_ipp_rr);
    cudaFree(d_cov_ipp_cc);
    cudaFree(d_cov_ipp_rc);
    cudaFree(d_shadow_x);
    cudaFree(d_shadow_y);
    cudaFree(d_cov_shadow_xx);
    cudaFree(d_cov_shadow_yy);
    cudaFree(d_cov_shadow_xy);
    cudaFree(d_gaussian_yr);
    cudaFree(d_gaussian_z);
    cudaFree(d_scattering);
    cudaFree(d_valid_mask);
    cudaFree(d_range_min);
    cudaFree(d_range_max);
    cudaFree(d_azi_min);
    cudaFree(d_azi_max);
    cudaFree(d_tile_gaussian_count);
    cudaFree(d_tile_gaussian_list);
    cudaFree(d_shadow_tile_gaussian_count);
    cudaFree(d_shadow_tile_gaussian_list);
    cudaFree(d_omega_sum);
    cudaFree(d_omega_count);
    cudaFree(d_omega);

    return {grad_means, grad_cov, grad_transmittance, grad_sh};
}

std::vector<torch::Tensor> render_sar(
    torch::Tensor means_world,
    torch::Tensor cov_world,
    torch::Tensor transmittance,
    torch::Tensor sh_coeffs,
    float radar_x, float radar_y, float radar_z,
    float track_angle,
    float incidence_angle,
    float azimuth_angle,
    float range_resolution,
    float azimuth_resolution,
    int range_samples,
    int azimuth_samples,
    bool prefiltered = false,
    bool debug = false
) {
    int n = means_world.size(0);

    SARRenderParams params;
    params.n_gaussians = n;
    params.means_world = means_world.data_ptr<float>();
    params.cov_world = cov_world.data_ptr<float>();
    params.transmittance = transmittance.data_ptr<float>();
    params.sh_coeffs = sh_coeffs.data_ptr<float>();
    params.radar_x = radar_x;
    params.radar_y = radar_y;
    params.radar_z = radar_z;
    params.track_angle = track_angle;
    params.incidence_angle = incidence_angle;
    params.azimuth_angle = azimuth_angle;
    params.range_resolution = range_resolution;
    params.azimuth_resolution = azimuth_resolution;
    params.range_samples = range_samples;
    params.azimuth_samples = azimuth_samples;
    params.prefiltered = prefiltered;
    params.debug = debug;

    auto output_image = torch::zeros({range_samples, azimuth_samples}, means_world.options());
    auto omega = torch::zeros({n}, means_world.options());

    CUDASARRenderer renderer;
    renderer.render(params, output_image.data_ptr<float>(), omega.data_ptr<float>());

    return {output_image, omega};
}

std::vector<torch::Tensor> render_sar_with_settings(
    torch::Tensor means_world,
    torch::Tensor cov_world,
    torch::Tensor transmittance,
    torch::Tensor sh_coeffs,
    SARRasterizationSettings settings
) {
    int n = means_world.size(0);

    SARRenderParams params;
    params.n_gaussians = n;
    params.means_world = means_world.data_ptr<float>();
    params.cov_world = cov_world.data_ptr<float>();
    params.transmittance = transmittance.data_ptr<float>();
    params.sh_coeffs = sh_coeffs.data_ptr<float>();
    params.radar_x = settings.radar_x;
    params.radar_y = settings.radar_y;
    params.radar_z = settings.radar_z;
    params.track_angle = settings.track_angle;
    params.incidence_angle = settings.incidence_angle;
    params.azimuth_angle = settings.azimuth_angle;
    params.range_resolution = settings.range_resolution;
    params.azimuth_resolution = settings.azimuth_resolution;
    params.range_samples = settings.range_samples;
    params.azimuth_samples = settings.azimuth_samples;
    params.prefiltered = settings.prefiltered;
    params.debug = settings.debug;

    auto output_image = torch::zeros({settings.range_samples, settings.azimuth_samples}, means_world.options());
    auto omega = torch::zeros({n}, means_world.options());

    CUDASARRenderer renderer;
    renderer.render(params, output_image.data_ptr<float>(), omega.data_ptr<float>());

    return {output_image, omega};
}

PYBIND11_MODULE(cuda_rasterizer_sar, m) {
    m.def("render_sar", &render_sar, "SAR forward rendering using CUDA");
    m.def("render_sar_with_settings", &render_sar_with_settings, "SAR forward rendering with settings object");
    m.def("render_sar_backward", &render_sar_backward, "SAR backward rendering using CUDA");
}