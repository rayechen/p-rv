#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV2D_COMMON_CONV2D_GEMM_KERNEL_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV2D_COMMON_CONV2D_GEMM_KERNEL_FP32_H_

// #include "ppl/kernel/riscv/fp16/conv2d/common/conv2d_ndarray_gemm_cto8c_kernel_fp16.h"

namespace ppl { namespace kernel {namespace riscv {

typedef void (*conv2d_gemm_kernel_m8nx_riscv_fp32_type_t)(
    const float *kernel_A,
    const float *kernel_B,
    float *kernel_C,
    int64_t k,
    int64_t total_n
);

typedef void (*conv2d_gemm_kernel_func_riscv_fp32_type_t) (
    const float *A,
    const float *B,
    float *C,
    int64_t m,
    int64_t n,
    int64_t k
);

template<int64_t align_n, int64_t align_left_n, conv2d_gemm_kernel_m8nx_riscv_fp32_type_t core_func, conv2d_gemm_kernel_m8nx_riscv_fp32_type_t core_left_func>
static void ppl3_conv_gemm_cto8c_kernel_fp16(
    const float *A,
    const float *B,
    float *C,
    int64_t m,
    int64_t n,
    int64_t k) {

    const int64_t atom_m = 4;

    int64_t mi, ni;
    int64_t kernel_m_stride = k * atom_m;

    for (mi = 0; mi < m; mi += atom_m) {
        auto temp_B = B;
        for (ni = 0; ni <= n - align_n; ni += align_n) {
            core_func(A, temp_B, C, k, n);

            C += align_n * atom_m;
            temp_B += align_n;
        }

        if (align_left_n != 0) {
            core_left_func(A, temp_B, C, k, n);
            C += align_left_n * atom_m;
        }
        A += kernel_m_stride;
    }
}

template <int64_t align_n, int64_t align_left_n, conv2d_gemm_kernel_m8nx_riscv_fp32_type_t core_func, conv2d_gemm_kernel_m8nx_riscv_fp32_type_t core_left_func>
static void ppl3_conv_gemm_kernel_fp16(
    const float *A,
    const float *B,
    float *C,
    int64_t m,
    int64_t n,
    int64_t k) {
    const int64_t atom_m = 4;

    int64_t mi, ni;
    int64_t kernel_n_stride = align_n * atom_m;
    int64_t kernel_n_left_stride = align_left_n * atom_m;
    int64_t kernel_m_stride = k * atom_m;

    for (mi = 0; mi < m; mi += atom_m) {
        auto temp_B = B;
        for (ni = 0; ni <= n - align_n; ni += align_n) {
            core_func(A, temp_B, C, k, n);

            C += kernel_n_stride;
            temp_B += kernel_n_stride;
        }

        if (align_left_n != 0) {
            core_left_func(A, temp_B, C, k, n);
            C += kernel_n_left_stride;
        }
        A += kernel_m_stride;
    }
}

template<bool first>
void conv2d_gemm_kernel_cto4c_fp32 (
    const float *A,
    const float *B,
    float *C,
    int64_t m,
    int64_t n,
    int64_t k) {

    const int atom_ic = 1, atom_oc = 4;

    if (first) {
        for (int mi = 0; mi < m; mi += 1) {
            for (int ni = 0; ni < n; ni += 1) {
                int c_idx = mi / atom_oc * n * atom_oc + ni * atom_oc + mi % atom_oc;
                C[c_idx] = 0.0f;
            }
        }
    }

    for (int mi = 0; mi < m; mi += 1) {
        for (int ni = 0; ni < n; ni += 1) {
            for (int ki = 0; ki < k; ki += 1) {
                int a_idx = mi / atom_oc * k * atom_oc + ki * atom_oc + mi % atom_oc;
                int b_idx = ki / atom_ic * n * atom_ic + ni * atom_ic + ki % atom_ic;
                int c_idx = mi / atom_oc * n * atom_oc + ni * atom_oc + mi % atom_oc;
                C[c_idx] += A[a_idx] * B[b_idx];
            }
        }
    }
}

template<bool first>
void conv2d_gemm_kernel_4cto4c_fp32 (
    const float *A,
    const float *B,
    float *C,
    int64_t m,
    int64_t n,
    int64_t k) {
    
    const int atom_ic = 4, atom_oc = 4;

    if (first) {
        for (int mi = 0; mi < m; mi += 1) {
            for (int ni = 0; ni < n; ni += 1) {
                int c_idx = mi / atom_oc * n * atom_oc + ni * atom_oc + mi % atom_oc;
                C[c_idx] = 0.0f;
            }
        }
    }

    for (int mi = 0; mi < m; mi += 1) {
        for (int ni = 0; ni < n; ni += 1) {
            for (int ki = 0; ki < k; ki += 1) {
                int a_idx = mi / atom_oc * k * atom_oc + ki * atom_oc + mi % atom_oc;
                int b_idx = ki / atom_ic * n * atom_ic + ni * atom_ic + ki % atom_ic;
                int c_idx = mi / atom_oc * n * atom_oc + ni * atom_oc + mi % atom_oc;
                C[c_idx] += A[a_idx] * B[b_idx];
            }
        }
    }
}

template<bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_cto4c_kernel_fp32_vec128(int64_t n) {
    return conv2d_gemm_kernel_cto4c_fp32<first>;
}

template <bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_4cto4c_kernel_fp32_vec128(int64_t m, int64_t n) {
    return conv2d_gemm_kernel_4cto4c_fp32<first>;
}

template<int64_t src_atom_c, bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_xcto4c_kernel_fp32_vec128(int64_t m, int64_t n) {
    switch (src_atom_c) {
        case 1: return conv2d_gemm_select_cto4c_kernel_fp32_vec128<first>(n);
        case 4: return conv2d_gemm_select_4cto4c_kernel_fp32_vec128<first>(m, n);
        default: return conv2d_gemm_select_4cto4c_kernel_fp32_vec128<first>(m, n);
    }
}

}}}; // namespace ppl::kernel::riscv

#endif