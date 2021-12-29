#ifndef PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_KERNEL_H_
#define PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_KERNEL_H_

#include "ppl/kernel/riscv/fp16/conv2d/common/conv2d_ndarray_gemm_cto8c_kernel_fp16.h"

namespace ppl { namespace kernel {namespace riscv {

typedef void (*PPL3_conv_gemm_riscv_kernel_m8nx)(
    const __fp16 *kernel_A,
    const __fp16 *kernel_B,
    __fp16 *kernel_C,
    int k,
    int total_n
);

typedef void (*PPL3_conv_gemm_riscv_kernel_func_type_t) (
    const __fp16 *A,
    const __fp16 *B,
    __fp16 *C,
    int m,
    int n,
    int k    
);

#ifdef __cplusplus
extern "C" {
#endif

    void gemm_common_m8n16_left15_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left15_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left14_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left14_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left13_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left13_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left12_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left12_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left11_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left11_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left10_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left10_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left9_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left9_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left8_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left8_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left7_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left7_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left6_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left6_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left5_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left5_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left4_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left4_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left3_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left3_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left2_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left2_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left1_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left1_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left0_first_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

    void gemm_common_m8n16_left0_rv64_fp16(
        const __fp16 *A,
        const __fp16 *B,
        __fp16 *C,
        int m,
        int n,
        int k
    );

#ifdef __cplusplus
}
#endif

template<int align_n, int align_left_n, PPL3_conv_gemm_riscv_kernel_m8nx core_func, PPL3_conv_gemm_riscv_kernel_m8nx core_left_func>
static void ppl3_conv_gemm_cto8c_kernel_fp16(
    const __fp16 *A,
    const __fp16 *B,
    __fp16 *C,
    int m,
    int n,
    int k) {

    int mi, ni;

    int kernel_m_stride = k * 8;

    for (mi = 0; mi < m; mi += 8) {
        auto temp_B = B;
        for (ni = 0; ni <= n - align_n; ni += align_n) {
            core_func(A, temp_B, C, k, n);

            C += align_n * 8;
            temp_B += align_n;
        }

        if (align_left_n != 0) {
            core_left_func(A, temp_B, C, k, n);
            C += align_left_n * 8;
        }
        A += kernel_m_stride;
    }
}

template<bool first>
PPL3_conv_gemm_riscv_kernel_func_type_t ppl3_conv_gemm_select_cto8c_kernel_fp16(int n) {
    // TODO: add "int m" to parameters
    switch (n % 24) {
    case 0: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 0,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<0>>;
    case 1: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 1,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<1>>;
    case 2: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 2,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<2>>;
    case 3: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 3,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<3>>;
    case 4: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 4,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<4>>;
    case 5: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 5,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<5>>;
    case 6: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 6,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<6>>;
    case 7: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 7,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<7>>;
    case 8: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 8,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<8>>;
    case 9: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 9,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<9>>;
    case 10: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 10, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<10>>;
    case 11: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 11, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<11>>;
    case 12: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 12, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<12>>;
    case 13: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 13, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<13>>;
    case 14: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 14, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<14>>;
    case 15: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 15, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<15>>;
    case 16: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 16, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<16>>;
    case 17: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 17, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<17>>;
    case 18: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 18, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<18>>;
    case 19: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 19, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<19>>;
    case 20: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 20, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<20>>;
    case 21: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 21, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<21>>;
    case 22: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 22, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<22>>;
    case 23: return ppl3_conv_gemm_cto8c_kernel_fp16<24, 23, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<23>>;
    }
    return ppl3_conv_gemm_cto8c_kernel_fp16<24, 0,   ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, ppl3_conv_gemm_cto8c_m8nx_kernel_core_fp16<0>>;;
}

template <bool first>
PPL3_conv_gemm_riscv_kernel_func_type_t ppl3_conv_gemm_select_kernel_fp16(int n) {
    switch (n % 16) {
        case 0 : return first? gemm_common_m8n16_left0_first_rv64_fp16  : gemm_common_m8n16_left0_rv64_fp16;
        case 1 : return first? gemm_common_m8n16_left1_first_rv64_fp16  : gemm_common_m8n16_left1_rv64_fp16;
        case 2 : return first? gemm_common_m8n16_left2_first_rv64_fp16  : gemm_common_m8n16_left2_rv64_fp16;
        case 3 : return first? gemm_common_m8n16_left3_first_rv64_fp16  : gemm_common_m8n16_left3_rv64_fp16;
        case 4 : return first? gemm_common_m8n16_left4_first_rv64_fp16  : gemm_common_m8n16_left4_rv64_fp16;
        case 5 : return first? gemm_common_m8n16_left5_first_rv64_fp16  : gemm_common_m8n16_left5_rv64_fp16;
        case 6 : return first? gemm_common_m8n16_left6_first_rv64_fp16  : gemm_common_m8n16_left6_rv64_fp16;
        case 7 : return first? gemm_common_m8n16_left7_first_rv64_fp16  : gemm_common_m8n16_left7_rv64_fp16;
        case 8 : return first? gemm_common_m8n16_left8_first_rv64_fp16  : gemm_common_m8n16_left8_rv64_fp16;
        case 9 : return first? gemm_common_m8n16_left9_first_rv64_fp16  : gemm_common_m8n16_left9_rv64_fp16;
        case 10: return first? gemm_common_m8n16_left10_first_rv64_fp16 : gemm_common_m8n16_left10_rv64_fp16;
        case 11: return first? gemm_common_m8n16_left11_first_rv64_fp16 : gemm_common_m8n16_left11_rv64_fp16;
        case 12: return first? gemm_common_m8n16_left12_first_rv64_fp16 : gemm_common_m8n16_left12_rv64_fp16;
        case 13: return first? gemm_common_m8n16_left13_first_rv64_fp16 : gemm_common_m8n16_left13_rv64_fp16;
        case 14: return first? gemm_common_m8n16_left14_first_rv64_fp16 : gemm_common_m8n16_left14_rv64_fp16;
        case 15: return first? gemm_common_m8n16_left15_first_rv64_fp16 : gemm_common_m8n16_left15_rv64_fp16;
    }
    return first? gemm_common_m8n16_left0_first_rv64_fp16 : gemm_common_m8n16_left0_rv64_fp16;
}

template<int src_atom_c, bool first>
PPL3_conv_gemm_riscv_kernel_func_type_t ppl3_conv_gemm_select_xcto8c_kernel_fp16(int m, int n) {
    switch (src_atom_c) {
        case 1: return ppl3_conv_gemm_select_cto8c_kernel_fp16<first>(n);
        case 8: return ppl3_conv_gemm_select_kernel_fp16<first>(n);
        default: return ppl3_conv_gemm_select_kernel_fp16<first>(n);
    }
}

}}}; // namespace ppl::kernel::riscv

#endif