#ifndef PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_MEM_H_
#define PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_MEM_H_

#include <riscv-vector.h>

namespace ppl { namespace kernel {namespace riscv {

template <bool with_relu>
void ppl3_conv_gemm_dst_blk_trans_o8_fp16(
    __fp16 * dst_blk,
    int dst_blk_h,
    int dst_blk_w,

    __fp16* dst,
    int dst_h,
    int dst_w,
    
    int real_dst_blk_m,
    int real_dst_blk_h,
    int real_dst_blk_w,

    const __fp16* bias)
{
    const int atom_c = 8;
    const int num_unroll = 8;
    const auto vl = vsetvli(atom_c, RVV_E16, RVV_M1);
    float16xm1_t _vzero = vfmvvf_float16xm1(0.f, vl);

    for (int mi = 0; mi < real_dst_blk_m; mi += atom_c) {
        auto temp_dst = dst + mi * dst_h * dst_w;
        auto temp_dst_blk = dst_blk + mi * dst_blk_h * dst_blk_w;
        float16xm1_t _vbias = vlev_float16xm1(bias, vl);

        for (int hi = 0; hi < real_dst_blk_h; hi += 1) {
            int wi;
            for (wi = 0; wi <= real_dst_blk_w - num_unroll; wi += num_unroll) {
                int temp_dst_loc = wi * atom_c;
                auto this_dst_blk_ptr = temp_dst_blk + temp_dst_loc;
                auto this_dst_ptr = temp_dst + temp_dst_loc;

                float16xm1_t _v0 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 0, vl);
                float16xm1_t _v1 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 1, vl);
                float16xm1_t _v2 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 2, vl);
                float16xm1_t _v3 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 3, vl);
                float16xm1_t _v4 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 4, vl);
                float16xm1_t _v5 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 5, vl);
                float16xm1_t _v6 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 6, vl);
                float16xm1_t _v7 = vlev_float16xm1(this_dst_blk_ptr + atom_c * 7, vl);
                
                _v0 = vfaddvv_float16xm1(_v0, _vbias, vl);
                _v1 = vfaddvv_float16xm1(_v1, _vbias, vl);
                _v2 = vfaddvv_float16xm1(_v2, _vbias, vl);
                _v3 = vfaddvv_float16xm1(_v3, _vbias, vl);
                _v4 = vfaddvv_float16xm1(_v4, _vbias, vl);
                _v5 = vfaddvv_float16xm1(_v5, _vbias, vl);
                _v6 = vfaddvv_float16xm1(_v6, _vbias, vl);
                _v7 = vfaddvv_float16xm1(_v7, _vbias, vl);

                if (with_relu) {
                    _v0 = vfmaxvv_float16xm1(_v0, _vzero, vl);
                    _v1 = vfmaxvv_float16xm1(_v1, _vzero, vl);
                    _v2 = vfmaxvv_float16xm1(_v2, _vzero, vl);
                    _v3 = vfmaxvv_float16xm1(_v3, _vzero, vl);
                    _v4 = vfmaxvv_float16xm1(_v4, _vzero, vl);
                    _v5 = vfmaxvv_float16xm1(_v5, _vzero, vl);
                    _v6 = vfmaxvv_float16xm1(_v6, _vzero, vl);
                    _v7 = vfmaxvv_float16xm1(_v7, _vzero, vl);
                }

                vsev_float16xm1(this_dst_ptr + atom_c * 0, _v0, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 1, _v1, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 2, _v2, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 3, _v3, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 4, _v4, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 5, _v5, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 6, _v6, vl);
                vsev_float16xm1(this_dst_ptr + atom_c * 7, _v7, vl);
            }

            for (; wi < real_dst_blk_w; wi += 1) {
                int temp_dst_loc = wi * atom_c;
                auto this_dst_blk_ptr = temp_dst_blk + temp_dst_loc;
                auto this_dst_ptr = temp_dst + temp_dst_loc;

                float16xm1_t _v0 = vlev_float16xm1(this_dst_blk_ptr + 0 , vl);
                _v0 = vfaddvv_float16xm1(_v0, _vbias, vl);
                if (with_relu) {
                    _v0 = vfmaxvv_float16xm1(_v0, _vzero, vl);
                }
                vsev_float16xm1(this_dst_ptr + 0 , _v0, vl);
            }

            temp_dst += dst_w * atom_c;
            temp_dst_blk += dst_blk_w * atom_c;
        }
        bias += atom_c;
    }
}

}}}; // namespace ppl::kernel::riscv

#endif // #define PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_MEM_H_