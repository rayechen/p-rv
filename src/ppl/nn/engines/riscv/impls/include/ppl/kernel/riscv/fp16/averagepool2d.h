#ifndef __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_
#define __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode averagepool2d_n8chw_1x16_fp16(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const __fp16 *src,
    __fp16 *dst
);

ppl::common::RetCode averagepool2d_nchw_normal_fp16(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const __fp16 *src,
    __fp16 *dst    
);

}}};    //  namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP16_AVERAFEPOOL2D_H_