#ifndef __ST_PPL_KERNEL_RISCV_FP32_AVERAFEPOOL2D_H_
#define __ST_PPL_KERNEL_RISCV_FP32_AVERAFEPOOL2D_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode averagepool2d_n4cx_1x16_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const float *src,
    float *dst
);

ppl::common::RetCode averagepool2d_nchw_normal_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const float *src,
    float *dst    
);

}}};    //  namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_AVERAFEPOOL2D_H_