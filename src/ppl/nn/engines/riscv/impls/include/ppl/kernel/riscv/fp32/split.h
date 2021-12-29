#ifndef __ST_PPL_KERNEL_RISCV_FP32_SPLIT_H_
#define __ST_PPL_KERNEL_RISCV_FP32_SPLIT_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode split_n4cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    float **dst_list);

}}};

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_SPLIT_H_