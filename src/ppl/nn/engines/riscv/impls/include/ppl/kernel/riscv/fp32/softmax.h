#ifndef __ST_PPL_KERNEL_RISCV_FP32_SOFTMAX_H_
#define __ST_PPL_KERNEL_RISCV_FP32_SOFTMAX_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode softmax_ndarray_fp32(
    const ppl::nn::TensorShape *shape,
    const int64_t axis,
    const float *src,
    float *dst
);

}}};

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_SOFTMAX_H_