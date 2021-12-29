#ifndef __ST_PPL_KERNEL_RISCV_FP32_RELU_H_
#define __ST_PPL_KERNEL_RISCV_FP32_RELU_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode relu_fp32(
    const ppl::nn::TensorShape *shape,
    const float *src,
    float *dst
);

}}}; // namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_RELU_H_