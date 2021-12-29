#ifndef __ST_PPL_KERNEL_RISCV_FP32_CLIP_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CLIP_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode clip_fp32(
    const ppl::nn::TensorShape *shape,
    const float clip_max,
    const float clip_min,
    const float *src,
    float *dst
);

}}};    //  namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_CLIP_H_