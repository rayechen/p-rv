#ifndef __ST_PPL_KERNEL_RISCV_FP16_SPLIT_H_
#define __ST_PPL_KERNEL_RISCV_FP16_SPLIT_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode split_n8cx_fp16(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const __fp16 *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    __fp16 **dst_list);

}}};

#endif  //  __ST_PPL_KERNEL_RISCV_FP16_SPLIT_H_