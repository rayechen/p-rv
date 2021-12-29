#ifndef __ST_PPL_KERNEL_RISCV_FP16_CONCAT_H_
#define __ST_PPL_KERNEL_RISCV_FP16_CONCAT_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_n8cx_fp16(
    const __fp16 **src_list,
    __fp16 *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis
);

ppl::common::RetCode concat_ndarray_fp16(
    const __fp16 **src_list,
    __fp16 *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis
);

ppl::common::RetCode concat_n8cx_interleave_channels_fp16(
    const __fp16 **src_list,
    __fp16 *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx
);

}}};    //  namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP16_CONCAT_H_