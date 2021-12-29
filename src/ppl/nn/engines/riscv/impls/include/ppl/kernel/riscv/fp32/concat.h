#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONCAT_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONCAT_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_n4cx_fp32(
    const float **src_list,
    float *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis
);

ppl::common::RetCode concat_ndarray_fp32(
    const float **src_list,
    float *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis
);

ppl::common::RetCode concat_n4cx_interleave_channels_fp32(
    const float **src_list,
    float *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx
);

}}};    //  namespace ppl::kernel::riscv

#endif  //  __ST_PPL_KERNEL_RISCV_FP32_CONCAT_H_