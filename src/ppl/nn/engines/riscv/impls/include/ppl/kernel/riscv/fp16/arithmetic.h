#ifndef __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_H_
#define __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_H_

#include <stdint.h>
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
// #include "ppl/common/riscv/sysinfo.h"
// #include "ppl/kernel/riscv/common/config.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode add_fp16(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const bool fuse_relu,
    const __fp16 *src0,
    const __fp16 *src1,
    __fp16 *dst);

ppl::common::RetCode sub_fp16(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const bool fuse_relu,
    const __fp16 *src0,
    const __fp16 *src1,
    __fp16 *dst);

ppl::common::RetCode mul_fp16(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const bool fuse_relu,
    const __fp16 *src0,
    const __fp16 *src1,
    __fp16 *dst);

ppl::common::RetCode div_fp16(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const bool fuse_relu,
    const __fp16 *src0,
    const __fp16 *src1,
    __fp16 *dst);

}}};    //  namespace ppl::kernel::riscv


#endif  //  __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_H_
