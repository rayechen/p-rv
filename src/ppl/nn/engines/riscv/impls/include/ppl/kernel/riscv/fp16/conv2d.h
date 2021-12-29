#ifndef __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_
#define __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_

#include <string>
#include <float.h>

#include "ppl/kernel/riscv/common/conv2d.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/allocator.h"
#include "ppl/common/sys.h"
#include "functional"

namespace ppl { namespace kernel { namespace riscv {

class conv2d_fp16_algo_selector {
public:
    static conv2d_common_algo_info select_algo(const ppl::nn::TensorShape& input_shape, const conv2d_common_param &param);
    static conv2d_offline_manager<__fp16> *gen_algo(const conv2d_common_param &param, const conv2d_common_algo_info &algo_info, ppl::common::Allocator *allocator);
};

}}}; // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_