
#include "ppl/kernel/riscv/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_ndarray_fp16(
    const __fp16 **src_list,
    __fp16 *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis    
) {
    return concat_ndarray<__fp16>(src_list, dst, src_shape_list, num_src, c_axis);
}

}}};    // namespace ppl::kernel::riscv