
#include "ppl/kernel/riscv/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_n4cx_interleave_channels_fp32(
    const float **src_list,
    float *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx
) {
    return concat_nbcx_interleave_channels<float, 4>(src_list, dst, src_shape_list, num_src, axis, c_dim_idx);
}

}}};    //  namespace ppl::kernel::riscv