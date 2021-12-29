
#include "ppl/kernel/riscv/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_n4cx_fp32(
    const float **src_list,
    float *dst,

    const ppl::nn::TensorShape **src_shape_list,
    const int32_t num_src,
    const int32_t c_axis    
) {
    return concat_nbcx<float, 4>(src_list, dst, src_shape_list, num_src, c_axis);
}

}}};    //  namespace ppl::kernel::riscv