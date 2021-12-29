#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_RISCV_COMMON_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_RISCV_COMMON_PARAM_H_

#include "ppl/common/types.h"
#include <vector>

namespace ppl { namespace nn { namespace riscv {

struct RISCVCommonParam {
    std::vector<ppl::common::dataformat_t> output_formats;
    std::vector<ppl::common::datatype_t> output_types;
};

}}} // namespace ppl::nn::riscv

#endif
