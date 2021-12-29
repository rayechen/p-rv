// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <new>
#include <cstring>

#include "ppl/kernel/riscv/fp32/fc/vec128/fc_fp32_vec128.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

void fc_cvt_flt_riscv_fp32(
    const float *flt,
    float *flt_cvt,

    int32_t num_outs,
    int32_t channels) {

}

template <int64_t atom_m>
void hgemm_n8chw_mxn8_riscv_fp32(
    const float *src,
    const float *flt,
    const float *bias,
    float *dst,

    int32_t channels,   // padded
    int32_t num_outs    // padded 
) {
   
}

void fc_n8chw_riscv_fp32(
    const float *src,
    const float *flt,
    const float *bias,
    float *dst,

    const int32_t batch,
    const int32_t channels,
    const int32_t num_outs    
) {

}

void fc_fp32_vec128_executor::cal_kernel_tunning_param(){}

uint64_t fc_fp32_vec128_executor::cal_temp_buffer_size()
{
    LOG(INFO) << "FC cal_temp_buffer_size";
    return 1;
}

ppl::common::RetCode fc_fp32_vec128_executor::prepare()
{
    if (!fc_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    cal_kernel_tunning_param();
    LOG(INFO) << "FC prepare";

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_vec128_executor::execute()
{
    if (!fc_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_  || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    LOG(INFO) << "FC execute";
    return common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_vec128_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    return ppl::common::RC_SUCCESS;
}

fc_executor<float> *fc_fp32_vec128_manager::gen_executor()
{
    return new fc_fp32_vec128_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::riscv