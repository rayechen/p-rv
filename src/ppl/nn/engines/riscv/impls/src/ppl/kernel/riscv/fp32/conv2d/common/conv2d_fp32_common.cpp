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
#include <chrono>

#include "ppl/kernel/riscv/fp32/conv2d/naive/conv2d_ndarray_naive_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/tile_gemm/vec128/conv2d_ndarray_tile_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/tile_gemm/vec128/conv2d_n4cx_tile_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d.h"
#include "ppl/common/log.h"
#include "ppl/common/types.h"

namespace ppl { namespace kernel { namespace riscv {

conv2d_common_algo_info conv2d_fp32_algo_selector::select_algo(const ppl::nn::TensorShape& input_shape, const conv2d_common_param &param)
{
    LOG(INFO) << "RISCV FP32 CONV select algo";
    static conv2d_common_algo_info unknown_info = {
        conv2d_common_algo::unknown,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATAFORMAT_UNKNOWN,
        ppl::common::DATATYPE_FLOAT32,
        ppl::common::DATATYPE_FLOAT32
    };

    if (ppl::common::DATAFORMAT_NDARRAY == input_shape.GetDataFormat()) {
        return {
            conv2d_common_algo::tile_gemm,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_N4CX,
            ppl::common::DATATYPE_FLOAT32,
            ppl::common::DATATYPE_FLOAT32            
        };
    }

    return {
        conv2d_common_algo::tile_gemm,
        ppl::common::DATAFORMAT_N4CX,
        ppl::common::DATAFORMAT_N4CX,
        ppl::common::DATATYPE_FLOAT32,
        ppl::common::DATATYPE_FLOAT32
    };

    return {
        conv2d_common_algo::naive,
        ppl::common::DATAFORMAT_NDARRAY,
        ppl::common::DATAFORMAT_NDARRAY,
        ppl::common::DATATYPE_FLOAT32,
        ppl::common::DATATYPE_FLOAT32
    };

    return unknown_info;
}

conv2d_offline_manager<float> *conv2d_fp32_algo_selector::gen_algo(const conv2d_common_param &param, const conv2d_common_algo_info &algo_info, ppl::common::Allocator *allocator)
{
    LOG(INFO) << "RISCV FP32 CONV gen algo";
    conv2d_offline_manager<float> *conv_mgr = nullptr;

    if (conv2d_common_algo::naive == algo_info.algo_type &&
        ppl::common::DATAFORMAT_NDARRAY == algo_info.input_format &&
        ppl::common::DATAFORMAT_NDARRAY == algo_info.output_format) {
        
        conv_mgr = new conv2d_ndarray_naive_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        ppl::common::DATAFORMAT_NDARRAY == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        
        conv_mgr = new conv2d_ndarray_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    if (conv2d_common_algo::tile_gemm == algo_info.algo_type &&
        ppl::common::DATAFORMAT_N4CX == algo_info.input_format &&
        ppl::common::DATAFORMAT_N4CX == algo_info.output_format) {
        
        conv_mgr = new conv2d_n4cx_tile_gemm_fp32_offline_manager(param, algo_info, allocator);
    }

    return conv_mgr;
}

}}}; // namespace ppl::kernel::riscv
