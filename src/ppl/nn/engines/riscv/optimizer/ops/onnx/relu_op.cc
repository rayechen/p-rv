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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/relu_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ReluOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = GenericInferDims;
    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

RetCode ReluOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (selected_input_formats->at(0) == DATAFORMAT_N8CX) {
        selected_input_formats->at(0) = DATAFORMAT_N8CX;
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    }
    selected_input_formats->at(0) = DATAFORMAT_N8CX;
    selected_output_formats->at(0) = DATAFORMAT_N8CX;
    return RC_SUCCESS;
}

RetCode ReluOp::SelectDataType(const InputOutputInfo& info,
                              std::vector<datatype_t>* selected_input_data_types,
                              std::vector<datatype_t>* selected_output_data_types) {
    if (selected_input_data_types->at(0) == DATATYPE_FLOAT16) {
        selected_input_data_types->at(0) = DATATYPE_FLOAT16;
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    }
    selected_input_data_types->at(0) = DATATYPE_FLOAT16;
    selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    return RC_SUCCESS;
}

KernelImpl* ReluOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReluKernel>();
}

}}} // namespace ppl::nn::riscv