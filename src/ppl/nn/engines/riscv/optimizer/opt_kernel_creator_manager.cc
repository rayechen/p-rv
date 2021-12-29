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

#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv/conv_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/sigmoid_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/unsqueeze_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/max_pool_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/average_pool_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_mean_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_max_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_min_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_sum_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/gather_op.h"

#include "ppl/nn/engines/riscv/optimizer/ops/ppl/shape_operation_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode OptKernelCreatorManager::Register(const string& domain, const string& type, OptKernelCreator creator) {
    auto domain_ret = domain_type_creator_.insert(make_pair(domain, map<string, OptKernelCreator>()));
    auto type_ret = domain_ret.first->second.insert(make_pair(type, creator));
    return type_ret.second ? RC_SUCCESS : RC_EXISTS;
}

void OptKernelCreatorManager::Remove(const string& domain, const string& type) {
    auto domain_ret = domain_type_creator_.find(domain);
    if (domain_ret != domain_type_creator_.end()) {
        auto& type2creator = domain_ret->second;
        type2creator.erase(type);
        if (type2creator.empty()) {
            domain_type_creator_.erase(domain_ret);
        }
    }
}

OptKernelCreator OptKernelCreatorManager::Find(const string& domain, const string& type) {
    auto type_creator_ref = domain_type_creator_.find(domain);
    if (type_creator_ref != domain_type_creator_.end()) {
        auto creator_ref = type_creator_ref->second.find(type);
        if (creator_ref != type_creator_ref->second.end()) {
            return creator_ref->second;
        }
    }
    return nullptr;
}

template <typename T>
static RISCVOptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

#define REGISTER_OPT_KERNEL_CREATOR(domain, type, classname) \
    domain_type_creator_[domain].insert(make_pair(type, GenericCreateOptKernel<classname>))

OptKernelCreatorManager::OptKernelCreatorManager() {
    // onnx op default domain is ""
    REGISTER_OPT_KERNEL_CREATOR("", "Conv", ConvOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Add", AddOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sub", SubOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Mul", MulOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Div", DivOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Relu", ReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Reshape", ReshapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Shape", ShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sigmoid", SigmoidOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Split", SplitOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Unsqueeze", UnsqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MaxPool", MaxPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "AveragePool", AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalAveragePool", AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Flatten", FlattenOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Softmax", SoftmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gemm", GemmOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Clip", ClipOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMean", ReduceMeanOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMax", ReduceMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMin", ReduceMinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceSum", ReduceSumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Concat", ConcatOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Transpose", TransposeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gather", GatherOp);
    // mmcv custom op

    // ppl
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Shape", PPLShapeOperationOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Reorder", ReorderOp);
}

}}} // namespace ppl::nn::riscv
