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

#include "ppl/nn/engines/riscv/riscv_device.h"
#include "ppl/nn/engines/riscv/engine.h"
#include "ppl/nn/engines/riscv/kernel.h"
#include "ppl/nn/engines/riscv/engine_context.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/riscv/optimizer/opt_graph.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode RISCVEngine::Init(const RISCVEngineOptions& options) {
    options_ = options;
    return ppl::common::RC_SUCCESS;
};

EngineContext* RISCVEngine::CreateEngineContext(const string&) {
    return new RISCVEngineContext(GetName(), &device_);
}

ppl::common::RetCode RISCVEngine::Configure(uint32_t, ...) {
    LOG(ERROR) << "invalid option[" << "]";
    return ppl::common::RC_UNSUPPORTED;
}

bool RISCVEngine::Supports(const ir::Node* node) const {
    auto& type = node->GetType();
    bool ok = OptKernelCreatorManager::Instance()->Find(type.domain, type.name) != nullptr;
    return ok;
}

RetCode RISCVEngine::DoOptimize(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info) {
    OptGraph opt_graph;
    auto status = opt_graph.Init(graph, resource, info, &options_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init OptGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = opt_graph.DoOptimize(&device_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode RISCVEngine::ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(graph, resource, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LoadConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv
