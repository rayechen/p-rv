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

#include "ppl/nn/engines/riscv/optimizer/opt_graph.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/common/logger.h"
// #include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv/conv_op.h"
#include "ppl/nn/engines/utils.h"
#include <string.h>

#define SHOW_GRAPH_VIS
#ifdef SHOW_GRAPH_VIS
#include "ppl/nn/auxtools/to_graphviz.h"
#include <fstream>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode OptGraph::InitKernels(const ir::Graph* graph) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        auto& type = node->GetType();
        auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
        if (!creator) {
            LOG(ERROR) << "cannot find creator for RISCVOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                       << type.name << "]";
            return RC_NOT_FOUND;
        }

        auto opt_kernel = unique_ptr<RISCVOptKernel>(creator(node));
        if (!opt_kernel) {
            LOG(ERROR) << "create RISCVOptKernel failed: oom";
            return RC_OUT_OF_MEMORY;
        }

        info_->kernels.emplace(node->GetId(), std::move(opt_kernel));
    }

    return RC_SUCCESS;
}

RetCode OptGraph::InitTensorImpls() {
    tensor_impls_.clear();
    auto& shapes = graph_->data->shapes;
    for (auto it = graph_->topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        auto edge_id = edge->GetId();
        auto tensor_type = graph_->data->constants.find(edge_id) == graph_->data->constants.end() ? TENSORTYPE_NORMAL
                                                                                                  : TENSORTYPE_RESERVED;
        TensorImpl* tensor = new TensorImpl(edge, tensor_type);
        if (shapes.find(edge_id) != shapes.end()) {
            utils::IrShape2TensorShape(shapes[edge_id], &tensor->GetShape());
        } else {
            tensor->GetShape().SetDataFormat(DATAFORMAT_NDARRAY);
        }
        tensor_impls_.emplace(edge_id, unique_ptr<TensorImpl>(tensor));
    }
    return RC_SUCCESS;
}

RetCode OptGraph::Init(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info, RISCVEngineOptions* options) {
    resource_ = resource;
    graph_ = graph;
    info_ = info;
    options_ = options;

    auto status = InitKernels(graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init kernels failed: " << GetRetCodeStr(status);
        return status;
    }

    status = InitTensorImpls();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init tensor impls failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

#define REORDER_INPUT 0
#define REORDER_OUTPUT 1
#define REORDER_EXTRA_INPUT 2

RetCode OptGraph::AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id, const nodeid_t& node_id,
                               const int32_t& reorder_type,
                               const ppl::common::dataformat_t& reorder_src_format,
                               const ppl::common::dataformat_t& reorder_dst_format,
                               const ppl::common::datatype_t& reorder_src_type,
                               const ppl::common::datatype_t& reorder_dst_type) {
    auto edge = graph_->topo->GetEdgeById(edge_id);
    auto node = graph_->topo->GetNodeById(node_id);

    std::string reorder_node_name = "";
    if (reorder_type == REORDER_INPUT) {
        reorder_node_name = "ReorderInput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_OUTPUT) {
        reorder_node_name = "ReorderOutput_" + edge->GetName() + "_of_" + node->GetName();
    } else if (reorder_type == REORDER_EXTRA_INPUT) {
        reorder_node_name = "ReorderExtraInput_" + edge->GetName() + "_of_" + node->GetName();
    }

    auto node_ret_pair = graph_->topo->AddNode(reorder_node_name);
    if (!node_ret_pair.second) {
        LOG(ERROR) << "node[" << reorder_node_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Node* reorder_node = node_ret_pair.first; // TODO: change name for easy to understand
    reorder_node->SetType(ir::Node::Type("ppl", "Reorder"));

    std::string reorder_edge_name = reorder_node_name + "_edge";
    auto edge_ret_pair = graph_->topo->AddEdge(reorder_edge_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "edge[" << reorder_edge_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Edge* reorder_edge = edge_ret_pair.first;

    if (reorder_type == REORDER_INPUT ||
        reorder_type == REORDER_EXTRA_INPUT) { // edge -> reorder_node -> reorder_edge -> node
        reorder_node->AddInput(edge_id);
        reorder_node->AddOutput(reorder_edge->GetId());
        reorder_edge->SetProducer(reorder_node->GetId());
        reorder_edge->AddConsumer(node_id);

        edge->DelConsumer(node_id);
        edge->AddConsumer(reorder_node->GetId());
        if (reorder_type == REORDER_INPUT) {
            node->ReplaceInput(edge_id, reorder_edge->GetId());
        } else if (reorder_type == REORDER_EXTRA_INPUT) {
            node->ReplaceExtraInput(edge_id, reorder_edge->GetId());
        }
    } else if (reorder_type == REORDER_OUTPUT) { // node -> reorder_edge -> reorder_node ->  edge
        reorder_node->AddInput(reorder_edge->GetId());
        reorder_node->AddOutput(edge_id);
        reorder_edge->SetProducer(node_id);
        reorder_edge->AddConsumer(reorder_node->GetId());

        edge->SetProducer(reorder_node->GetId());
        node->ReplaceOutput(edge_id, reorder_edge->GetId());
    }

    auto type = reorder_node->GetType();
    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for RISCVOptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<RISCVOptKernel>(creator(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create RISCVOptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    // opt_kernel->SetOutputDataFormat(0, reorder_out_format);
    opt_kernel->SetOutputDataFormat(0, reorder_dst_format);
    opt_kernel->SetOutputDataType(0, reorder_dst_type);

    info_->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);

    tensor->GetShape().SetDataFormat(reorder_dst_format);
    tensor->GetShape().SetDataType(reorder_dst_type);

    tensor_impls_.emplace(reorder_edge->GetId(), unique_ptr<TensorImpl>(tensor));

    LOG(INFO) << "successfully add reorder op " << reorder_node_name << " to reorder " <<
    GetDataFormatStr(reorder_src_format) << " " << GetDataTypeStr(reorder_src_type) << " to " <<
    GetDataFormatStr(reorder_dst_format) << " " << GetDataTypeStr(reorder_dst_type) << ".";
    return RC_SUCCESS;
}

inline bool IsGraphOutput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetOutputCount(); i++) {
        if (graph->topo->GetOutput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

inline bool IsGraphInput(const ir::Graph* graph, edgeid_t edge_id) {
    for (uint32_t i = 0; i < graph->topo->GetInputCount(); i++) {
        if (graph->topo->GetInput(i) == edge_id) {
            return true;
        }
    }
    return false;
}

RetCode OptGraph::LayoutOptimize(const OptKernelOptions& options) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        if (info_->kernels.find(node_id) == info_->kernels.end()) {
            LOG(ERROR) << "cannot find node_id " << node_id << " in RuntimePartitionInfo.";
            return RC_NOT_FOUND;
        }
        auto kernel = (RISCVOptKernel*)info_->kernels[node_id].get();
        auto node = kernel->GetNode();

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        {
            auto status = kernel->SelectAlgorithm(IOinfo, options);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "kernel[" << node->GetName() << "] SelectAlgorithm failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        vector<dataformat_t> selected_input_formats(node->GetInputCount(), DATAFORMAT_NDARRAY);
        vector<datatype_t> selected_input_data_types(node->GetInputCount(), DATATYPE_FLOAT32);
        vector<dataformat_t> selected_output_formats(node->GetOutputCount(), DATAFORMAT_NDARRAY);
        vector<datatype_t> selected_output_data_types(node->GetOutputCount(), DATATYPE_FLOAT32);
        {
            for (uint32_t i = 0; i < node->GetInputCount(); i++) {
                auto edge_id = node->GetInput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                } 
                selected_input_formats[i] = tensor_impls_[edge_id]->GetShape().GetDataFormat();
                selected_input_data_types[i] = tensor_impls_[edge_id]->GetShape().GetDataType();
            }
            if (options.engine_options->forward_precision == RISCV_USE_FP32) {
                selected_input_data_types[0] = DATATYPE_FLOAT32;
            } else if (options.engine_options->forward_precision == RISCV_USE_FP16) {
                selected_input_data_types[0] = DATATYPE_FLOAT16;
            }

            for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
                auto edge_id = node->GetOutput(i);
                if (edge_id == INVALID_EDGEID) {
                    continue;
                } 
                selected_output_formats[i] = tensor_impls_[edge_id]->GetShape().GetDataFormat();
                selected_output_data_types[i] = tensor_impls_[edge_id]->GetShape().GetDataType();
            }

            auto status = kernel->SelectFormat(IOinfo, &selected_input_formats, &selected_output_formats);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "kernel[" << node->GetName() << "] SelectFormat failed: " << GetRetCodeStr(status);
                return status;
            }

            status = kernel->SelectDataType(IOinfo, &selected_input_data_types, &selected_output_data_types);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "kernel[" << node->GetName() << "] SelectDataType failed: " << GetRetCodeStr(status);
            }
        }

        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto edge_id = node->GetInput(i);
            if (edge_id == INVALID_EDGEID) {
                continue;
            } 
            auto input_format = tensor_impls_[edge_id]->GetShape().GetDataFormat();
            auto input_data_type = tensor_impls_[edge_id]->GetShape().GetDataType();
            auto selected_input_format = selected_input_formats[i];
            auto selected_input_data_type = selected_input_data_types[i];
            if (input_format != selected_input_format || input_data_type != selected_input_data_type) {
                auto status = AddReorderOp(options, edge_id, node_id, REORDER_INPUT, input_format, selected_input_format, input_data_type, selected_input_data_type);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return status;
                }
            }
        }

        // extra input(used by if/loop op) force to be ndarray
        for (uint32_t i = 0; i < node->GetExtraInputCount(); i++) {
            auto edge_id = node->GetExtraInput(i);
            auto extra_input_format = tensor_impls_[edge_id]->GetShape().GetDataFormat();
            auto extra_input_data_type = tensor_impls_[edge_id]->GetShape().GetDataType();
            if (extra_input_format != ppl::common::DATAFORMAT_NDARRAY) {
                auto status = AddReorderOp(options, edge_id, node_id, REORDER_EXTRA_INPUT, extra_input_format,
                                      ppl::common::DATAFORMAT_NDARRAY, extra_input_data_type, extra_input_data_type);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return status;
                }
            }
        }

        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            auto edge_id = node->GetOutput(i);
            // auto output_format = tensor_impls_[edge_id]->GetShape().GetDataFormat();
            // auto output_type = tensor_impls_[edge_id]->GetShape().GetDataType();
            auto output_format = DATAFORMAT_NDARRAY;
            auto output_type = DATATYPE_FLOAT16;

            auto selected_output_format = selected_output_formats[i];
            auto selected_output_data_type = selected_output_data_types[i];

            tensor_impls_[edge_id]->GetShape().SetDataFormat(selected_output_format);
            tensor_impls_[edge_id]->GetShape().SetDataType(selected_output_data_type);
            kernel->SetOutputDataFormat(i, selected_output_format);
            kernel->SetOutputDataType(i, selected_output_data_type);
            if (IsGraphOutput(graph_, edge_id) && selected_output_format != output_format) {
                auto status = AddReorderOp(options, edge_id, node_id, REORDER_OUTPUT, selected_output_format, output_format, selected_output_data_type, output_type);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "add reorder op failed.";
                    return status;
                }
            }
        }

    }

#if 0
    auto status = FuseReorderOp();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FuseReorderOp failed: " << GetRetCodeStr(status);
        return status;
    }
#endif

    return RC_SUCCESS;
}

#if 0
RetCode OptGraph::SetIODataType() {
    /** by default i/o float data. TODO: can be specified by user using options*/
    for (int64_t i = 0; i < graph_->topo->GetInputCount(); i++) {
        auto edge_id = graph_->topo->GetInput(i);
        tensor_impls_[edge_id]->GetShape().SetDataType(DATATYPE_FLOAT32);
    }
    for (int64_t i = 0; i < graph_->topo->GetOutputCount(); i++) {
        auto edge_id = graph_->topo->GetOutput(i);
        tensor_impls_[edge_id]->GetShape().SetDataType(DATATYPE_FLOAT32);
    }
}
#endif

RetCode OptGraph::TryToInferType(RISCVDevice* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

#if 0
    /** get user specified forward precision */
    if (options_->forward_precision == ARM_USE_FP16) {
        for (int64_t i = 0; i < graph_->topo->GetInputCount(); i++) {
            auto edge_id = graph_->topo->GetInput(i);
            tensor_impls_[edge_id]->GetShape().SetDataType(DATATYPE_FLOAT16);
        }
    } else if (options_->forward_precision == ARM_USE_FP32) {
        for (int64_t i = 0; i < graph_->topo->GetInputCount(); i++) {
            auto edge_id = graph_->topo->GetInput(i);
            tensor_impls_[edge_id]->GetShape().SetDataType(DATATYPE_FLOAT32);
        }
    } else {
        LOG(ERROR) << "Unsupported forward precision";
        return RC_UNSUPPORTED;
    }
#endif

    for (int64_t i = 0; i < graph_->topo->GetInputCount(); i++) {
        auto edge_id = graph_->topo->GetInput(i);
    }

    /** try to infer types for each node's input and output*/
    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNodeById(node_id);
        bool all_inputs_has_type = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdgeById(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape().GetDataType() == DATATYPE_UNKNOWN) {
                all_inputs_has_type = false;
                break;
            }
        }
        if (!all_inputs_has_type) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto kernel = (RISCVOptKernel*)(info_->kernels[node_id].get());
        kernel->InferTypes(&IOinfo);
        // save output data type to common param
        for (uint32_t i = 0; i < node->GetOutputCount(); i++) {
            kernel->SetOutputDataType(i, IOinfo.GetOutput<TensorImpl>(i)->GetShape().GetDataType());
        }
    }

    // TODO: delete this test used for printing all types
    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNodeById(node_id);
        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });
        auto &in_shape = IOinfo.GetInput<TensorImpl>(0)->GetShape();
        auto &out_shape = IOinfo.GetOutput<TensorImpl>(0)->GetShape();
        
        LOG(DEBUG) << "node name " << node->GetName();
        LOG(DEBUG) << " input shape type" << GetDataTypeStr(in_shape.GetDataType());
        LOG(DEBUG) << " output shape type" <<  GetDataTypeStr(out_shape.GetDataType());
    }

    return RC_SUCCESS;
}

RetCode OptGraph::TryToInferDims(RISCVDevice* device) {
    vector<nodeid_t> sorted_nodes;
    graph_->topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        auto node = graph_->topo->GetNodeById(node_id);
        bool all_inputs_has_dims = true;
        for (uint32_t i = 0; i < node->GetInputCount(); i++) {
            auto input_edge = graph_->topo->GetEdgeById(node->GetInput(i));
            if (!input_edge) { // some op may have emtpy input
                continue;
            }
            if (tensor_impls_.find(input_edge->GetId()) == tensor_impls_.end() ||
                tensor_impls_[input_edge->GetId()]->GetShape().GetDimCount() == 0) {
                all_inputs_has_dims = false;
                break;
            }
        }
        if (!all_inputs_has_dims) {
            continue;
        }

        InputOutputInfo IOinfo;
        IOinfo.SetNode(node);
        IOinfo.SetAcquireObjectFunc([this](edgeid_t eid, uint32_t, Device*) -> EdgeObject* {
            auto iter = tensor_impls_.find(eid);
            if (iter == tensor_impls_.end()) {
                return nullptr;
            }
            return iter->second.get();
        });

        auto kernel = (RISCVOptKernel*)(info_->kernels[node_id].get());
        auto status = kernel->InferDims(&IOinfo);
        if (status != RC_SUCCESS) {
            continue;
        }
    }

    return RC_SUCCESS;
}

ppl::common::RetCode OptGraph::CreateRISCVOptKernel(const OptKernelOptions& options, const ir::Node* node,
                                                  RISCVOptKernel** kernel) {
    auto& type = node->GetType();

    auto creator = OptKernelCreatorManager::Instance()->Find(type.domain, type.name);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for RISCVOptKernel[" << node->GetName() << "] type[" << type.domain << ":"
                   << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = unique_ptr<RISCVOptKernel>(creator(node));
    if (!opt_kernel) {
        LOG(ERROR) << "create RISCVOptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    *kernel = opt_kernel.get();
    info_->kernels.emplace(node->GetId(), std::move(opt_kernel));

    return RC_SUCCESS;
}

/**
 * I/O tensor data type & layout inference and transformation:
 *  1. Infer and set data type using initial graph, with user-specified infer type.
 *      Note: a) Network input tensor data type won't be passed to runtime op kernel (whose defualt data type is default fp32), 
 *              as the runtime kernel only stores output data type.
 *            b) Reorder nodes have not been added.
 *  2. Infer tensor dims using intial graph.
 *  3. Transform data layout and inject necessary reorder nodes.
 *      Note: the types of injected reorder nodes are default (fp32)
 *  4. Infer and set data type again using optimized graph. (Set the data type of reorder nodes to user-specified type.)
 * */

RetCode OptGraph::DoOptimize(RISCVDevice* device) {
    OptKernelOptions options;
    options.resource = resource_;
    options.graph_data = graph_->data.get();
    options.device = device;
    options.engine_options = options_;

    for (auto it = info_->kernels.begin(); it != info_->kernels.end(); ++it) {
        auto kernel = (RISCVOptKernel*)(it->second.get());
        auto status = kernel->Init(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Init for kernel[" << kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (auto it = tensor_impls_.begin(); it != tensor_impls_.end(); ++it) {
        auto edge_id = it->first;
        if (graph_->data->constants.find(edge_id) != graph_->data->constants.end()) {
            auto tensor = it->second.get();
            tensor->SetDevice(device);
            tensor->ReallocBuffer();
            memcpy(tensor->GetBufferPtr<void>(), graph_->data->constants[edge_id].data.data(),
                   tensor->GetShape().GetBytesExcludingPadding());
        }
    }

    RetCode status = RC_SUCCESS;

    status = TryToInferType(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "TryToInferType failed: " << GetRetCodeStr(status);
        return status;
    }

    status = TryToInferDims(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "TryToInferDims failed: " << GetRetCodeStr(status);
        return status;
    }
#if 0
    FuseChannelShuffle(options);
#endif
    status = LayoutOptimize(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LayoutOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

#if 0
    while (FuseConvActivation() || FuseConvAdd() || FuseBNReLU() || FuseArithmeticReLU() || FuseFcActivation() ||
           FuseSwish(options))
        ;
#endif

    /** re-infer data type for added nodes*/
    status = TryToInferType(device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Re-TryToInferType failed: " << GetRetCodeStr(status);
        return status;
    }

#ifdef SHOW_GRAPH_VIS
    std::string vis = utils::ToGraphviz(graph_->topo.get());
    std::ofstream out_file("./graph.dot");
    if (out_file.is_open()) {
        out_file << vis;
    }
#endif

    for (auto it = tensor_impls_.begin(); it != tensor_impls_.end(); ++it) {
        auto edge_id = it->first;
        if (graph_->data->constants.find(edge_id) != graph_->data->constants.end()) {
            auto tensor = it->second.get();
            tensor->FreeBuffer();
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::riscv
