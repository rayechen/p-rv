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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_RISCV_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_RISCV_DEVICE_H_

#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/engines/riscv/data_converter.h"
#include "ppl/common/log.h"

namespace ppl { namespace nn { namespace riscv {

class RISCVDevice : public utils::GenericCpuDevice {
public:
    RISCVDevice(uint64_t alignment)
        : GenericCpuDevice(alignment), data_converter_() {}

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

private:
    RISCVDataConverter data_converter_;
};

}}} // namespace ppl::nn::riscv

#endif
