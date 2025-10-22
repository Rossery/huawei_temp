/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace MoeDistributeDispatchUT {

class MoeDistributeDispatchTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeDispatchTiling TearDown" << std::endl;
    }
};

//normal: share rank


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_basic_small) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2}, {64, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_basic_large) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 4}, {256, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_static_quant) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 3}, {128, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_dynamic_quant) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 3}, {128, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_with_tp) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 2}, {128, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_shared_expert) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 1}, {128, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_max_k) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 8}, {64, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_layered_comm) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 2}, {128, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_a2_dynamic_with_scales) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 2}, {128, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}


TEST_F(MoeDistributeDispatchTiling, moe_dispatch_a2_layered_dynamic) {
    struct MoeDistributeDispatchCompileInfo {};
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 2}, {128, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576}, {576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo,
        "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

}