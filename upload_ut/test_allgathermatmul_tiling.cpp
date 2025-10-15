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
#include "../../../op_kernel/all_gather_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"


class AllGatherMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
    }
};


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_basic_float16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 111UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_basic_bfloat16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_large_shape) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{32768, 8192}, {32768, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{8192, 16384}, {8192, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{32768, 16384}, {32768, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{32768, 8192}, {32768, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_small_shape) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_transpose_b) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{4096, 2048}, {4096, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_n_zero) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 0}, {2048, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_k_zero) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{0, 2048}, {0, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_bias_float16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{4096, 2048}, {4096, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{2048, 2048}, {2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 111UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_bias_bfloat16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{4096, 2048}, {4096, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{2048}, {2048}}, ge::DT_BF16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{2048, 2048}, {2048, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 111UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_world_size_2) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_world_size_4) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_world_size_16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {
                {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        }
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

