/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../../op_host/op_api/aclnn_matmul_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_aclnn_matmul_reduce_scatter_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_aclnn_matmul_reduce_scatter_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_aclnn_matmul_reduce_scatter_test TearDown" << endl; }
};


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_float16_basic) {
  TensorDesc x1_desc = TensorDesc({8192, 1536}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({1536, 12288}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({8192, 12288}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_float16_transpose_b) {
  TensorDesc x1_desc = TensorDesc({8192, 1536}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({12288, 1536}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({8192, 12288}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_bfloat16) {
  TensorDesc x1_desc = TensorDesc({8192, 1536}, ACL_DT_BF16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({1536, 12288}, ACL_DT_BF16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_BF16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({8192, 12288}, ACL_DT_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_large_shape) {
  TensorDesc x1_desc = TensorDesc({32768, 4096}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({4096, 8192}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({32768, 8192}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_small_shape) {
  TensorDesc x1_desc = TensorDesc({512, 256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 128}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({512, 128}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_with_bias) {
  TensorDesc x1_desc = TensorDesc({4096, 2048}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({2048, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4096, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_boundary_m) {
  TensorDesc x1_desc = TensorDesc({1, 2048}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({2048, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({1, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_boundary_n) {
  TensorDesc x1_desc = TensorDesc({2048, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({1024, 1}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({2048, 1}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_boundary_k) {
  TensorDesc x1_desc = TensorDesc({2048, 1}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({1, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({2048, 1024}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}


TEST_F(l2_aclnn_matmul_reduce_scatter_test, matmul_reduce_scatter_test_tiling_large_k) {
  TensorDesc x1_desc = TensorDesc({4096, 8192}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({8192, 2048}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4096, 2048}, ACL_DT_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

