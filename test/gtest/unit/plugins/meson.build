# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ucx_staged_pool_unit_test_dep = declare_dependency(
    sources: [
        '../../../../src/plugins/ucx/ucx_staged_pool.cpp',
        'ucx_staged_pool.cpp',
    ],
    include_directories: [
        include_directories('../../../../src/plugins/ucx'),
        nixl_inc_dirs,
        gtest_inc_dirs,
    ],
    dependencies: [gtest_dep],
)
