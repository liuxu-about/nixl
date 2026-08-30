/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ucx_staged_pool.h"

namespace {

class StagedPoolTest : public testing::Test {
protected:
    static constexpr size_t kSlotSize = 64;
    static constexpr size_t kSlotStride = 64;

    std::unique_ptr<nixlUcxStagedSlotPool>
    makePool(size_t tx_count,
             size_t rx_count,
             uint64_t lease_timeout_us = 10,
             size_t max_grants = 0) {
        slab_.assign((tx_count + rx_count) * kSlotStride, std::byte{0});
        nixlUcxStagedSlotPool::Backing backing{.base = slab_.data()};
        return std::make_unique<nixlUcxStagedSlotPool>(0,
                                                       17,
                                                       kSlotSize,
                                                       kSlotStride,
                                                       tx_count,
                                                       rx_count,
                                                       lease_timeout_us,
                                                       max_grants,
                                                       std::move(backing),
                                                       [this] { return nowUs_; });
    }

    static nixlUcxStagedSlotGrant
    reserve(nixlUcxStagedSlotPool &pool,
            const std::string &owner,
            uint64_t transfer,
            uint64_t chunk,
            uint64_t token = 7) {
        return pool.reserveRxSlot(owner, transfer, chunk, token, 0x1000, 0, kSlotSize);
    }

    static nixl_status_t
    begin(nixlUcxStagedSlotPool &pool,
          const std::string &owner,
          uint64_t transfer,
          uint64_t chunk,
          const nixlUcxStagedSlotGrant &grant) {
        nixlUcxStagedReadyLease ready;
        return pool.beginRemoteH2D(owner,
                                   transfer,
                                   chunk,
                                   grant.slotId,
                                   grant.leaseId,
                                   0x1000,
                                   0,
                                   kSlotSize,
                                   ready);
    }

    uint64_t nowUs_ = 100;
    std::vector<std::byte> slab_;
};

TEST_F(StagedPoolTest, TxAndRxPartitionsAreIsolated) {
    auto pool = makePool(2, 2);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_SUCCESS);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_SUCCESS);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_IN_PROG);
    EXPECT_EQ(reserve(*pool, "peer", 1, 1).status, NIXL_SUCCESS);

    auto other = makePool(2, 2);
    EXPECT_EQ(reserve(*other, "peer", 2, 1).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*other, "peer", 2, 2).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*other, "peer", 2, 3).status, NIXL_IN_PROG);
    EXPECT_EQ(other->acquireTxSlot().status, NIXL_SUCCESS);
}

TEST_F(StagedPoolTest, ExpiredReservedLeasesAreQuarantinedAndNeverRegranted) {
    auto pool = makePool(1, 2);
    const auto first = reserve(*pool, "peer", 1, 1);
    const auto second = reserve(*pool, "peer", 1, 2);
    ASSERT_EQ(first.status, NIXL_SUCCESS);
    ASSERT_EQ(second.status, NIXL_SUCCESS);

    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "peer", 1, 3).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->rxState(first.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(pool->rxState(second.slotId), nixlUcxStagedSlotState::QUARANTINED);
}

TEST_F(StagedPoolTest, LateMatchingReadyCannotReviveQuarantinedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1, 11);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_IN_PROG);

    EXPECT_EQ(begin(*pool, "other", 1, 1, grant), NIXL_ERR_MISMATCH);
    EXPECT_EQ(begin(*pool, "peer", 1, 1, grant), NIXL_ERR_MISMATCH);
    pool->finishRemoteLease(grant.slotId, grant.leaseId, NIXL_SUCCESS);
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_IN_PROG);
}

TEST_F(StagedPoolTest, LateReleaseCannotFreeQuarantinedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_IN_PROG);

    EXPECT_TRUE(pool->quarantineRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_IN_PROG);
}

TEST_F(StagedPoolTest, SlotReleaseQuarantinesReservedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);

    EXPECT_TRUE(pool->quarantineRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));

    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "other", 2, 1).status, NIXL_IN_PROG);
}

TEST_F(StagedPoolTest, SafeCancelCanReuseRxSlotRepeatedly) {
    auto pool = makePool(1, 4, 10, 4);
    for (uint64_t transfer = 1; transfer <= 20; ++transfer) {
        const auto grant = reserve(*pool, "peer", transfer, 1);
        ASSERT_EQ(grant.status, NIXL_SUCCESS);
        ASSERT_TRUE(
            pool->cancelRemoteLease("peer", transfer, 1, grant.slotId, grant.leaseId));
        EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::FREE);
    }
}

TEST_F(StagedPoolTest, SafeCancelCannotReviveStartedH2DLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    ASSERT_EQ(begin(*pool, "peer", 1, 1, grant), NIXL_SUCCESS);

    EXPECT_FALSE(pool->cancelRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::REMOTE_H2D);
}

TEST_F(StagedPoolTest, QuarantineDoesNotLeaveGhostGrantQuota) {
    auto pool = makePool(1, 4, 10, 2);
    const auto a = reserve(*pool, "a", 1, 1);
    ASSERT_EQ(a.status, NIXL_SUCCESS);
    ASSERT_TRUE(pool->quarantineRemoteLease("a", 1, 1, a.slotId, a.leaseId));

    ASSERT_EQ(reserve(*pool, "b", 2, 1).status, NIXL_SUCCESS);
    ASSERT_EQ(reserve(*pool, "b", 2, 2).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "b", 2, 3).status, NIXL_SUCCESS);
}

TEST_F(StagedPoolTest, OwnerDisconnectQuarantinesReservedLeaseOnly) {
    auto pool = makePool(1, 4, 10, 4);
    const auto a_reserved = reserve(*pool, "a", 1, 1);
    const auto a_h2d = reserve(*pool, "a", 1, 2);
    const auto b_reserved = reserve(*pool, "b", 2, 1);
    ASSERT_EQ(begin(*pool, "a", 1, 2, a_h2d), NIXL_SUCCESS);

    EXPECT_EQ(pool->quarantineLeasesForOwner("a"), 1u);
    EXPECT_EQ(pool->rxState(a_reserved.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(pool->rxState(a_h2d.slotId), nixlUcxStagedSlotState::REMOTE_H2D);
    EXPECT_EQ(pool->rxState(b_reserved.slotId), nixlUcxStagedSlotState::REMOTE_RESERVED);
}

TEST_F(StagedPoolTest, ErrorLeaseIsReclaimedUnderPressure) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "a", 1, 1);
    ASSERT_EQ(begin(*pool, "a", 1, 1, grant), NIXL_SUCCESS);
    pool->finishRemoteLease(grant.slotId, grant.leaseId, NIXL_ERR_BACKEND);
    ASSERT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::ERROR);

    const auto replacement = reserve(*pool, "b", 2, 1);
    EXPECT_EQ(replacement.status, NIXL_SUCCESS);
    EXPECT_EQ(replacement.slotId, grant.slotId);
    EXPECT_NE(replacement.leaseId, grant.leaseId);
}

TEST_F(StagedPoolTest, PerAgentCapAppliesOnlyWhenAnotherAgentIsActive) {
    auto pool = makePool(1, 4);
    const auto a_first = reserve(*pool, "a", 1, 1);
    const auto a_second = reserve(*pool, "a", 1, 2);
    const auto b = reserve(*pool, "b", 2, 1);
    ASSERT_EQ(a_first.status, NIXL_SUCCESS);
    ASSERT_EQ(a_second.status, NIXL_SUCCESS);
    ASSERT_EQ(b.status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "a", 1, 3).status, NIXL_IN_PROG);

    ASSERT_EQ(begin(*pool, "b", 2, 1, b), NIXL_SUCCESS);
    pool->finishRemoteLease(b.slotId, b.leaseId, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "a", 1, 3).status, NIXL_SUCCESS);
}

TEST_F(StagedPoolTest, ZeroSizedPartitionsReturnRoleErrors) {
    auto rx_only = makePool(0, 1);
    EXPECT_EQ(rx_only->acquireTxSlot().status, NIXL_ERR_NOT_SUPPORTED);

    auto tx_only = makePool(1, 0);
    EXPECT_EQ(reserve(*tx_only, "peer", 1, 1).status, NIXL_ERR_NOT_SUPPORTED);
}

TEST_F(StagedPoolTest, TxGenerationAdvancesAtPoolScopeOnReuse) {
    auto pool = makePool(1, 1);
    const auto first = pool->acquireTxSlot();
    ASSERT_EQ(first.status, NIXL_SUCCESS);
    EXPECT_EQ(pool->txGeneration(first.slotId), 1u);
    pool->releaseTxSlot(first.slotId);

    const auto second = pool->acquireTxSlot();
    ASSERT_EQ(second.status, NIXL_SUCCESS);
    EXPECT_EQ(second.slotId, first.slotId);
    EXPECT_EQ(pool->txGeneration(second.slotId), 2u);
}

TEST_F(StagedPoolTest, QuarantinedTxSlotIsNeverReused) {
    auto pool = makePool(1, 1);
    const auto slot = pool->acquireTxSlot();
    ASSERT_EQ(slot.status, NIXL_SUCCESS);

    pool->quarantineTxSlot(slot.slotId);

    EXPECT_EQ(pool->txState(slot.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_IN_PROG);
    pool->releaseTxSlot(slot.slotId);
    EXPECT_EQ(pool->txState(slot.slotId), nixlUcxStagedSlotState::QUARANTINED);
}

TEST_F(StagedPoolTest, TokenBookkeepingSurvivesReserveBeginAndQuarantine) {
    auto pool = makePool(1, 2);
    const auto h2d = reserve(*pool, "a", 1, 1, 41);
    const auto quarantine = reserve(*pool, "b", 2, 1, 42);
    ASSERT_TRUE(pool->hasLeasesForToken(41));
    ASSERT_TRUE(pool->hasLeasesForToken(42));
    EXPECT_EQ(pool->rxRegionToken(h2d.slotId), 41u);
    ASSERT_EQ(begin(*pool, "a", 1, 1, h2d), NIXL_SUCCESS);
    EXPECT_TRUE(pool->hasLeasesForToken(41));

    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "c", 3, 1, 43).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->rxState(quarantine.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_TRUE(pool->hasLeasesForToken(42));

    pool->finishRemoteLease(h2d.slotId, h2d.leaseId, NIXL_SUCCESS);
    EXPECT_FALSE(pool->hasLeasesForToken(41));
    EXPECT_TRUE(pool->hasLeasesForToken(42));
}

} // namespace
