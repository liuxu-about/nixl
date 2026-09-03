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
             size_t max_grants = 0,
             uint64_t waiter_timeout_us = 1000000) {
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
                                                       [this] { return nowUs_; },
                                                       waiter_timeout_us);
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
    EXPECT_EQ(reserve(*pool, "peer", 1, 3).status, NIXL_ERR_BACKEND);
    EXPECT_EQ(pool->rxState(first.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(pool->rxState(second.slotId), nixlUcxStagedSlotState::QUARANTINED);
}

TEST_F(StagedPoolTest, LateMatchingReadyCannotReviveQuarantinedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1, 11);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_ERR_BACKEND);

    EXPECT_EQ(begin(*pool, "other", 1, 1, grant), NIXL_ERR_MISMATCH);
    EXPECT_EQ(begin(*pool, "peer", 1, 1, grant), NIXL_ERR_MISMATCH);
    pool->finishRemoteLease(grant.slotId, grant.leaseId, NIXL_SUCCESS);
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_ERR_BACKEND);
}

TEST_F(StagedPoolTest, LateReleaseCannotFreeQuarantinedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_ERR_BACKEND);

    EXPECT_TRUE(pool->quarantineRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "peer", 2, 1).status, NIXL_ERR_BACKEND);
}

TEST_F(StagedPoolTest, SlotReleaseQuarantinesReservedLease) {
    auto pool = makePool(1, 1);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);

    EXPECT_TRUE(pool->quarantineRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));

    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(reserve(*pool, "other", 2, 1).status, NIXL_ERR_BACKEND);
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
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_ERR_BACKEND);
    pool->releaseTxSlot(slot.slotId);
    EXPECT_EQ(pool->txState(slot.slotId), nixlUcxStagedSlotState::QUARANTINED);
}

TEST_F(StagedPoolTest, MixedActiveAndQuarantinedTxSlotsRemainTransient) {
    auto pool = makePool(2, 1);
    const auto quarantined = pool->acquireTxSlot();
    const auto active = pool->acquireTxSlot();
    ASSERT_EQ(quarantined.status, NIXL_SUCCESS);
    ASSERT_EQ(active.status, NIXL_SUCCESS);

    pool->quarantineTxSlot(quarantined.slotId);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_IN_PROG);

    pool->releaseTxSlot(active.slotId);
    EXPECT_EQ(pool->acquireTxSlot().status, NIXL_SUCCESS);
}

TEST_F(StagedPoolTest, MixedActiveAndQuarantinedRxSlotsRemainTransient) {
    auto pool = makePool(1, 2, 0, 2);
    const auto quarantined = reserve(*pool, "a", 1, 1);
    const auto active = reserve(*pool, "b", 2, 1);
    ASSERT_EQ(quarantined.status, NIXL_SUCCESS);
    ASSERT_EQ(active.status, NIXL_SUCCESS);
    ASSERT_TRUE(pool->quarantineRemoteLease(
        "a", 1, 1, quarantined.slotId, quarantined.leaseId));

    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_IN_PROG);

    ASSERT_EQ(begin(*pool, "b", 2, 1, active), NIXL_SUCCESS);
    pool->finishRemoteLease(active.slotId, active.leaseId, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_SUCCESS);
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

// --- transport hardening: idempotent control messages, reclaim, FIFO ---------

TEST_F(StagedPoolTest, RepeatedSlotReqReplaysSameGrant) {
    auto pool = makePool(0, 2);
    const auto first = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(first.status, NIXL_SUCCESS);
    const auto again = reserve(*pool, "peer", 1, 1);
    EXPECT_EQ(again.status, NIXL_SUCCESS);
    EXPECT_EQ(again.slotId, first.slotId);
    EXPECT_EQ(again.leaseId, first.leaseId);
    EXPECT_EQ(pool->grantReplays(), 1u);
    // The replay did not consume a second slot.
    EXPECT_EQ(reserve(*pool, "peer", 1, 2).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "peer", 1, 3).status, NIXL_IN_PROG);
}

TEST_F(StagedPoolTest, RepeatedSlotReqAfterH2DStartedStillReplays) {
    auto pool = makePool(0, 2);
    const auto first = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(first.status, NIXL_SUCCESS);
    ASSERT_EQ(begin(*pool, "peer", 1, 1, first), NIXL_SUCCESS);
    const auto again = reserve(*pool, "peer", 1, 1);
    EXPECT_EQ(again.status, NIXL_SUCCESS);
    EXPECT_EQ(again.slotId, first.slotId);
    EXPECT_EQ(again.leaseId, first.leaseId);
}

TEST_F(StagedPoolTest, RetransmittedReadyReplaysCompletedStatus) {
    auto pool = makePool(0, 2);
    const auto ok = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(begin(*pool, "peer", 1, 1, ok), NIXL_SUCCESS);
    EXPECT_TRUE(pool->leaseInProgress("peer", 1, 1, ok.slotId, ok.leaseId));
    nixl_status_t replay = NIXL_IN_PROG;
    EXPECT_FALSE(pool->completedLeaseStatus("peer", 1, 1, ok.slotId, ok.leaseId, replay));
    pool->finishRemoteLease(ok.slotId, ok.leaseId, NIXL_SUCCESS);
    EXPECT_FALSE(pool->leaseInProgress("peer", 1, 1, ok.slotId, ok.leaseId));
    EXPECT_TRUE(pool->completedLeaseStatus("peer", 1, 1, ok.slotId, ok.leaseId, replay));
    EXPECT_EQ(replay, NIXL_SUCCESS);
    // A second READY for the same lease is a mismatch for beginRemoteH2D ...
    EXPECT_EQ(begin(*pool, "peer", 1, 1, ok), NIXL_ERR_MISMATCH);
    // ... and the slot is free for the next key.
    EXPECT_EQ(pool->rxState(ok.slotId), nixlUcxStagedSlotState::FREE);

    const auto bad = reserve(*pool, "peer", 2, 1);
    ASSERT_EQ(begin(*pool, "peer", 2, 1, bad), NIXL_SUCCESS);
    pool->finishRemoteLease(bad.slotId, bad.leaseId, NIXL_ERR_BACKEND);
    EXPECT_TRUE(pool->completedLeaseStatus("peer", 2, 1, bad.slotId, bad.leaseId, replay));
    EXPECT_EQ(replay, NIXL_ERR_BACKEND);
    // Wrong lease id or owner never matches.
    EXPECT_FALSE(pool->completedLeaseStatus("peer", 2, 1, bad.slotId, bad.leaseId + 7, replay));
    EXPECT_FALSE(pool->completedLeaseStatus("other", 2, 1, bad.slotId, bad.leaseId, replay));
}

TEST_F(StagedPoolTest, ExpiryRunsBeforeQuotaCheck) {
    auto pool = makePool(0, 4, /*lease_timeout_us=*/10, /*max_grants=*/2);
    EXPECT_EQ(reserve(*pool, "a", 1, 1).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "a", 1, 2).status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "b", 9, 1).status, NIXL_SUCCESS);
    // "a" is at its quota while "b" is active.
    EXPECT_EQ(reserve(*pool, "a", 1, 3).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->quotaWaits(), 1u);
    // Both of "a"'s grants expire: the reclaim must run before the quota check
    // so the expired grants stop counting against "a".
    nowUs_ += 11;
    EXPECT_EQ(reserve(*pool, "a", 1, 3).status, NIXL_SUCCESS);
    EXPECT_GE(pool->leasesExpired(), 2u);
    EXPECT_EQ(pool->rxState(0), nixlUcxStagedSlotState::QUARANTINED);
    EXPECT_EQ(pool->rxState(1), nixlUcxStagedSlotState::QUARANTINED);
}

TEST_F(StagedPoolTest, ReclaimExpiredLeasesRunsWithoutAllocationPressure) {
    auto pool = makePool(0, 2, /*lease_timeout_us=*/10);
    size_t callbacks = 0;
    pool->setQuarantineCallback(
        [&](size_t, const std::string &, uint64_t, uint64_t, uint64_t) { ++callbacks; });
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    EXPECT_EQ(pool->reclaimExpiredLeases(), 0u);
    nowUs_ += 11;
    EXPECT_EQ(pool->reclaimExpiredLeases(), 1u);
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(pool->leasesExpired(), 1u);
    EXPECT_EQ(pool->rxState(grant.slotId), nixlUcxStagedSlotState::QUARANTINED);
    // Started H2D leases are never reclaimed by the timer.
    const auto busy = reserve(*pool, "peer", 2, 1);
    ASSERT_EQ(begin(*pool, "peer", 2, 1, busy), NIXL_SUCCESS);
    nowUs_ += 100;
    EXPECT_EQ(pool->reclaimExpiredLeases(), 0u);
    EXPECT_EQ(pool->rxState(busy.slotId), nixlUcxStagedSlotState::REMOTE_H2D);
}

TEST_F(StagedPoolTest, FifoWaitersGetFreedSlotInArrivalOrder) {
    auto pool = makePool(0, 1, /*lease_timeout_us=*/1000000);
    const auto held = reserve(*pool, "a", 1, 1);
    ASSERT_EQ(held.status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "b", 2, 1).status, NIXL_IN_PROG);
    nowUs_ += 1;
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->waiterCount(), 2u);
    ASSERT_EQ(begin(*pool, "a", 1, 1, held), NIXL_SUCCESS);
    pool->finishRemoteLease(held.slotId, held.leaseId, NIXL_SUCCESS);
    // The slot is free, but "c" is behind "b" in the queue.
    nowUs_ += 1;
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->fifoWaits(), 1u);
    // A newcomer queues behind both.
    EXPECT_EQ(reserve(*pool, "d", 4, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->waiterCount(), 3u);
    const auto b = reserve(*pool, "b", 2, 1);
    EXPECT_EQ(b.status, NIXL_SUCCESS);
    EXPECT_EQ(pool->waiterCount(), 2u);
    ASSERT_EQ(begin(*pool, "b", 2, 1, b), NIXL_SUCCESS);
    pool->finishRemoteLease(b.slotId, b.leaseId, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "d", 4, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_SUCCESS);
    EXPECT_EQ(pool->waiterCount(), 1u);
}

TEST_F(StagedPoolTest, StaleWaiterIsPrunedFromQueueHead) {
    auto pool = makePool(0, 1, /*lease_timeout_us=*/1000000, 0, /*waiter_timeout_us=*/50);
    const auto held = reserve(*pool, "a", 1, 1);
    ASSERT_EQ(held.status, NIXL_SUCCESS);
    EXPECT_EQ(reserve(*pool, "b", 2, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->waiterCount(), 1u);
    ASSERT_EQ(begin(*pool, "a", 1, 1, held), NIXL_SUCCESS);
    pool->finishRemoteLease(held.slotId, held.leaseId, NIXL_SUCCESS);
    // "b" stopped polling; once its entry is stale a newcomer is served.
    nowUs_ += 51;
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_SUCCESS);
    EXPECT_EQ(pool->waiterCount(), 0u);
}

TEST_F(StagedPoolTest, QuotaBlockedRequesterDoesNotHoldQueueHead) {
    auto pool = makePool(0, 2, /*lease_timeout_us=*/1000000, /*max_grants=*/1);
    EXPECT_EQ(reserve(*pool, "a", 1, 1).status, NIXL_SUCCESS);
    const auto b = reserve(*pool, "b", 2, 1);
    ASSERT_EQ(b.status, NIXL_SUCCESS);
    // Pool is full; "a" asks for a second slot: quota-blocked, not queued.
    EXPECT_EQ(reserve(*pool, "a", 1, 2).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->waiterCount(), 0u);
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_IN_PROG);
    EXPECT_EQ(pool->waiterCount(), 1u);
    ASSERT_EQ(begin(*pool, "b", 2, 1, b), NIXL_SUCCESS);
    pool->finishRemoteLease(b.slotId, b.leaseId, NIXL_SUCCESS);
    // "a" is still over quota only while another agent holds a grant; "c" is
    // the head waiter and gets the freed slot.
    EXPECT_EQ(reserve(*pool, "a", 1, 2).status, NIXL_IN_PROG);
    EXPECT_EQ(reserve(*pool, "c", 3, 1).status, NIXL_SUCCESS);
}

TEST_F(StagedPoolTest, SafeCancelledLeaseIsNotReplayed) {
    auto pool = makePool(0, 2);
    const auto grant = reserve(*pool, "peer", 1, 1);
    ASSERT_EQ(grant.status, NIXL_SUCCESS);
    EXPECT_TRUE(pool->cancelRemoteLease("peer", 1, 1, grant.slotId, grant.leaseId));
    const auto fresh = reserve(*pool, "peer", 1, 1);
    EXPECT_EQ(fresh.status, NIXL_SUCCESS);
    EXPECT_NE(fresh.leaseId, grant.leaseId);
    EXPECT_EQ(pool->grantReplays(), 0u);
}

} // namespace
