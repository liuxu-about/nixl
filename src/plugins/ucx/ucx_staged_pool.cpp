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
#include "ucx_staged_pool.h"

#include <algorithm>
#include <limits>
#include <chrono>
#include <utility>

namespace {

[[nodiscard]] uint64_t
steadyNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

void
nixlUcxStagedSlotPool::RxLease::reset() {
    state = nixlUcxStagedSlotState::FREE;
    ownerAgent.clear();
    transferId = 0;
    chunkId = 0;
    leaseId = 0;
    regionToken = 0;
    gpuAddr = 0;
    gpuDev = 0;
    size = 0;
    grantedUs = 0;
    grantCycle = 0;
}

nixlUcxStagedSlotPool::nixlUcxStagedSlotPool(uint64_t gpu_dev_id,
                                             uint64_t pool_epoch,
                                             size_t slot_size,
                                             size_t slot_stride,
                                             size_t tx_count,
                                             size_t rx_count,
                                             uint64_t lease_timeout_us,
                                             size_t max_grants_per_agent,
                                             Backing pool_backing,
                                             NowUs now_us,
                                             uint64_t waiter_timeout_us,
                                             bool fifo_admission)
    : gpuDevId(gpu_dev_id),
      poolEpoch(pool_epoch),
      slotSize(slot_size),
      slotStride(slot_stride),
      txCount(tx_count),
      rxCount(rx_count),
      leaseTimeoutUs(lease_timeout_us),
      maxGrantsPerAgent(max_grants_per_agent == 0 ?
                           std::max<size_t>(1, rx_count / 2) :
                           max_grants_per_agent),
      waiterTimeoutUs(waiter_timeout_us),
      fifoAdmission(fifo_admission),
      backing(std::move(pool_backing)),
      txStates_(tx_count, nixlUcxStagedSlotState::FREE),
      txGenerations_(tx_count, 0),
      rxLeases_(rx_count),
      nowUs_(now_us ? std::move(now_us) : NowUs(steadyNowUs)),
      completed_(kCompletedRingSize) {}

nixlUcxStagedTxSlot
nixlUcxStagedSlotPool::acquireTxSlot() {
    const std::lock_guard lock(mutex_);
    if (txCount == 0) {
        return {.status = NIXL_ERR_NOT_SUPPORTED};
    }

    for (size_t i = 0; i < txStates_.size(); ++i) {
        if (txStates_[i] != nixlUcxStagedSlotState::FREE) {
            continue;
        }
        txStates_[i] = nixlUcxStagedSlotState::LOCAL_D2H;
        ++txGenerations_[i];
        if (txGenerations_[i] == 0) {
            txGenerations_[i] = 1;
        }
        return {.slotId = static_cast<uint64_t>(i), .status = NIXL_SUCCESS};
    }

    const bool has_active_slot =
        std::any_of(txStates_.begin(), txStates_.end(), [](const auto state) {
            return state == nixlUcxStagedSlotState::LOCAL_D2H;
        });
    if (!has_active_slot) {
        // No slot can become FREE without rebuilding this process-local pool.
        // Do not report permanent quarantine exhaustion as transient backpressure.
        return {.status = NIXL_ERR_BACKEND};
    }
    return {.status = NIXL_IN_PROG};
}

void
nixlUcxStagedSlotPool::releaseTxSlot(size_t slot_id) {
    const std::lock_guard lock(mutex_);
    if (slot_id < txStates_.size() &&
        txStates_[slot_id] == nixlUcxStagedSlotState::LOCAL_D2H) {
        txStates_[slot_id] = nixlUcxStagedSlotState::FREE;
    }
}

void
nixlUcxStagedSlotPool::quarantineTxSlot(size_t slot_id) {
    const std::lock_guard lock(mutex_);
    if (slot_id < txStates_.size() &&
        txStates_[slot_id] == nixlUcxStagedSlotState::LOCAL_D2H) {
        txStates_[slot_id] = nixlUcxStagedSlotState::QUARANTINED;
    }
}

uint64_t
nixlUcxStagedSlotPool::txGeneration(size_t slot_id) const {
    const std::lock_guard lock(mutex_);
    return slot_id < txGenerations_.size() ? txGenerations_[slot_id] : 0;
}

nixlUcxStagedSlotGrant
nixlUcxStagedSlotPool::grantRxSlot(size_t slot_id,
                                   const std::string &owner_agent,
                                   uint64_t transfer_id,
                                   uint64_t chunk_id,
                                   uint64_t region_token,
                                   uintptr_t gpu_addr,
                                   uint64_t gpu_dev,
                                   size_t size,
                                   uint64_t grant_cycle) {
    auto &lease = rxLeases_[slot_id];
    lease.reset();
    lease.state = nixlUcxStagedSlotState::REMOTE_RESERVED;
    lease.ownerAgent = owner_agent;
    lease.transferId = transfer_id;
    lease.chunkId = chunk_id;
    lease.leaseId = nextLeaseId_++;
    if (lease.leaseId == 0) {
        lease.leaseId = nextLeaseId_++;
    }
    lease.regionToken = region_token;
    lease.gpuAddr = gpu_addr;
    lease.gpuDev = gpu_dev;
    lease.size = size;
    lease.grantedUs = nowUs_();
    lease.grantCycle = grant_cycle;
    ++rxGrantsPerAgent_[owner_agent];
    return {.slotId = static_cast<uint64_t>(slot_id),
            .leaseId = lease.leaseId,
            .status = NIXL_SUCCESS};
}

bool
nixlUcxStagedSlotPool::quotaBlocks(const std::string &owner_agent) const {
    const auto owner_it = rxGrantsPerAgent_.find(owner_agent);
    const size_t owner_count = owner_it == rxGrantsPerAgent_.end() ? 0 : owner_it->second;
    if (owner_count < maxGrantsPerAgent) {
        return false;
    }

    return std::any_of(rxGrantsPerAgent_.begin(),
                       rxGrantsPerAgent_.end(),
                       [&](const auto &entry) {
                           return entry.first != owner_agent && entry.second != 0;
                       });
}

nixlUcxStagedSlotGrant
nixlUcxStagedSlotPool::reserveRxSlot(const std::string &owner_agent,
                                     uint64_t transfer_id,
                                     uint64_t chunk_id,
                                     uint64_t region_token,
                                     uintptr_t gpu_addr,
                                     uint64_t gpu_dev,
                                     size_t size,
                                     uint64_t grant_cycle) {
    const std::lock_guard lock(mutex_);
    if (rxCount == 0) {
        return {.status = NIXL_ERR_NOT_SUPPORTED};
    }
    if (size == 0 || size > slotSize) {
        return {.status = NIXL_ERR_INVALID_PARAM};
    }
    const uint64_t now_us = nowUs_();

    // Idempotent SLOT_REQ: a retransmit for a key that already holds a lease is
    // answered with the same grant. The initiator may have missed the first
    // GRANT; handing out a second slot would leak the first one until the lease
    // timeout quarantines it.
    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        auto &lease = rxLeases_[i];
        if ((lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED ||
             lease.state == nixlUcxStagedSlotState::REMOTE_H2D) &&
            lease.ownerAgent == owner_agent && lease.transferId == transfer_id &&
            lease.chunkId == chunk_id) {
            if (lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED &&
                grant_cycle > lease.grantCycle) {
                // The initiator abandoned this grant (its SAFE_CANCEL may still be
                // in flight or reordered behind this request). Release it here so
                // the abandoned lease is never handed back.
                decrementGrantCount(lease.ownerAgent);
                lease.reset();
                break;
            }
            grantReplays_.fetch_add(1, std::memory_order_relaxed);
            const auto waiter = findWaiterLocked(owner_agent, transfer_id, chunk_id);
            if (waiter != waiters_.end()) {
                waiters_.erase(waiter);
            }
            return {.slotId = static_cast<uint64_t>(i),
                    .leaseId = lease.leaseId,
                    .status = NIXL_SUCCESS};
        }
    }

    // Reclaim before the quota check: an agent whose grants all expired must
    // not be judged over quota by leases that are about to be quarantined.
    reclaimExpiredLeasesLocked(now_us);
    pruneWaitersLocked(now_us);

    if (quotaBlocks(owner_agent)) {
        quotaWaits_.fetch_add(1, std::memory_order_relaxed);
        const auto waiter = findWaiterLocked(owner_agent, transfer_id, chunk_id);
        if (waiter != waiters_.end()) {
            waiters_.erase(waiter);
        }
        return {.status = NIXL_IN_PROG};
    }

    size_t free_slot = rxLeases_.size();
    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        if (rxLeases_[i].state == nixlUcxStagedSlotState::FREE) {
            free_slot = i;
            break;
        }
    }

    auto waiter = findWaiterLocked(owner_agent, transfer_id, chunk_id);
    if (waiter != waiters_.end()) {
        waiter->lastSeenUs = now_us;
    }

    if (free_slot == rxLeases_.size()) {
        const bool has_active_lease =
            std::any_of(rxLeases_.begin(), rxLeases_.end(), [](const auto &lease) {
                return lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED ||
                       lease.state == nixlUcxStagedSlotState::REMOTE_H2D;
            });
        if (!has_active_lease) {
            return {.status = NIXL_ERR_BACKEND};
        }
        if (waiter == waiters_.end()) {
            waiters_.push_back({owner_agent, transfer_id, chunk_id, now_us});
        }
        return {.status = NIXL_IN_PROG};
    }

    // FIFO admission: a free slot goes to the oldest live waiter. Requesters that
    // never waited queue behind it; waiters that stop polling are pruned after
    // waiterTimeoutUs so a dead initiator cannot hold the head of the queue.
    if (fifoAdmission && !waiters_.empty() && waiter != waiters_.begin()) {
        if (waiter == waiters_.end()) {
            waiters_.push_back({owner_agent, transfer_id, chunk_id, now_us});
        }
        fifoWaits_.fetch_add(1, std::memory_order_relaxed);
        return {.status = NIXL_IN_PROG};
    }
    if (waiter != waiters_.end()) {
        waiters_.erase(waiter);
    }
    return grantRxSlot(free_slot,
                       owner_agent,
                       transfer_id,
                       chunk_id,
                       region_token,
                       gpu_addr,
                       gpu_dev,
                       size,
                       grant_cycle);
}

nixl_status_t
nixlUcxStagedSlotPool::beginRemoteH2D(const std::string &owner_agent,
                                      uint64_t transfer_id,
                                      uint64_t chunk_id,
                                      uint64_t slot_id,
                                      uint64_t lease_id,
                                      uintptr_t gpu_addr,
                                      uint64_t gpu_dev,
                                      size_t size,
                                      nixlUcxStagedReadyLease &ready) {
    const std::lock_guard lock(mutex_);
    if (slot_id >= rxLeases_.size() || size == 0 || size > slotSize) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto &lease = rxLeases_[slot_id];
    if (lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED ||
        lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
        lease.chunkId != chunk_id || lease.leaseId != lease_id ||
        lease.gpuAddr != gpu_addr || lease.gpuDev != gpu_dev || lease.size != size) {
        return NIXL_ERR_MISMATCH;
    }

    lease.state = nixlUcxStagedSlotState::REMOTE_H2D;
    ready.hostAddr = static_cast<char *>(backing.base) + txCount * slotStride +
                     slot_id * slotStride;
    ready.slotId = slot_id;
    ready.leaseId = lease_id;
    return NIXL_SUCCESS;
}

void
nixlUcxStagedSlotPool::decrementGrantCount(const std::string &owner_agent) {
    const auto it = rxGrantsPerAgent_.find(owner_agent);
    if (it == rxGrantsPerAgent_.end()) {
        return;
    }
    if (it->second > 1) {
        --it->second;
    } else {
        rxGrantsPerAgent_.erase(it);
    }
}

void
nixlUcxStagedSlotPool::finishRemoteLease(uint64_t slot_id,
                                         uint64_t lease_id,
                                         nixl_status_t status) {
    const std::lock_guard lock(mutex_);
    if (slot_id >= rxLeases_.size()) {
        return;
    }

    auto &lease = rxLeases_[slot_id];
    if (lease.leaseId != lease_id || lease.state != nixlUcxStagedSlotState::REMOTE_H2D) {
        return;
    }

    decrementGrantCount(lease.ownerAgent);
    rememberCompletedLocked(lease, slot_id, status);
    if (status == NIXL_SUCCESS) {
        lease.reset();
    } else {
        lease.state = nixlUcxStagedSlotState::ERROR;
    }
}

bool
nixlUcxStagedSlotPool::cancelRemoteLease(const std::string &owner_agent,
                                         uint64_t transfer_id,
                                         uint64_t chunk_id,
                                         uint64_t slot_id,
                                         uint64_t lease_id) {
    const std::lock_guard lock(mutex_);
    if (slot_id >= rxLeases_.size()) {
        return false;
    }

    auto &lease = rxLeases_[slot_id];
    if (lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
        lease.chunkId != chunk_id || lease.leaseId != lease_id ||
        lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED) {
        return false;
    }

    decrementGrantCount(lease.ownerAgent);
    lease.reset();
    return true;
}

bool
nixlUcxStagedSlotPool::quarantineRemoteLease(const std::string &owner_agent,
                                             uint64_t transfer_id,
                                             uint64_t chunk_id,
                                             uint64_t slot_id,
                                             uint64_t lease_id) {
    const std::lock_guard lock(mutex_);
    if (slot_id >= rxLeases_.size()) {
        return false;
    }

    auto &lease = rxLeases_[slot_id];
    if (lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
        lease.chunkId != chunk_id || lease.leaseId != lease_id ||
        (lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED &&
         lease.state != nixlUcxStagedSlotState::QUARANTINED)) {
        return false;
    }

    // SLOT_RELEASE can race with a previously posted data-plane write.  The
    // target cannot prove from the control message alone that the old writer
    // has stopped, so retain the lease as a fail-stop quarantine instead of
    // making its address available to a new request.
    if (lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED) {
        decrementGrantCount(lease.ownerAgent);
        lease.state = nixlUcxStagedSlotState::QUARANTINED;
        if (quarantineCallback_) {
            quarantineCallback_(
                slot_id, lease.ownerAgent, lease.transferId, lease.chunkId, lease.leaseId);
        }
    }
    return true;
}

size_t
nixlUcxStagedSlotPool::quarantineLeasesForOwner(const std::string &owner_agent) {
    const std::lock_guard lock(mutex_);
    size_t quarantined = 0;
    for (size_t slot_id = 0; slot_id < rxLeases_.size(); ++slot_id) {
        auto &lease = rxLeases_[slot_id];
        if (lease.ownerAgent != owner_agent ||
            lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED) {
            continue;
        }
        decrementGrantCount(lease.ownerAgent);
        lease.state = nixlUcxStagedSlotState::QUARANTINED;
        ++quarantined;
        if (quarantineCallback_) {
            quarantineCallback_(
                slot_id, lease.ownerAgent, lease.transferId, lease.chunkId, lease.leaseId);
        }
    }
    return quarantined;
}

bool
nixlUcxStagedSlotPool::hasLeasesForToken(uint64_t region_token) const {
    const std::lock_guard lock(mutex_);
    return std::any_of(rxLeases_.begin(), rxLeases_.end(), [&](const auto &lease) {
        return lease.state != nixlUcxStagedSlotState::FREE &&
               lease.regionToken == region_token;
    });
}

bool
nixlUcxStagedSlotPool::hasActiveWork() const {
    const std::lock_guard lock(mutex_);
    return std::any_of(txStates_.begin(), txStates_.end(), [](const auto state) {
               return state != nixlUcxStagedSlotState::FREE;
           }) ||
           std::any_of(rxLeases_.begin(), rxLeases_.end(), [](const auto &lease) {
               return lease.state != nixlUcxStagedSlotState::FREE;
           });
}

size_t
nixlUcxStagedSlotPool::countWritableLeases(uintptr_t gpu_addr,
                                           size_t size,
                                           uint64_t gpu_dev) const {
    if (size == 0) {
        return 0;
    }
    const uintptr_t range_end = size > std::numeric_limits<uintptr_t>::max() - gpu_addr ?
        std::numeric_limits<uintptr_t>::max() :
        gpu_addr + size;
    const std::lock_guard lock(mutex_);
    size_t count = 0;
    for (const auto &lease : rxLeases_) {
        if (lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED &&
            lease.state != nixlUcxStagedSlotState::REMOTE_H2D) {
            continue;
        }
        if (lease.gpuDev != gpu_dev || lease.size == 0) {
            continue;
        }
        const uintptr_t lease_end =
            lease.size > std::numeric_limits<uintptr_t>::max() - lease.gpuAddr ?
            std::numeric_limits<uintptr_t>::max() :
            lease.gpuAddr + lease.size;
        if (lease.gpuAddr < range_end && gpu_addr < lease_end) {
            ++count;
        }
    }
    return count;
}

void *
nixlUcxStagedSlotPool::txHostAddr(size_t slot_id) const {
    if (slot_id >= txCount || backing.base == nullptr) {
        return nullptr;
    }
    return static_cast<char *>(backing.base) + slot_id * slotStride;
}

void *
nixlUcxStagedSlotPool::rxHostAddr(size_t slot_id) const {
    if (slot_id >= rxCount || backing.base == nullptr) {
        return nullptr;
    }
    return static_cast<char *>(backing.base) + txCount * slotStride + slot_id * slotStride;
}

nixlUcxStagedSlotState
nixlUcxStagedSlotPool::txState(size_t slot_id) const {
    const std::lock_guard lock(mutex_);
    return slot_id < txStates_.size() ? txStates_[slot_id] : nixlUcxStagedSlotState::ERROR;
}

nixlUcxStagedSlotState
nixlUcxStagedSlotPool::rxState(size_t slot_id) const {
    const std::lock_guard lock(mutex_);
    return slot_id < rxLeases_.size() ? rxLeases_[slot_id].state :
                                       nixlUcxStagedSlotState::ERROR;
}

uint64_t
nixlUcxStagedSlotPool::rxRegionToken(size_t slot_id) const {
    const std::lock_guard lock(mutex_);
    return slot_id < rxLeases_.size() ? rxLeases_[slot_id].regionToken : 0;
}

void
nixlUcxStagedSlotPool::setQuarantineCallback(QuarantineCallback callback) {
    const std::lock_guard lock(mutex_);
    quarantineCallback_ = std::move(callback);
}

size_t
nixlUcxStagedSlotPool::reclaimExpiredLeasesLocked(uint64_t now_us) {
    size_t quarantined = 0;
    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        auto &lease = rxLeases_[i];
        if (lease.state == nixlUcxStagedSlotState::ERROR) {
            lease.reset();
            continue;
        }
        if (leaseTimeoutUs == 0 || lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED ||
            lease.grantedUs == 0 || now_us - lease.grantedUs < leaseTimeoutUs) {
            continue;
        }
        decrementGrantCount(lease.ownerAgent);
        lease.state = nixlUcxStagedSlotState::QUARANTINED;
        ++quarantined;
        leasesExpired_.fetch_add(1, std::memory_order_relaxed);
        if (quarantineCallback_) {
            quarantineCallback_(
                i, lease.ownerAgent, lease.transferId, lease.chunkId, lease.leaseId);
        }
    }
    return quarantined;
}

size_t
nixlUcxStagedSlotPool::reclaimExpiredLeases() {
    const std::lock_guard lock(mutex_);
    return reclaimExpiredLeasesLocked(nowUs_());
}

void
nixlUcxStagedSlotPool::pruneWaitersLocked(uint64_t now_us) {
    while (!waiters_.empty() && now_us - waiters_.front().lastSeenUs > waiterTimeoutUs) {
        waiters_.pop_front();
    }
    for (auto it = waiters_.begin(); it != waiters_.end();) {
        if (now_us - it->lastSeenUs > waiterTimeoutUs) {
            it = waiters_.erase(it);
        } else {
            ++it;
        }
    }
}

std::deque<nixlUcxStagedSlotPool::Waiter>::iterator
nixlUcxStagedSlotPool::findWaiterLocked(const std::string &owner_agent,
                                        uint64_t transfer_id,
                                        uint64_t chunk_id) {
    return std::find_if(waiters_.begin(), waiters_.end(), [&](const Waiter &waiter) {
        return waiter.transferId == transfer_id && waiter.chunkId == chunk_id &&
               waiter.ownerAgent == owner_agent;
    });
}

void
nixlUcxStagedSlotPool::rememberCompletedLocked(const RxLease &lease,
                                               uint64_t slot_id,
                                               nixl_status_t status) {
    if (completed_.empty()) {
        return;
    }
    CompletedLease &entry = completed_[completedNext_];
    completedNext_ = (completedNext_ + 1) % completed_.size();
    entry.ownerAgent = lease.ownerAgent;
    entry.transferId = lease.transferId;
    entry.chunkId = lease.chunkId;
    entry.slotId = slot_id;
    entry.leaseId = lease.leaseId;
    entry.status = status;
}

bool
nixlUcxStagedSlotPool::completedLeaseStatus(const std::string &owner_agent,
                                            uint64_t transfer_id,
                                            uint64_t chunk_id,
                                            uint64_t slot_id,
                                            uint64_t lease_id,
                                            nixl_status_t &status) const {
    const std::lock_guard lock(mutex_);
    for (const auto &entry : completed_) {
        if (entry.leaseId == lease_id && entry.slotId == slot_id &&
            entry.transferId == transfer_id && entry.chunkId == chunk_id &&
            entry.ownerAgent == owner_agent) {
            status = entry.status;
            return true;
        }
    }
    return false;
}

bool
nixlUcxStagedSlotPool::leaseInProgress(const std::string &owner_agent,
                                       uint64_t transfer_id,
                                       uint64_t chunk_id,
                                       uint64_t slot_id,
                                       uint64_t lease_id) const {
    const std::lock_guard lock(mutex_);
    if (slot_id >= rxLeases_.size()) {
        return false;
    }
    const auto &lease = rxLeases_[slot_id];
    return lease.state == nixlUcxStagedSlotState::REMOTE_H2D && lease.leaseId == lease_id &&
           lease.transferId == transfer_id && lease.chunkId == chunk_id &&
           lease.ownerAgent == owner_agent;
}

size_t
nixlUcxStagedSlotPool::waiterCount() const {
    const std::lock_guard lock(mutex_);
    return waiters_.size();
}

nixlUcxStagedReadyRetry
nixlUcxStagedSlotPool::classifyReadyRetry(const std::string &owner_agent,
                                          uint64_t transfer_id,
                                          uint64_t chunk_id,
                                          uint64_t slot_id,
                                          uint64_t lease_id,
                                          nixl_status_t &completed_status) const {
    const std::lock_guard lock(mutex_);
    if (slot_id < rxLeases_.size()) {
        const auto &lease = rxLeases_[slot_id];
        if (lease.state == nixlUcxStagedSlotState::REMOTE_H2D && lease.leaseId == lease_id &&
            lease.transferId == transfer_id && lease.chunkId == chunk_id &&
            lease.ownerAgent == owner_agent) {
            return nixlUcxStagedReadyRetry::IN_PROGRESS;
        }
    }
    for (const auto &entry : completed_) {
        if (entry.leaseId == lease_id && entry.slotId == slot_id &&
            entry.transferId == transfer_id && entry.chunkId == chunk_id &&
            entry.ownerAgent == owner_agent) {
            completed_status = entry.status;
            return nixlUcxStagedReadyRetry::COMPLETED;
        }
    }
    return nixlUcxStagedReadyRetry::UNKNOWN;
}
