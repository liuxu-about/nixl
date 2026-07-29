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
                                             NowUs now_us)
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
      backing(std::move(pool_backing)),
      txStates_(tx_count, nixlUcxStagedSlotState::FREE),
      txGenerations_(tx_count, 0),
      rxLeases_(rx_count),
      nowUs_(now_us ? std::move(now_us) : NowUs(steadyNowUs)) {}

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
                                   size_t size) {
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
                                     size_t size) {
    const std::lock_guard lock(mutex_);
    if (rxCount == 0) {
        return {.status = NIXL_ERR_NOT_SUPPORTED};
    }
    if (size == 0 || size > slotSize) {
        return {.status = NIXL_ERR_INVALID_PARAM};
    }
    if (quotaBlocks(owner_agent)) {
        return {.status = NIXL_IN_PROG};
    }

    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        if (rxLeases_[i].state == nixlUcxStagedSlotState::FREE) {
            return grantRxSlot(i,
                               owner_agent,
                               transfer_id,
                               chunk_id,
                               region_token,
                               gpu_addr,
                               gpu_dev,
                               size);
        }
    }

    const uint64_t now_us = nowUs_();
    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        auto &lease = rxLeases_[i];
        if (lease.state == nixlUcxStagedSlotState::ERROR) {
            lease.reset();
            continue;
        }
        if (leaseTimeoutUs == 0 ||
            lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED ||
            lease.grantedUs == 0 || now_us - lease.grantedUs < leaseTimeoutUs) {
            continue;
        }

        lease.state = nixlUcxStagedSlotState::QUARANTINED;
        if (quarantineCallback_) {
            quarantineCallback_(
                i, lease.ownerAgent, lease.transferId, lease.chunkId, lease.leaseId);
        }
    }

    for (size_t i = 0; i < rxLeases_.size(); ++i) {
        if (rxLeases_[i].state == nixlUcxStagedSlotState::FREE) {
            return grantRxSlot(i,
                               owner_agent,
                               transfer_id,
                               chunk_id,
                               region_token,
                               gpu_addr,
                               gpu_dev,
                               size);
        }
    }
    return {.status = NIXL_IN_PROG};
}

nixl_status_t
nixlUcxStagedSlotPool::beginRemoteH2D(const std::string &owner_agent,
                                      uint64_t transfer_id,
                                      uint64_t chunk_id,
                                      uint64_t region_token,
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
    if ((lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED &&
         lease.state != nixlUcxStagedSlotState::QUARANTINED) ||
        lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
        lease.chunkId != chunk_id || lease.leaseId != lease_id ||
        lease.regionToken != region_token || lease.gpuAddr != gpu_addr ||
        lease.gpuDev != gpu_dev || lease.size != size) {
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
    if (status == NIXL_SUCCESS) {
        lease.reset();
    } else {
        lease.state = nixlUcxStagedSlotState::ERROR;
    }
}

bool
nixlUcxStagedSlotPool::releaseRemoteLease(const std::string &owner_agent,
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

    decrementGrantCount(lease.ownerAgent);
    lease.reset();
    return true;
}

size_t
nixlUcxStagedSlotPool::releaseLeasesForOwner(const std::string &owner_agent) {
    const std::lock_guard lock(mutex_);
    size_t released = 0;
    for (auto &lease : rxLeases_) {
        if (lease.ownerAgent != owner_agent ||
            (lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED &&
             lease.state != nixlUcxStagedSlotState::QUARANTINED &&
             lease.state != nixlUcxStagedSlotState::ERROR)) {
            continue;
        }
        if (lease.state != nixlUcxStagedSlotState::ERROR) {
            decrementGrantCount(lease.ownerAgent);
        }
        lease.reset();
        ++released;
    }
    return released;
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
