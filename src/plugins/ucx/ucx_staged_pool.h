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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_STAGED_POOL_H
#define NIXL_SRC_PLUGINS_UCX_UCX_STAGED_POOL_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nixl_types.h>

enum class nixlUcxStagedSlotState {
    FREE,
    LOCAL_D2H,
    REMOTE_RESERVED,
    REMOTE_H2D,
    ERROR,
    QUARANTINED,
};

struct nixlUcxStagedSlotGrant {
    uint64_t slotId = 0;
    uint64_t leaseId = 0;
    nixl_status_t status = NIXL_IN_PROG;
};

struct nixlUcxStagedTxSlot {
    uint64_t slotId = 0;
    nixl_status_t status = NIXL_IN_PROG;
};

struct nixlUcxStagedReadyLease {
    void *hostAddr = nullptr;
    uint64_t slotId = 0;
    uint64_t leaseId = 0;
};

class nixlUcxStagedSlotPool {
public:
    struct Backing {
        void *base = nullptr;
        bool localShared = false;
        std::string sharedPath;
        std::string sharedCookie;
        size_t mappingSize = 0;
        int sharedFd = -1;
        bool hostRegistered = false;
        bool unlinkSharedPath = false;
        void *txRegistration = nullptr;
        void *rxRegistration = nullptr;
    };

    using NowUs = std::function<uint64_t()>;
    using QuarantineCallback =
        std::function<void(size_t, const std::string &, uint64_t, uint64_t, uint64_t)>;

    nixlUcxStagedSlotPool(uint64_t gpu_dev_id,
                          uint64_t pool_epoch,
                          size_t slot_size,
                          size_t slot_stride,
                          size_t tx_count,
                          size_t rx_count,
                          uint64_t lease_timeout_us,
                          size_t max_grants_per_agent,
                          Backing backing,
                          NowUs now_us = {});

    nixlUcxStagedSlotPool(const nixlUcxStagedSlotPool &) = delete;
    nixlUcxStagedSlotPool &
    operator=(const nixlUcxStagedSlotPool &) = delete;

    [[nodiscard]] nixlUcxStagedTxSlot
    acquireTxSlot();

    void
    releaseTxSlot(size_t slot_id);

    [[nodiscard]] uint64_t
    txGeneration(size_t slot_id) const;

    [[nodiscard]] nixlUcxStagedSlotGrant
    reserveRxSlot(const std::string &owner_agent,
                  uint64_t transfer_id,
                  uint64_t chunk_id,
                  uint64_t region_token,
                  uintptr_t gpu_addr,
                  uint64_t gpu_dev,
                  size_t size);

    [[nodiscard]] nixl_status_t
    beginRemoteH2D(const std::string &owner_agent,
                   uint64_t transfer_id,
                   uint64_t chunk_id,
                   uint64_t region_token,
                   uint64_t slot_id,
                   uint64_t lease_id,
                   uintptr_t gpu_addr,
                   uint64_t gpu_dev,
                   size_t size,
                   nixlUcxStagedReadyLease &ready);

    void
    finishRemoteLease(uint64_t slot_id, uint64_t lease_id, nixl_status_t status);

    [[nodiscard]] bool
    releaseRemoteLease(const std::string &owner_agent,
                       uint64_t transfer_id,
                       uint64_t chunk_id,
                       uint64_t slot_id,
                       uint64_t lease_id);

    size_t
    releaseLeasesForOwner(const std::string &owner_agent);

    [[nodiscard]] bool
    hasLeasesForToken(uint64_t region_token) const;

    [[nodiscard]] bool
    hasActiveWork() const;

    [[nodiscard]] void *
    txHostAddr(size_t slot_id) const;

    [[nodiscard]] void *
    rxHostAddr(size_t slot_id) const;

    [[nodiscard]] nixlUcxStagedSlotState
    txState(size_t slot_id) const;

    [[nodiscard]] nixlUcxStagedSlotState
    rxState(size_t slot_id) const;

    [[nodiscard]] uint64_t
    rxRegionToken(size_t slot_id) const;

    void
    setQuarantineCallback(QuarantineCallback callback);

    const uint64_t gpuDevId;
    const uint64_t poolEpoch;
    const size_t slotSize;
    const size_t slotStride;
    const size_t txCount;
    const size_t rxCount;
    const uint64_t leaseTimeoutUs;
    const size_t maxGrantsPerAgent;
    Backing backing;

private:
    struct RxLease {
        nixlUcxStagedSlotState state = nixlUcxStagedSlotState::FREE;
        std::string ownerAgent;
        uint64_t transferId = 0;
        uint64_t chunkId = 0;
        uint64_t leaseId = 0;
        uint64_t regionToken = 0;
        uintptr_t gpuAddr = 0;
        uint64_t gpuDev = 0;
        size_t size = 0;
        uint64_t grantedUs = 0;

        void
        reset();
    };

    [[nodiscard]] nixlUcxStagedSlotGrant
    grantRxSlot(size_t slot_id,
                const std::string &owner_agent,
                uint64_t transfer_id,
                uint64_t chunk_id,
                uint64_t region_token,
                uintptr_t gpu_addr,
                uint64_t gpu_dev,
                size_t size);

    void
    decrementGrantCount(const std::string &owner_agent);

    [[nodiscard]] bool
    quotaBlocks(const std::string &owner_agent) const;

    mutable std::mutex mutex_;
    std::vector<nixlUcxStagedSlotState> txStates_;
    std::vector<uint64_t> txGenerations_;
    std::vector<RxLease> rxLeases_;
    std::unordered_map<std::string, size_t> rxGrantsPerAgent_;
    uint64_t nextLeaseId_ = 1;
    NowUs nowUs_;
    QuarantineCallback quarantineCallback_;
};

#endif
