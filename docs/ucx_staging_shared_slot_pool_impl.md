<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Implementation Plan: Shared Per-GPU Staging Slot Pool

This is the execution contract for implementing
[`ucx_staging_shared_slot_pool.md`](ucx_staging_shared_slot_pool.md) (design v2). Read the design
doc first; this document adds the engineering decisions the design leaves open, fixes exact
names/fields, and prescribes commit slicing and acceptance criteria. Where this document and the
design doc conflict, this document wins. Where something is genuinely ambiguous, do NOT silently
improvise on architecture: implement the closest conservative reading and record the deviation in
the commit message body under a `NOTE:` line so the reviewer can rule on it.

## 0. Ground rules

- Work on branch `ucx-staging-shared-pool` (already created), base `main` @ `c1c430d`.
- Allowed to touch: `src/plugins/ucx/`, `test/gtest/`, `examples/cpp/staged_vram_write_smoke.cpp`,
  `examples/python/staged_vram_write_smoke.py`, `docs/`. Nothing else. Minimal `meson.build`
  edits only where a new test file must be listed.
- No public NIXL API changes. No changes to the non-staged UCX code paths beyond what the
  refactor forces (e.g. `registerMem` dispatch).
- No dual-protocol support: bump `kUcxStagedMagic` from `"NIXL_UCX_STAGED_V1"` to
  `"NIXL_UCX_STAGED_V2"`; a magic mismatch at `internalStagedMDHelper` fails with a clear
  NIXL_ERR_MISMATCH log naming both versions.
- This machine is macOS: the UCX plugin cannot be compiled or run here. Do not attempt
  `meson setup` builds of the plugin; correctness is enforced by careful code-motion, the
  CUDA-free unit tests (which must be structured to compile without CUDA/UCX — see §7), and
  reviewer verification on a Linux/CUDA host afterwards. Prefer mechanical, verifiable edits
  over clever rewrites.
- Match existing file style (4-space indent, existing naming patterns, NIXL_ERROR/NIXL_WARN
  logging idioms). No unrelated reformatting; keep diffs reviewable.
- Commit after each phase in §8 with message prefix `staging:`. Do not push. Do not amend
  earlier phase commits.

## 1. Pool data structures (`ucx_backend.cpp`, file-local like today's classes)

### 1.1 Slot state

Extend the existing staged slot state enum with `QUARANTINED`. Final set:
`FREE, LOCAL_D2H, REMOTE_RESERVED, REMOTE_H2D, ERROR, QUARANTINED`.

### 1.2 `nixlUcxStagedSlotPool`

One instance per (engine, GPU device). Two fixed partitions carved from one slab; **slot IDs on
the wire are RX-relative** (0..rxCount-1) and TX slot IDs are local-only (0..txCount-1). Remote
peers compute RX slot addresses as `rx_base + slot_id * slot_stride`; TX addresses never appear
on the RDMA wire (shm mode shares them by file offset, §6).

```cpp
class nixlUcxStagedSlotPool {
public:
    // testable core: allocation is injected, so lease/quota/quarantine logic
    // compiles and unit-tests without CUDA or UCX (see §7).
    struct Backing {
        void *base = nullptr;          // slab of (txCount + rxCount) * slotStride bytes
        // + ownership info: cudaHostAlloc vs shm-file (path/fd/mapping), filled by init
    };

    uint64_t gpuDevId;
    uint64_t poolEpoch;                // random non-zero 64-bit (reuse cookie RNG helper)
    size_t slotSize;                   // = config chunkSize
    size_t slotStride;                 // roundUp(slotSize, pageSize())
    size_t txCount, rxCount;
    uint64_t leaseTimeoutUs;

    // TX side (source D2H staging)
    //   states used: FREE, LOCAL_D2H
    //   generations kept per TX slot (shm-mode reuse guard), same semantics as
    //   today's localSlotGenerations.
    // RX side (remote-write landing)
    //   full lease records, same fields as today's nixlUcxStagedSlotLease,
    //   PLUS: uint64_t regionToken.
    // per-agent accounting: map<std::string, size_t> rxGrantsPerAgent;
    // single std::mutex covering both partitions (same scope as today's slotMutex).
```

Methods (move the bodies from `nixlUcxStagedPrivateMetadata`, adapting names; keep the logic
diffs minimal and flag anything non-mechanical):

- `acquireTxSlot()` / `releaseTxSlot(id)` / `txGeneration(id)` — from `acquireSlot` /
  `releaseSlot` / `slotGeneration`. If `txCount == 0`, `acquireTxSlot` returns a distinguishable
  "role not configured" failure (see §5 role enforcement), not a plain miss.
- `reserveRxSlot(owner_agent, transfer_id, chunk_id, region_token, gpu_addr, gpu_dev, size)`
  — from `reserveRemoteSlot`, with changes:
  - range validation moves OUT (done by the caller against the RegionRegistry, §3); the pool
    validates only `size <= slotSize` and role (`rxCount > 0`).
  - lease records `regionToken`.
  - **quarantine replaces re-grant**: when no FREE slot exists, first settle bookkeeping:
    `ERROR` leases -> FREE (H2D already consumed the RDMA write; no outstanding writer);
    `REMOTE_RESERVED` older than `leaseTimeoutUs` -> `QUARANTINED` (log one NIXL_WARN with
    owner/transfer/chunk/lease). Then retry the FREE scan. A `QUARANTINED` slot is never
    granted.
  - **per-agent cap**: let `cap = maxGrantsPerAgent` (config §5). Refuse (return IN_PROG) a
    grant to agent X when X currently holds >= cap RX leases (RESERVED+H2D+QUARANTINED
    combined) AND at least one other agent currently holds >= 1 RX lease. With a single active
    agent the cap is not enforced.
- `beginRemoteH2D(...)` — accepts only `REMOTE_RESERVED`; a quarantined lease cannot be revived.
  On success state -> `REMOTE_H2D` as usual.
- `finishRemoteLease(slot_id, lease_id, status)` — unchanged semantics (SUCCESS -> FREE,
  failure -> ERROR).
- `quarantineRemoteLease(...)` — a matching SLOT_RELEASE moves `REMOTE_RESERVED` to
  `QUARANTINED`; an already quarantined matching lease is an idempotent success.
- `quarantineLeasesForOwner(owner)` — disconnect moves that owner's `REMOTE_RESERVED` leases to
  `QUARANTINED`. It never touches `REMOTE_H2D` or another owner.
- `hasLeasesForToken(region_token)` — any RX lease (any non-FREE state) recording that token;
  used by `deregisterMem`.
- `hasActiveWork()` — any TX slot non-FREE or RX lease non-FREE; used at engine teardown for a
  diagnostic log (teardown order already guarantees quiescence).

### 1.3 Slab initialization (separate from the core, CUDA/UCX code lives here)

`initSlab(engine context)` performs, in order:

1. Allocate: `localStaging ? (pool-level shm file, mmap, cudaHostRegister)` —
   file name `nixl-ucx-pool-<sanitized agent>-<pid>-<gpuDev>-<epoch>.bin` in
   `localStagingShmDir`, size `(txCount + rxCount) * slotStride`, same create/size/mmap/register
   sequence and cleanup-on-error ordering as today's per-region branch —
   `: cudaHostAlloc(&base, size, cudaHostAllocPortable)`.
2. UCX-register the TX range `[base, txCount * slotStride)` into `txMem` (local memh only; no
   rkey is packed for it).
3. UCX-register the RX range `[base + txCount * slotStride, rxCount * slotStride)` into `rxMem`
   and pack ONE rkey string `rxRkeyStr`.
4. On any failure, unwind in reverse order (mirror today's `cleanup_staged` discipline).

Skip step 2 when `txCount == 0` and steps 3 when `rxCount == 0`.

Destruction (engine destructor only, after progress thread / AM / H2D worker stop): dereg
`txMem`/`rxMem`, then `cudaFreeHost` or (cudaHostUnregister, munmap, close, unlink).

## 2. Engine members (`ucx_backend.h`)

Replace the staging members:

```cpp
class nixlUcxStagedSlotPool;                       // fwd decl (defined in .cpp)— if a fwd-decl
                                                   // + unique_ptr member is not workable from the
                                                   // header, fall back to void* + accessors; NOTE it.
mutable std::mutex stagedPoolMutex_;
std::map<uint64_t, std::unique_ptr<nixlUcxStagedSlotPool>> stagedPools_;   // by gpuDevId
mutable std::atomic<uint64_t> nextRegionToken_{1};
// stagedRegionMutex_ stays; stagedRegions_ (vector) is replaced by:
std::unordered_map<uint64_t, nixlBackendMD *> stagedRegionsByToken_;
// RemotePool cache (initiator side), see §4:
mutable std::mutex remotePoolMutex_;
std::map<std::tuple<std::string, uint64_t, uint64_t>,
         std::shared_ptr<nixlUcxStagedRemotePool>> remotePools_;  // (agent, gpuDev, epoch)
```

Lock order (document at the declaration, it extends today's rule):
`stagedRegionMutex_` -> pool mutex; `stagedPoolMutex_` only guards map insert/lookup and is
never held while calling into a pool; `remotePoolMutex_` is independent and never held across
UCX send/progress calls.

`VramStagingConfig`: replace `slotsPerGpu` with `size_t txSlots = 4; size_t rxSlots = 4;
size_t maxGrantsPerAgent = 0;` (0 = auto `max(1, rxSlots / 2)`), keep everything else.

## 3. Region registry and `registerMem` / `deregisterMem`

`nixlUcxStagedPrivateMetadata` shrinks to: `regionToken, gpuBase, gpuLen, gpuDevId, hostId`
plus a pointer to its device pool. Everything slot/lease/shm related moves out.

`registerMem` (`ucx_backend.cpp:2586`) staged branch becomes:

1. Validate as today (`mem.len`, chunk size, and now `txSlots + rxSlots > 0`).
2. Under `stagedPoolMutex_`: find-or-create `stagedPools_[mem.devId]`; creation runs
   `initSlab` (outside the map lock is fine after insertion of a placeholder, but the simple
   version — init under the lock — is acceptable; NOTE which you chose).
3. `token = nextRegionToken_++`; build metadata; under `stagedRegionMutex_` insert into
   `stagedRegionsByToken_`.
4. Log one NIXL_INFO (region token, range, dev) — per-region log no longer mentions slots.

`deregisterMem` (`:2761`):

- Under `stagedRegionMutex_`: if `pool->hasLeasesForToken(token)` -> NIXL_ERR_NOT_ALLOWED (same
  single-critical-section discipline as today, same log shape); else erase from
  `stagedRegionsByToken_`. Free nothing else — the pool is untouched.

TX slots are not region-scoped, so region dereg does not check TX state (TX slots belong to
in-flight *local* transfers, whose request handles pin what they need).

## 4. Metadata v2, public MD, RemotePool cache

### 4.1 `serializeStagedMetadata` (`:790`)

Serialize exactly (order as listed; all `addBuf` for integers, `addStr` for strings):

```text
"magic"            = "NIXL_UCX_STAGED_V2"
"region_token"     u64
"gpu_base" "gpu_len" "gpu_dev" "host_id"          (as today)
"pool_epoch"       u64
"slot_size"        u64      // pool slotSize
"slot_stride"      u64
"rx_count"         u64
"rx_base"          u64      // uintptr of RX range start; 0 when rxCount == 0
"rx_rkey"          str      // packed rkey; empty when rxCount == 0
"ls_enabled"       u8/bool  // localStaging
"ls_path" "ls_cookie" "ls_mapping_size" "ls_tx_count"   // shm descriptor, empty/0 when disabled
```

Per-slot `slot_addr`/`slot_rkey` entries are gone.

### 4.2 `internalStagedMDHelper` and `nixlUcxStagedPublicMetadata`

Public MD v2 keeps `{conn, agent, regionToken, gpuBase, gpuLen, gpuDevId, hostId,
slotSize(remote)}` plus `std::shared_ptr<nixlUcxStagedRemotePool>`.

`nixlUcxStagedRemotePool` (new, initiator-side view of one remote pool):

```cpp
struct nixlUcxStagedRemotePool {
    std::string agent; uint64_t gpuDev; uint64_t poolEpoch;
    uintptr_t rxBase; size_t slotStride; size_t slotSize; size_t rxCount;
    std::vector<nixl::ucx::rkey> rxRkeys;      // one per worker, unpacked once from rx_rkey
    // request window, moved from per-region public MD:
    size_t windowLimit;                        // config slotRequestWindow, or rxCount*4 when 0
    std::atomic<size_t> windowInUse{0};        // + the existing tryAcquire/release logic
    // shm descriptor for same-host validation (path, cookie, mappingSize, txCount)
};
```

`internalStagedMDHelper` dedupes: under `remotePoolMutex_`, look up
`(agent, gpu_dev, pool_epoch)`; on miss build the entry (unpack `rx_rkey` per worker exactly the
way `makePublicMetadataRkeys` does today) and insert. Public MDs share the entry via
`shared_ptr`; the map entry is erased on disconnect of that agent (and stale epochs of the same
agent+dev are erased when a newer epoch arrives).

The same-host (`hostId` match) LOCAL path validation data now comes from this entry —
`registerLocalSharedRegion` / `unregisterLocalSharedRegion` / `LocalSharedRegionInfo`
(region-scoped) are deleted in favor of it; `validateLocalSharedReady` checks against the
RemotePool entry (path, cookie, mapping size, tx slot geometry, generation, §6).

### 4.3 Window semantics

`tryAcquireSlotWindow` / `releaseSlotWindow` move onto `nixlUcxStagedRemotePool` unchanged in
logic. `send_slot_request` (`:4307`) acquires the window from
`chunk.remoteMetadata->remotePool`. This intentionally changes scope from per-region to
per-(agent, device); the design doc §4 records why and mandates re-benchmarking — do not
"compensate" in code.

## 5. Config, role enforcement, control messages

### 5.1 Params/envs (`ucx_utils.h/.cpp`, `makeVramStagingConfig` `:1721`)

New (names follow existing `nixl_ucx_*_param_name` pattern):

| param | env | default |
|---|---|---|
| `staging_tx_slots_per_gpu` | `NIXL_UCX_STAGING_TX_SLOTS` | 4 |
| `staging_rx_slots_per_gpu` | `NIXL_UCX_STAGING_RX_SLOTS` | 4 |
| `staging_max_grants_per_agent` | `NIXL_UCX_STAGING_MAX_GRANTS_PER_AGENT` | 0 = `max(1, rx/2)` |

Legacy `staging_slots_per_gpu` / `NIXL_UCX_STAGING_SLOTS`: when the specific tx/rx knob is
absent, its value seeds BOTH tx and rx (param layer first, then env layer, matching the existing
two-stage override order in `makeVramStagingConfig`). The startup NIXL_INFO log (`:1850`) prints
`tx_slots`/`rx_slots`/`max_grants_per_agent` instead of `slots_per_gpu`.

### 5.2 Role enforcement (error, not hang)

- `registerMem` staged branch: `txSlots + rxSlots == 0` -> NIXL_ERR_INVALID_PARAM (existing
  invalid-config log extended).
- Target with `rxCount == 0` receiving SLOT_REQ: grant status `NIXL_ERR_NOT_SUPPORTED`
  (terminal — the initiator surfaces it as transfer failure; it must NOT be IN_PROG/retry).
  One NIXL_ERROR log naming the role config.
- Initiator with `txCount == 0` posting a staged write: `postStagedWrite` fails fast with
  `NIXL_ERR_NOT_SUPPORTED` and a clear log, before any SLOT_REQ is sent.

### 5.3 Message field changes (serdes keys)

- `sendStagedSlotReq` / `handleStagedSlotReq`: add `"region_token"` u64. Handler: O(1) lookup in
  `stagedRegionsByToken_` under `stagedRegionMutex_`; verify `gpu_dev` matches and
  `rangeCovers(region)` as today; verify the region's device pool exists; then
  `reserveRxSlot(...)`. Unknown/stale token -> grant status NIXL_ERR_NOT_FOUND (terminal).
- `sendStagedSlotGrant` / grant AM: add `"pool_epoch"` u64. Initiator on grant receipt: if
  `pool_epoch != remotePool->poolEpoch`, treat as terminal chunk error (metadata is stale) — log
  and fail the transfer; do NOT retry.
- `SLOT_RELEASE`: fields unchanged (slot_id now RX-relative pool id). Handler: `gpu_dev` routes
  to the pool (`by_id` releases scan all pools — there are at most a handful).
- `WRITE_READY` / `ACK`: fields unchanged; handler routes by `gpu_dev` to the pool.
- `LOCAL_WRITE_READY` (`sendStagedLocalWriteReady` / `handleStagedLocalWriteReady`): replace
  region-scoped fields (`source_region_id`, `source_region_cookie`, `source_shared_path`,
  `source_mapping_size`) with `"pool_epoch"`, `"ls_cookie"`, `"ls_path"`, `"ls_mapping_size"`;
  keep `tx slot id, generation, offset, size` fields (renames allowed for clarity). Validation
  (§4.2) recomputes expected offset `slot_id * slot_stride` and bounds-checks against
  `ls_mapping_size` and the TX range exactly as `validateLocalSharedReady` does today.

## 6. Data-path changes

### 6.1 Initiator FSM (`postStagedWrite` + helpers around `:4300-4700`)

- `chunk.localMetadata->acquireSlot()` (two call sites: prefetch `:4384`, non-prefetch `:4528`)
  -> `localPool->acquireTxSlot()` where `localPool` is resolved once per transfer from
  `chunk.localGpuDev`. TX slot host addr = `slab_base + slot_id * slot_stride`; local memh for
  the RDMA write source is the pool's `txMem`.
- RDMA write target (`:4595-4602`): address = `remotePool->rxBase + slot_id * slot_stride`;
  rkey = `remotePool->rxRkeys[worker_id]`. The `slotAddrs`/`slotRkeys` bounds check (`:4573`)
  becomes `slot_id < rxCount && worker_id < rxRkeys.size()`.
- Chunk sizing (`:4160`): `min(localPool->slotSize, remotePool->slotSize)`.
- Same-host branch chooses LOCAL path as today (hostId match + `ls_enabled`), consuming TX slots
  and sending pool-scoped LOCAL_WRITE_READY.
- Stale-grant guard, release paths, batch flush, profile counters: unchanged except that
  releases go through the pool. Rename profile fields only where the old name is now wrong
  (e.g. `localSlotMiss` stays fine).

### 6.2 Target H2D (`StagedH2DTask`, worker, `handleStagedWriteReady`)

`StagedH2DTask.region` (raw `nixlBackendMD*`) is replaced by
`{nixlUcxStagedSlotPool *pool, uint64_t slotId, uint64_t leaseId, uint64_t regionToken}` plus
the existing addr/dev/size fields. The H2D copy uses the lease-pinned slot host address and the
task's `gpuAddr`; completion calls `pool->finishRemoteLease`. No region pointer is ever
dereferenced after grant time. Pools outlive the H2D worker (teardown order in §1.3).

### 6.3 Disconnect path

Call `pool->quarantineLeasesForOwner(agent)` for every pool, and under `remotePoolMutex_` erase that
agent's `remotePools_` entries.

## 7. Tests (phase-1 priority, not an afterthought)

### 7.1 CUDA-free unit tests (new file `test/gtest/unit/plugins/ucx_staged_pool.cpp` or the
closest fitting location under `test/gtest/unit/`; add to that dir's `meson.build`)

Structural requirement on the pool class making this possible: the lease/quota/quarantine core
must not reference CUDA/UCX types except through the injected `Backing` (tests inject a
heap-allocated fake slab and never touch `initSlab`). If the current file layout makes the pool
class unreachable from tests (it is file-local in `ucx_backend.cpp`), extract the pool class
into `src/plugins/ucx/ucx_staged_pool.{h,cpp}` (CUDA/UCX-free core; the slab init with
CUDA/UCX calls stays in `ucx_backend.cpp` or an `#ifdef`-free separate TU) — this extraction is
the preferred option; NOTE the layout you chose.

Cases (assert exact statuses, not just "doesn't crash"):

1. TX/RX isolation: exhaust TX -> `reserveRxSlot` still grants; exhaust RX -> `acquireTxSlot`
   still succeeds.
2. Quarantine: fill RX, expire leases (inject clock or make `leaseTimeoutUs` tiny and the
   time-source injectable — prefer an injectable `now_us` functor, NOTE if you deviate),
   `reserveRxSlot` -> the expired leases become QUARANTINED and the call still returns IN_PROG
   (never a grant of a quarantined slot).
3. Late settle: quarantined lease + matching `beginRemoteH2D` is rejected and remains pinned.
4. Late release: SLOT_RELEASE cannot free a quarantined lease.
5. Owner disconnect: reserved leases for that owner become quarantined; never touch
   REMOTE_H2D or another owner's leases.
6. ERROR reclaim: ERROR lease -> next `reserveRxSlot` under pressure frees and re-grants it.
7. Per-agent cap: rx=4, cap auto=2; agent A takes 2, agent B takes 1; A's third request ->
   IN_PROG while B holds; after B settles and A is sole active agent, A can exceed cap.
8. Role: `rxCount == 0` pool refuses `reserveRxSlot` with the role error; `txCount == 0` refuses
   `acquireTxSlot` likewise.
9. `hasLeasesForToken`: reflects RESERVED/H2D/QUARANTINED leases; clears after settle.
10. Region-token bookkeeping through reserve -> begin -> finish keeps token stable and visible.

### 7.2 Smoke (CUDA hosts only; keep them compiling, do not run here)

Extend both `staged_vram_write_smoke.cpp/.py`: an optional `--regions N` (default 1) mode that
registers N small VRAM regions instead of one, transfers across all of them, and prints the
process's pinned/shmem usage so the operator can verify it stays at `(tx+rx) * chunk_size`
regardless of N. Keep existing modes working.

### 7.3 Existing tests

`test/gtest/error_handling.cpp` and others referencing staging params must keep compiling; update
param names only where they used `staging_slots_per_gpu` semantics in a way that breaks.

## 8. Commit plan (in order)

1. `staging: extract slot pool core with tx/rx partition and quarantine` — new pool class
   (+ file extraction per §7.1), enum value, config fields/parsing (§5.1), unit tests. No
   behavior change wired in yet; old path still compiles untouched where possible.
2. `staging: move target side to shared per-GPU pools` — registerMem/deregisterMem/region
   registry/tokens, metadata v2 serialize, SLOT_REQ/GRANT/RELEASE/READY handlers on pools,
   H2D task without region pointer, role enforcement.
3. `staging: move initiator side to remote pool descriptors` — public MD v2 +
   `nixlUcxStagedRemotePool` dedupe cache, FSM changes, computed RX addresses, window scope,
   epoch check.
4. `staging: pool-scoped local shared staging` — shm slab mode, LOCAL_WRITE_READY v2,
   validation against remote pool entry, attachment cache keyed per pool.
5. `staging: teardown, disconnect, and logging for shared pools` — disconnect path, engine
   destructor, startup/summary logs, profile counter touch-ups.
6. `staging: update smoke tests and docs for shared pools` — §7.2, plus updating
   `ucx_vram_staging.md` / `local_vram_staging.md` / `sglang_pd_staging_integration.md`
   param tables and the lowmem-script note (design doc §8).

If a phase turns out to be uncompilable in isolation because of coupling, merge it into the
adjacent phase rather than leaving a broken commit; NOTE it in the commit message.

## 9. Acceptance checklist (reviewer will verify; self-check before finishing)

- [ ] `grep -n "cudaMallocHost\|cudaHostRegister\|cudaHostAlloc" src/plugins/ucx/` hits only
      pool slab init (+ the unrelated pre-existing attachment path), never per-region code.
- [ ] Metadata v2 contains no per-slot entries; `slot_addr`/`slot_rkey` keys are gone from the
      codebase.
- [ ] Registering N regions on one device creates exactly one pool; deregistering all N frees
      nothing pinned.
- [ ] No code path grants a QUARANTINED slot; the only QUARANTINED exits are late-settle, late
      release, owner disconnect, engine teardown.
- [ ] TX and RX never borrow from each other (no code path allocates across the partition).
- [ ] Role misconfiguration produces NIXL_ERR_NOT_SUPPORTED, never a hang or IN_PROG loop.
- [ ] Region token is validated on every SLOT_REQ; dereg+re-register at the same GPU address
      cannot be written through a stale token.
- [ ] Lock order documented and consistent: stagedRegionMutex_ -> pool mutex; remotePoolMutex_
      never held across UCX calls.
- [ ] All unit tests of §7.1 present and passing (they must run on macOS: no CUDA/UCX includes
      in the test TU or the pool-core header).
- [ ] Non-staged UCX paths and files outside the §0 allowlist untouched
      (`git diff main --stat` audit).
- [ ] Every architectural deviation from this plan is flagged with `NOTE:` in a commit message.
