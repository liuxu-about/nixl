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

# UCX Staging: Shared Per-GPU Slot Pool (Design, v2)

Status: proposal, not implemented. Revision 2 — reworked after review: TX/RX pool partition,
contiguous slab with a single published rkey, pool-scoped request window flagged as a semantic
change requiring re-benchmarking, quarantine instead of re-grant on lease timeout, region tokens,
and evidence-based default sizing.

This document proposes moving the staged-VRAM slot pool from per-registered-region to a single
engine-owned pool per GPU device. It is a follow-up to
[`ucx_vram_staging.md`](ucx_vram_staging.md) and [`local_vram_staging.md`](local_vram_staging.md),
motivated by an OOM observed in the SGLang PD integration.

## Problem

Pinned staging memory currently scales as:

```text
pinned bytes per process = num_registered_VRAM_regions x staging_slots_per_gpu x chunk_size
```

The pool lives inside `nixlUcxStagedPrivateMetadata`, created per region in
`nixlUcxEngine::registerMem` (`ucx_backend.cpp`). NIXL core calls `registerMem` once per
descriptor (`nixl_memory_section.cpp`), and the parameter named `staging_slots_per_gpu` is in
fact multiplied by the region count. SGLang's NIXL connector registers the KV pool as one region
per layer per K/V buffer:

```text
48 layers x 2 (K,V)      =  96 regions per TP process
96 x 16 slots x 16 MiB   ~= 24.6 GiB pinned per process
x 4 processes (2 lanes)  ~= 98 GiB pinned on one host, allocated at startup, before any traffic
```

Dropping `staging_slots_per_gpu` to 2 was deployed as a stopgap (98 GiB -> ~13 GiB), but the
product still scales with layer count.

Note on what the stopgap does and does not prove: the slot-request window
(`staging_slot_request_window`) is enforced per remote *region* object
(`nixlUcxStagedPublicMetadata::tryAcquireSlotWindow`), so a deployment with 96 regions and
window 2 has up to 96 independent 2-slot windows, not 2 slots in flight total. The stopgap run
therefore shows that 2 slots *per region* sustain the SGLang workload; it does not establish how
many slots one shared per-GPU pool needs. Shared-pool sizing must be measured (section 8).

Secondary costs of the per-region design:

- Region metadata carries every slot's address and packed rkey (`serializeStagedMetadata`), so
  metadata exchange payload also scales with `regions x slots`.
- In local (same-host shm) mode, the target attaches and `cudaHostRegister`s one mapping per
  source region per peer (`getLocalSharedAttachment`): 96 attachments per peer instead of 1.
- Region registration performs `slots` pinned allocations, UCX registrations, and rkey packs per
  region.

## Where the pool is coupled to the region today

Any redesign has to unhook four couplings:

1. **Rkey pre-share.** Initiators learn slot host addresses and rkeys from region metadata at
   `loadRemoteMD` time (`nixlUcxStagedPublicMetadata.slotAddrs/slotRkeys`) and use them directly
   when posting the RDMA write (`start_granted_chunk`). Slots must exist and be registered before
   region metadata is serialized.
2. **Grant routing.** `handleStagedSlotReq` scans `stagedRegions_` for the region covering the
   requested GPU range and reserves from that region's pool. Slot IDs in SLOT_GRANT /
   SLOT_RELEASE / WRITE_READY are region-scoped.
3. **Local-shared validation.** In shm mode the target only attaches paths it learned through the
   metadata channel (`registerLocalSharedRegion` records path/cookie/geometry;
   `validateLocalSharedReady` cross-checks every LOCAL_WRITE_READY against it). The AM message
   alone must never be enough to make the target mmap an arbitrary path; this property must be
   preserved at pool scope.
4. **Lifecycle.** `deregisterMem` refuses to drop a region while its pool has active leases, and
   slot memory dies with the region. In-flight H2D tasks carry a raw `region` pointer
   (`StagedH2DTask`).

## Goals

- Pinned staging memory becomes `num_gpu_devices x (tx_slots + rx_slots) x chunk_size` per
  process — independent of how many regions the application registers.
- Slot capacity is expressed per GPU and per direction, with explicit role configuration for PD
  deployments (prefill = TX only, decode = RX only).
- No change to the application-facing NIXL API or to SGLang integration code.
- Keep the lease state machine, stale-grant guard, and retry (IN_PROG polling) model.
- Lease timeout must not become a cross-region corruption vector (quarantine, section 6).
- No throughput regression versus the slots=2 stopgap on the SGLang PD benchmark, validated by a
  sizing matrix before defaults are finalized.

Non-goals:

- Cross-process slot sharing on one host (see Alternatives).
- Wire compatibility between old and new plugin builds (see Rollout).
- Unconstrained borrowing between TX and RX capacity (deliberately excluded, section 5).

## Proposed design

### 1. Structure overview

```text
nixlUcxEngine
├── DeviceStagingPool[gpuDevId]            (created lazily at first VRAM registerMem)
│   ├── one contiguous pinned slab; slot_stride = roundUp(chunk_size, page)
│   ├── TX slot range   (source D2H staging; local UCX memh only, no published rkey)
│   ├── RX slot range   (remote-write landing; own UCX memh + ONE packed rkey, published)
│   ├── poolEpoch       (random cookie; identifies this slab/registration generation)
│   ├── lease table     (FREE / LOCAL_D2H / REMOTE_RESERVED / REMOTE_H2D / ERROR / QUARANTINED)
│   └── per-agent grant accounting
├── RegionRegistry: map<regionToken, {gpuBase, gpuLen, gpuDevId, activeLeaseCount}>
└── RemotePool cache: map<(agent, gpuDev, poolEpoch), {slabBase, stride, rx range,
        unpacked per-worker rkeys, pool-global request window, shm descriptor}>
```

### 2. The slab: one allocation, one published rkey

Per device pool, allocate one contiguous pinned slab of `(tx_slots + rx_slots) x slot_stride`
bytes (`cudaHostAlloc` with `cudaHostAllocPortable`; in local-shm mode, the existing per-pool
file + `mmap` + `cudaHostRegister`, which already has exactly this one-file/strided-slots
layout). Then, one time only:

- register the TX range with UCX for local use (RDMA-write source needs a local memh, never a
  remote rkey);
- register the RX range with UCX and pack **one** rkey for the whole range.

Registering TX and RX separately keeps the published rkey scoped to memory remote peers are
actually supposed to write — slightly stronger than today, where every slot's rkey (including
slots currently used for local D2H) is broadcast.

Remote slot addressing becomes computed, not tabulated: `slab_base + slot_id x slot_stride`.
This removes per-slot `cudaMallocHost`/`memReg`/`packRkey`, removes the per-slot rkey table from
metadata, keeps SLOT_GRANT small, and means the initiator unpacks exactly one rkey per
`(agent, gpuDev, poolEpoch)` per worker, cached in the RemotePool entry (dropped on disconnect
or epoch change).

`slot_stride` is fixed at `roundUp(chunk_size, page)`; a region smaller than a slot uses part of
one. Cross-peer chunk sizing stays `min(local_slot_size, remote_slot_size)`, with the remote's
slot size delivered via the pool descriptor.

### 3. Metadata v2 and region tokens

`registerMem(VRAM_SEG)` now only: validates, ensures the device pool exists, assigns a fresh
`regionToken` (monotonic u64, never reused), and records the region in RegionRegistry. No pinned
allocation, no UCX registration — region registration becomes O(1).

`serializeStagedMetadata` v2 (bump `kUcxStagedMagic`):

```text
magic (v2), region_token, gpu_base, gpu_len, gpu_dev, host_id
pool descriptor for gpu_dev:
  pool_epoch, slot_stride, slot_size, rx_slot_range, one packed rx rkey
  local_shared: enabled, path, cookie, mapping_size, tx_slot_range   // shm mode only
```

The pool descriptor is identical across all regions of one device; the remote side dedupes it
into the RemotePool entry keyed by `(agent, gpuDev, poolEpoch)`, refcounted by the loaded public
MDs and dropped on disconnect. The shm path still only ever arrives via the metadata channel
(coupling 3 preserved); LOCAL_WRITE_READY messages are validated against the deduped descriptor.

`regionToken` closes an ABA hazard the current address-scan has: if a region is deregistered and
a new one registered at the same GPU address, a peer still holding the old metadata could
otherwise direct writes into the new region. With tokens, SLOT_REQ carries
`region_token + gpu_addr + size`; the target looks the token up in RegionRegistry (O(1) instead
of scanning ~96 regions per request), verifies the range and device, then reserves from the
pool. A token that is gone or mismatched fails the grant with a clear error. Leases record the
token; `deregisterMem` refuses while `activeLeaseCount > 0` for that token.

### 4. Control plane

Message shapes are unchanged except where noted:

- **SLOT_REQ** — gains `region_token` (section 3). Target reserves an RX slot from
  `DeviceStagingPool[gpu_dev]`.
- **SLOT_GRANT** — gains `pool_epoch` only: `{xfer_id, chunk_id, status, slot_id, lease_id,
  pool_epoch}`. The initiator computes the target address from the RemotePool descriptor; an
  epoch mismatch (pool rebuilt since metadata load) is treated as a failed grant and surfaces as
  a transfer error prompting metadata reload.
- **SLOT_RELEASE** — gains `release_kind` (`SAFE_CANCEL` or `QUARANTINE`; missing/unknown values
  fail closed to `QUARANTINE`); lease lookup moves to the pool (release-by-id scans the handful
  of device pools instead of 96 regions).
- **WRITE_READY / ACK** — unchanged; lease lookup moves to the pool.
- **LOCAL_WRITE_READY** — region-scoped shm fields become pool-scoped (epoch, pool cookie/path,
  slot_id, generation, offset). The target's attachment cache holds one mapping per peer pool
  instead of one per peer region.

**Request window.** `tryAcquireSlotWindow` moves from the per-region public MD to the RemotePool
entry: it now bounds outstanding SLOT_REQs per `(agent, gpuDev)` rather than per region. This is
a real concurrency-semantics change, not a refactor: today a 96-region deployment with window 2
has up to 192 independent in-flight requests; afterwards one shared window governs the device.
Consequences:

- The window stays an independent tunable. Prior single-region tuning
  (`ucx_vram_staging.md`, tuning tables) shows a window *larger* than the slot count is
  fastest (slots=4: window=4 -> 15.1 GiB/s, window=32 -> 18.5 GiB/s); a window equal to the
  slot count removes control-plane retries but underfills the pipeline. So the default is NOT
  `rx_slots`; keep the existing auto rule scaled to the pool (`rx_slots x 4` when unset, window
  well above slot count remains allowed and expected).
- The sizing matrix (section 8) must re-measure window values against the shared pool before
  defaults are frozen.

Contention stays retry-based (target answers IN_PROG, initiator repolls from `checkStagedXfer`);
no target-side wait queue.

### 5. TX/RX partition: why no borrowing

One shared pool serving both directions deadlocks, and a grant-side reserve alone does not fix
it. Two concrete cycles with bidirectional traffic between A and B:

```text
(a) Grant exhaustion: all of A's slots granted to B's transfers and vice versa;
    neither side can acquire a local D2H slot, so neither ever reaches READY,
    so no grant is ever returned.
(b) Prefetch exhaustion (source_d2h_prefetch=true): A fills all its own slots
    with LOCAL_D2H/LOCAL_READY chunks before requesting remote slots; B does the
    same. Both then wait forever for remote grants. LOCAL_* states have no
    timeout reclaim, so unlike (a) this never unwinds — it is a permanent hang.
```

A single "reserve N slots from remote grants" rule blocks (a) but not (b): the local side can
starve the pool exactly as thoroughly as remote grants can. The fix is a fixed partition:

- **TX slots** serve `acquireSlot` (source D2H staging): `FREE -> LOCAL_D2H -> [RDMA/flush] ->
  FREE`.
- **RX slots** serve `reserveRemoteSlot` (incoming grants): `FREE -> REMOTE_RESERVED ->
  REMOTE_H2D -> FREE` (or `ERROR`/`QUARANTINED`).
- No borrowing in either direction. Each direction's chain completes using only its own local
  resources plus the peer's opposite-direction resources, so no cycle exists.

Role configuration makes the partition explicit in PD deployments: prefill runs `tx=N, rx=0`,
decode runs `tx=0, rx=N`, generic bidirectional runs both non-zero. A SLOT_REQ arriving at a
pool with `rx=0`, or a local acquire on `tx=0`, fails loudly with a clear "role not configured"
error — a misconfiguration should be an error, not a latent deadlock.

Per-agent grant accounting on RX (`staging_max_grants_per_agent`, default `rx_slots/2` when more
than one agent is actively requesting, floor 1) prevents one initiator from starving others and
— together with quarantine — bounds how much RX capacity a single frozen peer can pin down.

### 6. Lease timeout: quarantine, not re-grant

Today, when no slot is free, the target reclaims `REMOTE_RESERVED` leases older than
`staging_lease_timeout_ms` and re-grants them. The staging design doc itself documents the
residual risk: an initiator frozen after posting its RDMA write can wake up and land a late
write in a slot that has since been re-granted. In a per-region pool the blast radius is that
region; in a shared pool it becomes any region on the device. That trade is not acceptable, so
timed-out leases stop being re-granted:

- On timeout, a `REMOTE_RESERVED` lease moves to `QUARANTINED`: not grantable, but the lease
  record stays intact.
- A quarantined lease never returns to the live pool. Late WRITE_READY and SLOT_RELEASE messages
  cannot revive or free it, and owner disconnect quarantines remaining reserved leases. Process
  restart or a future pool rebuild is the recovery boundary; this deliberately trades capacity
  for protection against a late write reaching a newly assigned request.
- `ERROR` leases (H2D failed) are different: READY was already received, so the RDMA write has
  completed and no remote writer is outstanding. They return to FREE once the error ACK settles,
  as today.
- The initiator-side stale-grant guard (refuse grants older than half the lease timeout) stays
  as cheap defense in depth.

A frozen-but-connected peer can now pin RX slots in quarantine indefinitely. The per-agent grant
cap (section 5) limits simultaneous active grants while peers contend, but quarantined leases are
removed from active grant accounting to avoid ghost quota; repeated failures can therefore
accumulate quarantined slots until an epoch rebuild. This is strictly safer than the current
silent-corruption window and makes pool-health monitoring important.

### 7. Region and pool lifecycle

- **`deregisterMem`**: refuse (`NIXL_ERR_NOT_ALLOWED`, as today) while the region's
  `activeLeaseCount > 0` (leases — including quarantined — record their region token);
  otherwise drop the RegionRegistry entry. Pinned memory is untouched: region churn no longer
  touches pinned memory at all.
- **Pool teardown**: engine destructor only, after the progress thread, AM handlers, and H2D
  worker are stopped and connections are down.
- **`StagedH2DTask`** drops its `region` pointer in favor of `(gpuAddr, gpuDev, pool, slotId,
  leaseId, regionToken)`. The H2D copy needs only the slot host address and GPU address, both
  pinned by the lease; the region-pointer dangling hazard disappears.

### 8. Configuration and sizing

| Param | Meaning | Default |
|---|---|---|
| `staging_tx_slots_per_gpu` (new) | source-side D2H staging slots per GPU | 4 |
| `staging_rx_slots_per_gpu` (new) | remote-write landing slots per GPU | 4 |
| `staging_slots_per_gpu` / `NIXL_UCX_STAGING_SLOTS` | legacy shorthand: sets both tx and rx | 4 |
| `staging_max_grants_per_agent` (new) | RX grant cap per initiator when contended | `rx/2`, floor 1 |
| `staging_slot_request_window` | outstanding SLOT_REQs per (agent, gpuDev) — scope changed | auto = `rx x 4` |
| others (`chunk_size`, lease timeout, shm dir, …) | unchanged | unchanged |

The 4/4 baseline is chosen to preserve, for a single active region, exactly the source- and
target-side concurrency the current default (`slotsPerGpu = 4`) provides — not because it is
known optimal. Memory per process at 16 MiB chunks:

```text
generic bidirectional (tx=4, rx=4):  128 MiB   (~512 MiB per 4-process host)
PD role-split (tx=4 or rx=4 only):    64 MiB   (~256 MiB per 4-process host)
vs. original 96-region deployment:  24.6 GiB   (98 GiB per host)
vs. slots=2 stopgap:                 ~3 GiB    (~13 GiB per host)
```

Because the shared window is a semantic change (section 4), defaults are frozen only after a
sizing matrix on the SGLang PD benchmark: `{tx, rx} in {2, 4, 8} x window in {4, 8, 16, 32}`,
success criteria = zero-prefix 200-request throughput >= the slots=2 stopgap run
(15.23 req/s / 3900 tok/s) and pinned shmem at the predicted value. Single-region microbench
history (18.5 GiB/s at slots=4/window=32) suggests a small pool saturates the link, but it was
measured under per-region windows and does not transfer as-is.

> Deployment note: the low-memory scripts that set `NIXL_UCX_STAGING_SLOTS=2` must be revisited
> when this lands — under pool semantics that means 2 slots per direction for the whole GPU.
> Recommended: drop the override and take role-split defaults.

## Rollout and compatibility

Bump the staged-metadata magic (v2). An old plugin loading v2 metadata (or vice versa) fails
loudly at `loadRemoteMD` with a version-mismatch error — never silent corruption. Both ends of a
PD deployment ship in the same container image, so a flag-day upgrade is acceptable;
dual-protocol support is deliberately out of scope.

Parallel track, independent of this design: re-evaluate a CUDA-enabled UCX build. On the H20
side (data-center GPU, GPUDirect RDMA capable) it could remove staging entirely and should be
spiked. On the RTX 4090 side GPUDirect RDMA is not available on GeForce parts; CUDA-enabled UCX
there means UCX's internal pipelined host staging, which may or may not meet the completion
semantics and performance this plugin was written to guarantee (see `ucx_vram_staging.md`
goals). The staged path therefore remains the 4090 lane's plan of record, and this redesign
stands on its own either way.

## Alternatives considered

1. **Carry a packed rkey in every SLOT_GRANT** (the v1 draft of this document). Works, but
   requires per-slot registration and an initiator-side rkey cache keyed per slot, and makes
   grants bigger. The contiguous slab with one published RX rkey achieves the same decoupling
   with one registration, computed addresses, and an unchanged-size grant message. Strictly
   simpler; superseded.
2. **Shared pool, per-slot table still pre-shared in metadata.** Keeps `regions x slots`
   metadata bloat and freezes pool geometry at first serialize; the slab descriptor is smaller
   and cleaner. Rejected.
3. **Keep per-region pools, allocate slots lazily on first grant.** Lazy slots cannot have been
   pre-shared, so this needs grant-time key delivery anyway, while worst-case memory still
   scales with regions. Strictly dominated.
4. **Application-side: register the KV pool as one region.** Only works when the KV pool is one
   contiguous allocation (SGLang commonly allocates per-layer tensors), fixes only that
   framework, and leaves the plugin's scaling bug in place. Useful at most as an optimization to
   reduce metadata/registration overhead, not as the fix.
5. **One pool per host, shared across processes.** Best possible number, but requires a
   cross-process allocator, lease manager, and crash-cleanup protocol in shared memory.
   Per-process pools already reduce a host to well under 1 GiB. Not worth it now.

## Test plan

- Unit (gtest):
  - TX/RX isolation: saturating one direction never blocks the other.
  - Bidirectional A<->B with `tx=rx=2` completes (regression for deadlock (a)).
  - Bidirectional with `source_d2h_prefetch=true` and small pools completes (regression for
    deadlock (b), the permanent-hang variant).
  - Role enforcement: SLOT_REQ against `rx=0` and local acquire against `tx=0` fail with the
    role error, not a hang.
  - Quarantine: timed-out lease is never re-granted; late WRITE_READY on a quarantined lease
    still completes; disconnect releases the owner's quarantined leases; epoch bump invalidates
    the RemotePool cache.
  - Region tokens: dereg refused while `activeLeaseCount > 0`; SLOT_REQ with a stale token (dereg
    + re-register same address) fails cleanly.
  - Release-by-id across pools; shm generation checks at pool scope.
- Smoke: extend `staged_vram_write_smoke` (cpp/py) to register ~100 small regions and assert
  pinned/shmem stays at `(tx + rx) x chunk_size` regardless of region count, with transfers on
  many regions interleaving through one pool.
- Benchmark: the sizing matrix of section 8 on the SGLang PD zero-prefix setup (20 warmup + 200
  formal) before freezing defaults.

## Implementation map (for scoping)

All in `src/plugins/ucx/`:

- `ucx_backend.h` — `DeviceStagingPool`, RegionRegistry, RemotePool cache, config fields.
- `ucx_backend.cpp`:
  - `nixlUcxStagedPrivateMetadata` — shrinks to a region record (token, range, device); slot,
    lease, and generation machinery moves into `DeviceStagingPool` (mostly code motion) plus the
    TX/RX partition, quarantine state, and per-agent accounting.
  - `registerMem` / `deregisterMem` — pool ensure/lookup, token assignment, lease-count check.
  - `serializeStagedMetadata` / `internalStagedMDHelper` — metadata v2, pool-descriptor dedupe
    into RemotePool.
  - `handleStagedSlotReq` / `handleStagedSlotRelease` / `handleStagedWriteReady` /
    `handleStagedLocalWriteReady` — token lookup, pool routing, quarantine transitions.
  - `postStagedWrite` chunk FSM — TX slots from the pool; computed remote addresses from the
    RemotePool descriptor; window scope change; epoch check on grants.
  - `StagedH2DTask` / H2D worker — drop region pointer.
  - Disconnect path — pool-scoped quarantine of the owner's reserved leases; RemotePool cache
    eviction.
- `ucx_utils.h/.cpp` — new param/env names (`tx`/`rx`/`max_grants_per_agent`), legacy shorthand.
- Docs: update `ucx_vram_staging.md`, `local_vram_staging.md`, SGLang runbook (slot semantics,
  window semantics, lowmem script note).
