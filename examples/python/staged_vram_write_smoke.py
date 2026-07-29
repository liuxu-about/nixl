#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import torch

from nixl import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def wait_until(predicate, label, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}")
        time.sleep(0.01)


def make_agent(name, listen_port, backend_params):
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=listen_port,
        backends=[],
    )
    agent = nixl_agent(name, cfg)
    agent.create_backend("UCX", backend_params)
    return agent


def wait_for_xfer(agent, handle, state, timeout_s):
    def complete():
        nonlocal state
        if state != "PROC":
            return True
        state = agent.check_xfer_state(handle)
        return state != "PROC"

    wait_until(complete, "transfer completion", timeout_s)
    return state


def proc_status_kib(field):
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith(field):
                    return line.split()[1]
    except OSError:
        pass
    return "unavailable"


def log_pool_memory(args, role):
    expected_pool_bytes = 2 * args.slots * args.chunk_size
    logger.info(
        "%s pool_memory regions=%d expected_pool_bytes=%d vm_pin_kib=%s rss_shmem_kib=%s",
        role,
        args.regions,
        expected_pool_bytes,
        proc_status_kib("VmPin:"),
        proc_status_kib("RssShmem:"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["target", "initiator"], required=True)
    parser.add_argument("--ip", help="Target listener IP, required by initiator")
    parser.add_argument("--port", type=int, default=5561)
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--regions", type=int, default=1)
    parser.add_argument("--ucx-devices", default="")
    parser.add_argument("--ucx-tls", default="rc,ud,self")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.elements <= 0 or args.chunk_size <= 0 or args.slots <= 0 or args.regions <= 0:
        raise ValueError("--elements, --chunk-size, --slots, and --regions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for staged VRAM smoke test")

    backend_params = {
        "vram_staging": "true",
        "staging_chunk_size": str(args.chunk_size),
        "staging_tx_slots_per_gpu": str(args.slots),
        "staging_rx_slots_per_gpu": str(args.slots),
        "staging_force_progress_thread": "true",
    }
    if args.ucx_devices:
        backend_params["ucx_devices"] = args.ucx_devices
    if args.ucx_tls:
        os.environ.setdefault("UCX_TLS", args.ucx_tls)

    if args.mode == "target":
        agent = make_agent("target", args.port, backend_params)
        tensors = [
            torch.full(
                (args.elements,), -1, dtype=torch.int32, device="cuda:0"
            )
            for _ in range(args.regions)
        ]
        torch.cuda.synchronize()

        registration = agent.register_memory(tensors)
        target_descs = agent.get_xfer_descs(tensors)
        log_pool_memory(args, "target")
        target_desc_msg = b"DESC:" + agent.get_serialized_descs(target_descs)

        wait_until(
            lambda: agent.check_remote_metadata("initiator"),
            "initiator metadata",
            args.timeout,
        )
        agent.send_notif("initiator", target_desc_msg)
        logger.info("Target descriptors sent")

        wait_until(
            lambda: agent.check_remote_xfer_done(
                "initiator", b"DONE_WRITE", backends=["UCX"]
            ),
            "write completion notification",
            args.timeout,
        )
        torch.cuda.synchronize()

        expected = torch.arange(args.elements, dtype=torch.int32, device="cuda:0")
        for region, tensor in enumerate(tensors):
            if not torch.equal(tensor, expected):
                mismatches = torch.nonzero(tensor != expected, as_tuple=False).flatten()
                first = int(mismatches[0].item()) if mismatches.numel() else -1
                got = int(tensor[first].item()) if first >= 0 else 0
                want = int(expected[first].item()) if first >= 0 else 0
                raise RuntimeError(
                    f"verification failed in region {region} at index {first}: "
                    f"got={got} want={want}"
                )

        logger.info(
            "Target verification passed: %d regions, %d bytes total",
            args.regions,
            sum(tensor.numel() * tensor.element_size() for tensor in tensors),
        )
        agent.deregister_memory(registration)
        return

    if not args.ip:
        raise ValueError("--ip is required in initiator mode")

    agent = make_agent("initiator", 0, backend_params)
    tensors = [
        torch.arange(args.elements, dtype=torch.int32, device="cuda:0")
        for _ in range(args.regions)
    ]
    torch.cuda.synchronize()

    registration = agent.register_memory(tensors)
    local_descs = agent.get_xfer_descs(tensors)
    log_pool_memory(args, "initiator")

    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)

    target_descs = None

    def got_descs():
        nonlocal target_descs
        notifs = agent.get_new_notifs(backends=["UCX"])
        for msg in notifs.get("target", []):
            if msg.startswith(b"DESC:"):
                target_descs = agent.deserialize_descs(msg[len(b"DESC:") :])
                return True
        return False

    wait_until(got_descs, "target descriptors", args.timeout)
    wait_until(
        lambda: agent.check_remote_metadata("target"),
        "target metadata",
        args.timeout,
    )

    xfer_handle = agent.initialize_xfer(
        "WRITE",
        local_descs,
        target_descs,
        "target",
        notif_msg=b"DONE_WRITE",
        backends=["UCX"],
    )
    logger.info("Transfer backend: %s", agent.query_xfer_backend(xfer_handle))

    state = wait_for_xfer(agent, xfer_handle, agent.transfer(xfer_handle), args.timeout)
    if state != "DONE":
        raise RuntimeError(f"transfer ended in {state}")

    logger.info(
        "Initiator WRITE completed: %d regions, %d bytes total",
        args.regions,
        sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    )
    agent.release_xfer_handle(xfer_handle)
    agent.remove_remote_agent("target")
    agent.invalidate_local_metadata(args.ip, args.port)
    agent.deregister_memory(registration)


if __name__ == "__main__":
    main()
