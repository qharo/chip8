"""Tracer: converts CPU execution into structured token traces."""

from __future__ import annotations

from typing import Callable

from .cpu import CPU


# Max entries in a memory snapshot to keep token count manageable
MAX_MEM_SNAPSHOT_ENTRIES = 100


def format_trace_line(trace: dict) -> str:
    """Convert a single cycle trace dict into token string.

    Format: [PC:XXXX] [FETCH:XXXX] [OP:MNEMONIC] [W_REG:VAL] ...
    """
    if trace.get("halted") or trace.get("waiting_for_key"):
        return ""

    pc = trace["pc"]
    opcode = trace["opcode"]
    mnemonic = trace["mnemonic"]

    parts = [
        f"[PC:{pc:04X}]",
        f"[FETCH:{opcode:04X}]",
        f"[OP:{mnemonic}]",
    ]

    for write_type, write_val in trace["writes"]:
        if write_type.startswith("W_V"):
            parts.append(f"[{write_type}:{write_val:02X}]")
        elif write_type == "W_I":
            parts.append(f"[W_I:{write_val:03X}]")
        elif write_type == "W_PC":
            parts.append(f"[W_PC:{write_val:04X}]")
        elif write_type == "W_DT":
            parts.append(f"[W_DT:{write_val:02X}]")
        elif write_type == "W_ST":
            parts.append(f"[W_ST:{write_val:02X}]")
        elif write_type == "W_MEM":
            parts.append(f"[W_MEM:{write_val}]")
        elif write_type == "W_DISPLAY":
            parts.append("[W_DISPLAY:1]")

    return " ".join(parts)


def format_mem_snapshot(memory, written_addrs: set[int] | None = None) -> str:
    """Format memory non-zero bytes as token string.

    If written_addrs is provided, only snapshot those addresses (capped).
    Otherwise fall back to full non-zero snapshot from start_addr onward.
    """
    if written_addrs:
        entries = [(a, memory.read_byte(a)) for a in sorted(written_addrs)]
    else:
        entries = memory.snapshot()
    entries = entries[:MAX_MEM_SNAPSHOT_ENTRIES]
    parts = " ".join(f"<{addr:04X}:{val:02X}>" for addr, val in entries)
    return f"<MEM_START> {parts} <MEM_END>"


def format_reg_snapshot(state) -> str:
    """Format register state as token string."""
    parts = []
    for i in range(16):
        parts.append(f"<V{i:X}:{state.V[i]:02X}>")
    parts.append(f"<I:{state.I:03X}>")
    parts.append(f"<PC:{state.PC:04X}>")
    parts.append(f"<DT:{state.DT:02X}>")
    parts.append(f"<ST:{state.ST:02X}>")
    parts.append(f"<SP:{state.SP:02X}>")
    return f"<REG_START> {' '.join(parts)} <REG_END>"


def generate_trace(cpu: CPU, max_cycles: int = 10000,
                   snapshot_interval: int = 0,
                   key_event_fn: Callable | None = None) -> list[str]:
    """Run CPU and produce list of trace lines.

    Args:
        cpu: Running CPU instance
        max_cycles: Maximum cycles to execute
        snapshot_interval: If > 0, re-inject state snapshot every N cycles
        key_event_fn: Optional callback(cycle_lines, all_lines, cycle) to inject key events

    Returns:
        List of token strings (trace lines and optional snapshots)
    """
    lines = []
    lines.append(format_mem_snapshot(cpu.memory))
    lines.append(format_reg_snapshot(cpu.state))
    lines.append("<TRACE_START>")

    # Track which memory addresses have been written to (for trimmed snapshots)
    written_addrs: set[int] = set()

    for cycle in range(max_cycles):
        # Tick timers at ~60Hz (every cycle in emulation time)
        cpu.tick_timers()

        # Inject keyboard events before stepping
        if key_event_fn is not None:
            key_event_fn(lines, cycle)

        trace = cpu.step()

        if trace.get("halted"):
            break

        if trace.get("waiting_for_key"):
            continue

        # Track memory writes for trimmed snapshots
        for write_type, write_val in trace["writes"]:
            if write_type == "W_MEM":
                # write_val format: "03F:AB" (addr:byte)
                addr_str = write_val.split(":")[0]
                written_addrs.add(int(addr_str, 16))

        line = format_trace_line(trace)
        if line:
            lines.append(line)

        if snapshot_interval > 0 and (cycle + 1) % snapshot_interval == 0:
            lines.append(format_reg_snapshot(cpu.state))
            # Periodic trimmed memory snapshot
            if written_addrs:
                lines.append(format_mem_snapshot(cpu.memory, written_addrs))

    return lines
