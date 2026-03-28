"""Tracer: converts CPU execution into structured token traces."""

from __future__ import annotations

from .cpu import CPU


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


def format_mem_snapshot(memory) -> str:
    """Format memory non-zero bytes as token string."""
    snapshot = memory.snapshot()
    entries = " ".join(f"<{addr:04X}:{val:02X}>" for addr, val in snapshot)
    return f"<MEM_START> {entries} <MEM_END>"


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
                   snapshot_interval: int = 0) -> list[str]:
    """Run CPU and produce list of trace lines.

    Args:
        cpu: Running CPU instance
        max_cycles: Maximum cycles to execute
        snapshot_interval: If > 0, re-inject state snapshot every N cycles

    Returns:
        List of token strings (trace lines and optional snapshots)
    """
    lines = []
    lines.append(format_mem_snapshot(cpu.memory))
    lines.append(format_reg_snapshot(cpu.state))
    lines.append("<TRACE_START>")

    for cycle in range(max_cycles):
        trace = cpu.step()

        if trace.get("halted"):
            break

        if trace.get("waiting_for_key"):
            continue

        line = format_trace_line(trace)
        if line:
            lines.append(line)

        if snapshot_interval > 0 and (cycle + 1) % snapshot_interval == 0:
            lines.append(format_reg_snapshot(cpu.state))

    return lines
