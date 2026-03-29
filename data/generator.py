"""Data generator: produces training traces from ROMs and fuzzing."""

from __future__ import annotations

import os
import random
import struct
from pathlib import Path

from emulator.cpu import CPU
from emulator.tracer import generate_trace

# Classic CHIP-8 ROMs (common free ROMs)
# We'll generate synthetic ROMs since we don't have ROM files yet.
# Users can drop real .ch8 files into the roms/ directory.

OPCODE_TEMPLATES = [
    # (generator_func_name, weight)
    ("gen_cls", 2),
    ("gen_ret", 2),
    ("gen_jp", 5),
    ("gen_call", 3),
    ("gen_se_vx_nn", 5),
    ("gen_sne_vx_nn", 5),
    ("gen_se_vx_vy", 4),
    ("gen_ld_vx_nn", 10),
    ("gen_add_vx_nn", 8),
    ("gen_ld_vx_vy", 5),
    ("gen_or_vx_vy", 3),
    ("gen_and_vx_vy", 3),
    ("gen_xor_vx_vy", 3),
    ("gen_add_vx_vy", 5),
    ("gen_sub_vx_vy", 4),
    ("gen_shr_vx_vy", 2),
    ("gen_subn_vx_vy", 2),
    ("gen_shl_vx_vy", 2),
    ("gen_sne_vx_vy", 4),
    ("gen_ld_i_addr", 5),
    ("gen_jp_v0_addr", 2),
    ("gen_rnd_vx_nn", 3),
    ("gen_drw", 8),
    ("gen_skp", 2),
    ("gen_sknp", 2),
    ("gen_ld_vx_dt", 2),
    ("gen_ld_dt_vx", 3),
    ("gen_ld_st_vx", 2),
    ("gen_add_i_vx", 3),
    ("gen_ld_f_vx", 3),
    ("gen_ld_b_vx", 2),
    ("gen_ld_i_vx", 3),
    ("gen_ld_vx_i", 3),
]


def gen_cls(rng: random.Random) -> bytes:
    return bytes([0x00, 0xE0])

def gen_ret(rng: random.Random) -> bytes:
    return bytes([0x00, 0xEE])

def gen_jp(rng: random.Random) -> bytes:
    addr = rng.randint(0x200, 0xFFF)
    return bytes([0x10 | ((addr >> 8) & 0xF), addr & 0xFF])

def gen_call(rng: random.Random) -> bytes:
    addr = rng.randint(0x200, 0xFFF)
    return bytes([0x20 | ((addr >> 8) & 0xF), addr & 0xFF])

def gen_se_vx_nn(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    nn = rng.randint(0, 255)
    return bytes([0x30 | x, nn])

def gen_sne_vx_nn(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    nn = rng.randint(0, 255)
    return bytes([0x40 | x, nn])

def gen_se_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x50 | x, (y << 4) | 0x0])

def gen_ld_vx_nn(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    nn = rng.randint(0, 255)
    return bytes([0x60 | x, nn])

def gen_add_vx_nn(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    nn = rng.randint(0, 255)
    return bytes([0x70 | x, nn])

def gen_ld_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x0])

def gen_or_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x1])

def gen_and_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x2])

def gen_xor_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x3])

def gen_add_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x4])

def gen_sub_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x5])

def gen_shr_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x6])

def gen_subn_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0x7])

def gen_shl_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x80 | x, (y << 4) | 0xE])

def gen_sne_vx_vy(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    return bytes([0x90 | x, (y << 4) | 0x0])

def gen_ld_i_addr(rng: random.Random) -> bytes:
    addr = rng.randint(0x000, 0xFFF)
    return bytes([0xA0 | ((addr >> 8) & 0xF), addr & 0xFF])

def gen_jp_v0_addr(rng: random.Random) -> bytes:
    addr = rng.randint(0x200, 0xFFF)
    return bytes([0xB0 | ((addr >> 8) & 0xF), addr & 0xFF])

def gen_rnd_vx_nn(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    nn = rng.randint(0, 255)
    return bytes([0xC0 | x, nn])

def gen_drw(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    y = rng.randint(0, 15)
    n = rng.randint(1, 15)
    return bytes([0xD0 | x, (y << 4) | n])

def gen_skp(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xE0 | x, 0x9E])

def gen_sknp(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xE0 | x, 0xA1])

def gen_ld_vx_dt(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x07])

def gen_ld_dt_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x15])

def gen_ld_st_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x18])

def gen_add_i_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x1E])

def gen_ld_f_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x29])

def gen_ld_b_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x33])

def gen_ld_i_vx(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x55])

def gen_ld_vx_i(rng: random.Random) -> bytes:
    x = rng.randint(0, 15)
    return bytes([0xF0 | x, 0x65])


GENERATORS = {name: globals()[name] for name, _ in OPCODE_TEMPLATES}
WEIGHTS = [w for _, w in OPCODE_TEMPLATES]


def generate_random_rom(rng: random.Random, num_instructions: int = 64) -> bytes:
    """Generate a random ROM with valid opcodes."""
    gen_names = [name for name, _ in OPCODE_TEMPLATES]
    rom = bytearray()

    for _ in range(num_instructions):
        gen_name = rng.choices(gen_names, weights=WEIGHTS, k=1)[0]
        opcode_bytes = GENERATORS[gen_name](rng)
        rom.extend(opcode_bytes)

    return bytes(rom)


def _inject_key_events(rng: random.Random, cpu, lines: list[str], cycle: int):
    """Randomly inject key press/release events during trace generation."""
    if rng.random() < 0.05:  # 5% chance to press a key
        key = rng.randint(0, 15)
        cpu.keypad.press(key)
        lines.append(f"<KEY_{key:X}>")
    elif rng.random() < 0.03:  # 3% chance to release all keys
        cpu.keypad.reset()
        lines.append("<NO_KEY>")


def load_rom_file(path: str) -> bytes:
    """Load a .ch8 ROM file."""
    with open(path, "rb") as f:
        return f.read()


def generate_traces_from_rom(rom: bytes, seed: int = 42,
                              max_cycles: int = 2000,
                              snapshot_interval: int = 200) -> list[str]:
    """Run a ROM and generate trace lines with keyboard injection."""
    cpu = CPU(seed=seed)
    cpu.load_rom(rom)

    def _key_fn(lines, cycle):
        _inject_key_events(random.Random(seed + cycle), cpu, lines, cycle)

    return generate_trace(cpu, max_cycles=max_cycles,
                          snapshot_interval=snapshot_interval,
                          key_event_fn=_key_fn)


def generate_rom_trace(rom: bytes, rng: random.Random,
                       cycles: int = 2000,
                       snapshot_interval: int = 200) -> list[str]:
    """Generate a single ROM's trace (for per-ROM train/val splitting)."""
    rom_seed = rng.randint(0, 2**31)
    lines = generate_traces_from_rom(rom, seed=rom_seed,
                                     max_cycles=cycles,
                                     snapshot_interval=snapshot_interval)
    lines.append("<SEP>")
    return lines


def generate_dataset(num_random_roms: int = 500,
                     instructions_per_rom: int = 64,
                     cycles_per_rom: int = 2000,
                     rom_dir: str | None = None,
                     seed: int = 42,
                     per_rom: bool = False):
    """Generate a mixed dataset of traces from random ROMs and real ROMs.

    Args:
        per_rom: If True, return list[list[str]] (one list per ROM).
                 If False, return flat list[str] (backward compatible).

    Returns:
        list[str] if per_rom=False, list[list[str]] if per_rom=True.
    """
    rng = random.Random(seed)
    rom_traces = []

    # Random ROMs (fuzzing) — each ROM gets its own seeded RNG
    for i in range(num_random_roms):
        # Per-ROM RNG: unique seed derived from master seed + index
        rom_rng = random.Random(seed + i * 7919)

        # Variable ROM characteristics
        inst_count = rom_rng.randint(16, 256)
        cycle_count = rom_rng.randint(500, 5000)
        snap_interval = rom_rng.choice([100, 200, 500, 0])

        rom = generate_random_rom(rom_rng, inst_count)
        rom_seed = rng.randint(0, 2**31)
        lines = generate_traces_from_rom(rom, seed=rom_seed,
                                         max_cycles=cycle_count,
                                         snapshot_interval=snap_interval)
        lines.append("<SEP>")
        rom_traces.append(lines)

    # Real ROMs if directory provided
    if rom_dir and os.path.isdir(rom_dir):
        for fname in sorted(os.listdir(rom_dir)):
            if fname.endswith(".ch8") or fname.endswith(".rom"):
                path = os.path.join(rom_dir, fname)
                rom = load_rom_file(path)
                rom_seed = rng.randint(0, 2**31)
                lines = generate_traces_from_rom(rom, seed=rom_seed,
                                                 max_cycles=cycles_per_rom,
                                                 snapshot_interval=200)
                lines.append("<SEP>")
                rom_traces.append(lines)

    if per_rom:
        return rom_traces
    return [line for trace in rom_traces for line in trace]
