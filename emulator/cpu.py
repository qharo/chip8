"""CHIP-8 CPU: fetch-decode-execute with all 35 opcodes (COSMAC VIP quirks)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .memory import Memory
from .display import Display
from .keypad import Keypad

# Opcode mnemonics
OPCODE_NAMES = {
    0x00E0: "CLS",
    0x00EE: "RET",
    0x1: "JP_ADDR",
    0x2: "CALL_ADDR",
    0x3: "SE_VX_NN",
    0x4: "SNE_VX_NN",
    0x5: "SE_VX_VY",
    0x6: "LD_VX_NN",
    0x7: "ADD_VX_NN",
    0x80: "LD_VX_VY",
    0x81: "OR_VX_VY",
    0x82: "AND_VX_VY",
    0x83: "XOR_VX_VY",
    0x84: "ADD_VX_VY",
    0x85: "SUB_VX_VY",
    0x86: "SHR_VX_VY",
    0x87: "SUBN_VX_VY",
    0x8E: "SHL_VX_VY",
    0x9: "SNE_VX_VY",
    0xA: "LD_I_ADDR",
    0xB: "JP_V0_ADDR",
    0xC: "RND_VX_NN",
    0xD: "DRW_VX_VY_N",
    0xE9E: "SKP_VX",
    0xEA1: "SKNP_VX",
    0xF07: "LD_VX_DT",
    0xF0A: "LD_VX_K",
    0xF15: "LD_DT_VX",
    0xF18: "LD_ST_VX",
    0xF1E: "ADD_I_VX",
    0xF29: "LD_F_VX",
    0xF33: "LD_B_VX",
    0xF55: "LD_I_VX",
    0xF65: "LD_VX_I",
}


def decode_opcode(opcode: int) -> str:
    """Return mnemonic for an opcode."""
    if opcode == 0x00E0:
        return "CLS"
    if opcode == 0x00EE:
        return "RET"
    hi = opcode >> 8
    lo = opcode & 0xFF
    nnn = opcode & 0x0FFF
    x = (opcode >> 8) & 0x0F
    y = (opcode >> 4) & 0x0F
    n = opcode & 0x0F
    nn = opcode & 0xFF

    family = hi >> 4
    sub = hi & 0x0F

    if family == 0x1:
        return "JP_ADDR"
    if family == 0x2:
        return "CALL_ADDR"
    if family == 0x3:
        return "SE_VX_NN"
    if family == 0x4:
        return "SNE_VX_NN"
    if family == 0x5:
        return "SE_VX_VY"
    if family == 0x6:
        return "LD_VX_NN"
    if family == 0x7:
        return "ADD_VX_NN"
    if family == 0x8:
        return OPCODE_NAMES.get(0x80 | n, f"UNKNOWN_{opcode:04X}")
    if family == 0x9:
        return "SNE_VX_VY"
    if family == 0xA:
        return "LD_I_ADDR"
    if family == 0xB:
        return "JP_V0_ADDR"
    if family == 0xC:
        return "RND_VX_NN"
    if family == 0xD:
        return "DRW_VX_VY_N"
    if family == 0xE:
        if lo == 0x9E:
            return "SKP_VX"
        if lo == 0xA1:
            return "SKNP_VX"
    if family == 0xF:
        return OPCODE_NAMES.get(0xF00 | lo, f"UNKNOWN_{opcode:04X}")

    return f"UNKNOWN_{opcode:04X}"


@dataclass
class CPUState:
    V: list[int] = field(default_factory=lambda: [0] * 16)
    I: int = 0
    PC: int = 0x200
    SP: int = 0
    stack: list[int] = field(default_factory=lambda: [0] * 16)
    DT: int = 0
    ST: int = 0


class CPU:
    def __init__(self, seed: int | None = None):
        self.state = CPUState()
        self.memory = Memory()
        self.display = Display()
        self.keypad = Keypad()
        self.rng = random.Random(seed)
        self.halted = False
        self.waiting_for_key = False
        self.wait_key_reg = 0

    def reset(self):
        self.state = CPUState()
        self.display.clear()
        self.keypad.reset()
        self.halted = False
        self.waiting_for_key = False

    def load_rom(self, rom: bytes):
        self.memory.reset()
        self.memory.load_rom(rom)

    def tick_timers(self):
        if self.state.DT > 0:
            self.state.DT -= 1
        if self.state.ST > 0:
            self.state.ST -= 1

    def step(self) -> dict:
        """Execute one cycle. Returns trace dict of what happened."""
        if self.halted:
            return {"halted": True}

        if self.waiting_for_key:
            key = self.keypad.any_pressed()
            if key is not None:
                self.state.V[self.wait_key_reg] = key
                self.waiting_for_key = False
                self.state.PC += 2
            return {"waiting_for_key": True}

        s = self.state
        pc = s.PC
        opcode = self.memory.read_word(pc)
        mnemonic = decode_opcode(opcode)

        trace = {
            "pc": pc,
            "opcode": opcode,
            "mnemonic": mnemonic,
            "writes": [],
        }

        family = (opcode >> 12) & 0x0F
        x = (opcode >> 8) & 0x0F
        y = (opcode >> 4) & 0x0F
        n = opcode & 0x0F
        nn = opcode & 0xFF
        nnn = opcode & 0x0FFF

        if opcode == 0x00E0:
            # CLS
            self.display.clear()
            s.PC += 2

        elif opcode == 0x00EE:
            # RET
            s.SP = (s.SP - 1) & 0xF
            s.PC = s.stack[s.SP]
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x1:
            # JP addr
            s.PC = nnn
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x2:
            # CALL addr
            s.stack[s.SP] = s.PC + 2
            s.SP = (s.SP + 1) & 0xF
            s.PC = nnn
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x3:
            # SE Vx, nn
            if s.V[x] == nn:
                s.PC += 4
            else:
                s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x4:
            # SNE Vx, nn
            if s.V[x] != nn:
                s.PC += 4
            else:
                s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x5:
            # SE Vx, Vy
            if s.V[x] == s.V[y]:
                s.PC += 4
            else:
                s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x6:
            # LD Vx, nn
            s.V[x] = nn
            s.PC += 2
            trace["writes"].append((f"W_V{x:X}", nn))
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x7:
            # ADD Vx, nn (no carry)
            s.V[x] = (s.V[x] + nn) & 0xFF
            s.PC += 2
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x8:
            self._exec_arithmetic(opcode, x, y, n, trace)
            s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0x9:
            # SNE Vx, Vy
            if s.V[x] != s.V[y]:
                s.PC += 4
            else:
                s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xA:
            # LD I, addr
            s.I = nnn
            s.PC += 2
            trace["writes"].append(("W_I", nnn))
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xB:
            # JP V0, addr
            s.PC = nnn + s.V[0]
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xC:
            # RND Vx, nn
            r = self.rng.randint(0, 255)
            s.V[x] = r & nn
            s.PC += 2
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xD:
            # DRW Vx, Vy, n
            sprite = self.memory.read_range(s.I, n)
            collision = self.display.draw_sprite(s.V[x], s.V[y], sprite)
            s.V[0xF] = 1 if collision else 0
            s.PC += 2
            trace["writes"].append(("W_VF", s.V[0xF]))
            trace["writes"].append(("W_DISPLAY", 1))
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xE:
            if nn == 0x9E:
                # SKP Vx
                if self.keypad.is_pressed(s.V[x] & 0xF):
                    s.PC += 4
                else:
                    s.PC += 2
            elif nn == 0xA1:
                # SKNP Vx
                if not self.keypad.is_pressed(s.V[x] & 0xF):
                    s.PC += 4
                else:
                    s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        elif family == 0xF:
            self._exec_misc(opcode, x, nn, trace)
            s.PC += 2
            trace["writes"].append(("W_PC", s.PC))

        return trace

    def _exec_arithmetic(self, opcode: int, x: int, y: int, n: int, trace: dict):
        s = self.state
        if n == 0x0:
            # LD Vx, Vy
            s.V[x] = s.V[y]
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
        elif n == 0x1:
            # OR Vx, Vy (COSMAC: VF reset)
            s.V[x] = (s.V[x] | s.V[y]) & 0xFF
            s.V[0xF] = 0
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", 0))
        elif n == 0x2:
            # AND Vx, Vy (COSMAC: VF reset)
            s.V[x] = (s.V[x] & s.V[y]) & 0xFF
            s.V[0xF] = 0
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", 0))
        elif n == 0x3:
            # XOR Vx, Vy (COSMAC: VF reset)
            s.V[x] = (s.V[x] ^ s.V[y]) & 0xFF
            s.V[0xF] = 0
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", 0))
        elif n == 0x4:
            # ADD Vx, Vy
            result = s.V[x] + s.V[y]
            s.V[x] = result & 0xFF
            s.V[0xF] = 1 if result > 0xFF else 0
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", s.V[0xF]))
        elif n == 0x5:
            # SUB Vx, Vy
            s.V[0xF] = 1 if s.V[x] >= s.V[y] else 0
            s.V[x] = (s.V[x] - s.V[y]) & 0xFF
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", s.V[0xF]))
        elif n == 0x6:
            # SHR Vx, {Vy} — COSMAC: shift VY
            s.V[0xF] = s.V[y] & 1
            s.V[x] = s.V[y] >> 1
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", s.V[0xF]))
        elif n == 0x7:
            # SUBN Vx, Vy
            s.V[0xF] = 1 if s.V[y] >= s.V[x] else 0
            s.V[x] = (s.V[y] - s.V[x]) & 0xFF
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", s.V[0xF]))
        elif n == 0xE:
            # SHL Vx, {Vy} — COSMAC: shift VY
            s.V[0xF] = (s.V[y] >> 7) & 1
            s.V[x] = (s.V[y] << 1) & 0xFF
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
            trace["writes"].append(("W_VF", s.V[0xF]))

    def _exec_misc(self, opcode: int, x: int, nn: int, trace: dict):
        s = self.state
        if nn == 0x07:
            # LD Vx, DT
            s.V[x] = s.DT
            trace["writes"].append((f"W_V{x:X}", s.V[x]))
        elif nn == 0x0A:
            # LD Vx, K — wait for key
            self.waiting_for_key = True
            self.wait_key_reg = x
        elif nn == 0x15:
            # LD DT, Vx
            s.DT = s.V[x]
            trace["writes"].append(("W_DT", s.DT))
        elif nn == 0x18:
            # LD ST, Vx
            s.ST = s.V[x]
            trace["writes"].append(("W_ST", s.ST))
        elif nn == 0x1E:
            # ADD I, Vx
            s.I = (s.I + s.V[x]) & 0xFFF
            trace["writes"].append(("W_I", s.I))
        elif nn == 0x29:
            # LD F, Vx — font sprite location
            s.I = (s.V[x] & 0x0F) * 5
            trace["writes"].append(("W_I", s.I))
        elif nn == 0x33:
            # LD B, Vx — BCD
            val = s.V[x]
            self.memory.write_byte(s.I, val // 100)
            self.memory.write_byte(s.I + 1, (val // 10) % 10)
            self.memory.write_byte(s.I + 2, val % 10)
            trace["writes"].append(("W_MEM", f"{s.I:03X}:{val // 100:02X}"))
            trace["writes"].append(("W_MEM", f"{s.I + 1:03X}:{(val // 10) % 10:02X}"))
            trace["writes"].append(("W_MEM", f"{s.I + 2:03X}:{val % 10:02X}"))
        elif nn == 0x55:
            # LD [I], V0..Vx — COSMAC: I increments
            for i in range(x + 1):
                self.memory.write_byte(s.I + i, s.V[i])
                trace["writes"].append(("W_MEM", f"{s.I + i:03X}:{s.V[i]:02X}"))
            s.I = (s.I + x + 1) & 0xFFF
            trace["writes"].append(("W_I", s.I))
        elif nn == 0x65:
            # LD V0..Vx, [I] — COSMAC: I increments
            for i in range(x + 1):
                s.V[i] = self.memory.read_byte(s.I + i)
                trace["writes"].append((f"W_V{i:X}", s.V[i]))
            s.I = (s.I + x + 1) & 0xFFF
            trace["writes"].append(("W_I", s.I))
