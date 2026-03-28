"""CHIP-8 memory: 4KB RAM with font data."""

FONT_DATA = [
    0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
    0x20, 0x60, 0x20, 0x20, 0x70,  # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
    0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
    0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
    0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
    0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
    0xF0, 0x80, 0xF0, 0x80, 0x80,  # F
]


class Memory:
    """4096-byte CHIP-8 RAM."""

    def __init__(self):
        self.ram = bytearray(4096)
        self._load_font()

    def _load_font(self):
        for i, byte in enumerate(FONT_DATA):
            self.ram[i] = byte

    def reset(self):
        self.ram = bytearray(4096)
        self._load_font()

    def load_rom(self, rom: bytes, addr: int = 0x200):
        for i, byte in enumerate(rom):
            self.ram[addr + i] = byte

    def read_byte(self, addr: int) -> int:
        return self.ram[addr & 0xFFF]

    def write_byte(self, addr: int, value: int):
        self.ram[addr & 0xFFF] = value & 0xFF

    def read_word(self, addr: int) -> int:
        """Read 16-bit big-endian word."""
        return (self.ram[addr & 0xFFF] << 8) | self.ram[(addr + 1) & 0xFFF]

    def read_range(self, start: int, length: int) -> list[int]:
        return [self.ram[(start + i) & 0xFFF] for i in range(length)]

    def snapshot(self, start_addr: int = 0x200) -> list[tuple[int, int]]:
        """Return list of (addr, value) for non-zero bytes from start_addr onward."""
        return [(addr, val) for addr, val in enumerate(self.ram) if val != 0 and addr >= start_addr]
