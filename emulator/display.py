"""CHIP-8 display: 64x32 monochrome framebuffer."""


class Display:
    WIDTH = 64
    HEIGHT = 32

    def __init__(self):
        self.pixels = [[0] * self.WIDTH for _ in range(self.HEIGHT)]

    def clear(self):
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                self.pixels[y][x] = 0

    def get_pixel(self, x: int, y: int) -> int:
        return self.pixels[y % self.HEIGHT][x % self.WIDTH]

    def xor_pixel(self, x: int, y: int) -> bool:
        """XOR a pixel. Returns True if collision (pixel was on, now off)."""
        x %= self.WIDTH
        y %= self.HEIGHT
        old = self.pixels[y][x]
        self.pixels[y][x] ^= 1
        return old == 1 and self.pixels[y][x] == 0

    def draw_sprite(self, x: int, y: int, sprite: list[int]) -> bool:
        """Draw sprite at (x, y). Returns True if any collision occurred."""
        collision = False
        for row, byte in enumerate(sprite):
            for bit in range(8):
                if byte & (0x80 >> bit):
                    if self.xor_pixel(x + bit, y + row):
                        collision = True
        return collision

    def buffer(self) -> list[list[int]]:
        return [row[:] for row in self.pixels]
