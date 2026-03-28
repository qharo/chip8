"""CHIP-8 keypad: 16 keys (0x0 - 0xF)."""


class Keypad:
    def __init__(self):
        self.keys = [False] * 16

    def press(self, key: int):
        self.keys[key & 0xF] = True

    def release(self, key: int):
        self.keys[key & 0xF] = False

    def is_pressed(self, key: int) -> bool:
        return self.keys[key & 0xF]

    def any_pressed(self) -> int | None:
        """Return first pressed key value, or None."""
        for i, pressed in enumerate(self.keys):
            if pressed:
                return i
        return None

    def reset(self):
        self.keys = [False] * 16
