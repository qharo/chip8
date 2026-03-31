"""Custom CHIP-8 tokenizer. No BPE — every concept is one token."""

from __future__ import annotations


def _build_vocab() -> tuple[dict[str, int], dict[int, str], list[str]]:
    tokens = []
    idx = 0

    # Pad token
    PAD = "<PAD>"
    tokens.append(PAD)

    # Hex bytes: <00> through <FF>
    for i in range(256):
        tokens.append(f"<{i:02X}>")

    # Registers: <V0>..<VF>
    for i in range(16):
        tokens.append(f"<V{i:X}>")

    # Special registers
    for reg in ["<PC>", "<I>", "<DT>", "<ST>", "<SP>"]:
        tokens.append(reg)

    # Trace labels
    tokens.append("<PC:>")
    tokens.append("<FETCH:>")
    tokens.append("<OP:>")

    # Write labels: W_V0..W_VF, W_I, W_PC, W_DT, W_ST, W_SP, W_MEM:, W_DISPLAY:
    for i in range(16):
        tokens.append(f"<W_V{i:X}>")
    tokens.append("<W_I>")
    tokens.append("<W_PC>")
    tokens.append("<W_DT>")
    tokens.append("<W_ST>")
    tokens.append("<W_SP>")
    tokens.append("<W_MEM:>")
    tokens.append("<W_DISPLAY>")

    # Opcode mnemonics
    opcodes = [
        "CLS", "RET", "JP_ADDR", "CALL_ADDR", "SE_VX_NN", "SNE_VX_NN",
        "SE_VX_VY", "LD_VX_NN", "ADD_VX_NN", "LD_VX_VY", "OR_VX_VY",
        "AND_VX_VY", "XOR_VX_VY", "ADD_VX_VY", "SUB_VX_VY", "SHR_VX_VY",
        "SUBN_VX_VY", "SHL_VX_VY", "SNE_VX_VY", "LD_I_ADDR", "JP_V0_ADDR",
        "RND_VX_NN", "DRW_VX_VY_N", "SKP_VX", "SKNP_VX", "LD_VX_DT",
        "LD_VX_K", "LD_DT_VX", "LD_ST_VX", "ADD_I_VX", "LD_F_VX",
        "LD_B_VX", "LD_I_VX", "LD_VX_I",
    ]
    for op in opcodes:
        tokens.append(f"<{op}>")

    # Control tokens
    for ctrl in ["<MEM_START>", "<MEM_END>", "<REG_START>", "<REG_END>",
                 "<TRACE_START>", "<SEP>", "<ADDR:>", "<VAL:>"]:
        tokens.append(ctrl)

    # Keypad tokens
    for i in range(16):
        tokens.append(f"<KEY_{i:X}>")
    tokens.append("<KEY_UP>")
    tokens.append("<KEY_DOWN>")
    tokens.append("<KEY_LEFT>")
    tokens.append("<KEY_RIGHT>")
    tokens.append("<NO_KEY>")

    # Brackets for trace parsing
    tokens.append("[")
    tokens.append("]")

    # Colon
    tokens.append(":")

    # Build mappings
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}

    return token_to_id, id_to_token, tokens


class Tokenizer:
    """Custom CHIP-8 tokenizer."""

    def __init__(self):
        self.token_to_id, self.id_to_token, self.vocab = _build_vocab()
        self.vocab_size: int = len(self.vocab)
        self.pad_id = self.token_to_id["<PAD>"]

    def encode_line(self, line: str) -> list[int]:
        """Encode a single trace line or snapshot line into token IDs.

        We parse the structured format by splitting on spaces and colons,
        then looking up each token.
        """
        ids = []
        # Split on spaces first
        parts = line.split()
        for part in parts:
            # Handle bracketed tokens like [PC:0200]
            if part.startswith("[") and part.endswith("]"):
                inner = part[1:-1]
                ids.append(self.token_to_id["["])
                self._encode_inner(inner, ids)
                ids.append(self.token_to_id["]"])
            # Handle angle-bracket tokens like <MEM_START>, <0200:60>, <V0:00>
            elif part.startswith("<") and part.endswith(">"):
                inner = part[1:-1]
                # Check if it's a simple control token (e.g. <MEM_START>, <TRACE_START>, <SEP>)
                if part in self.token_to_id:
                    ids.append(self.token_to_id[part])
                else:
                    # It's a value token like <0200:60> or <V0:00>
                    self._encode_value_token(inner, ids)
            else:
                # Fallback: try direct lookup
                if part in self.token_to_id:
                    ids.append(self.token_to_id[part])
        return ids

    def _encode_inner(self, inner: str, ids: list[int]):
        """Encode the inside of a bracketed token like 'PC:0200' or 'OP:LD_VX_NN'."""
        if ":" in inner:
            label, value = inner.split(":", 1)
            label_token = f"<{label}:>"
            if label_token in self.token_to_id:
                ids.append(self.token_to_id[label_token])
            elif f"<{label}>" in self.token_to_id:
                ids.append(self.token_to_id[f"<{label}>"])

            # Encode value: either hex bytes or opcode name
            if label == "OP":
                op_token = f"<{value}>"
                if op_token in self.token_to_id:
                    ids.append(self.token_to_id[op_token])
            elif label in ("W_MEM",):
                # W_MEM format: "00:01:02:03:04" — address:byte pairs
                ids.append(self.token_to_id["<W_MEM:>"])
                for byte_str in value.split(":"):
                    tok = f"<{byte_str}>"
                    if tok in self.token_to_id:
                        ids.append(self.token_to_id[tok])
            else:
                # Hex value like "0200" or "05"
                self._encode_hex(value, ids)

    def _encode_hex(self, hex_str: str, ids: list[int]):
        """Encode a hex string byte by byte (2 chars at a time)."""
        hex_str = hex_str.upper()
        # Pad to even length
        if len(hex_str) % 2 != 0:
            hex_str = "0" + hex_str
        for i in range(0, len(hex_str), 2):
            byte = hex_str[i:i+2]
            tok = f"<{byte}>"
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])

    def _encode_value_token(self, inner: str, ids: list[int]):
        """Encode tokens like '0200:60' or 'V0:00'."""
        if ":" in inner:
            label, value = inner.split(":", 1)
            # Register value: <V0:00>
            reg_token = f"<{label}>"
            if reg_token in self.token_to_id:
                ids.append(self.token_to_id[reg_token])
                self._encode_hex(value, ids)
            else:
                # Memory value: <0200:60>
                self._encode_hex(label, ids)
                self._encode_hex(value, ids)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to string."""
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids)

    def decode_line(self, ids: list[int]) -> str:
        """Decode with reconstruction of bracket notation."""
        result = []
        i = 0
        while i < len(ids):
            tok = self.id_to_token.get(ids[i], "<UNK>")
            if tok == "[":
                # Collect until "]"
                inner_parts = []
                i += 1
                while i < len(ids) and self.id_to_token.get(ids[i]) != "]":
                    inner_parts.append(self.id_to_token.get(ids[i], "<UNK>"))
                    i += 1
                result.append("[" + "".join(inner_parts).replace("<", "").replace(">", "").replace(":", ":", 1) + "]")
                if i < len(ids):
                    i += 1  # skip "]"
            else:
                result.append(tok)
                i += 1
        return " ".join(result)

    def encode_trace(self, trace_lines: list[str]) -> list[int]:
        """Encode a full list of trace lines into one flat token ID list."""
        all_ids = []
        for line in trace_lines:
            ids = self.encode_line(line)
            all_ids.extend(ids)
        return all_ids
