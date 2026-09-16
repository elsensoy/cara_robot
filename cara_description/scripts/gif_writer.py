"""A minimal, dependency-free animated-GIF writer (stdlib only: zlib).

The project's .venv has no imageio/opencv/Pillow and the host has no ffmpeg
(checked: `pip list` and `which ffmpeg` both came back empty), so this exists
purely to let evaluate_policy.py save a rollout as a shareable clip without
adding a new dependency for it. It writes an uncompressed GIF89a: global
palette (up to 256 colors, built once from the first frame, then reused
via nearest-color lookup so files stay small) and uncompressed LZW frame
data (minimum-code-size LZW with no back-reference compression -- valid
per the GIF spec, just not maximally compact).
"""

from __future__ import annotations

import struct


def _quantize_palette(frames, max_colors=216):
    """Build one global palette from evenly spaced RGB channel levels
    (a 6x6x6 web-safe-style cube = 216 colors) -- avoids an expensive
    k-means/median-cut pass; adequate for a locomotion sanity clip."""
    levels = [0, 51, 102, 153, 204, 255]
    palette = [(r, g, b) for r in levels for g in levels for b in levels]
    return palette[:max_colors]


def _build_index_frame(frame, palette_arr):
    """Vectorized nearest-palette-color lookup (pure Python per-pixel would be
    far too slow: a 320x240 frame against a 216-color palette is ~16M scalar
    ops). `palette_arr` is an (n_colors, 3) numpy int array, built once and
    reused across every frame."""
    import numpy as np
    h, w = frame.shape[0], frame.shape[1]
    flat = frame.reshape(-1, 3).astype(np.int32)
    # (n_pixels, n_colors) squared-distance matrix -- fine at 216 colors.
    diff = flat[:, None, :] - palette_arr[None, :, :]
    dist = np.einsum("pcn,pcn->pc", diff, diff)
    idx = np.argmin(dist, axis=1).astype(np.uint8)
    return idx.tobytes()


def _lzw_encode(index_stream, min_code_size):
    """Degenerate, deliberately uncompressed LZW: every pixel is emitted as
    its own single-symbol code at a fixed width. This is NOT simply "no
    dictionary growth on the encoder side" -- a spec-compliant LZW decoder
    grows its OWN table by one entry per code decoded (after the first),
    regardless of what the encoder emits, so a naive fixed-width stream
    desyncs from the decoder the moment its table would need to widen past
    the current code size (a first attempt did exactly this and every
    decoder rejected frame 0 outright). The fix: periodically re-emit a
    CLEAR code, comfortably before the decoder's table (which starts at
    `clear_code + 2` entries and grows by one per code) would reach
    `2**code_size` entries -- a CLEAR resets the decoder's table back to its
    initial state, so the stream never needs an actual code-size increase."""
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    # Margin well under the true limit (2**code_size - clear_code - 2) so
    # off-by-one differences between decoders can't matter.
    reset_period = max(1, (1 << code_size) // 4)

    bits = []

    def emit(code, size):
        for i in range(size):
            bits.append((code >> i) & 1)

    emit(clear_code, code_size)
    since_clear = 0
    for byte in index_stream:
        if since_clear >= reset_period:
            emit(clear_code, code_size)
            since_clear = 0
        emit(byte, code_size)
        since_clear += 1
    emit(end_code, code_size)

    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        val = 0
        for b_i, b in enumerate(chunk):
            val |= (b << b_i)
        out.append(val)
    return bytes(out)


def write_gif(path, frames, fps=10):
    """frames: list of HxWx3 uint8 numpy arrays, all the same shape."""
    import numpy as np
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[0], frames[0].shape[1]
    palette = _quantize_palette(frames)
    palette_arr = np.array(palette, dtype=np.int32)
    n_colors = len(palette)
    # bpp = bits needed to INDEX n_colors (also the LZW min-code-size).
    # The GIF header's "Size of Global Color Table" field is a separate,
    # 3-bit quantity N where the table holds 2^(N+1) entries -- i.e. N = bpp-1,
    # NOT bpp itself. Conflating the two (using bpp directly as N, when
    # bpp=8 doesn't even fit in 3 bits) corrupted the header's packed byte
    # on the first attempt and produced an unreadable file.
    bpp = 1
    while (1 << bpp) < n_colors:
        bpp += 1
    n_field = bpp - 1
    min_code_size = max(2, bpp)
    delay_cs = max(1, round(100.0 / fps))

    with open(path, "wb") as f:
        f.write(b"GIF89a")
        f.write(struct.pack("<HH", w, h))
        gct_flag = 1
        packed = (gct_flag << 7) | (0b111 << 4) | (0 << 3) | n_field
        f.write(struct.pack("B", packed))
        f.write(struct.pack("BB", 0, 0))  # background color index, pixel aspect
        table_entries = 1 << bpp
        for i in range(table_entries):
            if i < n_colors:
                f.write(bytes(palette[i]))
            else:
                f.write(b"\x00\x00\x00")

        f.write(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")  # loop forever

        for frame in frames:
            f.write(b"\x21\xf9\x04")
            f.write(struct.pack("<BHB", 0, delay_cs, 255))
            f.write(b"\x00")
            f.write(b"\x2c")
            f.write(struct.pack("<HHHH", 0, 0, w, h))
            f.write(struct.pack("B", 0))

            idx_stream = _build_index_frame(frame, palette_arr)
            lzw_data = _lzw_encode(idx_stream, min_code_size)
            f.write(struct.pack("B", min_code_size))
            for i in range(0, len(lzw_data), 255):
                block = lzw_data[i:i + 255]
                f.write(struct.pack("B", len(block)))
                f.write(block)
            f.write(b"\x00")
        f.write(b"\x3b")
