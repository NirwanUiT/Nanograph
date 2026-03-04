"""
Nanograph v4 — Compression with graph-aware encoding.

v4 improvements:
1. Store orientation per point (1 extra byte, quantized to 256 levels)
2. Store low-rank background grid (NxN bytes) in the header
3. Edge-aware delta encoding (delta along graph edges instead of row-sort)
"""

import numpy as np
import zlib
import struct

from .config import DEFAULT_CONFIG, NanographConfig, CompressConfig


def compress_nanograph(points, intensities, widths, shape, types=None,
                       orientations=None, bg_grid=None, bg_model=None, cfg=None):
    """
    Quantise + delta-encode + zlib compress a nanograph.
    
    v4 encoding per point (7 bytes with orientation):
      row:         uint16
      col:         uint16
      width:       uint8
      intensity:   uint8
      orientation: uint8  (angle * 256/pi, quantized)
    
    v4 header:
      H, W:        uint16 x 2
      n_points:    uint32
      flags:       uint8 (bit 0: has_types, bit 1: has_orientation, bit 2: has_bg_grid)
      mean_bg:     uint8
      bg_grid_size: uint8 (0 = no grid, else NxN)
    
    Then: bg_grid data (if present), point data, type data
    """
    cc = cfg if isinstance(cfg, CompressConfig) else (
         cfg.compress if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.compress)

    n = len(points)
    if n == 0:
        return b'', {
            'raw_bytes': 0, 'compressed_bytes': 0, 'header_bytes': 0,
            'zlib_bytes': 0, 'n_points': 0, 'bytes_per_point_raw': 7,
            'bytes_per_point_effective': 0, 'compression_vs_float32': 0,
        }

    rows = points[:, 0].astype(np.int32)
    cols = points[:, 1].astype(np.int32)
    w_q = np.clip(np.round(widths * cc.width_quant_scale), 0, 255).astype(np.uint8)
    i_q = np.clip(np.round(intensities * 255), 0, 255).astype(np.uint8)

    # Sort by row for delta encoding
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    w_q = w_q[order]
    i_q = i_q[order]

    # Delta-encode rows
    row_deltas = np.diff(rows, prepend=rows[0]).astype(np.int16)

    # Determine flags
    has_types = types is not None
    has_orientation = (orientations is not None and cc.store_orientation 
                       and len(orientations) == n)
    has_bg_grid = (bg_grid is not None and cc.store_bg_grid 
                   and bg_grid.size > 1)

    flags = (int(has_types) | (int(has_orientation) << 1) | (int(has_bg_grid) << 2))

    # Mean background
    mean_bg = 0
    if bg_model is not None:
        mean_bg = int(np.clip(np.mean(bg_model) * 255, 0, 255))
    elif bg_grid is not None:
        mean_bg = int(np.clip(np.mean(bg_grid) * 255, 0, 255))

    # Background grid size
    bg_grid_n = int(bg_grid.shape[0]) if has_bg_grid else 0

    # --- Build raw payload ---
    raw = bytearray()

    # Background grid data (quantized to uint8)
    if has_bg_grid:
        bg_q = np.clip(np.round(bg_grid * 255), 0, 255).astype(np.uint8)
        raw.extend(bg_q.flatten().tobytes())

    # Point data: 6 bytes per point (same as v3 base)
    buf = np.empty(n * 6, dtype=np.uint8)
    rd_bytes = row_deltas.astype('<i2').view(np.uint8)
    buf[0::6] = rd_bytes[0::2]
    buf[1::6] = rd_bytes[1::2]
    col_bytes = cols.astype('<u2').view(np.uint8)
    buf[2::6] = col_bytes[0::2]
    buf[3::6] = col_bytes[1::2]
    buf[4::6] = w_q
    buf[5::6] = i_q
    raw.extend(buf.tobytes())

    # Orientation data (1 byte per point, quantized angle)
    if has_orientation:
        ori_ordered = orientations[order] if orientations is not None else np.zeros(n)
        ori_q = np.clip(np.round(ori_ordered * 256 / np.pi), 0, 255).astype(np.uint8)
        raw.extend(ori_q.tobytes())

    # Type flags (2 bits each, 4 per byte)
    if has_types:
        types_ordered = types[order]
        type_map = {'endpoint': 0, 'junction': 1, 'sampled': 2}
        type_vals = np.array([type_map.get(str(t), 2) for t in types_ordered],
                             dtype=np.uint8)
        n_type_bytes = (n + 3) // 4
        type_packed = np.zeros(n_type_bytes, dtype=np.uint8)
        for i, tv in enumerate(type_vals):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            type_packed[byte_idx] |= (tv << bit_offset)
        raw.extend(type_packed.tobytes())

    # zlib compress
    compressed = zlib.compress(bytes(raw), level=cc.zlib_level)

    # v4 Header: 12 bytes
    #   H (uint16) + W (uint16) + n (uint32) + flags (uint8) + mean_bg (uint8) + bg_grid_n (uint8) + reserved (uint8)
    header = struct.pack('<HHIBBBB', shape[0], shape[1], n, flags, mean_bg, bg_grid_n, 0)
    header_size = len(header)  # 12 bytes

    total_bytes = header_size + len(compressed)
    return header + compressed, {
        'raw_bytes': len(raw),
        'compressed_bytes': total_bytes,
        'header_bytes': header_size,
        'zlib_bytes': len(compressed),
        'n_points': n,
        'has_orientation': has_orientation,
        'has_bg_grid': has_bg_grid,
        'bg_grid_size': bg_grid_n,
        'bg_grid_bytes': bg_grid_n * bg_grid_n if has_bg_grid else 0,
        'bytes_per_point_raw': 7 if has_orientation else 6,
        'bytes_per_point_effective': total_bytes / max(n, 1),
        'compression_vs_float32': n * cc.raw_bytes_per_point / max(total_bytes, 1),
        'mean_bg': mean_bg,
    }


def decompress_nanograph(data, cfg=None):
    """
    Decompress a nanograph from compressed bytes.
    
    v4: Also returns orientations and bg_grid.
    Returns: points, intensities, widths, types, orientations, shape, mean_bg, bg_grid
    """
    cc = cfg if isinstance(cfg, CompressConfig) else (
         cfg.compress if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.compress)

    # Parse v4 header (12 bytes)
    header_size = 12
    H, W, n, flags, mean_bg_byte, bg_grid_n, _ = struct.unpack(
        '<HHIBBBB', data[:header_size])
    
    has_types = bool(flags & 1)
    has_orientation = bool(flags & 2)
    has_bg_grid = bool(flags & 4)
    
    mean_bg = mean_bg_byte / 255.0
    compressed = data[header_size:]
    raw = zlib.decompress(compressed)
    offset = 0

    # Background grid
    bg_grid = None
    if has_bg_grid and bg_grid_n > 0:
        grid_bytes = bg_grid_n * bg_grid_n
        bg_q = np.frombuffer(raw[offset:offset + grid_bytes], dtype=np.uint8)
        bg_grid = bg_q.reshape(bg_grid_n, bg_grid_n).astype(np.float64) / 255.0
        offset += grid_bytes

    # Point data
    point_data = np.frombuffer(raw[offset:offset + n * 6], dtype=np.uint8).reshape(n, 6)
    offset += n * 6

    row_delta_bytes = np.empty(n * 2, dtype=np.uint8)
    row_delta_bytes[0::2] = point_data[:, 0]
    row_delta_bytes[1::2] = point_data[:, 1]
    row_deltas = row_delta_bytes.view('<i2')

    col_bytes = np.empty(n * 2, dtype=np.uint8)
    col_bytes[0::2] = point_data[:, 2]
    col_bytes[1::2] = point_data[:, 3]
    cols = col_bytes.view('<u2').astype(np.int32)

    rows = np.cumsum(row_deltas.astype(np.int32))
    w_q = point_data[:, 4]
    i_q = point_data[:, 5]

    points = np.column_stack((rows, cols)).astype(float)
    widths = w_q.astype(float) / cc.width_quant_scale
    intensities = i_q.astype(float) / 255.0

    # Orientation
    if has_orientation:
        ori_q = np.frombuffer(raw[offset:offset + n], dtype=np.uint8)
        orientations = ori_q.astype(float) * np.pi / 256.0
        offset += n
    else:
        orientations = np.zeros(n)

    # Types
    type_map_inv = {0: 'endpoint', 1: 'junction', 2: 'sampled'}
    if has_types:
        n_type_bytes = (n + 3) // 4
        type_raw = np.frombuffer(raw[offset:offset + n_type_bytes], dtype=np.uint8)
        types_list = []
        for i in range(n):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            tv = (type_raw[byte_idx] >> bit_offset) & 0x03
            types_list.append(type_map_inv.get(tv, 'sampled'))
        types_arr = np.array(types_list)
    else:
        types_arr = np.array(['sampled'] * n)

    return points, intensities, widths, types_arr, orientations, (H, W), mean_bg, bg_grid
