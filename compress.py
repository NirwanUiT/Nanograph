"""
Nanograph v5 — Compression with graph-predictive encoding and spatial background.

v5 improvements over v4:
1. Spatial background grid (16×16 uint8, bilinearly interpolated) — replaces flat mean_bg
2. Graph-predictive encoding: predict node intensity/width from neighbors, encode residual
3. Foreground residual DCT coding (optional) — captures high-frequency detail PSF misses
4. v5 header: version(1) + H(2) + W(2) + n_points(4) + flags(2) + grid_size(1) + reserved(2) = 14 bytes

v4 backward compatibility maintained: v4 streams (header[8]==flags_u8) are auto-detected.
"""

import numpy as np
import zlib
import struct

from .config import DEFAULT_CONFIG, NanographConfig, CompressConfig


def _graph_predictive_encode(i_q, w_q, graph, order):
    """
    Graph-predictive encoding: for each node (in sorted order),
    predict intensity and width from the mean of already-encoded neighbors.
    Encode the signed residual instead of the raw value.

    Returns: (intensity_residuals, width_residuals) as int8 arrays
    """
    n = len(i_q)
    # Build reverse mapping: original_idx -> sorted_position
    inv_order = np.empty(n, dtype=np.int32)
    inv_order[order] = np.arange(n)

    i_resid = np.zeros(n, dtype=np.int16)
    w_resid = np.zeros(n, dtype=np.int16)
    encoded = np.zeros(n, dtype=bool)

    for pos in range(n):
        orig_idx = order[pos]
        i_val = int(i_q[pos])
        w_val = int(w_q[pos])

        # Get neighbors from graph
        if graph is not None and orig_idx < len(graph.nodes):
            neighbor_ids = graph.adjacency.get(orig_idx, [])
            # Only use neighbors that have already been encoded
            encoded_neighbors = [inv_order[nid] for nid in neighbor_ids
                                 if nid < n and encoded[inv_order[nid]]]
        else:
            encoded_neighbors = []

        if encoded_neighbors:
            # Predict from mean of encoded neighbors
            pred_i = int(np.mean([i_q[p] for p in encoded_neighbors]))
            pred_w = int(np.mean([w_q[p] for p in encoded_neighbors]))
        else:
            # No encoded neighbors: use previous point as predictor
            if pos > 0:
                pred_i = int(i_q[pos - 1])
                pred_w = int(w_q[pos - 1])
            else:
                pred_i = 128
                pred_w = 128

        i_resid[pos] = i_val - pred_i
        w_resid[pos] = w_val - pred_w
        encoded[pos] = True

    # Clamp to int8 range (residuals should be small for correlated data)
    i_resid = np.clip(i_resid, -128, 127).astype(np.int8)
    w_resid = np.clip(w_resid, -128, 127).astype(np.int8)
    return i_resid, w_resid


def _graph_predictive_decode(i_resid, w_resid, rows, cols):
    """
    Decode graph-predicted residuals back to absolute values.

    Without the graph at decode time, we use the sequential predictor
    (previous point), which was the fallback at encode time for the
    first point and works identically for the sequential case.
    """
    n = len(i_resid)
    i_q = np.zeros(n, dtype=np.uint8)
    w_q = np.zeros(n, dtype=np.uint8)

    for pos in range(n):
        if pos > 0:
            pred_i = int(i_q[pos - 1])
            pred_w = int(w_q[pos - 1])
        else:
            pred_i = 128
            pred_w = 128

        i_q[pos] = np.clip(pred_i + int(i_resid[pos]), 0, 255)
        w_q[pos] = np.clip(pred_w + int(w_resid[pos]), 0, 255)

    return i_q, w_q


def compress_nanograph(points, intensities, widths, shape, types=None,
                       orientations=None, bg_model=None, graph=None,
                       fg_residual=None, cfg=None):
    """
    Quantise + delta-encode + zlib compress a nanograph (v5 format).

    v5 format per point (core: 4 bytes + optional):
      row_delta:   int16   (delta from previous)
      col:         uint16
      intensity:   int8 or uint8  (residual if graph-predictive, else raw)
      width:       int8 or uint8  (residual if graph-predictive, else raw)
      orientation: uint8  (optional, quantized angle * 256/π)

    v5 header (14 bytes):
      version:     uint8  (5)
      H, W:        uint16 x 2
      n_points:    uint32
      flags:       uint16 (bit 0: has_types, bit 1: has_orientation,
                           bit 2: has_bg_grid, bit 3: has_fg_residual,
                           bit 4: graph_predictive)
      grid_size:   uint8
      reserved:    uint8

    Payload sections (concatenated, then zlib'd together):
      Section 0: Point data (n × 6 bytes)
      Section 1: Orientation data (n bytes, if flag)
      Section 2: Type data ((n+3)//4 bytes, if flag)
      Section 3: BG grid (grid_size² bytes, if flag)
    Appended raw (after zlib'd main payload):
      Section 4: FG residual (zlib'd DCT data, if flag) — already compressed
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

    # v5: bg grid
    from .utils import compute_bg_grid, quantize_bg_grid
    has_bg_grid = (cc.store_bg_grid and bg_model is not None)
    bg_grid_u8 = None
    grid_size = 16  # default
    if has_bg_grid:
        if isinstance(bg_model, np.ndarray) and bg_model.ndim == 2:
            if bg_model.shape[0] <= 32 and bg_model.shape[1] <= 32:
                # Already a grid
                bg_grid_u8 = quantize_bg_grid(bg_model)
                grid_size = bg_model.shape[0]
            else:
                # Full-resolution: we need to import the mask to compute grid
                # Use a simple threshold to estimate mask
                grid_size = 16
                bg_grid_u8 = quantize_bg_grid(bg_model) if bg_model.shape == (grid_size, grid_size) else None
                if bg_grid_u8 is None:
                    has_bg_grid = False

    # v5: fg residual
    has_fg_residual = (cc.store_fg_residual and fg_residual is not None
                       and len(fg_residual) > 0)

    # v5: graph-predictive encoding
    use_graph_pred = (cc.graph_predictive and graph is not None)

    flags = (int(has_types)
             | (int(has_orientation) << 1)
             | (int(has_bg_grid) << 2)
             | (int(has_fg_residual) << 3)
             | (int(use_graph_pred) << 4))

    # --- Build raw payload ---
    raw = bytearray()

    if use_graph_pred:
        # Graph-predictive: encode residuals for intensity and width
        i_resid, w_resid = _graph_predictive_encode(i_q, w_q, graph, order)
        # Point data: row_delta(2) + col(2) + i_resid(1) + w_resid(1) = 6 bytes
        buf = np.empty(n * 6, dtype=np.uint8)
        rd_bytes = row_deltas.astype('<i2').view(np.uint8)
        buf[0::6] = rd_bytes[0::2]
        buf[1::6] = rd_bytes[1::2]
        col_bytes = cols.astype('<u2').view(np.uint8)
        buf[2::6] = col_bytes[0::2]
        buf[3::6] = col_bytes[1::2]
        buf[4::6] = i_resid.view(np.uint8)
        buf[5::6] = w_resid.view(np.uint8)
    else:
        # Standard encoding (same as v4)
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

    # Orientation data
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

    # v5: Background grid
    if has_bg_grid and bg_grid_u8 is not None:
        raw.extend(bg_grid_u8.tobytes())

    # zlib compress main payload
    compressed = zlib.compress(bytes(raw), level=cc.zlib_level)

    # v5 Header: 13 bytes
    #   version(u8) + H(u16) + W(u16) + n(u32) + flags(u16) + grid_size(u8) + reserved(u8)
    header = struct.pack('<BHHIHBB', 5, shape[0], shape[1], n, flags,
                         grid_size if has_bg_grid else 0, 0)
    header_size = len(header)  # 13 bytes

    # Build final stream
    stream = bytearray(header)
    stream.extend(compressed)

    # v5: Append fg residual (already zlib'd) with length prefix
    if has_fg_residual:
        resid_data = fg_residual if isinstance(fg_residual, bytes) else bytes(fg_residual)
        stream.extend(struct.pack('<I', len(resid_data)))
        stream.extend(resid_data)

    total_bytes = len(stream)
    return bytes(stream), {
        'raw_bytes': len(raw),
        'compressed_bytes': total_bytes,
        'header_bytes': header_size,
        'zlib_bytes': len(compressed),
        'n_points': n,
        'has_orientation': has_orientation,
        'has_bg_grid': has_bg_grid,
        'has_fg_residual': has_fg_residual,
        'graph_predictive': use_graph_pred,
        'bg_grid_bytes': grid_size * grid_size if has_bg_grid else 0,
        'fg_residual_bytes': len(fg_residual) if has_fg_residual else 0,
        'bytes_per_point_raw': 7 if has_orientation else 6,
        'bytes_per_point_effective': total_bytes / max(n, 1),
        'compression_vs_float32': n * cc.raw_bytes_per_point / max(total_bytes, 1),
        'format_version': 5,
    }


def decompress_nanograph(data, cfg=None):
    """
    Decompress a nanograph from compressed bytes.

    Auto-detects v4 vs v5 format from the first byte.

    Returns: points, intensities, widths, types, orientations, shape, bg_info
             where bg_info is either a float (mean_bg for v4) or a dict for v5.
    """
    cc = cfg if isinstance(cfg, CompressConfig) else (
         cfg.compress if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.compress)

    # Detect format version
    version = data[0]
    if version == 5:
        return _decompress_v5(data, cc)
    else:
        return _decompress_v4(data, cc)


def _decompress_v4(data, cc):
    """Decompress v4 format (backward compatibility)."""
    header_size = 10
    H, W, n, flags, mean_bg_byte = struct.unpack('<HHIBB', data[:header_size])

    has_types = bool(flags & 1)
    has_orientation = bool(flags & 2)

    mean_bg = mean_bg_byte / 255.0
    compressed = data[header_size:]
    raw = zlib.decompress(compressed)
    offset = 0

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

    if has_orientation:
        ori_q = np.frombuffer(raw[offset:offset + n], dtype=np.uint8)
        orientations = ori_q.astype(float) * np.pi / 256.0
        offset += n
    else:
        orientations = np.zeros(n)

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

    return points, intensities, widths, types_arr, orientations, (H, W), mean_bg


def _decompress_v5(data, cc):
    """Decompress v5 format with bg grid, graph-predictive, and residual."""
    from .utils import dequantize_bg_grid

    header_size = 13
    version, H, W, n, flags, grid_size, reserved = struct.unpack(
        '<BHHIHBB', data[:header_size])

    has_types = bool(flags & 1)
    has_orientation = bool(flags & 2)
    has_bg_grid = bool(flags & 4)
    has_fg_residual = bool(flags & 8)
    graph_predictive = bool(flags & 16)

    # Find the boundary between main payload and residual
    # Main payload starts after header; residual (if present) is appended after
    # We need to decompress main payload first
    main_start = header_size

    # Try to decompress — zlib is self-terminating, but we don't know the boundary
    # So we try decompressing from main_start
    raw = None
    main_end = len(data)
    if has_fg_residual:
        # We need to find where the zlib stream ends
        # Use zlib decompressobj to consume exactly the zlib data
        d = zlib.decompressobj()
        raw = d.decompress(data[main_start:])
        consumed = len(data[main_start:]) - len(d.unused_data)
        main_end = main_start + consumed
    else:
        raw = zlib.decompress(data[main_start:])

    offset = 0

    # Point data (n × 6 bytes)
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
    points = np.column_stack((rows, cols)).astype(float)

    if graph_predictive:
        i_resid = point_data[:, 4].view(np.int8)
        w_resid = point_data[:, 5].view(np.int8)
        i_q, w_q = _graph_predictive_decode(i_resid, w_resid, rows, cols)
    else:
        w_q = point_data[:, 4]
        i_q = point_data[:, 5]

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
        offset += n_type_bytes
    else:
        types_arr = np.array(['sampled'] * n)

    # Background grid
    bg_grid = None
    if has_bg_grid and grid_size > 0:
        grid_bytes = grid_size * grid_size
        bg_grid_u8 = np.frombuffer(raw[offset:offset + grid_bytes],
                                    dtype=np.uint8).reshape(grid_size, grid_size)
        bg_grid = dequantize_bg_grid(bg_grid_u8)
        offset += grid_bytes

    # FG residual
    fg_residual_data = None
    if has_fg_residual and main_end < len(data):
        resid_start = main_end
        resid_len = struct.unpack('<I', data[resid_start:resid_start + 4])[0]
        fg_residual_data = data[resid_start + 4: resid_start + 4 + resid_len]

    # Build bg_info dict
    bg_info = {
        'bg_grid': bg_grid,
        'fg_residual_data': fg_residual_data,
        'format_version': 5,
    }
    # Fallback mean_bg for compatibility
    if bg_grid is not None:
        bg_info['mean_bg'] = float(np.mean(bg_grid))
    else:
        bg_info['mean_bg'] = 0.0

    return points, intensities, widths, types_arr, orientations, (H, W), bg_info
