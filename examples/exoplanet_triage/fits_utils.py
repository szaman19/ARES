"""Lightweight, zero-dependency NASA FITS reader for TESS `_lc.fits` and `_tp.fits`.

Parses standard 2880-byte-aligned FITS Primary, IMAGE, and BINTABLE HDUs directly
into NumPy arrays and Python metadata dicts using pure `numpy` + standard library.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

_FITS_BLOCK = 2880

_TFORM_RE = re.compile(r"^\s*(\d*)([LXBIJKAEDCM])\s*$")
_TDIM_RE = re.compile(r"\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)")

_BINTABLE_DTYPE_MAP: dict[str, tuple[str, int]] = {
    "L": ("|u1", 1),
    "B": ("|u1", 1),
    "I": (">i2", 2),
    "J": (">i4", 4),
    "K": (">i8", 8),
    "A": ("|S1", 1),
    "E": (">f4", 4),
    "D": (">f8", 8),
}

_BITPIX_DTYPE_MAP: dict[int, tuple[str, int]] = {
    8: ("|u1", 1),
    16: (">i2", 2),
    32: (">i4", 4),
    64: (">i8", 8),
    -32: (">f4", 4),
    -64: (">f8", 8),
}


def _parse_header_value(raw_val: str) -> Any:
    """Parse a single FITS 80-char card value (stripping inline `/` comments)."""
    s = raw_val.strip()
    if not s:
        return None
    if s.startswith("'"):
        # FITS single-quoted string; escaped quotes are ''
        idx = 1
        chars: list[str] = []
        while idx < len(s):
            if s[idx] == "'":
                if idx + 1 < len(s) and s[idx + 1] == "'":
                    chars.append("'")
                    idx += 2
                    continue
                break
            chars.append(s[idx])
            idx += 1
        return "".join(chars).strip()

    # Non-string value: strip comment after `/`
    val_part = s.split("/", 1)[0].strip()
    if val_part == "T":
        return True
    if val_part == "F":
        return False
    try:
        return int(val_part)
    except ValueError:
        pass
    try:
        return float(val_part.replace("D", "E").replace("d", "e"))
    except ValueError:
        return val_part


def _read_header(buf: memoryview, offset: int) -> tuple[dict[str, Any], int]:
    """Read one FITS header starting at `offset`; return `(header_dict, next_offset)`."""
    header: dict[str, Any] = {}
    pos = offset
    n_bytes = len(buf)
    found_end = False

    while pos + _FITS_BLOCK <= n_bytes and not found_end:
        block = bytes(buf[pos : pos + _FITS_BLOCK])
        pos += _FITS_BLOCK
        for i in range(0, _FITS_BLOCK, 80):
            card = block[i : i + 80].decode("ascii", errors="replace")
            key = card[:8].strip()
            if key == "END":
                found_end = True
                break
            if not key or key in ("COMMENT", "HISTORY", "CONTINUE"):
                continue
            if card[8:10] == "= ":
                header[key] = _parse_header_value(card[10:])

    return header, pos


def _align_2880(n: int) -> int:
    rem = n % _FITS_BLOCK
    return n if rem == 0 else n + (_FITS_BLOCK - rem)


def read_fits_hdus(path_or_bytes: str | Path | bytes) -> list[dict[str, Any]]:
    """Parse all HDUs from a FITS file or raw bytes buffer.

    Returns a list of HDU dicts with keys:
      - `extname`: str (`PRIMARY` for HDU 0, or `EXTNAME` header string)
      - `xtension`: str (`PRIMARY`, `IMAGE`, `BINTABLE`, ...)
      - `header`: dict[str, Any]
      - `data`: `np.ndarray`, `dict[str, np.ndarray]`, or `None`
    """
    if isinstance(path_or_bytes, (str, Path)):
        raw = Path(path_or_bytes).read_bytes()
    else:
        raw = path_or_bytes

    mv = memoryview(raw)
    pos = 0
    hdus: list[dict[str, Any]] = []
    hdu_idx = 0

    while pos + _FITS_BLOCK <= len(mv):
        header, pos = _read_header(mv, pos)
        if not header and hdu_idx > 0:
            break

        xtension = "PRIMARY" if hdu_idx == 0 else str(header.get("XTENSION", "")).upper()
        extname = "PRIMARY" if hdu_idx == 0 else str(header.get("EXTNAME", xtension)).upper()
        naxis = int(header.get("NAXIS", 0) or 0)
        pcount = int(header.get("PCOUNT", 0) or 0)
        gcount = int(header.get("GCOUNT", 1) or 1)

        data_bytes = 0
        parsed_data: Any = None

        if xtension in ("PRIMARY", "IMAGE") and naxis > 0:
            bitpix = int(header.get("BITPIX", 8))
            axes = [int(header.get(f"NAXIS{k}", 0)) for k in range(1, naxis + 1)]
            n_elem = 1
            for ax in axes:
                n_elem *= ax
            dtype_str, elem_bytes = _BITPIX_DTYPE_MAP[bitpix]
            data_bytes = gcount * (n_elem * elem_bytes + pcount)
            if n_elem > 0 and pos + n_elem * elem_bytes <= len(mv):
                # FITS NAXIS1 is fastest-varying (columns), NAXIS2 is rows -> reverse for C order
                arr_be = np.frombuffer(
                    mv[pos : pos + n_elem * elem_bytes], dtype=np.dtype(dtype_str)
                )
                c_shape = tuple(reversed(axes))
                arr = np.array(arr_be.reshape(c_shape), copy=True)
                if arr.dtype.byteorder == ">":
                    arr = arr.astype(arr.dtype.newbyteorder("="))
                bzero = header.get("BZERO")
                bscale = header.get("BSCALE")
                if bscale not in (None, 1, 1.0) or bzero not in (None, 0, 0.0):
                    arr = arr.astype(np.float64) * float(bscale or 1.0) + float(bzero or 0.0)
                parsed_data = arr

        elif xtension == "BINTABLE" and naxis >= 2:
            row_bytes = int(header.get("NAXIS1", 0))
            n_rows = int(header.get("NAXIS2", 0))
            tfields = int(header.get("TFIELDS", 0))
            table_bytes = row_bytes * n_rows
            data_bytes = gcount * (table_bytes + pcount)

            if n_rows > 0 and row_bytes > 0 and pos + table_bytes <= len(mv):
                dtype_fields: list[Any] = []
                col_shapes_c: dict[str, tuple[int, ...]] = {}
                byte_cursor = 0
                valid_layout = True

                for f_idx in range(1, tfields + 1):
                    ttype = str(header.get(f"TTYPE{f_idx}", f"COL{f_idx}")).strip()
                    tform = str(header.get(f"TFORM{f_idx}", "")).strip()
                    tdim = header.get(f"TDIM{f_idx}")
                    m = _TFORM_RE.match(tform)
                    if not m:
                        valid_layout = False
                        break
                    rep_str, code = m.group(1), m.group(2)
                    repeat = int(rep_str) if rep_str else 1

                    if code == "X":
                        # Packed bits
                        n_bit_bytes = (repeat + 7) // 8
                        dtype_fields.append((ttype, "|u1", (n_bit_bytes,)))
                        byte_cursor += n_bit_bytes
                        continue
                    if code not in _BINTABLE_DTYPE_MAP:
                        valid_layout = False
                        break

                    dt_str, unit_b = _BINTABLE_DTYPE_MAP[code]
                    if code == "A":
                        dtype_fields.append((ttype, f"|S{repeat}"))
                    elif repeat == 1:
                        dtype_fields.append((ttype, dt_str))
                    else:
                        dtype_fields.append((ttype, dt_str, (repeat,)))

                    if tdim and isinstance(tdim, str):
                        dm = _TDIM_RE.search(tdim)
                        if dm:
                            dims_fortran = [int(x.strip()) for x in dm.group(1).split(",")]
                            prod = 1
                            for d in dims_fortran:
                                prod *= d
                            if prod == repeat and len(dims_fortran) >= 2:
                                col_shapes_c[ttype] = tuple(reversed(dims_fortran))

                    byte_cursor += repeat * unit_b

                if valid_layout and byte_cursor == row_bytes:
                    rec_dtype = np.dtype(dtype_fields)
                    rec_arr = np.frombuffer(mv[pos : pos + table_bytes], dtype=rec_dtype)
                    cols: dict[str, np.ndarray] = {}
                    for name in rec_dtype.names or ():
                        col = np.array(rec_arr[name], copy=True)
                        if col.dtype.byteorder == ">":
                            col = col.astype(col.dtype.newbyteorder("="))
                        if name in col_shapes_c:
                            col = col.reshape((n_rows, *col_shapes_c[name]))
                        cols[name] = col
                    parsed_data = cols

        pos += _align_2880(data_bytes)
        hdus.append(
            {
                "extname": extname,
                "xtension": xtension,
                "header": header,
                "data": parsed_data,
            }
        )
        hdu_idx += 1

    return hdus
