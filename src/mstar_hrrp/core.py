"""Extraction matching the CW100Exp Classification 128-bin preprocessing."""

from pathlib import Path
import numpy as np


def read_phoenix_chip(path):
    """Return (complex image, header) from a magnitude/phase Phoenix chip.

    The payload is two row-major big-endian float32 planes: magnitude, then
    phase in radians. No pickle or executable metadata is loaded.
    """
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing EndofPhoenixHeader: {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            for separator in ("=", ":"):
                if separator in decoded:
                    key, value = decoded.split(separator, 1)
                    header[key.strip()] = value.strip()
                    break
            if "EndofPhoenixHeader" in decoded:
                header_end = stream.tell()
                break
    offset = int(header.get("PhoenixHeaderLength", header_end))
    if offset < header_end:
        raise ValueError("PhoenixHeaderLength ends inside the ASCII header")
    def dimension(*keys):
        for key in keys:
            if key in header:
                return int(header[key])
        raise ValueError(f"Missing dimension ({', '.join(keys)}) in {path}")
    rows = dimension("NumberOfRows", "NumRows", "NROWS")
    cols = dimension("NumberOfColumns", "NumColumns", "NCOLS")
    if rows <= 0 or cols <= 0:
        raise ValueError("Chip dimensions must be positive")
    count = rows * cols
    if path.stat().st_size != offset + 8 * count:
        raise ValueError(f"Unexpected payload length in {path}")
    with path.open("rb") as stream:
        stream.seek(offset)
        planes = np.fromfile(stream, dtype=">f4", count=2 * count)
    magnitude = planes[:count].reshape(rows, cols)
    phase = planes[count:].reshape(rows, cols)
    if not np.isfinite(magnitude).all() or not np.isfinite(phase).all():
        raise ValueError(f"Non-finite magnitude or phase in {path}")
    return magnitude * np.exp(1j * phase), header


def _smooth(profile, size):
    # scipy.ndimage.uniform_filter1d(..., mode="nearest") equivalent.
    left = size // 2
    padded = np.pad(profile, (left, size - left - 1), mode="edge")
    return np.convolve(padded, np.ones(size) / size, mode="valid")


def extract_profile(image, target_len=128, mode="target-window"):
    """IFFT cross-range, noncoherent mean amplitude, then prepare a HRRP.

    ``target-window`` reproduces the compact Classification profile. ``pad``
    provides the 1024-bin centered-padding variant; both use max normalization.
    """
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) == 0 or target_len < 2:
        raise ValueError("Expected a nonempty 2-D image and target_len >= 2")
    if mode not in ("target-window", "pad"):
        raise ValueError("mode must be target-window or pad")
    profile = np.abs(np.fft.ifft(image, axis=1)).mean(axis=1).astype(np.float32)
    if mode == "target-window":
        size = max(3, min(15, (len(profile) // 12) * 2 + 1))
        smooth = _smooth(profile, size)
        if smooth.max() > 1e-8:
            indices = np.flatnonzero(smooth >= 0.08 * smooth.max())
            start, end = int(indices[0]) - 24, int(indices[-1]) + 25
            if end - start < 96:
                center = round((start + end - 1) / 2)
                start, end = center - 48, center - 48 + 96
            profile = profile[max(0, start):min(len(profile), end)]
        profile = np.interp(np.linspace(0, 1, target_len),
                            np.linspace(0, 1, len(profile)), profile).astype(np.float32)
        if len(profile) > 10:
            for edge, reference in ((slice(None, 4), profile[4:20]),
                                    (slice(-4, None), profile[-20:-4])):
                profile[edge] = np.minimum(profile[edge], np.percentile(reference, 90))
    elif len(profile) < target_len:
        left = (target_len - len(profile)) // 2
        profile = np.pad(profile, (left, target_len - len(profile) - left))
    else:
        start = (len(profile) - target_len) // 2
        profile = profile[start:start + target_len]
    peak = float(profile.max())
    return (profile / peak if peak > 1e-8 else profile).astype(np.float32)


def process_chip(path, target_len=128, mode="target-window"):
    image, header = read_phoenix_chip(path)
    return extract_profile(image, target_len, mode), header
