"""Command-line batch extraction to portable NumPy and CSV files."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .core import process_chip


def _angle(header, names):
    for name in names:
        try:
            return float(header[name])
        except (KeyError, ValueError):
            pass
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract MSTAR HRRP from Phoenix SAR chips")
    parser.add_argument("input", type=Path, help="Chip file or directory containing class/chip files")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--mode", choices=("target-window", "pad"), default="target-window")
    parser.add_argument("--length", type=int, default=None, help="Default: 128 for target-window, 1024 for pad")
    parser.add_argument("--skip-errors", action="store_true", help="Record unreadable chips and continue")
    args = parser.parse_args(argv)
    length = args.length or (128 if args.mode == "target-window" else 1024)
    if length < 2:
        parser.error("--length must be >= 2")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.input.is_file():
        paths = [args.input]
    else:
        paths = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix[1:].isdigit())
    if not paths:
        parser.error("No Phoenix chip files found")
    profiles, metadata, errors = [], [], []
    for path in paths:
        try:
            profile, header = process_chip(path, length, args.mode)
        except (OSError, ValueError, OverflowError) as exc:
            if not args.skip_errors:
                parser.error(f"{path}: {exc}")
            errors.append({"path": str(path), "error": str(exc)})
            continue
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        metadata.append({
            "index": len(profiles), "source": relative.as_posix(),
            "class_name": relative.parts[0] if len(relative.parts) > 1 else "",
            "azimuth_deg": _angle(header, ("TargetAz", "TargetAzimuth", "AzimuthAngle", "AspectAngle", "Azimuth")),
            "depression_deg": _angle(header, ("MeasuredDepression", "DesiredDepression", "Depression")),
        })
        profiles.append(profile)
    if not profiles:
        parser.error("No valid chips were extracted")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "hrrp.npy", np.stack(profiles))
    with (args.output / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    (args.output / "run.json").write_text(json.dumps({
        "mode": args.mode, "length": length, "count": len(profiles),
        "errors": errors, "normalization": "per-profile maximum, linear amplitude",
    }, indent=2), encoding="utf-8")
    print(f"Extracted {len(profiles)} profiles of length {length} to {args.output}")


if __name__ == "__main__":
    main()
