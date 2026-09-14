# MSTAR HRRP

Reproducible extraction of **high-resolution range profiles (HRRP)** from **MSTAR** complex SAR image chips. A small Python package and command-line tool for radar target recognition research. Search terms: MSTAR HRRP, MSTAR SAR-to-HRRP, high-resolution range profile extraction, Phoenix chip preprocessing.

This implementation follows the preprocessing documented in `CW100Exp/Classification/pretraitement_mstar_hrrp.tex` and implemented in `CW100Exp/prep_MSTAR/extract_paper_hrrp.py` (local research workspace). The default produces one **128-bin linear-amplitude HRRP** per chip. It does not include MSTAR images or trained models.

## Install and run

Requires Python 3.10+ and NumPy. From a checkout:

```bash
python -m pip install -e .
mstar-hrrp /path/to/MSTAR/raw_toload outputs/mstar
```

The input may be one Phoenix chip or a directory with class subdirectories such as `2S1/HB19487.000`. The extractor recursively finds files with numeric suffixes (`.000`, `.001`, etc.). It stops on a malformed chip by default; `--skip-errors` records failures in `run.json` and continues.

```bash
# Alternative centered zero-padding representation, 1024 bins
mstar-hrrp /path/to/MSTAR/raw_toload outputs/mstar_1024 --mode pad

# Single chip, custom length
mstar-hrrp /path/to/HB19487.000 outputs/example --length 128
```

Outputs are `hrrp.npy` (float32 matrix, rows aligned with `metadata.csv`), `metadata.csv` (relative source path, class directory, azimuth and depression angle if present), and `run.json` (parameters and error log). Load with `np.load("outputs/mstar/hrrp.npy", allow_pickle=False)`.

## Method

Each Phoenix chip contains a big-endian float32 magnitude plane followed by a phase plane in radians. We reconstruct the focused image as `magnitude * exp(1j * phase)`. For each range row, an inverse FFT runs along cross-range; the mean **magnitude** of those complex returns gives the 1-D profile. The default smooths that profile to find samples above 8% of its peak, adds 24 bins of context on each side, enforces a 96-bin minimum window, linearly interpolates to 128 bins, caps four edge bins against the 90th percentile of 16 adjacent interior bins, and divides by the profile maximum. A zero profile remains zero. The output is in `[0, 1]` and is **not** a calibrated physical range axis. The `pad` mode center-pads or center-crops to 1024 bins before max normalization.

The original research script also offers local-mean normalization and Gamma augmentation. They are not used in the Classification 128-bin preprocessing and are not part of this package's default. The classifier workspace subsequently applies a *dataset-wide* min/max scaling to `[-1, 1]`; that is a training transform, not part of the exported HRRP. Compute its parameters on the training split only to avoid leakage.

The IFFT/mean-amplitude projection is an explicit processing choice inspired by the research workspace; it does **not** reconstruct raw radar returns from the focused image or guarantee equivalence to measured HRRPs. Cite the preprocessing choice and report all settings when publishing results.

## Reproducibility and data

MSTAR data are **not redistributed** here. To obtain them directly from AFRL:

1. Open the [official MSTAR overview](https://www.sdms.afrl.af.mil/index.php?collection=mstar), then the [MSTAR Public Targets page](https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets). Follow its **DOWNLOAD** link to [Public Data Downloads](https://www.sdms.afrl.af.mil/index.php?collection=registration).
2. Create a **Public SDMS Account** on that page, or sign in if you already have one. In the download catalog, select the MSTAR public target data product and follow the site's current download instructions. The site can change its package names and sign-in flow; use the official catalog rather than an unofficial mirror.
3. Unpack the downloaded archive **locally, outside this Git repository**. Locate the Phoenix complex chip files with numeric extensions such as `.000` or `.025`, preserving their class directories. Point `mstar-hrrp` at the directory containing those class folders, for example `mstar-hrrp /path/to/raw_toload outputs/mstar`. If the archive has extra parent directories, use the innermost directory where each class is a direct child so `class_name` in `metadata.csv` is meaningful.

This package needs the complex Phoenix chips, not JPEG previews or magnitude-only images: phase is required for the reconstruction. AFRL describes the [public target product](https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets) as available for immediate download, but the public site requires an account for its download portal. If access fails, use the contact address on the [official portal](https://www.sdms.afrl.af.mil/index.php?collection=registration). Read the terms attached to the package you obtain. **Publicly downloadable does not, by itself, establish permission to rehost the data or derived profiles on GitHub.** The MIT license in this repository covers only its code and documentation. The tool does not download data automatically. Output paths are relative to the input root, so moved datasets can be processed without embedding machine-specific absolute paths.

To check the software without MSTAR data, run `python -m unittest discover -s tests -v`. The tests generate synthetic Phoenix chips and verify reading, extraction, and CLI output. Real-data validation still requires access to MSTAR; no benchmark accuracy is claimed.

## Contributing

Issues and pull requests are welcome. Please include a small synthetic regression case for format or processing changes, describe any altered preprocessing parameters, and do not commit MSTAR chips or derived datasets without redistribution rights. Code is MIT licensed; data are governed separately by their provider.
