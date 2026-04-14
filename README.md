# Fibers

This repository contains a small Python script for generating synthetic fiber images. Each image is built by placing non-overlapping white ellipses on a black background until a target area fraction is reached. The script also computes the orientation tensor components of the generated ellipse set and stores them in the output filename.

## What the script does

`try.py`:

- Samples ellipse sizes and orientations within user-defined ranges.
- Places ellipses while avoiding overlaps.
- Stops when the requested in-image area fraction is reached or when the maximum number of placement attempts is exhausted.
- Computes the orientation tensor components `a_xx`, `a_yy`, and `a_zz`.
- Saves each generated image as a `.jpg` file.

## Requirements

- Python 3.9 or newer
- `numpy`
- `matplotlib`
- `shapely`
- `tqdm`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

or, if you prefer to install them manually:

```bash
pip install numpy matplotlib shapely tqdm
```

## Usage

Generate one image with the default settings:

```bash
python try.py
```

Generate ten images and save them to a custom output directory:

```bash
python try.py --num-images 10 --output-dir images
```

Generate images with a different ellipse orientation range:

```bash
python try.py --phi-min 15 --phi-max 75
```

## Parameters users are most likely to change

These command-line arguments are the main inputs a user is expected to tune:

| Argument | Default | Description |
| --- | --- | --- |
| `--num-images` | `1` | Number of images to generate in one run. |
| `--a` | `20` | Nominal minor-axis length. The script samples the actual value within `a +/- 0.5`. |
| `--b-ini` | `20.5` | Minimum major-axis length sampled for each ellipse. |
| `--b-end` | `200` | Maximum major-axis length sampled for each ellipse. |
| `--width` | `510` | Image width in arbitrary units. |
| `--height` | `310` | Image height in arbitrary units. |
| `--phi-min` | `0` | Minimum ellipse orientation angle in degrees. |
| `--phi-max` | `90` | Maximum ellipse orientation angle in degrees. |
| `--target-area-fraction` | `0.30` | Fraction of the image area that should be covered by ellipses. Valid range: `(0, 1]`. |
| `--max-attempts` | `20000` | Maximum number of ellipse placement attempts per image. |
| `--output-dir` | `images` | Directory where generated images are written. |
| `--show-image` | disabled | Displays each generated image interactively before closing it. |

## Output

- The script creates the output directory automatically if it does not exist.
- Each file is named as:

```text
a_xx,a_yy,a_zz.jpg
```

This makes it easy to associate each generated image with its orientation tensor values.

## Repository contents

- [try.py](./try.py): main script used to generate the synthetic images.
- [requirements.txt](./requirements.txt): Python dependencies needed to run the script.

## License

No license file has been added yet. If this repository is intended for public reuse, a `LICENSE` file should be included so users know the redistribution and reuse conditions.
