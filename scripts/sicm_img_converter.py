"""
sicm_img_converter.py

Converts raw SICM .img scan files into normalised 224x224 grayscale PNGs
ready for a cancer / non-cancer classifier.

File format (reverse-engineered from the raw bytes):
    - 830 byte header (scan metadata: size, mode, timestamp -- not parsed here)
    - 128 x 128 grid of little-endian int16 height values
    - sentinel value 32767 marks "no data" (pipette never registered the surface)

Only the base "t" scans are used (already smoothed by the instrument software).
SP1/SP2 variant scans and the raw "nt" scans are skipped -- see README notes.

Usage:
    python sicm_img_converter.py --input "2024_02_09" --output out/cancerous --label cancerous
    python sicm_img_converter.py --input "2024_02_22 2" --output out/noncancerous --label noncancerous
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

HEADER_SIZE = 830          # bytes of metadata before pixel data starts
GRID_SIZE = 128             # native resolution of every scan
SENTINEL = 32767            # "no data" marker in the raw int16 grid
OUTPUT_SIZE = 224           # target size for the classifier (ResNet/EfficientNet-friendly)
LOW_PCT, HIGH_PCT = 1, 99   # percentile clipping range for contrast normalisation


def load_img(path, header=HEADER_SIZE, size=GRID_SIZE):
    """Read one raw .img file and return its (size, size) height array (float64)."""
    with open(path, 'rb') as fh:
        data = fh.read()

    expected_pixel_bytes = size * size * 2  # int16 = 2 bytes per pixel
    actual_pixel_bytes = len(data) - header
    if actual_pixel_bytes != expected_pixel_bytes:
        raise ValueError(
            f"unexpected file size: got {len(data)} bytes, "
            f"expected header({header}) + pixels({expected_pixel_bytes})"
        )

    pixel_bytes = data[header:]
    # '<i2' = little-endian signed 16-bit integer, matching the instrument's byte order
    arr = np.frombuffer(pixel_bytes, dtype='<i2').reshape(size, size)
    return arr.astype(np.float64)


def render(arr, sentinel=SENTINEL, out_size=OUTPUT_SIZE):
    """
    Render a raw height array to a smooth 8-bit grayscale image, matching the
    look of the instrument's own SICM Image Viewer (linear black->white height
    map, smoothly upsampled -- not a harsh contrast-clipped/blocky render).

    - Sentinel ("no data") pixels are filled with the background floor value
      rather than forced to hard black, so there's no artificial sharp edge.
    - Upsampling happens in float (height) space using cubic interpolation
      BEFORE quantising to 8-bit -- this is what gives the soft continuous
      gradient look instead of a blocky one.
    - Contrast is a linear min/max stretch across the real data range (like
      the viewer's z [um] colour bar), not a percentile clip.

    Returns (image_uint8, sentinel_fraction) -- sentinel_fraction is useful
    for quality-control filtering later (a very high fraction usually means
    a failed/partial scan).
    """
    valid_mask = arr != sentinel
    valid = arr[valid_mask]

    if valid.size < 10:  # essentially no real data in this scan
        return np.zeros((out_size, out_size), dtype=np.uint8), 1.0

    # fill "no data" pixels with the background floor instead of a hard 0,
    # so cubic upsampling doesn't ring/blocky-artifact at the boundary
    floor = np.percentile(valid, 1)
    filled = np.where(valid_mask, arr, floor)

    # smooth upsample in float space to the final output resolution
    scale = out_size / arr.shape[0]
    upsampled = zoom(filled, scale, order=3)  # cubic spline

    # linear stretch across the full real data range
    lo, hi = valid.min(), valid.max()
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((upsampled - lo) / (hi - lo), 0, 1)

    sentinel_fraction = 1 - valid_mask.mean()
    return (norm * 255).astype(np.uint8), sentinel_fraction


def quality_flag(img_arr, row_std_thresh=2.0, row_mean_thresh=15, bad_frac_thresh=0.08):
    """
    Flag scans with corruption artifacts (tip crashes / stage collisions),
    which show up as a flat, non-black plateau spanning most of a row or column
    -- unlike a real cell blob, which has soft circular gradients, or normal
    background, which is flat but black.

    Returns True if the image looks corrupted and should be excluded/reviewed.
    """
    row_std = img_arr.std(axis=1)
    row_mean = img_arr.mean(axis=1)
    bad_rows = (row_std < row_std_thresh) & (row_mean > row_mean_thresh)

    col_std = img_arr.std(axis=0)
    col_mean = img_arr.mean(axis=0)
    bad_cols = (col_std < row_std_thresh) & (col_mean > row_mean_thresh)

    return bad_rows.mean() > bad_frac_thresh or bad_cols.mean() > bad_frac_thresh


def find_base_t_files(folder):
    """
    Return only the base 't' scan files for a session folder --
    e.g. 'SICM_090224_1425_004t.img', NOT '..._004_SP1t.img' or '..._004nt.img'.
    """
    all_imgs = glob.glob(os.path.join(folder, '**', '*.img'), recursive=True)
    base_t = [f for f in all_imgs if re.search(r'\d{3}t\.img$', os.path.basename(f))]
    return sorted(base_t)


def convert_folder(input_folder, output_folder, label, out_size=OUTPUT_SIZE):
    os.makedirs(output_folder, exist_ok=True)
    files = find_base_t_files(input_folder)
    print(f"Found {len(files)} base 't' scans in {input_folder}")

    manifest_rows = []
    skipped = 0
    for f in files:
        try:
            arr = load_img(f)
        except ValueError as e:
            print(f"  SKIP (unexpected file size): {os.path.basename(f)} -- {e}")
            skipped += 1
            continue

        img_arr, sentinel_frac = render(arr, out_size=out_size)
        flagged = quality_flag(img_arr)
        img = Image.fromarray(img_arr, mode='L')

        out_name = os.path.splitext(os.path.basename(f))[0] + '.png'
        out_path = os.path.join(output_folder, out_name)
        img.save(out_path)

        manifest_rows.append({
            'filename': out_name,
            'source_file': f,
            'label': label,
            'sentinel_fraction': round(sentinel_frac, 4),
            'qc_flag': flagged,
        })

    print(f"Converted {len(manifest_rows)} images, skipped {skipped}")
    return manifest_rows


def write_manifest(rows, manifest_path):
    if not rows:
        return
    with open(manifest_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest written to {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Folder containing raw .img files')
    parser.add_argument('--output', required=True, help='Folder to write PNGs to')
    parser.add_argument('--label', required=True, choices=['cancerous', 'noncancerous'])
    parser.add_argument('--manifest', default=None, help='Optional path for a CSV manifest')
    args = parser.parse_args()

    rows = convert_folder(args.input, args.output, args.label)
    if args.manifest:
        write_manifest(rows, args.manifest)
