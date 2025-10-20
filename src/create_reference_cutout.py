#!/usr/bin/env python3
"""
Create a small cutout from a reference FITS image centered on the WCS reference pixel.

Usage:
    python create_reference_cutout.py input_reference.fits output_reference.fits --size 100
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def create_cutout_reference(input_fits, output_fits, cutout_size=100):
    """
    Create a cutout from a FITS image centered on CRPIX.

    Args:
        input_fits: Path to input FITS file
        output_fits: Path to output FITS file
        cutout_size: Size of cutout (pixels, for square cutout)
    """
    print(f"Reading {input_fits}...")

    with fits.open(input_fits) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data

    # Get original dimensions and reference pixel
    orig_width = header['NAXIS1']
    orig_height = header['NAXIS2']
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']

    print(f"Original image: {orig_width} × {orig_height} pixels")
    print(f"Reference pixel: ({crpix1:.2f}, {crpix2:.2f})")

    # Calculate cutout boundaries centered on CRPIX
    # Convert CRPIX from 1-indexed (FITS) to 0-indexed (Python)
    center_x = int(crpix1 - 1)
    center_y = int(crpix2 - 1)

    half_size = cutout_size // 2

    x_min = max(0, center_x - half_size)
    x_max = min(orig_width, center_x + half_size)
    y_min = max(0, center_y - half_size)
    y_max = min(orig_height, center_y + half_size)

    # Adjust if we hit edges
    actual_width = x_max - x_min
    actual_height = y_max - y_min

    print(f"Cutout region: X=[{x_min}:{x_max}], Y=[{y_min}:{y_max}]")
    print(f"Cutout size: {actual_width} × {actual_height} pixels")

    # Extract cutout
    if data is not None:
        cutout_data = data[y_min:y_max, x_min:x_max]
        print(f"Cutout data shape: {cutout_data.shape}")
    else:
        # No data in the image, just create empty array
        cutout_data = np.zeros((actual_height, actual_width))
        print("Warning: No data in input image, creating empty cutout")

    # Update header
    # Update NAXIS keywords
    header['NAXIS1'] = actual_width
    header['NAXIS2'] = actual_height

    # Update CRPIX to reflect new position in cutout
    # CRPIX values are 1-indexed
    new_crpix1 = crpix1 - x_min
    new_crpix2 = crpix2 - y_min

    header['CRPIX1'] = new_crpix1
    header['CRPIX2'] = new_crpix2

    # CRVAL (reference coordinate) stays the same
    # PC matrix stays the same (rotation doesn't change)
    # CDELT stays the same (pixel scale doesn't change)

    # Add history
    header['HISTORY'] = f'Cutout from {input_fits}'
    header['HISTORY'] = f'Original size: {orig_width}x{orig_height}'
    header['HISTORY'] = f'Cutout region: [{x_min}:{x_max}, {y_min}:{y_max}]'
    header['HISTORY'] = f'Original CRPIX: ({crpix1:.2f}, {crpix2:.2f})'

    print(f"\nNew reference pixel: ({new_crpix1:.2f}, {new_crpix2:.2f})")

    # Verify the reference pixel is within the cutout
    if not (0 < new_crpix1 <= actual_width and 0 < new_crpix2 <= actual_height):
        print("\nWARNING: Reference pixel is outside cutout boundaries!")
        print("This may happen if the original reference pixel was near the edge.")

    # Save cutout
    hdu = fits.PrimaryHDU(cutout_data, header=header)
    hdu.writeto(output_fits, overwrite=True)

    print(f"\nCutout saved to: {output_fits}")

    # Print WCS summary
    print("\n=== WCS Summary ===")
    print(f"CRPIX1 = {header['CRPIX1']:.6f}")
    print(f"CRPIX2 = {header['CRPIX2']:.6f}")
    print(f"CRVAL1 = {header['CRVAL1']:.6f} (RA)")
    print(f"CRVAL2 = {header['CRVAL2']:.6f} (Dec)")
    print(f"PC1_1  = {header['PC1_1']:.10e}")
    print(f"PC1_2  = {header['PC1_2']:.10e}")
    print(f"PC2_1  = {header['PC2_1']:.10e}")
    print(f"PC2_2  = {header['PC2_2']:.10e}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a small cutout reference image centered on WCS reference pixel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python create_reference_cutout.py large_reference.fits small_reference.fits --size 100

This creates a 100×100 pixel cutout centered on CRPIX from the input image.
All WCS parameters are updated appropriately.
        """
    )

    parser.add_argument('input_fits', help='Input reference FITS image')
    parser.add_argument('output_fits', help='Output cutout FITS image')
    parser.add_argument('--size', type=int, default=100,
                        help='Size of square cutout in pixels (default: 100)')

    args = parser.parse_args()

    create_cutout_reference(args.input_fits, args.output_fits, args.size)


if __name__ == "__main__":
    main()