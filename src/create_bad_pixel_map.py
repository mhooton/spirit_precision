#!/usr/bin/env python3
"""
Create simulated bad pixel maps for testing.

Usage:
    python create_bad_pixel_map.py reference_image.fits output_map.fits --pixels x1,y1 x2,y2 x3,y3
    python create_bad_pixel_map.py reference_image.fits output_map.fits --random 100
    python create_bad_pixel_map.py reference_image.fits output_map.fits --clusters 10 --cluster_size 5
"""

import sys
import argparse
import numpy as np
from astropy.io import fits


def create_bad_pixel_map_from_coords(reference_fits, bad_pixel_coords, output_path):
    """
    Create a bad pixel map with specified pixel coordinates marked as bad.

    Args:
        reference_fits: Path to reference FITS image (for dimensions)
        bad_pixel_coords: List of (x, y) tuples for bad pixels
        output_path: Path to output bad pixel map FITS file
    """
    # Read reference image to get dimensions
    with fits.open(reference_fits) as hdul:
        header = hdul[0].header
        width = header['NAXIS1']
        height = header['NAXIS2']

    print(f"Creating bad pixel map: {width} × {height} pixels")

    # Initialize with all good pixels (0)
    bad_pixel_map = np.zeros((height, width), dtype=np.uint8)

    # Mark specified pixels as bad (1)
    n_bad = 0
    for x, y in bad_pixel_coords:
        # Check bounds
        if 0 <= x < width and 0 <= y < height:
            bad_pixel_map[y, x] = 1
            n_bad += 1
        else:
            print(f"Warning: Pixel ({x}, {y}) outside image bounds, skipping")

    print(f"Marked {n_bad} bad pixels")

    # Create FITS file
    hdu = fits.PrimaryHDU(bad_pixel_map)
    hdu.header['BUNIT'] = 'boolean'
    hdu.header['COMMENT'] = 'Bad pixel map: 0=good, 1=bad'
    hdu.header['REFIMAGE'] = reference_fits
    hdu.writeto(output_path, overwrite=True)

    print(f"Bad pixel map saved to: {output_path}")

    # Print statistics
    bad_rate = n_bad / (width * height) * 100
    print(f"Bad pixel rate: {bad_rate:.3f}%")


def create_random_bad_pixels(reference_fits, n_bad_pixels, output_path, seed=None):
    """
    Create a bad pixel map with randomly placed bad pixels.

    Args:
        reference_fits: Path to reference FITS image
        n_bad_pixels: Number of bad pixels to create
        output_path: Path to output bad pixel map
        seed: Random seed for reproducibility
    """
    # Read reference image
    with fits.open(reference_fits) as hdul:
        header = hdul[0].header
        width = header['NAXIS1']
        height = header['NAXIS2']

    print(f"Creating bad pixel map: {width} × {height} pixels")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize with all good pixels
    bad_pixel_map = np.zeros((height, width), dtype=np.uint8)

    # Generate random pixel coordinates
    n_pixels = width * height
    if n_bad_pixels > n_pixels:
        print(f"Warning: Requested {n_bad_pixels} bad pixels but image only has {n_pixels} pixels")
        n_bad_pixels = n_pixels

    # Randomly select pixels to be bad
    bad_indices = np.random.choice(n_pixels, size=n_bad_pixels, replace=False)
    bad_y, bad_x = np.unravel_index(bad_indices, (height, width))
    bad_pixel_map[bad_y, bad_x] = 1

    print(f"Marked {n_bad_pixels} random bad pixels")

    # Create FITS file
    hdu = fits.PrimaryHDU(bad_pixel_map)
    hdu.header['BUNIT'] = 'boolean'
    hdu.header['COMMENT'] = 'Bad pixel map: 0=good, 1=bad'
    hdu.header['REFIMAGE'] = reference_fits
    hdu.header['BPTYPE'] = 'random'
    if seed is not None:
        hdu.header['SEED'] = seed
    hdu.writeto(output_path, overwrite=True)

    print(f"Bad pixel map saved to: {output_path}")

    # Print statistics
    bad_rate = n_bad_pixels / n_pixels * 100
    print(f"Bad pixel rate: {bad_rate:.3f}%")


def create_clustered_bad_pixels(reference_fits, n_clusters, cluster_size, output_path, seed=None):
    """
    Create a bad pixel map with clusters of bad pixels.

    Args:
        reference_fits: Path to reference FITS image
        n_clusters: Number of clusters
        cluster_size: Approximate number of pixels per cluster
        output_path: Path to output bad pixel map
        seed: Random seed for reproducibility
    """
    # Read reference image
    with fits.open(reference_fits) as hdul:
        header = hdul[0].header
        width = header['NAXIS1']
        height = header['NAXIS2']

    print(f"Creating bad pixel map: {width} × {height} pixels")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize with all good pixels
    bad_pixel_map = np.zeros((height, width), dtype=np.uint8)

    # Generate clusters
    total_bad = 0
    for i in range(n_clusters):
        # Random cluster center
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        # Generate cluster with Gaussian distribution around center
        std_dev = np.sqrt(cluster_size) / 2
        for j in range(cluster_size):
            offset_x = int(np.random.normal(0, std_dev))
            offset_y = int(np.random.normal(0, std_dev))

            x = center_x + offset_x
            y = center_y + offset_y

            # Check bounds
            if 0 <= x < width and 0 <= y < height:
                if bad_pixel_map[y, x] == 0:  # Not already marked
                    bad_pixel_map[y, x] = 1
                    total_bad += 1

    print(f"Created {n_clusters} clusters with ~{cluster_size} pixels each")
    print(f"Total bad pixels: {total_bad}")

    # Create FITS file
    hdu = fits.PrimaryHDU(bad_pixel_map)
    hdu.header['BUNIT'] = 'boolean'
    hdu.header['COMMENT'] = 'Bad pixel map: 0=good, 1=bad'
    hdu.header['REFIMAGE'] = reference_fits
    hdu.header['BPTYPE'] = 'clustered'
    hdu.header['NCLUSTR'] = n_clusters
    hdu.header['CLUSSIZE'] = cluster_size
    if seed is not None:
        hdu.header['SEED'] = seed
    hdu.writeto(output_path, overwrite=True)

    print(f"Bad pixel map saved to: {output_path}")

    # Print statistics
    bad_rate = total_bad / (width * height) * 100
    print(f"Bad pixel rate: {bad_rate:.3f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Create simulated bad pixel maps for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specific pixels
  python create_bad_pixel_map.py ref.fits badpix.fits --pixels 100,200 150,250 500,600

  # 100 random bad pixels
  python create_bad_pixel_map.py ref.fits badpix.fits --random 100

  # 10 clusters of ~5 pixels each
  python create_bad_pixel_map.py ref.fits badpix.fits --clusters 10 --cluster_size 5

  # With random seed for reproducibility
  python create_bad_pixel_map.py ref.fits badpix.fits --random 100 --seed 42
        """
    )

    parser.add_argument('reference_fits', help='Reference FITS image (for dimensions)')
    parser.add_argument('output_fits', help='Output bad pixel map FITS file')

    # Mutually exclusive group for different modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pixels', nargs='+',
                       help='Specific pixel coordinates as x,y pairs (e.g., 100,200 150,250)')
    group.add_argument('--random', type=int, metavar='N',
                       help='Create N randomly placed bad pixels')
    group.add_argument('--clusters', type=int, metavar='N',
                       help='Create N clusters of bad pixels (use with --cluster_size)')

    parser.add_argument('--cluster_size', type=int, default=5,
                        help='Approximate number of pixels per cluster (default: 5)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Execute based on mode
    if args.pixels:
        # Parse pixel coordinates
        coords = []
        for pixel_str in args.pixels:
            try:
                x, y = map(int, pixel_str.split(','))
                coords.append((x, y))
            except ValueError:
                print(f"Error: Invalid pixel format '{pixel_str}'. Expected format: x,y")
                sys.exit(1)

        create_bad_pixel_map_from_coords(args.reference_fits, coords, args.output_fits)

    elif args.random:
        create_random_bad_pixels(args.reference_fits, args.random, args.output_fits, args.seed)

    elif args.clusters:
        create_clustered_bad_pixels(args.reference_fits, args.clusters,
                                    args.cluster_size, args.output_fits, args.seed)


if __name__ == "__main__":
    main()