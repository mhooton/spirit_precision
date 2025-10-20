import sys
import json
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import joblib
import pandas as pd
import warnings
from pathlib import Path
import argparse
import time
import csv
warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

# Get the absolute path to the spirit_precision directory
# Assumes script is in spirit_precision/src/
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Define standard directories
CONFIG_DIR = PROJECT_ROOT / "configs"
BPM_DIR = PROJECT_ROOT / "BPMs"
REF_IMAGE_DIR = PROJECT_ROOT / "ref_images"
RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_PATH = CONFIG_DIR / "precision_prediction_model.joblib"

# Ensure directories exist
BPM_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

def load_config(config_path=None):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON config file (if None, uses default)

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        config_path = CONFIG_DIR / "config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert relative paths in config to absolute paths
    if 'reference_image' in config:
        ref_path = Path(config['reference_image'])
        if not ref_path.is_absolute():
            config['reference_image'] = str(REF_IMAGE_DIR / ref_path)

    if 'bad_pixel_map_path' in config.get('detector', {}):
        bpm_path = Path(config['detector']['bad_pixel_map_path'])
        if not bpm_path.is_absolute():
            config['detector']['bad_pixel_map_path'] = str(BPM_DIR / bpm_path)

    # If reference_image is provided, extract detector/WCS parameters from it
    if 'reference_image' in config:
        print(f"Loading detector parameters from: {config['reference_image']}")
        fits_params = load_detector_from_fits(config['reference_image'])

        # Merge FITS-derived parameters into config
        if 'detector' not in config:
            config['detector'] = {}

        config['detector'].update(fits_params['detector'])
        config['wcs'] = fits_params['wcs']
        config['field_of_view'] = fits_params['field_of_view']

    return config

def load_detector_from_fits(fits_path):
    """
    Extract detector and WCS parameters from a plate-solved FITS image.

    Args:
        fits_path: Path to plate-solved FITS file

    Returns:
        Dictionary containing detector and WCS parameters
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        header = hdul[0].header

    # Extract detector dimensions
    width_pixels = header['NAXIS1']
    height_pixels = header['NAXIS2']

    # Extract WCS parameters
    pc1_1 = header['PC1_1']
    pc1_2 = header['PC1_2']
    pc2_1 = header['PC2_1']
    pc2_2 = header['PC2_2']
    cdelt1 = header.get('CDELT1', 1.0)
    cdelt2 = header.get('CDELT2', 1.0)

    # Calculate pixel scale in arcsec/pixel
    # Use the PC matrix magnitude and CDELT
    pixel_scale_deg = np.sqrt(pc1_1 ** 2 + pc1_2 ** 2) * abs(cdelt1)
    pixel_scale_arcsec = pixel_scale_deg * 3600.0

    # Calculate field of view in arcminutes
    fov_width_arcsec = width_pixels * pixel_scale_arcsec
    fov_height_arcsec = height_pixels * pixel_scale_arcsec
    fov_width_arcmin = fov_width_arcsec / 60.0
    fov_height_arcmin = fov_height_arcsec / 60.0

    print(f"\n=== Detector Parameters from FITS ===")
    print(f"Dimensions: {width_pixels} × {height_pixels} pixels")
    print(f"Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
    print(f"Field of view: {fov_width_arcmin:.2f}' × {fov_height_arcmin:.2f}'")

    return {
        'detector': {
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'pixel_scale_arcsec': pixel_scale_arcsec
        },
        'wcs': {
            'pc1_1': pc1_1,
            'pc1_2': pc1_2,
            'pc2_1': pc2_1,
            'pc2_2': pc2_2
        },
        'field_of_view': {
            'width_arcmin': fov_width_arcmin,
            'height_arcmin': fov_height_arcmin
        }
    }

def propagate_position(ra, dec, pmra, pmdec, ref_epoch, target_epoch):
    """
    Propagate star position from reference epoch to target epoch using proper motion.

    Args:
        ra: Right ascension at reference epoch (degrees)
        dec: Declination at reference epoch (degrees)
        pmra: Proper motion in RA * cos(dec) (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)
        ref_epoch: Reference epoch (Julian year, e.g., 2015.5)
        target_epoch: Target observation epoch (Julian year)

    Returns:
        ra_new, dec_new: Propagated coordinates (degrees)
    """
    # Handle missing proper motions (treat as zero)
    if pmra is None or np.ma.is_masked(pmra) or not np.isfinite(pmra):
        pmra = 0.0
    if pmdec is None or np.ma.is_masked(pmdec) or not np.isfinite(pmdec):
        pmdec = 0.0

    # Time difference in years
    dt = target_epoch - ref_epoch

    # Convert proper motions from mas/yr to degrees/yr
    pmra_deg = pmra / (3600.0 * 1000.0)  # mas -> degrees
    pmdec_deg = pmdec / (3600.0 * 1000.0)

    # Propagate positions
    # Note: pmra already includes cos(dec) factor in Gaia
    ra_new = ra + pmra_deg * dt
    dec_new = dec + pmdec_deg * dt

    return ra_new, dec_new


def get_current_julian_year():
    """
    Get current date as Julian year.

    Returns:
        Current Julian year (e.g., 2025.789)
    """
    from datetime import datetime
    from astropy.time import Time

    now = datetime.now()
    t = Time(now)
    return t.jyear

def get_field_jmag(gaia_id, config, expansion_factor=1.0):
    """
    Query Gaia DR2 for the target star and retrieve J-band magnitudes
    for all stars in the surrounding field.

    Args:
        gaia_id: Gaia DR2 source_id of the target star
        config: Configuration dictionary
        expansion_factor: Factor to expand FOV (1.0 = config FOV, >1.0 = larger)

    Returns:
        Astropy Table containing source_id, ra, dec, J-band mag, and Teff
        for all stars in the field
    """
    # Set Gaia table to DR2
    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

    # Get target star coordinates
    job = Gaia.launch_job_async(f"SELECT ra, dec FROM gaiadr2.gaia_source WHERE source_id={gaia_id}")
    result = job.get_results()
    if len(result) == 0:
        raise ValueError(f"Gaia ID {gaia_id} not found")
    ra, dec = result['ra'][0], result['dec'][0]

    # Define search box dimensions (in degrees) from config
    width = config['field_of_view']['width_arcmin'] / 60 * expansion_factor
    height = config['field_of_view']['height_arcmin'] / 60 * expansion_factor

    # Query for all stars in field with J-band magnitudes from 2MASS
    adql = f"""
    SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch, tm.j_m, g.teff_val
    FROM gaiadr2.gaia_source AS g
    JOIN gaiadr2.tmass_best_neighbour AS xmatch
      ON g.source_id = xmatch.source_id
    JOIN gaiadr1.tmass_original_valid AS tm
      ON xmatch.tmass_oid = tm.tmass_oid
    WHERE 1=CONTAINS(
        POINT('ICRS', g.ra, g.dec),
        BOX('ICRS', {ra}, {dec}, {width}, {height})
    )
    ORDER BY tm.j_m ASC
    """

    job2 = Gaia.launch_job_async(adql)
    results = job2.get_results()

    # Propagate all positions to current epoch
    current_epoch = get_current_julian_year()

    ra_corrected = []
    dec_corrected = []

    for row in results:
        ra_new, dec_new = propagate_position(
            row['ra'],
            row['dec'],
            row['pmra'],
            row['pmdec'],
            row['ref_epoch'],
            current_epoch
        )
        ra_corrected.append(ra_new)
        dec_corrected.append(dec_new)

    # Update positions in the table
    results['ra'] = ra_corrected
    results['dec'] = dec_corrected

    return results


def calculate_expansion_factor(config):
    """
    Calculate FOV expansion factor needed to cover all detector positions.

    When optimizing, the target can be placed anywhere on the detector (within
    edge constraints). The worst case is target at one edge - we need to query
    far enough to include comparison stars that could appear at the opposite edge
    of the detector.

    Args:
        config: Configuration dictionary

    Returns:
        Expansion factor (>= 1.0)
    """
    width_pix = config['detector']['width_pixels']
    height_pix = config['detector']['height_pixels']
    pixel_scale = config['detector']['pixel_scale_arcsec']
    edge_padding = config['detector']['edge_padding_pixels']
    aperture_radius = config['aperture']['radius_pixels']

    # Maximum distance from target position to detector edge
    # (target can be placed at edge + padding + aperture constraints)
    max_distance_x = width_pix - (edge_padding + aperture_radius)
    max_distance_y = height_pix - (edge_padding + aperture_radius)

    # Query needs to extend this far in all directions from target
    # So total query size is 2× this distance
    query_width_arcsec = 2 * max_distance_x * pixel_scale
    query_height_arcsec = 2 * max_distance_y * pixel_scale

    # Calculate expansion relative to configured FOV
    width_expansion = query_width_arcsec / (config['field_of_view']['width_arcmin'] * 60)
    height_expansion = query_height_arcsec / (config['field_of_view']['height_arcmin'] * 60)

    # Use the larger expansion factor and add small safety margin
    expansion = max(width_expansion, height_expansion) * 1.01

    return max(1.0, expansion)


def convert_j_to_zyj(jmag, config):
    """
    Convert J-band magnitude to zYJ filter magnitude using linear transformation.

    Args:
        jmag: J-band magnitude(s)
        config: Configuration dictionary

    Returns:
        zYJ magnitude(s)
    """
    slope = config['j_to_zyj_conversion']['slope']
    intercept = config['j_to_zyj_conversion']['intercept']
    return (jmag - intercept) / slope


def combined_mag(mag_array):
    """
    Calculate combined magnitude from multiple stars by adding their fluxes.

    Args:
        mag_array: Array of magnitudes

    Returns:
        Combined magnitude (scalar)
    """
    # Convert magnitudes to fluxes and sum
    artificial_flux = np.sum(10 ** (-0.4 * mag_array))
    # Convert back to magnitude
    artificial_mag = - 2.5 * np.log10(artificial_flux)
    return artificial_mag


def effective_mg(mag_array, target_index):
    """
    Calculate the effective magnitude representing the precision limit
    of differential photometry.

    Parameters:
    -----------
    mag_array : array-like
        Array of magnitudes (target + comparison stars)
    target_index : int
        Index of the target star in mag_array

    Returns:
    --------
    float
        Effective magnitude representing the combined noise floor
    """
    import numpy as np

    # Extract target magnitude
    m_target = mag_array[target_index]

    # Get all comparison star magnitudes (excluding target)
    comp_mags = np.delete(mag_array, target_index)

    # Convert comparison magnitudes to flux and sum to get artificial star flux
    comp_fluxes = 10 ** (-0.4 * comp_mags)
    artificial_flux = np.sum(comp_fluxes)

    # Convert artificial flux back to magnitude
    m_artificial = -2.5 * np.log10(artificial_flux)

    # Calculate effective differential magnitude
    print(m_target)
    print(comp_mags)
    print(m_artificial)
    m_diff = 2.5 * np.log10(10 ** (0.4 * m_target) + 10 ** (0.4 * m_artificial))

    return m_diff


def prediction_from_fit(total_mag, config):
    """
    Predict photometric precision using quadratic fit coefficients.

    Args:
        total_mag: Combined magnitude of field
        config: Configuration dictionary

    Returns:
        Predicted precision (log10 scale)
    """
    coeffs = config['quadratic_fit_coefficients']
    return np.polyval(coeffs, total_mag)  # Precision in log10

def prediction_from_DT(features):
    """
    Predict photometric precision using trained Decision Tree model.

    Args:
        features: DataFrame with columns ['Comp stars', 'zYJ mag', 'Combined mag', 'Teff']

    Returns:
        Predicted precision (linear scale)
    """
    # OLD:
    # model_dict = joblib.load('precision_prediction_model.joblib')

    # NEW:
    model_dict = joblib.load(MODEL_PATH)

    model = model_dict['model']
    feature_array = features[['Comp stars', 'zYJ mag', 'Combined mag', 'Teff']].values
    precision = model.predict(feature_array)[0]
    return precision

def to_float(value):
    """
    Convert various astropy/numpy types to plain Python float.

    Args:
        value: Quantity, masked array, numpy scalar, or numeric value

    Returns:
        Plain Python float
    """
    # Handle Quantity objects
    if hasattr(value, 'value'):
        value = value.value
    # Handle arrays/masked arrays - extract scalar
    if hasattr(value, 'item'):
        value = value.item()
    # Handle remaining masked array cases
    if hasattr(value, 'data'):
        value = value.data
        if hasattr(value, 'item'):
            value = value.item()
    return float(value)


def create_run_directory(gaia_id):
    """
    Create output directory for this run.

    Args:
        gaia_id: Gaia DR2 source_id

    Returns:
        Absolute path to run directory
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = RUNS_DIR / f"{gaia_id}_{date_str}"

    # Create directory (overwrite if exists)
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def download_latest_bpm(config):
    """
    Download the most recent bad pixel map from the SPECULOOS pipeline server.

    Searches for BPMs starting with v2/yesterday, then v3/yesterday, then goes
    back one day at a time, alternating versions. Stops at year 2021.

    Args:
        config: Configuration dictionary

    Returns:
        Path to downloaded BPM file, or None if not found
    """
    import paramiko
    from datetime import datetime, timedelta
    import os

    server = "appcs.ra.phy.cam.ac.uk"
    username = "speculoos"
    base_path = "/appct/data/SPECULOOSPipeline/PipelineOutput"
    telescope = "Callisto"
    versions = ["v2", "v3"]

    # Start with yesterday
    current_date = datetime.now() - timedelta(days=1)
    cutoff_date = datetime(2021, 1, 1)

    print("\n=== Searching for latest Bad Pixel Map on server ===")
    print(f"Connecting to {server}...")

    # Establish single SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(server, username=username)
        sftp = ssh.open_sftp()
        print("Connected successfully")

        while current_date >= cutoff_date:
            date_str = current_date.strftime("%Y%m%d")

            # Try each version for this date
            for version in versions:
                remote_path = f"{base_path}/{version}/{telescope}/output/{date_str}/reduction/1_BadPixelMap.fits"

                try:
                    # Check if file exists using SFTP stat
                    sftp.stat(remote_path)

                    local_filename = BPM_DIR / f"1_BadPixelMap_{date_str}.fits"

                    print(f"Found BPM: {version}/{date_str}")
                    print(f"Downloading to {local_filename}...")

                    sftp.get(remote_path, str(local_filename))

                    print(f"Successfully downloaded BPM from {version}/{date_str}")
                    sftp.close()
                    ssh.close()
                    return str(local_filename)  # Return absolute path

                except FileNotFoundError:
                    # File doesn't exist, continue searching
                    pass
                except IOError:
                    # File doesn't exist (alternative exception type)
                    pass
                except Exception as e:
                    print(f"Error checking {version}/{date_str}: {e}")

            # Move to previous day
            current_date -= timedelta(days=1)

        print("No BPM found on server (searched back to 2021)")
        sftp.close()
        ssh.close()
        return None

    except Exception as e:
        print(f"Error connecting to server: {e}")
        try:
            ssh.close()
        except:
            pass
        return None


def load_bad_pixel_map(config):
    """
    Load bad pixel map from FITS file.
    Downloads from server if configured, otherwise uses local path.

    Args:
        config: Configuration dictionary

    Returns:
        2D boolean numpy array where True = bad pixel, False = good pixel
    """
    from astropy.io import fits

    # Check if we should download from server
    if config.get('download_BPM_from_server', False):
        bpm_path = download_latest_bpm(config)

        if bpm_path is None:
            # Fall back to config path
            print(f"Falling back to configured BPM path: {config['detector']['bad_pixel_map_path']}")
            bpm_path = config['detector']['bad_pixel_map_path']
    else:
        bpm_path = config['detector']['bad_pixel_map_path']

    with fits.open(bpm_path) as hdul:
        # FITS convention: 1 = bad, 0 = good
        bad_pixel_data = hdul[0].data

    # Convert to boolean: True = bad pixel
    return bad_pixel_data.astype(bool)


def create_wcs(target_ra, target_dec, target_x, target_y, config):
    """
    Create WCS object for coordinate transformations.

    Args:
        target_ra: Target RA in degrees
        target_dec: Target Dec in degrees
        target_x: Target X pixel position on detector
        target_y: Target Y pixel position on detector
        config: Configuration dictionary

    Returns:
        astropy.wcs.WCS object
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [target_x, target_y]
    wcs.wcs.crval = [target_ra, target_dec]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.pc = [[config['wcs']['pc1_1'], config['wcs']['pc1_2']],
                  [config['wcs']['pc2_1'], config['wcs']['pc2_2']]]
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


def aperture_contains_bad_pixels(x, y, radius, bad_pixel_map):
    """
    Check if a circular aperture contains any bad pixels.

    Args:
        x: Aperture center X coordinate (pixel)
        y: Aperture center Y coordinate (pixel)
        radius: Aperture radius (pixels)
        bad_pixel_map: 2D boolean array where True = bad pixel

    Returns:
        True if aperture contains any bad pixels, False otherwise
    """
    # Get bounding box around aperture
    x_min = max(0, int(np.floor(x - radius)))
    x_max = min(bad_pixel_map.shape[1], int(np.ceil(x + radius)) + 1)
    y_min = max(0, int(np.floor(y - radius)))
    y_max = min(bad_pixel_map.shape[0], int(np.ceil(y + radius)) + 1)

    # Create coordinate grids for the bounding box
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    # Calculate distances from aperture center
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    # Check if any bad pixels are within the aperture
    mask = distances <= radius
    if np.any(mask):
        return np.any(bad_pixel_map[y_min:y_max, x_min:x_max][mask])
    return False


def aperture_on_detector(x, y, radius, detector_width, detector_height, padding):
    """
    Check if aperture is fully on detector with padding.

    Args:
        x: Aperture center X coordinate (pixel)
        y: Aperture center Y coordinate (pixel)
        radius: Aperture radius (pixels)
        detector_width: Detector width (pixels)
        detector_height: Detector height (pixels)
        padding: Edge padding (pixels)

    Returns:
        True if aperture is fully on detector (with padding), False otherwise
    """
    min_coord = radius + padding
    max_x = detector_width - radius - padding
    max_y = detector_height - radius - padding

    return (x >= min_coord and x <= max_x and
            y >= min_coord and y <= max_y)


def distance_to_nearest_hazard(x, y, bad_pixel_map, det_width, det_height, edge_padding):
    """
    Calculate distance from position to nearest hazard (bad pixel or detector edge).

    This is used as a tiebreaker among positions with equal precision - we prefer
    positions that are farther from both bad pixels and edges, maximizing the
    "safe zone" for small drifts.

    Args:
        x: Position X coordinate (pixel)
        y: Position Y coordinate (pixel)
        bad_pixel_map: 2D boolean array where True = bad pixel
        det_width: Detector width (pixels)
        det_height: Detector height (pixels)
        edge_padding: Edge padding (pixels)

    Returns:
        Distance to nearest hazard (pixels)
    """
    # Calculate distance to edges (with padding)
    dist_to_left = x - edge_padding
    dist_to_right = (det_width - edge_padding) - x
    dist_to_bottom = y - edge_padding
    dist_to_top = (det_height - edge_padding) - y

    dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

    # Find all bad pixel locations
    bad_y, bad_x = np.where(bad_pixel_map)

    if len(bad_y) == 0:
        # No bad pixels on detector - only edge distance matters
        return dist_to_edge

    # Calculate distances from (x, y) to all bad pixels
    distances_to_bad = np.sqrt((bad_x - x) ** 2 + (bad_y - y) ** 2)
    dist_to_bad_pixel = np.min(distances_to_bad)

    # Return the minimum of the two distances (nearest hazard)
    return min(dist_to_edge, dist_to_bad_pixel)

def sky_to_pixel(ra, dec, wcs_obj):
    """
    Transform RA/Dec to detector X/Y coordinates.

    Args:
        ra: Right ascension (degrees), scalar or array
        dec: Declination (degrees), scalar or array
        wcs_obj: astropy.wcs.WCS object

    Returns:
        x, y: Pixel coordinates (0-indexed)
    """
    x, y = wcs_obj.world_to_pixel_values(ra, dec)
    return x, y


def pixel_to_sky(x, y, wcs_obj):
    """
    Transform detector X/Y to RA/Dec coordinates.

    Args:
        x: X pixel coordinate (0-indexed), scalar or array
        y: Y pixel coordinate (0-indexed), scalar or array
        wcs_obj: astropy.wcs.WCS object

    Returns:
        ra, dec: Sky coordinates (degrees)
    """
    ra, dec = wcs_obj.pixel_to_world_values(x, y)
    return ra, dec


def read_target_list(filename):
    """
    Read Gaia DR2 IDs from target list file.

    Expects a space/comma-separated file with Gaia IDs in the 3rd column.
    Skips lines starting with # or that don't have at least 3 columns.

    Args:
        filename: Name of file in target_lists/ directory

    Returns:
        List of Gaia DR2 IDs as integers
    """
    target_list_path = PROJECT_ROOT / "target_lists" / filename

    if not target_list_path.exists():
        raise FileNotFoundError(f"Target list not found: {target_list_path}")

    gaia_ids = []

    with open(target_list_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split by whitespace or comma
            parts = line.replace(',', ' ').split()

            if len(parts) >= 3:
                try:
                    gaia_id = int(parts[2])  # 3rd column (0-indexed: 2)
                    gaia_ids.append(gaia_id)
                except ValueError:
                    # Skip lines where 3rd column isn't a valid integer
                    continue

    print(f"Read {len(gaia_ids)} targets from {filename}")
    return gaia_ids


def create_batch_directory():
    """
    Create batch output directory.

    Returns:
        Absolute path to batch directory
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y%m%d")
    batch_dir = RUNS_DIR / f"batch_{date_str}"

    # Create directory
    batch_dir.mkdir(parents=True, exist_ok=True)

    return batch_dir


def initialize_batch_csv(batch_dir):
    """
    Create and initialize batch summary CSV file.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Path to CSV file
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = batch_dir / f"batch_summary_{date_str}.csv"

    # Define CSV headers
    headers = [
        'gaia_id',
        'status',
        'timestamp',
        'target_ra',
        'target_dec',
        'optimal_x',
        'optimal_y',
        'detector_center_ra',
        'detector_center_dec',
        'target_offset_x',
        'target_offset_y',
        'target_offset_ra_arcsec',
        'target_offset_dec_arcsec',
        'precision',
        'n_comparison_stars',
        'combined_mag',
        'target_jmag',
        'target_zyj',
        'target_teff',
        'distance_to_hazard',
        'reference_image',
        'bad_pixel_map',
        'processing_time_seconds',
        'error_message'
    ]

    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return csv_path


def append_to_batch_csv(csv_path, result):
    """
    Append a result row to the batch CSV.

    Args:
        csv_path: Path to CSV file
        result: Dictionary with result data
    """
    row = [
        result.get('gaia_id', ''),
        result.get('status', ''),
        result.get('timestamp', ''),
        result.get('target_ra', ''),
        result.get('target_dec', ''),
        result.get('optimal_x', ''),
        result.get('optimal_y', ''),
        result.get('detector_center_ra', ''),
        result.get('detector_center_dec', ''),
        result.get('target_offset_x', ''),
        result.get('target_offset_y', ''),
        result.get('target_offset_ra_arcsec', ''),
        result.get('target_offset_dec_arcsec', ''),
        result.get('precision', ''),
        result.get('n_comparison_stars', ''),
        result.get('combined_mag', ''),
        result.get('target_jmag', ''),
        result.get('target_zyj', ''),
        result.get('target_teff', ''),
        result.get('distance_to_hazard', ''),
        result.get('reference_image', ''),
        result.get('bad_pixel_map', ''),
        result.get('processing_time_seconds', ''),
        result.get('error_message', '')
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def optimize_target_position(gaia_id, config, save_precision_map=False):
    """
    Find optimal detector position for target star to maximize photometric precision.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        save_precision_map: If True, save 2D precision map as FITS file

    Returns:
        dict containing:
            - 'optimal_x': Best X pixel position
            - 'optimal_y': Best Y pixel position
            - 'precision': Predicted precision at optimal position
            - 'n_comparison_stars': Number of usable comparison stars
            - 'target_ra': Target RA (degrees)
            - 'target_dec': Target Dec (degrees)
            - 'precision_map': 2D array of precision values (if save_precision_map=True)
    """
    print("\n=== Starting Position Optimization ===")

    # Load bad pixel map
    print("Loading bad pixel map...")
    bad_pixel_map = load_bad_pixel_map(config)
    n_bad_pixels = np.sum(bad_pixel_map)
    print(f"Bad pixel map loaded: {n_bad_pixels} bad pixels ({n_bad_pixels / bad_pixel_map.size * 100:.2f}%)")

    # Query expanded field
    print("Querying Gaia for expanded field...")
    expansion = calculate_expansion_factor(config)
    print(f"Expansion factor: {expansion:.3f}")
    jmag_data = get_field_jmag(gaia_id, config, expansion_factor=expansion)
    print(f"Gaia query returned {len(jmag_data)} stars")

    # Get target coordinates
    target_row = jmag_data[jmag_data['source_id'] == int(gaia_id)]
    if len(target_row) == 0:
        raise ValueError(f"Target star {gaia_id} not found in query results")
    target_ra = float(target_row['ra'][0])
    target_dec = float(target_row['dec'][0])
    target_jmag = float(target_row['j_m'][0])
    target_teff = to_float(target_row['teff_val'][0])

    print(f"Target: RA={target_ra:.6f}, Dec={target_dec:.6f}, J={target_jmag:.3f}, Teff={target_teff:.0f}K")

    # Get detector parameters
    det_width = config['detector']['width_pixels']
    det_height = config['detector']['height_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    edge_padding = config['detector']['edge_padding_pixels']
    grid_spacing = config['optimization']['grid_spacing_pixels']

    print(f"Detector: {det_width}×{det_height} px, aperture={aperture_radius} px, padding={edge_padding} px")
    print(f"Grid spacing: {grid_spacing} px")

    # Filter comparison stars by magnitude (relative to target)
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']

    print(
        f"Magnitude limits: target J={target_jmag:.3f}, range=[{target_jmag + brighter_limit:.3f}, {target_jmag + fainter_limit:.3f}]")

    comp_stars = jmag_data[
        (jmag_data['source_id'] != int(gaia_id)) &
        (jmag_data['j_m'] < target_jmag + fainter_limit) &
        (jmag_data['j_m'] > target_jmag + brighter_limit)
        ]

    print(f"Potential comparison stars in magnitude range: {len(comp_stars)}")

    if len(comp_stars) == 0:
        print("ERROR: No comparison stars found in magnitude range!")
        print("Cannot create precision map without comparison stars.")
        raise ValueError("Cannot optimize without comparison stars")

    # Create grid of target positions
    x_positions = np.arange(aperture_radius + edge_padding,
                            det_width - aperture_radius - edge_padding,
                            grid_spacing)
    y_positions = np.arange(aperture_radius + edge_padding,
                            det_height - aperture_radius - edge_padding,
                            grid_spacing)

    total_positions = len(x_positions) * len(y_positions)
    print(f"Testing {total_positions} grid positions (spacing={grid_spacing} pixels)...")

    # Store results
    results = []

    # Convert target J-mag to zYJ once
    target_zyj = to_float(convert_j_to_zyj(target_jmag, config))

    # Counters for diagnostics
    n_tested = 0
    n_target_bad_pix = 0
    n_target_off_detector = 0
    n_no_valid_comps = 0

    # Evaluate each grid position
    for i, target_x in enumerate(x_positions):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(x_positions)} rows... (valid positions so far: {len(results)})")

        for target_y in y_positions:
            n_tested += 1

            # Check if target aperture is on detector (should always be true given how we generate grid, but check anyway)
            if not aperture_on_detector(target_x, target_y, aperture_radius, det_width, det_height, edge_padding):
                n_target_off_detector += 1
                continue

            # Check if target aperture is clean
            if aperture_contains_bad_pixels(target_x, target_y, aperture_radius, bad_pixel_map):
                n_target_bad_pix += 1
                continue

            # Create WCS for this target position
            wcs_obj = create_wcs(target_ra, target_dec, target_x, target_y, config)

            # Transform all comparison stars to pixel coordinates
            comp_x, comp_y = sky_to_pixel(comp_stars['ra'], comp_stars['dec'], wcs_obj)

            # Filter comparison stars: on detector and clean apertures
            valid_comps = []
            valid_comp_mags = []

            for j, (cx, cy, jmag) in enumerate(zip(comp_x, comp_y, comp_stars['j_m'])):
                # Check if on detector
                if not aperture_on_detector(cx, cy, aperture_radius, det_width, det_height, edge_padding):
                    continue
                # Check if aperture is clean
                if aperture_contains_bad_pixels(cx, cy, aperture_radius, bad_pixel_map):
                    continue
                valid_comps.append(j)
                valid_comp_mags.append(jmag)

            # Skip if no valid comparison stars
            if len(valid_comps) == 0:
                n_no_valid_comps += 1
                continue

            # Convert to zYJ magnitudes
            valid_comp_mags = np.array(valid_comp_mags)
            comp_zyj = convert_j_to_zyj(valid_comp_mags, config)

            # Calculate combined magnitude (without target)
            combined_mag_wo_target = combined_mag(comp_zyj)

            # Predict precision using Decision Tree
            features = pd.DataFrame({
                'Comp stars': [len(valid_comps)],
                'zYJ mag': [target_zyj],
                'Combined mag': [to_float(combined_mag_wo_target)],
                'Teff': [target_teff]
            })
            precision = to_float(prediction_from_DT(features))

            # Store result
            results.append({
                'x': target_x,
                'y': target_y,
                'precision': precision,
                'n_comp': len(valid_comps),
                'combined_mag': to_float(combined_mag_wo_target)
            })

    print(f"\n=== Optimization Statistics ===")
    print(f"Total positions tested: {n_tested}")
    print(f"Target aperture off detector: {n_target_off_detector} ({n_target_off_detector / n_tested * 100:.1f}%)")
    print(f"Target aperture contains bad pixels: {n_target_bad_pix} ({n_target_bad_pix / n_tested * 100:.1f}%)")
    print(f"No valid comparison stars: {n_no_valid_comps} ({n_no_valid_comps / n_tested * 100:.1f}%)")
    print(f"Successful positions: {len(results)} ({len(results) / n_tested * 100:.1f}%)")

    if len(results) == 0:
        print("\nWARNING: No valid positions found!")
        print("\nPossible issues:")
        print(f"  - Detector too small ({det_width}×{det_height} px)")
        print(f"  - Aperture + padding too large ({aperture_radius + edge_padding} px from edges)")
        print(f"  - All comparison stars fall outside detector at all positions")
        print(f"  - Bad pixel contamination too high")
        raise ValueError("No valid positions found - detector may be too contaminated with bad pixels")

    # Find best precision
    precisions = np.array([r['precision'] for r in results])
    best_precision = np.min(precisions)

    # Get all positions with best precision
    optimal_positions = [r for r in results if r['precision'] == best_precision]

    print(f"\nBest precision: {best_precision:.6f}")
    print(f"Number of positions with best precision: {len(optimal_positions)}")
    print(f"Precision range: [{np.min(precisions):.6f}, {np.max(precisions):.6f}]")
    print(f"Mean precision: {np.mean(precisions):.6f}")

    # Break ties by maximizing distance to nearest hazard (bad pixel or edge)
    if len(optimal_positions) > 1:
        best_distance = -np.inf
        best_result = None

        for result in optimal_positions:
            dist = distance_to_nearest_hazard(result['x'], result['y'],
                                              bad_pixel_map, det_width, det_height, edge_padding)
            if dist > best_distance:
                best_distance = dist
                best_result = result
    else:
        best_result = optimal_positions[0]
        best_distance = distance_to_nearest_hazard(best_result['x'], best_result['y'],
                                                   bad_pixel_map, det_width, det_height, edge_padding)

    print(f"\nOptimal position: X={best_result['x']:.1f}, Y={best_result['y']:.1f}")
    print(f"Distance to nearest hazard (bad pixel or edge): {best_distance:.2f} pixels")
    print(f"Number of comparison stars: {best_result['n_comp']}")

    # Calculate additional positioning information
    # Center of detector in detector coordinates
    center_x = det_width / 2.0
    center_y = det_height / 2.0

    # Offset of target from center in detector coordinates
    offset_x = best_result['x'] - center_x
    offset_y = best_result['y'] - center_y

    # Convert detector center to RA/Dec using optimal WCS
    wcs_optimal = create_wcs(target_ra, target_dec, best_result['x'], best_result['y'], config)
    center_ra, center_dec = pixel_to_sky(center_x, center_y, wcs_optimal)

    # Calculate offset in RA/Dec (target is at the reference position of the WCS)
    # The target RA/Dec is the WCS reference point, so offset from center:
    offset_ra = target_ra - center_ra  # degrees
    offset_dec = target_dec - center_dec  # degrees

    # Convert to arcseconds for readability
    offset_ra_arcsec = offset_ra * 3600.0
    offset_dec_arcsec = offset_dec * 3600.0

    print(f"\n--- Detector Center Information ---")
    print(f"Detector center (optimized): RA={center_ra:.6f}°, Dec={center_dec:.6f}°")
    print(f"Target offset from center: ΔX={offset_x:.1f} px, ΔY={offset_y:.1f} px")
    print(f"Target offset from center: ΔRA={offset_ra_arcsec:.2f}\", ΔDec={offset_dec_arcsec:.2f}\"")

    # Create precision map if requested
    precision_map = None
    if save_precision_map:
        print("\n=== Creating Precision Map ===")

        # Calculate coarse grid dimensions
        coarse_height = len(y_positions)
        coarse_width = len(x_positions)

        print(f"Coarse grid dimensions: {coarse_width} × {coarse_height}")
        print(f"Each pixel represents {grid_spacing} × {grid_spacing} detector pixels")

        # Initialize coarse map with NaN
        precision_map = np.full((coarse_height, coarse_width), np.nan)

        # Fill in computed values at coarse grid positions
        # Map from detector coordinates to coarse grid indices
        x_pos_to_idx = {int(x): i for i, x in enumerate(x_positions)}
        y_pos_to_idx = {int(y): i for i, y in enumerate(y_positions)}

        for result in results:
            coarse_x_idx = x_pos_to_idx[int(result['x'])]
            coarse_y_idx = y_pos_to_idx[int(result['y'])]
            precision_map[coarse_y_idx, coarse_x_idx] = result['precision']

        n_filled = np.sum(np.isfinite(precision_map))
        print(f"Precision map contains {n_filled} valid positions")

        # Generate output filename
        import os
        run_dir = create_run_directory(gaia_id)
        ref_image_base = 'noref'
        if 'reference_image' in config:
            ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
        badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
        map_output_path = run_dir / f'precision_map_{gaia_id}_{ref_image_base}_{badpix_base}.fits'

        print(f"Saving precision map FITS to: {map_output_path}")

        # Save to FITS with coarse grid parameters
        from astropy.io import fits
        hdu = fits.PrimaryHDU(precision_map)
        hdu.header['BUNIT'] = 'precision'
        hdu.header['GAIA_ID'] = str(gaia_id)
        hdu.header['TARG_RA'] = (target_ra, 'Target RA (deg)')
        hdu.header['TARG_DEC'] = (target_dec, 'Target Dec (deg)')
        hdu.header['TARG_J'] = (target_jmag, 'Target J magnitude')
        hdu.header['TARG_TEFF'] = (target_teff, 'Target Teff (K)')
        hdu.header['OPT_X'] = (best_result['x'], 'Optimal X position (detector coords)')
        hdu.header['OPT_Y'] = (best_result['y'], 'Optimal Y position (detector coords)')
        hdu.header['OPT_PREC'] = (best_precision, 'Optimal precision')

        if 'reference_image' in config:
            hdu.header['REFIMAGE'] = (config['reference_image'], 'Reference FITS image')
        hdu.header['BADPXMAP'] = (config['detector']['bad_pixel_map_path'], 'Bad pixel map file')

        # Coarse grid parameters
        hdu.header['APERTURE'] = (aperture_radius, 'Aperture radius (detector pixels)')
        hdu.header['EDGEPAD'] = (edge_padding, 'Edge padding (detector pixels)')
        hdu.header['GRIDSPAC'] = (grid_spacing, 'Grid spacing (detector pixels)')
        hdu.header['PIXSCALE'] = (config['detector']['pixel_scale_arcsec'] * grid_spacing,
                                  'Pixel scale (arcsec/pixel, coarse grid)')
        hdu.header['DET_NX'] = (det_width, 'Detector width (detector pixels)')
        hdu.header['DET_NY'] = (det_height, 'Detector height (detector pixels)')

        hdu.header['COMPFNT'] = (fainter_limit, 'Comp star fainter limit (mag)')
        hdu.header['COMPBRT'] = (brighter_limit, 'Comp star brighter limit (mag)')
        hdu.header['NCOMPALL'] = (len(comp_stars), 'Total potential comp stars')
        hdu.header['NVALID'] = (len(results), 'Number of valid positions')

        hdu.header['COMMENT'] = 'Predicted photometric precision at grid positions'
        hdu.header['COMMENT'] = f'Coarse grid: each pixel = {grid_spacing}x{grid_spacing} detector pixels'
        hdu.header['COMMENT'] = 'NaN values indicate invalid positions'

        hdu.writeto(map_output_path, overwrite=True)

        print(f"Precision map FITS saved successfully")

        # Create PNG visualization
        save_precision_map_png(precision_map, config, gaia_id,
                               best_result['x'], best_result['y'], best_precision,
                               det_width, det_height)

    target_jmag_val = float(target_jmag)

    return {
        'optimal_x': best_result['x'],
        'optimal_y': best_result['y'],
        'precision': best_result['precision'],
        'n_comparison_stars': best_result['n_comp'],
        'combined_mag': best_result['combined_mag'],
        'target_ra': target_ra,
        'target_dec': target_dec,
        'distance_to_hazard': best_distance,
        'detector_center_ra': center_ra,
        'detector_center_dec': center_dec,
        'target_offset_x': offset_x,
        'target_offset_y': offset_y,
        'target_offset_ra_arcsec': offset_ra_arcsec,
        'target_offset_dec_arcsec': offset_dec_arcsec,
        'target_jmag': target_jmag_val,
        'target_zyj': target_zyj,
        'target_teff': target_teff,
        'precision_map': precision_map
    }

def save_optimization_results(result, gaia_id, output_path=None):
    """
    Save optimization results to JSON file in run directory.

    Args:
        result: Dictionary returned from optimize_target_position()
        gaia_id: Gaia DR2 source_id
        output_path: Path to output JSON file (if None, auto-generate in run dir)
    """
    import json

    if output_path is None:
        run_dir = create_run_directory(gaia_id)
        output_path = run_dir / f"optimization_{gaia_id}.json"

    # Convert numpy types to Python native types for JSON serialization
    serializable_result = {
        'gaia_id': int(gaia_id),
        'status': result.get('status', 'SUCCESS'),
        'timestamp': result.get('timestamp', ''),
        'optimal_x': float(result['optimal_x']),
        'optimal_y': float(result['optimal_y']),
        'precision': float(result['precision']),
        'n_comparison_stars': int(result['n_comparison_stars']),
        'combined_mag': float(result['combined_mag']),
        'target_ra': float(result['target_ra']),
        'target_dec': float(result['target_dec']),
        'distance_to_hazard': float(result['distance_to_hazard']),
        'detector_center_ra': float(result['detector_center_ra']),
        'detector_center_dec': float(result['detector_center_dec']),
        'target_offset_x': float(result['target_offset_x']),
        'target_offset_y': float(result['target_offset_y']),
        'target_offset_ra_arcsec': float(result['target_offset_ra_arcsec']),
        'target_offset_dec_arcsec': float(result['target_offset_dec_arcsec']),
        'target_jmag': float(result.get('target_jmag', 0)),
        'target_zyj': float(result.get('target_zyj', 0)),
        'target_teff': float(result.get('target_teff', 0)),
        'reference_image': result.get('reference_image', ''),
        'bad_pixel_map': result.get('bad_pixel_map', ''),
        'processing_time_seconds': float(result.get('processing_time_seconds', 0)),
        'error_message': result.get('error_message', '')
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)

    print(f"\nResults saved to: {output_path}")

def save_precision_map_png(precision_map, config, gaia_id, optimal_x_det, optimal_y_det,
                          best_precision, det_width, det_height):
    """
    Create a PNG visualization of the precision map.

    Args:
        precision_map: 2D array of precision values (coarse grid)
        config: Configuration dictionary
        gaia_id: Gaia DR2 source_id
        optimal_x_det: Optimal X position (detector coordinates)
        optimal_y_det: Optimal Y position (detector coordinates)
        best_precision: Best precision value
        det_width: Full detector width (detector pixels)
        det_height: Full detector height (detector pixels)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize
    import os

    print("\n=== Creating Precision Map PNG ===")

    # Get parameters
    coarse_height, coarse_width = precision_map.shape
    edge_padding = config['detector']['edge_padding_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    grid_spacing = config['optimization']['grid_spacing_pixels']

    # Calculate vmin and vmax
    finite_mask = np.isfinite(precision_map)
    if not np.any(finite_mask):
        print("ERROR: No finite precision values in map")
        return

    finite_values = precision_map[finite_mask]
    vmin = np.min(finite_values)
    vmax = np.max(finite_values)

    print(f"Precision range: [{vmin:.6f}, {vmax:.6f}]")

    # Create figure with proportional dimensions
    # Base size on detector aspect ratio
    aspect_ratio = det_width / det_height
    if aspect_ratio >= 1:
        # Wider than tall
        fig_width = 12
        fig_height = 12 / aspect_ratio
    else:
        # Taller than wide
        fig_height = 12
        fig_width = 12 * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create RGB image manually to handle NaN coloring
    cmap = plt.colormaps.get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Initialize RGB array
    rgb_image = np.ones((coarse_height, coarse_width, 3))

    # Fill in valid precision values with colormap
    for i in range(coarse_height):
        for j in range(coarse_width):
            if np.isfinite(precision_map[i, j]):
                rgb_image[i, j, :3] = cmap(norm(precision_map[i, j]))[:3]
            else:
                # NaN pixels (bad pixels or no comparison stars) -> red
                rgb_image[i, j] = [1.0, 0.0, 0.0]

    # Display the image with extent in detector coordinates
    # The coarse grid starts at (first_x, first_y) and ends at (last_x, last_y)
    first_x = edge_padding + aperture_radius
    last_x = det_width - edge_padding - aperture_radius
    first_y = edge_padding + aperture_radius
    last_y = det_height - edge_padding - aperture_radius

    im = ax.imshow(rgb_image, origin='lower',
                   extent=[first_x, last_x, first_y, last_y],
                   aspect='equal', interpolation='nearest')

    # Draw edge padding box (safe zone boundary) - in detector coordinates
    padding_rect = patches.Rectangle(
        (edge_padding, edge_padding),
        det_width - 2 * edge_padding,
        det_height - 2 * edge_padding,
        linewidth=2, edgecolor='black', facecolor='none',
        linestyle='--', label='Edge padding boundary'
    )
    ax.add_patch(padding_rect)

    # Mark optimal position with a cross - in detector coordinates
    ax.plot(optimal_x_det, optimal_y_det, 'wx', markersize=20, markeredgewidth=3,
            label=f'Optimal position ({optimal_x_det:.1f}, {optimal_y_det:.1f})')

    # Axis labels (detector coordinates)
    ax.set_xlabel('Detector X Position (pixels)', fontsize=12)
    ax.set_ylabel('Detector Y Position (pixels)', fontsize=12)
    ax.set_title(f'Precision Map - Gaia DR2 {gaia_id}\n'
                f'Grid spacing: {grid_spacing} px | Best precision: {best_precision:.6f}',
                fontsize=14, weight='bold')

    # Set axis limits to full detector dimensions (shows untested edge regions)
    ax.set_xlim(0, det_width)
    ax.set_ylim(0, det_height)

    # Add colorbar for valid precision values
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Precision', fontsize=12)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        ax.get_legend_handles_labels()[0][0],  # Edge padding box
        ax.get_legend_handles_labels()[0][1],  # Optimal position
        Patch(facecolor='red', edgecolor='black', label='Invalid (bad pixels/no comps)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

    # Generate output filename
    run_dir = create_run_directory(gaia_id)
    ref_image_base = 'noref'
    if 'reference_image' in config:
        ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
    badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
    png_output_path = run_dir / f'precision_map_{gaia_id}_{ref_image_base}_{badpix_base}.png'

    try:
        plt.tight_layout()
    except ValueError:
        print("Warning: tight_layout failed, saving without layout adjustment")

    plt.savefig(png_output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Precision map PNG saved to: {png_output_path}")

def create_optimization_visualization(gaia_id, config, opt_result, output_path=None):
    """
    Create a PNG visualization showing the queried field with sky survey background,
    stars, and optimal detector position.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        opt_result: Dictionary returned from optimize_target_position()
        output_path: Path to output PNG file (if None, auto-generate)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle, Rectangle, Polygon
    import os
    from astroquery.skyview import SkyView
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    print("\n=== Creating Optimization Visualization ===")

    # Auto-generate output filename if not provided
    if output_path is None:
        run_dir = create_run_directory(gaia_id)
        ref_image_base = 'noref'
        if 'reference_image' in config:
            ref_image_base = os.path.splitext(os.path.basename(config['reference_image']))[0]
        badpix_base = os.path.splitext(os.path.basename(config['detector']['bad_pixel_map_path']))[0]
        output_path = run_dir / f'optimization_viz_{gaia_id}_{ref_image_base}_{badpix_base}.png'

    # Query expanded field
    expansion = calculate_expansion_factor(config)
    jmag_data = get_field_jmag(gaia_id, config, expansion_factor=expansion)

    # Get target info
    target_ra = opt_result['target_ra']
    target_dec = opt_result['target_dec']
    optimal_x = opt_result['optimal_x']
    optimal_y = opt_result['optimal_y']

    # Get target row
    target_row = jmag_data[jmag_data['source_id'] == int(gaia_id)]
    target_jmag = float(target_row['j_m'][0])

    # Filter comparison stars by magnitude
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']
    comp_stars = jmag_data[
        (jmag_data['source_id'] != int(gaia_id)) &
        (jmag_data['j_m'] < target_jmag + fainter_limit) &
        (jmag_data['j_m'] > target_jmag + brighter_limit)
        ]

    # Calculate field size for sky survey query
    all_ra = jmag_data['ra']
    all_dec = jmag_data['dec']
    ra_range = np.max(all_ra) - np.min(all_ra)
    dec_range = np.max(all_dec) - np.min(all_dec)

    # Add margin
    margin_factor = 1.1
    ra_width = ra_range * margin_factor
    dec_height = dec_range * margin_factor

    # Center position
    center_ra = np.mean(all_ra)
    center_dec = np.mean(all_dec)

    print(f"Querying sky survey at RA={center_ra:.6f}, Dec={center_dec:.6f}")
    print(f"Field size: {ra_width * 60:.2f}' × {dec_height * 60:.2f}'")

    # Query sky survey image
    try:
        center_coord = SkyCoord(center_ra, center_dec, unit='deg', frame='icrs')
        img_list = SkyView.get_images(
            position=center_coord,
            survey='DSS2 Red',
            width=ra_width * u.deg,
            height=dec_height * u.deg,
            pixels=[1000, 1000]
        )

        if len(img_list) == 0:
            raise ValueError("No image returned from SkyView")

        survey_hdu = img_list[0][0]
        survey_data = survey_hdu.data
        survey_wcs = WCS(survey_hdu.header)

        print("Sky survey image retrieved successfully")

    except Exception as e:
        print(f"Warning: Could not retrieve sky survey image: {e}")
        print("Creating visualization without background image")
        survey_data = None
        survey_wcs = None

    # Create WCS for optimal detector position
    detector_wcs = create_wcs(target_ra, target_dec, optimal_x, optimal_y, config)

    # Get detector parameters
    det_width = config['detector']['width_pixels']
    det_height = config['detector']['height_pixels']
    aperture_radius = config['aperture']['radius_pixels']
    pixel_scale_arcsec = config['detector']['pixel_scale_arcsec']

    # Create figure with proportional dimensions based on RA/Dec range
    aspect_ratio = (ra_width * np.cos(np.radians(center_dec))) / dec_height
    if aspect_ratio >= 1:
        fig_width = 14
        fig_height = 14 / aspect_ratio
    else:
        fig_height = 14
        fig_width = 14 * aspect_ratio

    # Create figure with WCS projection
    if survey_wcs is not None:
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.subplot(projection=survey_wcs)

        # Plot sky survey image
        ax.imshow(survey_data, origin='lower', cmap='gray',
                  vmin=np.percentile(survey_data, 5),
                  vmax=np.percentile(survey_data, 99.5))

        # Invert the Y-axis to match detector orientation (south-up)
        ax.invert_yaxis()

        # Configure RA/Dec display
        ax.coords[0].set_format_unit(u.deg)
        ax.coords[1].set_format_unit(u.deg)
        ax.coords[0].set_major_formatter('d.ddd')
        ax.coords[1].set_major_formatter('d.ddd')

        # Set equal aspect for RA/Dec (accounting for declination)
        # ax.set_aspect('equal')  # This should work with WCS projection

    else:
        # Fallback without survey image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_facecolor('black')
        ax.set_aspect('equal')

    # Plot all queried stars
    ax.plot(all_ra, all_dec, 'o', color='cyan', markersize=3,
            alpha=0.6, label='All queried stars', transform=ax.get_transform('world'))

    # Plot comparison stars with circles and full Gaia IDs
    for ra, dec, source_id in zip(comp_stars['ra'], comp_stars['dec'], comp_stars['source_id']):
        # Convert aperture radius to sky coordinates (approximate)
        aperture_radius_deg = aperture_radius * pixel_scale_arcsec / 3600.0

        # Draw circle around comparison star
        circle = Circle((ra, dec), aperture_radius_deg, fill=False,
                        edgecolor='blue', linewidth=1.5, alpha=0.8,
                        transform=ax.get_transform('world'))
        ax.add_patch(circle)

        # Add star marker
        ax.plot(ra, dec, '*', color='blue', markersize=10, alpha=0.9,
                transform=ax.get_transform('world'))

        # Add full Gaia ID label
        label = str(source_id)
        ax.annotate(label, xy=(ra, dec), xycoords=ax.get_transform('world'),
                    xytext=(15, 0), textcoords='offset points',  # 15 points to the right
                    fontsize=6, color='blue', alpha=0.8,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.7, edgecolor='blue'))

    # Plot target star
    aperture_radius_deg = aperture_radius * pixel_scale_arcsec / 3600.0
    target_circle = Circle((target_ra, target_dec), aperture_radius_deg,
                           fill=False, edgecolor='red', linewidth=2.5, alpha=0.95,
                           transform=ax.get_transform('world'))
    ax.add_patch(target_circle)

    ax.plot(target_ra, target_dec, '*', color='red', markersize=15,
            alpha=0.95, label='Target star', transform=ax.get_transform('world'))

    # Add target label with full Gaia ID
    target_label = f"Target\n{gaia_id}"
    ax.annotate(target_label, xy=(target_ra, target_dec), xycoords=ax.get_transform('world'),
                xytext=(15, 0), textcoords='offset points',  # 15 points to the right
                fontsize=8, color='red', weight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.8, edgecolor='red'))

    # Draw detector boundary at optimal position
    # Get detector center from optimization result
    detector_center_ra = opt_result['detector_center_ra']
    detector_center_dec = opt_result['detector_center_dec']

    # Draw detector boundary at optimal position
    # Get detector corners in detector coordinates (X, Y order)
    detector_corners_x = np.array([0, det_width, det_width, 0, 0])
    detector_corners_y = np.array([0, 0, det_height, det_height, 0])

    # Convert to RA/Dec using the detector WCS
    # This WCS has target at position (optimal_x, optimal_y)
    detector_corners_ra, detector_corners_dec = pixel_to_sky(
        detector_corners_x, detector_corners_y, detector_wcs
    )

    # Debug: print corner positions
    print(f"Detector corners RA range: [{np.min(detector_corners_ra):.6f}, {np.max(detector_corners_ra):.6f}]")
    print(f"Detector corners Dec range: [{np.min(detector_corners_dec):.6f}, {np.max(detector_corners_dec):.6f}]")
    print(f"Target RA/Dec: ({target_ra:.6f}, {target_dec:.6f})")
    print(f"Detector center RA/Dec: ({detector_center_ra:.6f}, {detector_center_dec:.6f})")

    # Plot detector FOV
    ax.plot(detector_corners_ra, detector_corners_dec,
            color='lime', linewidth=2.5, linestyle='--',
            alpha=0.9, label='Detector FOV (optimal)',
            transform=ax.get_transform('world'))

    # Mark detector center with X
    ax.plot(detector_center_ra, detector_center_dec, 'x',
            color='lime', markersize=18, markeredgewidth=3, alpha=0.9,
            transform=ax.get_transform('world'))

    # Mark detector center with X
    detector_center_ra = opt_result['detector_center_ra']
    detector_center_dec = opt_result['detector_center_dec']
    ax.plot(detector_center_ra, detector_center_dec, 'x',
            color='lime', markersize=18, markeredgewidth=3, alpha=0.9,
            transform=ax.get_transform('world'))

    # Labels and title
    if survey_wcs is not None:
        ax.coords[0].set_axislabel('RA (deg)', fontsize=12)
        ax.coords[1].set_axislabel('Dec (deg)', fontsize=12)
    else:
        ax.set_xlabel('RA (deg)', fontsize=12)
        ax.set_ylabel('Dec (deg)', fontsize=12)

    ax.set_title(f'Optimal Detector Positioning - Gaia DR2 {gaia_id}\n'
                 f'Precision: {opt_result["precision"]:.6f} | '
                 f'Comparison stars: {opt_result["n_comparison_stars"]}',
                 fontsize=14, weight='bold')

    # Grid
    if survey_wcs is not None:
        ax.coords.grid(color='white', alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

    # Create secondary axes for Detector X/Y coordinates
    # We need to create transformation functions with bounds checking
    def ra_to_detx(ra_vals):
        """Transform RA to detector X coordinate"""
        if isinstance(ra_vals, (int, float)):
            ra_vals = np.array([ra_vals])
        # For each RA, use the mean Dec to get approximate X
        dec_vals = np.full_like(ra_vals, target_dec)
        x_vals, _ = sky_to_pixel(ra_vals, dec_vals, detector_wcs)
        # Replace any invalid values with reasonable bounds
        x_vals = np.where(np.isfinite(x_vals), x_vals,
                          np.where(x_vals > 0, det_width, 0))
        return x_vals

    def detx_to_ra(x_vals):
        """Transform detector X to RA"""
        if isinstance(x_vals, (int, float)):
            x_vals = np.array([x_vals])
        # For each X, use center Y to get RA
        y_vals = np.full_like(x_vals, optimal_y)
        ra_vals, _ = pixel_to_sky(x_vals, y_vals, detector_wcs)
        # Ensure finite values
        ra_vals = np.where(np.isfinite(ra_vals), ra_vals, target_ra)
        return ra_vals

    def dec_to_dety(dec_vals):
        """Transform Dec to detector Y coordinate"""
        if isinstance(dec_vals, (int, float)):
            dec_vals = np.array([dec_vals])
        # For each Dec, use the mean RA to get approximate Y
        ra_vals = np.full_like(dec_vals, target_ra)
        _, y_vals = sky_to_pixel(ra_vals, dec_vals, detector_wcs)
        # Replace any invalid values with reasonable bounds
        y_vals = np.where(np.isfinite(y_vals), y_vals,
                          np.where(y_vals > 0, det_height, 0))
        return y_vals

    def dety_to_dec(y_vals):
        """Transform detector Y to Dec"""
        if isinstance(y_vals, (int, float)):
            y_vals = np.array([y_vals])
        # For each Y, use center X to get Dec
        x_vals = np.full_like(y_vals, optimal_x)
        _, dec_vals = pixel_to_sky(x_vals, y_vals, detector_wcs)
        # Ensure finite values
        dec_vals = np.where(np.isfinite(dec_vals), dec_vals, target_dec)
        return dec_vals

    # # Create secondary X axis (top) for detector X
    # try:
    #     ax_top = ax.secondary_xaxis('top', functions=(ra_to_detx, detx_to_ra))
    #     ax_top.set_xlabel('Detector X (pixels)', fontsize=12)
    # except Exception as e:
    #     print(f"Warning: Could not create secondary X axis: {e}")
    #
    # # Create secondary Y axis (right) for detector Y
    # try:
    #     ax_right = ax.secondary_yaxis('right', functions=(dec_to_dety, dety_to_dec))
    #     ax_right.set_ylabel('Detector Y (pixels)', fontsize=12)
    # except Exception as e:
    #     print(f"Warning: Could not create secondary Y axis: {e}")

    # Legend
    ax.legend(loc='upper left', fontsize=10, fancybox=True,
              framealpha=0.8)

    # Add info text box
    info_text = (f"Aperture radius: {aperture_radius} px\n"
                 f"Detector: {det_width}×{det_height} px\n"
                 f"Distance to hazard: {opt_result['distance_to_hazard']:.1f} px\n"
                 f"Optimal position: ({optimal_x:.1f}, {optimal_y:.1f})\n"
                 f"Target offset: ({opt_result['target_offset_x']:.1f}, {opt_result['target_offset_y']:.1f}) px\n"
                 f"Detector center: ({opt_result['detector_center_ra']:.6f}°, {opt_result['detector_center_dec']:.6f}°)")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color='black')

    try:
        plt.tight_layout()
    except ValueError:
        print("Warning: tight_layout failed, saving without layout adjustment")

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def predict(gaia_id, config, optimize=False, save_results=False, create_viz=False, save_precision_map=False):
    """
    Main prediction function for a given target star.

    Args:
        gaia_id: Gaia DR2 source_id of target star
        config: Configuration dictionary
        optimize: If True, optimize target position on detector. If False, assume centered.
        save_results: If True, save optimization results to JSON file
        create_viz: If True, create PNG visualization of optimization
        save_precision_map: If True, save precision map FITS file (only when optimize=True)
    """
    from datetime import datetime

    if optimize:
        # Run optimization to find best position (optionally creating precision map)
        start_time = time.time()
        opt_result = optimize_target_position(gaia_id, config, save_precision_map=save_precision_map)
        processing_time = time.time() - start_time

        # Add metadata to result
        opt_result['gaia_id'] = gaia_id
        opt_result['status'] = 'SUCCESS'
        opt_result['timestamp'] = datetime.now().isoformat()
        opt_result['processing_time_seconds'] = processing_time
        opt_result['reference_image'] = config.get('reference_image', '')
        opt_result['bad_pixel_map'] = config['detector'].get('bad_pixel_map_path', '')
        opt_result['error_message'] = ''

        print("\n=== Optimization Results ===")
        print(f"Optimal detector position: X={opt_result['optimal_x']:.1f}, Y={opt_result['optimal_y']:.1f}")
        print(f"Predicted precision: {opt_result['precision']:.6f}")
        print(f"Usable comparison stars: {opt_result['n_comparison_stars']}")
        print(f"Distance to nearest hazard: {opt_result['distance_to_hazard']:.2f} pixels")
        print(f"Processing time: {processing_time:.1f} seconds")

        if save_results:
            save_optimization_results(opt_result, gaia_id)

        if create_viz:
            create_optimization_visualization(gaia_id, config, opt_result)

        return opt_result

    jmag_data = get_field_jmag(gaia_id, config)

    # Extract target star row
    target_row = jmag_data[jmag_data['source_id'] == int(gaia_id)]
    target_index = np.where(target_row)[0][0]

    # Filter comparison stars by magnitude range (from config)
    fainter_limit = config['comparison_star_limits']['fainter_limit']
    brighter_limit = config['comparison_star_limits']['brighter_limit']
    jmag_data = jmag_data[jmag_data['j_m'] < (target_row['j_m'] + fainter_limit)]
    jmag_data = jmag_data[jmag_data['j_m'] > (target_row['j_m'] + brighter_limit)]

    # Extract comparison star magnitudes (excluding target)
    comp_star_mag = jmag_data[jmag_data['source_id'] != int(gaia_id)]['j_m']

    # Convert all J-band magnitudes to zYJ
    zyj_mags_all = convert_j_to_zyj(jmag_data['j_m'], config)

    # Calculate combined magnitude (including target)
    combined_mags = combined_mag(zyj_mags_all)
    effective_mag = effective_mg(zyj_mags_all, target_index)

    # Predict precision using quadratic fit
    predicted_precision = 10 ** prediction_from_fit(combined_mags, config)
    eff_predicted_precision = 10 ** prediction_from_fit(effective_mag, config)

    # Calculate combined magnitude without target
    combined_mags_wo_target = combined_mag(convert_j_to_zyj(comp_star_mag, config))

    # Extract target properties and convert to floats
    n_comp = len(comp_star_mag)
    target_zyj = to_float(convert_j_to_zyj(target_row['j_m'], config))
    target_teff = to_float(target_row['teff_val'])
    combined_mags = to_float(combined_mags)
    effective_mag = to_float(effective_mag)
    combined_mags_wo_target = to_float(combined_mags_wo_target)
    predicted_precision = to_float(predicted_precision)

    # Print target information
    print(f"\n--- Target Star Information ---")
    print(f"Gaia DR2 ID: {gaia_id}")
    print(f"zYJ magnitude: {target_zyj:.3f}")
    print(f"Effective temperature: {target_teff:.0f} K")

    # Print field information
    print(f"\n--- Field Information ---")
    print(f"Number of comparison stars: {n_comp}")
    print(f"Combined magnitude (with target): {combined_mags:.3f}")
    print(f"Effective magnitude: {effective_mag:.3f}")
    print(f"Combined magnitude (without target): {combined_mags_wo_target:.3f}")

    # Print quadratic fit prediction
    print(f"\n--- Precision Predictions ---")
    print(f"Quadratic fit combined prediction: {predicted_precision:.6f}")
    print(f"Quadratic fit effective prediction: {eff_predicted_precision:.6f}")

    # Prepare features for Decision Tree prediction
    features = pd.DataFrame({
        'Comp stars': [n_comp],
        'zYJ mag': [target_zyj],
        'Combined mag': [combined_mags_wo_target],
        'Teff': [target_teff]
    })

    # Predict using Decision Tree model
    predicted_precision_DT = to_float(prediction_from_DT(features))
    print(f"Decision Tree prediction: {predicted_precision_DT:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict photometric precision for SPECULOOS targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target
  python predict_target_precision.py --target 123456789 --optimize --save --viz

  # Batch processing
  python predict_target_precision.py --batch targets.txt --optimize --save --viz --map
        """
    )

    # Target specification (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target', type=int, help='Single Gaia DR2 source ID')
    target_group.add_argument('--batch', type=str, help='Batch target list filename (in target_lists/)')

    # Operation modes
    parser.add_argument('--optimize', action='store_true', help='Run position optimization')
    parser.add_argument('--centered', action='store_true', help='Run centered prediction')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    parser.add_argument('--viz', action='store_true', help='Create visualization')
    parser.add_argument('--map', action='store_true', help='Create precision map')

    args = parser.parse_args()

    # Load configuration once
    config = load_config()

    # Single target mode
    if args.target:
        gaia_id = args.target

        print("\n" + "=" * 60)
        print(f"PROCESSING TARGET: {gaia_id}")
        print("=" * 60)

        # Run centered prediction if requested
        if args.centered:
            print("\n" + "=" * 60)
            print("RUNNING CENTERED PREDICTION")
            print("=" * 60)
            predict(gaia_id, config, optimize=False, save_results=False, create_viz=False)

        # Run optimization if requested
        if args.optimize:
            print("\n" + "=" * 60)
            print("RUNNING OPTIMIZATION")
            print("=" * 60)
            predict(gaia_id, config, optimize=True, save_results=args.save,
                    create_viz=args.viz, save_precision_map=args.map)

        # If no mode specified, default to centered prediction
        if not (args.centered or args.optimize):
            print("\n" + "=" * 60)
            print("RUNNING CENTERED PREDICTION (default)")
            print("=" * 60)
            predict(gaia_id, config, optimize=False, save_results=False, create_viz=False)

    # Batch mode
    elif args.batch:
        from datetime import datetime

        print("\n" + "=" * 60)
        print(f"BATCH PROCESSING: {args.batch}")
        print("=" * 60)

        # Read target list
        try:
            gaia_ids = read_target_list(args.batch)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        if len(gaia_ids) == 0:
            print("Error: No valid Gaia IDs found in target list")
            return

        # Create batch directory and CSV
        batch_dir = create_batch_directory()
        csv_path = initialize_batch_csv(batch_dir)

        print(f"Batch directory: {batch_dir}")
        print(f"Results will be written to: {csv_path}")
        print(f"Processing {len(gaia_ids)} targets...\n")

        # Process each target
        for i, gaia_id in enumerate(gaia_ids, 1):
            print("\n" + "=" * 60)
            print(f"TARGET {i}/{len(gaia_ids)}: {gaia_id}")
            print("=" * 60)

            try:
                # Run optimization
                result = predict(gaia_id, config, optimize=True,
                                 save_results=args.save, create_viz=args.viz,
                                 save_precision_map=args.map)

                # Append to batch CSV
                append_to_batch_csv(csv_path, result)

                print(f"✓ Target {gaia_id} completed successfully")

            except Exception as e:
                # Handle failure
                print(f"✗ Target {gaia_id} FAILED: {e}")

                # Write failure to CSV
                failed_result = {
                    'gaia_id': gaia_id,
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat(),
                    'reference_image': config.get('reference_image', ''),
                    'bad_pixel_map': config['detector'].get('bad_pixel_map_path', ''),
                    'error_message': str(e)
                }
                append_to_batch_csv(csv_path, failed_result)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
