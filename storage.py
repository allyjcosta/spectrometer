import glob
import json
import os
from datetime import datetime

import numpy as np


SAVE_DIR = "measurements"


def calculate_noise_stats(freqs_hz, asd_v_per_sqrt_hz, stats_min_hz, stats_max_hz):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    asd_v_per_sqrt_hz = np.asarray(asd_v_per_sqrt_hz, dtype=float)

    valid = (
        np.isfinite(freqs_hz)
        & np.isfinite(asd_v_per_sqrt_hz)
        & (freqs_hz > stats_min_hz)
        & (freqs_hz < stats_max_hz)
        & (asd_v_per_sqrt_hz > 0)
    )

    if not np.any(valid):
        return {
            "stats_min_Hz": float(stats_min_hz),
            "stats_max_Hz": float(stats_max_hz),
            "mean_V_per_sqrtHz": None,
            "median_V_per_sqrtHz": None,
            "floor90_V_per_sqrtHz": None,
            "mean_nV_per_sqrtHz": None,
            "median_nV_per_sqrtHz": None,
            "floor90_nV_per_sqrtHz": None,
        }

    stats_amps = asd_v_per_sqrt_hz[valid]

    mean_v = float(np.mean(stats_amps))
    median_v = float(np.median(stats_amps))
    floor90_v = float(np.percentile(stats_amps, 90))

    return {
        "stats_min_Hz": float(stats_min_hz),
        "stats_max_Hz": float(stats_max_hz),
        "mean_V_per_sqrtHz": mean_v,
        "median_V_per_sqrtHz": median_v,
        "floor90_V_per_sqrtHz": floor90_v,
        "mean_nV_per_sqrtHz": mean_v * 1e9,
        "median_nV_per_sqrtHz": median_v * 1e9,
        "floor90_nV_per_sqrtHz": floor90_v * 1e9,
    }


def make_filename(instrument, measurement_type, target_freq_hz=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if target_freq_hz is None:
        return f"{instrument}_{measurement_type}_{timestamp}.json"

    return f"{instrument}_{measurement_type}_{target_freq_hz:.3f}Hz_{timestamp}.json"


def save_measurement(
    instrument,
    measurement_type,
    freqs_hz,
    mag_v,
    asd_v_per_sqrt_hz,
    sample_rate_hz,
    samples,
    plot_config,
    stats_min_hz,
    stats_max_hz,
    test=None,
    filename=None,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    mag_v = np.asarray(mag_v, dtype=float)
    asd_v_per_sqrt_hz = np.asarray(asd_v_per_sqrt_hz, dtype=float)

    nyquist_hz = float(sample_rate_hz) / 2.0
    physical = (
        np.isfinite(freqs_hz)
        & (freqs_hz > 0)
        & (freqs_hz <= nyquist_hz)
    )
    freqs_hz = freqs_hz[physical]
    mag_v = mag_v[physical]
    asd_v_per_sqrt_hz = asd_v_per_sqrt_hz[physical]

    stats = calculate_noise_stats(
        freqs_hz=freqs_hz,
        asd_v_per_sqrt_hz=asd_v_per_sqrt_hz,
        stats_min_hz=stats_min_hz,
        stats_max_hz=stats_max_hz,
    )

    if filename is None:
        target_freq_hz = None
        if test is not None:
            target_freq_hz = test.get("target_freq_Hz")

        filename = make_filename(
            instrument=instrument,
            measurement_type=measurement_type,
            target_freq_hz=target_freq_hz,
        )

    if not filename.startswith(SAVE_DIR):
        filename = os.path.join(SAVE_DIR, filename)

    measurement = {
        "instrument": instrument,
        "measurement_type": measurement_type,

        "sample_rate_Hz": float(sample_rate_hz),
        "nyquist_Hz": nyquist_hz,
        "samples": int(samples),

        "plot_config": plot_config,

        "units": {
            "frequency": "Hz",
            "magnitude": "V",
            "asd": "V/sqrt(Hz)",
        },

        "stats": stats,
        "test": test,

        "freqs_Hz": freqs_hz.tolist(),
        "mag_V": mag_v.tolist(),
        "asd_V_per_sqrtHz": asd_v_per_sqrt_hz.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(measurement, f, indent=4)

    print("Saved:", filename)
    return filename


def load_measurement(filename):
    with open(filename, "r") as f:
        return json.load(f)


def discover_measurements():
    files = []

    for filename in sorted(glob.glob(os.path.join(SAVE_DIR, "*.json"))):
        try:
            measurement = load_measurement(filename)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        if "freqs_Hz" not in measurement:
            continue

        if "mag_V" not in measurement:
            continue

        if "asd_V_per_sqrtHz" not in measurement:
            continue

        files.append(filename)

    return files


def clear_saved_measurements():
    deleted = 0

    for filename in glob.glob(os.path.join(SAVE_DIR, "*.json")):
        try:
            os.remove(filename)
            deleted += 1
            print("Deleted:", filename)
        except OSError as e:
            print("Could not delete:", filename, e)

    if deleted == 0:
        print(f"No saved measurements found in {SAVE_DIR}/")
    else:
        print(f"Deleted {deleted} saved measurements from {SAVE_DIR}/")
