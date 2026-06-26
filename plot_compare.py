import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = "measurements"

COLOR_CYCLE = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "magenta",
    "cyan",
    "black",
]


def _color_for_index(index):
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


def save_json(
    filename,
    instrument,
    stitched_freqs,
    stitched_amps_V,
    sample_rate_Hz,
    samples,
    plot_config,
    stats_min_Hz=100,
    stats_max_Hz=10000,
    test_mode=False,
    test_input_freq_Hz=None,
    test_results=None,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not filename.startswith(SAVE_DIR):
        filename = os.path.join(SAVE_DIR, filename)

    stitched_freqs = np.asarray(stitched_freqs, dtype=float)
    stitched_amps_V = np.asarray(stitched_amps_V, dtype=float)

    valid_all = (
        np.isfinite(stitched_freqs)
        & np.isfinite(stitched_amps_V)
        & (stitched_freqs > 0)
        & (stitched_amps_V > 0)
    )

    valid_stats = (
        valid_all
        & (stitched_freqs > stats_min_Hz)
        & (stitched_freqs < stats_max_Hz)
    )

    if np.any(valid_stats):
        stats_amps_V = stitched_amps_V[valid_stats]
    else:
        stats_amps_V = stitched_amps_V[valid_all]

    if len(stats_amps_V) > 0:
        median_V = float(np.median(stats_amps_V))
        floor90_V = float(np.percentile(stats_amps_V, 90))
    else:
        median_V = None
        floor90_V = None

    # Determine structural key-naming and units based on operational state
    if test_mode:
        amp_unit = "V"
        median_key = "median_V"
        floor_key = "floor90_V"
    else:
        amp_unit = "V/sqrt(Hz)"
        median_key = "median_V_per_sqrtHz"
        floor_key = "floor90_V_per_sqrtHz"

    measurement = {
        "instrument": instrument,
        "sample_rate_Hz": sample_rate_Hz,
        "samples": samples,
        "plot_config": plot_config,
        "test_mode": bool(test_mode),
        "test_input_freq_Hz": test_input_freq_Hz,
        "units": {
            "amplitude": amp_unit,
            "frequency": "Hz",
        },
        "stats": {
            "stats_min_Hz": float(stats_min_Hz),
            "stats_max_Hz": float(stats_max_Hz),
            median_key: median_V,
            floor_key: floor90_V,
            f"{median_key.replace('_V', '_nV')}": None if median_V is None else median_V * 1e9,
            f"{floor_key.replace('_V', '_nV')}": None if floor90_V is None else floor90_V * 1e9,
        },
        "freqs_Hz": stitched_freqs.tolist(),
        "amps_raw_stream": stitched_amps_V.tolist(),
        "test_results": test_results,
    }

    with open(filename, "w") as f:
        json.dump(measurement, f, indent=4)

    print("Saved:", filename)


COMPARE_FILES = {}


def _measurement_has_supported_units(measurement):
    # Backward compatible check for standard key parsing structures
    return "freqs_Hz" in measurement and (
        "amps_raw_stream" in measurement
        or "amps_V_per_sqrtHz" in measurement
        or "amps_dBV_per_sqrtHz" in measurement
    )


def _discover_measurement_files():
    discovered = {}
    for filename in sorted(glob.glob(os.path.join(SAVE_DIR, "*.json"))):
        try:
            with open(filename, "r") as f:
                measurement = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        if not _measurement_has_supported_units(measurement):
            continue

        instrument = measurement.get("instrument", filename)
        discovered[f"{instrument} ({filename})"] = {
            "file": filename,
            "enabled": True,
        }
    return discovered


def _load_asd_V_per_sqrtHz(measurement):
    freqs = np.asarray(measurement["freqs_Hz"], dtype=float)

    if "amps_raw_stream" in measurement:
        amps_V = np.asarray(measurement["amps_raw_stream"], dtype=float)
    elif "amps_V_per_sqrtHz" in measurement:
        amps_V = np.asarray(measurement["amps_V_per_sqrtHz"], dtype=float)
    elif "amps_dBV_per_sqrtHz" in measurement:
        amps_dBV = np.asarray(measurement["amps_dBV_per_sqrtHz"], dtype=float)
        amps_V = 10 ** (amps_dBV / 20.0)
    else:
        raise KeyError("Measurement does not contain supported amplitude units.")

    valid = np.isfinite(freqs) & np.isfinite(amps_V) & (freqs > 0) & (amps_V > 0)
    return freqs[valid], amps_V[valid]


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


def _is_test_mode_measurement(measurement):
    instrument = str(measurement.get("instrument", "")).upper()
    return bool(measurement.get("test_mode", False)) or "TEST" in instrument


def _format_freq(freq_Hz):
    if freq_Hz is None:
        return "unknown"
    freq_Hz = float(freq_Hz)
    if freq_Hz >= 1000:
        return f"{freq_Hz / 1000:.3f} kHz"
    return f"{freq_Hz:.3f} Hz"


def compare_saved_measurements(files=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    if files is None:
        files = COMPARE_FILES if COMPARE_FILES else _discover_measurement_files()

    if not files:
        print("No saved measurement files found to compare.")
        plt.close(fig)
        return

    plotted_any = False
    plot_index = 0
    y_min = None
    y_max = None

    for instrument, file_config in files.items():
        if not file_config.get("enabled", True):
            continue

        filename = file_config["file"]

        try:
            with open(filename, "r") as f:
                measurement = json.load(f)
        except FileNotFoundError:
            print(f"Missing file: {filename}")
            continue
        except json.JSONDecodeError:
            print(f"Invalid JSON: {filename}")
            continue

        try:
            freqs, amps_V = _load_asd_V_per_sqrtHz(measurement)
        except KeyError as err:
            print(f"Skipping {filename}: {err}")
            continue

        if len(freqs) == 0:
            print(f"No valid data in: {filename}")
            continue

        plot_config = measurement.get("plot_config", {})
        color = _color_for_index(plot_index)
        base_label = plot_config.get("label", instrument)

        if y_min is None and "y_min" in plot_config and "y_max" in plot_config:
            y_min = plot_config["y_min"]
            y_max = plot_config["y_max"]

        is_test_mode = _is_test_mode_measurement(measurement)
        label = base_label

        if is_test_mode:
            results = measurement.get("test_results", {})
            input_freq = results.get("input_freq_Hz", measurement.get("test_input_freq_Hz"))
            measured_freq = results.get("measured_freq_Hz")
            freq_error_bins = results.get("freq_error_bins")
            gain_dB = results.get("gain_dB")

            freq_label = f"Δf={float(freq_error_bins):.3f} bins" if freq_error_bins is not None else "Δf=?"
            gain_label = f"Gain={float(gain_dB):.3f} dB" if gain_dB is not None else "Gain=?"

            label = (
                f"{_format_freq(input_freq)} → {_format_freq(measured_freq)}"
                f" | {freq_label} | {gain_label}"
            )

        ax.plot(
            freqs,
            amps_V,
            linestyle=plot_config.get("linestyle", "-"),
            color=color,
            linewidth=plot_config.get("linewidth", 0.8),
            label=label,
        )

        plotted_any = True
        plot_index += 1

        # Only draw horizontal noise ceiling anchors if evaluating noise profiles
        if not is_test_mode:
            stats = measurement.get("stats", {})
            median_V = stats.get("median_V_per_sqrtHz", None)
            floor90_V = stats.get("floor90_V_per_sqrtHz", None)

            if median_V is not None and np.isfinite(median_V) and median_V > 0:
                ax.axhline(
                    median_V,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.45,
                )

            if floor90_V is not None and np.isfinite(floor90_V) and floor90_V > 0:
                ax.axhline(
                    floor90_V,
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.65,
                )

    if not plotted_any:
        print("No enabled measurement files could be plotted.")
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude Spectral Density (V/√Hz) or Amplitude (V)")
    ax.set_title("ADC Noise Spectrum Comparison")
    ax.grid(True, which="both", alpha=0.3)

    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.show()


def plot_test_errors(files=None):
    if files is None:
        files = COMPARE_FILES if COMPARE_FILES else _discover_measurement_files()

    rows = []
    for instrument, file_config in files.items():
        if not file_config.get("enabled", True):
            continue

        filename = file_config["file"]
        try:
            with open(filename, "r") as f:
                measurement = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        if not _is_test_mode_measurement(measurement):
            continue

        results = measurement.get("test_results", None)
        if not results:
            continue

        input_freq = results.get("input_freq_Hz")
        measured_freq = results.get("measured_freq_Hz")
        freq_error_Hz = results.get("freq_error_Hz")
        freq_error_bins = results.get("freq_error_bins")
        gain_dB = results.get("gain_dB")

        if input_freq is None or gain_dB is None or freq_error_bins is None:
            continue
        if not np.isfinite(input_freq) or input_freq <= 0:
            continue

        rows.append({
            "input_freq_Hz": float(input_freq),
            "measured_freq_Hz": float(measured_freq),
            "freq_error_Hz": float(freq_error_Hz),
            "freq_error_bins": float(freq_error_bins),
            "gain_dB": float(gain_dB),
            "filename": filename,
        })

    if len(rows) == 0:
        print("No valid test-mode error data found.")
        return

    rows = sorted(rows, key=lambda r: r["input_freq_Hz"])

    input_freqs = np.array([r["input_freq_Hz"] for r in rows])
    gain_errors_dB = np.array([r["gain_dB"] for r in rows])
    freq_errors_bins = np.array([r["freq_error_bins"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xscale("log")

    line1 = ax1.plot(
        input_freqs,
        gain_errors_dB,
        marker="o",
        linewidth=1.5,
        color = "red",
        label="Gain Error (dB)",
    )[0]

    ax1.axhline(0, linestyle="-", linewidth=1.0, alpha=0.5)
    ax1.axhline(0.5, linestyle="--", linewidth=1.0, alpha=0.4)
    ax1.axhline(-0.5, linestyle="--", linewidth=1.0, alpha=0.4)

    ax1.set_ylabel("Vertical Error: Gain Error (dB)")
    ax1.set_xlabel("Input Frequency (Hz)")
    ax1.set_ylim(-2, 2)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        input_freqs,
        freq_errors_bins,
        marker="s",
        linewidth=1.5,
        color = "blue",
        label="Frequency Error (bins)",
    )[0]

    ax2.axhline(0, linestyle="-", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.5, linestyle=":", linewidth=1.0, alpha=0.4)
    ax2.axhline(-0.5, linestyle=":", linewidth=1.0, alpha=0.4)

    ax2.set_ylabel("Horizontal Error: Frequency Error (bins)")
    ax2.set_ylim(-1, 1)

    ax1.set_title("ADC Spectrometer Accuracy")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend([line1, line2], ["Gain Error (dB)", "Frequency Error (bins)"], loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_saved_measurements()
    plot_test_errors()