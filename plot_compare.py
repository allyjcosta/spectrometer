import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style
from config import (
    INPUT_REFERRED_GAIN,
    INPUT_REFERRED_GAIN_LABEL,
    STATS_MIN_HZ,
    TEST_FREQS_HZ,
    TEST_SIGNAL_AMP_V,
    band_for_frequency,
    make_bands,
)
from FFT_analysis import calculate_test_results
from storage import discover_measurements, load_measurement


COMPARISON_COLORS = [
    "#B85C5C",  # muted red
    "#4E79A7",  # muted blue
    "#D28B45",  # muted orange
    "#5C8F68",  # muted green
    "#8E6A9E",  # muted purple
    "#8A7A45",  # muted olive
]

NOISE_STAT_RANGES_HZ = [
    ("200-2k", 200.0, 2000.0),
    ("50-2k", 50.0, 2000.0),
]

ONE_OVER_F_FIT_RANGE_HZ = (0.1, float(STATS_MIN_HZ))


def input_refer_values(values):
    return np.asarray(values, dtype=float) / INPUT_REFERRED_GAIN


def input_refer_stat_nv(value_nv):
    if value_nv is None:
        return None
    return float(value_nv) / INPUT_REFERRED_GAIN


def valid_plot_mask(freqs, values, nyquist_hz):
    return (
        np.isfinite(freqs)
        & np.isfinite(values)
        & (freqs > 0)
        & (freqs <= nyquist_hz)
        & (values > 0)
    )


def calculate_range_stats(freqs, values, ranges_hz):
    stats = {}

    for label, f_min_hz, f_max_hz in ranges_hz:
        valid = (
            np.isfinite(freqs)
            & np.isfinite(values)
            & (freqs >= f_min_hz)
            & (freqs <= f_max_hz)
            & (values > 0)
        )

        if not np.any(valid):
            stats[label] = {
                "mean_V_per_sqrtHz": None,
                "median_V_per_sqrtHz": None,
                "mean_nV_per_sqrtHz": None,
                "median_nV_per_sqrtHz": None,
            }
            continue

        range_values = values[valid]
        mean_v = float(np.mean(range_values))
        median_v = float(np.median(range_values))
        stats[label] = {
            "mean_V_per_sqrtHz": mean_v,
            "median_V_per_sqrtHz": median_v,
            "mean_nV_per_sqrtHz": mean_v * 1e9,
            "median_nV_per_sqrtHz": median_v * 1e9,
        }

    return stats


def evaluate_power_law(freqs, slope, intercept):
    freqs = np.asarray(freqs, dtype=float)
    return 10 ** (intercept + slope * np.log10(freqs))


def reference_power_law_intercept(anchor_freq_hz, anchor_value, slope):
    return float(np.log10(anchor_value) - slope * np.log10(anchor_freq_hz))


def plot_minus_one_reference(ax, freqs, values, valid, color):
    f_min_hz, f_max_hz = ONE_OVER_F_FIT_RANGE_HZ
    anchor_target_hz = 10.0
    anchor_candidates = np.where(
        valid
        & np.isfinite(freqs)
        & np.isfinite(values)
        & (freqs > 0)
        & (values > 0)
        & (freqs >= f_min_hz)
        & (freqs <= f_max_hz)
    )[0]
    if anchor_candidates.size == 0:
        return

    anchor_index = anchor_candidates[
        np.argmin(np.abs(np.log10(freqs[anchor_candidates]) - np.log10(anchor_target_hz)))
    ]
    anchor_freq_hz = float(freqs[anchor_index])
    anchor_value = float(values[anchor_index])
    reference_intercept = reference_power_law_intercept(
        anchor_freq_hz=anchor_freq_hz,
        anchor_value=anchor_value,
        slope=-1.0,
    )
    reference_freqs = np.geomspace(f_min_hz, f_max_hz, 200)

    ax.plot(
        reference_freqs,
        evaluate_power_law(reference_freqs, -1.0, reference_intercept),
        color=color,
        linewidth=0.9,
        alpha=0.6,
        linestyle="--",
        rasterized=True,
    )


def format_range_stats(range_stats):
    parts = []
    for label, _f_min_hz, _f_max_hz in NOISE_STAT_RANGES_HZ:
        stats = range_stats[label]
        median_nv = stats["median_nV_per_sqrtHz"]
        mean_nv = stats["mean_nV_per_sqrtHz"]
        if median_nv is None or mean_nv is None:
            parts.append(f"{label}: N/A")
        else:
            parts.append(f"{label}: med {median_nv:.0f}, mean {mean_nv:.0f}")

    return "\n".join(parts)


def format_input_referred_summary(instrument, mean_nv, median_nv, range_stats):
    parts = [instrument]
    if mean_nv is None:
        parts.append("Mean: N/A")
    else:
        parts.append(f"Mean: {mean_nv:.2f} nV/√Hz")

    if median_nv is not None:
        parts.append(f"Median: {median_nv:.2f} nV/√Hz")

    parts.append(format_range_stats(range_stats))
    parts.append(f"Gain label: {INPUT_REFERRED_GAIN_LABEL}")
    return "\n".join(parts)


def _plot_saved_measurement_grid(measurements, value_key, ylabel, figure_title):
    apply_plot_style()
    measurement_count = len(measurements)
    columns = 1
    rows = measurement_count
    fig, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(15, max(4.0, 1.9 * rows)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid.ravel()
    plotted_frequencies = []

    for index, (ax, measurement) in enumerate(zip(axes, measurements)):
        freqs = np.asarray(measurement["freqs_Hz"], dtype=float)
        values = input_refer_values(measurement[value_key])
        sample_rate_hz = measurement.get("sample_rate_Hz")
        nyquist_hz = (
            float(sample_rate_hz) / 2.0
            if sample_rate_hz is not None
            else np.inf
        )
        valid = valid_plot_mask(freqs, values, nyquist_hz)

        if np.any(valid):
            plotted_frequencies.append(freqs[valid])
            ax.plot(
                freqs[valid],
                values[valid],
                color=COMPARISON_COLORS[index % len(COMPARISON_COLORS)],
                linewidth=0.8,
                rasterized=True,
            )

        instrument = measurement.get("instrument", "Unknown instrument")
        if value_key == "asd_V_per_sqrtHz":
            stats = measurement.get("stats") or {}
            mean_nv = input_refer_stat_nv(stats.get("mean_nV_per_sqrtHz"))
            median_nv = input_refer_stat_nv(stats.get("median_nV_per_sqrtHz"))
            range_stats = calculate_range_stats(
                freqs=freqs,
                values=values,
                ranges_hz=NOISE_STAT_RANGES_HZ,
            )
            plot_minus_one_reference(
                ax=ax,
                freqs=freqs,
                values=values,
                valid=valid,
                color=COMPARISON_COLORS[index % len(COMPARISON_COLORS)],
            )

            statistic_label = (
                "Mean: N/A"
                if mean_nv is None
                else f"Mean: {mean_nv:.2f} nV/√Hz"
            )
            if median_nv is not None:
                statistic_label += f"\nMedian: {median_nv:.2f} nV/√Hz"
            statistic_label += f"\n{format_range_stats(range_stats)}"
            statistic_label += f"\nInput gain: {INPUT_REFERRED_GAIN_LABEL}"
        else:
            peak_v = float(np.max(values[valid])) if np.any(valid) else None
            statistic_label = (
                "Peak: N/A"
                if peak_v is None
                else f"Peak: {peak_v * 1e3:.3f} mV"
            )

        ax.text(
            1.01,
            0.50,
            f"{index + 1}. {instrument} — {statistic_label}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.82),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=12)

    for ax in axes[measurement_count:]:
        ax.set_visible(False)

    if plotted_frequencies:
        all_frequencies = np.concatenate(plotted_frequencies)
        axes[0].set_xlim(np.min(all_frequencies), np.max(all_frequencies))

    fig.supxlabel("Frequency (Hz)")
    fig.supylabel(ylabel)
    fig.suptitle(figure_title)
    return fig


def plot_saved_measurements_stacked(files=None):
    if files is None:
        files = discover_measurements()

    if not files:
        print("No saved measurements found.")
        return

    measurements = [load_measurement(filename) for filename in files]
    _plot_saved_measurement_grid(
        measurements=measurements,
        value_key="asd_V_per_sqrtHz",
        ylabel="Input-Referred ASD (V/√Hz)",
        figure_title="Saved Measurements — Input-Referred ASD Small Multiples",
    )
    _plot_saved_measurement_grid(
        measurements=measurements,
        value_key="mag_V",
        ylabel="Input-Referred Amplitude (V)",
        figure_title="Saved Measurements — Input-Referred Amplitude Small Multiples",
    )
    plt.show()


def compare_saved_measurements(files=None):
    apply_plot_style()
    if files is None:
        files = discover_measurements()

    if not files:
        print("No saved measurements found.")
        return

    # Create a layout with a dedicated structural column on the right for the legend
    fig, ax_dict = plt.subplot_mosaic(
        [["asd", "legend"],
         ["mag", "legend"]],
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1, 0.42], "wspace": 0.04},
        constrained_layout=True,
    )
    
    ax_asd = ax_dict["asd"]
    ax_mag = ax_dict["mag"]
    ax_leg = ax_dict["legend"]

    # Use a high-contrast, colorblind-friendly order for saved instruments.
    ax_asd.set_prop_cycle(color=COMPARISON_COLORS)
    
    # Cleanly hide the ticks and spine lines for the legend's panel
    ax_leg.axis("off")

    handles = []
    labels = []
    plotted_frequencies = []

    for filename in files:
        measurement = load_measurement(filename)

        freqs = np.asarray(measurement["freqs_Hz"], dtype=float)
        asd = input_refer_values(measurement["asd_V_per_sqrtHz"])
        mag = input_refer_values(measurement["mag_V"])
        sample_rate_hz = measurement.get("sample_rate_Hz")
        nyquist_hz = (
            float(sample_rate_hz) / 2.0
            if sample_rate_hz is not None
            else np.inf
        )

        valid_asd = valid_plot_mask(freqs, asd, nyquist_hz)
        valid_mag = valid_plot_mask(freqs, mag, nyquist_hz)

        valid_frequency = valid_asd | valid_mag
        if np.any(valid_frequency):
            plotted_frequencies.append(freqs[valid_frequency])

        instrument = measurement.get("instrument", "Unknown instrument")
        stats = measurement.get("stats") or {}
        mean_nv = input_refer_stat_nv(stats.get("mean_nV_per_sqrtHz"))
        median_nv = input_refer_stat_nv(stats.get("median_nV_per_sqrtHz"))
        range_stats = calculate_range_stats(
            freqs=freqs,
            values=asd,
            ranges_hz=NOISE_STAT_RANGES_HZ,
        )
        label = format_input_referred_summary(
            instrument=instrument,
            mean_nv=mean_nv,
            median_nv=median_nv,
            range_stats=range_stats,
        )

        line, = ax_asd.plot(
            freqs[valid_asd],
            asd[valid_asd],
            linewidth=0.8,
            alpha=0.78,
            label=label,
        )

        ax_mag.plot(
            freqs[valid_mag],
            mag[valid_mag],
            linewidth=0.8,
            alpha=0.78,
            color=line.get_color(),
        )

        plot_minus_one_reference(
            ax=ax_asd,
            freqs=freqs,
            values=asd,
            valid=valid_asd,
            color=line.get_color(),
        )

        handles.append(line)
        labels.append(label)

    # LINK AXES SAFELY HERE
    ax_asd.sharex(ax_mag)
    plt.setp(ax_asd.get_xticklabels(), visible=False)

    ax_asd.set_xscale("log")
    ax_asd.set_yscale("log")
    ax_asd.set_ylabel("Input-Referred ASD (V/√Hz)")
    ax_asd.set_title("Saved Measurement Comparison — Input-Referred")
    ax_asd.grid(True, which="both", alpha=0.3)

    ax_mag.set_xscale("log")
    ax_mag.set_yscale("log")
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Input-Referred Amplitude (V)")
    ax_mag.grid(True, which="both", alpha=0.3)

    if plotted_frequencies:
        all_plotted_frequencies = np.concatenate(plotted_frequencies)
        ax_mag.set_xlim(
            np.min(all_plotted_frequencies),
            np.max(all_plotted_frequencies),
        )

    # Render the legend inside its safe side panel
    if handles:
        ax_leg.legend(
            handles,
            labels,
            fontsize=8.5,
            loc="center left",
            frameon=True,
            borderpad=0.55,
            labelspacing=0.8,
            handlelength=2.2,
        )

    plt.show()


def plot_test_errors(files=None):
    apply_plot_style()
    if files is None:
        files = discover_measurements()

    rows = []
    live_test_index = 0
    for filename in files:
        measurement = load_measurement(filename)
        measurement_type = measurement.get("measurement_type")
        old_test_mode = bool(measurement.get("test_mode", False))
        live_test_mode = measurement_type in ("live", "live_test")
        new_test_mode = measurement_type == "synthetic_test" or live_test_mode

        if not old_test_mode and not new_test_mode:
            continue

        if live_test_mode:
            if live_test_index >= len(TEST_FREQS_HZ):
                continue

            target_freq_hz = TEST_FREQS_HZ[live_test_index]
            live_test_index += 1
            sample_rate_hz = measurement.get("sample_rate_Hz")
            if sample_rate_hz is None:
                continue

            bands = make_bands(sample_rate_hz)
            active_band_name = band_for_frequency(bands, target_freq_hz)
            if active_band_name is None:
                continue

            test = calculate_test_results(
                freqs_hz=measurement["freqs_Hz"],
                mag_v=measurement["mag_V"],
                target_freq_hz=target_freq_hz,
                target_amp_v=TEST_SIGNAL_AMP_V,
                fmin_hz=bands[active_band_name]["f_min_Hz"],
                use_global_peak=True,
            )
        else:
            test = measurement.get("test") or measurement.get("test_results")

        if not test:
            continue

        input_freq_hz = (
            test.get("input_freq_Hz")
            or measurement.get("test_input_freq_Hz")
            or test.get("target_freq_Hz")
        )
        measured_freq_hz = test.get("measured_freq_Hz")
        freq_error_hz = test.get("freq_error_Hz")
        freq_error_bins = test.get("freq_error_bins")
        gain_db = test.get("gain_dB")
        leakage_db = test.get("spectral_leakage_dB")

        if any(v is None for v in [input_freq_hz, measured_freq_hz, freq_error_hz, freq_error_bins, gain_db]):
            continue

        rows.append({
            "input_freq_Hz": float(input_freq_hz),
            "measured_freq_Hz": float(measured_freq_hz),
            "freq_error_Hz": float(freq_error_hz),
            "freq_error_bins": float(freq_error_bins),
            "gain_dB": float(gain_db),
            "spectral_leakage_dB": (
                None if leakage_db is None else float(leakage_db)
            ),
        })

    if len(rows) == 0:
        print("No valid test error data found.")
        return

    rows = sorted(rows, key=lambda r: r["input_freq_Hz"])

    input_freqs = np.array([r["input_freq_Hz"] for r in rows])
    gain_errors_db = np.array([r["gain_dB"] for r in rows])
    freq_errors_bins = np.array([r["freq_error_bins"] for r in rows])

    gain_rms_db = np.sqrt(np.mean(gain_errors_db ** 2))
    gain_median_abs_db = np.median(np.abs(gain_errors_db))
    gain_max_abs_db = np.max(np.abs(gain_errors_db))

    freq_rms_bins = np.sqrt(np.mean(freq_errors_bins ** 2))
    freq_median_abs_bins = np.median(np.abs(freq_errors_bins))
    freq_max_abs_bins = np.max(np.abs(freq_errors_bins))

    leakage_values_db = np.array([
        row["spectral_leakage_dB"]
        for row in rows
        if row["spectral_leakage_dB"] is not None
    ])

    # Widen layout space for the sidebar text
    fig, ax_dict = plt.subplot_mosaic(
        [["gain_plot", "summary_panel"],
         ["freq_plot", "points_panel"]],
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [1, 0.62], "wspace": 0.25, "hspace": 0.22}
    )

    ax_gain = ax_dict["gain_plot"]
    ax_freq = ax_dict["freq_plot"]
    ax_sum = ax_dict["summary_panel"]
    ax_pts = ax_dict["points_panel"]

    ax_sum.axis("off")
    ax_pts.axis("off")

    # --- TOP PLOT: GAIN ERROR ---
    ax_gain.set_xscale("log")
    ax_gain.plot(input_freqs, gain_errors_db, marker="o", color="crimson", linestyle="none")
    ax_gain.axhline(0, color="black", linewidth=1.2)
    ax_gain.axhline(0.5, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax_gain.axhline(-0.5, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax_gain.set_ylabel("Gain Error (dB)", color="crimson")
    ax_gain.tick_params(axis='y', labelcolor="crimson")
    
    max_gain_abs = max(np.max(np.abs(gain_errors_db)), 0.5)
    ax_gain.set_ylim(-max_gain_abs * 1.2, max_gain_abs * 1.2)
    ax_gain.set_title(
        "Stitched Multi-Band Spectrometer Accuracy Profile",
        fontweight='bold',
        pad=10,
    )
    ax_gain.grid(True, which="both", alpha=0.25)
    plt.setp(ax_gain.get_xticklabels(), visible=False)

    # --- BOTTOM PLOT: FREQUENCY ERROR ---
    ax_freq.set_xscale("log")
    ax_freq.plot(input_freqs, freq_errors_bins, marker="D", color="royalblue", linestyle="none", markersize=5)
    ax_freq.axhline(0, color="black", linewidth=1.2)
    ax_freq.axhline(0.5, linestyle=":", linewidth=0.8, color="gray", alpha=0.5)
    ax_freq.axhline(-0.5, linestyle=":", linewidth=0.8, color="gray", alpha=0.5)
    ax_freq.set_xlabel("Test Tone Input Frequency (Hz)")
    ax_freq.set_ylabel("Frequency Error (FFT Bins)", color="royalblue")
    ax_freq.tick_params(axis='y', labelcolor="royalblue")
    
    max_freq_abs = max(np.max(np.abs(freq_errors_bins)), 1.0)
    ax_freq.set_ylim(-max_freq_abs * 1.2, max_freq_abs * 1.2)
    ax_freq.grid(True, which="both", alpha=0.25)

    ax_gain.sharex(ax_freq)

    # --- UPPER TABLE: SYSTEM SUMMARY STATS ---
    summary_content = [
        ["Total Swept Points", f"{len(rows)}"],
        ["Gain Error RMS", f"{gain_rms_db:.3f} dB"],
        ["Gain Median |Err|", f"{gain_median_abs_db:.3f} dB"],
        ["Gain Max |Err|", f"{gain_max_abs_db:.3f} dB"],
        ["Freq Error RMS", f"{freq_rms_bins:.3f} bins"],
        ["Freq Median |Err|", f"{freq_median_abs_bins:.3f} bins"],
        ["Freq Max |Err|", f"{freq_max_abs_bins:.3f} bins"]
    ]

    if len(leakage_values_db) > 0:
        summary_content.extend([
            ["Leakage Median", f"{np.median(leakage_values_db):.3f} dB"],
            ["Leakage Worst", f"{np.max(leakage_values_db):.3f} dB"],
        ])

    summary_table = ax_sum.table(
        cellText=summary_content,
        colLabels=["System Metric", "Performance Value"],
        loc="center",
        cellLoc="left"
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(12)
    summary_table.scale(1.0, 1.3)

    # --- LOWER TABLE: INDIVIDUAL SWEEP POINTS ---
    points_headers = [
        "Target\nFrequency",
        "Measured\nFrequency",
        "Gain\nError",
        "Frequency\nError",
        "Leakage",
    ]
    points_content = []
    

    display_rows = rows


    for r in display_rows:
        leakage_db = r["spectral_leakage_dB"]
        points_content.append([
            f"{r['input_freq_Hz']:.2f} Hz",
            f"{r['measured_freq_Hz']:.2f} Hz",
            f"{r['gain_dB']:.3f} dB",
            f"{r['freq_error_bins']:.3f} bins",
            "N/A" if leakage_db is None else f"{leakage_db:.3f} dB",
        ])

    points_table = ax_pts.table(
        cellText=points_content,
        colLabels=points_headers,
        loc="center",
        cellLoc="left"
    )
    points_table.auto_set_font_size(False)
    points_table.set_fontsize(11)
    points_table.scale(1.0, 1.08)

    # Manual cell width expansion & formatting overrides to prevent cutoffs
    for (row, col), cell in summary_table.get_celld().items():
        cell.set_width(0.52) 
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')

    for (row, col), cell in points_table.get_celld().items():
        cell.set_width(0.20)
        if row == 0:
            cell.set_height(cell.get_height() * 1.55)
            cell.set_text_props(weight='bold', color='white', fontsize=10)
            cell.set_facecolor('#34495E')

    plt.show()

def plot_thd2(files=None):
    apply_plot_style()
    if files is None:
        files = discover_measurements()

    rows = []
    live_test_index = 0

    for filename in files:
        measurement = load_measurement(filename)
        measurement_type = measurement.get("measurement_type")

        if measurement_type in ("live", "live_test"):
            if live_test_index >= len(TEST_FREQS_HZ):
                continue

            target_freq_hz = TEST_FREQS_HZ[live_test_index]
            live_test_index += 1
            sample_rate_hz = measurement.get("sample_rate_Hz")
            if sample_rate_hz is None:
                continue

            bands = make_bands(sample_rate_hz)
            active_band_name = band_for_frequency(bands, target_freq_hz)
            if active_band_name is None:
                continue

            test = calculate_test_results(
                freqs_hz=measurement["freqs_Hz"],
                mag_v=measurement["mag_V"],
                target_freq_hz=target_freq_hz,
                target_amp_v=TEST_SIGNAL_AMP_V,
                fmin_hz=bands[active_band_name]["f_min_Hz"],
                use_global_peak=False,
            )
        elif measurement_type == "synthetic_test":
            test = measurement.get("test")
        else:
            continue

        if test is None:
            continue

        target_freq_hz = test.get("target_freq_Hz")
        thd2_percent = test.get("thd2_percent")

        if target_freq_hz is None or thd2_percent is None:
            continue

        rows.append({
            "target_freq_Hz": float(target_freq_hz),
            "thd2_percent": float(thd2_percent),
        })

    if len(rows) == 0:
        print("No valid THD2 data found.")
        return

    rows = sorted(rows, key=lambda r: r["target_freq_Hz"])

    target_freqs = np.array([r["target_freq_Hz"] for r in rows])
    thd2 = np.array([r["thd2_percent"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(
        target_freqs,
        thd2,
        marker="o",
        linewidth=1.5,
        label="THD2 (%)",
    )

    ax.set_xlabel("Target Frequency (Hz)")
    ax.set_ylabel("THD2 (%)")
    ax.set_title("Second Harmonic Distortion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_amplitude_comparison(files=None):
    apply_plot_style()
    if files is None:
        files = discover_measurements()

    rows = []
    live_test_index = 0

    for filename in files:
        measurement = load_measurement(filename)

        measurement_type = measurement.get("measurement_type")

        if measurement_type in ("live", "live_test"):
            if live_test_index >= len(TEST_FREQS_HZ):
                continue

            target_freq_hz = TEST_FREQS_HZ[live_test_index]
            live_test_index += 1
            sample_rate_hz = measurement.get("sample_rate_Hz")
            if sample_rate_hz is None:
                continue

            bands = make_bands(sample_rate_hz)
            active_band_name = band_for_frequency(bands, target_freq_hz)
            if active_band_name is None:
                continue

            test = calculate_test_results(
                freqs_hz=measurement["freqs_Hz"],
                mag_v=measurement["mag_V"],
                target_freq_hz=target_freq_hz,
                target_amp_v=TEST_SIGNAL_AMP_V,
                fmin_hz=bands[active_band_name]["f_min_Hz"],
                use_global_peak=True,
            )
            measurement_type = "live_test"
        elif measurement_type == "synthetic_test":
            test = measurement.get("test")
        else:
            continue

        if test is None:
            continue

        target_freq_hz = test.get("target_freq_Hz")
        target_amp_v = test.get("target_amp_V")
        measured_amp_v = test.get("measured_amp_V")

        if (
            target_freq_hz is None
            or target_amp_v is None
            or measured_amp_v is None
        ):
            continue

        rows.append({
            "measurement_type": measurement_type,
            "target_freq_Hz": float(target_freq_hz),
            "target_amp_V": float(target_amp_v),
            "measured_amp_V": float(measured_amp_v),
        })

    if len(rows) == 0:
        print("No valid amplitude comparison data found.")
        return

    rows = sorted(rows, key=lambda r: (r["measurement_type"], r["target_freq_Hz"]))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    for measurement_type in ("synthetic_test", "live_test"):
        subset = [r for r in rows if r["measurement_type"] == measurement_type]

        if not subset:
            continue

        target_freqs = np.array([r["target_freq_Hz"] for r in subset])
        measured_amps = np.array([r["measured_amp_V"] for r in subset])
        target_amps = np.array([r["target_amp_V"] for r in subset])

        label = measurement_type.replace("_", " ")

        ax1.plot(
            target_freqs,
            measured_amps,
            marker="o",
            linestyle = None,
            linewidth=1.5,
            color = "green",
            label=f"{label} measured",
        )

        ax2.plot(
            target_freqs,
            20 * np.log10(measured_amps / target_amps),
            marker="o",
            linestyle = None,
            linewidth=1.5,
            color = "green",
            label=f"{label} gain error",
        )

    # Ideal amplitude line
    all_freqs = np.array([r["target_freq_Hz"] for r in rows])
    all_targets = np.array([r["target_amp_V"] for r in rows])

    order = np.argsort(all_freqs)

    ax1.plot(
        all_freqs[order],
        all_targets[order],
        linestyle="--",
        linewidth=1.2,
        color = "lime",
        label="ideal target",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Measured Amplitude (V)")
    ax1.set_title("Amplitude Comparison")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.axhline(0, linewidth=1.0, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Target Frequency (Hz)")
    ax2.set_ylabel("Gain Error (dB)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
