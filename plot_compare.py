import matplotlib.pyplot as plt
import numpy as np

from storage import discover_measurements, load_measurement


def compare_saved_measurements(files=None):
    if files is None:
        files = discover_measurements()

    if not files:
        print("No saved measurements found.")
        return

    # Create a layout with a dedicated structural column on the right for the legend
    fig, ax_dict = plt.subplot_mosaic(
        [["asd", "legend"],
         ["mag", "legend"]],
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [1, 0.16], "wspace": 0.05}
    )
    
    ax_asd = ax_dict["asd"]
    ax_mag = ax_dict["mag"]
    ax_leg = ax_dict["legend"]
    
    # Cleanly hide the ticks and spine lines for the legend's panel
    ax_leg.axis("off")

    handles = []
    labels = []

    for filename in files:
        measurement = load_measurement(filename)

        freqs = np.asarray(measurement["freqs_Hz"], dtype=float)
        asd = np.asarray(measurement["asd_V_per_sqrtHz"], dtype=float)
        mag = np.asarray(measurement["mag_V"], dtype=float)

        valid_asd = (
            np.isfinite(freqs)
            & np.isfinite(asd)
            & (freqs > 0)
            & (asd > 0)
        )

        valid_mag = (
            np.isfinite(freqs)
            & np.isfinite(mag)
            & (freqs > 0)
            & (mag > 0)
        )

        label = measurement.get("instrument", filename)

        test = measurement.get("test")
        if test is None:
            test = measurement.get("test_results")

        if test is not None:
            input_freq = (
                test.get("input_freq_Hz")
                or measurement.get("test_input_freq_Hz")
                or test.get("target_freq_Hz")
            )
            if input_freq is not None:
                label = f"{float(input_freq):.3f} Hz"

        line, = ax_asd.plot(
            freqs[valid_asd],
            asd[valid_asd],
            linewidth=0.8,
            label=label,
        )

        ax_mag.plot(
            freqs[valid_mag],
            mag[valid_mag],
            linewidth=0.8,
            color=line.get_color(),
        )

        handles.append(line)
        labels.append(label)

    # LINK AXES SAFELY HERE
    ax_asd.sharex(ax_mag)
    plt.setp(ax_asd.get_xticklabels(), visible=False)

    ax_asd.set_xscale("log")
    ax_asd.set_yscale("log")
    ax_asd.set_ylabel("ASD (V/√Hz)")
    ax_asd.set_title("Saved Measurement Comparison")
    ax_asd.grid(True, which="both", alpha=0.3)

    ax_mag.set_xscale("log")
    ax_mag.set_yscale("log")
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Amplitude (V)")
    ax_mag.grid(True, which="both", alpha=0.3)

    # Render the legend inside its safe side panel
    if handles:
        ax_leg.legend(
            handles,
            labels,
            fontsize=8,
            loc="center left",
            frameon=True,
        )

    plt.tight_layout()
    plt.show()


def plot_test_errors(files=None):
    if files is None:
        files = discover_measurements()

    rows = []
    for filename in files:
        measurement = load_measurement(filename)
        measurement_type = measurement.get("measurement_type")
        old_test_mode = bool(measurement.get("test_mode", False))
        new_test_mode = measurement_type in ("synthetic_test", "hardware_test")

        if not old_test_mode and not new_test_mode:
            continue

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

        if any(v is None for v in [input_freq_hz, measured_freq_hz, freq_error_hz, freq_error_bins, gain_db]):
            continue

        rows.append({
            "input_freq_Hz": float(input_freq_hz),
            "measured_freq_Hz": float(measured_freq_hz),
            "freq_error_Hz": float(freq_error_hz),
            "freq_error_bins": float(freq_error_bins),
            "gain_dB": float(gain_db),
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

    # Widen layout space for the sidebar text
    fig, ax_dict = plt.subplot_mosaic(
        [["gain_plot", "summary_panel"],
         ["freq_plot", "points_panel"]],
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [1, 0.52], "wspace": 0.25, "hspace": 0.22}
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
    ax_gain.set_title("Stitched Multi-Band Spectrometer Accuracy Profile", fontsize=12, fontweight='bold', pad=10)
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

    summary_table = ax_sum.table(
        cellText=summary_content,
        colLabels=["System Metric", "Performance Value"],
        loc="center",
        cellLoc="left"
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(8.5)
    summary_table.scale(1.0, 1.3)

    # --- LOWER TABLE: INDIVIDUAL SWEEP POINTS ---
    points_headers = ["Target Freq", "Measured Freq", "Gain Error", "Freq Error"]
    points_content = []
    

    display_rows = rows


    for r in display_rows:
        points_content.append([
            f"{r['input_freq_Hz']:.2f} Hz",
            f"{r['measured_freq_Hz']:.2f} Hz",
            f"{r['gain_dB']:.3f} dB",
            f"{r['freq_error_bins']:.3f} bins"
        ])

    points_table = ax_pts.table(
        cellText=points_content,
        colLabels=points_headers,
        loc="center",
        cellLoc="left"
    )
    points_table.auto_set_font_size(False)
    points_table.set_fontsize(8)
    points_table.scale(1.0, 1.1)

    # Manual cell width expansion & formatting overrides to prevent cutoffs
    for (row, col), cell in summary_table.get_celld().items():
        cell.set_width(0.52) 
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')

    for (row, col), cell in points_table.get_celld().items():
        cell.set_width(0.28) # Added width column allocation space
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=7.5)
            cell.set_facecolor('#34495E')

    plt.show()

def plot_thd2(files=None):
    if files is None:
        files = discover_measurements()

    rows = []

    for filename in files:
        measurement = load_measurement(filename)

        if measurement.get("measurement_type") != "hardware_test":
            continue

        test = measurement.get("test")

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
        print("No valid hardware THD2 data found.")
        return

    rows = sorted(rows, key=lambda r: r["target_freq_Hz"])

    target_freqs = np.array([r["target_freq_Hz"] for r in rows])
    thd2 = np.array([r["thd2_percent"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xscale("log")

    ax.plot(
        target_freqs,
        thd2,
        marker="o",
        linewidth=1.5,
        label="THD2 (%)",
    )

    ax.set_xlabel("Target Frequency (Hz)")
    ax.set_ylabel("THD2 (%)")
    ax.set_title("Hardware Second Harmonic Distortion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_amplitude_comparison(files=None):
    if files is None:
        files = discover_measurements()

    rows = []

    for filename in files:
        measurement = load_measurement(filename)

        measurement_type = measurement.get("measurement_type")

        if measurement_type not in ("synthetic_test", "hardware_test"):
            continue

        test = measurement.get("test")

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

    for measurement_type in ("synthetic_test", "hardware_test"):
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
            linewidth=1.5,
            label=f"{label} measured",
        )

        ax2.plot(
            target_freqs,
            20 * np.log10(measured_amps / target_amps),
            marker="o",
            linewidth=1.5,
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
        label="ideal target",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Measured Amplitude (V)")
    ax1.set_title("Amplitude Comparison")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(0, linewidth=1.0, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Target Frequency (Hz)")
    ax2.set_ylabel("Gain Error (dB)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()