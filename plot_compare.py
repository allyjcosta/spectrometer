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

        test = measurement.get("test")
        if test is None:
            test = measurement.get("test_results")

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

        if (
            input_freq_hz is None
            or measured_freq_hz is None
            or freq_error_hz is None
            or freq_error_bins is None
            or gain_db is None
        ):
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

    stats_text = (
        f"Test points: {len(rows)}\n"
        f"Gain error RMS: {gain_rms_db:.3f} dB\n"
        f"Gain error median |error|: {gain_median_abs_db:.3f} dB\n"
        f"Gain error max |error|: {gain_max_abs_db:.3f} dB\n"
        f"Freq error RMS: {freq_rms_bins:.3f} bins\n"
        f"Freq error median |error|: {freq_median_abs_bins:.3f} bins\n"
        f"Freq error max |error|: {freq_max_abs_bins:.3f} bins"
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xscale("log")

    # Changed: Added linestyle="none" to disconnect points
    line1 = ax1.plot(
        input_freqs,
        gain_errors_db,
        marker="o",
        color="red",
        linestyle="none",
        label="Gain Error (dB)",
    )[0]

    ax1.axhline(0, color="black", linewidth=1.5)
    ax1.axhline(0.5, linestyle="--", linewidth=1.0, alpha=0.4)
    ax1.axhline(-0.5, linestyle="--", linewidth=1.0, alpha=0.4)

    ax1.set_xlabel("Input Frequency (Hz)")
    

    ax1.set_ylabel("Vertical Error: Gain Error (dB)", color="red")

    

    ax1.tick_params(axis='y', labelcolor="red", color="red")
    max_gain_abs = np.max(np.abs(gain_errors_db))
    gain_limit = max(max_gain_abs * 1.15, 0.5)
    ax1.set_ylim(-gain_limit, gain_limit)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        input_freqs,
        freq_errors_bins,
        marker="s",
        color="blue",
        linestyle="none",
        label="Frequency Error (FFT bins)",
    )[0]

    ax2.axhline(0, color="black", linewidth=1.5)
    ax2.axhline(0.5, linestyle=":", linewidth=1.0, alpha=0.4)
    ax2.axhline(-0.5, linestyle=":", linewidth=1.0, alpha=0.4)

    ax2.set_ylabel("Horizontal Error: Frequency Error (FFT bins)", color="blue")

    max_freq_abs = np.max(np.abs(freq_errors_bins))
    freq_limit = max(max_freq_abs * 1.15, 1.0)
    ax2.set_ylim(-freq_limit, freq_limit)
    
    ax2.tick_params(axis='y', labelcolor="blue", color="blue")

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax1.set_title("ADC Spectrometer Accuracy")
    ax1.grid(True, which="both", alpha=0.3)

    ax1.legend(
        [line1, line2],
        ["Gain Error (dB)", "Frequency Error (FFT bins)"],
        loc="upper right", # Shifted slightly so it won't crash into the floating text box
    )

    plt.tight_layout()
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