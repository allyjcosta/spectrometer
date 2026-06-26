import matplotlib.pyplot as plt
import numpy as np

from storage import discover_measurements, load_measurement


def compare_saved_measurements(files=None):
    if files is None:
        files = discover_measurements()

    if not files:
        print("No saved measurements found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for filename in files:
        measurement = load_measurement(filename)

        freqs = np.asarray(measurement["freqs_Hz"], dtype=float)
        asd = np.asarray(measurement["asd_V_per_sqrtHz"], dtype=float)

        valid = (
            np.isfinite(freqs)
            & np.isfinite(asd)
            & (freqs > 0)
            & (asd > 0)
        )

        label = measurement.get("instrument", filename)

        test = measurement.get("test")
        if test is not None:
            label = f"{test['target_freq_Hz']:.3f} Hz"

        ax.plot(freqs[valid], asd[valid], linewidth=0.8, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ASD (V/√Hz)")
    ax.set_title("Saved Measurement Comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_test_errors(files=None):
    if files is None:
        files = discover_measurements()

    rows = []

    for filename in files:

        measurement = load_measurement(filename)

        measurement_type = measurement.get("measurement_type")

        if measurement_type not in (
            "synthetic_test",
            "hardware_test",
        ):
            continue

        test = measurement.get("test")

        if test is None:
            continue

        target_freq_hz = test.get("target_freq_Hz")
        measured_freq_hz = test.get("measured_freq_Hz")
        freq_error_hz = test.get("freq_error_Hz")
        freq_error_bins = test.get("freq_error_bins")
        gain_db = test.get("gain_dB")

        if target_freq_hz is None:
            continue

        if gain_db is None or freq_error_bins is None:
            continue

        rows.append({
            "target_freq_Hz": float(target_freq_hz),
            "measured_freq_Hz": float(measured_freq_hz),
            "freq_error_Hz": float(freq_error_hz),
            "freq_error_bins": float(freq_error_bins),
            "gain_dB": float(gain_db),
            "filename": filename,
        })

    if len(rows) == 0:
        print("No valid test error data found.")
        return

    rows = sorted(rows, key=lambda r: r["target_freq_Hz"])

    target_freqs = np.array([r["target_freq_Hz"] for r in rows])
    gain_errors_db = np.array([r["gain_dB"] for r in rows])
    freq_errors_bins = np.array([r["freq_error_bins"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xscale("log")

    line1 = ax1.plot(
        target_freqs,
        gain_errors_db,
        marker="o",
        color ="red",
        linewidth=1.5,
        label="Gain Error (dB)",
    )[0]

    ax1.axhline(0, linestyle="-", linewidth=1.0, alpha=0.6)
    ax1.axhline(0.5, linestyle="--", linewidth=1.0, alpha=0.4)
    ax1.axhline(-0.5, linestyle="--", linewidth=1.0, alpha=0.4)

    ax1.set_xlabel("Target Frequency (Hz)")
    ax1.set_ylabel("Vertical Error: Gain Error (dB)")
    ax1.set_ylim(-2, 2)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        target_freqs,
        freq_errors_bins,
        marker="s",
        color ="red",
        linewidth=1.5,
        label="Frequency Error (bins)",
    )[0]

    ax2.axhline(0, linestyle="-", linewidth=1.0, alpha=0.6)
    ax2.axhline(0.5, linestyle=":", linewidth=1.0, alpha=0.4)
    ax2.axhline(-0.5, linestyle=":", linewidth=1.0, alpha=0.4)

    ax2.set_ylabel("Horizontal Error: Frequency Error (bins)")
    ax2.set_ylim(-1, 1)

    ax1.set_title("ADC Spectrometer Accuracy")
    ax1.grid(True, which="both", alpha=0.3)

    ax1.legend(
        [line1, line2],
        ["Gain Error (dB)", "Frequency Error (bins)"],
        loc="best",
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