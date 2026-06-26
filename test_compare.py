import time
from datetime import datetime
import serial

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import blackmanharris

from stitching import stitch_latest_results
from plot_compare import (
    save_json,
    compare_saved_measurements,
    clear_saved_measurements,
    plot_test_errors,
)


#RUN_MODE = "LIVE_NOISE"      # live ADC noise, save/compare ASD only
#RUN_MODE = "SYNTH_TEST"    # synthetic sine sweep, no serial
RUN_MODE = "EXT_SINE_TEST" # real external sine into ADC, calculate accuracy

#LIVE_NOISE:
#   plot ASD V/√Hz
#   save ASD V/√Hz
#    compare noise floors
#    no gain/frequency error stats

#SYNTH_TEST:
#   generate sine in Python
#    use mag_V for gain/frequency error
#    save test_results

#EXT_SINE_TEST:
#    read real ADC blocks from external sine input
#    use mag_V for gain/frequency error
#    save test_results

TEST_MODE = RUN_MODE in ["SYNTH_TEST", "EXT_SINE_TEST"]
USE_SERIAL = RUN_MODE in ["LIVE_NOISE", "EXT_SINE_TEST"]
DO_ERROR_ANALYSIS = RUN_MODE in ["SYNTH_TEST", "EXT_SINE_TEST"]
DO_NOISE_STATS = RUN_MODE == "LIVE_NOISE"

PORT = "/dev/cu.usbmodem101"
BAUD = 2000000
SAMPLES = 2048

TEST_RAW_SAMPLE_RATE_HZ = 100_000.0
TEST_FREQS_HZ = [
    0.1, 0.2, 0.5, 0.75,
    1, 5, 7.5, 10, 25, 50,
    100, 150, 500, 750, 1000, 1500,
    5000, 7500, 10000, 25000, 40000,
]

sweep_index = 0
TEST_SIGNAL_FREQ_HZ = TEST_FREQS_HZ[sweep_index]
TEST_SIGNAL_AMP_V = 1e-3

STATS_MIN_HZ = 50
STATS_MAX_HZ = 10000

Y_MIN = 1e-15
Y_MAX = 1e0


def send_cmd(cmd):
    ser.write((cmd + "\n").encode())
    ser.flush()


def request_raw_sample_rate():
    send_cmd("RATE")

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line.startswith("RAW_SAMPLE_RATE_HZ:"):
            rate = float(line.split(":")[1])
            print(f"RAW_SAMPLE_RATE_HZ = {rate:.3f}")
            return rate


def read_one_band_block():
    data = []

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("BAND:"):
            band_name = line.split(":", 1)[1]
            break

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "START":
            break

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if line == "END":
            break

        try:
            data.append(float(line))
        except ValueError:
            pass

    return band_name, np.array(data)


def read_low_high_blocks():
    blocks = {}

    send_cmd("BLOCK")

    while len(blocks) < len(BAND_ORDER):
        band_name, data = read_one_band_block()

        if band_name in ["LOW", "MID", "HIGH"]:
            blocks[band_name] = data
            print(f"Read {band_name}: {len(data)} samples")
        else:
            print("Ignoring unknown band:", band_name)

    return blocks


def generate_test_data(samples, sample_rate_hz, signal_freq_hz, signal_amp_v):
    t = np.arange(samples) / sample_rate_hz
    return signal_amp_v * np.sin(2 * np.pi * signal_freq_hz * t)


def generate_low_high_blocks():
    blocks = {}

    for band_name in BAND_ORDER:
        band = BANDS[band_name]
        fft_sample_rate_hz = band["f_min_Hz"] * SAMPLES

        in_band = (
            TEST_SIGNAL_FREQ_HZ >= band["stitch_min"]
            and (
                band["stitch_max"] is None
                or TEST_SIGNAL_FREQ_HZ <= band["stitch_max"]
            )
        )

        if in_band:
            data = generate_test_data(
                SAMPLES,
                fft_sample_rate_hz,
                TEST_SIGNAL_FREQ_HZ,
                TEST_SIGNAL_AMP_V,
            )
        else:
            data = np.zeros(SAMPLES)

        blocks[band_name] = data

    return blocks


if RUN_MODE == "SYNTH_TEST":
    print("RUN_MODE = SYNTH_TEST - using synthetic clean sine wave, no serial connection.")
    RAW_SAMPLE_RATE_HZ = TEST_RAW_SAMPLE_RATE_HZ
    clear_saved_measurements()

elif USE_SERIAL:
    ser = serial.Serial(PORT, BAUD, timeout=0.05)
    time.sleep(2)
    print("Connected to:", ser.name)
    RAW_SAMPLE_RATE_HZ = request_raw_sample_rate()

else:
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


BANDS = {
    "LOW": {
        "f_min_Hz": 0.1,
        "stitch_min": 0.1,
        "stitch_max": (0.1*SAMPLES)/2
    },
    "MID": {
        "f_min_Hz": 1.0,
        "stitch_min": (0.1*SAMPLES)/2,
        "stitch_max": (1.0*SAMPLES)/2
    },
    "HIGH": {
        "f_min_Hz": RAW_SAMPLE_RATE_HZ/SAMPLES,
        "stitch_min": (1.0*SAMPLES)/2,
        "stitch_max": None,
    },
}

BAND_ORDER = ["LOW","MID", "HIGH"]
band_results = {}

INSTRUMENT_LABEL = RUN_MODE if TEST_MODE else "SAMD21"

PLOT_CONFIG = {
    "label": INSTRUMENT_LABEL,
    "color": "blue",
    "linestyle": "-",
    "linewidth": 0.8,
    "y_min": Y_MIN,
    "y_max": Y_MAX,
}


latest_stitched_freqs = None
latest_stitched_amps = None
latest_stitched_mags = None  # Added to track magnitude separate from ASD
pending_action = None


def on_key_press(event):
    global pending_action

    if event.key == "s":
        pending_action = "save"
    elif event.key == "p":
        pending_action = "compare"
    elif event.key == "x":
        pending_action = "clear"


plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.mpl_connect("key_press_event", on_key_press)

# Real-time plot remains focused on Spectral Density (V/√Hz)
line, = ax.plot(
    [],
    [],
    linestyle=PLOT_CONFIG["linestyle"],
    color=PLOT_CONFIG["color"],
    linewidth=PLOT_CONFIG["linewidth"],
    label="Stitched ASD",
)
if DO_NOISE_STATS:
    median_line = ax.axhline(
        y=Y_MIN,
        linestyle="--",
        color="cyan",
        linewidth=1.2,
        alpha=0.8,
        label="Median",
    )

    floor_line = ax.axhline(
        y=Y_MIN,
        linestyle=":",
        color="orange",
        linewidth=1.2,
        alpha=0.8,
        label="90% Floor",
    )

ax.set_title("ADC Noise Spectrometer" + (f"  [{RUN_MODE}]" if TEST_MODE else ""))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude Spectral Density (V/√Hz)")  # Retained ASD label
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, RAW_SAMPLE_RATE_HZ / 2)
ax.set_ylim(Y_MIN, Y_MAX)
ax.grid(True, which="both", alpha=0.3)

stats_text = ax.text(
    0.02,
    0.02,
    "",
    transform=ax.transAxes,
    fontsize=9,
    color="white",
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
)

control_text = ax.text(
    0.02,
    0.98,
    "Keys: s = save noise ASD, p = compare, x = clear saved",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax.legend(fontsize=8)
plt.tight_layout()
plt.show(block=False)


try:
    while True:

        if RUN_MODE == "SYNTH_TEST":
            blocks = generate_low_high_blocks()
        else:
            blocks = read_low_high_blocks()

        band_results_asd = {}
        band_results_mag = {}

        for BAND_NAME in BAND_ORDER:
            band = BANDS[BAND_NAME]
            data = blocks[BAND_NAME]

            if len(data) != SAMPLES:
                print(f"Warning: expected {SAMPLES}, got {len(data)}")

                if len(data) < SAMPLES:
                    continue

                data = data[:SAMPLES]

            FFT_SAMPLE_RATE_HZ = band["f_min_Hz"] * SAMPLES

            sample_interval_s = 1.0 / FFT_SAMPLE_RATE_HZ
            bin_width_Hz = FFT_SAMPLE_RATE_HZ / SAMPLES
            nyquist_Hz = FFT_SAMPLE_RATE_HZ / 2
            capture_time_s = SAMPLES / FFT_SAMPLE_RATE_HZ
            averaging_ratio = RAW_SAMPLE_RATE_HZ / FFT_SAMPLE_RATE_HZ

            print()
            print(f"Band: {BAND_NAME}")
            print(f"FFT sample rate = {FFT_SAMPLE_RATE_HZ:.3f} Hz")
            print(f"Bin width = {bin_width_Hz:.6f} Hz")
            print(f"Nyquist = {nyquist_Hz:.3f} Hz")
            print(f"Capture time = {capture_time_s:.3f} s")
            print(f"Approx averaging ratio = {averaging_ratio:.2f}")

            if BAND_NAME == "HIGH":
                print(f"Raw sample rate = {RAW_SAMPLE_RATE_HZ:.3f} Hz")
                print(f"Stitch frequency = {RAW_SAMPLE_RATE_HZ / SAMPLES:.6f} Hz")

            window = blackmanharris(SAMPLES)
            coherent_gain = np.sum(window) / SAMPLES

            enbw_Hz = (
                FFT_SAMPLE_RATE_HZ
                * np.sum(window ** 2)
                / (np.sum(window) ** 2)
            )

            freqs = np.fft.rfftfreq(SAMPLES, d=sample_interval_s)
            freqs = freqs[1:]

            data = data - np.mean(data)

            fft_result = np.fft.rfft(data * window)

            mag_V = np.abs(fft_result) / ((SAMPLES / 2) * coherent_gain)
            
            # 1. Calculate ASD for the real-time viewport
            asd_V_per_sqrtHz = mag_V / np.sqrt(enbw_Hz)
            asd_V_per_sqrtHz[asd_V_per_sqrtHz <= 0] = 1e-12
            asd_V_trimmed = asd_V_per_sqrtHz[1:]

            # 2. Keep Raw Amplitude for saving/comparison
            mag_V[mag_V <= 0] = 1e-15
            mag_V_trimmed = mag_V[1:]

            # Real-time processing dictionary (ASD)
            band_results_asd[BAND_NAME] = {
                "freqs": freqs.copy(),
                "amps_V": asd_V_trimmed.copy(),
                "mag_V": mag_V_trimmed.copy(),
                "fft_sample_rate_Hz": FFT_SAMPLE_RATE_HZ,
                "bin_width_Hz": bin_width_Hz,
                "nyquist_Hz": nyquist_Hz,
            }

            # File storage processing dictionary (Magnitude)
            band_results_mag[BAND_NAME] = {
                "freqs": freqs.copy(),
                "amps_V": mag_V_trimmed.copy(),
                "mag_V": mag_V_trimmed.copy(),
                "fft_sample_rate_Hz": FFT_SAMPLE_RATE_HZ,
                "bin_width_Hz": bin_width_Hz,
                "nyquist_Hz": nyquist_Hz,
            }

        # Stitch the ASD stream for the plot window
        stitched_freqs, stitched_amps = stitch_latest_results(
            band_results_asd,
            BANDS,
            BAND_ORDER,
        )

        # Stitch the Magnitude stream for comparison exports
        _, stitched_mags = stitch_latest_results(
            band_results_mag,
            BANDS,
            BAND_ORDER,
        )

        if stitched_freqs is None or stitched_mags is None:
            continue

        latest_stitched_freqs = stitched_freqs.copy()
        latest_stitched_amps = stitched_amps.copy()
        latest_stitched_mags = stitched_mags.copy()

        valid = (
            np.isfinite(stitched_amps)
            & np.isfinite(stitched_freqs)
            & (stitched_amps > 0)
            & (stitched_freqs > STATS_MIN_HZ)
            & (stitched_freqs < STATS_MAX_HZ)
        )
        if DO_NOISE_STATS:
            stats_amps_V = stitched_amps[valid]

            if len(stats_amps_V) > 0:
                median_V = np.median(stats_amps_V)
                floor90_V = np.percentile(stats_amps_V, 90)

                median_nV = median_V * 1e9
                floor90_nV = floor90_V * 1e9

                median_line.set_ydata([median_V, median_V])
                floor_line.set_ydata([floor90_V, floor90_V])

                median_line.set_label(f"Median = {median_nV:.2f} nV/√Hz")
                floor_line.set_label(f"90% Floor = {floor90_nV:.2f} nV/√Hz")

                print(f"Median = {median_nV:.2f} nV/√Hz")
                print(f"90% Floor = {floor90_nV:.2f} nV/√Hz")
            else:
                print("No valid stats points in selected stats range.")

        line.set_data(stitched_freqs, stitched_amps)

        valid_freqs = stitched_freqs[
            np.isfinite(stitched_freqs) & (stitched_freqs > 0)
        ]

        if len(valid_freqs) > 0:
            ax.set_xlim(np.min(valid_freqs), np.max(valid_freqs))

        ax.set_ylim(Y_MIN, Y_MAX)

        if DO_ERROR_ANALYSIS:

            peak_band = None

            for band_name in BAND_ORDER:

                band = BANDS[band_name]

                if (
                    TEST_SIGNAL_FREQ_HZ >= band["stitch_min"]
                    and (
                        band["stitch_max"] is None
                        or TEST_SIGNAL_FREQ_HZ <= band["stitch_max"]
                    )
                ):
                    peak_band = band_name
                    break

            if peak_band is None:
                peak_band = "HIGH"

            band_freqs = band_results_asd[peak_band]["freqs"]
            band_mag_V = band_results_asd[peak_band]["mag_V"]
            bin_width_Hz = band_results_asd[peak_band]["bin_width_Hz"]

            nearest_idx = np.argmin(np.abs(band_freqs - TEST_SIGNAL_FREQ_HZ))
            search_lo = max(0, nearest_idx - 5)
            search_hi = min(len(band_freqs), nearest_idx + 6)
            local_peak_idx = search_lo + np.argmax(band_mag_V[search_lo:search_hi])

            measured_freq_Hz = band_freqs[local_peak_idx]
            measured_amp_V = band_mag_V[local_peak_idx]

            freq_error_Hz = measured_freq_Hz - TEST_SIGNAL_FREQ_HZ
            freq_error_bins = freq_error_Hz / bin_width_Hz

            gain = measured_amp_V / TEST_SIGNAL_AMP_V

            if gain > 0:
                gain_dB = 20 * np.log10(gain)
            else:
                gain_dB = -np.inf

            test_results = {
                "input_freq_Hz": float(TEST_SIGNAL_FREQ_HZ),
                "measured_freq_Hz": float(measured_freq_Hz),
                "freq_error_Hz": float(freq_error_Hz),
                "freq_error_bins": float(freq_error_bins),

                "input_amp_V": float(TEST_SIGNAL_AMP_V),
                "measured_amp_V": float(measured_amp_V),
                "gain": float(gain),
                "gain_dB": float(gain_dB),
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{INSTRUMENT_LABEL}_{TEST_SIGNAL_FREQ_HZ:.3f}Hz_{timestamp}.json"

            # Save magnitude in V for sine-wave accuracy mode.
            # The test_results contain the gain/frequency error data.
            save_json(
                filename=filename,
                instrument=INSTRUMENT_LABEL,
                stitched_freqs=latest_stitched_freqs,
                stitched_amps_V=latest_stitched_mags,
                sample_rate_Hz=RAW_SAMPLE_RATE_HZ,
                samples=SAMPLES,
                plot_config=PLOT_CONFIG,
                stats_min_Hz=STATS_MIN_HZ,
                stats_max_Hz=STATS_MAX_HZ,
                test_mode=True,
                test_input_freq_Hz=TEST_SIGNAL_FREQ_HZ,
                test_results=test_results,
            )

            stats_str = (
                f"Sweep point:      {sweep_index + 1}/{len(TEST_FREQS_HZ)}\n"
                f"Input signal:     {TEST_SIGNAL_FREQ_HZ:.3f} Hz @ {TEST_SIGNAL_AMP_V*1e6:.4f} uV\n"
                f"Measured peak:    {measured_freq_Hz:.3f} Hz @ {measured_amp_V*1e6:.4f} uV  band: {peak_band}\n"
                f"FFT resolution:   {bin_width_Hz:.6f} Hz/bin\n"
                f"Freq error:       {freq_error_Hz:.6f} Hz  ({freq_error_bins:.3f} bins)\n"
                f"Gain:             {gain:.6f}  ({gain_dB:.3f} dB)"
            )

            stats_text.set_text(stats_str)
            print(stats_str)
        else:
            stats_text.set_text("")

        ax.legend(fontsize=8)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

        if pending_action == "save" and RUN_MODE == "LIVE_NOISE":
            if latest_stitched_freqs is not None and latest_stitched_amps is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{INSTRUMENT_LABEL}_{timestamp}.json"

                save_json(
                    filename=filename,
                    instrument=INSTRUMENT_LABEL,
                    stitched_freqs=latest_stitched_freqs,
                    stitched_amps_V=latest_stitched_amps,  # Save ASD for noise-floor comparison
                    sample_rate_Hz=RAW_SAMPLE_RATE_HZ,
                    samples=SAMPLES,
                    plot_config=PLOT_CONFIG,
                    stats_min_Hz=STATS_MIN_HZ,
                    stats_max_Hz=STATS_MAX_HZ,
                    test_mode=False,
                    test_input_freq_Hz=None,
                    test_results=None,
                )
            else:
                print("No stitched data available to save yet.")

            pending_action = None

        elif pending_action == "compare":
            compare_saved_measurements()
            if DO_ERROR_ANALYSIS:
                plot_test_errors()
            pending_action = None

        elif pending_action == "clear":
            clear_saved_measurements()
            pending_action = None

        if DO_ERROR_ANALYSIS:
            sweep_index += 1

            if sweep_index >= len(TEST_FREQS_HZ):

                print("Sweep complete.")

                plt.ioff()

                plot_test_errors()

                plt.show()

                break

        if DO_ERROR_ANALYSIS:
            TEST_SIGNAL_FREQ_HZ = TEST_FREQS_HZ[sweep_index]

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if USE_SERIAL:
        ser.close()
        print("Serial closed.")