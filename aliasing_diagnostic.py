"""Display each acquired band before stitching to diagnose aliasing."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import BAUD, PORT, SAMPLES, active_band_order, make_bands
from FFT_analysis import compute_fft_analysis
from serial_io import open_serial, request_raw_sample_rate, send_cmd


DIAGNOSTIC_Y_MIN = 1e-15
DIAGNOSTIC_Y_MAX = 1.0


def folded_frequency(signal_hz, sample_rate_hz):
    """Fold a frequency into the one-sided interval [0, sample_rate / 2]."""
    return abs(
        (signal_hz + sample_rate_hz / 2.0) % sample_rate_hz
        - sample_rate_hz / 2.0
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot unstitched bands and predicted aliases for a test sine."
    )
    parser.add_argument(
        "frequency_hz",
        type=float,
        nargs="?",
        help="Applied sine-wave frequency in Hz (for example: 150)",
    )
    return parser.parse_args()


def read_band_blocks_with_rates(ser, band_order):
    """Read one BLOCK response, including the measured rate metadata."""
    blocks = {}
    sample_rates_hz = {}
    send_cmd(ser, "BLOCK")

    while len(blocks) < len(band_order):
        band_name = None
        while band_name is None:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("BAND:"):
                band_name = line.split(":", 1)[1]

        measured_rate_hz = None
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("BAND_SAMPLE_RATE_HZ:"):
                measured_rate_hz = float(line.split(":", 1)[1])
            elif line == "START":
                break

        data = []
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "END":
                break
            try:
                data.append(float(line))
            except ValueError:
                pass

        if band_name in band_order:
            blocks[band_name] = np.asarray(data, dtype=float)
            if measured_rate_hz is not None:
                sample_rates_hz[band_name] = measured_rate_hz

    return blocks, sample_rates_hz


def main():
    args = parse_args()
    target_hz = args.frequency_hz
    if target_hz is None:
        target_hz = float(input("Applied sine-wave frequency (Hz): ").strip())
    ser = open_serial(PORT, BAUD)

    try:
        raw_sample_rate_hz = request_raw_sample_rate(ser)
        bands = make_bands(raw_sample_rate_hz)
        band_order = [
            band_name
            for band_name in active_band_order(bands)
            if band_name != "MID"
        ]

        print(f"Raw sample rate: {raw_sample_rate_hz:.3f} Hz")
        print(f"Applied sine frequency: {target_hz:.3f} Hz")
        print("Close the diagnostic figure or press Ctrl+C to stop.")

        plt.ion()
        fig, axes = plt.subplots(
            len(band_order),
            1,
            figsize=(10, 2.8 * len(band_order)),
            squeeze=False,
        )
        axes = axes[:, 0]
        lines = {}
        alias_lines = {}

        for ax, band_name in zip(axes, band_order):
            sample_rate_hz = bands[band_name]["f_min_Hz"] * SAMPLES
            alias_hz = folded_frequency(target_hz, sample_rate_hz)

            line, = ax.plot([], [], linewidth=0.8, label="Measured magnitude")
            lines[band_name] = line
            alias_line = ax.axvline(
                max(alias_hz, bands[band_name]["f_min_Hz"]),
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label=(
                    f"Predicted fold: {alias_hz:.3f} Hz"
                    if alias_hz > 0
                    else "Predicted fold: DC"
                ),
            )
            alias_line.set_visible(alias_hz > 0)
            alias_lines[band_name] = alias_line

            ax.set_title(
                f"{band_name}: FFT sample rate {sample_rate_hz:.3f} Hz, "
                f"Nyquist {sample_rate_hz / 2.0:.3f} Hz"
            )
            ax.set_ylabel("Magnitude (V)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(
                bands[band_name]["f_min_Hz"],
                sample_rate_hz / 2.0,
            )
            ax.set_ylim(1e-15, 1.0)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

            print(
                f"{band_name}: FFT rate={sample_rate_hz:.3f} Hz, "
                f"Nyquist={sample_rate_hz / 2.0:.3f} Hz, "
                f"predicted observed frequency={alias_hz:.3f} Hz"
            )

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.suptitle(f"Unstitched Band Aliasing Test — input {target_hz:g} Hz")
        fig.tight_layout()
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            blocks, measured_rates_hz = read_band_blocks_with_rates(
                ser,
                band_order,
            )

            for ax, band_name in zip(axes, band_order):
                data = np.asarray(blocks.get(band_name, []), dtype=float)
                if data.size < SAMPLES:
                    continue

                nominal_rate_hz = bands[band_name]["f_min_Hz"] * SAMPLES
                sample_rate_hz = measured_rates_hz.get(
                    band_name,
                    nominal_rate_hz,
                )
                alias_hz = folded_frequency(target_hz, sample_rate_hz)
                analysis = compute_fft_analysis(
                    data=data[:SAMPLES],
                    sample_rate_hz=sample_rate_hz,
                )
                freqs = np.asarray(analysis["freqs_Hz"], dtype=float)
                magnitude = np.asarray(analysis["mag_V"], dtype=float)
                valid = (
                    np.isfinite(freqs)
                    & np.isfinite(magnitude)
                    & (freqs > 0)
                    & (magnitude > 0)
                )
                if not np.any(valid):
                    continue

                lines[band_name].set_data(freqs[valid], magnitude[valid])
                marker_hz = max(alias_hz, bands[band_name]["f_min_Hz"])
                alias_lines[band_name].set_xdata([marker_hz, marker_hz])
                alias_lines[band_name].set_visible(alias_hz > 0)
                alias_lines[band_name].set_label(
                    f"Predicted fold: {alias_hz:.3f} Hz"
                    if alias_hz > 0
                    else "Predicted fold: DC"
                )
                ax.set_title(
                    f"{band_name}: measured sample rate {sample_rate_hz:.3f} Hz, "
                    f"Nyquist {sample_rate_hz / 2.0:.3f} Hz"
                )
                ax.set_xlim(
                    bands[band_name]["f_min_Hz"],
                    sample_rate_hz / 2.0,
                )
                ax.set_ylim(DIAGNOSTIC_Y_MIN, DIAGNOSTIC_Y_MAX)
                ax.legend(fontsize=8)

                rate_source = "measured" if band_name in measured_rates_hz else "nominal"
                print(
                    f"{band_name}: {rate_source} rate={sample_rate_hz:.6f} Hz, "
                    f"predicted fold={alias_hz:.6f} Hz"
                )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\nAliasing diagnostic stopped.")
    finally:
        ser.close()
        print("Serial closed.")


if __name__ == "__main__":
    main()
