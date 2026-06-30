from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

from config import (
    SAMPLES,
    TEST_FREQS_HZ,
    TEST_SIGNAL_AMP_V,
    STATS_MIN_HZ,
    STATS_MAX_HZ,
    PLOT_CONFIG,
    BAND_ORDER,
    make_bands,
    PORT,
    BAUD,
)

from signal_source import generate_synthetic_blocks
from spectrum_processing import process_blocks
from FFT_analysis import calculate_spectral_leakage, calculate_test_results
from storage import save_measurement, calculate_noise_stats
from serial_io import open_serial, request_raw_sample_rate, read_band_blocks
from plot_live import create_live_plot, update_live_plot
from live_input_handler import create_input_handler


def fmt(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def run_synthetic_sweep():
    instrument = "TEST_MODE"
    measurement_type = "synthetic_test"

    raw_sample_rate_hz = 100000.0
    bands = make_bands(raw_sample_rate_hz)

    for index, target_freq_hz in enumerate(TEST_FREQS_HZ):
        print()
        print(f"Synthetic sweep point {index + 1}/{len(TEST_FREQS_HZ)}")
        print(f"Target: {target_freq_hz:.3f} Hz")

        blocks = generate_synthetic_blocks(
            bands=bands,
            band_order=BAND_ORDER,
            samples=SAMPLES,
            signal_freq_hz=target_freq_hz,
            signal_amp_v=TEST_SIGNAL_AMP_V,
        )

        spectrum = process_blocks(
            blocks=blocks,
            bands=bands,
            band_order=BAND_ORDER,
            samples=SAMPLES,
        )

        if target_freq_hz >= bands["HIGH"]["stitch_min"]:
            active_band_name = "HIGH"
            active_fmin = bands["HIGH"]["f_min_Hz"]
        elif target_freq_hz >= bands["MID"]["stitch_min"]:
            active_band_name = "MID"
            active_fmin = bands["MID"]["f_min_Hz"]
        else:
            active_band_name = "LOW"
            active_fmin = bands["LOW"]["f_min_Hz"]

        test = calculate_test_results(
            freqs_hz=spectrum["freqs_Hz"],
            mag_v=spectrum["mag_V"],
            target_freq_hz=target_freq_hz,
            target_amp_v=TEST_SIGNAL_AMP_V,
            fmin_hz=active_fmin
        )

        active_band = spectrum["band_results_mag"][active_band_name]
        test.update(
            calculate_spectral_leakage(
                freqs_hz=active_band["freqs"],
                mag_v=active_band["amps_V"],
                target_freq_hz=target_freq_hz,
            )
        )

        print(
            f"Measured: {fmt(test['measured_freq_Hz'])} Hz, "
            f"Gain error: {fmt(test['gain_dB'])} dB, "
            f"Freq error: {fmt(test['freq_error_bins'])} bins, "
            f"THD2: {fmt(test['thd2_percent'])}%, "
            f"Leakage: {fmt(test['spectral_leakage_dB'])} dB"
        )

        save_measurement(
            instrument=instrument,
            measurement_type=measurement_type,
            freqs_hz=spectrum["freqs_Hz"],
            mag_v=spectrum["mag_V"],
            asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
            sample_rate_hz=raw_sample_rate_hz,
            samples=SAMPLES,
            plot_config=PLOT_CONFIG,
            stats_min_hz=STATS_MIN_HZ,
            stats_max_hz=STATS_MAX_HZ,
            test=test,
        )

    print()
    print("Synthetic sweep complete.")


def run_live_mode():
    input_handler = None
    ser = open_serial(PORT, BAUD)

    try:
        raw_sample_rate_hz = request_raw_sample_rate(ser)
        bands = make_bands(raw_sample_rate_hz)

        plot = create_live_plot(
            y_min=PLOT_CONFIG["y_min"],
            y_max=PLOT_CONFIG["y_max"],
            raw_sample_rate_hz=raw_sample_rate_hz,
            title="Live Stitched ASD ADC Noise Spectrometer",
        )
        input_handler = create_input_handler()

        print("Live mode running.")
        print("Close the plot window, enter q, or press Ctrl+C to stop.")

        stop_requested = False

        with ThreadPoolExecutor(max_workers=1) as acquisition_executor:
            while plot["running"] and not stop_requested:
                capture = acquisition_executor.submit(
                    read_band_blocks,
                    ser,
                    BAND_ORDER,
                )

                while not capture.done():
                    stop_requested = input_handler.process_commands(
                        instrument="SAMD21",
                        sample_rate_hz=raw_sample_rate_hz,
                        samples=SAMPLES,
                        plot_config=PLOT_CONFIG,
                        stats_min_hz=STATS_MIN_HZ,
                        stats_max_hz=STATS_MAX_HZ,
                    )
                    plt.pause(0.05)

                    if stop_requested or not plot["running"]:
                        break

                # Finish consuming the current serial response before another
                # command is sent or the serial port is closed.
                blocks = capture.result()

                if stop_requested or not plot["running"]:
                    break

                spectrum = process_blocks(
                    blocks=blocks,
                    bands=bands,
                    band_order=BAND_ORDER,
                    samples=SAMPLES,
                )

                if spectrum is not None:
                    stats = calculate_noise_stats(
                        freqs_hz=spectrum["freqs_Hz"],
                        asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
                        stats_min_hz=STATS_MIN_HZ,
                        stats_max_hz=STATS_MAX_HZ,
                    )

                    input_handler.update_spectrum(spectrum)
                    update_live_plot(
                        plot=plot,
                        freqs_hz=spectrum["freqs_Hz"],
                        asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
                        stats=stats,
                        y_min=PLOT_CONFIG["y_min"],
                        y_max=PLOT_CONFIG["y_max"],
                    )

                stop_requested = input_handler.process_commands(
                    instrument="SAMD21",
                    sample_rate_hz=raw_sample_rate_hz,
                    samples=SAMPLES,
                    plot_config=PLOT_CONFIG,
                    stats_min_hz=STATS_MIN_HZ,
                    stats_max_hz=STATS_MAX_HZ,
                )

    except KeyboardInterrupt:
        print()
        print("Live mode stopped.")

    finally:
        if input_handler is not None:
            input_handler.stop()
        ser.close()
        print("Serial closed.")
