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
from FFT_analysis import calculate_test_results
from storage import save_measurement, clear_saved_measurements, calculate_noise_stats
from plot_compare import compare_saved_measurements, plot_test_errors, plot_thd2
from serial_io import open_serial, request_raw_sample_rate, read_band_blocks
from plot_live import create_live_plot, update_live_plot



def fmt(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"

def run_synthetic_sweep():
    instrument = "TEST_MODE"
    measurement_type = "synthetic_test"

    raw_sample_rate_hz = 100000.0
    bands = make_bands(raw_sample_rate_hz)

    clear_saved_measurements()

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

        test = calculate_test_results(
            freqs_hz=spectrum["freqs_Hz"],
            mag_v=spectrum["mag_V"],
            target_freq_hz=target_freq_hz,
            target_amp_v=TEST_SIGNAL_AMP_V,
            bin_width_hz= 1
        )

        print(
            f"Measured: {fmt(test['measured_freq_Hz'])} Hz, "
            f"Gain error: {fmt(test['gain_dB'])} dB, "
            f"Freq error: {fmt(test['freq_error_bins'])} bins, "
            f"THD2: {fmt(test['thd2_percent'])}%"
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

    compare_saved_measurements()
    plot_test_errors()


def run_hardware_sweep():
    instrument = "SAMD21"
    measurement_type = "hardware_test"

    ser = open_serial(PORT, BAUD)

    try:
        raw_sample_rate_hz = request_raw_sample_rate(ser)
        bands = make_bands(raw_sample_rate_hz)

        clear_saved_measurements()

        sweep_index = 0

        while sweep_index < len(TEST_FREQS_HZ):
            target_freq_hz = TEST_FREQS_HZ[sweep_index]

            print()
            print(f"Hardware sweep point {sweep_index + 1}/{len(TEST_FREQS_HZ)}")
            print(f"Set generator to {target_freq_hz:.3f} Hz")
            print(f"Target amplitude: {TEST_SIGNAL_AMP_V:.6g} V")

            choice = input("Enter = capture/save, r = redo, q = quit: ").strip().lower()

            if choice == "q":
                break

            if choice == "r":
                print("Redoing same sweep point.")
                continue

            blocks = read_band_blocks(ser, BAND_ORDER)

            spectrum = process_blocks(
                blocks=blocks,
                bands=bands,
                band_order=BAND_ORDER,
                samples=SAMPLES,
            )

            if spectrum is None:
                print("No valid spectrum generated. Redo this point.")
                continue

            test = calculate_test_results(
                freqs_hz=spectrum["freqs_Hz"],
                mag_v=spectrum["mag_V"],
                target_freq_hz=target_freq_hz,
                target_amp_v=TEST_SIGNAL_AMP_V,
                bin_width_hz=1,
            )

            print(
                f"Measured: {fmt(test['measured_freq_Hz'])} Hz, "
                f"Gain error: {fmt(test['gain_dB'])} dB, "
                f"Freq error: {fmt(test['freq_error_bins'])} bins, "
                f"THD2: {fmt(test['thd2_percent'])}%"
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

            sweep_index += 1

        compare_saved_measurements()
        plot_test_errors()
        plot_thd2()

    finally:
        ser.close()
        print("Serial closed.")


def run_live_mode():
    instrument = "SAMD21"

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

        print("Live mode running.")
        print("Close plot window or press Ctrl+C to stop.")

        while True:
            blocks = read_band_blocks(ser, BAND_ORDER)

            spectrum = process_blocks(
                blocks=blocks,
                bands=bands,
                band_order=BAND_ORDER,
                samples=SAMPLES,
            )

            if spectrum is None:
                continue

            stats = calculate_noise_stats(
                freqs_hz=spectrum["freqs_Hz"],
                asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
                stats_min_hz=STATS_MIN_HZ,
                stats_max_hz=STATS_MAX_HZ,
            )

            update_live_plot(
                plot=plot,
                freqs_hz=spectrum["freqs_Hz"],
                asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
                stats=stats,
                y_min=PLOT_CONFIG["y_min"],
                y_max=PLOT_CONFIG["y_max"],
            )

    except KeyboardInterrupt:
        print()
        print("Live mode stopped.")

    finally:
        ser.close()
        print("Serial closed.")