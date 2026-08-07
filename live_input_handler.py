import os
import queue
import select
import sys
import threading

from config import (
    TEST_FREQS_HZ,
    TEST_SIGNAL_AMP_V,
    INSTRUMENT_LABEL,
    band_for_frequency,
    make_bands,
)
from FFT_analysis import calculate_test_results
from plot_compare import (
    compare_saved_measurements,
    plot_saved_measurements_stacked,
    plot_amplitude_comparison,
    plot_test_errors,
    plot_thd2,
)
from plot_time_domain import create_time_domain_plot, update_time_domain_plot
from storage import (
    clear_saved_measurements,
    discover_measurements,
    load_measurement,
    save_measurement,
)


class LiveInputHandler:
    """Queue terminal commands without blocking live acquisition."""

    def __init__(self):
        self.command_queue = queue.Queue()
        self.command_completed = threading.Event()
        self.input_thread = None
        self.running = False
        self.current_spectrum = None
        self.current_blocks = None
        self.current_band_sample_rates_hz = None
        self.time_domain_plot = None
        self.saved_test_points = []
        self.prepared_live_test_files = []
        self.next_test_index = 0

    def start(self):
        self.running = True
        self.command_completed.set()
        self._display_menu()
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def stop(self):
        self.running = False
        self.command_completed.set()
        if self.input_thread is not None:
            self.input_thread.join(timeout=0.5)

    def _display_menu(self):
        print()
        print("Live Mode")
        print("1 = Save current live spectrum")
        print("2 = List saved measurements")
        print("3 = Compare saved measurements")
        print("4 = Plot test errors")
        print("5 = Plot THD2")
        print("6 = Plot amplitude comparison")
        print("7 = Clear saved measurements")
        print("8 = Stack saved measurements")
        print("9 = Plot time domain signal")
        print("q = Quit live mode")

    def _input_loop(self):
        while self.running:
            self.command_completed.wait()
            self.command_completed.clear()
            if not self.running:
                return

            print("Command> ", end="", flush=True)

            while self.running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue

                line = sys.stdin.readline()
                if line == "":
                    self.command_queue.put("q")
                    return

                command = line.strip().lower()
                if command:
                    self.command_queue.put(command)
                    print(f"Command {command!r} queued; it will run after the current capture.")

                if command == "q":
                    return

                if not command:
                    self.command_completed.set()

                break

    def update_spectrum(self, spectrum):
        self.current_spectrum = spectrum.copy()

    def update_time_domain(self, blocks, band_sample_rates_hz):
        self.current_blocks = {
            name: data.copy()
            for name, data in blocks.items()
        }
        self.current_band_sample_rates_hz = dict(band_sample_rates_hz)
        if (
            self.time_domain_plot is not None
            and self.time_domain_plot.get("running", False)
        ):
            update_time_domain_plot(
                plot=self.time_domain_plot,
                blocks=self.current_blocks,
                sample_rates_hz=self.current_band_sample_rates_hz,
            )

    def process_commands(
        self,
        instrument,
        sample_rate_hz,
        samples,
        plot_config,
        stats_min_hz,
        stats_max_hz,
    ):
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return False

            command_labels = {
                "1": "save current live spectrum",
                "2": "list saved measurements",
                "3": "compare saved measurements",
                "4": "plot test errors",
                "5": "plot THD2",
                "6": "plot amplitude comparison",
                "7": "clear saved measurements",
                "8": "stack saved measurements",
                "9": "plot time domain signal",
                "q": "quit live mode",
            }
            command_label = command_labels.get(command)
            if command_label is not None:
                print(f"Processing: {command_label}...")

            if command == "1":
                self._save_live_snapshot(
                    instrument=instrument,
                    sample_rate_hz=sample_rate_hz,
                    samples=samples,
                    plot_config=plot_config,
                    stats_min_hz=stats_min_hz,
                    stats_max_hz=stats_max_hz,
                )
            elif command == "3":
                self._run_plot("comparison", compare_saved_measurements)
            elif command == "4":
                files = self._prepare_live_test_measurements(
                    instrument=instrument,
                    sample_rate_hz=sample_rate_hz,
                    samples=samples,
                    plot_config=plot_config,
                    stats_min_hz=stats_min_hz,
                    stats_max_hz=stats_max_hz,
                )
                if files:
                    self._run_plot("test errors", plot_test_errors, files=files)
                else:
                    self._run_sweep_plot("test errors", plot_test_errors)
            elif command == "5":
                self._run_plot("THD2", plot_thd2)
            elif command == "6":
                self._run_plot("amplitude comparison", plot_amplitude_comparison)
            elif command == "7":
                clear_saved_measurements()
                self.saved_test_points.clear()
                self.prepared_live_test_files.clear()
                self.next_test_index = 0
            elif command == "8":
                self._run_plot("stacked measurements", plot_saved_measurements_stacked)
            elif command == "9":
                self._plot_time_domain()
            elif command == "2":
                self._list_measurements()
            elif command == "q":
                print("Live mode quit requested.")
                return True
            else:
                print(f"Unknown command: {command!r}")

            if command_label is not None:
                print(f"Completed: {command_label}.")

            self._display_menu()
            self.command_completed.set()

    def _plot_time_domain(self):
        if self.current_blocks is None:
            print("No live time-domain data is available yet.")
            return

        try:
            if (
                self.time_domain_plot is None
                or not self.time_domain_plot.get("running", False)
            ):
                self.time_domain_plot = create_time_domain_plot(
                    blocks=self.current_blocks,
                    sample_rates_hz=self.current_band_sample_rates_hz,
                    title="Live Time Domain Signal",
                )
            else:
                update_time_domain_plot(
                    plot=self.time_domain_plot,
                    blocks=self.current_blocks,
                    sample_rates_hz=self.current_band_sample_rates_hz,
                )
        except Exception as error:
            print(f"Could not plot time-domain signal: {error}")

    def _save_live_snapshot(
        self,
        instrument,
        sample_rate_hz,
        samples,
        plot_config,
        stats_min_hz,
        stats_max_hz,
    ):
        if self.current_spectrum is None:
            print("No live spectrum is available yet.")
            return

        spectrum = self.current_spectrum
        try:
            filename = save_measurement(
                instrument=instrument,
                measurement_type="live",
                freqs_hz=spectrum["freqs_Hz"],
                mag_v=spectrum["mag_V"],
                asd_v_per_sqrt_hz=spectrum["asd_V_per_sqrtHz"],
                sample_rate_hz=sample_rate_hz,
                samples=samples,
                plot_config=plot_config,
                stats_min_hz=stats_min_hz,
                stats_max_hz=stats_max_hz,
            )
        except Exception as error:
            print(f"Could not save live spectrum: {error}")
            return

        target_freq_hz = None
        if self.next_test_index < len(TEST_FREQS_HZ):
            target_freq_hz = TEST_FREQS_HZ[self.next_test_index]

        self.saved_test_points.append((filename, target_freq_hz))
        self.next_test_index += 1

    def _prepare_live_test_measurements(
        self,
        instrument,
        sample_rate_hz,
        samples,
        plot_config,
        stats_min_hz,
        stats_max_hz,
    ):
        prepared_files = []

        live_files = []
        for filename in discover_measurements():
            try:
                measurement = load_measurement(filename)
            except Exception as error:
                print(f"Could not load {filename}: {error}")
                continue

            if measurement.get("instrument") != instrument:
                continue
            if measurement.get("measurement_type") not in ("live", "live_test"):
                continue

            live_files.append((filename, measurement))

        if len(live_files) > len(TEST_FREQS_HZ):
            print(
                f"Only the first {len(TEST_FREQS_HZ)} of {len(live_files)} "
                "saved live measurements have configured test frequencies."
            )

        for index, (filename, measurement) in enumerate(live_files):
            if index >= len(TEST_FREQS_HZ):
                break

            target_freq_hz = TEST_FREQS_HZ[index]
            print(
                f"Test index {index}: {os.path.basename(filename)} "
                f"-> {target_freq_hz:.3f} Hz"
            )

            measurement_sample_rate_hz = measurement.get(
                "sample_rate_Hz",
                sample_rate_hz,
            )
            bands = make_bands(measurement_sample_rate_hz)
            active_band_name = band_for_frequency(bands, target_freq_hz)
            if active_band_name is None:
                print(
                    f"Skipping {target_freq_hz:.3f} Hz: outside instrument range."
                )
                continue
            active_fmin = bands[active_band_name]["f_min_Hz"]

            test = calculate_test_results(
                freqs_hz=measurement["freqs_Hz"],
                mag_v=measurement["mag_V"],
                target_freq_hz=target_freq_hz,
                target_amp_v=TEST_SIGNAL_AMP_V,
                fmin_hz=active_fmin,
                use_global_peak=False,
            )

            try:
                save_measurement(
                    instrument=measurement.get("instrument", INSTRUMENT_LABEL),
                    measurement_type="live_test",
                    freqs_hz=measurement["freqs_Hz"],
                    mag_v=measurement["mag_V"],
                    asd_v_per_sqrt_hz=measurement["asd_V_per_sqrtHz"],
                    sample_rate_hz=measurement_sample_rate_hz,
                    samples=measurement.get("samples", samples),
                    plot_config=measurement.get("plot_config", plot_config),
                    stats_min_hz=stats_min_hz,
                    stats_max_hz=stats_max_hz,
                    test=test,
                    filename=filename,
                )
            except Exception as error:
                print(f"Could not prepare {filename} for error plotting: {error}")
                continue

            prepared_files.append(filename)

        self.prepared_live_test_files = prepared_files
        return prepared_files

    def _run_plot(self, label, plot_function, files=None):
        try:
            plot_function(files=files)
        except Exception as error:
            print(f"Could not plot {label}: {error}")

    def _run_sweep_plot(self, label, plot_function):
        files = self.prepared_live_test_files or self._discover_current_sweep_measurements()
        if not files:
            print("No saved sweep data matches the configured sweep frequencies.")
            return
        self._run_plot(label, plot_function, files=files)

    def _discover_current_sweep_measurements(self):
        matching_files = []

        for filename in discover_measurements():
            try:
                measurement = load_measurement(filename)
            except Exception:
                continue

            test = measurement.get("test") or measurement.get("test_results") or {}
            target_freq_hz = test.get("target_freq_Hz")
            if target_freq_hz is None:
                target_freq_hz = test.get("input_freq_Hz")
            if target_freq_hz is None:
                target_freq_hz = measurement.get("test_input_freq_Hz")
            if target_freq_hz is None:
                continue

            target_freq_hz = float(target_freq_hz)
            if target_freq_hz in TEST_FREQS_HZ:
                matching_files.append(filename)

        return matching_files

    def _list_measurements(self):
        files = discover_measurements()
        if not files:
            print("No saved measurements found.")
            return

        for index, filename in enumerate(files, 1):
            print(f"{index:>3}: {os.path.basename(filename)}")


def create_input_handler():
    handler = LiveInputHandler()
    handler.start()
    return handler
