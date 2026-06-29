from sweep_runner import run_live_mode, run_hardware_sweep, run_synthetic_sweep
from plot_compare import (
    compare_saved_measurements,
    plot_test_errors,
    plot_thd2,
    plot_amplitude_comparison,
)
from storage import clear_saved_measurements


def main():
    while True:
        print()
        print("ADC Spectrometer")
        print("1 = Live ASD mode")
        print("2 = Hardware test sweep")
        print("3 = Synthetic test sweep")
        print("4 = Compare saved measurements")
        print("5 = Plot test errors")
        print("6 = Plot THD2")
        print("7 = Plot amplitude comparison")
        print("8 = Clear saved measurements")
        print("q = Quit")

        choice = input("> ").strip().lower()

        if choice == "1":
            run_live_mode()

        elif choice == "2":
            run_hardware_sweep()
            plot_test_errors()
            plot_thd2()

        elif choice == "3":
            clear_saved_measurements()
            run_synthetic_sweep()

        elif choice == "4":
            compare_saved_measurements()

        elif choice == "5":
            plot_test_errors()

        elif choice == "6":
            plot_thd2()

        elif choice == "7":
            plot_amplitude_comparison()

        elif choice == "8":
            clear_saved_measurements()

        elif choice == "q":
            break

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()