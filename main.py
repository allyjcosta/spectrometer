from sweep_runner import run_live_mode, run_synthetic_sweep
from plot_compare import (
    compare_saved_measurements,
    plot_saved_measurements_stacked,
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
        print("2 = Synthetic test sweep")
        print("3 = Compare saved measurements")
        print("4 = Plot test errors")
        print("5 = Plot THD2")
        print("6 = Plot amplitude comparison")
        print("7 = Clear saved measurements")
        print("8 = Stack saved measurements")
        print("q = Quit")

        choice = input("> ").strip().lower()

        if choice == "1":
            run_live_mode()

        elif choice == "2":
            clear_saved_measurements()
            run_synthetic_sweep()
            compare_saved_measurements()
            plot_test_errors()

        elif choice == "3":
            compare_saved_measurements()

        elif choice == "4":
            plot_test_errors()

        elif choice == "5":
            plot_thd2()

        elif choice == "6":
            plot_amplitude_comparison()

        elif choice == "7":
            clear_saved_measurements()

        elif choice == "8":
            plot_saved_measurements_stacked()

        elif choice == "q":
            break

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()
