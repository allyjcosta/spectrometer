import numpy as np
from scipy.signal.windows import blackmanharris

#compute_fft_analysis()
#find_test_peak()
#find_second_harmonic()
#calculate_test_results()

def compute_fft_analysis(data, sample_rate_hz):
    data = np.asarray(data, dtype=float)
    samples = len(data)

    sample_interval_s = 1.0 / sample_rate_hz
    bin_width_hz = sample_rate_hz / samples

    window = blackmanharris(samples)
    coherent_gain = np.sum(window) / samples

    enbw_hz = (
        sample_rate_hz
        * np.sum(window ** 2)
        / (np.sum(window) ** 2)
    )

    data = data - np.mean(data)

    freqs_hz = np.fft.rfftfreq(samples, d=sample_interval_s)
    fft_result = np.fft.rfft(data * window)

    mag_v = np.abs(fft_result) / ((samples / 2.0) * coherent_gain)
    asd_v_per_sqrt_hz = mag_v / np.sqrt(enbw_hz)

    # Remove DC bin
    freqs_hz = freqs_hz[1:]
    mag_v = mag_v[1:]
    asd_v_per_sqrt_hz = asd_v_per_sqrt_hz[1:]

    # Keep log plots safe
    mag_v[mag_v <= 0] = 1e-15
    asd_v_per_sqrt_hz[asd_v_per_sqrt_hz <= 0] = 1e-12

    return {
        "freqs_Hz": freqs_hz,
        "mag_V": mag_v,
        "asd_V_per_sqrtHz": asd_v_per_sqrt_hz,
        "bin_width_Hz": bin_width_hz,
        "enbw_Hz": enbw_hz,
    }

def find_test_peak(freqs_hz, mag_v, target_freq_hz, search_bins=5):
    nearest_idx = np.argmin(np.abs(freqs_hz - target_freq_hz))

    search_lo = max(0, nearest_idx - search_bins)
    search_hi = min(len(freqs_hz), nearest_idx + search_bins + 1)

    local_peak_idx = search_lo + np.argmax(mag_v[search_lo:search_hi])

    measured_freq_hz = freqs_hz[local_peak_idx]
    measured_amp_v = mag_v[local_peak_idx]

    return {
        "measured_freq_Hz": float(measured_freq_hz),
        "measured_amp_V": float(measured_amp_v),
        "peak_index": int(local_peak_idx),
    }


def find_global_peak(freqs_hz, mag_v):
    peak_index = int(np.argmax(mag_v))
    return {
        "measured_freq_Hz": float(freqs_hz[peak_index]),
        "measured_amp_V": float(mag_v[peak_index]),
        "peak_index": peak_index,
    }


def calculate_spectral_leakage(
    freqs_hz,
    mag_v,
    target_freq_hz,
    main_lobe_half_width_bins=4,
):
    """Measure power outside a Blackman-Harris fundamental main lobe."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    mag_v = np.asarray(mag_v, dtype=float)

    valid = np.isfinite(freqs_hz) & np.isfinite(mag_v) & (mag_v >= 0)
    freqs_hz = freqs_hz[valid]
    mag_v = mag_v[valid]

    if len(freqs_hz) == 0:
        return {
            "spectral_leakage_ratio": None,
            "spectral_leakage_percent": None,
            "spectral_leakage_dB": None,
        }

    peak = find_test_peak(
        freqs_hz=freqs_hz,
        mag_v=mag_v,
        target_freq_hz=target_freq_hz,
    )
    peak_index = peak["peak_index"]
    main_lo = max(0, peak_index - main_lobe_half_width_bins)
    main_hi = min(len(mag_v), peak_index + main_lobe_half_width_bins + 1)

    bin_power = mag_v ** 2
    fundamental_power = float(np.sum(bin_power[main_lo:main_hi]))
    total_power = float(np.sum(bin_power))
    leakage_power = max(0.0, total_power - fundamental_power)

    if fundamental_power <= 0:
        leakage_ratio = None
        leakage_db = None
    else:
        leakage_ratio = leakage_power / fundamental_power
        leakage_db = (
            None
            if leakage_ratio <= 0
            else float(10.0 * np.log10(leakage_ratio))
        )

    return {
        "spectral_leakage_ratio": (
            None if leakage_ratio is None else float(leakage_ratio)
        ),
        "spectral_leakage_percent": (
            None if leakage_ratio is None else float(100.0 * leakage_ratio)
        ),
        "spectral_leakage_dB": leakage_db,
        "leakage_main_lobe_half_width_bins": int(main_lobe_half_width_bins),
    }


def find_second_harmonic(freqs_hz, mag_v, fundamental_freq_hz, search_bins=5):
    second_harmonic_hz = 2.0 * fundamental_freq_hz

    if second_harmonic_hz > freqs_hz[-1]:
        return {
            "second_harmonic_freq_Hz": None,
            "second_harmonic_amp_V": None,
        }

    nearest_idx = int(np.argmin(np.abs(freqs_hz - second_harmonic_hz)))

    # Estimate resolution locally because stitched bands can have different
    # bin widths. A Blackman-Harris fundamental occupies roughly +/-4 bins;
    # if its second harmonic falls inside that lobe, THD2 is not resolvable.
    resolution_lo = max(0, nearest_idx - 3)
    resolution_hi = min(len(freqs_hz), nearest_idx + 4)
    local_diffs = np.diff(freqs_hz[resolution_lo:resolution_hi])
    local_diffs = local_diffs[local_diffs > 0]
    if len(local_diffs) == 0:
        return {
            "second_harmonic_freq_Hz": None,
            "second_harmonic_amp_V": None,
        }

    local_bin_width_hz = float(np.median(local_diffs))
    harmonic_separation_hz = second_harmonic_hz - fundamental_freq_hz
    if harmonic_separation_hz <= 4.0 * local_bin_width_hz:
        return {
            "second_harmonic_freq_Hz": None,
            "second_harmonic_amp_V": None,
        }

    separation_bins = harmonic_separation_hz / local_bin_width_hz
    nonoverlap_search_bins = max(0, int(np.floor(separation_bins / 2.0)) - 1)
    effective_search_bins = min(search_bins, nonoverlap_search_bins)

    search_lo = max(0, nearest_idx - effective_search_bins)
    search_hi = min(len(freqs_hz), nearest_idx + effective_search_bins + 1)

    local_peak_idx = search_lo + np.argmax(mag_v[search_lo:search_hi])

    return {
        "second_harmonic_freq_Hz": float(freqs_hz[local_peak_idx]),
        "second_harmonic_amp_V": float(mag_v[local_peak_idx]),
    }


def calculate_test_results(
    freqs_hz,
    mag_v,
    target_freq_hz,
    target_amp_v,
    fmin_hz,
    search_bins=5,
    use_global_peak=False,
):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    mag_v = np.asarray(mag_v, dtype=float)

    if use_global_peak:
        peak = find_global_peak(freqs_hz=freqs_hz, mag_v=mag_v)
    else:
        peak = find_test_peak(
            freqs_hz=freqs_hz,
            mag_v=mag_v,
            target_freq_hz=target_freq_hz,
            search_bins=search_bins,
        )

    measured_freq_hz = peak["measured_freq_Hz"]
    measured_amp_v = peak["measured_amp_V"]

    harmonic = find_second_harmonic(
        freqs_hz=freqs_hz,
        mag_v=mag_v,
        fundamental_freq_hz=measured_freq_hz,
        search_bins=search_bins,
    )

    # Calculate frequency errors purely based on the local band's fmin_hz
    freq_error_hz = measured_freq_hz - target_freq_hz
    freq_error_bins = freq_error_hz / fmin_hz

    gain = measured_amp_v / target_amp_v if target_amp_v > 0 else None

    if gain is not None and gain > 0:
        gain_dB = 20 * np.log10(gain)
    else:
        gain_dB = None

    second_amp_v = harmonic["second_harmonic_amp_V"]

    if second_amp_v is not None and measured_amp_v > 0:
        thd2_percent = 100.0 * second_amp_v / measured_amp_v
    else:
        thd2_percent = None

    return {
        "target_freq_Hz": float(target_freq_hz),
        "target_amp_V": float(target_amp_v),

        "measured_freq_Hz": float(measured_freq_hz),
        "measured_amp_V": float(measured_amp_v),

        "freq_error_Hz": float(freq_error_hz),
        "freq_error_bins": float(freq_error_bins),

        "gain": None if gain is None else float(gain),
        "gain_dB": None if gain_dB is None else float(gain_dB),

        "second_harmonic_freq_Hz": harmonic["second_harmonic_freq_Hz"],
        "second_harmonic_amp_V": harmonic["second_harmonic_amp_V"],
        "thd2_percent": None if thd2_percent is None else float(thd2_percent),
    }
