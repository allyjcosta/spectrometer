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

def find_second_harmonic(freqs_hz, mag_v, target_freq_hz, search_bins=5):
    second_harmonic_hz = 2.0 * target_freq_hz

    if second_harmonic_hz > freqs_hz[-1]:
        return {
            "second_harmonic_freq_Hz": None,
            "second_harmonic_amp_V": None,
        }

    nearest_idx = np.argmin(np.abs(freqs_hz - second_harmonic_hz))

    search_lo = max(0, nearest_idx - search_bins)
    search_hi = min(len(freqs_hz), nearest_idx + search_bins + 1)

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
):

    peak = find_test_peak(
        freqs_hz=freqs_hz,
        mag_v=mag_v,
        target_freq_hz=target_freq_hz,
        search_bins=search_bins,
    )

    harmonic = find_second_harmonic(
        freqs_hz=freqs_hz,
        mag_v=mag_v,
        target_freq_hz=target_freq_hz,
        search_bins=search_bins,
    )

    measured_freq_hz = peak["measured_freq_Hz"]
    measured_amp_v = peak["measured_amp_V"]

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