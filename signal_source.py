import numpy as np


def generate_test_data(samples, sample_rate_hz, signal_freq_hz, signal_amp_v):
    t = np.arange(samples) / sample_rate_hz
    return signal_amp_v * np.sin(2 * np.pi * signal_freq_hz * t)


def generate_synthetic_blocks(
    bands,
    band_order,
    samples,
    signal_freq_hz,
    signal_amp_v,
):
    blocks = {}

    for band_name in band_order:
        band = bands[band_name]
        fft_sample_rate_hz = band["f_min_Hz"] * samples

        in_band = (
            signal_freq_hz >= band["stitch_min"]
            and (
                band["stitch_max"] is None
                or signal_freq_hz < band["stitch_max"]
            )
        )

        if in_band:
            data = generate_test_data(
                samples=samples,
                sample_rate_hz=fft_sample_rate_hz,
                signal_freq_hz=signal_freq_hz,
                signal_amp_v=signal_amp_v,
            )
        else:
            data = np.zeros(samples)

        blocks[band_name] = data

    return blocks