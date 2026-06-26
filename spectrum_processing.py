from FFT_analysis import compute_fft_analysis
from stitching import stitch_latest_results


def process_blocks(blocks, bands, band_order, samples):
    band_results_asd = {}
    band_results_mag = {}
    band_metadata = {}

    for band_name in band_order:
        if band_name not in blocks:
            continue

        data = blocks[band_name]

        if len(data) < samples:
            print(f"Warning: {band_name} expected {samples}, got {len(data)}")
            continue

        data = data[:samples]

        band = bands[band_name]
        fft_sample_rate_hz = band["f_min_Hz"] * samples

        analysis = compute_fft_analysis(
            data=data,
            sample_rate_hz=fft_sample_rate_hz,
        )

        band_results_asd[band_name] = {
            "freqs": analysis["freqs_Hz"],
            "amps_V": analysis["asd_V_per_sqrtHz"],
        }

        band_results_mag[band_name] = {
            "freqs": analysis["freqs_Hz"],
            "amps_V": analysis["mag_V"],
        }

        band_metadata[band_name] = {
            "bin_width_Hz": analysis["bin_width_Hz"],
            "enbw_Hz": analysis["enbw_Hz"],
            "fft_sample_rate_Hz": fft_sample_rate_hz,
        }

    stitched_freqs, stitched_asd = stitch_latest_results(
        band_results_asd,
        bands,
        band_order,
    )

    _, stitched_mag = stitch_latest_results(
        band_results_mag,
        bands,
        band_order,
    )

    return {
        "freqs_Hz": stitched_freqs,
        "asd_V_per_sqrtHz": stitched_asd,
        "mag_V": stitched_mag,
        "band_results_mag": band_results_mag,
        "band_metadata": band_metadata,
    }