import numpy as np


def stitch_latest_results(band_results, bands, band_order):
    stitched_freqs = []
    stitched_amps = []

    for band_name in band_order:
        if band_name not in band_results:
            continue

        band = bands[band_name]
        freqs = band_results[band_name]["freqs"]
        amps = band_results[band_name]["amps_V"]

        mask = freqs >= band["stitch_min"]

        if band["stitch_max"] is not None:
            mask &= freqs < band["stitch_max"]

        band_freqs = freqs[mask]
        band_amps = amps[mask]

        if len(band_freqs) == 0:
            continue

        order = np.argsort(band_freqs)
        band_freqs = band_freqs[order]
        band_amps = band_amps[order]

        stitched_freqs.append(band_freqs)
        stitched_amps.append(band_amps)

    if len(stitched_freqs) == 0:
        return None, None

    stitched_freqs = np.concatenate(stitched_freqs)
    stitched_amps = np.concatenate(stitched_amps)

    return stitched_freqs, stitched_amps
