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

        stitched_freqs.append(freqs[mask])
        stitched_amps.append(amps[mask])

    if len(stitched_freqs) == 0:
        return None, None

    stitched_freqs = np.concatenate(stitched_freqs)
    stitched_amps = np.concatenate(stitched_amps)

    order = np.argsort(stitched_freqs)
    return stitched_freqs[order], stitched_amps[order]