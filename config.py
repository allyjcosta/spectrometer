# config.py

PORT = "/dev/cu.usbmodem101"
BAUD = 2000000
SAMPLES = 2048

INSTRUMENT_LABEL = "SAMD21"

STATS_MIN_HZ = 50
STATS_MAX_HZ = 10000

Y_MIN = 1e-15
Y_MAX = 1e0

TEST_SIGNAL_AMP_V = 1e-3

TEST_FREQS_HZ = [
    0.1, 0.2, 0.5, 0.75,
    1, 5, 7.5, 10, 25, 50,
    100, 150, 500, 750, 1000, 1500,
    5000, 7500, 10000, 25000, 40000,
]


def make_bands(raw_sample_rate_hz):
    return {
        "LOW": {
            "f_min_Hz": 0.1,
            "stitch_min": 0.1,
            "stitch_max": (0.1 * SAMPLES) / 2,
        },
        "MID": {
            "f_min_Hz": 1.0,
            "stitch_min": (0.1 * SAMPLES) / 2,
            "stitch_max": (1.0 * SAMPLES) / 2,
        },
        "HIGH": {
            "f_min_Hz": raw_sample_rate_hz / SAMPLES,
            "stitch_min": (1.0 * SAMPLES) / 2,
            "stitch_max": None,
        },
    }


BAND_ORDER = ["LOW", "MID", "HIGH"]


PLOT_CONFIG = {
    "label": INSTRUMENT_LABEL,
    "linestyle": "-",
    "linewidth": 0.8,
    "y_min": Y_MIN,
    "y_max": Y_MAX,
}