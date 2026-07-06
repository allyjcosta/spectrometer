# config.py

PORT = "/dev/cu.usbmodem101"
BAUD = 2000000
SAMPLES = 2048

#INSTRUMENT_LABEL = "SAMD21"
#INSTRUMENT_LABEL = "ADS1115"
INSTRUMENT_LABEL = "MCP3202"
#INSTRUMENT_LABEL = "ATMEGA328"

STATS_MIN_HZ = 50
STATS_MAX_HZ = 10000

Y_MIN = 1e-15
Y_MAX = 1e0

TEST_SIGNAL_AMP_V = 500e-3

TEST_FREQS_HZ = [
    0.1, 0.2, 0.5, 0.75,
    1, 5, 7.5, 10, 25, 50,
    100, 150, 500, 750, 1000, 1500,
    5000, 7500, 10000, 25000, 40000,
]


def make_bands(raw_sample_rate_hz):
    raw_sample_rate_hz = float(raw_sample_rate_hz)
    if raw_sample_rate_hz <= 0:
        raise ValueError("Raw sample rate must be positive.")

    candidates = [
        ("LOW", 0.1),
        ("MID", 1.0),
    ]

    # A decimated band is useful only when its FFT rate is below the raw rate.
    active = [
        (name, f_min_hz)
        for name, f_min_hz in candidates
        if f_min_hz * SAMPLES < raw_sample_rate_hz
    ]
    active.append(("HIGH", raw_sample_rate_hz / SAMPLES))

    bands = {}
    # The first positive FFT bin is one bin-width above DC. For a device
    # without a decimated LOW band, this follows its measured sample rate.
    stitch_min_hz = active[0][1]
    for name, f_min_hz in active:
        band_nyquist_hz = (f_min_hz * SAMPLES) / 2.0
        bands[name] = {
            "f_min_Hz": f_min_hz,
            "stitch_min": stitch_min_hz,
            "stitch_max": None if name == "HIGH" else band_nyquist_hz,
        }
        stitch_min_hz = band_nyquist_hz

    return bands


BAND_ORDER = ["LOW", "MID", "HIGH"]


def active_band_order(bands):
    return [name for name in BAND_ORDER if name in bands]


def band_for_frequency(bands, frequency_hz):
    for name in active_band_order(bands):
        band = bands[name]
        if frequency_hz < band["stitch_min"]:
            continue
        if band["stitch_max"] is None or frequency_hz < band["stitch_max"]:
            return name
    return None


PLOT_CONFIG = {
    "label": INSTRUMENT_LABEL,
    "linestyle": "-",
    "linewidth": 0.8,
    "y_min": Y_MIN,
    "y_max": Y_MAX,
}
