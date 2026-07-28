import matplotlib.pyplot as plt
import numpy as np

from config import INPUT_REFERRED_GAIN, INPUT_REFERRED_GAIN_LABEL
from plot_style import apply_plot_style


def create_live_plot(
    y_min,
    y_max,
    raw_sample_rate_hz,
    min_frequency_hz,
    title="ADC Noise Spectrometer",
):
    apply_plot_style()
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))

    plot = {
        "fig": fig,
        "ax": ax,
        "running": True,
        "x_min_hz": float(min_frequency_hz),
        "nyquist_hz": raw_sample_rate_hz / 2.0,
    }

    def on_close(_event):
        plot["running"] = False

    fig.canvas.mpl_connect("close_event", on_close)

    line, = ax.plot([], [], linewidth=0.8, label="Live Input-Referred ASD")

    median_line = ax.axhline(
        y=y_min,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Median (Input-Referred ASD)",
    )

    mean_line = ax.axhline(
        y=y_min,
        linestyle="-.",
        linewidth=1.0,
        alpha=0.85,
        label="Mean (Input-Referred ASD)",
    )

    stats_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=13,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Input-Referred ASD (V/√Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(plot["x_min_hz"], plot["nyquist_hz"])
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show(block=False)

    plot.update({
        "line": line,
        "median_line": median_line,
        "mean_line": mean_line,
        "stats_text": stats_text,
    })
    return plot


def update_live_plot(plot, freqs_hz, asd_v_per_sqrt_hz, stats=None, y_min=None, y_max=None):
    fig = plot["fig"]
    ax = plot["ax"]
    line = plot["line"]

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    asd_v_per_sqrt_hz = np.asarray(asd_v_per_sqrt_hz, dtype=float) / INPUT_REFERRED_GAIN

    if stats is not None:
        stats = dict(stats)
        for key in (
            "mean_V_per_sqrtHz",
            "median_V_per_sqrtHz",
            "floor90_V_per_sqrtHz",
            "mean_nV_per_sqrtHz",
            "median_nV_per_sqrtHz",
            "floor90_nV_per_sqrtHz",
        ):
            if stats.get(key) is not None:
                stats[key] = stats[key] / INPUT_REFERRED_GAIN

    positive_frequencies = freqs_hz[np.isfinite(freqs_hz) & (freqs_hz > 0)]
    if positive_frequencies.size:
        plot["x_min_hz"] = float(np.min(positive_frequencies))

    valid = (
        np.isfinite(freqs_hz)
        & np.isfinite(asd_v_per_sqrt_hz)
        & (freqs_hz >= plot["x_min_hz"])
        & (freqs_hz <= plot["nyquist_hz"])
        & (asd_v_per_sqrt_hz > 0)
    )

    if not np.any(valid):
        return

    plot_freqs = freqs_hz.copy()
    plot_asd = asd_v_per_sqrt_hz.copy()
    separators = np.isnan(plot_freqs) | np.isnan(plot_asd)
    plot_freqs[~valid & ~separators] = np.nan
    plot_asd[~valid & ~separators] = np.nan

    line.set_data(plot_freqs, plot_asd)

    ax.set_xlim(plot["x_min_hz"], plot["nyquist_hz"])

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    if stats is not None:
        median_v = stats.get("median_V_per_sqrtHz")

        if median_v is not None:
            plot["median_line"].set_ydata([median_v, median_v])

        mean_v = stats.get("mean_V_per_sqrtHz")
        if mean_v is not None:
            plot["mean_line"].set_ydata([mean_v, mean_v])

        mean_nv = stats.get("mean_nV_per_sqrtHz")
        median_nv = stats.get("median_nV_per_sqrtHz")
        stats_min_hz = stats.get("stats_min_Hz")
        stats_max_hz = stats.get("stats_max_Hz")

        stat_lines = []
        if mean_nv is not None:
            stat_lines.append(f"Mean (ASD): {mean_nv:.2f} nV/√Hz")
        else:
            stat_lines.append("Mean (ASD): N/A")

        if median_nv is not None:
            stat_lines.append(f"Median (ASD): {median_nv:.2f} nV/√Hz")
        else:
            stat_lines.append("Median (ASD): N/A")

        if stats_min_hz is not None and stats_max_hz is not None:
            stat_lines.append(
                f"Stats band: {stats_min_hz:g}-{stats_max_hz:g} Hz"
            )
        stat_lines.append(f"Input gain: {INPUT_REFERRED_GAIN_LABEL}")

        plot["stats_text"].set_text("\n".join(stat_lines))

    ax.legend()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)
