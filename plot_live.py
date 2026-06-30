import matplotlib.pyplot as plt
import numpy as np


def create_live_plot(y_min, y_max, raw_sample_rate_hz, title="ADC Noise Spectrometer"):
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))

    plot = {
        "fig": fig,
        "ax": ax,
        "running": True,
        "x_min_hz": 0.1,
        "nyquist_hz": raw_sample_rate_hz / 2.0,
    }

    def on_close(_event):
        plot["running"] = False

    fig.canvas.mpl_connect("close_event", on_close)

    line, = ax.plot([], [], linewidth=0.8, label="Live ASD")

    median_line = ax.axhline(
        y=y_min,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Median (ASD)",
    )

    stats_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude Spectral Density (V/√Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(plot["x_min_hz"], plot["nyquist_hz"])
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    plot.update({
        "line": line,
        "median_line": median_line,
        "stats_text": stats_text,
    })
    return plot


def update_live_plot(plot, freqs_hz, asd_v_per_sqrt_hz, stats=None, y_min=None, y_max=None):
    fig = plot["fig"]
    ax = plot["ax"]
    line = plot["line"]

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    asd_v_per_sqrt_hz = np.asarray(asd_v_per_sqrt_hz, dtype=float)

    valid = (
        np.isfinite(freqs_hz)
        & np.isfinite(asd_v_per_sqrt_hz)
        & (freqs_hz >= plot["x_min_hz"])
        & (freqs_hz <= plot["nyquist_hz"])
        & (asd_v_per_sqrt_hz > 0)
    )

    if not np.any(valid):
        return

    line.set_data(freqs_hz[valid], asd_v_per_sqrt_hz[valid])

    ax.set_xlim(plot["x_min_hz"], plot["nyquist_hz"])

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    if stats is not None:
        median_v = stats.get("median_V_per_sqrtHz")

        if median_v is not None:
            plot["median_line"].set_ydata([median_v, median_v])

        median_nv = stats.get("median_nV_per_sqrtHz")

        plot["stats_text"].set_text(
            f"Median (ASD): {median_nv:.2f} nV/√Hz\n"
        )

    ax.legend(fontsize=8)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)
