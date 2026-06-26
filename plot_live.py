import matplotlib.pyplot as plt
import numpy as np


def create_live_plot(y_min, y_max, raw_sample_rate_hz, title="ADC Noise Spectrometer"):
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))

    line, = ax.plot([], [], linewidth=0.8, label="Live ASD")

    median_line = ax.axhline(
        y=y_min,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Median",
    )

    floor_line = ax.axhline(
        y=y_min,
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="90% Floor",
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

    control_text = ax.text(
        0.02,
        0.98,
        "Keys: s = save, p = compare, q = quit",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude Spectral Density (V/√Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, raw_sample_rate_hz / 2)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    return {
        "fig": fig,
        "ax": ax,
        "line": line,
        "median_line": median_line,
        "floor_line": floor_line,
        "stats_text": stats_text,
        "control_text": control_text,
    }


def update_live_plot(plot, freqs_hz, asd_v_per_sqrt_hz, stats=None, y_min=None, y_max=None):
    fig = plot["fig"]
    ax = plot["ax"]
    line = plot["line"]

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    asd_v_per_sqrt_hz = np.asarray(asd_v_per_sqrt_hz, dtype=float)

    valid = (
        np.isfinite(freqs_hz)
        & np.isfinite(asd_v_per_sqrt_hz)
        & (freqs_hz > 0)
        & (asd_v_per_sqrt_hz > 0)
    )

    if not np.any(valid):
        return

    line.set_data(freqs_hz[valid], asd_v_per_sqrt_hz[valid])

    ax.set_xlim(np.min(freqs_hz[valid]), np.max(freqs_hz[valid]))

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    if stats is not None:
        median_v = stats.get("median_V_per_sqrtHz")
        floor90_v = stats.get("floor90_V_per_sqrtHz")

        if median_v is not None:
            plot["median_line"].set_ydata([median_v, median_v])

        if floor90_v is not None:
            plot["floor_line"].set_ydata([floor90_v, floor90_v])

        median_nv = stats.get("median_nV_per_sqrtHz")
        floor90_nv = stats.get("floor90_nV_per_sqrtHz")

        plot["stats_text"].set_text(
            f"Median: {median_nv:.2f} nV/√Hz\n"
            f"90% Floor: {floor90_nv:.2f} nV/√Hz"
        )

    ax.legend(fontsize=8)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)