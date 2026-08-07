import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

from plot_style import apply_plot_style


TIME_DOMAIN_Y_PADDING_FRACTION = 0.08
TIME_DOMAIN_MIN_Y_SPAN_V = 1e-6
TIME_DOMAIN_MAX_DISPLAY_SECONDS = 0.100
TIME_DOMAIN_MAJOR_TICK_SECONDS = 0.010
TIME_DOMAIN_MINOR_TICK_SECONDS = 0.002
TIME_DOMAIN_MAJOR_TICK_VOLTS = 0.001
TIME_DOMAIN_MINOR_TICK_VOLTS = 0.0002


def available_block_names(blocks):
    return [
        name
        for name, data in blocks.items()
        if data is not None and len(data) > 0
    ]


def block_time_axis(data, sample_rate_hz):
    if sample_rate_hz is not None and sample_rate_hz > 0:
        return np.arange(len(data)) / sample_rate_hz, "Time (s)"

    return np.arange(len(data)), "Sample Index"


def block_stats_text(data):
    mean_v = float(np.mean(data))
    min_v = float(np.min(data))
    max_v = float(np.max(data))
    vpp = max_v - min_v
    amplitude_v = 0.5 * vpp
    ac_rms_v = float(np.std(data - mean_v))
    return (
        f"Mean: {mean_v:.6g} V\n"
        f"Vpp: {vpp:.6g} V\n"
        f"Amp: {amplitude_v:.6g} V\n"
        f"AC RMS: {ac_rms_v:.6g} V"
    )


def padded_limits(values, padding_fraction=TIME_DOMAIN_Y_PADDING_FRACTION):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return -1.0, 1.0

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max(max_value - min_value, TIME_DOMAIN_MIN_Y_SPAN_V)
    center = 0.5 * (min_value + max_value)
    half_span = 0.5 * span * (1.0 + 2.0 * padding_fraction)

    return center - half_span, center + half_span


def configure_time_axis(ax, start_seconds, display_seconds):
    if display_seconds <= 0.2:
        ax.set_xlabel("Time in window (ms)")
        ax.xaxis.set_major_locator(MultipleLocator(TIME_DOMAIN_MAJOR_TICK_SECONDS))
        ax.xaxis.set_minor_locator(MultipleLocator(TIME_DOMAIN_MINOR_TICK_SECONDS))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{(value - start_seconds) * 1e3:.0f}")
        )
    else:
        ax.set_xlabel("Time (s)")
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))


def configure_voltage_axis(ax):
    ax.yaxis.set_major_locator(MultipleLocator(TIME_DOMAIN_MAJOR_TICK_VOLTS))
    ax.yaxis.set_minor_locator(MultipleLocator(TIME_DOMAIN_MINOR_TICK_VOLTS))


def create_time_domain_plot(
    blocks,
    sample_rates_hz=None,
    title="Time Domain Signal",
):
    apply_plot_style()
    plt.ion()

    if sample_rates_hz is None:
        sample_rates_hz = {}

    band_names = available_block_names(blocks)
    if not band_names:
        print("No time-domain blocks are available to plot.")
        return None

    fig, axes = plt.subplots(
        len(band_names),
        1,
        figsize=(14, 3.2 * len(band_names)),
        sharex=False,
    )

    if len(band_names) == 1:
        axes = [axes]

    plot = {
        "fig": fig,
        "axes": {},
        "lines": {},
        "stats_text": {},
        "band_names": band_names,
        "running": True,
    }

    def on_close(_event):
        plot["running"] = False

    fig.canvas.mpl_connect("close_event", on_close)

    for ax, band_name in zip(axes, band_names):
        line, = ax.plot([], [], linewidth=0.8)
        stats_text = ax.text(
            0.01,
            0.95,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_ylabel("Voltage (V)")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)

        plot["axes"][band_name] = ax
        plot["lines"][band_name] = line
        plot["stats_text"][band_name] = stats_text

    fig.suptitle(title)
    update_time_domain_plot(plot, blocks, sample_rates_hz)
    plt.show(block=False)
    return plot


def update_time_domain_plot(plot, blocks, sample_rates_hz=None):
    if plot is None or not plot.get("running", False):
        return

    if sample_rates_hz is None:
        sample_rates_hz = {}

    for band_name in plot["band_names"]:
        if band_name not in blocks:
            continue

        data = np.asarray(blocks[band_name], dtype=float)
        if len(data) == 0:
            continue

        sample_rate_hz = sample_rates_hz.get(band_name)
        time_axis, xlabel = block_time_axis(data, sample_rate_hz)

        if sample_rate_hz is not None and sample_rate_hz > 0:
            rate_label = f"{sample_rate_hz:.3f} Hz"
        else:
            rate_label = "unknown rate"

        ax = plot["axes"][band_name]
        line = plot["lines"][band_name]
        line.set_data(time_axis, data)

        ax.set_title(f"{band_name} band ({rate_label})")
        if len(time_axis) == 1:
            center = float(time_axis[0])
            ax.set_xlim(center - 0.5, center + 0.5)
            ax.set_xlabel(xlabel)
        elif sample_rate_hz is not None and sample_rate_hz > 0:
            display_seconds = min(TIME_DOMAIN_MAX_DISPLAY_SECONDS, float(time_axis[-1] - time_axis[0]))
            start_seconds = float(time_axis[-1] - display_seconds)
            ax.set_xlim(start_seconds, float(time_axis[-1]))
            configure_time_axis(ax, start_seconds, display_seconds)
        else:
            ax.set_xlim(float(time_axis[0]), float(time_axis[-1]))
            ax.set_xlabel(xlabel)

        ax.set_ylim(*padded_limits(data))
        configure_voltage_axis(ax)
        plot["stats_text"][band_name].set_text(block_stats_text(data))

    plot["fig"].tight_layout()
    plot["fig"].canvas.draw_idle()
    plot["fig"].canvas.flush_events()
    plt.pause(0.01)


def plot_time_domain_blocks(blocks, sample_rates_hz=None, title="Time Domain Signal"):
    return create_time_domain_plot(
        blocks=blocks,
        sample_rates_hz=sample_rates_hz,
        title=title,
    )


if __name__ == "__main__":
    print(
        "plot_time_domain.py is a helper module. "
        "Run live mode from main.py and enter command 9 after at least one "
        "capture to open a live-updating LOW/HIGH time-domain plot."
    )
