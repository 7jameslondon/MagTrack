import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # ask Matplotlib to use Tk backend
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path

CSV_PATH = "benchmark_results.csv"
OUTFILE  = Path("plots_combined.png")
MATCH_AXES = True

def _plot_lines(ax, df_agg, ycol, yerr_col, ylabel, min_y = 1e50, max_y = -1e50):
    min_x = 1e50
    max_x = 0

    funcs = pd.unique(df_agg["function"].astype(str))
    cmap = plt.get_cmap("tab10")
    base_colors = {f: cmap(i % 10) for i, f in enumerate(funcs)}

    for (func, s), sub in df_agg.groupby(["function", "size"]):
        sub = sub.sort_values("n_images")
        x = sub["n_images"]
        y = sub[ycol].to_numpy()
        yerr = sub[yerr_col].to_numpy()

        sizes_in_func = sorted(df_agg.loc[df_agg["function"] == func, "size"].unique(), reverse=True)
        rank = sizes_in_func.index(s)
        n_levels = max(len(sizes_in_func) - 1, 1)
        lighten = 0.6 * (rank / n_levels)  # 0 -> darkest, up to 0.6 lighter
        r, g, b = mcolors.to_rgb(base_colors[func])
        c = (r + (1 - r) * lighten, g + (1 - g) * lighten, b + (1 - b) * lighten)

        ax.errorbar(x, y, yerr=yerr, fmt="-o", ms=3.5, lw=1.4, capsize=2.5, label=f"{func} | ROI {s}", color=c)
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min_x/2, max_x*2)
    ax.set_ylim(min_y/2, max_y*2)
    # ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.4)

    return min_y, max_y

def build_figure():
    df  = pd.read_csv(CSV_PATH)
    cpu = df[df["processor"] == "CPU"]
    gpu = df[df["processor"] == "GPU"]

    def _add_tp(d):
        d = d.copy()
        d["throughput"] = d["n_images"] / d["mean"]
        d["sd_throughput"] = d["n_images"] * d["sd"] / (d["mean"] ** 2)
        return d

    cpu = _add_tp(cpu)
    gpu = _add_tp(gpu)

    # Optional axis matching (based on CPU panels)
    r_xlim = (cpu["n_images"].min(), cpu["n_images"].max())
    r_ylim = (cpu["mean"].min(),    cpu["mean"].max())
    t_xlim = (cpu["n_images"].min(),  cpu["n_images"].max())
    t_ylim = (cpu["throughput"].min(),cpu["throughput"].max())

    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 9),
                            dpi=140,
                            constrained_layout=False,
                            sharex=True,
                            sharey=("row" if MATCH_AXES else False))
    fig.subplots_adjust(right=0.82)

    # Runtime Plots
    min_y, max_y = _plot_lines(axs[0,0], cpu, "mean", "sd",
                "Runtime (s)")

    _plot_lines(axs[0,1], gpu, "mean", "sd",
                "Runtime (s)", min_y, max_y)


    # Throughput Plots
    min_y, max_y = _plot_lines(axs[1,0], cpu, "throughput", "sd_throughput",
                "Throughput\n(images processed in 1 second)")

    _plot_lines(axs[1,1], gpu, "throughput", "sd_throughput",
                "Throughput\n(images processed in 1 second)", min_y, max_y)

    # Legend
    handles, labels = [], []
    for ax in axs.ravel():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    handles, labels = zip(*uniq) if uniq else ([], [])
    fig.legend(handles, labels, title="ROI", fontsize=8, frameon=True,
               loc="center left", bbox_to_anchor=(0.84, 0.5))

    # Save file
    fig.savefig(OUTFILE, bbox_inches="tight")

if __name__ == "__main__":
    build_figure()
    plt.show()
