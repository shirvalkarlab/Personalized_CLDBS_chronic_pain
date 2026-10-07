"""
Figure S6 - Battery efficiency of closed-loop versus open-loop stimulation.

This script reproduces the manuscript's battery-efficiency figure. It answers a
single question: does closed-loop (adaptive) deep brain stimulation actually
save battery compared with conventional always-on open-loop stimulation?

Closed-loop stimulation runs an always-on power-domain sensing pipeline to
compute its control signal. That sensing draws current whether or not the
device is stimulating. Closed-loop usually stimulates at a lower effective duty
cycle than always-on open-loop, so it saves therapy energy; the question is
whether that saving is larger than the fixed battery cost of the sensing it
requires.

The figure has three panels:

  A. Measured battery discharge rate (milliamp-hours per day) grouped by the
     dominant control mode of each discharge segment (off/mixed, open-loop,
     closed-loop), shown as a box plot with the raw segment values swarmed on
     top. Each group is annotated with its median delivered-therapy energy.

  B. Per-hemisphere break-even closed-loop duty cycle: the duty cycle below
     which closed-loop saves battery relative to 100%-duty open-loop. The
     measured (actual) closed-loop duty cycle is drawn alongside for
     comparison. Hemispheres whose sensing cost already exceeds their
     full-duty therapy have no positive break-even and are labelled instead.

  C. Full-duty therapy drain versus the fixed power-domain sensing cost, per
     hemisphere. Hemispheres left of the sensing-cost line spend more battery
     on sensing than they could ever save on therapy, so closed-loop can never
     save battery for them.

DATA INPUTS (both in data/figureS6_battery_efficiency/):
  discharge_segments.csv - one row per inter-recharge battery-discharge
      segment, with the measured discharge rate, the delivered therapy energy,
      the fraction of the segment spent in each control mode, the home-
      streaming rate, and the dominant control mode.
  breakeven_duty.csv - one row per implanted hemisphere, with the fitted
      break-even and actual closed-loop duty cycles and the save/no-save status.

Panels A and B read these two files directly. The mixed-effects drain model
that sets the sensing-cost line in panel C is re-fitted at run time from
discharge_segments.csv, so all three panels follow from the two files above.

USAGE:
    python plot_figureS6_battery_efficiency.py
    # optional: set the repository root if running from elsewhere
    NATMED_REPO=/path/to/Shirvalkar_2026_NatMed_deposit \
        python plot_figureS6_battery_efficiency.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repository root: the folder that contains data/, figures/, python_code/.
# Defaults to two levels up from this file (python_code/ -> repository root).
repository_root = os.environ.get(
    "NATMED_REPO",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_directory = os.path.join(
    repository_root, "data", "figureS6_battery_efficiency")
figure_directory = os.path.join(repository_root, "figures")

segments_file = os.path.join(data_directory, "discharge_segments.csv")
breakeven_file = os.path.join(data_directory, "breakeven_duty.csv")

# Discharge segments at or above this rate (mAh/day) are excluded from both the
# drain model and panel A.
maximum_discharge_rate = 60


# ---------------------------------------------------------------------------
# Figure appearance
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fit_sensing_cost(segments):
    """Re-fit the mixed-effects battery-drain model to recover the sensing cost.

    Battery discharge rate is modelled as a per-patient baseline plus a term
    for delivered therapy energy, a term for the closed-loop (power-domain
    sensing) fraction, and a term for the home-streaming rate. The coefficient
    on the closed-loop fraction is the marginal battery cost of running the
    power-domain sensing, in milliamp-hours per day, which panel C draws as a
    vertical line.
    """
    data = segments.dropna(subset=["discharge_mAh_per_day", "teed_J_per_day",
                                   "frac_closed_loop"]).copy()
    data = data[(data["dur_days"] >= 1)
                & (data["discharge_mAh_per_day"] > 0)
                & (data["discharge_mAh_per_day"] < maximum_discharge_rate)]
    data["streaming_per_day"] = data["streaming_per_day"].fillna(0).clip(0, 20)

    model = smf.mixedlm(
        "discharge_mAh_per_day ~ teed_J_per_day + frac_closed_loop + streaming_per_day",
        data, groups=data["patient"]).fit(method="lbfgs", reml=True)
    return model.params["frac_closed_loop"]


def draw_box_with_swarm(axis, groups, labels, colors, ylabel, title, point_size=10):
    """Box plot with the raw points swarmed on top.

    Box outline, whiskers, and caps use a light 0.5-weight stroke in the group
    colour with a translucent fill and no outlier markers; the raw points are
    drawn on top with a small horizontal jitter so the whole distribution is
    visible rather than summarized away.
    """
    for index, (values, color) in enumerate(zip(groups, colors)):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        axis.boxplot(values, positions=[index], widths=0.55, patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color=color, lw=1.2),
                     boxprops=dict(facecolor=color, alpha=0.12, edgecolor=color, lw=0.5),
                     whiskerprops=dict(color=color, lw=0.5),
                     capprops=dict(color=color, lw=0.5))
        jitter = np.random.default_rng(index).normal(index, 0.06, size=len(values))
        axis.scatter(jitter, values, s=point_size, c=color, alpha=0.55,
                     edgecolor="none", zorder=3)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(0, None)


def make_battery_efficiency_figure(segments, breakeven, sensing_cost, output_file):
    """Draw the three-panel battery-efficiency composite and save it as a PDF."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    # ---- Panel A: measured discharge rate by dominant control mode ---------
    loops = ["mixed_or_off", "open_loop", "closed_loop"]
    loop_labels = ["Off / mixed", "Open-loop", "Closed-loop\n(power-domain)"]
    loop_colors = ["#8C8C8C", "#2166AC", "#B2182B"]
    plotted = segments[(segments["dur_days"] >= 1)
                       & (segments["discharge_mAh_per_day"] > 0)
                       & (segments["discharge_mAh_per_day"] < maximum_discharge_rate)]
    groups = [plotted[plotted["dominant_loop"] == loop]["discharge_mAh_per_day"].values
              for loop in loops]
    draw_box_with_swarm(axes[0], groups, loop_labels, loop_colors,
                        "Battery discharge (mAh / day)",
                        "A. Measured discharge by loop mode")
    for index, loop in enumerate(loops):
        median_energy = plotted[plotted["dominant_loop"] == loop]["teed_J_per_day"].median()
        axes[0].annotate("%.1f J/day\ntherapy" % median_energy, (index, 0.5),
                         ha="center", va="bottom", fontsize=6.5, color=loop_colors[index])

    # ---- Panel B: per-hemisphere break-even versus actual duty --------------
    rows = np.arange(len(breakeven))
    is_real = (breakeven["breakeven_cl_duty"] > 0) & (breakeven["breakeven_cl_duty"] <= 1)
    real_rows = breakeven[is_real]
    axes[1].errorbar(
        real_rows["breakeven_cl_duty"] * 100, real_rows.index,
        xerr=[(real_rows["breakeven_cl_duty"] - real_rows["breakeven_lo"]).clip(lower=0) * 100,
              (real_rows["breakeven_hi"] - real_rows["breakeven_cl_duty"]).clip(lower=0) * 100],
        fmt="o", color="#B2182B", capsize=3, ms=6,
        label="Break-even closed-loop duty", zorder=4)
    axes[1].scatter(breakeven["actual_cl_duty"] * 100, rows, marker="D",
                    color="#2166AC", zorder=5, label="Actual closed-loop duty (median)")
    axes[1].axvline(100, color="gray", ls=":", lw=1, label="100% open-loop reference")
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_yticks(rows)
    axes[1].set_yticklabels(breakeven["hem"])
    axes[1].set_xlabel("Closed-loop effective duty cycle (%)")
    axes[1].set_title("B. Break-even vs actual duty (per hemisphere)")
    axes[1].set_xlim(0, 150)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   frameon=False, fontsize=8)
    label_by_status = {
        "saves": ("saves", "#1a7f37"),
        "no save": ("no save", "#B2182B"),
        "CL never saves": ("never saves*", "#B2182B"),
        "CL always saves": ("always saves", "#1a7f37"),
        "no CL data": ("n/a", "#666")}
    for index, record in breakeven.iterrows():
        text, color = label_by_status.get(record["status"], (record["status"], "#666"))
        axes[1].annotate(text, (147, index), ha="right", va="center",
                         fontsize=7, color=color)
    axes[1].text(103, 0.3,
                 "* sensing cost exceeds\n  full-duty therapy;\n  no positive break-even",
                 fontsize=6.3, color="#B2182B", va="bottom")

    # ---- Panel C: full-duty therapy drain versus fixed sensing cost --------
    axes[2].scatter(breakeven["therapy_drain_full_mAhday"], rows, s=65,
                    color="#4D4D4D", zorder=5, label="Therapy drain @100% duty")
    axes[2].axvline(sensing_cost, color="#B2182B", lw=2,
                    label="Power-domain sensing cost")
    axes[2].fill_betweenx([-0.5, len(breakeven) - 0.5], 0, sensing_cost,
                          color="#B2182B", alpha=0.12)
    axes[2].set_yticks(rows)
    axes[2].set_yticklabels(breakeven["hem"])
    axes[2].set_xlabel("Battery drain (mAh / day)")
    axes[2].set_title("C. Can closed-loop ever save battery?")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   frameon=False, fontsize=8)
    axes[2].text(sensing_cost + 0.15, len(breakeven) - 1.3,
                 "left of line:\nsensing > therapy,\nCL never saves",
                 fontsize=7, color="#B2182B", va="top")
    axes[2].set_ylim(-0.5, len(breakeven) - 0.5)

    figure.tight_layout()
    figure.savefig(output_file, bbox_inches="tight")
    plt.close(figure)


def main():
    segments = pd.read_csv(segments_file)
    breakeven = pd.read_csv(breakeven_file)

    def relabel(hemisphere):
        # Split the participant label from the side for display: "P3L" -> "P3 L".
        return "%s %s" % (hemisphere[:-1], hemisphere[-1])

    breakeven = breakeven.copy()
    breakeven["hem"] = breakeven["hem"].map(relabel)

    sensing_cost = fit_sensing_cost(segments)

    output_file = os.path.join(figure_directory, "FigureS6_battery_efficiency.pdf")
    make_battery_efficiency_figure(segments, breakeven, sensing_cost, output_file)
    print("wrote", output_file)
    print("fitted power-domain sensing cost: %.2f mAh/day" % sensing_cost)


if __name__ == "__main__":
    main()
