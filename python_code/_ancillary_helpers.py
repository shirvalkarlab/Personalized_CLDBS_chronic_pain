"""
_ancillary_helpers.py
================================================================================
Shared plotting + fitting helpers for the ancillary behavioral score figures
(main Figure 6 d/e and Extended Figure 3 b/c).

Shared drawing helpers, so every box + swarm panel in this deposit renders with
the same conventions:

  * draw_summary_box     -> IQR rectangle + black median + 1.5*IQR whiskers
  * draw_box_swarm       -> summary boxes with each survey swarmed on top
  * condition_box_face   -> pale grey (baseline/sham) vs pale blue (on therapy)
  * PARTICIPANT_COLORS   -> Okabe-Ito colours per participant (colourblind safe)
  * format_p, add_panel_letter -> p-value text and panel letters

Also carries the analysis reused by both plotters:
  * fit_mixed_effects    -> per-measure MixedLM(value ~ phase + (1|patient))
  * fit_constructs       -> orient + z within participant, MixedLM per construct
  * mcid_responders      -> per-patient MCID responder counts

Both models are fitted by maximum likelihood (Powell optimizer); confidence
intervals and p values use a t distribution on n_obs - 2 degrees of freedom.

Responder status is reported as MCID counts only. The pre-implant baseline
sample is too small to support a stable reliable-change index.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats


# ---------------------------------------------------------------------------
# Colour + measure conventions
# ---------------------------------------------------------------------------
PARTICIPANT_COLORS = {
    "P1": (0.902, 0.624, 0.000),
    "P2": (0.337, 0.706, 0.914),
    "P3": (0.000, 0.620, 0.451),
    "P4": (0.800, 0.475, 0.655),
    "P5": (0.835, 0.369, 0.000),
}

BASELINE_FACE = (0.90, 0.90, 0.90)   # light grey  -> baseline / sham / pre
CHRONIC_FACE = (0.80, 0.87, 0.95)    # light blue  -> chronic / on-therapy
CHRONIC_LINE = (0.00, 0.447, 0.698)  # saturated blue for construct forest
GREY_LINE = (0.55, 0.55, 0.55)

# Direction of "better" for each measure.
HIGHER_IS_BETTER = {
    "sf36_composite": True,
    "randPhysFxn": True, "randRolePhysical": True, "randRoleEmotional": True,
    "randEnergy": True, "randEmotion": True, "randSocial": True,
    "randPain": True, "randGeneral": True,
    "BDI": False, "BAI": False, "HamD6": False,
}

CONSTRUCT_OF = {
    "randPain": "pain",
    "randPhysFxn": "function", "randRolePhysical": "function",
    "randEnergy": "function", "randSocial": "function", "randGeneral": "function",
    "BDI": "mood", "BAI": "mood", "HamD6": "mood",
    "randEmotion": "mood", "randRoleEmotional": "mood",
}

MCID = {
    "sf36_composite": 10, "randPhysFxn": 10, "randRolePhysical": 12,
    "randRoleEmotional": 12, "randEnergy": 10, "randEmotion": 10,
    "randSocial": 12, "randPain": 10, "randGeneral": 10,
    "BDI": 5, "BAI": 5, "HamD6": 3,
}


def condition_box_face(condition: str) -> tuple:
    """Pale grey for baseline / sham / pre; pale blue otherwise."""
    c = condition.lower()
    if any(tag in c for tag in ("sham", "baseline", "pre")):
        return BASELINE_FACE
    return CHRONIC_FACE


def format_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def print_p(p: float) -> str:
    """p as printed in the paper: 3 decimals, 4 below 0.001, '<0.0001' below that."""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def write_reference_tables(mixed: pd.DataFrame, constructs: pd.DataFrame, data_dir: str):
    """Write the two reference CSVs at the precision printed in the paper."""
    m = mixed.rename(columns={"direction_better": "direction", "n_baseline": "n_base"}).copy()
    m[["coef", "ci_lo", "ci_hi"]] = m[["coef", "ci_lo", "ci_hi"]].round(1)
    m["p"] = m["p"].map(print_p)
    m.to_csv(os.path.join(data_dir, "mixed_effects_reference.csv"), index=False)
    c = constructs.copy()
    c[["coef_sd", "ci_lo", "ci_hi"]] = c[["coef_sd", "ci_lo", "ci_hi"]].round(2)
    c["p"] = c["p"].map(print_p)
    c.to_csv(os.path.join(data_dir, "construct_pooling_reference.csv"), index=False)


# ---------------------------------------------------------------------------
# Box + swarm helpers
# ---------------------------------------------------------------------------
def draw_summary_box(ax, values, x_center, box_width, face_color):
    """One clean summary box: IQR rectangle (black outline), median, 1.5*IQR
    whiskers with caps. NaNs ignored. No outlier markers (swarm sits on top).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    # Hazen quartiles (midpoint rule), as in the published box plots.
    q1, med, q3 = np.percentile(v, [25, 50, 75], method="hazen")
    iqr = q3 - q1
    whisker_low = max(v.min(), q1 - 1.5 * iqr)
    whisker_high = min(v.max(), q3 + 1.5 * iqr)
    half = box_width / 2

    ax.add_patch(plt.Rectangle(
        (x_center - half, q1), box_width, iqr,
        facecolor=face_color, edgecolor="k", linewidth=0.8, zorder=2))
    ax.plot([x_center - half, x_center + half], [med, med],
            "k-", linewidth=1.2, zorder=3)
    ax.plot([x_center, x_center], [q3, whisker_high],
            "k-", linewidth=0.8, zorder=2)
    ax.plot([x_center, x_center], [whisker_low, q1],
            "k-", linewidth=0.8, zorder=2)
    ax.plot([x_center - half / 2, x_center + half / 2],
            [whisker_high, whisker_high], "k-", linewidth=0.8, zorder=2)
    ax.plot([x_center - half / 2, x_center + half / 2],
            [whisker_low, whisker_low], "k-", linewidth=0.8, zorder=2)


def draw_box_swarm(ax, df, phases, xpos, valcol,
                   jitter=0.13, msize=18, malpha=0.8, rng_seed=0):
    """Grey/blue summary box for each phase at xpos, patient-coloured swarm.

    df       : rows with columns 'patient','phase','value' (or valcol).
    phases   : sequence of phase labels (e.g. ['baseline','chronic']).
    xpos     : scalar or per-phase list of x centres.
    """
    rng = np.random.default_rng(rng_seed)
    for j, phase in enumerate(phases):
        sub = df[df["phase"] == phase]
        v = sub[valcol].dropna().values
        if v.size == 0:
            continue
        xc = xpos if np.isscalar(xpos) else xpos[j]
        face = condition_box_face(phase)
        box_w = 0.42 if len(phases) == 1 else 0.5
        draw_summary_box(ax, v, xc, box_w, face)
        for pat, psub in sub.groupby("patient"):
            pv = psub[valcol].dropna().values
            if pv.size == 0:
                continue
            col = PARTICIPANT_COLORS.get(pat, (0.5, 0.5, 0.5))
            xs = xc + (rng.uniform(-jitter, jitter, size=pv.size))
            ax.scatter(xs, pv, s=msize, color=col, alpha=malpha,
                       edgecolors="none", zorder=4)


def add_panel_letter(ax, letter):
    """Bold lowercase panel letter, top-left, outside the axes."""
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Analysis (reused by both plotters)
# ---------------------------------------------------------------------------
def fit_mixed_effects(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for measure in long["measure"].unique():
            d = long[long["measure"] == measure].dropna(subset=["value"]).copy()
            if d.empty or d["phase"].nunique() < 2:
                continue
            d["phase_c"] = (d["phase"] == "chronic").astype(int)
            try:
                fit = smf.mixedlm("value ~ phase_c", d, groups=d["patient"]).fit(
                    method="powell", reml=False)
                coef = fit.fe_params["phase_c"]
                se = fit.bse_fe["phase_c"]
                dof = len(d) - 2
                t_crit = stats.t.ppf(0.975, dof)
                p = 2 * stats.t.sf(abs(coef / se), dof)
                rows.append({
                    "measure": measure,
                    "direction_better": "higher" if HIGHER_IS_BETTER.get(measure, True) else "lower",
                    "coef": coef,
                    "ci_lo": coef - t_crit * se, "ci_hi": coef + t_crit * se,
                    "p": p,
                    "n_obs": len(d),
                    "n_baseline": int((d["phase_c"] == 0).sum()),
                })
            except Exception as exc:
                print(f"  [warn] MixedLM {measure} failed: {exc}")
    return pd.DataFrame(rows)


def fit_constructs(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_z = long.copy()
    long_z["oriented"] = long_z.apply(
        lambda r: r["value"] * (1 if HIGHER_IS_BETTER.get(r["measure"], True) else -1),
        axis=1,
    )
    long_z["z"] = np.nan
    long_z["construct"] = long_z["measure"].map(CONSTRUCT_OF)

    for (pat, meas), sub in long_z.groupby(["patient", "measure"]):
        vals = sub["oriented"].dropna()
        if vals.empty:
            continue
        sd = vals.std(ddof=1)
        if sd and np.isfinite(sd):
            long_z.loc[sub.index, "z"] = (sub["oriented"] - vals.mean()) / sd
        else:
            long_z.loc[sub.index, "z"] = sub["oriented"] * 0.0

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for construct in ("pain", "function", "mood"):
            d = long_z[(long_z["construct"] == construct) & long_z["z"].notna()].copy()
            if d.empty or d["phase"].nunique() < 2:
                continue
            d["phase_c"] = (d["phase"] == "chronic").astype(int)
            try:
                fit = smf.mixedlm("z ~ phase_c", d, groups=d["patient"]).fit(
                    method="powell", reml=False)
                coef = fit.fe_params["phase_c"]
                se = fit.bse_fe["phase_c"]
                dof = len(d) - 2
                t_crit = stats.t.ppf(0.975, dof)
                p = 2 * stats.t.sf(abs(coef / se), dof)
                rows.append({
                    "construct": construct, "coef_sd": coef,
                    "ci_lo": coef - t_crit * se, "ci_hi": coef + t_crit * se,
                    "p": p, "n_obs": len(d),
                })
            except Exception as exc:
                print(f"  [warn] construct MixedLM {construct} failed: {exc}")
    return pd.DataFrame(rows), long_z


def mcid_responders(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per patient x measure: first baseline vs last chronic; reached_mcid = True
    if |change| >= MCID in the better direction."""
    long_sorted = long.sort_values(["patient", "measure", "phase", "date"])
    rows = []
    for (pat, meas), sub in long_sorted.groupby(["patient", "measure"]):
        base = sub[sub["phase"] == "baseline"]["value"].dropna()
        chron = sub[sub["phase"] == "chronic"]["value"].dropna()
        if base.empty or chron.empty:
            continue
        change = chron.iloc[-1] - base.iloc[0]
        hib = HIGHER_IS_BETTER.get(meas, True)
        improved = (change > 0) == hib
        mcid = MCID.get(meas, np.nan)
        rows.append({
            "measure": meas, "patient": pat, "change": change,
            "reached_mcid": bool(np.isfinite(mcid) and abs(change) >= mcid and improved),
        })
    det = pd.DataFrame(rows)
    summary = (det.groupby("measure")
                 .agg(n=("patient", "count"),
                      reached_mcid=("reached_mcid", "sum"))
                 .reset_index())
    return det, summary
