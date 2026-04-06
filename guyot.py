"""
Guyot et al. (2012) algorithm for reconstructing Individual Patient Data (IPD)
from published Kaplan-Meier curves.

Python port of R package `reconstructKM` (Ryan Sun).
Reference: Guyot P, et al. BMC Med Res Methodol 2012;12:9.
"""

import numpy as np
import pandas as pd


def collapse_corners(df):
    """
    Preprocess digitized KM coordinates:
    - Group by time, take min(survival) per time
    - Enforce monotone non-increasing via cummin
    - Clamp to [0, 1]
    - Ensure (0, 1) start point exists
    """
    df = df.copy()
    df = df.groupby("time", as_index=False).agg({"survival": "min"})
    df = df.sort_values("time").reset_index(drop=True)
    df["survival"] = np.minimum.accumulate(np.minimum(df["survival"], 1.0))
    df.loc[df["survival"] < 0, "survival"] = 0.0

    # Ensure (0, 1) start
    if len(df) == 0 or df.iloc[0]["time"] != 0 or df.iloc[0]["survival"] != 1.0:
        start = pd.DataFrame({"time": [0.0], "survival": [1.0]})
        df = pd.concat([start, df[~((df["time"] == 0) & (df["survival"] == 1.0))]], ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)

    return df


def format_raw_tabs(raw_NAR, raw_surv):
    """
    Port of reconstructKM::format_raw_tabs

    Parameters
    ----------
    raw_NAR : pd.DataFrame with columns ['time', 'NAR']
    raw_surv : pd.DataFrame with columns ['time', 'survival']

    Returns
    -------
    dict with 'aug_NAR' and 'aug_surv' DataFrames
    """
    # Validate raw_surv
    surv = raw_surv[["time", "survival"]].copy()
    surv = surv.sort_values("time").reset_index(drop=True)

    if surv["survival"].is_monotonic_decreasing is False:
        # Check non-increasing (allows equal values)
        diffs = np.diff(surv["survival"].values)
        if np.any(diffs > 1e-10):
            raise ValueError("survival must be non-increasing in time")

    if surv.iloc[0]["time"] != 0 or surv.iloc[0]["survival"] != 1.0:
        raise ValueError("raw_surv must start with t=0, S=1")

    # Validate raw_NAR
    nar = raw_NAR[["time", "NAR"]].copy()
    nar = nar.sort_values("time").reset_index(drop=True)

    # Build interval indices (1-based in R, 0-based here)
    n_intervals = len(nar) - 1
    lower = [None] * n_intervals
    upper = [None] * n_intervals

    for i in range(n_intervals):
        t_lo = nar["time"].iloc[i]
        t_hi = nar["time"].iloc[i + 1]
        mask = (surv["time"] >= t_lo) & (surv["time"] < t_hi)
        rows = np.where(mask)[0]
        if len(rows) > 0:
            lower[i] = int(rows.min())
            upper[i] = int(rows.max())

    # Filter out intervals with no clicks
    valid = [(l, u) for l, u in zip(lower, upper) if l is not None]
    valid_idx = [i for i, l in enumerate(lower) if l is not None]

    aug_lower = [v[0] for v in valid]
    aug_upper = [v[1] for v in valid]
    aug_time = [nar["time"].iloc[i] for i in valid_idx]
    aug_nar_vals = [nar["NAR"].iloc[i] for i in valid_idx]

    # Last NAR row
    fup_end = nar["time"].max()
    last_surv_val = surv["survival"].iloc[-1]
    last_upper = aug_upper[-1] if aug_upper else 0

    aug_time.append(fup_end)
    aug_nar_vals.append(int(nar["NAR"].min()))
    aug_lower.append(last_upper + 1)
    aug_upper.append(last_upper + 1)

    aug_NAR = pd.DataFrame({
        "time": aug_time,
        "NAR": aug_nar_vals,
        "lower": aug_lower,
        "upper": aug_upper
    })

    # Last surv row
    last_surv_row = pd.DataFrame({"time": [fup_end], "survival": [last_surv_val]})
    aug_surv = pd.concat([surv, last_surv_row], ignore_index=True)

    # Rename survival -> surv for KM_reconstruct compatibility
    aug_surv = aug_surv.rename(columns={"survival": "surv"})

    return {"aug_NAR": aug_NAR, "aug_surv": aug_surv}


def KM_reconstruct(aug_NAR, aug_surv):
    """
    Port of reconstructKM::KM_reconstruct

    Parameters
    ----------
    aug_NAR : pd.DataFrame with columns ['time', 'NAR', 'lower', 'upper']
    aug_surv : pd.DataFrame with columns ['time', 'surv']

    Returns
    -------
    dict with 'IPD_time', 'IPD_event', and diagnostic arrays
    """
    TAR = aug_NAR["time"].values.astype(float)
    NAR = aug_NAR["NAR"].values.astype(float).copy()
    lower = aug_NAR["lower"].values.astype(int)
    upper = aug_NAR["upper"].values.astype(int)
    t_surv = aug_surv["time"].values.astype(float)
    surv = aug_surv["surv"].values.astype(float)

    total_ints = len(NAR)
    total_e_times = upper[total_ints - 1]  # last upper index

    int_censor = np.zeros(total_ints - 1, dtype=float)
    last_event = np.ones(total_ints, dtype=int)  # 1-based "last" tracker

    n_hat = np.full(total_e_times + 1, NAR[0] + 1, dtype=float)
    n_cen = np.zeros(total_e_times + 1, dtype=float)
    n_event = np.zeros(total_e_times + 1, dtype=float)
    KM_hat = np.ones(total_e_times + 1, dtype=float)

    for int_idx in range(total_ints - 1):
        lo = lower[int_idx]
        hi = upper[int_idx]
        lo_next = lower[int_idx + 1]

        if surv[lo] == 0:
            int_censor[int_idx] = 0
        else:
            int_censor[int_idx] = round(
                NAR[int_idx] * surv[lo_next] / surv[lo] - NAR[int_idx + 1]
            )

        # Iterative loop to match n_hat at next boundary to NAR
        max_iter = 1000
        iteration = 0
        while iteration < max_iter:
            iteration += 1

            if int_censor[int_idx] <= 0:
                n_cen[lo:hi + 1] = 0
                int_censor[int_idx] = 0
            else:
                # Distribute censorings uniformly
                cen_count = int(int_censor[int_idx])
                cen_times = (
                    t_surv[lo]
                    + np.arange(1, cen_count + 1)
                    * (t_surv[lo_next] - t_surv[lo])
                    / (cen_count + 1)
                )
                # Histogram to count censorings per click interval
                breaks = t_surv[lo:lo_next + 1]
                if len(breaks) > 1 and len(cen_times) > 0:
                    counts, _ = np.histogram(cen_times, bins=breaks)
                    n_cen[lo:lo + len(counts)] = counts
                else:
                    n_cen[lo:hi + 1] = 0

            # Process clicks in this interval
            n_hat[lo] = NAR[int_idx]
            last = last_event[int_idx]

            for click_idx in range(lo, hi + 1):
                if click_idx == 0:
                    n_event[click_idx] = 0
                    KM_hat[click_idx] = 1
                else:
                    if KM_hat[last] == 0:
                        n_event[click_idx] = 0
                        KM_hat[click_idx] = 0
                    else:
                        n_event[click_idx] = round(
                            n_hat[click_idx] * (1 - surv[click_idx] / KM_hat[last])
                        )
                        KM_hat[click_idx] = KM_hat[last] * (
                            1 - n_event[click_idx] / n_hat[click_idx]
                        ) if n_hat[click_idx] > 0 else 0

                if click_idx + 1 < len(n_hat):
                    n_hat[click_idx + 1] = (
                        n_hat[click_idx] - n_event[click_idx] - n_cen[click_idx]
                    )

                if n_event[click_idx] != 0:
                    last = click_idx

            # Check convergence
            if lo_next < len(n_hat):
                diff = n_hat[lo_next] - NAR[int_idx + 1]
                if abs(diff) < 0.5:
                    break
                int_censor[int_idx] = int_censor[int_idx] + diff
                if int_censor[int_idx] < 0:
                    int_censor[int_idx] = 0
            else:
                break

        # Adjust NAR if needed
        if lo_next < len(n_hat) and n_hat[lo_next] < NAR[int_idx + 1]:
            NAR[int_idx + 1] = n_hat[lo_next]

        last_event[int_idx + 1] = last

    # Build IPD
    total_n = int(NAR[0])
    IPD_event = np.zeros(total_n, dtype=int)
    total_events = int(np.sum(n_event))
    if total_events > total_n:
        total_events = total_n
    IPD_event[:total_events] = 1

    # Event times
    IPD_time = []
    for click_idx in range(total_e_times + 1):
        ne = int(n_event[click_idx])
        IPD_time.extend([t_surv[click_idx]] * ne)

    # Censoring times (midpoint between consecutive clicks)
    for click_idx in range(total_e_times):
        nc = int(n_cen[click_idx])
        mid = (t_surv[click_idx] + t_surv[click_idx + 1]) / 2
        IPD_time.extend([mid] * nc)

    # Remaining people censored at max time
    ppl_remain = total_n - len(IPD_time)
    if ppl_remain < 0:
        # Truncate
        IPD_time = IPD_time[:total_n]
    else:
        IPD_time.extend([t_surv.max()] * ppl_remain)

    IPD_time = np.array(IPD_time[:total_n], dtype=float)

    return {
        "IPD_time": IPD_time,
        "IPD_event": IPD_event,
        "n_hat": n_hat,
        "KM_hat": KM_hat,
        "n_cen": n_cen,
        "n_event": n_event,
        "int_censor": int_censor,
    }


def reconstruct_arm(clicks_df, nar_df, arm_name="Arm 1", is_cumulative_incidence=False):
    """
    Full pipeline: digitized clicks + NAR -> IPD for one arm.

    Parameters
    ----------
    clicks_df : pd.DataFrame with columns ['time', 'survival'] or ['time', 'cum_inc']
    nar_df : pd.DataFrame with columns ['time', 'NAR']
    arm_name : str
    is_cumulative_incidence : bool - if True, converts CI to survival

    Returns
    -------
    pd.DataFrame with columns ['arm', 'time', 'status']
    """
    df = clicks_df.copy()

    if is_cumulative_incidence:
        if "cum_inc" in df.columns:
            df["survival"] = 1 - df["cum_inc"] / 100
        elif "survival" in df.columns:
            df["survival"] = 1 - df["survival"] / 100
        df = df[["time", "survival"]]

    # Preprocess
    df = collapse_corners(df)

    # Trim to NAR max time
    max_time = nar_df["time"].max()
    df = df[df["time"] <= max_time].reset_index(drop=True)

    # Format and reconstruct
    aug = format_raw_tabs(nar_df, df)
    recon = KM_reconstruct(aug["aug_NAR"], aug["aug_surv"])

    ipd = pd.DataFrame({
        "arm": arm_name,
        "time": recon["IPD_time"],
        "status": recon["IPD_event"]
    })

    return ipd
