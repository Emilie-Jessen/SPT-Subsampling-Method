#!/usr/bin/env python3
"""Core analysis utilities for the diffusivity-density analysis.

This module contains the numerical routines used by ``LineDistance_Stats.ipynb``
to analyse single-particle tracking (SPT) data from the publication.  It can
also be executed as a small command-line program to analyse all configured
datasets and reproduce the diffusivity-density summary plot.

Input data
----------
Trajectory files are expected to be whitespace-separated text files with at
least four columns in the order::

    trajectory_id    x_um    y_um    time_s

Additional columns are ignored. Coordinates are in micrometres and time is in
seconds.

Analysis overview
-----------------
For each cell (or segmented focus), the pipeline:

1. estimates the fast diffusion coefficient from trajectory lengths;
2. fits a two-component Rayleigh mixture to step lengths;
3. estimates the slow diffusion coefficient using the covariance estimator
   (CVE; alternative Rayleigh/MSD estimates are also retained);
4. estimates and subtracts the moving focus centre with a Kalman smoother;
5. samples local log densities from the corrected localization cloud; and
6. estimates the potential-depth contrast ``dU`` from the two density modes.

The diffusivity coordinate used in the publication is

    x = (D_slow - D_b) / (D_fast - D_b),

and the corresponding density coordinate is

    y = exp(-dU) = rho_out / rho_in.

The identity relation ``y = x`` is the PPPS reference used in the companion
notebook.

Notes
-----
The numerical algorithms are intentionally kept close to the analysis used for
the publication.  The refactoring in this public version focuses on naming,
documentation, input checking, deterministic random-number generation, and
fixing implementation inconsistencies without changing the intended method.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import gaussian_kde

Array = np.ndarray
PathLike = str | Path


# =============================================================================
# Configuration
# =============================================================================

# By default, expect ``AllData/`` next to this file. The notebook uses its own
# explicit data root and can override any module-level setting below.
DATA_ROOT = Path(__file__).resolve().parent

# name -> (subfolder under AllData, filename glob, frame interval [s])
DATASETS = {
    "Rad51": ("Rad51", "*.trxyt", 0.03),
    "SUMO": ("SUMO", "*.trxyt", 0.02),
    "Rad52": ("Rad52", "*.txt", 0.02),
    "Rfa1": ("Rfa1", "*.txt", 0.02),
    "TwoDSB": ("TwoDSB", "*.txt", 0.02),
}

# Diffusion / trajectory settings.
Db = 0.0025  # baseline diffusion coefficient used in the diffusivity ratio [um^2/s]
MIN_LONG = 10  # trajectories longer than this can anchor focus-centre inference
MIN_TOTAL = 100  # minimum localizations required to analyse a cell
SIGMA_LOC = 0.025  # localization precision per coordinate [um]
DC_FOCUS = 0.002  # focus-centre diffusion coefficient [um^2/s]
D_FAST_MAX = 2.0  # upper bound on D_fast in the Rayleigh fit [um^2/s]

# Local-density sampling settings.
N_TEST = 8_000
R_MIN = 0.015  # [um]
R_MAX = 0.15  # [um]
N_TEST_GROUP = 30_000

# Slow-diffusion estimator settings.
CVE_NBOOT = 60
CVE_FIX_SIGMA = True
SLOW_SOURCE = "cve"  # one of: "rayleigh", "cve", "msd"
MSD_NMAX = 3

# Optional focus filtering / segmentation settings.
R_FOCUS = 0.20
FOCUS_MIN = 0.50
USE_FOCUS_FILTER = False
GROUP_MODE = "mean"  # "mean" or "group"
SEGMENT_SOURCES = {"Rad51"}
R_FOC_DET = 0.15
MIN_SEP = 0.30
R_CROP = 0.45
MIN_PEAK = 25
SEG_MIN_TOTAL = 60

# Simulation-only parameters.
A = 4.0
b = 1_000.0
r_f = 0.15
r_n = 1.0
D_0 = 0.05
D_1 = 0.5
Z_L = 0.15
sigma_loc = 0.03


# =============================================================================
# Model helper functions
# =============================================================================


def sig(u: Array | float) -> Array | float:
    """Logistic sigmoid used to define the smooth focus interface."""
    return 1.0 / (1.0 + np.exp(-u))


def U_r(r: Array | float) -> Array | float:
    """Dimensionless radial potential ``U(r) / kBT`` used in simulations."""
    return A * sig(b * (r - r_f))


def dU_r(r: Array | float) -> Array | float:
    """Radial derivative of :func:`U_r`."""
    s = sig(b * (r - r_f))
    return A * b * s * (1.0 - s)


def D_r(r: Array | float) -> Array | float:
    """Position-dependent diffusion coefficient used in simulations."""
    return D_0 + (D_1 - D_0) * sig(b * (r - r_f))


def dD_r(r: Array | float) -> Array | float:
    """Radial derivative of :func:`D_r`."""
    s = sig(b * (r - r_f))
    return (D_1 - D_0) * b * s * (1.0 - s)


# =============================================================================
# I/O and trajectory utilities
# =============================================================================


def load_trxyt(path: PathLike) -> Array | None:
    """Load an SPT trajectory file.

    Parameters
    ----------
    path
        Whitespace-separated file containing at least four columns:
        ``trajectory_id, x, y, time``.

    Returns
    -------
    numpy.ndarray or None
        Array with shape ``(n_localizations, 4)``. ``None`` is returned if the
        file cannot be read or does not contain the required columns.
    """
    try:
        data = np.genfromtxt(path)
    except (OSError, ValueError):
        return None

    if data.ndim != 2 or data.shape[1] < 4:
        return None
    return np.asarray(data[:, :4], dtype=float)


def trxyt_to_trajs(trxyt: Array) -> list[Array]:
    """Convert a ``trxyt`` array into time-ordered 2D trajectories.

    Returns one ``(n_i, 2)`` coordinate array per trajectory ID.
    """
    trajectories: list[Array] = []
    for trajectory_id in np.unique(trxyt[:, 0]):
        sub = trxyt[trxyt[:, 0] == trajectory_id]
        sub = sub[np.argsort(sub[:, 3])]
        trajectories.append(sub[:, 1:3])
    return trajectories


def estimate_Dfast_from_lengths(
    trajs: Sequence[Array], ts: float, focal_depth: float
) -> float:
    """Estimate ``D_fast`` from the short-trajectory length distribution.

    This reproduces the estimator used in the publication analysis.  The first
    two histogram bins of trajectory lengths are fit in log-count space and
    converted to a diffusion coefficient using the focal depth.
    """
    if not trajs:
        return np.nan

    lengths = np.asarray([len(traj) for traj in trajs], dtype=float)
    counts, edges = np.histogram(lengths, 50)
    if len(edges) < 3:
        return np.nan

    half_bin = 0.5 * (edges[1] - edges[0])
    edge_x = half_bin + edges[:-1]
    log_counts = np.log(np.maximum(counts[:2], 1e-9))
    slope, _ = np.polyfit(edge_x[:2], log_counts, 1)
    return max(-slope * 4.0 * focal_depth**2 / (np.pi * ts), 1e-3)


# =============================================================================
# Diffusion estimation
# =============================================================================


def Raleigh_fit_MCMC(
    data: Array,
    ts: float,
    D2: float,
    n_mh: int = 5_000,
    q_s: float = 0.01,
    tol: float = 1e-3,
    D_fast_max: float = D_FAST_MAX,
    seed: int | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Fit a two-Rayleigh step-size mixture with Metropolis sampling.

    Parameters
    ----------
    data
        ``trxyt`` array. Only steps separated by one frame within ``tol`` are
        used; steps spanning missed frames are discarded.
    ts
        Frame interval in seconds.
    D2
        Initial estimate of the fast diffusion coefficient.
    n_mh
        Number of Metropolis-Hastings iterations.
    q_s
        Proposal standard deviation for all three fitted parameters.
    tol
        Absolute tolerance used to identify one-frame steps.
    D_fast_max
        Upper bound imposed on the fast diffusion coefficient.
    seed
        Random seed. If omitted, a deterministic seed is derived from the
        cell coordinates so results do not depend on processing order.

    Returns
    -------
    tuple
        ``(D_slow, sd_D_slow, D_fast, sd_D_fast, slow_fraction,
        sd_slow_fraction)`` estimated from the second half of the chain.

    Notes
    -----
    The historical function name is retained because it is used by the
    publication notebook. The fitted components themselves are Rayleigh
    distributions.
    """
    if seed is None:
        seed = int(np.mod(np.sum(np.abs(data[:, 1:3])) * 1e6, 2**32))
    rng = np.random.default_rng(seed)

    step_lengths: list[Array] = []
    for trajectory_id in np.unique(data[:, 0]):
        trajectory = data[data[:, 0] == trajectory_id]
        trajectory = trajectory[np.argsort(trajectory[:, 3])]
        if len(trajectory) < 2:
            continue

        dx = np.diff(trajectory[:, 1])
        dy = np.diff(trajectory[:, 2])
        in_time = np.abs(np.diff(trajectory[:, 3]) - ts) < tol
        values = np.hypot(dx, dy)[in_time]
        if values.size:
            step_lengths.append(values)

    if not step_lengths:
        return (np.nan,) * 6

    steps = np.concatenate(step_lengths)
    steps = steps[steps > 0]
    if len(steps) < 2:
        return (np.nan,) * 6

    D1 = 0.02
    D2 = min(float(D2), D_fast_max)
    fraction = 0.5

    def log_probability(x: Array, d1: float, d2: float, frac: float) -> float:
        slow = frac * x / (2 * d1 * ts) * np.exp(-x**2 / (4 * d1 * ts))
        fast = (1 - frac) * x / (2 * d2 * ts) * np.exp(-x**2 / (4 * d2 * ts))
        return float(np.sum(np.log(slow + fast + 1e-300)))

    chain = np.zeros((n_mh, 3))
    for i in range(n_mh):
        D1_new = D1 + q_s * rng.standard_normal()
        D2_new = D2 + q_s * rng.standard_normal()
        fraction_new = abs(fraction + q_s * rng.standard_normal())

        valid = (
            D1_new > 0
            and D2_new > 0
            and D2_new <= D_fast_max
            and 0 <= fraction_new <= 1
        )
        if valid:
            log_alpha = log_probability(steps, D1_new, D2_new, fraction_new) - log_probability(
                steps, D1, D2, fraction
            )
            if np.log(rng.random()) < log_alpha:
                D1, D2, fraction = D1_new, D2_new, fraction_new

        chain[i] = (D1, D2, fraction)

    burn_in = n_mh // 2
    posterior = chain[burn_in:]
    return (
        float(posterior[:, 0].mean()),
        float(posterior[:, 0].std()),
        float(posterior[:, 1].mean()),
        float(posterior[:, 1].std()),
        float(posterior[:, 2].mean()),
        float(posterior[:, 2].std()),
    )


def fit_two_rayleigh(s: Array) -> tuple[float, float, float]:
    """Fit a two-component Rayleigh mixture by EM.

    Returns ``(b_slow, b_fast, slow_fraction)``, where ``b`` denotes the
    Rayleigh scale parameter.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return np.nan, np.nan, np.nan

    base = np.sqrt(np.mean(s**2) / 2.0)
    b1, b2, w1 = 0.5 * base, 1.5 * base, 0.5

    for _ in range(200):
        f1 = (s / b1**2) * np.exp(-s**2 / (2 * b1**2))
        f2 = (s / b2**2) * np.exp(-s**2 / (2 * b2**2))
        responsibility = (w1 * f1) / (w1 * f1 + (1 - w1) * f2 + 1e-300)
        w1 = float(np.mean(responsibility))

        slow_weight = np.sum(responsibility)
        fast_weight = np.sum(1 - responsibility)
        if slow_weight <= 0 or fast_weight <= 0:
            break

        b1 = np.sqrt(np.sum(responsibility * s**2) / (2 * slow_weight))
        b2 = np.sqrt(np.sum((1 - responsibility) * s**2) / (2 * fast_weight))

    if b1 > b2:
        b1, b2 = b2, b1
        w1 = 1 - w1
    return float(b1), float(b2), float(w1)


def cve_slow_diffusion(
    data: Array,
    dt_base: float,
    n_boot: int = 300,
    tol: float = 1e-3,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate the slow diffusion coefficient using the CVE correction.

    The slow Rayleigh scale is obtained from a two-component mixture. The
    localization variance is either fixed to ``SIGMA_LOC**2`` or estimated from
    pooled lag-1 step covariances. Track-level bootstrap resampling provides the
    reported standard error.

    Returns
    -------
    tuple
        ``(D_slow, sigma_localization, bootstrap_sd)``.
    """
    per_track: list[tuple[Array, Array, Array]] = []
    for trajectory_id in np.unique(data[:, 0]):
        trajectory = data[data[:, 0] == trajectory_id]
        trajectory = trajectory[np.argsort(trajectory[:, 3])]
        dx = np.diff(trajectory[:, 1])
        dy = np.diff(trajectory[:, 2])
        in_time = np.abs(np.diff(trajectory[:, 3]) - dt_base) < tol
        if in_time.sum() > 0:
            per_track.append((dx, dy, in_time))

    if not per_track or sum(mask.sum() for _, _, mask in per_track) < 40:
        return np.nan, np.nan, np.nan

    def estimate(tracks: Sequence[tuple[Array, Array, Array]]) -> tuple[float, float]:
        steps = np.concatenate([np.hypot(dx, dy)[mask] for dx, dy, mask in tracks])
        if len(steps) < 40:
            return np.nan, np.nan

        b1, _, _ = fit_two_rayleigh(steps)
        if CVE_FIX_SIGMA:
            sigma2 = SIGMA_LOC**2
        else:
            numerator = 0.0
            denominator = 0.0
            for dx, dy, mask in tracks:
                adjacent = mask[:-1] & mask[1:]
                numerator += np.sum(
                    (dx[:-1] * dx[1:] + dy[:-1] * dy[1:])[adjacent]
                )
                denominator += np.sum(adjacent)
            sigma2 = max(-0.5 * numerator / denominator, 0.0) if denominator > 0 else 0.0

        D_slow = max((b1**2 - 2.0 * sigma2) / (2.0 * dt_base), 0.0)
        return float(D_slow), float(np.sqrt(sigma2))

    D_slow, sigma = estimate(per_track)

    rng = np.random.default_rng(seed)
    n_tracks = len(per_track)
    bootstrap: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n_tracks, n_tracks)
        D_boot, _ = estimate([per_track[i] for i in indices])
        bootstrap.append(D_boot)

    bootstrap_array = np.asarray(bootstrap, dtype=float)
    bootstrap_array = bootstrap_array[np.isfinite(bootstrap_array)]
    bootstrap_sd = np.std(bootstrap_array) if len(bootstrap_array) else np.nan
    return D_slow, sigma, float(bootstrap_sd)


def msd_slow(
    data: Array,
    dt_base: float,
    nmax: int = MSD_NMAX,
    min_len: int = MIN_LONG,
    tol: float = 1e-3,
) -> tuple[float, float]:
    """Estimate slow diffusion from the first lags of the time-averaged MSD.

    Only trajectories longer than ``min_len`` are included. For
    ``MSD(t) = 4 D t + 4 sigma^2``, the returned values are ``(D, sigma^2)``.
    """
    sums = np.zeros(nmax)
    counts = np.zeros(nmax)

    for trajectory_id in np.unique(data[:, 0]):
        trajectory = data[data[:, 0] == trajectory_id]
        trajectory = trajectory[np.argsort(trajectory[:, 3])]
        if len(trajectory) <= min_len:
            continue

        x, y, t = trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
        for lag in range(1, nmax + 1):
            in_time = np.abs((t[lag:] - t[:-lag]) - lag * dt_base) < tol
            displacement2 = (x[lag:] - x[:-lag]) ** 2 + (y[lag:] - y[:-lag]) ** 2
            sums[lag - 1] += np.sum(displacement2[in_time])
            counts[lag - 1] += np.sum(in_time)

    valid = counts > 0
    if valid.sum() < 2:
        return np.nan, np.nan

    msd = sums[valid] / counts[valid]
    lags = (np.arange(1, nmax + 1) * dt_base)[valid]
    design = np.vstack([lags, np.ones(valid.sum())]).T
    slope, intercept = np.linalg.lstsq(design, msd, rcond=None)[0]
    return float(slope / 4.0), float(intercept / 4.0)


# =============================================================================
# HMM / Kalman focus-centre correction
# =============================================================================


def kalman_center(
    X: Array,
    dt: Array,
    kappa: float,
    D: float,
    DC: float,
    P0: float = 0.1,
) -> tuple[Array, float]:
    """Infer a moving focus centre with a Kalman filter and smoother.

    Parameters follow the OU/Kalman centre model used in the publication.
    ``X`` must have shape ``(2, n_positions)`` and ``dt`` length
    ``n_positions - 1``.
    """
    n_positions = X.shape[1]
    a = np.exp(-kappa * dt)
    H = 1.0 - a
    observation_var = D**2 / kappa * (1.0 - a**2)
    process_var = 2.0 * DC * dt
    Y = X[:, 1:] - a * X[:, :-1]

    filtered_mean = np.zeros((n_positions, 2))
    filtered_var = np.zeros(n_positions)
    mean = np.zeros(2)
    variance = P0
    log_likelihood = 0.0

    for n in range(n_positions):
        if n > 0:
            variance += process_var[n - 1]
        if n <= n_positions - 2:
            innovation_var = H[n] ** 2 * variance + observation_var[n]
            gain = H[n] * variance / innovation_var
            innovation = Y[:, n] - H[n] * mean
            mean = mean + gain * innovation
            variance = (1.0 - gain * H[n]) * variance
            log_likelihood += (
                -0.5 * np.sum(innovation**2) / innovation_var
                - np.log(2 * np.pi * innovation_var)
            )
        filtered_mean[n] = mean
        filtered_var[n] = variance

    smoothed_mean = filtered_mean.copy()
    for n in range(n_positions - 2, -1, -1):
        predicted_var = filtered_var[n] + process_var[n]
        smoother_gain = filtered_var[n] / predicted_var
        smoothed_mean[n] = filtered_mean[n] + smoother_gain * (
            smoothed_mean[n + 1] - filtered_mean[n]
        )

    return smoothed_mean.T, float(log_likelihood)


def infer_center(
    X: Array,
    dt: Array,
    DC: float,
    D2_prior: float | None = None,
    D2_prior_sd: float | None = None,
) -> Array:
    """Estimate Kalman/OU parameters and return the smoothed focus centre."""
    D0 = np.sqrt(D2_prior) if D2_prior is not None and D2_prior > 0 else 0.05

    def negative_log_likelihood(theta: Array) -> float:
        kappa, D = theta
        if kappa <= 0 or D <= 0:
            return 1e9

        _, log_likelihood = kalman_center(X, dt, kappa, D, DC)
        if D2_prior is not None:
            prior_sd = 1e-4 if D2_prior_sd is None else max(D2_prior_sd, 1e-4)
            log_likelihood += -0.5 * ((D**2 - D2_prior) / prior_sd) ** 2
        return -float(log_likelihood)

    result = minimize(
        negative_log_likelihood,
        [50.0, D0],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200},
    )
    center, _ = kalman_center(X, dt, result.x[0], result.x[1], DC)
    return center


def correct_cell(
    data: Array,
    D2_prior: float | None = None,
    D2_prior_sd: float | None = None,
    D_c: float = DC_FOCUS,
) -> tuple[Array | None, list[tuple[Array, Array]] | None, float]:
    """Subtract the inferred moving focus centre from a cell's localizations.

    Long trajectories (``len > MIN_LONG``) independently anchor centre
    inference. Short trajectories are translated using the temporally closest
    inferred centre position.

    Returns
    -------
    corrected
        ``trxyt`` array with corrected x/y coordinates, or ``None`` when no
        trajectory is long enough to anchor centre inference.
    centers
        List of ``(centre_positions, time_points)`` for the anchor trajectories.
    fraction_corrected
        Fraction of input localizations included in the corrected output.
    """
    trajectory_ids = np.unique(data[:, 0])
    centers: list[tuple[Array, Array]] = []
    Xc: list[float] = []
    Yc: list[float] = []
    Ic: list[float] = []
    Tc: list[float] = []
    n_points_corrected = 0

    for trajectory_id in trajectory_ids:
        trajectory = data[data[:, 0] == trajectory_id]
        trajectory = trajectory[np.argsort(trajectory[:, 3])]
        if len(trajectory) <= MIN_LONG:
            continue

        xy = trajectory[:, 1:3].astype(float).copy()
        x0 = xy[0].copy()
        xy -= x0
        dt = np.diff(trajectory[:, 3])
        center = infer_center(
            xy.T,
            dt,
            D_c,
            D2_prior=D2_prior,
            D2_prior_sd=D2_prior_sd,
        )
        corrected_xy = xy - center.T

        Xc.extend(corrected_xy[:, 0])
        Yc.extend(corrected_xy[:, 1])
        Ic.extend(trajectory[:, 0])
        Tc.extend(trajectory[:, 3])
        centers.append((center + x0[:, None], trajectory[:, 3]))
        n_points_corrected += len(trajectory)

    if not centers:
        return None, None, 0.0

    all_centers = np.hstack([center for center, _ in centers])
    all_times = np.hstack([time for _, time in centers])

    for trajectory_id in trajectory_ids:
        trajectory = data[data[:, 0] == trajectory_id]
        if len(trajectory) > MIN_LONG or len(trajectory) == 0:
            continue

        nearest = np.argmin(np.abs(all_times - trajectory[0, 3]))
        if abs(all_times[nearest] - trajectory[0, 3]) < 1:
            Xc.extend(trajectory[:, 1] - all_centers[0, nearest])
            Yc.extend(trajectory[:, 2] - all_centers[1, nearest])
            Ic.extend(trajectory[:, 0])
            Tc.extend(trajectory[:, 3])
            n_points_corrected += len(trajectory)

    corrected = np.column_stack([Ic, Xc, Yc, Tc])
    fraction_corrected = n_points_corrected / len(data)
    return corrected, centers, float(fraction_corrected)


# =============================================================================
# Local-density sampling and potential-depth estimation
# =============================================================================


def estimate_rn(xy: Array) -> float:
    """Estimate an effective 2D nuclear radius from localization coordinates."""
    try:
        area = ConvexHull(xy).volume  # ``volume`` is area for a 2D hull.
        return float(np.sqrt(area / np.pi))
    except (ValueError, RuntimeError):
        center = xy.mean(axis=0)
        radius = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
        return float(np.percentile(radius, 99))


def box_sampler_Our(
    xy: Array,
    N: int,
    r_min: float,
    r_max: float,
    seed: int,
    r_n: float,
) -> Array:
    """Sample local log densities using random disks.

    For each trial, a point is sampled within the analysis disk and the local
    density is evaluated in a random-radius disk. If the disk contains no
    localization, its radius is increased until at least one point is found.

    The function name is retained from the original analysis notebook for
    backwards compatibility.
    """
    rng = np.random.default_rng(seed)
    log_rhos: list[float] = []
    tree = cKDTree(xy)

    for _ in range(N):
        radial_position = r_n * rng.uniform()
        angle = 2 * np.pi * rng.uniform()
        x = radial_position * np.cos(angle)
        y = radial_position * np.sin(angle)

        scale = 1
        radius = rng.uniform(r_min, r_max)
        while True:
            neighbors = tree.query_ball_point([x, y], radius)
            if neighbors:
                density = len(neighbors) / (np.pi * radius**2)
                log_rhos.append(np.log(density))
                break
            scale += 1
            radius = rng.uniform(r_min * scale, r_max * scale)

    return np.asarray(log_rhos, dtype=float)


def find_energy(log_rhos: Array) -> tuple[float, float]:
    """Estimate the density-mode separation ``dU`` and its approximate SE.

    A Gaussian KDE is fit to the sampled log densities. The two most prominent
    modes are identified and their separation is returned as ``dU`` in units of
    ``kBT``. Curvature around the modes provides an approximate uncertainty.
    """
    x_data = np.asarray(log_rhos, dtype=float).ravel()
    x_data = x_data[np.isfinite(x_data)]
    if x_data.size < 10:
        return np.nan, np.nan

    kde = gaussian_kde(x_data, bw_method="scott")
    x = np.linspace(x_data.min(), x_data.max(), 500)
    y = kde(x)
    peaks, properties = find_peaks(y, prominence=0.01 * y.max())
    if len(peaks) < 2:
        return np.nan, np.nan

    top_two = peaks[np.argsort(properties["prominences"])[::-1][:2]]
    p_low, p_high = np.sort(top_two)
    dU = x[p_high] - x[p_low]

    log_y = np.log(y + 1e-300)
    dx = x[1] - x[0]
    second_derivative = np.gradient(np.gradient(log_y, dx), dx)
    curvature_low = -second_derivative[p_low]
    curvature_high = -second_derivative[p_high]
    if curvature_low <= 0 or curvature_high <= 0:
        return float(dU), np.nan

    sigma_low = 1 / np.sqrt(curvature_low)
    sigma_high = 1 / np.sqrt(curvature_high)
    split = x[p_low + np.argmin(y[p_low:p_high])]
    n_low = max((x_data < split).sum(), 1)
    n_high = max((x_data >= split).sum(), 1)
    se = np.hypot(sigma_low / np.sqrt(n_low), sigma_high / np.sqrt(n_high))
    return float(dU), float(se)


# =============================================================================
# Cell- and source-level analysis
# =============================================================================


def _selected_slow_diffusion(
    D_slow_rayleigh: float,
    D_slow_msd: float,
    D_slow_cve: float,
) -> float:
    """Return the slow-diffusion estimate selected by ``SLOW_SOURCE``."""
    try:
        return {
            "rayleigh": D_slow_rayleigh,
            "msd": D_slow_msd,
            "cve": D_slow_cve,
        }[SLOW_SOURCE]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SLOW_SOURCE={SLOW_SOURCE!r}; use 'rayleigh', 'cve', or 'msd'."
        ) from exc


def analyze_cell(
    trxyt: Array | None,
    ts: float,
    D_c: float = DC_FOCUS,
    seed: int = 1,
) -> dict[str, float] | None:
    """Run the full publication pipeline for one cell.

    Returns a dictionary containing ``dU``, the diffusivity coordinate ``x``,
    the alternative slow-diffusion estimates, and the number of localizations.
    Cells with fewer than ``MIN_TOTAL`` localizations return ``None``.
    """
    if trxyt is None or len(trxyt) < MIN_TOTAL:
        return None

    trajectories = trxyt_to_trajs(trxyt)
    D_fast_est = estimate_Dfast_from_lengths(trajectories, ts, Z_L)
    (
        D_slow_rayleigh,
        sd_D_slow,
        D_fast,
        _,
        _,
        _,
    ) = Raleigh_fit_MCMC(trxyt, ts, D_fast_est)

    D_slow_cve, _, _ = cve_slow_diffusion(
        trxyt, ts, n_boot=CVE_NBOOT, tol=1e-3, seed=1
    )
    D_slow_msd, _ = msd_slow(trxyt, ts, nmax=MSD_NMAX, min_len=MIN_LONG)

    corrected, _, _ = correct_cell(
        trxyt,
        D2_prior=D_slow_rayleigh,
        D2_prior_sd=sd_D_slow,
        D_c=D_c,
    )
    if corrected is None:
        dU = np.nan
    else:
        xy_corrected = corrected[:, 1:3]
        r_n_cell = np.percentile(
            np.hypot(xy_corrected[:, 0], xy_corrected[:, 1]), 98
        )
        log_rhos = box_sampler_Our(
            xy_corrected, N_TEST, R_MIN, R_MAX, seed, r_n_cell
        )
        dU, _ = find_energy(log_rhos)

    D_slow = _selected_slow_diffusion(
        D_slow_rayleigh, D_slow_msd, D_slow_cve
    )
    denominator = D_fast - Db
    x_value = (
        (D_slow - Db) / denominator
        if np.isfinite(D_slow) and denominator > 0 and D_slow > Db
        else np.nan
    )

    return {
        "dU": float(dU),
        "x": float(x_value),
        "D_slow": float(D_slow_rayleigh),
        "D_fast": float(D_fast),
        "D_slow_cve": float(D_slow_cve),
        "D_slow_msd": float(D_slow_msd),
        "n": float(len(trxyt)),
    }


def has_focus(d: Array, R: float = R_FOCUS, frac_min: float = FOCUS_MIN) -> bool:
    """Return whether a localization cloud contains a sufficiently dense core."""
    xy = d[:, 1:3]
    tree = cKDTree(xy)
    counts = np.asarray([len(indices) for indices in tree.query_ball_point(xy, R)])
    return bool(counts.max() / len(xy) >= frac_min)


def segment_foci(
    d: Array,
    R_focus: float = R_FOC_DET,
    min_sep: float = MIN_SEP,
    R_crop: float = R_CROP,
    min_peak: int = MIN_PEAK,
    min_total: int = SEG_MIN_TOTAL,
    max_foci: int = 40,
) -> list[Array]:
    """Split a multi-focus field into focus-centred sub-cells.

    Compact localization-density peaks are detected at scale ``R_focus``.
    Candidate centres must be separated by at least ``min_sep``. Whole
    trajectories are then assigned to their nearest centre when their mean
    position lies within ``R_crop``. Sub-cells with fewer than ``min_total``
    localizations are discarded.
    """
    xy = d[:, 1:3]
    ids = d[:, 0]
    unique_ids = np.unique(ids)
    trajectory_centres = np.asarray([xy[ids == tid].mean(axis=0) for tid in unique_ids])

    local_counts = np.asarray(
        [len(indices) for indices in cKDTree(xy).query_ball_point(xy, R_focus)]
    )

    centres: list[Array] = []
    for index in np.argsort(-local_counts):
        if local_counts[index] < min_peak:
            break
        candidate = xy[index]
        if centres:
            distances = np.linalg.norm(np.asarray(centres) - candidate, axis=1)
            if np.min(distances) < min_sep:
                continue
        centres.append(candidate)
        if len(centres) >= max_foci:
            break

    if not centres:
        return []

    distance, nearest_centre = cKDTree(np.asarray(centres)).query(trajectory_centres)
    subcells: list[Array] = []
    for focus_index in range(len(centres)):
        selected_ids = unique_ids[
            (nearest_centre == focus_index) & (distance < R_crop)
        ]
        if len(selected_ids) == 0:
            continue
        subcell = d[np.isin(ids, selected_ids)]
        if len(subcell) >= min_total:
            subcells.append(subcell)
    return subcells


def analyze_source(
    name: str,
    data_list: Sequence[Array | None],
    ts: float,
    seed0: int = 0,
) -> dict[str, object]:
    """Analyse all cells belonging to one experimental source/condition."""
    cells: list[dict[str, float]] = []
    pooled_xy: list[Array] = []
    pooled_raw: list[Array] = []
    id_offset = 0.0
    n_small = n_no_focus = n_no_correction = 0

    for index, data in enumerate(data_list):
        if data is None or len(data) < MIN_TOTAL:
            n_small += 1
            continue
        if USE_FOCUS_FILTER and not has_focus(data):
            n_no_focus += 1
            continue

        trajectories = trxyt_to_trajs(data)
        D_fast_est = estimate_Dfast_from_lengths(trajectories, ts, Z_L)
        D_slow_rayleigh, sd_D_slow, D_fast, *_ = Raleigh_fit_MCMC(
            data, ts, D_fast_est
        )
        D_slow_msd, _ = msd_slow(data, ts)
        D_slow_cve, _, _ = cve_slow_diffusion(
            data, ts, n_boot=CVE_NBOOT, tol=1e-3, seed=1
        )

        corrected, _, _ = correct_cell(
            data,
            D2_prior=D_slow_rayleigh,
            D2_prior_sd=sd_D_slow,
            D_c=DC_FOCUS,
        )
        if corrected is None:
            n_no_correction += 1
            continue

        xy_corrected = corrected[:, 1:3]
        r_n_cell = np.percentile(
            np.hypot(xy_corrected[:, 0], xy_corrected[:, 1]), 98
        )
        dU_cell, _ = find_energy(
            box_sampler_Our(
                xy_corrected,
                N_TEST,
                R_MIN,
                R_MAX,
                seed0 + index + 1,
                r_n_cell,
            )
        )

        D_slow = _selected_slow_diffusion(
            D_slow_rayleigh, D_slow_msd, D_slow_cve
        )
        denominator = D_fast - Db
        x_cell = (
            (D_slow - Db) / denominator
            if np.isfinite(D_slow) and D_slow > Db and denominator > 0
            else np.nan
        )
        cells.append(
            {
                "x": float(x_cell),
                "dU": float(dU_cell),
                "D_slow": float(D_slow),
                "D_fast": float(D_fast),
                "n": float(len(data)),
            }
        )

        if GROUP_MODE == "group":
            pooled_xy.append(xy_corrected)
            shifted = data.copy()
            shifted[:, 0] = shifted[:, 0] + id_offset
            id_offset += float(data[:, 0].max() + 1)
            pooled_raw.append(shifted)

    print(
        f"  kept {len(cells)}/{len(data_list)} cells "
        f"(dropped {n_small} small, {n_no_focus} no-focus, "
        f"{n_no_correction} no-anchor)"
    )

    group: dict[str, float] | None = None
    if GROUP_MODE == "group" and pooled_xy:
        XY = np.vstack(pooled_xy)
        RAW = np.vstack(pooled_raw)
        r_n_pooled = np.percentile(np.hypot(XY[:, 0], XY[:, 1]), 98)
        dU_group, _ = find_energy(
            box_sampler_Our(
                XY, N_TEST_GROUP, R_MIN, R_MAX, seed0, r_n_pooled
            )
        )

        trajectories = trxyt_to_trajs(RAW)
        D_fast_est = estimate_Dfast_from_lengths(trajectories, ts, Z_L)
        D_slow_rayleigh, _, D_fast_group, *_ = Raleigh_fit_MCMC(
            RAW, ts, D_fast_est
        )
        D_slow_msd, _ = msd_slow(RAW, ts)
        D_slow_cve, _, _ = cve_slow_diffusion(
            RAW, ts, n_boot=CVE_NBOOT, tol=1e-3, seed=1
        )
        D_slow_group = _selected_slow_diffusion(
            D_slow_rayleigh, D_slow_msd, D_slow_cve
        )
        denominator = D_fast_group - Db
        x_group = (
            (D_slow_group - Db) / denominator
            if np.isfinite(D_slow_group)
            and D_slow_group > Db
            and denominator > 0
            else np.nan
        )
        group = {
            "x": float(x_group),
            "dU": float(dU_group),
            "D_slow": float(D_slow_group),
            "D_fast": float(D_fast_group),
            "n_cells": float(len(pooled_xy)),
            "n_pts": float(len(XY)),
        }
        print(
            f"  GROUPED: dU={dU_group:.5g}  x={x_group:.5g}  "
            f"D_slow={D_slow_group:.3f}  D_fast={D_fast_group:.3f}  "
            f"({len(pooled_xy)} cells, {len(XY)} pts)"
        )

    return {"cells": cells, "group": group}


# =============================================================================
# Data and simulation drivers
# =============================================================================


def run_real(data_root: PathLike | None = None) -> dict[str, dict[str, object]]:
    """Analyse all real-data sources in :data:`DATASETS`.

    Parameters
    ----------
    data_root
        Directory containing ``AllData/``. Defaults to :data:`DATA_ROOT`.
    """
    root = Path(DATA_ROOT if data_root is None else data_root)
    all_data = root / "AllData"
    results: dict[str, dict[str, object]] = {}

    for name, (subfolder, pattern, ts) in DATASETS.items():
        files = sorted((all_data / subfolder).glob(pattern))
        files = [
            path
            for path in files
            if "2Foci" not in path.name and "2Cells" not in path.name
        ]

        data_list: list[Array] = []
        for path in files:
            data = load_trxyt(path)
            if data is None:
                continue
            if name in SEGMENT_SOURCES:
                data_list.extend(segment_foci(data))
            else:
                data_list.append(data)

        print(
            f"\n=== {name}: {len(files)} files -> {len(data_list)} cells "
            f"(dt={ts}) ==="
        )
        results[name] = analyze_source(name, data_list, ts)

    return results


def simulate(
    N_particles: int,
    tmax: float,
    dt: float,
    ts: float,
    seed: int,
    sigma_loc: float,
    Z_L: float,
    D_c: float = 0.005,
    focus_center: Array | None = None,
) -> Array:
    """Simulate projected SPT trajectories for the smooth-focus model."""
    rng = np.random.default_rng(seed)
    n_steps = int(tmax / dt)
    n_save = int(tmax / ts) + 1

    r_grid = np.linspace(0.0, r_n, 5_000)
    radial_pdf = r_grid**2 * np.exp(-U_r(r_grid))
    radial_cdf = np.cumsum(radial_pdf)
    radial_cdf /= radial_cdf[-1]

    if focus_center is None:
        magnitude = (r_n - r_f) * rng.random() ** (1 / 3)
        cos_theta = 1 - 2 * rng.random()
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2 * np.pi * rng.random()
        focus_center = np.array(
            [
                magnitude * sin_theta * np.cos(phi),
                magnitude * sin_theta * np.sin(phi),
                0.0 * magnitude * cos_theta,
            ]
        )

    cx, cy, cz = focus_center
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    while len(xs) < N_particles:
        n_missing = N_particles - len(xs)
        radius = np.interp(rng.random(n_missing), radial_cdf, r_grid)
        cos_theta = 1 - 2 * rng.random(n_missing)
        sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2))
        phi = 2 * np.pi * rng.random(n_missing)
        x_new = cx + radius * sin_theta * np.cos(phi)
        y_new = cy + radius * sin_theta * np.sin(phi)
        z_new = cz + radius * cos_theta
        inside = x_new**2 + y_new**2 + z_new**2 < r_n**2
        xs.extend(x_new[inside])
        ys.extend(y_new[inside])
        zs.extend(z_new[inside])

    x = np.asarray(xs[:N_particles])
    y = np.asarray(ys[:N_particles])
    z = np.asarray(zs[:N_particles])

    saved = np.zeros((n_save, N_particles, 3))
    save_index = 0
    next_save = 0.0
    max_center_radius = r_n / 2

    for step in range(n_steps):
        time = (step + 1) * dt

        cx_new = cx + np.sqrt(2 * D_c * dt) * rng.standard_normal()
        cy_new = cy + np.sqrt(2 * D_c * dt) * rng.standard_normal()
        if cx_new**2 + cy_new**2 < max_center_radius**2:
            cx, cy = cx_new, cy_new

        dx_center, dy_center, dz_center = x - cx, y - cy, z - cz
        radial_distance = np.maximum(
            np.sqrt(dx_center**2 + dy_center**2 + dz_center**2), 1e-9
        )
        diffusion = D_r(radial_distance)
        potential_gradient = dU_r(radial_distance)
        diffusion_gradient = dD_r(radial_distance)
        ux = dx_center / radial_distance
        uy = dy_center / radial_distance
        uz = dz_center / radial_distance
        drift = -diffusion * potential_gradient + diffusion_gradient
        noise_scale = np.sqrt(2 * diffusion * dt)

        x_trial = x + dt * ux * drift + noise_scale * rng.standard_normal(N_particles)
        y_trial = y + dt * uy * drift + noise_scale * rng.standard_normal(N_particles)
        z_trial = z + dt * uz * drift + noise_scale * rng.standard_normal(N_particles)
        inside = x_trial**2 + y_trial**2 + z_trial**2 < r_n**2
        x = np.where(inside, x_trial, x)
        y = np.where(inside, y_trial, y)
        z = np.where(inside, z_trial, z)

        if time >= next_save and save_index < n_save:
            saved[save_index, :, 0] = x
            saved[save_index, :, 1] = y
            saved[save_index, :, 2] = z
            save_index += 1
            next_save = save_index * ts

    saved = saved[:save_index]
    observation_rng = np.random.default_rng(seed + 99)
    n_frames = saved.shape[0]
    current_id = np.full(N_particles, -1, dtype=int)
    next_id = 1
    rows: list[tuple[int, float, float, float]] = []

    for frame in range(n_frames):
        in_slice = np.abs(saved[frame, :, 2]) < Z_L
        newly_visible = in_slice & (current_id < 0)
        n_new = int(newly_visible.sum())
        if n_new > 0:
            current_id[newly_visible] = np.arange(next_id, next_id + n_new)
            next_id += n_new
        current_id[~in_slice] = -1

        indices = np.where(in_slice)[0]
        if indices.size:
            x_obs = saved[frame, indices, 0].copy()
            y_obs = saved[frame, indices, 1].copy()
            if sigma_loc > 0:
                x_obs += sigma_loc * observation_rng.standard_normal(indices.size)
                y_obs += sigma_loc * observation_rng.standard_normal(indices.size)
            for j, particle_index in enumerate(indices):
                rows.append(
                    (
                        int(current_id[particle_index]),
                        float(x_obs[j]),
                        float(y_obs[j]),
                        frame * ts,
                    )
                )

    trxyt = np.asarray(rows, dtype=float)
    if len(trxyt):
        trxyt = trxyt[np.lexsort((trxyt[:, 3], trxyt[:, 0]))]
    return trxyt


SIM_SOURCES = {
    "sim_A2": {"A": 2.0},
    "sim_A4": {"A": 4.0},
    "sim_A6": {"A": 6.0},
    "sim_A8": {"A": 8.0},
}
SIM_NCELLS = 20
SIM_NP = 30
SIM_TMAX = 2.0
SIM_TS = 0.02
SIM_DC = 0.002


def run_sim() -> dict[str, dict[str, object]]:
    """Run the configured synthetic validation datasets."""
    global A

    original_A = A
    results: dict[str, dict[str, object]] = {}
    try:
        for name, overrides in SIM_SOURCES.items():
            A = overrides["A"]
            print(f"\n=== {name}: A={A}, {SIM_NCELLS} cells ===")
            data = [
                simulate(
                    SIM_NP,
                    SIM_TMAX,
                    1e-4,
                    SIM_TS,
                    1000 * int(A) + index,
                    sigma_loc,
                    Z_L,
                    SIM_DC,
                )
                for index in range(SIM_NCELLS)
            ]
            results[name] = analyze_source(name, data, SIM_TS)
    finally:
        A = original_A
    return results


# =============================================================================
# Result serialization and plotting
# =============================================================================


def save_csv(results: dict[str, dict[str, object]], path: PathLike) -> None:
    """Save cell-level and optional pooled results to a compact CSV file."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "source", "x", "dU", "D_slow", "D_fast", "n"])
        for name, result in results.items():
            for cell in result["cells"]:  # type: ignore[index]
                writer.writerow(
                    [
                        "cell",
                        name,
                        cell["x"],
                        cell["dU"],
                        cell["D_slow"],
                        cell["D_fast"],
                        cell["n"],
                    ]
                )
            group = result["group"]
            if group:
                writer.writerow(
                    [
                        "group",
                        name,
                        group["x"],
                        group["dU"],
                        group["D_slow"],
                        group["D_fast"],
                        group["n_pts"],
                    ]
                )


def load_csv(path: PathLike) -> dict[str, dict[str, object]]:
    """Load results previously written by :func:`save_csv`."""
    results: dict[str, dict[str, object]] = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            result = results.setdefault(row["source"], {"cells": [], "group": None})
            record = {
                "x": float(row["x"]),
                "dU": float(row["dU"]),
                "D_slow": float(row["D_slow"]),
                "D_fast": float(row["D_fast"]),
                "n": float(row["n"]),
            }
            if row["kind"] == "cell":
                result["cells"].append(record)  # type: ignore[union-attr]
            else:
                record["n_pts"] = record.pop("n")
                record["n_cells"] = float(len(result["cells"]))  # type: ignore[arg-type]
                result["group"] = record
    return results


def make_figure(results: dict[str, dict[str, object]], out_png: PathLike) -> None:
    """Create the diffusivity-density summary figure and matching PDF."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    cmap = plt.get_cmap("tab10")
    order = list(DATASETS.keys()) + list(SIM_SOURCES.keys())
    color = {name: cmap(i % 10) for i, name in enumerate(order)}
    mean_x: list[float] = []
    mean_y: list[float] = []

    for i, (name, result) in enumerate(results.items()):
        source_color = color.get(name, cmap(i % 10))
        cells = result["cells"]
        xs = np.asarray([cell["x"] for cell in cells], dtype=float)
        ys = np.exp(-np.asarray([cell["dU"] for cell in cells], dtype=float))
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)

        ax.scatter(
            xs[valid],
            ys[valid],
            s=12,
            color=source_color,
            alpha=0.18,
            edgecolors="none",
            zorder=1,
        )

        group = result["group"]
        if group and np.isfinite(group["x"]) and np.isfinite(group["dU"]):
            mx = float(group["x"])
            my = float(np.exp(-group["dU"]))
            sx = sy = 0.0
            n = int(group["n_cells"])
        elif valid.sum() > 0:
            mx = float(np.mean(xs[valid]))
            my = float(np.mean(ys[valid]))
            sx = float(np.std(xs[valid]) / np.sqrt(valid.sum()))
            sy = float(np.std(ys[valid]) / np.sqrt(valid.sum()))
            n = int(valid.sum())
        else:
            ax.scatter([], [], color=source_color, label=f"{name} (n=0)")
            continue

        xerr = [[min(sx, mx * 0.95)], [sx]]
        yerr = [[min(sy, my * 0.95)], [sy]]
        ax.errorbar(
            mx,
            my,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=11,
            color=source_color,
            mec="k",
            mew=1.2,
            ecolor="k",
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
            label=f"{name} (n={n})",
        )
        mean_x.append(mx)
        mean_y.append(my)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if mean_x:
        ax.set_xlim(min(mean_x) / 2.5, max(mean_x) * 2.5)
        ax.set_ylim(min(mean_y) / 3.0, max(mean_y) * 3.0)

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    low = min(x_limits[0], y_limits[0])
    high = max(x_limits[1], y_limits[1])
    ax.plot([low, high], [low, high], "k:", lw=1.0, zorder=0, label=r"$y=x$")
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel(r"$\dfrac{D_{\rm slow}-D_b}{D_{\rm fast}-D_b}$")
    ax.set_ylabel(r"$\rho_{\rm out}/\rho_{\rm in}$")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()

    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nSaved figure -> {out_png} (+ {out_png.with_suffix('.pdf').name})")


# =============================================================================
# Command-line interface
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for standalone use."""
    parser = argparse.ArgumentParser(
        description="Run the publication diffusivity-density analysis."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("data", "sim"),
        default="data",
        help="Analyse experimental data (default) or synthetic validation data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute results even when a cached CSV already exists.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing AllData/ (default: directory of this script).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    cache = Path(f"results_{args.mode}.csv")

    if cache.exists() and not args.force:
        print(f"Loading cached {cache} (use --force to recompute).")
        results = load_csv(cache)
    else:
        results = run_real(args.data_root) if args.mode == "data" else run_sim()
        save_csv(results, cache)
        print(f"Wrote {cache}")

    make_figure(results, f"dU_vs_diffratio_{args.mode}.png")


if __name__ == "__main__":
    main()
