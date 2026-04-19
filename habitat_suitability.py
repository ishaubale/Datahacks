"""
Habitat Suitability Index (HSI) for Southern California fish species.

Given predicted water conditions (temperature, DO, depth, chlorophyll),
compute a 0-1 suitability score per species.

Ranges come from published fisheries literature. These are simplified
"preferendum" models — real HSI work uses weighted multivariate fuzzy logic,
but for a hackathon demo this is defensible and interpretable.

References (for the pitch slide):
- Pacific sardine (Sardinops sagax): Lluch-Belda et al., optimum 13-25 C
- Northern anchovy (Engraulis mordax): 11.5-21 C optimum
- Pacific mackerel (Scomber japonicus): 14-22 C
- DO tolerances: most pelagic species require > 2 mL/L (hypoxia threshold)
"""

import numpy as np


SPECIES = {
    "Pacific Sardine": {
        "temp_opt": (14.0, 22.0),    # optimum temperature range (C)
        "temp_tol": (10.0, 25.0),    # tolerable range
        "depth_opt": (0, 50),         # pelagic, upper water column
        "depth_tol": (0, 200),
        "do_min": 2.0,                # mL/L
        "chl_opt": (1.0, 10.0),       # productive waters
    },
    "Northern Anchovy": {
        "temp_opt": (13.0, 18.0),
        "temp_tol": (11.0, 21.0),
        "depth_opt": (0, 40),
        "depth_tol": (0, 150),
        "do_min": 2.0,
        "chl_opt": (2.0, 15.0),
    },
    "Pacific Mackerel": {
        "temp_opt": (15.0, 20.0),
        "temp_tol": (14.0, 22.0),
        "depth_opt": (0, 80),
        "depth_tol": (0, 300),
        "do_min": 2.5,
        "chl_opt": (0.5, 8.0),
    },
    "Pacific Hake": {
        "temp_opt": (7.0, 12.0),      # cooler, deeper
        "temp_tol": (5.0, 15.0),
        "depth_opt": (100, 400),
        "depth_tol": (50, 500),
        "do_min": 1.5,
        "chl_opt": (0.2, 3.0),
    },
}


def _trapezoid(value, opt_range, tol_range):
    """Trapezoidal membership: 1 inside opt, linear decay to 0 at tol edges."""
    opt_lo, opt_hi = opt_range
    tol_lo, tol_hi = tol_range
    if value < tol_lo or value > tol_hi:
        return 0.0
    if opt_lo <= value <= opt_hi:
        return 1.0
    if value < opt_lo:
        return (value - tol_lo) / (opt_lo - tol_lo)
    return (tol_hi - value) / (tol_hi - opt_hi)


def suitability(species: str, temp: float, depth: float,
                do: float, chl: float) -> float:
    """Return HSI in [0, 1] for a species given conditions."""
    if species not in SPECIES:
        raise ValueError(f"Unknown species: {species}")
    s = SPECIES[species]

    t_score = _trapezoid(temp, s["temp_opt"], s["temp_tol"])
    d_score = _trapezoid(depth, s["depth_opt"], s["depth_tol"])

    # DO is a hard threshold — fish suffocate below it
    do_score = 0.0 if do < s["do_min"] else min(1.0, (do - s["do_min"]) / 2.0)

    # Chlorophyll — forage availability proxy
    c_lo, c_hi = s["chl_opt"]
    if c_lo <= chl <= c_hi:
        c_score = 1.0
    elif chl < c_lo:
        c_score = max(0.0, chl / c_lo) if c_lo > 0 else 0.0
    else:
        c_score = max(0.0, 1.0 - (chl - c_hi) / c_hi)

    # Geometric mean — any single failing factor tanks overall suitability
    # (a fish can't live without oxygen even if temp is perfect)
    scores = [t_score, d_score, do_score, c_score]
    if min(scores) == 0:
        return 0.0
    return float(np.prod(scores) ** (1 / len(scores)))


def suitability_all(temp, depth, do, chl) -> dict:
    """Return HSI for all species."""
    return {sp: suitability(sp, temp, depth, do, chl) for sp in SPECIES}
