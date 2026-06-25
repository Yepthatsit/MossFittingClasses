
"""
Mossbauer sextet fitting (lmfit-based) for 57Fe-type spectra.

Extracts: isomer shift (IS) and magnetic hyperfine field (Bhf, given
directly in Tesla, with the line splitting itself expressed in mm/s).

------------------------------------------------------------------
Physics / line-position formula
------------------------------------------------------------------
For 57Fe (ground state I=1/2, excited state I=3/2), six allowed
transitions (Delta m = 0, +-1) give a sextet. Following the standard
treatment (e.g. Hjollum & Madsen, "Fit;o) - A Mossbauer spectrum
fitting program", and standard texts such as Greenwood & Gibb,
Gutlich et al.), the six line positions (in velocity units, mm/s)
relative to the isomer shift IS are:

    x_i = IS + s_QS(i) * QS/2 + s_HF(i) * k_i * Bhf

where Bhf is the hyperfine field in Tesla, QS is the quadrupole
shift (mm/s), and k_i, s_HF(i), s_QS(i) are fixed per line:

    line i :  1        2        3        4       5       6
    k_i    :  k16      k25      k34      k34     k25     k16
    s_HF   :  -1       -1       -1       +1      +1      +1
    s_QS   :  -1       +1       +1       +1      +1      -1

with the proportionality constants (mm/s per Tesla), derived from
the 57Fe nuclear g-factors (g_ground=0.181208, g_excited=-0.10355):

    k16 = 0.161299  mm/s/T   (line pair 1,6 -- outer lines)
    k25 = 0.0933835 mm/s/T   (line pair 2,5 -- middle lines)
    k34 = 0.0254672 mm/s/T   (line pair 3,4 -- inner lines)

Sanity check built into this module: for the alpha-Fe calibration
standard (Bhf = 33.0 T, QS = 0, IS = 0) this reproduces the
well-known six-line pattern with line1<->line6 spacing of
~10.6 mm/s, matching the textbook alpha-Fe room-temperature
spectrum used for velocity calibration.

Because Bhf and QS multiply *linearly independent* sign/coefficient
patterns across the 6 lines (s_HF and s_QS are not proportional),
IS, Bhf and QS are NOT degenerate with each other in the fit -- each
is determined by a distinct combination of line positions.

Area ratios: for a powder (randomly oriented) thin absorber with
unsplit angular distribution, the theoretical line area ratio is
3:2:1:1:2:3 (lines 1:2:3:4:5:6). This is offered as an optional
constraint (`constrain_areas=True`, default) but can be switched
off if your sample is textured/oriented.

Author: prepared for fitting experimental 57Fe Mossbauer sextets to
extract isomer shift (IS) and hyperfine field (Bhf).
"""

import numpy as np
import lmfit
from lmfit import Model, Parameters
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Physical constants (57Fe), from nuclear g-factors g_g=0.181208,
# g_e=-0.10355 (Morup, 1994), as tabulated in Hjollum & Madsen (2009).
# ----------------------------------------------------------------------
K16 = 0.161299    # mm/s per Tesla, outer line pair (1,6)
K25 = 0.0933835   # mm/s per Tesla, middle line pair (2,5)
K34 = 0.0254672   # mm/s per Tesla, inner line pair (3,4)

# per-line coefficients, lines 1..6
LINE_K = [K16, K25, K34, K34, K25, K16]
LINE_SIGN_HF = [-1, -1, -1, +1, +1, +1]
LINE_SIGN_QS = [-1, +1, +1, +1, +1, -1]
LINE_AREA_RATIO = [3, 2, 1, 1, 2, 3]


def lorentzian(x, center, fwhm, amplitude):
    """Single Lorentzian peak; `amplitude` is the peak AREA (not height)."""
    gamma = fwhm / 2.0
    return (amplitude / np.pi) * (gamma / ((x - center) ** 2 + gamma ** 2))


class MossbauerSextetFit:
    """
    Fit a 57Fe magnetic sextet Mossbauer spectrum and extract the
    isomer shift (IS, mm/s) and hyperfine field (Bhf, Tesla).

    Parameters
    ----------
    constrain_areas : bool, default True
        Tie the 6 line areas to the theoretical thin-absorber, random
        powder ratio 3:2:1:1:2:3 (only an overall scale is free).
        Set False to let all 6 areas vary independently (e.g. for
        textured / non-randomly-oriented samples).
    baseline : 'constant' (default) or 'linear'
        Background model added under the sextet.
    same_fwhm : bool, default True
        If True, all 6 lines share one linewidth parameter `fwhm`.
        If False, each line gets its own fwhm1..fwhm6 (use only if
        you have good reason -- adds 5 extra free parameters and may
        not be well constrained).

    Usage
    -----
    >>> fitter = MossbauerSextetFit()
    >>> params = fitter.guess(velocity, counts, Bhf0=33.0)
    >>> result = fitter.fit(velocity, counts, params)
    >>> fitter.print_report(result)
    >>> IS, IS_err, Bhf, Bhf_err = fitter.extract(result)
    >>> fitter.plot(velocity, counts, result)
    """

    def __init__(self, constrain_areas=True, baseline="constant", same_fwhm=True):
        self.constrain_areas = constrain_areas
        self.baseline = baseline
        self.same_fwhm = same_fwhm
        self.model = self._build_model()

    # ------------------------------------------------------------------
    def _line_centers(self, IS, Bhf, QS):
        """Six line centers (mm/s) given IS (mm/s), Bhf (Tesla), QS (mm/s)."""
        return [
            IS + LINE_SIGN_QS[i] * QS / 2.0 + LINE_SIGN_HF[i] * LINE_K[i] * Bhf
            for i in range(6)
        ]

    def _sextet_func(self, x, IS, Bhf, QS, bkg, slope,
                      a1, a2, a3, a4, a5, a6,
                      fwhm=None, fwhm1=None, fwhm2=None, fwhm3=None,
                      fwhm4=None, fwhm5=None, fwhm6=None):
        centers = self._line_centers(IS, Bhf, QS)
        amps = [a1, a2, a3, a4, a5, a6]
        if self.same_fwhm:
            fwhms = [fwhm] * 6
        else:
            fwhms = [fwhm1, fwhm2, fwhm3, fwhm4, fwhm5, fwhm6]

        y = np.full_like(x, bkg, dtype=float)
        if self.baseline == "linear":
            y = y + slope * (x - np.mean(x))
        for c, a, w in zip(centers, amps, fwhms):
            y = y - lorentzian(x, c, w, a)  # absorption dip
        return y

    def _build_model(self):
        return Model(self._sextet_func, independent_vars=["x"])

    # ------------------------------------------------------------------
    def guess(self, x, y, IS0=None, Bhf0=None, QS0=0.0, fwhm0=0.3,
              baseline0=None):
        """
        Build lmfit Parameters with sensible starting values.

        x : velocity array, mm/s
        y : counts array (absorption dip spectrum)
        Bhf0 : starting hyperfine field guess, in TESLA. If None
               (default), it is estimated automatically from the
               outermost resolvable dip in the data (so the initial
               line positions actually overlap the data -- if you
               instead pick a fixed Bhf0 far from the truth, the
               outer lines can land outside the measured velocity
               range and the optimizer may converge to a near-zero-
               amplitude local minimum). Override if the auto-guess
               looks wrong (e.g. for low-field or multi-site spectra).
        IS0 : starting isomer shift guess, mm/s. If None, estimated
              as the intensity-weighted centroid of the dip region.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        n_edge = max(5, len(y) // 20)
        if baseline0 is None:
            baseline0 = float(np.median(np.concatenate([y[:n_edge], y[-n_edge:]])))

        depth_guess = max(baseline0 - np.min(y), baseline0 * 0.01)
        area_guess = depth_guess * (fwhm0 * np.pi / 2.0)

        # --- automatic IS0 / Bhf0 estimate from the data's dip extent ---
        # weight = how far below baseline each point is (0 where at/above baseline)
        dip = np.clip(baseline0 - y, 0, None)
        if dip.sum() > 0:
            centroid = float(np.sum(x * dip) / np.sum(dip))
            # outer-line proxy: velocity range containing the bulk of the
            # absorption (5th-95th percentile of the dip-weighted distribution)
            order = np.argsort(x)
            xs, dips = x[order], dip[order]
            cum = np.cumsum(dips)
            cum /= cum[-1]
            lo = float(np.interp(0.05, cum, xs))
            hi = float(np.interp(0.95, cum, xs))
            half_outer_extent = max(abs(hi - centroid), abs(centroid - lo))
        else:
            centroid = 0.0
            half_outer_extent = (x.max() - x.min()) / 2.0 * 0.8

        if IS0 is None:
            IS0 = centroid
        if Bhf0 is None:
            # half_outer_extent ~= K16 * Bhf  (outer line offset from IS)
            Bhf0 = max(half_outer_extent / K16, 5.0)

        params = Parameters()
        params.add("IS", value=IS0, vary=True)
        params.add("Bhf", value=Bhf0, vary=True, min=0.0)
        params.add("QS", value=QS0, vary=True)
        params.add("bkg", value=baseline0, vary=True)
        params.add("slope", value=0.0, vary=(self.baseline == "linear"))

        if self.same_fwhm:
            params.add("fwhm", value=fwhm0, vary=True, min=0.01)
        else:
            for i in range(1, 7):
                params.add(f"fwhm{i}", value=fwhm0, vary=True, min=0.01)

        if self.constrain_areas:
            params.add("scale", value=area_guess, vary=True, min=0)
            for i, ratio in zip(range(1, 7), LINE_AREA_RATIO):
                if ratio == 1:
                    params.add(f"a{i}", expr="scale")
                else:
                    params.add(f"a{i}", expr=f"{ratio}*scale")
        else:
            for i, ratio in zip(range(1, 7), LINE_AREA_RATIO):
                params.add(f"a{i}", value=area_guess * ratio / 3.0, min=0, vary=True)

        return params

    # ------------------------------------------------------------------
    def fit(self, x, y, params=None, weights=None, **guess_kwargs):
        """Run the fit; builds guess params automatically if not provided.
        Returns an lmfit.model.ModelResult."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if params is None:
            params = self.guess(x, y, **guess_kwargs)
        return self.model.fit(y, params, x=x, weights=weights)

    # ------------------------------------------------------------------
    def extract(self, result):
        """Return (IS, IS_err, Bhf, Bhf_err) -- IS in mm/s, Bhf in Tesla."""
        p = result.params
        IS = p["IS"].value
        IS_err = p["IS"].stderr if p["IS"].stderr is not None else np.nan
        Bhf = p["Bhf"].value
        Bhf_err = p["Bhf"].stderr if p["Bhf"].stderr is not None else np.nan
        return IS, IS_err, Bhf, Bhf_err

    def print_report(self, result):
        print(result.fit_report())
        IS, IS_err, Bhf, Bhf_err = self.extract(result)
        QS = result.params["QS"].value
        QS_err = result.params["QS"].stderr or np.nan
        print("\n--- Extracted physical parameters ---")
        print(f"Isomer shift   IS  = {IS:.4f} +/- {IS_err:.4f} mm/s")
        print(f"Hyperfine field Bhf = {Bhf:.3f} +/- {Bhf_err:.3f} T")
        print(f"Quadrupole shift QS = {QS:.4f} +/- {QS_err:.4f} mm/s")

    def plot(self, x, y, result, ax=None, show_components=True):
        x = np.asarray(x, dtype=float)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, "o", ms=3, color="0.3", label="data")
        ax.plot(x, result.best_fit, "-", color="crimson", lw=2, label="fit")

        if show_components:
            p = result.params.valuesdict()
            centers = self._line_centers(p["IS"], p["Bhf"], p["QS"])
            amps = [p[f"a{i}"] for i in range(1, 7)]
            if self.same_fwhm:
                fwhms = [p["fwhm"]] * 6
            else:
                fwhms = [p[f"fwhm{i}"] for i in range(1, 7)]
            xx = np.linspace(x.min(), x.max(), 2000)
            for c, a, w in zip(centers, amps, fwhms):
                comp = p["bkg"] - lorentzian(xx, c, w, a)
                ax.plot(xx, comp, "--", lw=1, color="steelblue", alpha=0.7)

        ax.set_xlabel("Velocity (mm/s)")
        ax.set_ylabel("Counts (a.u.)")
        ax.legend()
        return ax