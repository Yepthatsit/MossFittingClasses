import numpy as np
from lmfit import Model
from lmfit.lineshapes import lorentzian


class MossbauerSextetModel(Model):
    """
    A model for a 57Fe Mössbauer sextet.
    Parameters:
        is_shift: Isomer shift (mm/s)
        b_hf: Hyperfine field (Tesla)
        eps: Quadrupole shift (mm/s)
        gamma: FWHM linewidth (mm/s)
        amplitude: Total area
        ratio: Intensity ratio of line 2 to 3 (default 2 for powder)
    """

    def __init__(self, *args, **kwargs):
        def sextet_func(x, is_shift=0.0, b_hf=33.0, eps=0.0, gamma=0.25, amplitude=-1000, ratio=2.0):
            # Physical constants for 57Fe
            gg = 0.181208
            ge = -0.103542
            # Conversion factor: mm/s per Tesla
            C = 0.65569

            # Line positions (order: 1 is leftmost, 6 is rightmost)
            # Factors derived from energy level transitions
            v1 = is_shift + (1.5 * ge - 0.5 * gg) * C * b_hf + eps
            v2 = is_shift + (0.5 * ge - 0.5 * gg) * C * b_hf - eps
            v3 = is_shift + (-0.5 * ge - 0.5 * gg) * C * b_hf - eps
            v4 = is_shift + (0.5 * ge + 0.5 * gg) * C * b_hf - eps
            v5 = is_shift + (-0.5 * ge + 0.5 * gg) * C * b_hf - eps
            v6 = is_shift + (-1.5 * ge + 0.5 * gg) * C * b_hf + eps

            # Relative intensities (3 : ratio : 1 : 1 : ratio : 3)
            # Total relative sum = 8 + 2*ratio
            norm = 8.0 + 2.0 * ratio
            i1 = i6 = 3.0 / norm
            i2 = i5 = ratio / norm
            i3 = i4 = 1.0 / norm

            # Sum of 6 Lorentzians
            # Note: lmfit.lineshapes.lorentzian uses 'sigma' (FWHM = 2*sigma)
            sigma = gamma / 2.0
            y = amplitude * (
                    i1 * lorentzian(x, amplitude=1.0, center=v1, sigma=sigma) +
                    i2 * lorentzian(x, amplitude=1.0, center=v2, sigma=sigma) +
                    i3 * lorentzian(x, amplitude=1.0, center=v3, sigma=sigma) +
                    i4 * lorentzian(x, amplitude=1.0, center=v4, sigma=sigma) +
                    i5 * lorentzian(x, amplitude=1.0, center=v5, sigma=sigma) +
                    i6 * lorentzian(x, amplitude=1.0, center=v6, sigma=sigma)
            )
            return y

        super().__init__(sextet_func, *args, **kwargs)

    def guess(self, x, y):
        # Rough initialization for a typical iron foil
        params = self.make_params(is_shift=0, b_hf=33.0, eps=0, gamma=0.3, ratio=2.0)
        # Background is usually handled by adding a ConstantModel()
        params['amplitude'].set(value=np.min(y) * (np.max(x) - np.min(x)), min=-1e9, max=0)
        return params


class MossbauerRelaxationSextetModel(Model):
    def __init__(self, *args, **kwargs):
        def relaxation_func(x, is_shift=0.0, b_hf=33.0, eps=0.0, gamma=0.3,
                            amplitude=-1000, ratio=2.0, log10_w=6.0):
            """
            log10_w: log10 of the relaxation rate (Hz).
            Typically between 6 (static) and 9 (fully collapsed).
            """
            # Physical Constants
            gg, ge, C = 0.181208, -0.103542, 0.65569
            w = 10 ** log10_w  # Convert log rate to Hz

            # Line positions (omega_j) relative to center
            # These are the positions in the static limit
            v_pos = np.array([
                (1.5 * ge - 0.5 * gg) * C * b_hf + eps,
                (0.5 * ge - 0.5 * gg) * C * b_hf - eps,
                (-0.5 * ge - 0.5 * gg) * C * b_hf - eps,
                (0.5 * ge + 0.5 * gg) * C * b_hf - eps,
                (-0.5 * ge + 0.5 * gg) * C * b_hf - eps,
                (-1.5 * ge + 0.5 * gg) * C * b_hf + eps
            ])

            # Intensities
            norm = 8.0 + 2.0 * ratio
            weights = np.array([3.0, ratio, 1.0, 1.0, ratio, 3.0]) / norm

            # Blume-Tjon simplified formula for a two-state flip
            # Gamma is half-width at half-max in angular freq units
            gamma_nat = gamma / 2.0

            total_abs = np.zeros_like(x, dtype=complex)

            # Loop through the 6 transitions
            for oj, weight in zip(v_pos, weights):
                # The Blume-Tjon expression for one line flipping between +oj and -oj
                # This handles the transition from sextet to singlet
                p = gamma_nat - 1j * (x - is_shift)
                line_shape = (p + 2 * w) / (p ** 2 + 2 * w * p + oj ** 2)
                total_abs += weight * line_shape

            return amplitude * total_abs.real

        super().__init__(relaxation_func, *args, **kwargs)

    def guess(self, x, y):
        params = self.make_params(is_shift=0, b_hf=33.0, eps=0, gamma=0.3,
                                  amplitude=-1000, ratio=2.0, log10_w=6.0)
        return params