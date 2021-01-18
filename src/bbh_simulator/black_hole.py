"""
Module to facilate merging of BHs
"""

import logging
from typing import List, Optional

import numpy as np
import surfinBH

BH_FIT = surfinBH.LoadFits("NRSur7dq4Remnant")

PM = "\u00B1"  # ±
CHI = "\u03C7"  # χ
c = 299792  # speed of liught in km/s


class BlackHole:
    bh_counter = 0

    def __init__(
        self,
        mass: float,
        spin: List[float],
        parents: Optional = None,
        kick: Optional[float] = [0, 0, 0],
        mass_unc: Optional[float] = None,
        spin_unc: Optional[List[float]] = None,
        kick_unc: Optional[List[float]] = [0, 0, 0],
    ):
        self.mass = mass
        self.spin = spin
        self.kick = kick
        self.mass_unc = mass_unc
        self.spin_unc = spin_unc
        self.kick_unc = kick_unc
        self.parents = parents
        self.generation_number = self._get_generation_number()
        self.id = BlackHole.bh_counter
        BlackHole.bh_counter += 1

    def __str__(self):
        if self.mass_unc:
            vk = mag(self.kick) * c
            vk_unc = mag(self.kick_unc) * c

            return (
                f"Gen# {self.generation_number}\n"
                f"Mass {self.mass:.2f} {PM} {self.mass_unc:.2f}\n"
                f"|{CHI}| {mag(self.spin):.2f} {PM} {mag(self.spin_unc):.2f}\n"
                f"|vk| {vk:.2f}  {PM} {vk_unc:.2f} km/s"
            )
        else:
            return (
                f"Gen# {self.generation_number}\n"
                f"Mass {self.mass:.2f}\n"
                f"|{CHI}| {mag(self.spin):.2f}"
            )

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return dict(
            mass=self.mass,
            spin_mag=self.spin_mag,
            spin_x=self.spin[0],
            spin_y=self.spin[1],
            spin_z=self.spin[2],
            kick_mag=self.kick_mag,
            kick_x=self.kick[0] * c,
            kick_y=self.kick[1] * c,
            kick_z=self.kick[2] * c,
            spin_unc_x=self.spin_unc[0],
            kick_unc_x=self.kick_unc[0] * c,
            spin_unc_y=self.spin_unc[1],
            kick_unc_y=self.kick_unc[1] * c,
            spin_unc_z=self.spin_unc[2],
            kick_unc_z=self.kick_unc[2] * c,
        )

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, parents):
        if parents:
            assert len(parents) == 2
        self._parents = parents

    @property
    def spin(self):
        return self._spin

    @property
    def kick_mag(self):
        return mag(self.kick) * c

    @property
    def spin_mag(self):
        return mag(self.spin)

    @spin.setter
    def spin(self, spin):
        assert len(spin) == 3
        self._spin = spin

    def _get_generation_number(self):
        if not self.parents:
            return 0
        else:
            return self.parents[0].generation_number + 1


def merge_bbh_pair(bh_1, bh_2):
    """Merges two BH and returns the final mass with kick and final spin

    This uses the NRSur7dq4Remnant model to predict the final mass mf,
    final spin vector chif and final kick velocity vector vf, for the remnants
    of precessing binary black hole systems.  The fits are done using Gaussian
    Process Regression (GPR) and also provide an error estimate along with the
    fit value.

    See arxiv:1905.09300

    NOTE:
    |  This model has been trained in the parameter space:
    |      q <= 4, |chi_a| <= 0.8, |chi_b| <= 0.8
    |
    |  However, it extrapolates reasonably to:
    |      q <= 6, |chi_a| <= 1, |chi_b| <= 1

    q: float
        Mass ratio (q = mA/mB >= 1)
    chi_b: [float, float, float]
        Dimensionless spin vector of the heavier black hole at reference epoch.
    chi_b: [float, float, float]
        Dimensionless spin vector of the lighter black hole at reference epoch.

    Notes on chi_a and chi_b:
    Follows the same convention as LAL, where the spin
    components are defined as:
    -> \chi_z = \chi \cdot \hat{L},
    -> \chi_x = \chi \cdot \hat{n},
    -> \chi_y = \chi \cdot \hat{L \cross n}.

    where L is the orbital angular momentum vector at the epoch.
    where n = body2 -> body1 is the separation vector at the epoch
    (body1 is the heavier body)
    These spin components are frame-independent as they are defined
    using vector inner products. This is equivalent to specifying
    the spins in the coorbital frame at the reference epoch.

    To use aligned spin, set chi[0]=chi[1]=0

    https://github.com/vijayvarma392/surfinBH/blob/master/examples/example_7dq4.ipynb

    :param bh_1: BlackHole
        1st black hole being merged
    :param bh_2: BlackHole
        2nd black hole being merged

    :return: BlackHole
        The merger remnant

    """
    if bh_1.mass / bh_2.mass >= 1:
        q = bh_1.mass / bh_2.mass
        chi_a = bh_1.spin
        chi_b = bh_2.spin
    else:
        q = bh_2.mass / bh_1.mass
        chi_a = bh_2.spin
        chi_b = bh_1.spin

    # Check if merging BH compatible with fit
    m_chi_a = mag(chi_a)
    m_chi_b = mag(chi_b)
    if q > 4:
        logging.warning(f"q={q:.1f} > 4")
    if m_chi_a > 0.8:
        logging.warning(f"|chi_a|={m_chi_a:.1f} > 0.8")
    if m_chi_b > 0.8:
        logging.warning(f"|chi_b|={m_chi_b:.1f} > 0.8")

    # Merging BH
    total_mass = bh_1.mass + bh_2.mass
    try:
        mf, chif, vf, mf_err, chif_err, vf_err = BH_FIT.all(
            q=q, chiA=chi_a, chiB=chi_b
        )
        remnant = BlackHole(
            mass=mf * total_mass,
            mass_unc=mf_err * total_mass,
            spin=chif,
            spin_unc=chif_err,
            kick=vf,
            kick_unc=vf_err,  # units of c
            parents=[bh_1, bh_2],
        )
    except ValueError as e:
        logging.error(f"Cannot merge {bh_2} {bh_2}: {e}. Returning Nans.")
        nan_vec = [np.NaN, np.NaN, np.NaN]
        remnant = BlackHole(
            mass=np.NaN,
            mass_unc=np.NaN,
            spin=nan_vec,
            spin_unc=nan_vec,
            kick=nan_vec,
            kick_unc=nan_vec,  # units of c
            parents=[bh_1, bh_2],
        )
    return remnant


def mag(x):
    return np.linalg.norm(x)


if __name__ == "__main__":
    pass
