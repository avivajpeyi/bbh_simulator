# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import logging
import os

import corner
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from bilby.gw import conversion
from bilby.gw.prior import BBHPriorDict

from bbh_simulator.black_hole import BlackHole, merge_bbh_pair
from bbh_simulator.corner_kwargs import CORNER_KWARGS

logging.getLogger().setLevel(logging.INFO)
ALIGNED_SAMPLES = "aligned_spin_samples.csv"
PRECESSING_SAMPLES = "precession_spin_samples.csv"

DIRNAME = os.path.dirname(__file__)

ALIGNED_POPULATION_PRIOR = os.path.join(
    DIRNAME, "datafiles/aligned_population.prior"
)
PRECESSING_POPULATION_PRIOR = os.path.join(
    DIRNAME, "datafiles/precessing_spin_population.prior"
)
AGN_POPULATION_PRIOR = os.path.join(DIRNAME, "datafiles/agn_population.prior")

REMNANT_KICK = "remnant_kick"
REMNANT_MASS = "remnant_mass"
REMNANT_SPIN = "remnant_spin"


class Samples:
    def __init__(self, prior_file, num_samples=1000):
        self.prior_filename = prior_file
        self.prior = BBHPriorDict(filename=prior_file)
        self.num_samples = num_samples
        self.samples = self.__generate_samples()

    def __generate_samples(self):
        samples = pd.DataFrame(self.prior.sample(size=self.num_samples))
        samples = conversion.generate_all_bbh_parameters(samples)
        samples = conversion.generate_component_spins(samples)
        if "chi_1" not in samples:
            samples["chi_1"] = samples["spin_1z"]
            samples["chi_2"] = samples["spin_2z"]
        samples[REMNANT_KICK] = None
        samples[REMNANT_MASS] = None
        samples[REMNANT_SPIN] = None
        return samples

    def add_remnant_data_to_samples(self):
        for idx, s in tqdm.tqdm(
            self.samples.iterrows(), total=self.num_samples, desc="Merging BH"
        ):
            remnant = merge_bbh_pair(
                bh_1=BlackHole(
                    mass=s.mass_1, spin=[s.spin_1x, s.spin_1y, s.spin_1z]
                ),
                bh_2=BlackHole(
                    mass=s.mass_2, spin=[s.spin_2x, s.spin_2y, s.spin_2z]
                ),
            )
            self.samples.at[idx, REMNANT_MASS] = remnant.mass
            self.samples.at[idx, REMNANT_SPIN] = remnant.spin_mag
            self.samples.at[idx, REMNANT_KICK] = remnant.kick_mag

    def plot_corner(self, filename):
        param_of_interest = [
            "chi_1",
            "chi_2",
            "mass_ratio",
            "remnant_kick",
            "remnant_spin",
        ]
        if "chi_1_in_plane" in self.samples:
            param_of_interest += ["chi_1_in_plane", "chi_2_in_plane"]

        corner.corner(self.samples[param_of_interest], **CORNER_KWARGS)
        font = {"color": "darkblue", "weight": "bold"}
        plt.suptitle(filename, fontdict=font, fontsize=40)
        plt.savefig(f"{filename}.png")
        plt.close()

    def save_samples(self):
        self.samples.to_csv(
            self.prior_filename.replace(".prior", ".csv"), index=False
        )


def main():
    agn_samples = Samples(AGN_POPULATION_PRIOR)
    agn_samples.add_remnant_data_to_samples()
    agn_samples.save_samples()

    # aligned_samples = Samples(ALIGNED_POPULATION_PRIOR)
    # aligned_samples.add_remnant_data_to_samples()
    # aligned_samples.save_samples()
    #
    # precessing_samples = Samples(PRECESSING_POPULATION_PRIOR)
    # precessing_samples.add_remnant_data_to_samples()
    # precessing_samples.save_samples()


if __name__ == "__main__":
    main()
