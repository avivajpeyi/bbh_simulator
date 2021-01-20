"""Python module to load postetrior_samples and calculate their kicks"""

import logging
import os
import sys
import warnings

import pandas as pd
import tqdm
from bilby.gw import conversion

from bbh_simulator.black_hole import BlackHole, merge_bbh_pair

warnings.filterwarnings("ignore")


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.tqdm.write(msg, end="")


logging.basicConfig(level=logging.INFO, stream=TqdmStream)
log = logging.getLogger(__name__)


class Samples:
    def __init__(self, posterior: pd.DataFrame):
        self.posterior = posterior

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename
        self.outfile = self.filename.replace(".dat", "_with_kicks.dat")
        if os.path.isfile(self.outfile):
            raise FileExistsError(
                f"{self.outfile} already present! Terminating job."
            )

    @classmethod
    def from_file(cls, filename):
        posteior = Samples.read_file(filename)
        samples = cls(posteior)
        samples.filename = filename
        return samples

    @staticmethod
    def read_file(filename: str):
        log.info(f"Loading posterior from {filename}")
        assert os.path.isfile(filename)
        _, file_extension = os.path.splitext(filename)
        if file_extension == ".dat":
            posterior = pd.read_csv(filename, " ")
        else:
            posterior = pd.read_csv(filename)
        assert (
            len(posterior.columns.values) > 2
        ), f"Error reading posterior: {posterior}"
        posterior = conversion.generate_all_bbh_parameters(posterior)
        posterior = conversion.generate_component_spins(posterior)
        posterior["id"] = posterior.index + 1
        posterior = posterior.set_index("id")
        return posterior

    def calculate_remnant_kick_velocity(self):
        p = self.posterior
        p["id"] = p.index
        p = p.set_index("id")
        remnant_data = []
        progress_bar = tqdm.tqdm(
            p.iterrows(), total=len(p), desc="Calculating Kicks"
        )
        for idx, sample in progress_bar:
            remnant_dict = get_sample_kick(sample)
            remnant_dict = {f"remnant_{k}": v for k, v in remnant_dict.items()}
            remnant_dict.update(dict(id=idx))
            remnant_data.append(remnant_dict)
        remnant_df = pd.DataFrame(remnant_data)
        remnant_df = remnant_df.set_index("id")
        assert len(remnant_df) == len(p)
        self.posterior = p.merge(remnant_df, on="id")

    def save_samples_with_kicks(self):
        self.calculate_remnant_kick_velocity()
        self.posterior.to_csv(self.outfile)
        log.info(f"Saved posterior with kicks in {self.outfile}")


def get_sample_kick(s):
    """

    :param s: A posterior sample
    :return:
    """
    remnant = merge_bbh_pair(
        bh_1=BlackHole(
            mass=s.mass_1_source, spin=[s.spin_1x, s.spin_1y, s.spin_1z]
        ),
        bh_2=BlackHole(
            mass=s.mass_2_source, spin=[s.spin_2x, s.spin_2y, s.spin_2z]
        ),
    )
    return remnant.to_dict()


def main():
    samples_filename = sys.argv[1]
    log.info(f"Calculating kicks for {samples_filename}")
    samples = Samples(filename=samples_filename)
    samples.save_samples_with_kicks()


if __name__ == "__main__":
    main()
