import os
import shutil
import unittest

import corner
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams

from bbh_simulator.utils import CORNER_KWARGS


class TestCornerKwargs(unittest.TestCase):
    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_something(self):
        rcParams["font.size"] = 20
        rcParams["font.family"] = "serif"
        rcParams["font.sans-serif"] = ["Computer Modern Sans"]
        rcParams["text.usetex"] = False
        rcParams["axes.labelsize"] = 30
        rcParams["axes.titlesize"] = 30
        rcParams["axes.labelpad"] = 20
        m1_true, m2_true = 35, 40

        samples = pd.DataFrame(
            dict(
                m1=np.random.normal(loc=m1_true, scale=15, size=1000),
                m2=np.random.normal(loc=m2_true, scale=30, size=1000),
            )
        )
        truths = [m1_true + 20, m2_true - 5]
        corner.corner(samples, **CORNER_KWARGS, truths=truths)
        fname = f"{self.outdir}/bayesian.png"
        plt.savefig(fname)
        self.assertTrue(os.path.isfile(fname))

        CORNER_KWARGS.update(
            dict(fill_contours=False, quantiles=None, alpha=0, color="white")
        )
        corner.corner(samples, **CORNER_KWARGS, truths=truths)
        fname = f"{self.outdir}/freq.png"
        plt.savefig(fname)
        self.assertTrue(os.path.isfile(fname))


if __name__ == "__main__":
    unittest.main()
