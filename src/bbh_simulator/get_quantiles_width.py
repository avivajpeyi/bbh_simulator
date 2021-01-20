# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import pandas as pd
import plot_corner_weighted_with_kick
import tqdm

FILE = "/fred/oz117/avajpeyi/projects/phase-marginalisation-test/jobs/out_hundred_injections_gstar/out_injection_{num}/result/injection_{num}_0_posterior_samples_with_kicks.dat"


def main():
    widths = []
    truths = pd.read_csv("datafiles/injections.csv")
    for index, truth in tqdm.tqdm(truths.iterrows(), total=len(truths)):
        f = FILE.format(num=index)
        kick_mean = truth["remnant_kick_mag"]
        kick_sigma = 50

        samples = plot_corner_weighted_with_kick.Samples(
            samples_csv=f,
            kick_mean=kick_mean,
            kick_sigma=kick_sigma,
            truths=truth.to_dict(),
        )
        try:
            widths.append(samples.get_quantiles_width("chi_p"))
        except Exception as e:
            print(f"Error: {e}")

    df = pd.DataFrame(widths)
    df.to_csv("quantiles_widths.csv")


if __name__ == "__main__":
    main()
