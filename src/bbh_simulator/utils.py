import math

import numpy as np

c = 299792  # speed of liught in km/s

BILBY_BLUE_COLOR = "#0072C1"
VIOLET_COLOR = "#8E44AD"

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=True,
    use_math_text=True,
)


def remove_duplicate_edges(lst):
    return [t for t in (set(tuple(i) for i in lst))]


def is_power_of_two(x):
    return x and (not (x & (x - 1)))


def mag(x):
    return math.sqrt(math.fsum([i ** 2 for i in x]))
