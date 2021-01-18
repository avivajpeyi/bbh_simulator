import logging
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tqdm
from bilby.core.prior import Constraint, Uniform
from bilby.gw.prior import BBHPriorDict

from . import utils
from .black_hole import BlackHole, merge_bbh_pair

logging.getLogger().setLevel(logging.INFO)

POPULATION_PRIOR = BBHPriorDict(
    dictionary=dict(
        # Mass
        mass_1=Uniform(name="mass_1", minimum=18, maximum=32),
        mass_2=Uniform(name="mass_1", minimum=18, maximum=32),
        mass_ratio=Constraint(name="mass_ratio", minimum=0.7, maximum=1),
        a_1=Uniform(name="a_1", minimum=0, maximum=0.2),
        a_2=Uniform(name="a_2", minimum=0, maximum=0.2),
    )
)


class BlackHolePopulation:
    def __init__(self, number_of_initial_bh):
        self.number_of_initial_bh = number_of_initial_bh
        self.population = {}
        self.__init_population()

    @property
    def population_size(self):
        return len(self.population)

    @property
    def number_of_generation(self):
        return len(self.get_generation_counts())

    @property
    def number_of_initial_bh(self):
        return self._number_of_initial_bh

    @number_of_initial_bh.setter
    def number_of_initial_bh(self, number_of_initial_bh):
        if not utils.is_power_of_two(number_of_initial_bh):
            raise ValueError(
                f"{number_of_initial_bh} needs to be a power of 2."
            )
        self._number_of_initial_bh = number_of_initial_bh

    def __init_population(self):
        logging.info(f"Initialising {self.number_of_initial_bh} BH")
        BlackHole.bh_counter = 0
        bbh_pop_prior = POPULATION_PRIOR
        num_bbh_pairs = self.number_of_initial_bh // 2
        bbh_population_data = bbh_pop_prior.sample(num_bbh_pairs)
        self.population = {}
        for i in range(num_bbh_pairs):
            bh_1 = BlackHole(
                mass=bbh_population_data["mass_1"][i],
                spin=[0, 0, bbh_population_data["a_1"][i]],
            )
            bh_2 = BlackHole(
                mass=bbh_population_data["mass_2"][i],
                spin=[0, 0, bbh_population_data["a_2"][i]],
            )
            self.population.update({bh.id: bh for bh in [bh_1, bh_2]})

    def conduct_multiple_mergers(self):
        """Repeatedly conduct mergers until only 1 BH remaining"""
        merging_bh_list = list(self.population.values())
        while len(merging_bh_list) > 1:
            merging_bh_list = self.__truncate_bh_list(merging_bh_list)
            remnant_list = self.merge_bh_list(merging_bh_list)
            self.population.update({bh.id: bh for bh in remnant_list})
            merging_bh_list = remnant_list

        assert self.population_size == BlackHole.bh_counter
        logging.info(
            f"All BH mergers complete resulting "
            f"in total number of BH: {self.population_size}"
        )

    @staticmethod
    def __truncate_bh_list(
        merging_bh_list: List[BlackHole]
    ) -> List[BlackHole]:
        if not len(merging_bh_list) % 2 == 0:
            logging.info(f"Merging BH population uneven, removing a BH")
            merging_bh_list.pop()
        return merging_bh_list

    @staticmethod
    def pair_up_bh(
        merging_bh_list: List[BlackHole]
    ) -> List[Tuple[BlackHole, BlackHole]]:
        # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
        paired_up_population = list(zip(merging_bh_list, merging_bh_list[1:]))[
            ::2
        ]
        return paired_up_population

    def merge_bh_list(
        self, merging_bh_list: List[BlackHole]
    ) -> List[BlackHole]:
        logging.info(f"Merging pairs in list of {len(merging_bh_list)} BH")
        paired_up_population = self.pair_up_bh(merging_bh_list)
        remnants = []
        for bh_1, bh_2 in paired_up_population:
            remnant = merge_bbh_pair(bh_1, bh_2)
            remnants.append(remnant)
        return remnants

    def get_generation_counts(self) -> Dict[int, int]:
        generation_counts = dict()
        for bh in self.population.values():
            count = generation_counts.get(bh.generation_number, 0)
            generation_counts.update({bh.generation_number: count + 1})
        return generation_counts

    def get_generation_stats(self) -> pd.DataFrame:
        init_col_vals = [0.0 for _ in range(self.number_of_generation)]
        stats = pd.DataFrame(
            dict(
                count=[i for i in self.get_generation_counts().values()],
                avg_kick=init_col_vals,
                avg_mass=init_col_vals,
                avg_spin=init_col_vals,
            )
        )
        # totaling the various stats
        for bh in self.population.values():
            stats.at[bh.generation_number, "avg_kick"] += (
                utils.mag(bh.kick) * utils.c
            )
            stats.at[bh.generation_number, "avg_mass"] += bh.mass
            stats.at[bh.generation_number, "avg_spin"] += utils.mag(bh.spin)
        # averaging the vals
        stats["avg_kick"] = stats["avg_kick"].astype("float") / stats[
            "count"
        ].astype("float")
        stats["avg_mass"] = stats["avg_mass"].astype("float") / stats[
            "count"
        ].astype("float")
        stats["avg_spin"] = stats["avg_spin"].astype("float") / stats[
            "count"
        ].astype("float")

        return stats

    def _get_graph_generation_data(self):
        nodes, edges, labels = [], [], {}
        for bh_id, bh in self.population.items():
            labels.update({bh_id: str(bh)})
            nodes.append(bh_id)
            if bh.parents:
                edges += [(bh.id, p.id) for p in bh.parents]
        edges = utils.remove_duplicate_edges(edges)
        return nodes, edges, labels

    def render_population(self, filename):
        nodes, edges, labels = self._get_graph_generation_data()

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        fig = plt.figure(
            figsize=(
                self.number_of_initial_bh * 2,
                self.number_of_generation + 5,
            )
        )
        fig.suptitle(
            f"{self.number_of_generation - 1} Generations of BH mergers",
            fontsize=16,
        )
        pos = hierarchy_pos(
            graph=graph,
            root=nodes[-1],
            width=self.number_of_initial_bh * 2,
            vert_gap=0.01,
        )
        nx.draw(graph, pos=pos, labels=labels)
        plt.savefig(filename)

    def render_spin_and_mass(self, filename, stats=None):
        if not isinstance(stats, pd.DataFrame):
            stats = self.get_generation_stats()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(stats.index, stats.avg_mass, color="red", marker="o")
        ax2.plot(stats.index, stats.avg_spin, color="blue", marker="o")
        ax1.set_xlabel("Generation Number", fontsize=14)
        ax1.set_ylabel("Average Mass", color="red", fontsize=14)
        ax2.set_ylabel("Average |\u03C7|", color="blue", fontsize=14)
        ax2.set_ylim(0, 1)
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.margins(0.05)
        plt.tight_layout()
        plt.savefig(filename)

    def run_expiriment(self) -> pd.DataFrame:
        self.__init_population()
        self.conduct_multiple_mergers()
        return self.get_generation_stats()

    def repeat_expirement(self, num_expt: int) -> pd.DataFrame:
        expt_stats = []
        expt_progress_bar = tqdm.tqdm(range(num_expt))
        for expt_num in expt_progress_bar:
            expt_progress_bar.set_description(f"Expt {expt_num}")
            expt_stats.append(self.run_expiriment())
        avg_stats = expt_stats[0]
        for i in range(len(expt_stats)):
            avg_stats = avg_stats.add(expt_stats[i])
        avg_stats = avg_stats / len(expt_stats)
        return avg_stats


def hierarchy_pos(
    graph, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5
):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to overlaid_corner this in a
    hierarchical layout.

    graph: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(graph):
        raise TypeError(
            "cannot use hierarchy_pos on a graph that is not a tree"
        )

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(graph))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(graph.nodes))

    def _hierarchy_pos(
        graph,
        root,
        width=1.0,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(graph.neighbors(root))
        if not isinstance(graph, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    graph,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)


def main():
    pop = BlackHolePopulation(number_of_initial_bh=2 ** 5)
    pop.conduct_multiple_mergers()
    pop.render_population("mergers.png")
    stats = pop.repeat_expirement(num_expt=5)
    pop.render_spin_and_mass("stats.png", stats=stats)


if __name__ == "__main__":
    main()
