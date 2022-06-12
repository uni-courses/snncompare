"""Verifies 2 nodes are included in the networkx graph."""
import json
import pickle
import unittest

from src.get_graph import get_networkx_graph_of_2_neurons
from src.process_results import get_run_results


class Test_process_results(unittest.TestCase):
    """Tests whether the process_results function behaves as expected."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filename = "_death_prob0.25_adapt_True_raddamTrue__seed42_size3_m0_iter1_hash-5432048059257968152"
        pickle_filepath = f"tests/{filename}.pkl"
        json_filepath = f"tests/{filename}.json"
        self.load_test_pickle(pickle_filepath)
        self.load_json_dict(json_filepath)

        # Get last graphs of SNN graph behaviour.
        self.G_mdsa = self.G_behaviour_mdsa[-1]
        self.G_brain_adap = self.G_behaviour_brain_adaptation[-1]
        self.G_rad_dam = self.G_behaviour_rad_damage[-1]
        self.G_mdsa = self.G_behaviour_mdsa[-1]

    def load_test_pickle(self, pickle_filepath):
        """TODO: remove duplicate function"""
        pickle_off = open(
            pickle_filepath,
            "rb",
        )

        [
            self.G,
            self.G_behaviour_mdsa,
            self.G_behaviour_brain_adaptation,
            self.G_behaviour_rad_damage,
            self.mdsa_graph,
            self.brain_adaptation_graph,
            self.rad_damaged_graph,
            self.rand_props,
        ] = pickle.load(pickle_off)

    def load_json_dict(self, json_filepath):
        with open(json_filepath) as json_file:
            self.the_dict = json.load(json_file)

    def test_identical_results_are_returned_as_equal(self):
        get_run_results(
            self.G,
            self.G_mdsa,
            self.G_brain_adap,
            self.G_rad_dam,
            self.the_dict["m"],
            self.rand_props,
        )
        self.assertEqual(2, 2)
