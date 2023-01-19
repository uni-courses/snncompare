"""Tests whether the exp_config gets the same unique_id at all times if its
content is the same, and that it gets different unique ids for different
exp_config settings."""
import copy
import unittest

# pylint: disable=R0801
from typing import Dict, List

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from typeguard import typechecked

from snncompare.exp_config.custom_setts.run_configs.algo_test import (
    long_exp_config_for_mdsa_testing,
)
from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import verify_exp_config


class Test_setting_unique_id_exp_config(unittest.TestCase):
    """Tests whether the exp_config gets the same unique_id at all times if its
    content is the same, and that it gets different unique ids for different
    exp_config settings."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        # Generate default experiment config.
        self.exp_config_list: List = [
            get_config_one(),
            get_config_two(),
        ]

        for exp_config in self.exp_config_list:
            exp_config.show_snns = False
            exp_config.export_images = False
            verify_exp_config(
                Supported_experiment_settings(),
                exp_config,
                has_unique_id=False,
                allow_optional=True,
            )

    @typechecked
    def test_same_exp_config_same_unique_id(self) -> None:
        """Verifies the same run config gets the same unique_id."""

        # Generate run configurations.
        self.assertGreaterEqual(len(self.exp_config_list), 2)

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for row, row_exp_config in enumerate(self.exp_config_list):
            for col, col_exp_config in enumerate(
                copy.deepcopy(self.exp_config_list)
            ):
                if row == col:
                    self.assertTrue(row_exp_config == col_exp_config)
                else:
                    self.assertFalse(row_exp_config == col_exp_config)

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for index, exp_config in enumerate(self.exp_config_list):
            supp_setts = Supported_experiment_settings()
            supp_setts.append_unique_exp_config_id(
                exp_config, allow_optional=True
            )
            if index == 0:
                self.assertEqual(
                    exp_config.unique_id,
                    "77352e201ee23bab9958b0a6b9e2bae409e9f9234cb79e33"
                    + "c181855cfeeb7a98",
                )
            if index == 1:
                self.assertEqual(
                    exp_config.unique_id,
                    "471ce9c875abffe41f6b23bea5bc5ed67d869c1962acf0"
                    + "5cf7937cf2348826b9",
                )


def get_config_one() -> Dict:
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""
    # Create prerequisites
    # supp_exp_config = Supported_experiment_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    mdsa_creation_only_size_3_4: Dict = {
        "adaptations": None,
        # TODO: set using a verification setting.
        "algorithms": {
            "MDSA": get_algo_configs(MDSA(list(range(1, 2, 1))).__dict__)
        },
        "iterations": list(range(0, 1, 1)),
        # TODO: Change into list with "Seeds"
        "seed": 7,
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 2,
        "min_graph_size": 3,
        "max_graph_size": 4,
        # "size_and_max_graphs": [(3, 1), (4, 3)],
        # "size_and_max_graphs": [(3, 1),(4, 1)],
        "size_and_max_graphs": [(3, 1), (4, 1)],
        # Move into "overwrite options"
        "recreate_s1": True,
        "recreate_s2": True,
        "overwrite_images_only": True,
        "recreate_s4": True,
        "radiations": None,
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }

    verify_exp_config(
        Supported_experiment_settings(),
        mdsa_creation_only_size_3_4,
        has_unique_id=False,
        allow_optional=True,
    )
    return mdsa_creation_only_size_3_4


def get_config_two() -> Dict:
    """Returns a default experiment setting with  graph size 7, m=4."""
    mdsa_creation_only_size_7_m_4: Dict = long_exp_config_for_mdsa_testing()
    mdsa_creation_only_size_7_m_4["algorithms"] = {
        "MDSA": get_algo_configs(MDSA(list(range(4, 5, 1))).__dict__)
    }
    mdsa_creation_only_size_7_m_4["size_and_max_graphs"] = [(5, 1)]
    return mdsa_creation_only_size_7_m_4
