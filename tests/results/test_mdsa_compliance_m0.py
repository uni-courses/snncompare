""""Runs the snns on a specific run config, and verifies that the degree
receiver nodes at m_0 have the same:

- random values
- weights (=degree for m=0)
- marks
- countermarks
as the corresponding neighbours in the Neumann algorithm.
"""


# python -m pytest tests/results/Test_mdsa_compliance_m0.py
# --capture=tee-sys
# python -m pytest tests/results/Test_mdsa_compliance_m0.py --capture=tee-sys

# Run a run config.

from pprint import pprint
from typing import Dict
from unittest import TestCase

from typeguard import typechecked

from snncompare.arg_parser.arg_parser import parse_cli_args
from snncompare.arg_parser.process_args import manage_export_parsing
from snncompare.exp_config.Exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.json_configurations.algo_test import load_exp_config_from_file
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config


@typechecked
def tested_run_config() -> Run_config:
    """Returns the specific run config that is being tested."""
    run_config: Run_config = Run_config(
        **{
            "adaptation": {"redundancy": 2},
            "algorithm": {"MDSA": {"m_val": 0}},
            "graph_nr": 2,
            "graph_size": 6,
            "max_duration": None,
            "radiation": {"neuron_death": 0.01},
            "seed": 7,
            "simulator": "simsnn",
        }
    )
    return run_config


@typechecked
def custom_mdsa_compliance_m0_exp_config() -> Exp_config:
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    long_mdsa_testing: Dict = {
        "adaptations": {"redundancy": [2]},
        "algorithms": {
            "MDSA": [
                {"m_val": 0},
            ]
        },
        # TODO: Change into list with "Seeds"
        "seeds": [7],
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "max_graph_size": 20,
        "max_max_graphs": 4,
        "min_graph_size": 20,
        "min_max_graphs": 4,
        "radiations": {
            "neuron_death": [
                0.01,
            ]
        },
        "size_and_max_graphs": [(6, 4)],
        "simulators": ["simsnn"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }

    # The ** loads the dict into the object.
    exp_config = Exp_config(**long_mdsa_testing)
    return exp_config


class Test_something(TestCase):
    """Tests whether the function output_stage_json() creates valid output json
    files."""

    # def setUp(self):

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        mock_parser = parse_cli_args(parse=False)
        # parsed = self.parser.parse_args(['--something', 'test'])
        mock_args = mock_parser.parse_args(
            ["-e", "debug6", "-j1", "-j2", "-dr", "-rev"]
        )
        pprint(mock_args.__dict__)

        custom_config_path: str = "src/snncompare/json_configurations"
        exp_config: Exp_config = load_exp_config_from_file(
            custom_config_path=custom_config_path,
            filename=mock_args.experiment_settings_name,
        )
        print("")
        pprint(exp_config.__dict__)

        # if mock_args.run_config_path is not None:
        # # specific_run_config: Union[
        # # # None, Run_config
        # # # ] = load_run_config_from_file(
        # # # custom_config_path=custom_config_path,
        # # filename=f"{mock_args.run_config_path}",
        # # )
        # # else:
        # pass
        output_config: Output_config = manage_export_parsing(args=mock_args)

        full_exp_runner = Experiment_runner(
            exp_config=exp_config,
            output_config=output_config,
            reverse=False,
            perform_run=False,
            specific_run_config=None,
        )

        for one_at_a_time_run_config in full_exp_runner.run_configs:
            # python -m src.snncompare -e mdsa_creation_only_size_3_4 -v
            exp_runner: Experiment_runner = Experiment_runner(
                exp_config=exp_config,
                output_config=output_config,
                perform_run=any(
                    x in output_config.output_json_stages for x in [1, 2, 3, 4]
                ),
                reverse=mock_args.reverse,
                specific_run_config=one_at_a_time_run_config,
                # specific_run_config=tested_run_config(),
            )
        pprint(exp_runner.results_nx_graphs)

    def test_marks_equality_m_0(
        self,
    ) -> None:
        """Tests whether the marks of the snn equal the Alipour/Neumann
        marks."""
        print(self.__dict__)
        self.assertEqual(True, True)
        self.assertEqual(True, False)


# For SNN:
# Get the total offset in the rand spike input.
# input_graph.graph["alg_props"]["rand_edge_weights"][node_index]
# Probably retrieved at:
# get_degree_receiver_offset
# Optionally disect it into: -degree_buffer+ rand_nrs+1selector_input+margin.

# Get the random values per node in that offset of the rand spike input.
# get_rand__degree_receiver_edge_weights
# Verify the random values per node go into
# the neighbour in degree_receiver_<node_circuit>_<neighbour>. (DONE MANUALLY.)

# The marks are, (for m=0) the values of the spike_once inputs at t=0.

# Get the weights:
# (=degree+rand, = for SNN marks+rand for m=0)

# The countermarks are the nr of spike_once inputs


# For Neumann
# Get the random values per node.
# Get the weight (=degree for m=0)
# Get the countermarks per node
# Get the marks per node

# Assert the mark of Neumann equals ==
# sum of input spike_once_neurons + rand nr of that node.
# Optionally, assert that the degree_receiver - neumann mark all yield the
# same starting baseline.

# Assert Neumann weights equal spike_once input sum.
