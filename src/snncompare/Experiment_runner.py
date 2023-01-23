"""Contains the object that runs the entire experiment. Also Contains a single
setting of the experiment configuration settings.

(The values of the settings may vary, yet the types should be the same.)
"""
import functools
import timeit
from decimal import Decimal
from pprint import pprint
from typing import Dict, List, Optional

from snnbackends.plot_graphs import create_root_dir_if_not_exists
from snnbackends.verify_nx_graphs import verify_results_nx_graphs
from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    Exp_config,
    Supported_experiment_settings,
    append_unique_exp_config_id,
)
from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.helper import dicts_are_equal, generate_run_configs

from .exp_config.run_config.verify_run_completion import (
    assert_stage_is_completed,
)
from .exp_config.run_config.verify_run_settings import verify_has_unique_id
from .export_results.load_json_to_nx_graph import (
    load_json_to_nx_graph_from_file,
)
from .export_results.Output_stage_12 import output_files_stage_1_and_2
from .export_results.Output_stage_34 import output_stage_files_3_and_4
from .graph_generation.stage_1_get_input_graphs import get_used_graphs
from .import_results.check_completed_stages import has_outputted_stage
from .import_results.stage_1_load_input_graphs import load_results_stage_1
from .process_results.process_results import (
    export_results_to_json,
    set_results,
)
from .simulation.stage2_sim import sim_graphs

template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""
timeit.template = template  # type: ignore[attr-defined]


class Experiment_runner:
    """Experiment manager.

    First prepares the environment for running the experiment, and then
    calls a private method that executes the experiment consisting of 4
    stages.
    """

    # pylint: disable=R0903

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        specific_run_config: Optional[Run_config] = None,
        perform_run: bool = True,
    ) -> None:

        # Ensure output directories are created for stages 1 to 4.
        create_root_dir_if_not_exists("results")

        # Store the experiment configuration settings.
        self.exp_config = exp_config

        # Load the ranges of supported settings.
        self.supp_exp_config = Supported_experiment_settings()

        # Verify the experiment exp_config are complete and valid.
        # pylint: disable=R0801
        # verify_exp_config(
        #    self.supp_exp_config,
        #    exp_config,
        #    has_unique_id=False,
        #    allow_optional=True,
        # )

        # If the experiment exp_config does not contain a hash-code,
        # create the unique hash code for this configuration.
        # TODO: restore
        if not self.supp_exp_config.has_unique_config_id(self.exp_config):
            append_unique_exp_config_id(
                self.exp_config,
            )

        # Verify the unique hash code for this configuration is valid.
        verify_has_unique_id(self.exp_config.__dict__)

        self.run_configs = generate_run_configs(
            exp_config, specific_run_config
        )

        # Perform runs accordingly.
        if perform_run:
            self.__perform_run(self.exp_config, self.run_configs)

    # pylint: disable=W0238
    @typechecked
    def __perform_run(
        self, exp_config: Exp_config, run_configs: List[Run_config]
    ) -> None:
        """Private method that performs a run of the experiment.

        The 2 underscores indicate it is private. This method executes
        the run in the way the processed configuration settings specify.
        """
        duration: float
        results_nx_graphs: Dict
        for i, run_config in enumerate(run_configs):

            print(f"\n{i+1}/{len(run_configs)} [runs]")
            pprint(run_config.__dict__)
            self.to_run = determine_what_to_run(run_config)
            print("\nstart stage I:  ", end=" ")

            duration, results_nx_graphs = timeit.Timer(  # type:ignore[misc]
                functools.partial(
                    self.__perform_run_stage_1,
                    exp_config,
                    run_config,
                    self.to_run,
                )
            ).timeit(1)
            print(f"{round(duration,5)} [s]")
            print("Start stage II  ", end=" ")
            duration, _ = timeit.Timer(  # type:ignore[misc]
                functools.partial(
                    self.__perform_run_stage_2, results_nx_graphs, self.to_run
                )
            ).timeit(1)
            print(f"{round(duration,5)} [s]")
            print("Start stage III ", end=" ")
            duration, _ = timeit.Timer(  # type:ignore[misc]
                functools.partial(
                    self.__perform_run_stage_3, results_nx_graphs, self.to_run
                )
            ).timeit(1)
            print(f"{round( Decimal(round(float(duration), 5)),5)} [s]")

            print("Start stage IV  ", end=" ")
            duration, _ = timeit.Timer(  # type:ignore[misc]
                functools.partial(
                    self.__perform_run_stage_4,
                    self.exp_config.export_images,
                    results_nx_graphs,
                    self.to_run,
                )
            ).timeit(1)
            print(f"{round(duration,5)} [s]")
            # Store run results in dict of Experiment_runner.
            self.results_nx_graphs: Dict = {
                run_config.unique_id: results_nx_graphs  # type:ignore[index]
            }

    @typechecked
    def __perform_run_stage_1(
        self,
        exp_config: Exp_config,
        run_config: Run_config,
        to_run: Dict,
    ) -> Dict:
        """Performs the run for stage 1 or loads the data from file depending
        on the run configuration.

        Stage 1 applies a conversion that the user specified to run an
        SNN algorithm. This is done by taking an input graph, and
        generating an SNN (graph) that runs the intended algorithm.
        """
        if to_run["stage_1"]:

            # Run first stage of experiment, get input graph.
            stage_1_graphs: Dict = get_used_graphs(run_config)
            results_nx_graphs = {
                "exp_config": exp_config,
                "run_config": run_config,
                "graphs_dict": stage_1_graphs,
            }

            # Exports results, including graphs as dict.
            output_files_stage_1_and_2(results_nx_graphs, 1, to_run)
        else:
            results_nx_graphs = load_results_stage_1(run_config)
        self.equalise_loaded_run_config(
            results_nx_graphs["run_config"],
            run_config,
        )
        assert_stage_is_completed(run_config, 1, to_run)
        return results_nx_graphs

    @typechecked
    def equalise_loaded_run_config(
        self,
        loaded_from_json: Run_config,
        incoming: Run_config,
    ) -> None:
        """Ensures the non-impactfull run config that is loaded from json are
        identical to those of the  incoming run_config."""
        for key, val in incoming.__dict__.items():

            if loaded_from_json.__dict__[key] != val:
                print(
                    f"setting run config[{key}] from "
                    + f"{loaded_from_json.__dict__[key]} to:{val}"
                )
                loaded_from_json.__dict__[key] = val
        if not dicts_are_equal(
            loaded_from_json.__dict__,
            incoming.__dict__,
            without_unique_id=False,
        ):
            pprint(loaded_from_json.__dict__)
            pprint(incoming.__dict__)
            raise AttributeError(
                "Run_config and loaded run_config from json are not equal."
            )

    @typechecked
    def __perform_run_stage_2(
        self,
        results_nx_graphs: Dict,
        to_run: Dict,
    ) -> None:
        """Performs the run for stage 2 or loads the data from file depending
        on the run configuration.

        Stage two simulates the SNN graphs over time and, if desired,
        exports each timestep of those SNN graphs to a json dictionary.
        """
        # Verify incoming results dict.
        verify_results_nx_graphs(
            results_nx_graphs, results_nx_graphs["run_config"]
        )

        if to_run["stage_2"]:
            if not results_nx_graphs["run_config"].recreate_s4:
                # TODO: check if the stage 2 graphs already are loaded from
                # file correctly. If loaded incorrectly, raise exception, if
                # not loaded, perform simulation.

                # TODO: check if the graphs can be loaded from file,
                if has_outputted_stage(
                    results_nx_graphs["run_config"], 2, to_run
                ):
                    # Load results from file.
                    nx_graphs_dict = load_json_to_nx_graph_from_file(
                        results_nx_graphs["run_config"], 2, to_run
                    )
                    results_nx_graphs["graphs_dict"] = nx_graphs_dict

            # TODO: Verify the (incoming (and loaded)) graph types are as
            # expected.

            # Run simulation on networkx or lava backend.
            sim_graphs(
                results_nx_graphs["run_config"],
                results_nx_graphs["graphs_dict"],
            )
            output_files_stage_1_and_2(results_nx_graphs, 2, to_run)

        assert_stage_is_completed(
            results_nx_graphs["run_config"], 2, to_run, verbose=True
        )

    @typechecked
    def __perform_run_stage_3(
        self,
        results_nx_graphs: Dict,
        to_run: Dict,
    ) -> None:
        """Performs the run for stage 3, which visualises the behaviour of the
        SNN graphs over time. This behaviour is shown as a sequence of images.

        The behaviour is described with:
        - Green neuron: means a neuron spikes in that timestep.
        - Green synapse: means the input neuron of that synapse spiked at that
        timestep.
        - Red neuron: radiation has damaged/killed the neuron, it won't spike.
        - Red synapse: means the input neuron of that synapse has died and will
        not spike at that timestep.
        - White/transparent neuron: works as expected, yet does not spike at
        that timestep.
        - A circular synapse: a recurrent connection of a neuron into itself.
        """
        if to_run["stage_3"]:
            # Generate output json dicts (and plots) of propagated graphs.
            # TODO: pass the stage index and re-use it to export the
            # stage 4 graphs
            if (
                results_nx_graphs["run_config"].overwrite_images_only
                or results_nx_graphs["run_config"].export_images
            ):
                output_stage_files_3_and_4(results_nx_graphs, 3, to_run)
            assert_stage_is_completed(
                results_nx_graphs["run_config"],
                3,
                to_run,
                verbose=True,
                results_nx_graphs=results_nx_graphs,
            )

            # TODO: assert gif file exists

    @typechecked
    def __perform_run_stage_4(
        self, export_images: bool, results_nx_graphs: Dict, to_run: Dict
    ) -> None:
        """Performs the run for stage 4.

        Stage 4 computes the results of the SNN against the
        default/Neumann implementation. Then stores this result in the
        last entry of each graph.
        """
        if set_results(
            results_nx_graphs["run_config"],
            results_nx_graphs["graphs_dict"],
        ):
            export_results_to_json(export_images, results_nx_graphs, 4, to_run)
        assert_stage_is_completed(
            results_nx_graphs["run_config"], 4, to_run, verbose=True
        )


@typechecked
def determine_what_to_run(
    run_config: Run_config,
) -> Dict[str, bool]:
    """Scans for existing output and then combines the run configuration
    settings to determine what still should be computed."""
    # Initialise default: run everything.
    to_run = {
        "stage_1": False,
        "stage_2": False,
        "stage_3": False,
        "stage_4": False,
    }

    # Check if the input graphs exist, (the graphs that can still be adapted.)
    if (
        not has_outputted_stage(run_config, 1, to_run)
        or run_config.recreate_s1
        or run_config.recreate_s2
        or run_config.recreate_s4
    ):
        # If original graphs do not yet exist, or a manual overwrite is
        # requested, create them (Note it only asks for an overwrite of
        # the sim results, but it is assumed this means re-do the
        # simulation).
        to_run["stage_1"] = True

    # Check if the incoming graphs have been supplemented with adaptation
    # and/or radiation.
    if (
        not has_outputted_stage(run_config, 2, to_run)
        or run_config.recreate_s1
        or run_config.recreate_s2
        or run_config.recreate_s4
    ):
        to_run["stage_2"] = True
    # Check if the visualisation of the graph behaviour needs to be created.
    if run_config.export_images and (
        not has_outputted_stage(run_config, 3, to_run)
        or run_config.overwrite_images_only
    ):
        # Note this allows the user to create inconsistent simulation
        # results and visualisation. E.g. the simulated behaviour may
        # have changed due to code changes, yet the visualisation would
        # not be updated stage 3 has already been performed, with
        # recreate_s4=True, and overwrite_images_only=False.
        to_run["stage_3"] = True
        print(f'to_run["stage_3"]={to_run["stage_3"]}')

    # Throw warning to user about potential discrepancy between graph
    # behaviour and old visualisation.

    if (
        not has_outputted_stage(run_config, 4, to_run)
        or run_config.recreate_s1
        or run_config.recreate_s2
        or run_config.recreate_s4
    ):
        to_run["stage_4"] = True
    return to_run
