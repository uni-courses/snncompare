"""Contains the object that runs the entire experiment. Also Contains a single
setting of the experiment configuration settings.

(The values of the settings may vary, yet the types should be the same.)
"""


from typing import Any, Dict, List, Tuple

from src.snncompare.exp_setts.run_config.Supported_run_settings import (
    Supported_run_settings,
)
from src.snncompare.exp_setts.run_config.verify_run_completion import (
    assert_stage_is_completed,
)
from src.snncompare.exp_setts.run_config.verify_run_settings import (
    verify_run_config,
)
from src.snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
    verify_has_unique_id,
)
from src.snncompare.export_results.load_json_to_nx_graph import (
    load_json_to_nx_graph_from_file,
)
from src.snncompare.export_results.Output_stage_12 import (
    output_files_stage_1_and_2,
)
from src.snncompare.export_results.Output_stage_34 import (
    output_stage_files_3_and_4,
)
from src.snncompare.export_results.plot_graphs import (
    create_root_dir_if_not_exists,
)
from src.snncompare.export_results.verify_nx_graphs import (
    verify_results_nx_graphs,
)
from src.snncompare.graph_generation.stage_1_get_input_graphs import (
    get_used_graphs,
)
from src.snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)
from src.snncompare.import_results.stage_1_load_input_graphs import (
    load_results_stage_1,
)
from src.snncompare.process_results.process_results import (
    export_results_to_json,
    set_results,
)
from src.snncompare.simulation.stage2_sim import sim_graphs


class Experiment_runner:
    """Experiment manager.

    First prepares the environment for running the experiment, and then
    calls a private method that executes the experiment consisting of 4
    stages.
    """

    # pylint: disable=R0903

    def __init__(
        self, experiment_config: dict, export_images: bool, show_snns: bool
    ) -> None:

        # Ensure output directories are created for stages 1 to 4.
        create_root_dir_if_not_exists("results")

        # Store the experiment configuration settings.
        self.experiment_config = experiment_config

        # Load the ranges of supported settings.
        self.supp_exp_setts = Supported_experiment_settings()

        # Verify the experiment experiment_config are complete and valid.
        # pylint: disable=R0801
        verify_experiment_config(
            self.supp_exp_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

        # If the experiment experiment_config does not contain a hash-code,
        # create the unique hash code for this configuration.
        if not self.supp_exp_setts.has_unique_config_id(
            self.experiment_config
        ):
            self.supp_exp_setts.append_unique_config_id(self.experiment_config)

        # Verify the unique hash code for this configuration is valid.
        verify_has_unique_id(self.experiment_config)

        # Append the export_images and show_snns arguments.
        self.experiment_config["export_images"] = export_images
        self.experiment_config["show_snns"] = show_snns

        # Perform runs accordingly.
        # TODO: see if self.run_configs can be removed.
        self.run_configs = self.__perform_run(self.experiment_config)

    # pylint: disable=W0238
    def __perform_run(self, experiment_config: dict) -> List[dict]:
        """Private method that performs a run of the experiment.

        The 2 underscores indicate it is private. This method executes
        the run in the way the processed configuration settings specify.
        """
        # Generate run configurations.
        run_configs: List[dict] = experiment_config_to_run_configs(
            experiment_config
        )

        for run_config in run_configs:
            to_run = determine_what_to_run(run_config)
            results_nx_graphs = self.__perform_run_stage_1(
                experiment_config, run_config, to_run
            )
            self.__perform_run_stage_2(results_nx_graphs, to_run)
            self.__perform_run_stage_3(results_nx_graphs, to_run)
            self.__perform_run_stage_4(
                self.experiment_config["export_images"],
                results_nx_graphs,
                to_run,
            )

        return run_configs

    def __perform_run_stage_1(
        self, experiment_config: dict, run_config: dict, to_run: dict
    ) -> dict:
        """Performs the run for stage 1 or loads the data from file depending
        on the run configuration.

        Stage 1 applies a conversion that the user specified to run an
        SNN algorithm. This is done by taking an input graph, and
        generating an SNN (graph) that runs the intended algorithm.
        """
        if to_run["stage_1"]:

            # Run first stage of experiment, get input graph.
            stage_1_graphs: dict = get_used_graphs(run_config)
            results_nx_graphs = {
                "experiment_config": experiment_config,
                "run_config": run_config,
                "graphs_dict": stage_1_graphs,
            }

            # Exports results, including graphs as dict.
            output_files_stage_1_and_2(results_nx_graphs, 1, to_run)
        else:
            results_nx_graphs = load_results_stage_1(run_config)

        assert_stage_is_completed(run_config, 1, to_run)
        return results_nx_graphs

    def __perform_run_stage_2(
        self,
        results_nx_graphs: dict,
        to_run: dict,
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
            if not results_nx_graphs["run_config"]["overwrite_sim_results"]:
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
                results_nx_graphs["graphs_dict"],
                results_nx_graphs["run_config"],
            )
            output_files_stage_1_and_2(results_nx_graphs, 2, to_run)
        assert_stage_is_completed(results_nx_graphs["run_config"], 2, to_run)

    def __perform_run_stage_3(
        self,
        results_nx_graphs: dict,
        to_run: dict,
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
            print("Generating plots for stage 3.")
            # TODO: pass the stage index and re-use it to export the
            # stage 4 graphs
            output_stage_files_3_and_4(results_nx_graphs, 3, to_run)
            print('"Done generating output plots for stage 3.')
            assert_stage_is_completed(
                results_nx_graphs["run_config"], 3, to_run
            )

    def __perform_run_stage_4(
        self, export_images: bool, results_nx_graphs: dict, to_run: dict
    ) -> None:
        """Performs the run for stage 4.

        Stage 4 computes the results of the SNN against the
        default/Neumann implementation. Then stores this result in the
        last entry of each graph.
        """
        set_results(
            results_nx_graphs["run_config"],
            results_nx_graphs["graphs_dict"],
        )
        export_results_to_json(export_images, results_nx_graphs, 4, to_run)
        assert_stage_is_completed(results_nx_graphs["run_config"], 4, to_run)


def experiment_config_to_run_configs(
    experiment_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generates all the run_config dictionaries of a single experiment
    configuration. Then verifies whether each run_config is valid.

    TODO: Ensure this can be done modular, and lower the depth of the loop.
    """
    # pylint: disable=R0914
    supp_run_setts = Supported_run_settings()
    run_configs = []

    # pylint: disable=R1702
    # TODO: make it loop through a list of keys.
    # for algorithm in experiment_config["algorithms"]:
    for algorithm_name, algo_specs in experiment_config["algorithms"].items():
        for algo_config in algo_specs:
            algorithm = {algorithm_name: algo_config}
            for adaptation_name, adaptation_setts_list in experiment_config[
                "adaptations"
            ].items():
                for adaptation_config in adaptation_setts_list:
                    adaptation = {adaptation_name: adaptation_config}

                    for (
                        radiation_name,
                        radiation_setts_list,
                    ) in experiment_config["radiations"].items():
                        # TODO: verify it is of type list.
                        for rad_config in radiation_setts_list:
                            radiation = {radiation_name: rad_config}

                            for iteration in experiment_config["iterations"]:
                                for size_and_max_graph in experiment_config[
                                    "size_and_max_graphs"
                                ]:
                                    for simulator in experiment_config[
                                        "simulators"
                                    ]:
                                        for graph_nr in range(
                                            0, size_and_max_graph[1]
                                        ):
                                            run_configs.append(
                                                run_parameters_to_dict(
                                                    adaptation,
                                                    algorithm,
                                                    iteration,
                                                    size_and_max_graph,
                                                    graph_nr,
                                                    radiation,
                                                    experiment_config,
                                                    simulator,
                                                )
                                            )

    for run_config in run_configs:
        verify_run_config(
            supp_run_setts, run_config, has_unique_id=False, strict=True
        )

        # Append unique_id to run_config
        supp_run_setts.append_unique_config_id(run_config)

        # Append show_snns and export_images to run config.
        supp_run_setts.assert_has_key(experiment_config, "show_snns", bool)
        supp_run_setts.assert_has_key(experiment_config, "export_images", bool)
        run_config["show_snns"] = experiment_config["show_snns"]
        run_config["export_images"] = experiment_config["export_images"]
    return run_configs


# pylint: disable=R0913
def run_parameters_to_dict(
    adaptation: Dict[str, Any],
    algorithm: Dict[str, Any],
    iteration: int,
    size_and_max_graph: Tuple[int, int],
    graph_nr: int,
    radiation: Dict[str, Any],
    experiment_config: Dict[str, Any],
    simulator: str,
) -> dict:
    """Stores selected parameters into a dictionary.

    Written as separate argument to keep code width under 80 lines. #
    TODO: verify typing.
    """
    return {
        "adaptation": adaptation,
        "algorithm": algorithm,
        "iteration": iteration,
        "graph_size": size_and_max_graph[0],
        "graph_nr": graph_nr,
        "radiation": radiation,
        "overwrite_sim_results": experiment_config["overwrite_sim_results"],
        "overwrite_visualisation": experiment_config[
            "overwrite_visualisation"
        ],
        "seed": experiment_config["seed"],
        "simulator": simulator,
    }


def determine_what_to_run(run_config: Dict[str, Any]) -> Dict[str, bool]:
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
        or run_config["overwrite_sim_results"]
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
        or run_config["overwrite_sim_results"]
    ):
        to_run["stage_2"] = True
    # Check if the visualisation of the graph behaviour needs to be created.
    if (
        not has_outputted_stage(run_config, 3, to_run)
        and (run_config["export_images"] or run_config["show_snns"])
    ) or run_config["overwrite_visualisation"]:
        # Note this allows the user to create inconsistent simulation
        # results and visualisation. E.g. the simulated behaviour may
        # have changed due to code changes, yet the visualisation would
        # not be updated stage 3 has already been performed, with
        # overwrite_sim_results=True, and overwrite_visualisation=False.
        to_run["stage_3"] = True
    else:
        to_run["stage_3"] = False

    # Throw warning to user about potential discrepancy between graph
    # behaviour and old visualisation.
    if (
        has_outputted_stage(run_config, 3, to_run)
        and run_config["overwrite_sim_results"]
        and not run_config["overwrite_visualisation"]
    ):
        input(
            "Warning, if you have changed the graph behaviour without "
            + "overwrite_visualisation=True, your visualisation may/will "
            + "not match with what the graphs actually do. We suggest you "
            + "try this again with:overwrite_visualisation=True"
        )
    return to_run
