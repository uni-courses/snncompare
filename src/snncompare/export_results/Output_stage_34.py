""""Outputs the results for stage 3 and/or 4."""
from typeguard import typechecked

from snncompare.export_plots.create_snn_gif import create_gif_of_run_config
from snncompare.helper import add_stage_completion_to_graph

from .helper import run_config_to_filename
from .Output import output_stage_json, plot_graph_behaviours


@typechecked
def output_stage_files_3_and_4(
    results_nx_graphs: dict, stage_index: int, to_run: dict
) -> None:
    """Merges the experiment configuration dict, run configuration dict into a
    single dict. This method assumes only the graphs that are to be exported
    are passed into this method.

    If the networkx (nx) simulator is used, the graphs should be
    convertible into dicts. This merged dict is then written to file. If
    the lava simulator is used, the graphs cannot (easily) be converted
    into dicts, hence in that case, only the experiment and run settings
    are merged into a single dict that is exported.

    % TODO: The lava graphs will be terminated and exported as pickle.
    If the graphs are not exported, pickle throws an error because some
    pipe/connection/something is still open.

    The unique_id of the experiment is added to the file as a filetag, as well
    as the unique_id of the run. Furthermore, all run parameter values are
    added as file tags, to make it easier to filter certain runs to
    manually inspect the results.

    Also exports the images of the graph behaviour.

    :param experiment_config: param run_config:
    :param graphs_stage_2:
    :param run_config:
    """
    # TODO: merge experiment config, results_nx_graphs['run_config'] into
    # single dict.
    if results_nx_graphs["run_config"].simulator == "nx":
        filename = run_config_to_filename(results_nx_graphs["run_config"])
        # TODO: Check if plots are already generated and if they must be
        # overwritten.
        # TODO: Distinguish between showing snns and outputting snns.
        if results_nx_graphs["run_config"].export_images and stage_index == 3:
            # Output graph behaviour for stage stage_index.
            plot_graph_behaviours(
                filename,
                results_nx_graphs["graphs_dict"],
                results_nx_graphs["run_config"],
            )
            create_gif_of_run_config(results_nx_graphs)
            for nx_graph in results_nx_graphs["graphs_dict"].values():
                add_stage_completion_to_graph(nx_graph, 3)

        if (
            # results_nx_graphs["run_config"].export_images or
            stage_index
            == 4
        ):
            # Output the json dictionary of the files.
            output_stage_json(
                results_nx_graphs,
                filename,
                stage_index,
                to_run,
            )

    elif results_nx_graphs["run_config"].simulator == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        raise Exception("Error, lava export method not yet implemented.")
    else:
        raise Exception("Simulator not supported.")
    # TODO: write merged dict to file.

    # TODO: append tags to output file(s).
