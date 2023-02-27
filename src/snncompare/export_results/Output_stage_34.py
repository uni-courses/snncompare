""""Outputs the results for stage 3 and/or 4."""
from typing import Dict

from typeguard import typechecked

from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.export_plots.create_snn_gif import create_gif_of_run_config
from snncompare.helper import add_stage_completion_to_graph
from snncompare.optional_config.Output_config import Output_config

from .helper import run_config_to_filename
from .Output import output_stage_json, plot_graph_behaviours


@typechecked
def output_stage_files_3_and_4(
    *, output_config: Output_config, results_nx_graphs: Dict, stage_index: int
) -> None:
    """Merges the experiment configuration Dict, run configuration dict into a
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

    :param exp_config: param run_config:
    :param graphs_stage_2:
    :param run_config:
    """
    # TODO: merge experiment config, results_nx_graphs['run_config'] into
    # single dict.
    if results_nx_graphs["run_config"].simulator == "nx":
        run_config_filename = run_config_to_filename(
            run_config_dict=results_nx_graphs["run_config"].__dict__
        )
        if stage_index == 3:
            print(f"output_config.export_types={output_config.export_types}")
            if "pdf" in output_config.export_types:
                # TODO: Check if plots are already generated and if they must
                # be overwritten.
                # Output graph behaviour for stage stage_index.
                plot_graph_behaviours(
                    run_config_filename=run_config_filename,
                    output_config=output_config,
                    stage_2_graphs=results_nx_graphs["graphs_dict"],
                )
            if "svg" in output_config.export_types:
                create_svg_plot(
                    run_config_filename=run_config_filename,
                    graph_names=["rad_adapted_snn_graph"],
                    graphs=results_nx_graphs["graphs_dict"],
                    output_config=output_config,
                )
            elif "pdf" in output_config.export_types:
                pass
            if "gif" in output_config.export_types:
                create_gif_of_run_config(results_nx_graphs=results_nx_graphs)
                for nx_graph in results_nx_graphs["graphs_dict"].values():
                    add_stage_completion_to_graph(
                        input_graph=nx_graph, stage_index=3
                    )

        if (
            # results_nx_graphs["run_config"].export_images or
            stage_index
            == 4
        ):
            # Output the json dictionary of the files.
            output_stage_json(
                results_nx_graphs=results_nx_graphs,
                run_config_filename=run_config_filename,
                stage_index=stage_index,
            )

    elif results_nx_graphs["run_config"].simulator == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        raise ValueError("Error, lava export method not yet implemented.")
    else:
        raise ValueError("Simulator not supported.")
    # TODO: write merged dict to file.

    # TODO: append tags to output file(s).
