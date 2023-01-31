"""Creates a gif of an SNN propagation."""

from typing import Dict

import imageio

from snncompare.export_results.helper import (
    get_expected_image_paths_stage_3,
    run_config_to_filename,
)
from snncompare.graph_generation.stage_1_get_input_graphs import (
    get_input_graph,
)


def create_gif_of_run_config(*, results_nx_graphs: Dict) -> None:
    """Creates a gif of an SNN propagation."""
    run_config = results_nx_graphs["run_config"]
    if run_config.gif:
        expected_filepaths = []
        extensions = (run_config.export_types,)
        print(f"extensions={extensions}")
        output_filename = run_config_to_filename(run_config=run_config)

        # Get expected png filenames.
        if "png" in extensions:  # No png, no gif
            if run_config.export_images:
                expected_filepaths.extend(
                    get_expected_image_paths_stage_3(
                        nx_graphs_dict=results_nx_graphs["graphs_dict"],
                        input_graph=get_input_graph(run_config=run_config),
                        run_config=run_config,
                        extensions=["png"],
                    )
                )

            # Verify expected png filenames exist.

            # Convert pngs to a single gif.
            with imageio.get_writer(
                f"latex/Images/graphs/{output_filename}.gif",
                mode="I",
                duration=0.5,
            ) as writer:
                for filepath in expected_filepaths:
                    image = imageio.imread(filepath)
                    writer.append_data(image)
