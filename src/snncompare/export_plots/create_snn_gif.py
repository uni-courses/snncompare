"""Creates a gif of an SNN propagation."""
import imageio

from snncompare.export_results.helper import (
    get_expected_image_paths_stage_3,
    run_config_to_filename,
)
from snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from snncompare.graph_generation.stage_1_get_input_graphs import (
    get_input_graph,
)
from snncompare.helper import get_extensions_list


def create_gif_of_run_config(run_config: dict) -> None:
    """Creates a gif of an SNN propagation."""
    expected_filepaths = []
    extensions = get_extensions_list(run_config, stage_index=3)
    output_filename = run_config_to_filename(run_config)

    # Get expected png filenames.
    if run_config["export_images"]:
        expected_filepaths.extend(
            get_expected_image_paths_stage_3(
                get_expected_stage_1_graph_names(run_config),
                get_input_graph(run_config),
                run_config,
                extensions,
            )
        )

    # Verify expected png filenames exist.

    # Convert pngs to a single gif.
    with imageio.get_writer(
        f"latex/Images/graphs/{output_filename}.gif", mode="I"
    ) as writer:
        for filepath in expected_filepaths:
            image = imageio.imread(filepath)
            writer.append_data(image)
