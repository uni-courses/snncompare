"""Contains helper functions for exporting simulation results."""
import collections
import copy
from typing import Any, Dict, List, Union

import networkx as nx

from src.snncompare.helper import get_sim_duration


def flatten(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Union[Dict[str, float], Dict[str, int]]:
    """Flattens a dictionary (makes multiple lines into a oneliner)."""
    items: List = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(dict(v), new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# >>> flatten({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
# {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}


def run_config_to_filename(run_config: dict) -> str:
    """Converts a run_config dictionary into a filename.

    Does that by flattining the dictionary (and all its child-
    dictionaries).
    """
    # TODO: order dictionaries by alphabetical order by default.
    # TODO: allow user to specify a custom order of parameters.

    stripped_run_config = copy.deepcopy(run_config)
    stripped_run_config.pop("unique_id")  # Unique Id will be added as tag
    stripped_run_config.pop("overwrite_sim_results")  # Irrellevant
    stripped_run_config.pop("overwrite_visualisation")  # Irrellevant
    stripped_run_config.pop("show_snns")  # Irrellevant
    stripped_run_config.pop("export_images")  # Irrellevant
    # instead (To reduce filename length).
    filename = str(flatten(stripped_run_config))

    # Remove the ' symbols.
    # Don't, that makes it more difficult to load the dict again.
    # filename=filename.replace("'","")

    # Don't, that makes it more difficult to load the dict again.
    # Remove the spaces.
    filename = filename.replace(" ", "")

    if len(filename) > 256:
        raise Exception(f"Filename={filename} is too long:{len(filename)}")
    return filename


def get_expected_image_paths_stage_3(
    graph_names: List[str],
    input_graph: nx.DiGraph,
    run_config: dict,
    extensions: List[str],
) -> List:
    """Returns the expected image filepaths for stage 3.

    (If export is on).
    """
    image_filepaths = []
    filename: str = run_config_to_filename(run_config)

    if "alg_props" not in input_graph.graph.keys():
        raise Exception("Error, algo_props is not set.")

    sim_duration = get_sim_duration(
        input_graph,
        run_config,
    )

    # TODO: move this into hardcoded setting.
    image_dir = "latex/Images/graphs/"
    for extension in extensions:
        for graph_name in graph_names:
            if graph_name == "input_graph":
                image_filepaths.append(
                    f"results/{graph_name}_{filename}{extension}"
                )
            else:
                for t in range(0, sim_duration):
                    image_filepaths.append(
                        image_dir + f"{graph_name}_{filename}_{t}{extension}"
                    )
    return image_filepaths
