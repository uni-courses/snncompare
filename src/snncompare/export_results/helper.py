"""Contains helper functions for exporting simulation results."""
import collections
import copy
import hashlib
import json
from typing import Any, Dict, List, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from ..helper import get_some_duration


@typechecked
def flatten(
    *, d: Dict, parent_key: str = "", sep: str = "_"
) -> Union[Dict, Dict[str, float], Dict[str, int]]:
    """Flattens a dictionary (makes multiple lines into a oneliner)."""
    items: List = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(
                flatten(d=dict(v), parent_key=new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


@typechecked
def run_config_to_filename(
    *,
    run_config_dict: Dict,
) -> str:
    """Converts a run_config dictionary into a filename.

    Does that by flattining the dictionary (and all its child-
    dictionaries).
    """
    # TODO: order dictionaries by alphabetical order by default.
    # TODO: allow user to specify a custom order of parameters.

    # stripped_run_config:Dict = copy.deepcopy(run_config).__dict__
    stripped_run_config: Dict = copy.deepcopy(run_config_dict)

    # instead (To reduce filename length).
    filename = str(flatten(d=stripped_run_config))

    # Remove the ' symbols.
    # Don't, that makes it more difficult to load the dict again.
    # filename=filename.replace("'","")

    # Don't, that makes it more difficult to load the dict again.
    # Remove the spaces.
    filename = filename.replace(" ", "")
    filename = filename.replace("adaptation_", "")
    filename = filename.replace("algorithm_", "")
    filename = filename.replace("graph_", "")
    filename = filename.replace("radiation_", "")
    filename = filename.replace("unique_", "")

    if len(filename) > 256:
        raise NameError(f"Filename={filename} is too long:{len(filename)}")
    return filename


@typechecked
def exp_config_to_filename(
    *,
    exp_config_dict: Dict,
) -> str:
    """Converts an Exp_config dictionary into a filename.

    Does that by flattining the dictionary (and all its child-
    dictionaries).
    """
    # TODO: order dictionaries by alphabetical order by default.
    # TODO: allow user to specify a custom order of parameters.

    # stripped_run_config:Dict = copy.deepcopy(run_config).__dict__
    stripped_exp_config: Dict = copy.deepcopy(exp_config_dict)
    stripped_exp_config.pop("unique_id")

    # instead (To reduce filename length).
    filename = str(flatten(d=stripped_exp_config))

    # Remove the ' symbols.
    # Don't, that makes it more difficult to load the dict again.
    # filename=filename.replace("'","")

    # Don't, that makes it more difficult to load the dict again.
    # Remove the spaces.
    filename = filename.replace(" ", "")
    filename = filename.replace("'", "")
    filename = filename.replace("[", "")
    filename = filename.replace("]", "")
    filename = filename.replace("{", "")
    filename = filename.replace("}", "")
    filename = filename.replace("adaptations_", "")
    filename = filename.replace("algorithms_", "")
    filename = filename.replace("graphs_", "")
    filename = filename.replace("radiations_", "")
    filename = filename.replace("unique_", "")

    if len(filename) > 256:
        raise NameError(f"Filename={filename} is too long:{len(filename)}")
    return filename


@typechecked
def get_expected_image_paths_stage_3(  # type:ignore[misc]
    *,
    nx_graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    input_graph: nx.Graph,
    run_config: Any,
    extensions: List[str],
) -> List[str]:
    """Returns the expected image filepaths for stage 3.

    (If export is on).
    """
    image_filepaths = []
    filename: str = run_config_to_filename(run_config_dict=run_config.__dict__)

    if "alg_props" not in input_graph.graph.keys():
        raise KeyError("Error, algo_props is not set.")

    # TODO: move this into hardcoded setting.
    image_dir = "latex/Images/graphs/"
    for extension in extensions:
        for graph_name, snn_graph in nx_graphs_dict.items():
            if graph_name != "input_graph":
                sim_duration = get_some_duration(
                    simulator=run_config.simulator,
                    snn_graph=snn_graph,
                    duration_name="actual_duration",
                )
                for t in range(0, sim_duration):
                    image_filepaths.append(
                        image_dir + f"{graph_name}_{filename}_{t}.{extension}"
                    )
    return image_filepaths


@typechecked
def get_unique_id(  # type:ignore[misc]
    *,
    some_config: Any,
) -> str:
    """Checks if an experiment configuration dictionary already has a unique
    identifier, and if not it computes and appends it.

    If it does, throws an error.
    TODO: move into helper

    :param exp_config: Exp_config:
    """
    if "unique_id" in some_config.__dict__.keys():
        raise KeyError(
            f"Error, the exp_config:{some_config}\n"
            + "already contains a unique identifier."
        )

    # Compute a unique code belonging to this particular experiment
    # configuration.
    unique_id = str(
        hashlib.sha256(
            json.dumps(some_config.__dict__).encode("utf-8")
        ).hexdigest()
    )
    return unique_id
